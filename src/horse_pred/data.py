"""Data-contract helpers for the approved race-results CSV.

The streaming audit uses only the Python standard library; the public DataFrame
API imports pandas lazily. The module never embeds a workstation-specific raw
path: callers inject it explicitly or through the manifest-named environment
variable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


RAW_COLUMNS: tuple[str, ...] = (
    "raceid",
    "race_class",
    "course_type",
    "distance",
    "ground_state",
    "around",
    "weather",
    "着順",
    "枠番",
    "馬番",
    "馬名",
    "horse_id",
    "sex",
    "age",
    "騎手",
    "jockey_id",
    "trainer",
    "タイム",
    "着差",
    "通過順位",
    "上がり3F",
    "単勝",
    "人気",
    "馬体重",
    "馬体重増減",
    "date",
)

VENUE_CODES: dict[str, str] = {
    "01": "sapporo",
    "02": "hakodate",
    "03": "fukushima",
    "04": "niigata",
    "05": "tokyo",
    "06": "nakayama",
    "07": "chukyo",
    "08": "kyoto",
    "09": "hanshin",
    "10": "kokura",
}

SURFACE_CODES: dict[str, str] = {
    "芝": "turf",
    "ダート": "dirt",
    "障害": "jump",
}

NULL_TOKENS = frozenset({"", "--", "---"})
NONSTARTER_STATUSES = frozenset({"scratched", "excluded"})
STARTER_STATUSES = frozenset({"finished", "demoted", "did_not_finish", "disqualified"})

_DEMOTED_RE = re.compile(r"^(?P<position>[0-9]+)[(（]降[)）]$")
_RACE_ID_RE = re.compile(r"^[0-9]{12}$")
_HORSE_ID_RE = re.compile(r"^[0-9]{10}$")
_JOCKEY_ID_RE = re.compile(r"^[0-9]{3,4}$")


class DataContractError(ValueError):
    """Base exception for a data-contract violation."""


class SchemaMismatchError(DataContractError):
    """Raised when the CSV columns differ from the frozen raw schema."""


class FingerprintMismatchError(DataContractError):
    """Raised when raw bytes differ from the manifest fingerprint."""


def _pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without project deps
        raise DataContractError(
            "pandas is required for the DataFrame API; install the project dependencies"
        ) from exc
    return pd


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON manifest without resolving or mutating the raw data."""

    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise DataContractError("manifest root must be a JSON object")
    return manifest


def resolve_raw_path(
    explicit_path: str | os.PathLike[str] | None = None,
    *,
    environment_variable: str = "HORSE_PRED_RAW_CSV",
) -> Path:
    """Resolve an injected raw path; no workstation path is used as a default."""

    raw_value = os.fspath(explicit_path) if explicit_path is not None else os.getenv(environment_variable)
    if not raw_value:
        raise DataContractError(
            f"raw CSV path is required via --raw-path or {environment_variable}"
        )
    path = Path(raw_value).expanduser()
    if not path.is_file():
        raise DataContractError(f"raw CSV does not exist or is not a file: {path}")
    return path


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def has_utf8_bom(path: str | os.PathLike[str]) -> bool:
    with Path(path).open("rb") as handle:
        return handle.read(3) == b"\xef\xbb\xbf"


def verify_raw_file(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Verify byte-level identity before parsing the raw CSV."""

    raw_spec = manifest.get("raw_file")
    if not isinstance(raw_spec, Mapping):
        raise DataContractError("manifest.raw_file must be an object")

    raw_path = Path(path)
    actual_size = raw_path.stat().st_size
    expected_size = raw_spec.get("size_bytes")
    if expected_size is not None and actual_size != expected_size:
        raise FingerprintMismatchError(
            f"size mismatch: expected {expected_size}, got {actual_size}"
        )

    actual_bom = has_utf8_bom(raw_path)
    expected_bom = raw_spec.get("has_utf8_bom")
    if expected_bom is not None and actual_bom is not expected_bom:
        raise FingerprintMismatchError(
            f"UTF-8 BOM mismatch: expected {expected_bom}, got {actual_bom}"
        )

    result: dict[str, Any] = {
        "size_bytes": actual_size,
        "has_utf8_bom": actual_bom,
    }
    if verify_hash:
        actual_hash = sha256_file(raw_path)
        expected_hash = raw_spec.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise FingerprintMismatchError(
                f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        result["sha256"] = actual_hash
    return result


def _verify_expected_sha256(
    path: str | os.PathLike[str], expected_sha256: str | None
) -> str | None:
    if expected_sha256 is None:
        return None
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise FingerprintMismatchError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_raw(
    path: str | os.PathLike[str], expected_sha256: str | None = None
) -> pd.DataFrame:
    """Load the exact raw schema as strings into a pandas DataFrame.

    String loading is intentional: source-native IDs keep leading zeros, and
    exceptional outcome/null tokens survive until ``normalize_raw``.
    """

    _verify_expected_sha256(path, expected_sha256)
    pd = _pandas()
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    actual = tuple(frame.columns)
    if actual != RAW_COLUMNS:
        raise SchemaMismatchError(
            f"CSV header mismatch; expected={list(RAW_COLUMNS)}, actual={list(actual)}"
        )
    return frame


def normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Vector-normalize a raw DataFrame without dropping exceptional runners."""

    pd = _pandas()
    actual = tuple(raw.columns)
    if actual != RAW_COLUMNS:
        raise SchemaMismatchError(
            f"DataFrame columns mismatch; expected={list(RAW_COLUMNS)}, actual={list(actual)}"
        )

    normalized = raw.copy()

    def clean_series(column: str) -> pd.Series:
        return normalized[column].fillna("").astype(str).str.strip()

    def numeric_series(column: str, *, integer: bool) -> pd.Series:
        text = clean_series(column)
        null_mask = text.isin(NULL_TOKENS)
        numeric = pd.to_numeric(text.where(~null_mask), errors="coerce")
        invalid_mask = ~null_mask & numeric.isna()
        if invalid_mask.any():
            invalid = text.loc[invalid_mask].iloc[0]
            raise DataContractError(f"{column} is not numeric: {invalid!r}")
        if integer:
            fractional_mask = numeric.notna() & numeric.mod(1).ne(0)
            if fractional_mask.any():
                invalid = text.loc[fractional_mask].iloc[0]
                raise DataContractError(f"{column} must be integer-valued: {invalid!r}")
            return numeric.astype("Int64")
        return numeric.astype("Float64")

    race_id = clean_series("raceid")
    horse_id = clean_series("horse_id")
    jockey_id = clean_series("jockey_id")
    for values, pattern, message in (
        (race_id, r"[0-9]{12}", "raceid must be 12 ASCII digits"),
        (horse_id, r"[0-9]{10}", "horse_id must be 10 ASCII digits"),
        (jockey_id, r"[0-9]{3,4}", "jockey_id must be 3 or 4 ASCII digits"),
    ):
        invalid_mask = ~values.str.fullmatch(pattern)
        if invalid_mask.any():
            raise DataContractError(f"{message}: {values.loc[invalid_mask].iloc[0]!r}")

    segment_start = race_id.ne(race_id.shift())
    segment_race_ids = race_id.loc[segment_start]
    repeated_segment = segment_race_ids.duplicated()
    if repeated_segment.any():
        repeated_id = segment_race_ids.loc[repeated_segment].iloc[0]
        raise DataContractError(f"race {repeated_id} is not stored contiguously")

    date_text = clean_series("date")
    date_format_valid = date_text.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
    race_timestamp = pd.to_datetime(
        date_text.where(date_format_valid), format="%Y-%m-%d", errors="coerce"
    )
    invalid_date = race_timestamp.isna()
    if invalid_date.any():
        raise DataContractError(
            f"date is not ISO YYYY-MM-DD: {date_text.loc[invalid_date].iloc[0]!r}"
        )
    year_mismatch = race_id.str[:4].astype(int).ne(race_timestamp.dt.year)
    if year_mismatch.any():
        index = year_mismatch.loc[year_mismatch].index[0]
        raise DataContractError(
            f"raceid year {race_id.loc[index][:4]} disagrees with date "
            f"{date_text.loc[index]}"
        )

    finish_raw = clean_series("着順")
    numeric_finish = finish_raw.str.fullmatch(r"[0-9]+")
    demoted_position = finish_raw.str.extract(_DEMOTED_RE, expand=True)["position"]
    finish_position = pd.to_numeric(
        finish_raw.where(numeric_finish, demoted_position), errors="coerce"
    ).astype("Int64")
    status = pd.Series("unknown", index=normalized.index, dtype="string")
    status.loc[numeric_finish] = "finished"
    status.loc[demoted_position.notna()] = "demoted"
    status.loc[finish_raw.isin({"中", "中止", "競走中止"})] = "did_not_finish"
    status.loc[finish_raw.isin({"失", "失格"})] = "disqualified"
    status.loc[finish_raw.isin({"取", "取消"})] = "scratched"
    status.loc[finish_raw.isin({"除", "除外"})] = "excluded"

    started = pd.Series(pd.NA, index=normalized.index, dtype="boolean")
    started.loc[status.isin(STARTER_STATUSES)] = True
    started.loc[status.isin(NONSTARTER_STATUSES)] = False
    starter_mask = started.eq(True).fillna(False)
    nonstarter_mask = started.eq(False).fillna(False)

    venue_code = race_id.str[4:6]
    surface_raw = clean_series("course_type")
    distance_m = numeric_series("distance", integer=True)
    race_date = race_timestamp.dt.date
    race_fields = pd.DataFrame(
        {
            "race_id": race_id,
            "race_date": race_date,
            "venue_code": venue_code,
            "surface_raw": surface_raw,
            "distance_m": distance_m,
        },
        index=normalized.index,
    )
    for field in ("race_date", "venue_code", "surface_raw", "distance_m"):
        conflict_counts = race_fields.groupby("race_id", sort=False)[field].nunique(dropna=False)
        if conflict_counts.gt(1).any():
            conflict_race = conflict_counts.loc[conflict_counts.gt(1)].index[0]
            raise DataContractError(
                f"race-level field {field} conflicts within race {conflict_race}"
            )

    rank_frame = pd.DataFrame(
        {"race_id": race_id, "finish_position": finish_position},
        index=normalized.index,
    )
    dead_heat_size = (
        rank_frame.groupby(
            ["race_id", "finish_position"], sort=False, dropna=False
        )["race_id"]
        .transform("size")
        .astype("Int64")
        .where(finish_position.notna())
    )
    is_dead_heat = dead_heat_size.gt(1).fillna(False).astype(bool)
    winner_indicator = finish_position.eq(1).fillna(False).astype(int)
    winner_count = winner_indicator.groupby(race_id, sort=False).transform("sum")

    winner_label = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    winner_label.loc[starter_mask] = 0
    winner_label.loc[starter_mask & finish_position.eq(1).fillna(False)] = 1
    coherent_win_target = pd.Series(pd.NA, index=normalized.index, dtype="Float64")
    coherent_win_target.loc[starter_mask] = 0.0
    winner_mask = starter_mask & finish_position.eq(1).fillna(False)
    coherent_win_target.loc[winner_mask] = 1.0 / winner_count.loc[winner_mask]

    race_has_nonstarter = (
        nonstarter_mask.astype(int).groupby(race_id, sort=False).transform("max").astype(bool)
    )
    normalized = normalized.assign(
        race_id=race_id,
        race_date=race_date,
        venue_code=venue_code,
        venue=venue_code.map(VENUE_CODES).fillna("unknown"),
        venue_is_known=venue_code.isin(VENUE_CODES),
        surface_raw=surface_raw,
        surface=surface_raw.map(SURFACE_CODES).fillna("unknown"),
        surface_is_known=surface_raw.isin(SURFACE_CODES),
        distance_m=distance_m,
        frame_number=numeric_series("枠番", integer=True),
        horse_number=numeric_series("馬番", integer=True),
        horse_age=numeric_series("age", integer=True),
        finish_raw=finish_raw,
        finish_position=finish_position,
        status=status,
        finish_status=status,
        started=started,
        history_update_eligible=starter_mask,
        winner_label=winner_label,
        coherent_win_target=coherent_win_target,
        is_dead_heat=is_dead_heat,
        dead_heat_size=dead_heat_size,
        pit_c_scoring_eligible=~race_has_nonstarter,
        time_raw=normalized["タイム"],
        margin_raw=normalized["着差"],
        passing_order_raw=normalized["通過順位"],
        last_3f_seconds=numeric_series("上がり3F", integer=False),
        final_win_odds=numeric_series("単勝", integer=False),
        final_popularity=numeric_series("人気", integer=True),
        body_weight_kg=numeric_series("馬体重", integer=True),
        body_weight_change_kg=numeric_series("馬体重増減", integer=True),
    )
    return normalized


def audit_raw(
    path: str | os.PathLike[str], expected_sha256: str | None = None
) -> dict[str, Any]:
    """Integrated API: optionally verify bytes, then run the streaming audit."""

    actual_hash = sha256_file(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise FingerprintMismatchError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    report = audit_csv(path)
    report["fingerprint"] = {
        "sha256": actual_hash,
        "size_bytes": Path(path).stat().st_size,
        "has_utf8_bom": has_utf8_bom(path),
    }
    return report


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _nullable_int(value: Any, field: str) -> int | None:
    raw = _clean(value)
    if raw in NULL_TOKENS:
        return None
    try:
        number = float(raw)
    except ValueError as exc:
        raise DataContractError(f"{field} is not numeric: {raw!r}") from exc
    if not number.is_integer():
        raise DataContractError(f"{field} must be integer-valued: {raw!r}")
    return int(number)


def _nullable_float(value: Any, field: str) -> float | None:
    raw = _clean(value)
    if raw in NULL_TOKENS:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise DataContractError(f"{field} is not numeric: {raw!r}") from exc


def parse_finish(raw_finish: Any) -> tuple[int | None, str, bool | None]:
    """Return official position, status, and whether the runner started.

    Unknown tokens are preserved by the caller and represented as ``unknown``;
    they are never silently dropped or guessed to be starters.
    """

    raw = _clean(raw_finish)
    if raw.isdigit():
        return int(raw), "finished", True
    demoted = _DEMOTED_RE.fullmatch(raw)
    if demoted:
        return int(demoted.group("position")), "demoted", True
    if raw in {"中", "中止", "競走中止"}:
        return None, "did_not_finish", True
    if raw in {"失", "失格"}:
        return None, "disqualified", True
    if raw in {"取", "取消"}:
        return None, "scratched", False
    if raw in {"除", "除外"}:
        return None, "excluded", False
    return None, "unknown", None


def normalize_raw_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one row while retaining every original raw field."""

    missing = [column for column in RAW_COLUMNS if column not in row]
    extra = [column for column in row if column not in RAW_COLUMNS]
    if missing or extra:
        raise SchemaMismatchError(f"row schema mismatch; missing={missing}, extra={extra}")

    normalized: dict[str, Any] = {
        column: "" if row[column] is None else str(row[column]) for column in RAW_COLUMNS
    }
    race_id = _clean(normalized["raceid"])
    if not _RACE_ID_RE.fullmatch(race_id):
        raise DataContractError(f"raceid must be 12 ASCII digits: {race_id!r}")
    horse_id = _clean(normalized["horse_id"])
    if not _HORSE_ID_RE.fullmatch(horse_id):
        raise DataContractError(f"horse_id must be 10 ASCII digits: {horse_id!r}")
    jockey_id = _clean(normalized["jockey_id"])
    if not _JOCKEY_ID_RE.fullmatch(jockey_id):
        raise DataContractError(f"jockey_id must be 3 or 4 ASCII digits: {jockey_id!r}")

    venue_code = race_id[4:6]
    surface_raw = _clean(normalized["course_type"])
    finish_position, finish_status, started = parse_finish(normalized["着順"])
    try:
        race_date = date.fromisoformat(_clean(normalized["date"]))
    except ValueError as exc:
        raise DataContractError(f"date is not ISO YYYY-MM-DD: {normalized['date']!r}") from exc
    if int(race_id[:4]) != race_date.year:
        raise DataContractError(
            f"raceid year {race_id[:4]} disagrees with date {race_date.isoformat()}"
        )

    normalized.update(
        {
            "race_id": race_id,
            "race_date": race_date,
            "venue_code": venue_code,
            "venue": VENUE_CODES.get(venue_code, "unknown"),
            "venue_is_known": venue_code in VENUE_CODES,
            "surface_raw": surface_raw,
            "surface": SURFACE_CODES.get(surface_raw, "unknown"),
            "surface_is_known": surface_raw in SURFACE_CODES,
            "distance_m": _nullable_int(normalized["distance"], "distance"),
            "frame_number": _nullable_int(normalized["枠番"], "枠番"),
            "horse_number": _nullable_int(normalized["馬番"], "馬番"),
            "horse_age": _nullable_int(normalized["age"], "age"),
            "finish_raw": normalized["着順"],
            "finish_position": finish_position,
            "status": finish_status,
            "finish_status": finish_status,
            "started": started,
            "history_update_eligible": started is True,
            "winner_label": (1 if finish_position == 1 else 0) if started is True else None,
            "coherent_win_target": None,
            "is_dead_heat": False,
            "dead_heat_size": 1 if finish_position is not None else None,
            "time_raw": normalized["タイム"],
            "margin_raw": normalized["着差"],
            "passing_order_raw": normalized["通過順位"],
            "last_3f_seconds": _nullable_float(normalized["上がり3F"], "上がり3F"),
            "final_win_odds": _nullable_float(normalized["単勝"], "単勝"),
            "final_popularity": _nullable_int(normalized["人気"], "人気"),
            "body_weight_kg": _nullable_int(normalized["馬体重"], "馬体重"),
            "body_weight_change_kg": _nullable_int(
                normalized["馬体重増減"], "馬体重増減"
            ),
        }
    )
    return normalized


def normalize_race_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one race and annotate every member of a dead-heat group."""

    normalized = [normalize_raw_row(row) for row in rows]
    if not normalized:
        return normalized
    race_ids = {row["race_id"] for row in normalized}
    if len(race_ids) != 1:
        raise DataContractError(f"normalize_race_rows received multiple races: {sorted(race_ids)}")
    for field in ("race_date", "venue_code", "surface_raw", "distance_m"):
        values = {row[field] for row in normalized}
        if len(values) != 1:
            race_id = normalized[0]["race_id"]
            raise DataContractError(
                f"race-level field {field} conflicts within race {race_id}: {sorted(values, key=str)}"
            )

    rank_counts = Counter(
        row["finish_position"] for row in normalized if row["finish_position"] is not None
    )
    winner_count = rank_counts.get(1, 0)
    has_late_nonstarter = any(row["finish_status"] in NONSTARTER_STATUSES for row in normalized)
    for row in normalized:
        position = row["finish_position"]
        if position is not None:
            size = rank_counts[position]
            row["dead_heat_size"] = size
            row["is_dead_heat"] = size > 1
        if row["started"] is True:
            row["coherent_win_target"] = (
                1.0 / winner_count if row["finish_position"] == 1 and winner_count else 0.0
            )
        row["pit_c_scoring_eligible"] = not has_late_nonstarter
    return normalized


def summarize_race_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize field sizes without deleting non-finishers or non-starters."""

    if not rows:
        raise DataContractError("cannot summarize an empty race")
    race_ids = {str(row["race_id"]) for row in rows}
    if len(race_ids) != 1:
        raise DataContractError(f"race summary received multiple races: {sorted(race_ids)}")
    status_counts = Counter(str(row["finish_status"]) for row in rows)
    dead_heat_groups: dict[str, int] = {}
    rank_counts = Counter(
        int(row["finish_position"])
        for row in rows
        if row.get("finish_position") is not None
    )
    for position, count in sorted(rank_counts.items()):
        if count > 1:
            dead_heat_groups[str(position)] = count
    return {
        "race_id": next(iter(race_ids)),
        "declared_runner_count": len(rows),
        "starter_count": sum(row.get("started") is True for row in rows),
        "nonstarter_count": sum(row.get("started") is False for row in rows),
        "unknown_start_count": sum(row.get("started") is None for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "dead_heat_groups": dead_heat_groups,
        "pit_c_scoring_eligible": all(
            bool(row.get("pit_c_scoring_eligible")) for row in rows
        ),
    }


def iter_raw_rows(
    path: str | os.PathLike[str],
    expected_columns: Sequence[str] = RAW_COLUMNS,
) -> Iterator[dict[str, str]]:
    """Read a BOM-safe CSV and require an exact, ordered schema."""

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        expected = tuple(expected_columns)
        if actual != expected:
            raise SchemaMismatchError(
                f"CSV header mismatch; expected={list(expected)}, actual={list(actual)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise SchemaMismatchError(f"row {line_number} has more fields than the header")
            if any(value is None for value in row.values()):
                raise SchemaMismatchError(f"row {line_number} has fewer fields than the header")
            yield {column: str(row[column]) for column in expected}


def iter_normalized_races(path: str | os.PathLike[str]) -> Iterator[list[dict[str, Any]]]:
    """Stream contiguous race batches and reject a race that reappears later."""

    current_id: str | None = None
    current_rows: list[dict[str, str]] = []
    closed: set[str] = set()
    for raw_row in iter_raw_rows(path):
        race_id = _clean(raw_row["raceid"])
        if current_id is None:
            current_id = race_id
        if race_id != current_id:
            closed.add(current_id)
            yield normalize_race_rows(current_rows)
            if race_id in closed:
                raise DataContractError(f"race {race_id} is not stored contiguously")
            current_id = race_id
            current_rows = []
        current_rows.append(raw_row)
    if current_rows:
        yield normalize_race_rows(current_rows)


def iter_normalized_rows(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    for race_rows in iter_normalized_races(path):
        yield from race_rows


def expand_race_id_ranges(ranges: Iterable[Mapping[str, Any]]) -> list[str]:
    """Expand inclusive race-number ranges recorded in the manifest."""

    expanded: list[str] = []
    for item in ranges:
        day_prefix = str(item["day_prefix"])
        start = int(item["race_number_start"])
        end = int(item["race_number_end"])
        if not re.fullmatch(r"[0-9]{10}", day_prefix) or not (1 <= start <= end <= 99):
            raise DataContractError(f"invalid missing-race range: {dict(item)}")
        expanded.extend(f"{day_prefix}{number:02d}" for number in range(start, end + 1))
    if len(expanded) != len(set(expanded)):
        raise DataContractError("missing-race ranges overlap")
    return expanded


def _nested_counts(counter: Counter[tuple[int, str | int]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for (year, key), count in sorted(counter.items()):
        result[str(year)][str(key)] = count
    return dict(result)


def audit_csv(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return serializable coverage and quality counts for a CSV."""

    rows = 0
    races = 0
    date_min: str | None = None
    date_max: str | None = None
    race_by_year: Counter[int] = Counter()
    race_by_venue: Counter[tuple[int, str]] = Counter()
    race_by_month: Counter[tuple[int, int]] = Counter()
    race_by_surface: Counter[tuple[int, str]] = Counter()
    runner_by_surface: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    finish_raw_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    unknown_venues: Counter[str] = Counter()
    unknown_surfaces: Counter[str] = Counter()
    seen_race_horse: set[tuple[str, str]] = set()
    seen_race_number: set[tuple[str, int | None]] = set()
    duplicate_race_horse = 0
    duplicate_race_number = 0
    tie_races = 0
    tie_extra_runners = 0
    pit_c_ineligible_races = 0
    starter_count = 0
    nonstarter_count = 0
    unknown_start_count = 0
    observed_race_ids: set[str] = set()

    for race_rows in iter_normalized_races(path):
        races += 1
        summary = summarize_race_rows(race_rows)
        if summary["dead_heat_groups"]:
            tie_races += 1
            tie_extra_runners += sum(
                count - 1 for count in summary["dead_heat_groups"].values()
            )
        if not summary["pit_c_scoring_eligible"]:
            pit_c_ineligible_races += 1

        first = race_rows[0]
        race_id = str(first["race_id"])
        observed_race_ids.add(race_id)
        race_date = first["race_date"]
        year = race_date.year
        iso_date = race_date.isoformat()
        date_min = iso_date if date_min is None or iso_date < date_min else date_min
        date_max = iso_date if date_max is None or iso_date > date_max else date_max
        race_by_year[year] += 1
        race_by_venue[(year, str(first["venue_code"]))] += 1
        race_by_month[(year, race_date.month)] += 1
        race_by_surface[(year, str(first["surface_raw"]))] += 1

        for row in race_rows:
            rows += 1
            starter_count += row["started"] is True
            nonstarter_count += row["started"] is False
            unknown_start_count += row["started"] is None
            outcome_counts[str(row["finish_status"])] += 1
            finish_raw_counts[str(row["finish_raw"])] += 1
            runner_by_surface[str(row["surface_raw"])] += 1
            if not row["venue_is_known"]:
                unknown_venues[str(row["venue_code"])] += 1
            if not row["surface_is_known"]:
                unknown_surfaces[str(row["surface_raw"])] += 1
            for column in RAW_COLUMNS:
                if _clean(row[column]) in NULL_TOKENS:
                    null_counts[column] += 1
            horse_key = (race_id, str(row["horse_id"]))
            number_key = (race_id, row["horse_number"])
            duplicate_race_horse += horse_key in seen_race_horse
            duplicate_race_number += number_key in seen_race_number
            seen_race_horse.add(horse_key)
            seen_race_number.add(number_key)

    official_comparison: dict[str, dict[str, int]] = {}
    known_missing: dict[str, Any] = {
        "expected_count": 0,
        "absent_count": 0,
        "unexpectedly_present": [],
    }
    if manifest:
        coverage_spec = manifest.get("coverage", {})
        official_counts = coverage_spec.get("jra_official_race_counts", {})
        for year_text, official_value in sorted(official_counts.items(), key=lambda item: int(item[0])):
            raw_value = race_by_year[int(year_text)]
            official = int(official_value)
            official_comparison[str(year_text)] = {
                "raw_race_count": raw_value,
                "jra_official_race_count": official,
                "shortfall": official - raw_value,
            }
        missing_ids = expand_race_id_ranges(coverage_spec.get("known_missing_race_id_ranges", []))
        present = sorted(set(missing_ids) & observed_race_ids)
        known_missing = {
            "expected_count": len(missing_ids),
            "absent_count": len(missing_ids) - len(present),
            "unexpectedly_present": present,
        }

    report = {
        "row_count": rows,
        "race_count": races,
        "column_count": len(RAW_COLUMNS),
        "date_min": date_min,
        "date_max": date_max,
        "coverage": {
            "race_count_by_year": {str(key): value for key, value in sorted(race_by_year.items())},
            "race_count_by_year_and_venue_code": _nested_counts(race_by_venue),
            "race_count_by_year_and_month": _nested_counts(race_by_month),
            "race_count_by_year_and_surface": _nested_counts(race_by_surface),
            "runner_count_by_surface": dict(sorted(runner_by_surface.items())),
            "jra_official_comparison": official_comparison,
            "known_missing_race_ids": known_missing,
        },
        "outcomes": {
            "declared_runner_count": rows,
            "starter_count": starter_count,
            "nonstarter_count": nonstarter_count,
            "unknown_start_count": unknown_start_count,
            "normalized_status_counts": dict(sorted(outcome_counts.items())),
            "non_numeric_finish_raw_counts": {
                raw: count
                for raw, count in sorted(finish_raw_counts.items())
                if not raw.isdigit()
            },
            "dead_heat_race_count": tie_races,
            "dead_heat_extra_runner_count": tie_extra_runners,
            "pit_c_scoring_ineligible_race_count": pit_c_ineligible_races,
        },
        "quality": {
            "null_token_counts": dict(sorted(null_counts.items())),
            "duplicate_race_horse_id_count": duplicate_race_horse,
            "duplicate_race_horse_number_count": duplicate_race_number,
            "unknown_venue_runner_count": sum(unknown_venues.values()),
            "unknown_venue_codes": dict(sorted(unknown_venues.items())),
            "unknown_surface_runner_count": sum(unknown_surfaces.values()),
            "unknown_surface_values": dict(sorted(unknown_surfaces.items())),
        },
    }
    return report


def verify_audit_against_manifest(
    report: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    raw_spec = manifest.get("raw_file", {})
    mismatches: list[str] = []
    for key in ("row_count", "race_count", "column_count", "date_min", "date_max"):
        expected = raw_spec.get(key)
        actual = report.get(key)
        if expected is not None and actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    missing_audit = report.get("coverage", {}).get("known_missing_race_ids", {})
    if missing_audit.get("unexpectedly_present"):
        mismatches.append(
            "manifest-declared missing race IDs are present: "
            + ", ".join(missing_audit["unexpectedly_present"])
        )

    coverage_spec = manifest.get("coverage", {})
    expected_year_counts = coverage_spec.get("observed_raw_race_counts", {})
    actual_year_counts = report.get("coverage", {}).get("race_count_by_year", {})
    if expected_year_counts and actual_year_counts != expected_year_counts:
        mismatches.append(
            "race_count_by_year differs from manifest: "
            f"expected {expected_year_counts!r}, got {actual_year_counts!r}"
        )

    outcome_spec = manifest.get("outcome_audit", {})
    actual_outcomes = report.get("outcomes", {})
    expected_status_counts = outcome_spec.get("normalized_runner_counts", {})
    actual_status_counts = actual_outcomes.get("normalized_status_counts", {})
    for status, expected in expected_status_counts.items():
        actual = actual_status_counts.get(status, 0)
        if actual != expected:
            mismatches.append(
                f"outcome {status}: expected {expected!r}, got {actual!r}"
            )
    for key in (
        "declared_runner_count",
        "starter_count",
        "nonstarter_count",
        "dead_heat_race_count",
        "dead_heat_extra_runner_count",
        "pit_c_scoring_ineligible_race_count",
    ):
        expected = outcome_spec.get(key)
        actual = actual_outcomes.get(key)
        if expected is not None and actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise DataContractError("audit does not match manifest: " + "; ".join(mismatches))
