"""PIT-safe final recorded passing-position history for PACE-03."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import (
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_raw_file,
)
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.features import FeatureConfig, is_flat_race

PACE_FINAL_COLUMN = "pace_final__decay_90d__mean_final_position_percentile"
PACE_FINAL_TRANSFORMATION_HASH = "b0bb8efb6b696cabd52e7da2f9e72ce9d5bbbd6d3ca9b6a95180a643e66be52c"
_PRESENCE_COLUMN = "_pace_final_history_row_present"
_ELIGIBLE_PASSING_ORDER = re.compile(r"^[0-9]+(?:-[0-9]+)+$")


@dataclass(frozen=True)
class PaceFinalSpec:
    decay_half_life_days: int = 90

    def __post_init__(self) -> None:
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")

    def transformation_dict(self) -> dict[str, Any]:
        return {
            "feature_id": "PACE-03",
            "source": "passing_order_raw",
            "eligible_race": "JRA flat",
            "eligible_runner": (
                "started and numeric hyphen-separated passing order with at least two segments"
            ),
            "structural_exception": (
                "races with no two-or-more-segment starter supply no observation, "
                "including Niigata turf straight 1000m"
            ),
            "within_race_transform": {
                "checkpoint": "final recorded passing position",
                "rank": "ascending average tie rank among eligible runners",
                "value": "1-(rank-1)/(n_valid-1)",
                "meaning": "1 frontmost, 0 rearmost",
                "min_valid": 2,
            },
            "history": {
                "half_life_days": self.decay_half_life_days,
                "same_date_batch": True,
                "missing_history": "NaN",
            },
            "output_column": PACE_FINAL_COLUMN,
        }


def _validate_frozen_transformation(spec: PaceFinalSpec) -> str:
    actual = canonical_json_hash(spec.transformation_dict())
    if actual != PACE_FINAL_TRANSFORMATION_HASH:
        raise ValueError(f"PACE-03 transformation differs from preregistration: {actual}")
    return actual


def final_position_percentiles(starters: pd.DataFrame) -> pd.Series:
    """Return one race's final-token relative position observations."""

    required = {"started", "passing_order_raw"}
    missing = sorted(required.difference(starters.columns))
    if missing:
        raise ValueError(f"PACE-03 input is missing columns: {missing}")
    output = pd.Series(np.nan, index=starters.index, dtype="float64")
    text = starters["passing_order_raw"].fillna("").astype(str).str.strip()
    eligible_token = text.map(lambda value: bool(_ELIGIBLE_PASSING_ORDER.fullmatch(value)))
    valid = starters["started"].eq(True).fillna(False) & eligible_token  # noqa: E712
    n_valid = int(valid.sum())
    if n_valid < 2:
        return output
    final_position = pd.to_numeric(text.loc[valid].str.split("-").str[-1], errors="raise").astype(float)
    if not np.isfinite(final_position.to_numpy()).all() or final_position.lt(1).any():
        raise ValueError("eligible final passing positions must be finite positive integers")
    ranks = final_position.rank(method="average", ascending=True)
    output.loc[valid] = 1.0 - (ranks - 1.0) / (n_valid - 1.0)
    return output


def build_pace_final_history(
    normalized: pd.DataFrame,
    *,
    spec: PaceFinalSpec | None = None,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Emit PACE-03 history for every flat starter through 2024."""

    if through_year > 2024:
        raise ValueError("PACE-03 must not generate or inspect 2025+ history")
    spec = spec or PaceFinalSpec()
    _validate_frozen_transformation(spec)
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "started",
        "passing_order_raw",
        "course_type",
        "race_class",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"normalized PACE-03 input is missing columns: {missing}")
    frame = normalized.copy()
    years = pd.to_datetime(frame["race_date"], errors="raise").dt.year
    frame = frame.loc[years.le(through_year)].copy()
    frame["_source_position"] = np.arange(len(frame), dtype=np.int64)
    races: list[pd.DataFrame] = []
    feature_config = FeatureConfig()
    for _, race in frame.groupby("race_id", sort=False):
        if not is_flat_race(race, feature_config):
            continue
        starters = race.loc[race["started"].eq(True).fillna(False)].copy()  # noqa: E712
        if starters.empty:
            continue
        starters["_pace_final_observation"] = final_position_percentiles(race).loc[starters.index]
        races.append(starters)
    output_columns = (
        "race_id",
        "horse_id",
        "race_date",
        PACE_FINAL_COLUMN,
        _PRESENCE_COLUMN,
    )
    if not races:
        return pd.DataFrame(columns=output_columns)
    ordered = pd.concat(races, ignore_index=True)
    if ordered.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("PACE-03 history contains duplicate runner keys")
    ordered = ordered.sort_values(["race_date", "race_id", "_source_position"], kind="stable").copy()

    state: dict[object, tuple[float, float, pd.Timestamp]] = {}
    emitted: list[float] = []
    decay_rate = log(2.0) / spec.decay_half_life_days
    for event_date, day in ordered.groupby("race_date", sort=True):
        event_date = pd.Timestamp(event_date).normalize()
        for horse_id in day["horse_id"]:
            record = state.get(horse_id)
            if record is None:
                emitted.append(np.nan)
                continue
            total, weight, state_date = record
            factor = exp(-decay_rate * (event_date - state_date).days)
            total *= factor
            weight *= factor
            emitted.append(total / weight if weight > 0.0 else np.nan)
        for index in day.index:
            observation = float(ordered.at[index, "_pace_final_observation"])
            if not np.isfinite(observation):
                continue
            horse_id = ordered.at[index, "horse_id"]
            record = state.get(horse_id)
            if record is None:
                total = weight = 0.0
            else:
                total, weight, state_date = record
                factor = exp(-decay_rate * (event_date - state_date).days)
                total *= factor
                weight *= factor
            state[horse_id] = (total + observation, weight + 1.0, event_date)

    ordered[PACE_FINAL_COLUMN] = np.asarray(emitted, dtype=np.float64)
    ordered[_PRESENCE_COLUMN] = True
    result = ordered.sort_values("_source_position", kind="stable")
    return result.loc[:, list(output_columns)].reset_index(drop=True)


def build_pace_final_augmented_cache(
    baseline_cache_path: str | Path,
    history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append PACE-03 while preserving the PACE-01 incumbent cache."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite PACE-03 cache: {target}")
    frame, metadata = read_model_frame_cache(baseline_path)
    if any(column.startswith("pace_final__") for column in frame.columns):
        raise ValueError("baseline cache already contains PACE-03")
    required = {"race_id", "horse_id", PACE_FINAL_COLUMN, _PRESENCE_COLUMN}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"PACE-03 history is missing columns: {missing}")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("PACE-03 history contains duplicate runner keys")

    base = frame.copy()
    base["_cache_order"] = np.arange(len(base), dtype=np.int64)
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = history.loc[:, ["race_id", "horse_id", PACE_FINAL_COLUMN, _PRESENCE_COLUMN]].copy()
    selected["race_id"] = selected["race_id"].astype("string")
    selected["horse_id"] = selected["horse_id"].astype("string")
    augmented = base.merge(
        selected,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_cache_order", kind="stable")
    augmented = augmented.drop(columns="_cache_order").reset_index(drop=True)
    years = pd.to_datetime(augmented["race_date"], errors="raise").dt.year
    pre2025 = years.le(2024)
    if not augmented.loc[pre2025, _PRESENCE_COLUMN].eq(True).all():  # noqa: E712
        raise ValueError("PACE-03 history does not cover every pre-2025 row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025+ PACE-03 history rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[PACE_FINAL_COLUMN] = pd.to_numeric(
        augmented[PACE_FINAL_COLUMN], errors="coerce"
    ).astype("float32")
    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].reset_index(drop=True).equals(augmented.loc[:, old_features]):
        raise ValueError("an existing feature changed during PACE-03 augmentation")
    feature_columns = [*old_features, PACE_FINAL_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["pace_final"] = [PACE_FINAL_COLUMN]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    write_json(
        sidecar_path,
        {
            **metadata,
            "row_count": len(augmented),
            "race_count": int(augmented["race_id"].nunique()),
            "feature_columns": feature_columns,
            "feature_groups_v1": groups,
            "pace_final": {
                "baseline_cache_sha256": sha256_file(baseline_path),
                "experiment_config_hash": canonical_json_hash(config),
                "transformation_hash": PACE_FINAL_TRANSFORMATION_HASH,
                "transformation": config["transformation"],
                "stored_dtype": "float32",
                "retrospective_2025_used": False,
            },
        },
    )
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(old_features),
        "candidate_feature_count": len(feature_columns),
        "old_feature_exact": True,
        "pre2025_nonmissing": int(augmented.loc[pre2025, PACE_FINAL_COLUMN].notna().sum()),
        "pre2025_missing": int(augmented.loc[pre2025, PACE_FINAL_COLUMN].isna().sum()),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, PACE_FINAL_COLUMN].notna().sum()
        ),
    }


def load_pace_final_config(path: str | Path) -> tuple[dict[str, Any], PaceFinalSpec]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("candidate_column") != PACE_FINAL_COLUMN:
        raise ValueError("registered PACE-03 column differs from code contract")
    raw = config.get("transformation")
    if not isinstance(raw, dict):
        raise ValueError("PACE-03 config is missing transformation")
    spec = PaceFinalSpec(decay_half_life_days=int(raw["history"]["half_life_days"]))
    if raw != spec.transformation_dict():
        raise ValueError("PACE-03 config transformation differs from code contract")
    actual = _validate_frozen_transformation(spec)
    if config.get("transformation_hash") != actual:
        raise ValueError("PACE-03 config transformation_hash is invalid")
    return config, spec


def build_pace_final_cache_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    baseline_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config, spec = load_pace_final_config(config_file)
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(2024)].copy()
    history = build_pace_final_history(normalize_raw(raw), spec=spec, through_year=2024)
    result = build_pace_final_augmented_cache(
        baseline_cache_path, history, output_path, config=config
    )
    result["raw_rows_after_pre_normalization_cutoff"] = len(raw)
    result["transformation_hash"] = PACE_FINAL_TRANSFORMATION_HASH
    return result
