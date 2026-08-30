"""Forward-only historical race-content features from official result clocks.

The result clock belongs to the completed historical race.  It is never a
feature for that race: all rows on a date are emitted before any result from
that date updates state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.features import FeatureConfig, is_flat_race

RACE_CONTENT_COLUMN = (
    "race_content__decay_90d__mean_signed_time_gap_per_1000m"
)
_PRESENCE_COLUMN = "_race_content_history_row_present"


@dataclass(frozen=True)
class RaceContentSpec:
    """Frozen PV-01 transformation parameters."""

    absolute_cap: float = 5.0
    decay_half_life_days: int = 90
    excluded_statuses: tuple[str, ...] = ("demoted", "disqualified")

    def __post_init__(self) -> None:
        if not np.isfinite(self.absolute_cap) or self.absolute_cap <= 0:
            raise ValueError("absolute_cap must be finite and positive")
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")
        if not self.excluded_statuses:
            raise ValueError("excluded_statuses must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "absolute_cap": self.absolute_cap,
            "decay_half_life_days": self.decay_half_life_days,
            "excluded_statuses": list(self.excluded_statuses),
        }


def parse_result_time_seconds(value: object) -> float:
    """Parse the frozen raw ``M:SS.t`` format, returning NaN otherwise."""

    if value is None:
        return np.nan
    try:
        if bool(pd.isna(value)):
            return np.nan
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    match = pd.Series([text], dtype="string").str.extract(
        r"^(\d+):(\d{2})\.(\d)$", expand=True
    )
    if match.isna().any(axis=None):
        return np.nan
    minutes, seconds, tenths = (int(match.iloc[0, index]) for index in range(3))
    if seconds >= 60:
        return np.nan
    return float(minutes * 60 + seconds + tenths / 10.0)


def signed_time_content_scores(
    race: pd.DataFrame,
    spec: RaceContentSpec | None = None,
) -> pd.Series:
    """Return post-race signed time content indexed like ``race``.

    A sole winner receives its distance-normalized separation from the fastest
    nonwinner.  Numeric nonwinners receive the negative distance-normalized gap
    to the official winner.  Dead-heat winners receive zero.  A race with a
    demotion or disqualification is intentionally left missing.
    """

    spec = spec or RaceContentSpec()
    result = pd.Series(np.nan, index=race.index, dtype="float64")
    required = {
        "finish_position",
        "status",
        "started",
        "time_raw",
        "distance",
    }
    missing = sorted(required.difference(race.columns))
    if missing:
        raise ValueError(f"race-content input is missing columns: {missing}")
    if race["status"].astype("string").isin(spec.excluded_statuses).any():
        return result

    distance_values = pd.to_numeric(race["distance"], errors="coerce").dropna().unique()
    if len(distance_values) != 1 or distance_values[0] <= 0:
        return result
    distance = float(distance_values[0])
    finish = pd.to_numeric(race["finish_position"], errors="coerce")
    started = race["started"].astype("boolean").eq(True)  # noqa: E712
    times = race["time_raw"].map(parse_result_time_seconds)
    timed = started & finish.notna() & times.notna()
    winners = timed & finish.eq(1)
    nonwinners = timed & finish.gt(1)
    if not winners.any() or not nonwinners.any():
        return result

    winner_time = float(times.loc[winners].min())
    if int(winners.sum()) == 1:
        separation = max(float(times.loc[nonwinners].min()) - winner_time, 0.0)
        result.loc[winners] = separation * 1000.0 / distance
    else:
        result.loc[winners] = 0.0
    result.loc[nonwinners] = (
        -np.maximum(times.loc[nonwinners].astype(float) - winner_time, 0.0)
        * 1000.0
        / distance
    )
    return result.clip(-spec.absolute_cap, spec.absolute_cap)


def build_race_content_history(
    normalized: pd.DataFrame,
    *,
    spec: RaceContentSpec | None = None,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Emit one PIT feature row per supplied runner through ``through_year``."""

    if through_year > 2024:
        raise ValueError("PV-01 must not generate or inspect 2025+ race content")
    spec = spec or RaceContentSpec()
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "finish_position",
        "status",
        "started",
        "time_raw",
        "distance",
        "course_type",
        "race_class",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"normalized race-content input is missing columns: {missing}")

    dates = pd.to_datetime(normalized["race_date"], errors="raise").dt.normalize()
    source = normalized.loc[dates.dt.year.le(through_year)].copy()
    source["_content_date"] = dates.loc[source.index]
    source["_content_order"] = np.arange(len(source), dtype=np.int64)
    source = source.sort_values(
        ["_content_date", "race_id", "_content_order"], kind="stable"
    )

    # horse -> (decayed weighted sum, decayed observation weight, state date)
    state: dict[object, tuple[float, float, pd.Timestamp]] = {}
    emitted: list[pd.DataFrame] = []
    feature_config = FeatureConfig()
    decay_rate = log(2.0) / spec.decay_half_life_days

    for event_date, date_frame in source.groupby("_content_date", sort=True):
        values: list[float] = []
        for horse_id in date_frame["horse_id"]:
            record = state.get(horse_id)
            if record is None:
                values.append(np.nan)
                continue
            total, weight, state_date = record
            factor = exp(-decay_rate * (event_date - state_date).days)
            weight *= factor
            values.append(total * factor / weight if weight > 0.0 else np.nan)

        emitted.append(
            pd.DataFrame(
                {
                    "race_id": date_frame["race_id"].astype("string").to_numpy(),
                    "horse_id": date_frame["horse_id"].astype("string").to_numpy(),
                    "race_date": event_date,
                    RACE_CONTENT_COLUMN: np.asarray(values, dtype=np.float64),
                    _PRESENCE_COLUMN: True,
                    "_content_order": date_frame["_content_order"].to_numpy(),
                }
            )
        )

        # No result from event_date becomes visible until every row above was emitted.
        pending: list[tuple[object, float]] = []
        for _, race in date_frame.groupby("race_id", sort=False):
            if not is_flat_race(race, feature_config):
                continue
            scores = signed_time_content_scores(race, spec)
            for index, score in scores.items():
                if np.isfinite(score):
                    pending.append((race.at[index, "horse_id"], float(score)))

        for horse_id, score in pending:
            if pd.isna(horse_id):
                continue
            record = state.get(horse_id)
            if record is None:
                total = weight = 0.0
            else:
                total, weight, state_date = record
                factor = exp(-decay_rate * (event_date - state_date).days)
                total *= factor
                weight *= factor
            state[horse_id] = (total + score, weight + 1.0, event_date)

    if not emitted:
        return pd.DataFrame(
            columns=(
                "race_id",
                "horse_id",
                "race_date",
                RACE_CONTENT_COLUMN,
                _PRESENCE_COLUMN,
            )
        )
    result = pd.concat(emitted, ignore_index=True).sort_values(
        "_content_order", kind="stable"
    )
    return result.drop(columns="_content_order").reset_index(drop=True)


def build_race_content_augmented_cache(
    baseline_cache_path: str | Path,
    content_history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Join PV-01 history to the ignored baseline cache with a 2025 firewall."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite race-content cache: {target}")
    frame, metadata = read_model_frame_cache(baseline_path)
    required = {"race_id", "horse_id", RACE_CONTENT_COLUMN, _PRESENCE_COLUMN}
    missing = sorted(required.difference(content_history.columns))
    if missing:
        raise ValueError(f"race-content history is missing columns: {missing}")
    if content_history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("race-content history contains duplicate runner keys")

    base = frame.copy()
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = content_history.loc[
        :, ["race_id", "horse_id", RACE_CONTENT_COLUMN, _PRESENCE_COLUMN]
    ].copy()
    selected["race_id"] = selected["race_id"].astype("string")
    selected["horse_id"] = selected["horse_id"].astype("string")
    augmented = base.merge(
        selected,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    pre2025 = ~augmented["split"].eq("retrospective_test")
    if not augmented.loc[pre2025, _PRESENCE_COLUMN].eq(True).all():  # noqa: E712
        raise ValueError("race-content history does not cover every pre-2025 model row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025 race-content rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[RACE_CONTENT_COLUMN] = pd.to_numeric(
        augmented[RACE_CONTENT_COLUMN], errors="coerce"
    ).astype("float32")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    feature_columns = [*metadata["feature_columns"], RACE_CONTENT_COLUMN]
    sidecar = {
        **metadata,
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "feature_columns": feature_columns,
        "race_content_time": {
            "baseline_cache_sha256": sha256_file(baseline_path),
            "experiment_config_hash": canonical_json_hash(config),
            "spec": config["race_content"],
            "retrospective_2025_used": False,
        },
    }
    write_json(sidecar_path, sidecar)
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(metadata["feature_columns"]),
        "candidate_feature_count": len(feature_columns),
        "pre2025_nonmissing": int(
            augmented.loc[pre2025, RACE_CONTENT_COLUMN].notna().sum()
        ),
        "pre2025_missing": int(
            augmented.loc[pre2025, RACE_CONTENT_COLUMN].isna().sum()
        ),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, RACE_CONTENT_COLUMN].notna().sum()
        ),
    }


def load_race_content_config(path: str | Path) -> tuple[dict[str, Any], RaceContentSpec]:
    """Load the registered config and construct its frozen transformation."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = config["race_content"]
    if config.get("candidate_column") != RACE_CONTENT_COLUMN:
        raise ValueError("registered race-content column differs from code contract")
    spec = RaceContentSpec(
        absolute_cap=float(raw["absolute_cap"]),
        decay_half_life_days=int(raw["decay_half_life_days"]),
        excluded_statuses=tuple(raw["exclude_race_if_status_present"]),
    )
    return config, spec
