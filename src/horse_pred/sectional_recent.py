"""PIT-safe race-relative last-3F history for SEC-3F-001.

Each eligible past observation is the horse's own within-race last-3F speed
percentile.  Absolute sectional seconds never cross race boundaries.  Every
target row on a date is emitted before any observation from that date updates
the 90-day state.
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
from horse_pred.data import (
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_raw_file,
)
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.features import FeatureConfig, is_flat_race

SECTIONAL_RECENT_COLUMN = "sectional__decay_90d__mean_last_3f_speed_percentile"
SECTIONAL_RECENT_TRANSFORMATION_HASH = "744a157bb4a5053775b990a56ebf9e1208cd9f7b83a00f72327766f612d0b1b6"
_PRESENCE_COLUMN = "_sectional_recent_history_row_present"


@dataclass(frozen=True)
class SectionalRecentSpec:
    """Frozen SEC-3F transformation parameters."""

    decay_half_life_days: int = 90

    def __post_init__(self) -> None:
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")

    def transformation_dict(self) -> dict[str, Any]:
        """Return the exact audited payload whose canonical hash is frozen."""

        return {
            "feature_id": "SEC-3F",
            "source": "last_3f_seconds",
            "eligible_race": "JRA flat",
            "eligible_runner": "started and finite last_3f_seconds",
            "within_race_transform": {
                "rank": "ascending average tie rank",
                "value": "1-(rank-1)/(n_valid-1)",
                "meaning": "1 fastest, 0 slowest",
                "min_valid": 2,
            },
            "history": {
                "half_life_days": self.decay_half_life_days,
                "same_date_batch": True,
                "missing_history": "NaN",
            },
            "output_column": SECTIONAL_RECENT_COLUMN,
        }


def _validate_frozen_transformation(spec: SectionalRecentSpec) -> str:
    actual = canonical_json_hash(spec.transformation_dict())
    if actual != SECTIONAL_RECENT_TRANSFORMATION_HASH:
        raise ValueError(f"SEC-3F transformation differs from the preregistered hash: {actual}")
    return actual


def last_3f_speed_percentiles(starters: pd.DataFrame) -> pd.Series:
    """Return one race's frozen relative last-3F observations.

    The input may contain non-starters and missing/non-finite sectionals. Only
    started rows with finite ``last_3f_seconds`` enter ``n_valid`` and ranking.
    When fewer than two valid values exist, every observation is missing.
    """

    required = {"started", "last_3f_seconds"}
    missing = sorted(required.difference(starters.columns))
    if missing:
        raise ValueError(f"sectional input is missing columns: {missing}")
    output = pd.Series(np.nan, index=starters.index, dtype="float64")
    seconds = pd.to_numeric(starters["last_3f_seconds"], errors="coerce")
    finite = pd.Series(np.isfinite(seconds.to_numpy(dtype="float64")), index=starters.index)
    valid = starters["started"].eq(True).fillna(False) & finite  # noqa: E712
    n_valid = int(valid.sum())
    if n_valid < 2:
        return output
    ranks = seconds.loc[valid].rank(method="average", ascending=True)
    output.loc[valid] = 1.0 - (ranks - 1.0) / (n_valid - 1.0)
    return output


def build_sectional_recent_history(
    normalized: pd.DataFrame,
    *,
    spec: SectionalRecentSpec | None = None,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Emit SEC-3F history for every flat-race starter through 2024."""

    if through_year > 2024:
        raise ValueError("SEC-3F must not generate or inspect 2025+ history")
    spec = spec or SectionalRecentSpec()
    _validate_frozen_transformation(spec)
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "started",
        "last_3f_seconds",
        "course_type",
        "race_class",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"normalized SEC-3F input is missing columns: {missing}")

    frame = normalized.copy()
    years = pd.to_datetime(frame["race_date"], errors="raise").dt.year
    frame = frame.loc[years.le(through_year)].copy()
    frame["_source_position"] = np.arange(len(frame), dtype=np.int64)
    eligible_races: list[pd.DataFrame] = []
    feature_config = FeatureConfig()
    for _, race in frame.groupby("race_id", sort=False):
        if not is_flat_race(race, feature_config):
            continue
        starters = race.loc[race["started"].eq(True).fillna(False)].copy()  # noqa: E712
        if starters.empty:
            continue
        starters["_sectional_observation"] = last_3f_speed_percentiles(race).loc[starters.index]
        eligible_races.append(starters)

    output_columns = (
        "race_id",
        "horse_id",
        "race_date",
        SECTIONAL_RECENT_COLUMN,
        _PRESENCE_COLUMN,
    )
    if not eligible_races:
        return pd.DataFrame(columns=output_columns)
    ordered = pd.concat(eligible_races, ignore_index=True)
    if ordered.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("SEC-3F history contains duplicate runner keys")
    ordered = ordered.sort_values(["race_date", "race_id", "_source_position"], kind="stable").copy()

    # horse -> (decayed total, decayed weight, state date)
    state: dict[object, tuple[float, float, pd.Timestamp]] = {}
    emitted_values: list[float] = []
    decay_rate = log(2.0) / spec.decay_half_life_days

    for event_date, day in ordered.groupby("race_date", sort=True):
        event_date = pd.Timestamp(event_date).normalize()
        for horse_id in day["horse_id"]:
            record = state.get(horse_id)
            if record is None:
                emitted_values.append(np.nan)
                continue
            total, weight, state_date = record
            factor = exp(-decay_rate * (event_date - state_date).days)
            total *= factor
            weight *= factor
            emitted_values.append(total / weight if weight > 0.0 else np.nan)

        # No observation from event_date is visible to another race that day.
        for index in day.index:
            observation = float(ordered.at[index, "_sectional_observation"])
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

    ordered[SECTIONAL_RECENT_COLUMN] = np.asarray(emitted_values, dtype=np.float64)
    ordered[_PRESENCE_COLUMN] = True
    result = ordered.sort_values("_source_position", kind="stable")
    return result.loc[:, list(output_columns)].reset_index(drop=True)


def build_sectional_recent_augmented_cache(
    baseline_cache_path: str | Path,
    sectional_history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append one SEC-3F column while preserving an arbitrary baseline cache."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite SEC-3F cache: {target}")

    frame, metadata = read_model_frame_cache(baseline_path)
    if any(column.startswith("sectional__") for column in frame.columns):
        raise ValueError("baseline cache already contains a sectional column")
    required = {
        "race_id",
        "horse_id",
        SECTIONAL_RECENT_COLUMN,
        _PRESENCE_COLUMN,
    }
    missing = sorted(required.difference(sectional_history.columns))
    if missing:
        raise ValueError(f"sectional history is missing columns: {missing}")
    if sectional_history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("sectional history contains duplicate runner keys")
    if "race_date" not in frame:
        raise ValueError("baseline cache is missing race_date")

    base = frame.copy()
    base["_cache_order"] = np.arange(len(base), dtype=np.int64)
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = sectional_history.loc[
        :,
        ["race_id", "horse_id", SECTIONAL_RECENT_COLUMN, _PRESENCE_COLUMN],
    ].copy()
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
        raise ValueError("sectional history does not cover every pre-2025 model row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025+ sectional-history rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[SECTIONAL_RECENT_COLUMN] = pd.to_numeric(augmented[SECTIONAL_RECENT_COLUMN], errors="coerce").astype(
        "float32"
    )

    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].reset_index(drop=True).equals(augmented.loc[:, old_features]):
        raise ValueError("an existing baseline feature changed during SEC-3F augmentation")
    feature_columns = [*old_features, SECTIONAL_RECENT_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["sectional"] = [SECTIONAL_RECENT_COLUMN]

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    sidecar = {
        **metadata,
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "feature_columns": feature_columns,
        "feature_groups_v1": groups,
        "sectional_recent": {
            "baseline_cache_sha256": sha256_file(baseline_path),
            "experiment_config_hash": canonical_json_hash(config),
            "transformation_hash": SECTIONAL_RECENT_TRANSFORMATION_HASH,
            "transformation": config["transformation"],
            "stored_dtype": "float32",
            "retrospective_2025_used": False,
        },
    }
    write_json(sidecar_path, sidecar)
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(old_features),
        "candidate_feature_count": len(feature_columns),
        "old_feature_exact": True,
        "pre2025_nonmissing": int(augmented.loc[pre2025, SECTIONAL_RECENT_COLUMN].notna().sum()),
        "pre2025_missing": int(augmented.loc[pre2025, SECTIONAL_RECENT_COLUMN].isna().sum()),
        "retrospective_2025_feature_nonmissing": int(augmented.loc[~pre2025, SECTIONAL_RECENT_COLUMN].notna().sum()),
    }


def load_sectional_recent_config(
    path: str | Path,
) -> tuple[dict[str, Any], SectionalRecentSpec]:
    """Load and validate the frozen SEC-3F preregistration."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("candidate_column") != SECTIONAL_RECENT_COLUMN:
        raise ValueError("registered SEC-3F column differs from code contract")
    raw = config.get("transformation")
    if not isinstance(raw, dict):
        raise ValueError("SEC-3F config is missing transformation")
    history = raw.get("history", {})
    spec = SectionalRecentSpec(decay_half_life_days=int(history["half_life_days"]))
    if raw != spec.transformation_dict():
        raise ValueError("SEC-3F config transformation differs from code contract")
    actual = _validate_frozen_transformation(spec)
    if config.get("transformation_hash") != actual:
        raise ValueError("SEC-3F config transformation_hash is invalid")
    return config, spec


def build_sectional_recent_cache_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    baseline_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build SEC-3F after removing 2025+ before normalization."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config, spec = load_sectional_recent_config(config_file)
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(2024)].copy()
    history = build_sectional_recent_history(normalize_raw(raw), spec=spec, through_year=2024)
    result = build_sectional_recent_augmented_cache(
        baseline_cache_path,
        history,
        output_path,
        config=config,
    )
    result["raw_rows_after_pre_normalization_cutoff"] = len(raw)
    result["transformation_hash"] = SECTIONAL_RECENT_TRANSFORMATION_HASH
    return result
