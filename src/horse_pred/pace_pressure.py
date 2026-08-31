"""Current-field rival pressure derived from frozen PIT PACE-01 histories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.pace_recent import PACE_RECENT_COLUMN

PACE_PRESSURE_COLUMN = "pace_pressure__current_field__rival_front_excess_sum"
PACE_PRESSURE_TRANSFORMATION_HASH = "59c5d5a818498561956b1101fc990944ad475d7c8645e59269c10c414c697285"


@dataclass(frozen=True)
class PacePressureSpec:
    """Frozen PACE-02 transformation contract."""

    def transformation_dict(self) -> dict[str, Any]:
        return {
            "feature_id": "PACE-02",
            "source": PACE_RECENT_COLUMN,
            "eligible_runner": (
                "current-field row with at least one other runner having finite PACE-01 history"
            ),
            "within_race_transform": {
                "runner_front_excess": "max(x-0.5,0)",
                "meaning": "continuous excess above the frozen PACE-01 percentile midpoint",
                "aggregate": "sum over finite other-runner front excess",
                "exclude_target_runner": True,
                "missing": "NaN when no finite opponent history exists",
            },
            "timestamp_semantics": (
                "current field composed only from PIT PACE-01 histories already available at target date"
            ),
            "output_column": PACE_PRESSURE_COLUMN,
        }


def _validate_frozen_transformation(spec: PacePressureSpec) -> str:
    actual = canonical_json_hash(spec.transformation_dict())
    if actual != PACE_PRESSURE_TRANSFORMATION_HASH:
        raise ValueError(f"PACE-02 transformation differs from preregistration: {actual}")
    return actual


def rival_front_excess_sum(frame: pd.DataFrame) -> pd.Series:
    """Compute the target-excluded pressure for every current-field row."""

    required = {"race_id", "horse_id", "context__field_size_rows", PACE_RECENT_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"PACE-02 input is missing columns: {missing}")
    if frame.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("PACE-02 input contains duplicate runner keys")

    field_size = pd.to_numeric(frame["context__field_size_rows"], errors="coerce")
    group_size = frame.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)
    if field_size.isna().any() or not np.array_equal(field_size.to_numpy(), group_size.to_numpy()):
        raise ValueError("PACE-02 field size differs from current race rows")

    source = pd.to_numeric(frame[PACE_RECENT_COLUMN], errors="coerce").astype(float)
    invalid = source.notna() & (~np.isfinite(source) | source.lt(0.0) | source.gt(1.0))
    if invalid.any():
        raise ValueError("PACE-01 source must be finite in [0,1] or missing")
    finite = source.notna()
    own_excess = (source - 0.5).clip(lower=0.0).where(finite, 0.0)
    grouped = frame["race_id"]
    total_excess = own_excess.groupby(grouped, sort=False).transform("sum")
    valid_count = finite.astype(int).groupby(grouped, sort=False).transform("sum")
    rival_valid_count = valid_count - finite.astype(int)
    pressure = (total_excess - own_excess).astype(float)
    return pressure.where(rival_valid_count.ge(1), np.nan)


def load_pace_pressure_config(path: str | Path) -> tuple[dict[str, Any], PacePressureSpec]:
    """Load and validate the frozen PACE-02 preregistration."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("candidate_column") != PACE_PRESSURE_COLUMN:
        raise ValueError("registered PACE-02 column differs from code contract")
    spec = PacePressureSpec()
    raw = config.get("transformation")
    if raw != spec.transformation_dict():
        raise ValueError("PACE-02 config transformation differs from code contract")
    actual = _validate_frozen_transformation(spec)
    if config.get("transformation_hash") != actual:
        raise ValueError("PACE-02 config transformation_hash is invalid")
    return config, spec


def build_pace_pressure_cache(
    input_cache_path: str | Path,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append one PACE-02 column to the accepted PACE-01 cache."""

    input_path = Path(input_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite PACE-02 cache: {target}")
    frame, metadata = read_model_frame_cache(input_path)
    if any(column.startswith("pace_pressure__") for column in frame.columns):
        raise ValueError("input cache already contains pace pressure")
    if PACE_RECENT_COLUMN not in metadata["feature_columns"]:
        raise ValueError("PACE-02 requires frozen PACE-01 in the input feature contract")

    augmented = frame.copy()
    augmented[PACE_PRESSURE_COLUMN] = rival_front_excess_sum(frame).astype("float32")
    years = pd.to_datetime(augmented["race_date"], errors="raise").dt.year
    if augmented.loc[years.ge(2025), PACE_PRESSURE_COLUMN].notna().any():
        raise ValueError("2025+ PACE-02 values must remain unavailable")

    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].equals(augmented.loc[:, old_features]):
        raise ValueError("an existing feature changed during PACE-02 augmentation")
    feature_columns = [*old_features, PACE_PRESSURE_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["pace_pressure"] = [PACE_PRESSURE_COLUMN]

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
        "pace_pressure": {
            "input_cache_sha256": sha256_file(input_path),
            "experiment_config_hash": canonical_json_hash(config),
            "transformation_hash": PACE_PRESSURE_TRANSFORMATION_HASH,
            "transformation": config["transformation"],
            "stored_dtype": "float32",
            "retrospective_2025_used": False,
        },
    }
    write_json(sidecar_path, sidecar)
    pre2025 = years.le(2024)
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(old_features),
        "candidate_feature_count": len(feature_columns),
        "old_feature_exact": True,
        "pre2025_nonmissing": int(augmented.loc[pre2025, PACE_PRESSURE_COLUMN].notna().sum()),
        "pre2025_missing": int(augmented.loc[pre2025, PACE_PRESSURE_COLUMN].isna().sum()),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, PACE_PRESSURE_COLUMN].notna().sum()
        ),
        "transformation_hash": PACE_PRESSURE_TRANSFORMATION_HASH,
    }


def build_pace_pressure_cache_from_config(
    *,
    repo_root: str | Path,
    input_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Resolve the frozen config and build PACE-02 without raw outcome access."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config, _ = load_pace_pressure_config(config_file)
    expected = config["source_scope"]["input_cache_sha256"]
    actual = sha256_file(input_cache_path)
    if actual != expected:
        raise ValueError(f"PACE-02 input cache fingerprint differs: {actual}")
    return build_pace_pressure_cache(input_cache_path, output_path, config=config)
