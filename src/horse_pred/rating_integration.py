"""Join a frozen standalone rating artifact to the local model-frame cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.cache_control import MODULAR_RATING_COLUMNS
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache


def build_rating_augmented_cache(
    baseline_cache_path: str | Path,
    rating_predictions_path: str | Path,
    rating_spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a 2025-unread rating cache while preserving all baseline columns."""

    baseline_path = Path(baseline_cache_path).resolve()
    predictions_path = Path(rating_predictions_path).resolve()
    spec_path = Path(rating_spec_path).resolve()
    target = Path(output_path).resolve()
    if target.exists() or target.with_suffix(f"{target.suffix}.meta.json").exists():
        raise FileExistsError(f"refusing to overwrite rating cache: {target}")
    frame, metadata = read_model_frame_cache(baseline_path)
    ratings = pd.read_pickle(predictions_path)
    missing = sorted(
        set(("race_id", "horse_id", *MODULAR_RATING_COLUMNS)).difference(ratings.columns)
    )
    if missing:
        raise ValueError(f"rating artifact is missing columns: {missing}")
    if ratings["split"].eq("retrospective_test").any():
        raise ValueError("rating artifact must not contain 2025 retrospective rows")
    if ratings.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("rating artifact contains duplicate runner keys")
    base = frame.copy()
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = ratings.loc[:, ["race_id", "horse_id", *MODULAR_RATING_COLUMNS]].copy()
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
    if augmented.loc[pre2025, MODULAR_RATING_COLUMNS].isna().any().any():
        raise ValueError("rating artifact does not cover every pre-2025 model row")
    if augmented.loc[~pre2025, MODULAR_RATING_COLUMNS].notna().any().any():
        raise ValueError("2025 modular ratings must remain unavailable")
    for column in MODULAR_RATING_COLUMNS:
        augmented[column] = pd.to_numeric(augmented[column], errors="coerce").astype(
            "float32"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    feature_columns = [*metadata["feature_columns"], *MODULAR_RATING_COLUMNS]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    sidecar = {
        **metadata,
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "feature_columns": feature_columns,
        "rating_module": {
            "predictions_sha256": sha256_file(predictions_path),
            "spec_sha256": sha256_file(spec_path),
            "spec": spec,
            "retrospective_2025_used": False,
        },
    }
    write_json(target.with_suffix(f"{target.suffix}.meta.json"), sidecar)
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(metadata["feature_columns"]),
        "candidate_feature_count": len(feature_columns),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, MODULAR_RATING_COLUMNS].notna().sum().sum()
        ),
    }
