"""Local, ignored cache for the expensive point-in-time model frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.features import FeatureDataset


def write_model_frame_cache(
    path: str | Path,
    frame: pd.DataFrame,
    dataset: FeatureDataset,
    *,
    data_fingerprint: str,
) -> None:
    """Atomically write a local pandas cache and its reproducibility metadata."""

    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite model-frame cache: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    frame.to_pickle(temporary)
    temporary.replace(target)
    write_json(
        target.with_suffix(f"{target.suffix}.meta.json"),
        {
            "schema_version": 1,
            "format": "pandas_pickle",
            "data_fingerprint": data_fingerprint,
            "row_count": len(frame),
            "race_count": int(frame["race_id"].nunique()),
            "feature_columns": list(dataset.feature_columns),
            "feature_groups_v1": {
                name: list(columns) for name, columns in dataset.feature_groups.items()
            },
        },
    )


def read_model_frame_cache(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a cache only when its sidecar and shape agree."""

    target = Path(path).resolve()
    sidecar = target.with_suffix(f"{target.suffix}.meta.json")
    if not target.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"model-frame cache or sidecar is missing: {target}")
    import json

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    frame = pd.read_pickle(target)
    if len(frame) != int(metadata["row_count"]):
        raise ValueError("model-frame cache row count disagrees with sidecar")
    if frame["race_id"].nunique() != int(metadata["race_count"]):
        raise ValueError("model-frame cache race count disagrees with sidecar")
    missing = sorted(set(metadata["feature_columns"]).difference(frame.columns))
    if missing:
        raise ValueError(f"model-frame cache is missing feature columns: {missing}")
    return frame, metadata
