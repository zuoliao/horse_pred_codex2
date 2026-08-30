"""Build the PV-04 opt-in cache containing one frozen margin-rating score."""

from __future__ import annotations

import json
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
from horse_pred.rating import (
    RatingSpec,
    attach_scoring_population,
    build_rating_history_from_events,
    prepare_rating_events,
)

MARGIN_RATING_COLUMN = "margin_rating__score_pre"
_PRESENCE_COLUMN = "_margin_rating_present"


def build_margin_rating_score_augmented_cache(
    baseline_cache_path: str | Path,
    rating_history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
    pv03_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join one frozen score to PV-01 while preserving every old feature exactly."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite margin-rating cache: {target}")
    frame, metadata = read_model_frame_cache(baseline_path)
    if any(column.startswith("margin_rating__") for column in frame.columns):
        raise ValueError("baseline cache already contains a margin-rating column")
    scored = attach_scoring_population(rating_history, frame)
    selected = scored.loc[
        :, ["race_id", "horse_id", "modular_rating__score_pre"]
    ].rename(columns={"modular_rating__score_pre": MARGIN_RATING_COLUMN})
    selected[_PRESENCE_COLUMN] = True
    if selected.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("margin-rating history contains duplicate runner keys")
    if not np.isfinite(selected[MARGIN_RATING_COLUMN]).all():
        raise ValueError("margin-rating history contains non-finite scores")

    if pv03_predictions_path is not None:
        reference = pd.read_pickle(Path(pv03_predictions_path).resolve())
        reference = reference.loc[
            :, ["race_id", "horse_id", "modular_rating__score_pre"]
        ].rename(columns={"modular_rating__score_pre": "reference_score"})
        check = selected.merge(
            reference,
            on=["race_id", "horse_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(check) != len(reference):
            raise ValueError("fresh rating history does not cover the PV-03 reference")
        reference_diff = (
            check[MARGIN_RATING_COLUMN].astype(float) - check["reference_score"].astype(float)
        ).abs()
        if float(reference_diff.max()) != 0.0:
            raise ValueError("fresh rating score does not exactly reproduce PV-03")
    else:
        reference_diff = pd.Series([np.nan])

    base = frame.copy()
    base["_cache_order"] = np.arange(len(base), dtype=np.int64)
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
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
    pre2025 = ~augmented["split"].eq("retrospective_test")
    if not augmented.loc[pre2025, _PRESENCE_COLUMN].eq(True).all():  # noqa: E712
        raise ValueError("margin-rating history does not cover every pre-2025 model row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025 margin-rating rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[MARGIN_RATING_COLUMN] = pd.to_numeric(
        augmented[MARGIN_RATING_COLUMN], errors="coerce"
    ).astype("float32")

    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].reset_index(drop=True).equals(
        augmented.loc[:, old_features]
    ):
        raise ValueError("an existing baseline feature changed during cache augmentation")
    feature_columns = [*old_features, MARGIN_RATING_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["margin_rating"] = [MARGIN_RATING_COLUMN]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    sidecar = {
        **metadata,
        "feature_columns": feature_columns,
        "feature_groups_v1": groups,
        "margin_rating": {
            "baseline_cache_sha256": sha256_file(baseline_path),
            "experiment_config_hash": canonical_json_hash(config),
            "rating_spec": config["rating_spec"],
            "source_column": "modular_rating__score_pre",
            "output_column": MARGIN_RATING_COLUMN,
            "stored_dtype": "float32",
            "standalone_temperature_used": False,
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
        "pre2025_nonmissing": int(
            augmented.loc[pre2025, MARGIN_RATING_COLUMN].notna().sum()
        ),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, MARGIN_RATING_COLUMN].notna().sum()
        ),
        "pv03_reference_max_abs_diff": (
            None if reference_diff.isna().all() else float(reference_diff.max())
        ),
    }


def build_margin_rating_cache_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    baseline_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    pv03_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build PV-04 score history after a hard pre-normalization 2024 cutoff."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("candidate_column") != MARGIN_RATING_COLUMN:
        raise ValueError("PV-04 candidate column differs from the code contract")
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(2024)].copy()
    events = prepare_rating_events(normalize_raw(raw), through_year=2024)
    spec = RatingSpec(**config["rating_spec"])
    history = build_rating_history_from_events(events, spec)
    result = build_margin_rating_score_augmented_cache(
        baseline_cache_path,
        history,
        output_path,
        config=config,
        pv03_predictions_path=pv03_predictions_path,
    )
    result["data_fingerprint"] = manifest["raw_file"]["sha256"]
    result["config_sha256"] = sha256_file(config_file)
    return result
