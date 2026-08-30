"""Paired 2024 comparison of cached ablations against a frozen baseline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from horse_pred.artifacts import write_artifact_manifest, write_json
from horse_pred.data import sha256_file
from horse_pred.uncertainty import (
    ModelSpec,
    development_race_metric_table,
    paired_block_bootstrap,
)


def compare_ablation_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 20240830,
) -> dict[str, Any]:
    """Return paired model-family deltas on an identical 2024 race population."""

    baseline_dev = baseline.loc[baseline["split"].eq("development")].copy()
    candidate_dev = candidate.loc[candidate["split"].eq("development")].copy()
    keys = ["race_id", "horse_id"]
    required_identity = [
        "race_date",
        "model_finish_position",
        "course_type",
        "distance",
        "race_class",
        "field_size",
    ]
    for name, frame in (("baseline", baseline_dev), ("candidate", candidate_dev)):
        missing = sorted(set(keys + required_identity + ["split"]).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} predictions have duplicate runner keys")
    baseline_dev["race_id"] = baseline_dev["race_id"].astype("string")
    baseline_dev["horse_id"] = baseline_dev["horse_id"].astype("string")
    candidate_dev["race_id"] = candidate_dev["race_id"].astype("string")
    candidate_dev["horse_id"] = candidate_dev["horse_id"].astype("string")

    candidate_columns = {
        "pred_binary_raw": "candidate_pred_binary_raw",
        "score_lambdarank": "candidate_score_lambdarank",
        "prob_binary_logit_softmax_temperature_2023": "candidate_prob_binary",
        "prob_lambdarank_softmax_temperature_2023": "candidate_prob_lambdarank",
    }
    missing_predictions = sorted(set(candidate_columns).difference(candidate_dev.columns))
    if missing_predictions:
        raise ValueError(f"candidate predictions are missing model columns: {missing_predictions}")
    joined = baseline_dev.merge(
        candidate_dev.loc[:, keys + required_identity + list(candidate_columns)].rename(
            columns=candidate_columns
        ),
        on=keys,
        how="inner",
        sort=False,
        validate="one_to_one",
        suffixes=("", "_candidate_identity"),
    )
    if len(joined) != len(baseline_dev) or len(joined) != len(candidate_dev):
        raise ValueError("baseline and candidate do not share an identical runner population")
    for column in required_identity:
        other = f"{column}_candidate_identity"
        left = joined[column].astype("string").fillna("<NA>")
        right = joined[other].astype("string").fillna("<NA>")
        if not left.eq(right).all():
            raise ValueError(f"baseline and candidate disagree on {column}")
        joined = joined.drop(columns=other)

    specs: Mapping[str, ModelSpec] = {
        "baseline_binary": ModelSpec(
            "prob_binary_logit_softmax_temperature_2023", "pred_binary_raw"
        ),
        "candidate_binary": ModelSpec("candidate_prob_binary", "candidate_pred_binary_raw"),
        "baseline_lambdarank": ModelSpec(
            "prob_lambdarank_softmax_temperature_2023", "score_lambdarank"
        ),
        "candidate_lambdarank": ModelSpec(
            "candidate_prob_lambdarank", "candidate_score_lambdarank"
        ),
    }
    race_metrics = development_race_metric_table(joined, model_specs=specs)
    bootstrap = paired_block_bootstrap(
        race_metrics,
        comparisons=(
            ("candidate_binary", "baseline_binary"),
            ("candidate_lambdarank", "baseline_lambdarank"),
        ),
        n_resamples=n_resamples,
        seed=seed,
        block_length_dates=4,
    )
    return {
        "scope": {
            "race_count": int(race_metrics["race_id"].nunique()),
            "runner_count": len(joined),
            "date_count": int(race_metrics["race_date"].nunique()),
            "retrospective_2025_used": False,
        },
        "bootstrap": bootstrap,
    }


def run_ablation_analysis(
    baseline_predictions: str | Path,
    candidate_predictions: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    n_resamples: int = 10_000,
    seed: int = 20240830,
) -> dict[str, Any]:
    """Compare every registered ablation and write one aggregate artifact."""

    baseline_path = Path(baseline_predictions).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ablation analysis: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.mkdir(parents=True)
    baseline = pd.read_csv(
        baseline_path, dtype={"race_id": "string", "horse_id": "string"}
    )
    results: dict[str, Any] = {}
    sources: dict[str, Any] = {
        "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)}
    }
    for index, (experiment_id, raw_path) in enumerate(candidate_predictions.items()):
        path = Path(raw_path).resolve()
        candidate = pd.read_csv(path, dtype={"race_id": "string", "horse_id": "string"})
        results[experiment_id] = compare_ablation_predictions(
            baseline,
            candidate,
            n_resamples=n_resamples,
            seed=seed + index + 1,
        )
        sources[experiment_id] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {
        "schema_version": 1,
        "analysis_id": "semantic_feature_ablation_2024",
        "n_resamples": n_resamples,
        "seed_root": seed,
        "retrospective_2025_used": False,
        "sources": sources,
        "experiments": results,
    }
    write_json(temporary / "ablation_analysis.json", payload)
    write_artifact_manifest(temporary)
    temporary.rename(output)
    return payload
