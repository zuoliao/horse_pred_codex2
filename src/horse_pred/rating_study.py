"""Reproducible R0-R5 standalone rating study orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from horse_pred.artifacts import write_artifact_manifest, write_json
from horse_pred.data import load_manifest, load_raw, normalize_raw, sha256_file, verify_raw_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import (
    ndcg_at_k,
    race_brier_score,
    race_log_loss,
    top_k_winner_mass,
)
from horse_pred.modeling import apply_temperature, fit_temperature
from horse_pred.rating import (
    RatingSpec,
    attach_scoring_population,
    build_rating_history_from_events,
    evaluate_rating_rows,
    prepare_rating_events,
)


def _primary_metrics(frame: pd.DataFrame, years: list[int]) -> dict[str, float | int]:
    selected = frame.loc[frame["race_date"].dt.year.isin(years)].sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    probabilities = selected["modular_rating__raw_win_probability_pre"].tolist()
    positions = selected["model_finish_position"].astype(int).tolist()
    race_ids = selected["race_id"].tolist()
    scores = selected["modular_rating__score_pre"].tolist()
    return {
        "race_count": int(selected["race_id"].nunique()),
        "runner_count": len(selected),
        "race_log_loss": race_log_loss(probabilities, positions, race_ids),
        "race_brier": race_brier_score(probabilities, positions, race_ids),
        "ndcg_at_3": ndcg_at_k(scores, positions, race_ids, k=3),
        "top_1": top_k_winner_mass(scores, positions, race_ids, k=1),
    }


def _annual_selection_metrics(
    frame: pd.DataFrame, years: list[int]
) -> dict[str, Any]:
    annual = {str(year): _primary_metrics(frame, [year]) for year in years}
    metric_names = ("race_log_loss", "race_brier", "ndcg_at_3", "top_1")
    macro = {
        metric: sum(float(payload[metric]) for payload in annual.values()) / len(annual)
        for metric in metric_names
    }
    return {"annual": annual, "annual_macro": macro}


def _selection_key(payload: dict[str, Any], complexity: tuple[float, ...]) -> tuple[float, ...]:
    metrics = payload["annual_macro"]
    return (
        float(metrics["race_log_loss"]),
        float(metrics["race_brier"]),
        -float(metrics["ndcg_at_3"]),
        *complexity,
    )


def _candidate_improvement(
    baseline: dict[str, float | int], candidate: dict[str, float | int]
) -> dict[str, float]:
    return {
        "race_log_loss": float(baseline["race_log_loss"])
        - float(candidate["race_log_loss"]),
        "race_brier": float(baseline["race_brier"])
        - float(candidate["race_brier"]),
        "ndcg_at_3": float(candidate["ndcg_at_3"])
        - float(baseline["ndcg_at_3"]),
        "top_1": float(candidate["top_1"]) - float(baseline["top_1"]),
    }


def _accepted(improvement: dict[str, float], guardrails: dict[str, float]) -> bool:
    return bool(
        improvement["race_log_loss"] > 0.0
        and improvement["race_brier"]
        >= float(guardrails["race_brier_improvement_min"])
        and improvement["ndcg_at_3"]
        >= float(guardrails["ndcg_at_3_improvement_min"])
        and improvement["top_1"] >= float(guardrails["top_1_improvement_min"])
    )


def run_rating_study(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    model_cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run preregistered R0-R5 without reading or evaluating 2025."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rating study: {output}")
    output.mkdir(parents=True)
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    model_frame, model_meta = read_model_frame_cache(root / model_cache_path)
    raw = normalize_raw(load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"]))
    events = prepare_rating_events(raw, through_year=2024)
    del raw

    def score(spec: RatingSpec) -> pd.DataFrame:
        history = build_rating_history_from_events(events, spec)
        return attach_scoring_population(history, model_frame)

    r0_spec = RatingSpec(**config["r0"])
    r0_frame = score(r0_spec)
    reference = model_frame.loc[
        ~model_frame["split"].eq("retrospective_test"),
        ["race_id", "horse_id", "rating__horse_elo_pre"],
    ].copy()
    reference["race_id"] = reference["race_id"].astype("string")
    reference["horse_id"] = reference["horse_id"].astype("string")
    r0_control = r0_frame[["race_id", "horse_id", "global_state_pre"]].merge(
        reference,
        on=["race_id", "horse_id"],
        validate="one_to_one",
    )
    differences = (
        r0_control["global_state_pre"].astype(float)
        - r0_control["rating__horse_elo_pre"].astype(float)
    ).abs()
    r0_result = {
        "passed": bool(len(r0_control) == len(reference) and differences.max() == 0.0),
        "runner_count": len(r0_control),
        "max_abs_diff": float(differences.max()),
        "mismatch_count": int(differences.ne(0.0).sum()),
        "spec": r0_spec.as_dict(),
    }
    if not r0_result["passed"]:
        raise RuntimeError(f"R0 current-Elo control failed: {r0_result}")

    r1 = {
        split: evaluate_rating_rows(r0_frame, split)
        for split in ("model_validation", "calibration", "development")
    }
    selection_years = [int(year) for year in config["data"]["parameter_selection_years"]]

    r2_candidates: list[dict[str, Any]] = []
    r2_frames: dict[tuple[float, float], pd.DataFrame] = {}
    for k in config["r2_grid"]["k"]:
        for scale in config["r2_grid"]["scale"]:
            spec = RatingSpec(family="pairwise_elo", k=float(k), scale=float(scale))
            frame = score(spec)
            metrics = _annual_selection_metrics(frame, selection_years)
            r2_candidates.append({"spec": spec.as_dict(), **metrics})
            r2_frames[(float(k), float(scale))] = frame
    r2_selected = min(
        r2_candidates,
        key=lambda candidate: _selection_key(
            candidate,
            (
                abs(float(candidate["spec"]["k"]) - 24.0),
                abs(float(candidate["spec"]["scale"]) - 400.0),
            ),
        ),
    )
    r2_spec = RatingSpec(**r2_selected["spec"])
    r2_frame = r2_frames[(r2_spec.k, r2_spec.scale)]
    r2_validation = _primary_metrics(r2_frame, [2022])
    del r2_frames

    r3_candidates: list[dict[str, Any]] = []
    r3_frames: dict[float, pd.DataFrame] = {}
    for learning_rate in config["r3_grid"]["learning_rate"]:
        spec = RatingSpec(
            family="online_top1_pl", learning_rate=float(learning_rate)
        )
        frame = score(spec)
        metrics = _annual_selection_metrics(frame, selection_years)
        r3_candidates.append({"spec": spec.as_dict(), **metrics})
        r3_frames[float(learning_rate)] = frame
    r3_selected = min(
        r3_candidates,
        key=lambda candidate: _selection_key(
            candidate, (float(candidate["spec"]["learning_rate"]),)
        ),
    )
    r3_spec = RatingSpec(**r3_selected["spec"])
    r3_frame = r3_frames[r3_spec.learning_rate]
    r3_validation = _primary_metrics(r3_frame, [2022])
    r3_improvement = _candidate_improvement(r2_validation, r3_validation)
    r3_accepted = _accepted(r3_improvement, config["selection"]["guardrails"])
    selected_global_spec = r3_spec if r3_accepted else r2_spec
    selected_global_frame = r3_frame if r3_accepted else r2_frame
    selected_global_validation = r3_validation if r3_accepted else r2_validation
    del r3_frames

    r4_candidates: list[dict[str, Any]] = []
    r4_frames: dict[float, pd.DataFrame] = {}
    for weight in config["r4_surface_blend_weights"]:
        payload = selected_global_spec.as_dict()
        payload["surface_blend_weight"] = float(weight)
        spec = RatingSpec(**payload)
        frame = score(spec)
        metrics = _annual_selection_metrics(frame, selection_years)
        r4_candidates.append({"spec": spec.as_dict(), **metrics})
        r4_frames[float(weight)] = frame
    r4_selected = min(
        r4_candidates,
        key=lambda candidate: _selection_key(
            candidate, (float(candidate["spec"]["surface_blend_weight"]),)
        ),
    )
    r4_spec = RatingSpec(**r4_selected["spec"])
    r4_frame = r4_frames[r4_spec.surface_blend_weight]
    r4_validation = _primary_metrics(r4_frame, [2022])
    r4_improvement = _candidate_improvement(
        selected_global_validation, r4_validation
    )
    r4_accepted = _accepted(r4_improvement, config["selection"]["guardrails"])
    final_spec = r4_spec if r4_accepted else selected_global_spec
    final_frame = r4_frame if r4_accepted else selected_global_frame

    calibration = final_frame.loc[final_frame["split"].eq("calibration")].sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    calibrator = fit_temperature(
        calibration["modular_rating__score_pre"],
        calibration["race_id"],
        calibration["model_finish_position"],
    )
    development = final_frame.loc[final_frame["split"].eq("development")].copy()
    development = development.sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    development["modular_rating__calibrated_win_probability_2023"] = apply_temperature(
        calibrator,
        development["modular_rating__score_pre"],
        development["race_id"],
    )
    calibrated_frame = final_frame.copy()
    calibrated_frame["modular_rating__calibrated_win_probability_2023"] = pd.NA
    calibrated_frame.loc[
        calibrated_frame["split"].eq("development"),
        "modular_rating__calibrated_win_probability_2023",
    ] = development["modular_rating__calibrated_win_probability_2023"].to_numpy()
    original_probability = calibrated_frame["modular_rating__raw_win_probability_pre"]
    calibrated_frame["modular_rating__raw_win_probability_pre"] = calibrated_frame[
        "modular_rating__calibrated_win_probability_2023"
    ]
    r5_calibrated = evaluate_rating_rows(calibrated_frame, "development")
    calibrated_frame["modular_rating__raw_win_probability_pre"] = original_probability
    r5_raw = evaluate_rating_rows(final_frame, "development")

    final_frame.to_pickle(output / "rating_predictions_2014_2024.pkl")
    write_json(output / "final_spec.json", {
        "schema_version": 1,
        "spec": final_spec.as_dict(),
        "temperature_2023": calibrator.temperature,
        "feature_columns": config["r5_outputs"],
    })
    result = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "scope": {
            "events_through_2024": len(events),
            "scoring_rows_2014_2024": len(final_frame),
            "scoring_races_2014_2024": int(final_frame["race_id"].nunique()),
            "retrospective_2025_used": False,
            "odds_used": False,
        },
        "r0": r0_result,
        "r1": r1,
        "r2": {
            "candidates": r2_candidates,
            "selected_spec": r2_spec.as_dict(),
            "validation_2022": r2_validation,
        },
        "r3": {
            "candidates": r3_candidates,
            "selected_candidate_spec": r3_spec.as_dict(),
            "baseline_2022": r2_validation,
            "candidate_2022": r3_validation,
            "candidate_improvement": r3_improvement,
            "accepted": r3_accepted,
            "selected_global_spec": selected_global_spec.as_dict(),
        },
        "r4": {
            "candidates": r4_candidates,
            "selected_candidate_spec": r4_spec.as_dict(),
            "baseline_2022": selected_global_validation,
            "candidate_2022": r4_validation,
            "candidate_improvement": r4_improvement,
            "accepted": r4_accepted,
        },
        "r5": {
            "final_spec": final_spec.as_dict(),
            "temperature_2023": calibrator.temperature,
            "development_raw": r5_raw,
            "development_calibrated": r5_calibrated,
        },
        "reproducibility": {
            "config_sha256": sha256_file(root / config_path),
            "model_cache_sha256": sha256_file(root / model_cache_path),
            "data_fingerprint": manifest["raw_file"]["sha256"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "metrics.json", result)
    write_artifact_manifest(output)
    return result
