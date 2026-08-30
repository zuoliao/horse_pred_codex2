"""PV-03: out-of-period temperature calibration for margin-aware Elo."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.data import load_manifest, sha256_file, verify_raw_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.margin_rating_study import (
    _coverage,
    _decision,
    _load_normalized_through,
    _race_metric_table,
    _score,
    derive_adjacent_margin_scale,
)
from horse_pred.modeling import apply_temperature, fit_temperature
from horse_pred.rating import RatingSpec, prepare_rating_events
from horse_pred.rating_study import _candidate_improvement, _primary_metrics
from horse_pred.uncertainty import paired_block_bootstrap


def _calibrate_for_year(
    frame: pd.DataFrame, calibration_year: int, evaluation_year: int
) -> tuple[pd.DataFrame, float, dict[str, float | int]]:
    calibration = frame.loc[frame["race_date"].dt.year.eq(calibration_year)].sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    evaluation = frame.loc[frame["race_date"].dt.year.eq(evaluation_year)].copy()
    evaluation = evaluation.sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    if calibration.empty or evaluation.empty:
        raise ValueError("calibration and evaluation years must both contain rows")
    calibrator = fit_temperature(
        calibration["modular_rating__score_pre"],
        calibration["race_id"],
        calibration["model_finish_position"],
    )
    evaluation["calibrated_probability"] = apply_temperature(
        calibrator,
        evaluation["modular_rating__score_pre"],
        evaluation["race_id"],
    )
    metric_frame = evaluation.copy()
    metric_frame["modular_rating__raw_win_probability_pre"] = metric_frame[
        "calibrated_probability"
    ]
    return metric_frame, float(calibrator.temperature), _primary_metrics(
        metric_frame, [evaluation_year]
    )


def _rolling_decision(
    annual_improvements: dict[str, dict[str, float]],
    latest_bootstrap: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    years = sorted(annual_improvements)
    macro = {
        metric: float(np.mean([annual_improvements[year][metric] for year in years]))
        for metric in ("race_log_loss", "race_brier", "ndcg_at_3", "top_1")
    }
    positive_years = sum(
        annual_improvements[year]["race_log_loss"] > 0.0 for year in years
    )
    latest_interval = latest_bootstrap["paired"]["candidate_vs_control"][
        "race_log_loss"
    ]
    passed = bool(
        positive_years >= int(selection["minimum_positive_evaluation_years"])
        and macro["race_log_loss"]
        > float(selection["annual_macro_log_loss_improvement_exclusive"])
        and macro["race_brier"]
        >= float(selection["annual_macro_brier_improvement_min"])
        and float(latest_interval["lower"])
        > float(selection["latest_year_log_loss_improvement_ci_lower_exclusive"])
    )
    diagnostics = {
        "positive_log_loss_year_count": positive_years,
        "evaluation_year_count": len(years),
        "annual_macro_improvement": macro,
        "latest_log_loss_improvement_interval": latest_interval,
    }
    if passed:
        return "go", diagnostics
    if (
        positive_years < int(selection["minimum_positive_evaluation_years"])
        or macro["race_log_loss"] <= 0.0
        or macro["race_brier"] < float(selection["annual_macro_brier_improvement_min"])
        or float(latest_interval["upper"]) < 0.0
    ):
        return "reject", diagnostics
    return "inconclusive", diagnostics


def run_margin_rating_calibration_study(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    model_cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run PV-03 rolling calibration and conditionally open 2024."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PV-03 output: {output}")
    output.mkdir(parents=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    cache_file = Path(model_cache_path)
    if not cache_file.is_absolute():
        cache_file = root / cache_file
    config = json.loads(config_file.read_text(encoding="utf-8"))
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    model_frame, _ = read_model_frame_cache(cache_file)
    model_frame["race_date"] = pd.to_datetime(model_frame["race_date"], errors="raise")
    control_spec = RatingSpec(**config["control"])
    candidate_spec = RatingSpec(**config["candidate"])

    normalized_2022 = _load_normalized_through(
        raw_path, manifest["raw_file"]["sha256"], 2022
    )
    events_2022 = prepare_rating_events(normalized_2022, through_year=2022)
    del normalized_2022
    tau = derive_adjacent_margin_scale(
        events_2022, set(config["data"]["margin_scale_years"])
    )
    if not np.isclose(
        tau["tau"], candidate_spec.time_margin_tau_seconds_per_1000m, atol=1e-12, rtol=0.0
    ):
        raise ValueError(f"derived early-period tau disagrees with frozen value: {tau['tau']}")
    stage_frame = model_frame.loc[model_frame["race_date"].dt.year.le(2022)].copy()
    arms = {
        "control": _score(events_2022, control_spec, stage_frame),
        "candidate": _score(events_2022, candidate_spec, stage_frame),
    }

    annual: dict[str, Any] = {}
    annual_improvements: dict[str, dict[str, float]] = {}
    annual_evaluation_frames: dict[str, list[pd.DataFrame]] = {
        "control": [],
        "candidate": [],
    }
    latest_year = max(pair[1] for pair in config["data"]["rolling_calibration_evaluation_pairs"])
    latest_race_metrics: pd.DataFrame | None = None
    for calibration_year, evaluation_year in config["data"][
        "rolling_calibration_evaluation_pairs"
    ]:
        calibrated_frames: dict[str, pd.DataFrame] = {}
        temperatures: dict[str, float] = {}
        metrics: dict[str, dict[str, float | int]] = {}
        for arm, frame in arms.items():
            calibrated, temperature, arm_metrics = _calibrate_for_year(
                frame, int(calibration_year), int(evaluation_year)
            )
            calibrated_frames[arm] = calibrated
            temperatures[arm] = temperature
            metrics[arm] = arm_metrics
            annual_evaluation_frames[arm].append(calibrated)
        improvement = _candidate_improvement(metrics["control"], metrics["candidate"])
        annual_improvements[str(evaluation_year)] = improvement
        annual[str(evaluation_year)] = {
            "calibration_year": int(calibration_year),
            "temperatures": temperatures,
            "metrics": metrics,
            "candidate_improvement": improvement,
        }
        if int(evaluation_year) == latest_year:
            latest_race_metrics = _race_metric_table(
                calibrated_frames, int(evaluation_year), "calibrated_probability"
            )
    if latest_race_metrics is None:
        raise AssertionError("latest rolling evaluation was not constructed")
    uncertainty = config["uncertainty"]
    latest_bootstrap = paired_block_bootstrap(
        latest_race_metrics,
        comparisons=(("candidate", "control"),),
        n_resamples=int(uncertainty["bootstrap_resamples"]),
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        block_length_dates=int(uncertainty["block_length_dates"]),
    )
    decision, rolling_diagnostics = _rolling_decision(
        annual_improvements, latest_bootstrap, config["selection"]
    )
    latest_race_metrics.to_csv(
        output / f"race_metrics_{latest_year}.csv.gz", index=False, compression="gzip"
    )
    for arm, frames in annual_evaluation_frames.items():
        pd.concat(frames, ignore_index=True).to_pickle(
            output / f"rolling_predictions_{arm}.pkl"
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "scope": {
            "development_2024_opened": False,
            "retrospective_2025_used": False,
            "odds_used": False,
        },
        "tau_derivation": tau,
        "annual": annual,
        "rolling_gate": {
            **rolling_diagnostics,
            "latest_paired_bootstrap": latest_bootstrap,
            "decision": decision,
        },
        "reproducibility": {
            "git": git_state(root),
            "config_sha256": sha256_file(config_file),
            "model_cache_sha256": sha256_file(cache_file),
            "data_fingerprint": manifest["raw_file"]["sha256"],
            "seed": config["seed"],
        },
    }

    if decision == "go":
        normalized_2024 = _load_normalized_through(
            raw_path, manifest["raw_file"]["sha256"], 2024
        )
        events_2024 = prepare_rating_events(normalized_2024, through_year=2024)
        del normalized_2024
        full_frame = model_frame.loc[model_frame["race_date"].dt.year.le(2024)].copy()
        full_arms = {
            "control": _score(events_2024, control_spec, full_frame),
            "candidate": _score(events_2024, candidate_spec, full_frame),
        }
        calibrated_arms: dict[str, pd.DataFrame] = {}
        temperatures: dict[str, float] = {}
        calibrated_metrics: dict[str, dict[str, float | int]] = {}
        raw_metrics = {
            arm: _primary_metrics(frame, [2024]) for arm, frame in full_arms.items()
        }
        for arm, frame in full_arms.items():
            calibrated, temperature, metrics = _calibrate_for_year(frame, 2023, 2024)
            calibrated_arms[arm] = calibrated
            temperatures[arm] = temperature
            calibrated_metrics[arm] = metrics
        improvement = _candidate_improvement(
            calibrated_metrics["control"], calibrated_metrics["candidate"]
        )
        race_metrics_2024 = _race_metric_table(
            calibrated_arms, 2024, "calibrated_probability"
        )
        bootstrap_2024 = paired_block_bootstrap(
            race_metrics_2024,
            comparisons=(("candidate", "control"),),
            n_resamples=int(uncertainty["bootstrap_resamples"]),
            confidence_level=float(uncertainty["confidence_level"]),
            seed=int(uncertainty["bootstrap_seed"]),
            block_length_dates=int(uncertainty["block_length_dates"]),
        )
        development_config = {"selection": config["development_selection"]}
        development_decision = _decision(
            improvement, bootstrap_2024, development_config
        )
        race_metrics_2024.to_csv(
            output / "race_metrics_2024.csv.gz", index=False, compression="gzip"
        )
        for arm, frame in calibrated_arms.items():
            frame.to_pickle(output / f"predictions_2024_{arm}.pkl")
        result["scope"]["development_2024_opened"] = True
        result["coverage_2024"] = _coverage(events_2024, 2024)
        result["development_2024"] = {
            "temperature_2023": temperatures,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "candidate_improvement": improvement,
            "paired_bootstrap": bootstrap_2024,
            "decision": development_decision,
        }

    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "metrics.json", result)
    write_json(output / "config.json", config)
    write_artifact_manifest(output)
    return result
