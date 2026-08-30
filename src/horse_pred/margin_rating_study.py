"""PV-02: preregistered time-margin pairwise Elo comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.data import (
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_raw_file,
)
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import (
    ndcg_at_k,
    race_brier_score,
    race_log_loss,
    top_k_winner_mass,
)
from horse_pred.modeling import apply_temperature, fit_temperature
from horse_pred.rating import (
    RatingEvent,
    RatingSpec,
    attach_scoring_population,
    build_rating_history_from_events,
    prepare_rating_events,
)
from horse_pred.rating_study import _candidate_improvement, _primary_metrics
from horse_pred.uncertainty import paired_block_bootstrap


def derive_adjacent_margin_scale(
    events: tuple[RatingEvent, ...], years: set[int]
) -> dict[str, Any]:
    """Derive train-only tau from positive adjacent official-rank clock gaps."""

    gaps: list[float] = []
    clean_races = 0
    excluded_status_races = 0
    zero_count = 0
    negative_count = 0
    eligible_pair_count = 0
    for event in events:
        if event.race_date.year not in years:
            continue
        if any(status in {"demoted", "disqualified"} for status in event.finish_statuses):
            excluded_status_races += 1
            continue
        clean_races += 1
        if not np.isfinite(event.distance_m) or event.distance_m <= 0:
            continue
        rank_groups: dict[float, list[int]] = {}
        for index, (finish, result_time, status) in enumerate(
            zip(event.finishes, event.result_times_seconds, event.finish_statuses)
        ):
            if status != "finished" or not np.isfinite(finish) or not np.isfinite(result_time):
                continue
            rank_groups.setdefault(float(finish), []).append(index)
        ranks = sorted(rank_groups)
        for upper_rank, lower_rank in zip(ranks, ranks[1:]):
            for upper_index in rank_groups[upper_rank]:
                for lower_index in rank_groups[lower_rank]:
                    eligible_pair_count += 1
                    gap = (
                        event.result_times_seconds[lower_index]
                        - event.result_times_seconds[upper_index]
                    ) * 1000.0 / event.distance_m
                    if gap > 0:
                        gaps.append(float(gap))
                    elif gap == 0:
                        zero_count += 1
                    else:
                        negative_count += 1
    if not gaps:
        raise ValueError("no positive adjacent time margins for tau derivation")
    values = np.asarray(gaps, dtype=np.float64)
    quantiles = {
        str(probability): float(np.quantile(values, probability, method="linear"))
        for probability in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    return {
        "years": sorted(years),
        "estimator": "median_strictly_positive_adjacent_distinct_official_finish_gap",
        "unit": "seconds_per_1000m",
        "clean_race_count": clean_races,
        "excluded_status_race_count": excluded_status_races,
        "eligible_adjacent_pair_count": eligible_pair_count,
        "strictly_positive_pair_count": len(values),
        "zero_pair_count": zero_count,
        "negative_pair_count": negative_count,
        "strictly_positive_quantiles": quantiles,
        "tau": float(np.median(values)),
    }


def _coverage(events: tuple[RatingEvent, ...], year: int) -> dict[str, int]:
    counts = {
        "race_count": 0,
        "total_pair_count": 0,
        "continuous_positive_pair_count": 0,
        "continuous_zero_pair_count": 0,
        "same_finish_pair_count": 0,
        "pair_fallback_count": 0,
        "whole_race_ordinal_pair_count": 0,
        "clock_order_inversion_pair_count": 0,
    }
    for event in events:
        if event.race_date.year != year:
            continue
        counts["race_count"] += 1
        force_ordinal = any(
            status in {"demoted", "disqualified", "unknown"}
            for status in event.finish_statuses
        )
        for i in range(len(event.horse_ids)):
            for j in range(i + 1, len(event.horse_ids)):
                counts["total_pair_count"] += 1
                if force_ordinal:
                    counts["whole_race_ordinal_pair_count"] += 1
                    continue
                finish_i, finish_j = event.finishes[i], event.finishes[j]
                time_i, time_j = event.result_times_seconds[i], event.result_times_seconds[j]
                if np.isfinite(finish_i) and np.isfinite(finish_j) and finish_i == finish_j:
                    counts["same_finish_pair_count"] += 1
                    continue
                if not (
                    np.isfinite(event.distance_m)
                    and event.distance_m > 0
                    and np.isfinite(finish_i)
                    and np.isfinite(finish_j)
                    and np.isfinite(time_i)
                    and np.isfinite(time_j)
                ):
                    counts["pair_fallback_count"] += 1
                    continue
                margin = (time_j - time_i) * 1000.0 / event.distance_m
                direction = 1.0 if finish_i < finish_j else -1.0
                if margin * direction < 0:
                    counts["clock_order_inversion_pair_count"] += 1
                elif margin == 0:
                    counts["continuous_zero_pair_count"] += 1
                else:
                    counts["continuous_positive_pair_count"] += 1
    return counts


def _race_metric_table(
    arms: dict[str, pd.DataFrame], year: int, probability_column: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in arms.items():
        selected = frame.loc[frame["race_date"].dt.year.eq(year)].sort_values(
            ["race_date", "race_id", "source_position"], kind="stable"
        )
        for race_id, race in selected.groupby("race_id", sort=False):
            probabilities = pd.to_numeric(race[probability_column], errors="raise")
            scores = pd.to_numeric(race["modular_rating__score_pre"], errors="raise")
            positions = pd.to_numeric(race["model_finish_position"], errors="raise").astype(int)
            ids = race["race_id"].tolist()
            if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-8, rtol=0.0):
                raise ValueError(f"incoherent probabilities for {model} race {race_id}")
            rows.append(
                {
                    "race_id": str(race_id),
                    "race_date": pd.Timestamp(race["race_date"].iloc[0]).normalize(),
                    "model": model,
                    "ndcg_at_3": ndcg_at_k(scores, positions, ids, k=3),
                    "top_1_winner_mass": top_k_winner_mass(scores, positions, ids, k=1),
                    "race_log_loss": race_log_loss(probabilities, positions, ids),
                    "race_brier": race_brier_score(probabilities, positions, ids),
                }
            )
    result = pd.DataFrame(rows)
    support = result.groupby("model")["race_id"].agg(frozenset)
    if support.nunique() != 1:
        raise ValueError("PV-02 arms do not share the same race population")
    return result.sort_values(["race_date", "race_id", "model"], kind="stable").reset_index(
        drop=True
    )


def _decision(
    improvement: dict[str, float], bootstrap: dict[str, Any], config: dict[str, Any]
) -> str:
    paired = bootstrap["paired"]["candidate_vs_control"]
    lower = float(paired["race_log_loss"]["lower"])
    upper = float(paired["race_log_loss"]["upper"])
    guardrails = config["selection"]["guardrails"]
    guardrails_pass = bool(
        improvement["race_brier"] >= guardrails["race_brier_improvement_min"]
        and improvement["ndcg_at_3"] >= guardrails["ndcg_at_3_improvement_min"]
        and improvement["top_1"] >= guardrails["top_1_improvement_min"]
    )
    if guardrails_pass and lower > config["selection"]["race_log_loss_improvement_ci_lower_exclusive"]:
        return "go"
    if not guardrails_pass or upper < 0.0:
        return "reject"
    return "inconclusive"


def _load_normalized_through(
    raw_path: str | Path, expected_sha256: str, through_year: int
) -> pd.DataFrame:
    raw = load_raw(raw_path, expected_sha256=expected_sha256)
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(through_year)].copy()
    return normalize_raw(raw)


def _score(
    events: tuple[RatingEvent, ...], spec: RatingSpec, model_frame: pd.DataFrame
) -> pd.DataFrame:
    return attach_scoring_population(
        build_rating_history_from_events(events, spec), model_frame
    )


def run_margin_rating_study(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    model_cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run PV-02 with a hard 2022 gate before any 2023/2024 outcome access."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PV-02 output: {output}")
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
    frozen = (control_spec.initial_rating, control_spec.k, control_spec.scale)
    if frozen != (1500.0, 48.0, 200.0) or (
        candidate_spec.initial_rating,
        candidate_spec.k,
        candidate_spec.scale,
    ) != frozen:
        raise ValueError("PV-02 must preserve the frozen R5 initial/K/scale")

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
        raise ValueError(f"derived tau disagrees with preregistration: {tau['tau']}")

    stage_frame = model_frame.loc[model_frame["race_date"].dt.year.le(2022)].copy()
    arms_2022 = {
        "control": _score(events_2022, control_spec, stage_frame),
        "candidate": _score(events_2022, candidate_spec, stage_frame),
    }
    metrics_2022 = {
        arm: _primary_metrics(frame, [2022]) for arm, frame in arms_2022.items()
    }
    improvement_2022 = _candidate_improvement(
        metrics_2022["control"], metrics_2022["candidate"]
    )
    race_metrics_2022 = _race_metric_table(
        arms_2022, 2022, "modular_rating__raw_win_probability_pre"
    )
    uncertainty = config["uncertainty"]
    bootstrap_2022 = paired_block_bootstrap(
        race_metrics_2022,
        comparisons=(("candidate", "control"),),
        n_resamples=int(uncertainty["bootstrap_resamples"]),
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        block_length_dates=int(uncertainty["block_length_dates"]),
    )
    decision_2022 = _decision(improvement_2022, bootstrap_2022, config)
    race_metrics_2022.to_csv(output / "race_metrics_2022.csv.gz", index=False, compression="gzip")
    for arm, frame in arms_2022.items():
        frame.loc[frame["race_date"].dt.year.eq(2022)].to_pickle(
            output / f"predictions_2022_{arm}.pkl"
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "scope": {
            "retrospective_2025_used": False,
            "odds_used": False,
            "development_2024_opened": False,
        },
        "specs": {
            "control": {**control_spec.as_dict(), "pairwise_actual": "ordinal"},
            "candidate": candidate_spec.as_dict(),
        },
        "tau_derivation": tau,
        "coverage_2022": _coverage(events_2022, 2022),
        "validation_2022": {
            "metrics": metrics_2022,
            "candidate_improvement": improvement_2022,
            "paired_bootstrap": bootstrap_2022,
            "decision": decision_2022,
        },
        "reproducibility": {
            "git": git_state(root),
            "config_sha256": sha256_file(config_file),
            "model_cache_sha256": sha256_file(cache_file),
            "data_fingerprint": manifest["raw_file"]["sha256"],
            "seed": config["seed"],
        },
    }

    if decision_2022 == "go":
        normalized_2024 = _load_normalized_through(
            raw_path, manifest["raw_file"]["sha256"], 2024
        )
        events_2024 = prepare_rating_events(normalized_2024, through_year=2024)
        del normalized_2024
        full_frame = model_frame.loc[model_frame["race_date"].dt.year.le(2024)].copy()
        arms_2024 = {
            "control": _score(events_2024, control_spec, full_frame),
            "candidate": _score(events_2024, candidate_spec, full_frame),
        }
        temperatures: dict[str, float] = {}
        calibrated_arms: dict[str, pd.DataFrame] = {}
        for arm, frame in arms_2024.items():
            calibration = frame.loc[frame["race_date"].dt.year.eq(2023)].sort_values(
                ["race_date", "race_id", "source_position"], kind="stable"
            )
            calibrator = fit_temperature(
                calibration["modular_rating__score_pre"],
                calibration["race_id"],
                calibration["model_finish_position"],
            )
            temperatures[arm] = float(calibrator.temperature)
            calibrated = frame.copy()
            calibrated["calibrated_probability"] = np.nan
            development_mask = calibrated["race_date"].dt.year.eq(2024)
            calibrated.loc[development_mask, "calibrated_probability"] = apply_temperature(
                calibrator,
                calibrated.loc[development_mask, "modular_rating__score_pre"],
                calibrated.loc[development_mask, "race_id"],
            )
            calibrated_arms[arm] = calibrated
        raw_2024 = {
            arm: _primary_metrics(frame, [2024]) for arm, frame in arms_2024.items()
        }
        calibrated_2024 = {}
        for arm, frame in calibrated_arms.items():
            metric_frame = frame.copy()
            mask = metric_frame["race_date"].dt.year.eq(2024)
            metric_frame.loc[mask, "modular_rating__raw_win_probability_pre"] = (
                metric_frame.loc[mask, "calibrated_probability"]
            )
            calibrated_2024[arm] = _primary_metrics(metric_frame, [2024])
        improvement_2024 = _candidate_improvement(
            calibrated_2024["control"], calibrated_2024["candidate"]
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
        decision_2024 = _decision(improvement_2024, bootstrap_2024, config)
        race_metrics_2024.to_csv(
            output / "race_metrics_2024.csv.gz", index=False, compression="gzip"
        )
        for arm, frame in calibrated_arms.items():
            frame.loc[frame["race_date"].dt.year.eq(2024)].to_pickle(
                output / f"predictions_2024_{arm}.pkl"
            )
        result["scope"]["development_2024_opened"] = True
        result["coverage_2024"] = _coverage(events_2024, 2024)
        result["development_2024"] = {
            "temperature_2023": temperatures,
            "raw_metrics": raw_2024,
            "calibrated_metrics": calibrated_2024,
            "candidate_improvement": improvement_2024,
            "paired_bootstrap": bootstrap_2024,
            "decision": decision_2024,
        }

    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "metrics.json", result)
    write_json(output / "config.json", config)
    write_artifact_manifest(output)
    return result
