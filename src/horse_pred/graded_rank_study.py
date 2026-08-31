"""GR-001: isolated 2022 gate for a coarser field-aware LambdaRank label."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.cached_experiment import (
    feature_columns_checksum,
    resolve_semantic_feature_selection,
)
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import (
    evaluate_predictions,
    ndcg_at_k,
    race_brier_score,
    race_log_loss,
    top_k_winner_mass,
)
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    grouped_ranking_relevance_targets,
    predict,
    race_softmax,
    train_ranker,
)
from horse_pred.rating_study import _candidate_improvement
from horse_pred.uncertainty import paired_block_bootstrap


def _year_frame(frame: pd.DataFrame, maximum_year: int) -> pd.DataFrame:
    required = {
        "race_id",
        "horse_id",
        "race_date",
        "field_size",
        "model_finish_position",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"GR-001 cache is missing metadata: {missing}")
    dates = pd.to_datetime(frame["race_date"], errors="raise")
    selected = frame.loc[dates.dt.year.le(maximum_year)].copy()
    selected["race_date"] = dates.loc[selected.index]
    selected["_source_order"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.sort_values(
        ["race_date", "race_id", "_source_order"], kind="stable"
    ).drop(columns="_source_order")
    if selected["race_date"].dt.year.gt(maximum_year).any():
        raise AssertionError("post-gate outcomes survived GR-001 isolation")
    grouped = selected.groupby("race_id", sort=False, observed=True)
    sizes = grouped.size()
    declared = grouped["field_size"].first()
    if not sizes.eq(pd.to_numeric(declared, errors="raise").astype(int)).all():
        raise ValueError("GR-001 requires cache field_size to equal query row count")
    if grouped["field_size"].nunique().gt(1).any():
        raise ValueError("GR-001 field_size must be constant within each race")
    return selected.reset_index(drop=True)


def _study_split(frame: pd.DataFrame, data_config: dict[str, Any]) -> pd.Series:
    years = frame["race_date"].dt.year
    fit_years = {int(year) for year in data_config["model_fit_years"]}
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[years.isin(fit_years)] = "train"
    result.loc[years.eq(int(data_config["early_stopping_year"]))] = "model_validation"
    result.loc[years.eq(int(data_config["temperature_calibration_year"]))] = "calibration"
    result.loc[years.eq(int(data_config["validation_year"]))] = "evaluation"
    if result.isna().any():
        unexpected = sorted(years.loc[result.isna()].unique().tolist())
        raise ValueError(f"GR-001 encountered unregistered years: {unexpected}")
    return result


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return [
        int(value)
        for value in frame.groupby("race_id", sort=False, observed=True).size().tolist()
    ]


def _label_audit(frame: pd.DataFrame, schemes: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "model_validation"):
        selected = frame.loc[frame["study_split"].eq(split)]
        positions = pd.to_numeric(
            selected["model_finish_position"], errors="raise"
        ).astype(int).tolist()
        sizes = _group_sizes(selected)
        result[split] = {}
        for arm, scheme in schemes.items():
            labels = grouped_ranking_relevance_targets(
                positions, sizes, relevance_scheme=scheme
            )
            counts = Counter(labels)
            result[split][arm] = {
                "race_count": len(sizes),
                "runner_count": len(labels),
                "label_counts": {str(label): int(counts[label]) for label in range(4)},
                "label_fractions": {
                    str(label): counts[label] / len(labels) for label in range(4)
                },
            }
    candidate_rows = frame.loc[
        frame["study_split"].isin(("train", "model_validation"))
    ]
    positions = pd.to_numeric(
        candidate_rows["model_finish_position"], errors="raise"
    ).astype(int)
    cutoffs = (pd.to_numeric(candidate_rows["field_size"], errors="raise") + 1) // 2
    relevance_one = positions.ge(4) & positions.le(cutoffs)
    result["candidate_relevance_one_by_field_size"] = {
        str(int(field_size)): int(count)
        for field_size, count in candidate_rows.loc[relevance_one]
        .groupby("field_size", observed=True)
        .size()
        .items()
    }
    return result


def _race_metric_table(scoring: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ("control", "candidate"):
        score_column = f"score_{arm}"
        probability_column = f"prob_{arm}"
        for race_id, race in scoring.groupby("race_id", sort=False, observed=True):
            ids = race["race_id"].tolist()
            positions = pd.to_numeric(
                race["model_finish_position"], errors="raise"
            ).astype(int).tolist()
            scores = race[score_column].tolist()
            probabilities = race[probability_column].tolist()
            rows.append(
                {
                    "race_id": str(race_id),
                    "race_date": pd.Timestamp(race["race_date"].iloc[0]).normalize(),
                    "model": arm,
                    "ndcg_at_3": ndcg_at_k(scores, positions, ids, k=3),
                    "top_1_winner_mass": top_k_winner_mass(
                        scores, positions, ids, k=1
                    ),
                    "race_log_loss": race_log_loss(probabilities, positions, ids),
                    "race_brier": race_brier_score(probabilities, positions, ids),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["race_date", "race_id", "model"], kind="stable"
    ).reset_index(drop=True)


def _primary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "race_log_loss": float(payload["probability"]["race_log_loss"]),
        "race_brier": float(payload["probability"]["race_brier"]),
        "ndcg_at_3": float(payload["ranking"]["ndcg_at_3"]),
        "top_1": float(payload["ranking"]["top_1"]),
    }


def graded_rank_decision(
    improvement: dict[str, float], bootstrap: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    paired = bootstrap["paired"]["candidate_vs_control"]
    acceptance = config["acceptance"]
    probability = acceptance["probability_path"]
    ranking = acceptance["ranking_path"]
    probability_passed = bool(
        improvement["race_log_loss"] >= probability["log_loss_improvement_min"]
        and paired["race_log_loss"]["lower"] > 0.0
        and improvement["race_brier"] >= probability["brier_improvement_min"]
        and improvement["ndcg_at_3"] >= probability["ndcg_at_3_improvement_min"]
        and improvement["top_1"] >= probability["top_1_improvement_min"]
    )
    ranking_passed = bool(
        paired["ndcg_at_3"]["lower"] > 0.0
        and improvement["race_log_loss"] >= ranking["log_loss_improvement_min"]
        and improvement["race_brier"] >= ranking["brier_improvement_min"]
        and improvement["top_1"] >= ranking["top_1_improvement_min"]
    )
    guardrail_failed = bool(
        improvement["race_log_loss"] < ranking["log_loss_improvement_min"]
        or improvement["race_brier"] < ranking["brier_improvement_min"]
        or improvement["ndcg_at_3"] < probability["ndcg_at_3_improvement_min"]
        or improvement["top_1"] < ranking["top_1_improvement_min"]
    )
    if probability_passed or ranking_passed:
        decision = "accept"
    elif (
        guardrail_failed
        or paired["ndcg_at_3"]["upper"] < 0.0
        or paired["race_log_loss"]["upper"] < 0.0
    ):
        decision = "reject"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "probability_path_passed": probability_passed,
        "ranking_path_passed": ranking_passed,
        "guardrail_failed": guardrail_failed,
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None and int(value) > 0 else None


def _write_model_outputs(
    output: Path,
    models: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> None:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for arm, model in models.items():
        model.booster_.save_model(str(model_dir / f"{arm}.txt"))
        table = pd.DataFrame(
            {
                "feature": feature_columns,
                "gain": model.booster_.feature_importance(importance_type="gain"),
                "split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values(["gain", "feature"], ascending=[False, True], kind="stable")
        table.to_csv(output / f"feature_importance_{arm}.csv", index=False)


def run_graded_rank_study(
    *,
    repo_root: str | Path,
    cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the preregistered GR-001 comparison and stop after 2022."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GR-001 output: {output}")
    output.mkdir(parents=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    cache_file = Path(cache_path)
    if not cache_file.is_absolute():
        cache_file = root / cache_file
    config = load_json(config_file)
    base_config = load_json(root / config["base_feature_config"])
    ranker_config = load_json(root / config["ranker_config"])
    cached, cache_meta = read_model_frame_cache(cache_file)
    frame = _year_frame(cached, int(config["data"]["maximum_outcome_year"]))
    del cached
    frame["study_split"] = _study_split(frame, config["data"])

    all_features = tuple(cache_meta["feature_columns"])
    feature_columns, feature_groups, feature_resolution = (
        resolve_semantic_feature_selection(all_features, base_config)
    )
    feature_hash = feature_columns_checksum(feature_columns)
    if len(feature_columns) != int(config["features"]["expected_count"]):
        raise ValueError("GR-001 feature count disagrees with preregistration")
    if feature_hash != config["features"]["expected_columns_sha256"]:
        raise ValueError("GR-001 feature hash disagrees with preregistration")

    schemes = {
        "control": str(config["control_relevance_scheme"]),
        "candidate": str(config["candidate_relevance_scheme"]),
    }
    label_audit = _label_audit(frame, schemes)
    common = {
        "frame": frame,
        "feature_columns": feature_columns,
        "train_split": "train",
        "model_validation_split": "model_validation",
        "race_id_column": "race_id",
        "finish_position_column": "model_finish_position",
        "split_column": "study_split",
        "params": ranker_config["parameters"],
        "early_stopping_rounds": ranker_config.get("early_stopping_rounds"),
    }
    models = {
        arm: train_ranker(relevance_scheme=scheme, **common)
        for arm, scheme in schemes.items()
    }
    scoring = frame.loc[
        frame["study_split"].isin(("calibration", "evaluation"))
    ].copy()
    for arm, model in models.items():
        scoring[f"score_{arm}"] = predict(
            model, scoring, feature_columns=feature_columns, model_kind="lambdarank"
        )
    calibration = scoring.loc[scoring["study_split"].eq("calibration")]
    evaluation = scoring.loc[scoring["study_split"].eq("evaluation")].copy()
    temperatures: dict[str, float] = {}
    for arm in schemes:
        calibrator = fit_temperature(
            calibration[f"score_{arm}"],
            calibration["race_id"],
            calibration["model_finish_position"],
        )
        temperatures[arm] = float(calibrator.temperature)
        evaluation[f"prob_{arm}"] = apply_temperature(
            calibrator, evaluation[f"score_{arm}"], evaluation["race_id"]
        )
        evaluation[f"prob_t1_{arm}"] = race_softmax(
            evaluation[f"score_{arm}"], evaluation["race_id"]
        )

    positions = pd.to_numeric(
        evaluation["model_finish_position"], errors="raise"
    ).astype(int).tolist()
    race_ids = evaluation["race_id"].tolist()
    calibrated_payloads = {
        arm: evaluate_predictions(
            evaluation[f"prob_{arm}"],
            positions,
            race_ids,
            ranking_scores=evaluation[f"score_{arm}"],
        )
        for arm in schemes
    }
    t1_payloads = {
        arm: evaluate_predictions(
            evaluation[f"prob_t1_{arm}"],
            positions,
            race_ids,
            ranking_scores=evaluation[f"score_{arm}"],
        )
        for arm in schemes
    }
    primary = {arm: _primary_metrics(payload) for arm, payload in calibrated_payloads.items()}
    improvement = _candidate_improvement(primary["control"], primary["candidate"])
    race_metrics = _race_metric_table(evaluation)
    uncertainty = config["uncertainty"]
    bootstrap = paired_block_bootstrap(
        race_metrics,
        comparisons=(("candidate", "control"),),
        n_resamples=int(uncertainty["bootstrap_resamples"]),
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        block_length_dates=int(uncertainty["block_length_dates"]),
    )
    decision = graded_rank_decision(improvement, bootstrap, config)
    split_rows = {
        str(name): int(count)
        for name, count in frame["study_split"].value_counts().items()
    }
    split_races = {
        str(name): int(count)
        for name, count in frame.groupby("study_split", observed=True)["race_id"]
        .nunique()
        .items()
    }
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "hypothesis": config["hypothesis"],
        "scope": {
            "model_fit": "2014-2019",
            "early_stopping": "2020",
            "temperature_calibration": "2021",
            "evaluation": "2022 only",
            "maximum_outcome_year": 2022,
            "outcome_rows_used_2023": 0,
            "outcome_rows_used_2024": 0,
            "outcome_rows_used_2025": 0,
            "odds_used": False,
        },
        "data": {
            "fingerprint": cache_meta["data_fingerprint"],
            "cache_sha256": sha256_file(cache_file),
            "split_rows": split_rows,
            "split_races": split_races,
        },
        "features": {
            "count": len(feature_columns),
            "columns_sha256": feature_hash,
            "groups": {name: list(columns) for name, columns in feature_groups.items()},
            "resolution": feature_resolution,
        },
        "label_audit": label_audit,
        "models": {
            arm: {
                "relevance_scheme": schemes[arm],
                "best_iteration": _best_iteration(models[arm]),
                "temperature_2021": temperatures[arm],
            }
            for arm in schemes
        },
        "validation_2022": {
            "calibrated": calibrated_payloads,
            "t1_descriptive": t1_payloads,
            "primary_metrics": primary,
            "candidate_improvement": improvement,
            "paired_bootstrap": bootstrap,
            **decision,
        },
        "reproducibility": {
            "git": git_state(root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_file),
            "config_hash": canonical_json_hash(config),
            "ranker_config_hash": canonical_json_hash(ranker_config),
            "seed": config["seed"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "config.json", config)
    write_json(output / "ranker_config.json", ranker_config)
    write_json(
        output / "feature_schema.json",
        {
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "feature_columns_sha256": feature_hash,
        },
    )
    predictions = evaluation.loc[
        :,
        [
            "race_id",
            "horse_id",
            "race_date",
            "model_finish_position",
            "field_size",
            "score_control",
            "prob_control",
            "score_candidate",
            "prob_candidate",
        ],
    ]
    predictions.to_csv(output / "predictions_2022.csv.gz", index=False, compression="gzip")
    race_metrics.to_csv(output / "race_metrics_2022.csv.gz", index=False, compression="gzip")
    _write_model_outputs(output, models, feature_columns)
    write_artifact_manifest(output)
    return result
