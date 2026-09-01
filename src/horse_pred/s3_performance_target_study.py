"""Preregistered S3 rolling study for continuous performance supervision."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.cached_experiment import feature_columns_checksum
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.evaluation import evaluate_predictions, race_balanced_reliability
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    predict,
    probability_logits,
    train_binary,
    train_huber_regressor,
    train_ranker,
    validate_prediction_feature_columns,
)
from horse_pred.performance_target import (
    PERFORMANCE_TARGET_COLUMN,
    ConditionAdjustedPerformanceTargetSpec,
    build_fold_performance_targets,
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.rolling_evaluation import assign_fold_roles
from horse_pred.s1_two_axis_study import (
    _build_base_frame,
    _metric_payload,
    _paired_bootstrap,
    _race_metric_table,
    _resolve_controls,
    _software_versions,
    _year_summary,
    assign_s1_slice_flags,
    classify_s1_comparison,
    isolate_s1_source,
    verify_artifact_manifest,
)

_METHODS = (
    "binary_control",
    "lambdarank_control",
    "huber_binary_scope",
    "huber_lambdarank_scope",
)
_METRICS = (
    "ndcg_at_3",
    "top_1_winner_mass",
    "winner_reciprocal_rank",
    "race_log_loss",
    "race_brier",
)
_FORBIDDEN_FEATURE_FRAGMENTS = (
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "race_date",
    "finish_position",
    "winner_label",
    "target__",
    "odds",
    "popularity",
    "payout",
    "オッズ",
    "人気",
    "払戻",
)


def validate_s3_preregistration(config: dict[str, Any]) -> None:
    """Fail closed unless the metric-blind S3 protocol is unchanged."""

    required = {
        "schema_version",
        "experiment_id",
        "raw_sha256",
        "maximum_outcome_year",
        "forbidden_years",
        "market_used",
        "folds",
        "controls",
        "target",
        "methods",
        "regression_parameters",
        "probability_mapping",
        "comparisons",
        "metrics",
        "uncertainty",
        "slices",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"S3 preregistration is missing: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("S3 preregistration schema_version must be 1")
    if int(config["maximum_outcome_year"]) != 2022:
        raise ValueError("S3 maximum_outcome_year must remain 2022")
    if set(map(int, config["forbidden_years"])) != {2023, 2024, 2025}:
        raise ValueError("S3 forbidden years must be exactly 2023, 2024, 2025")
    if config["market_used"] is not False:
        raise ValueError("S3 must not enable market information")
    if config.get("s1_performance_feature_added") is not False:
        raise ValueError("S3 must remain a target-only experiment")
    folds = config["folds"]
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("S3 requires exactly three rolling folds")
    for fold, evaluation_year in zip(folds, (2020, 2021, 2022)):
        expected = {
            "train_start_year": 2014,
            "train_end_year": evaluation_year - 3,
            "early_stopping_year": evaluation_year - 2,
            "calibration_year": evaluation_year - 1,
            "evaluation_year": evaluation_year,
        }
        if any(int(fold[key]) != value for key, value in expected.items()):
            raise ValueError("S3 rolling fold rule changed")
    target = config["target"]
    if target.get("column") != PERFORMANCE_TARGET_COLUMN:
        raise ValueError("S3 target column changed")
    if target.get("normalizer") != "pooled_fold_train_2014_to_train_end_then_frozen":
        raise ValueError("S3 target normalizer scope changed")
    if float(target.get("ridge_alpha")) != 1.0 or list(target.get("clip", [])) != [-5.0, 5.0]:
        raise ValueError("S3 target ridge/clip changed")
    params = config["regression_parameters"]
    if params.get("objective") != "huber" or float(params.get("alpha")) != 0.9:
        raise ValueError("S3 Huber objective or alpha changed")
    if tuple(config["methods"]) != _METHODS:
        raise ValueError("S3 methods changed")
    expected_comparisons = {
        "huber_binary_scope_vs_binary_control",
        "huber_lambdarank_scope_vs_lambdarank_control",
    }
    if {item["id"] for item in config["comparisons"]} != expected_comparisons:
        raise ValueError("S3 paired comparisons changed")
    if tuple(config["metrics"]) != _METRICS:
        raise ValueError("S3 metric list changed")
    if config["probability_mapping"].get("fit_role") != "calibration":
        raise ValueError("S3 probability mapping must fit calibration only")


def isolate_s3_source(
    raw_path: str | Path,
    *,
    maximum_outcome_year: int,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Physically isolate the S3 source before normalization."""

    return isolate_s1_source(
        raw_path,
        maximum_outcome_year=maximum_outcome_year,
        expected_sha256=expected_sha256,
    )


def _feature_scope(
    control: tuple[str, ...], additions: list[str]
) -> tuple[str, ...]:
    """Freeze S3 to target-only changes and reject outcome/ID/market inputs."""

    if additions:
        if len(tuple((*control, *additions))) != len(set((*control, *additions))):
            raise ValueError("S3 feature scope contains duplicate columns")
        raise ValueError("S3 does not permit feature additions")
    columns = tuple(control)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("S3 feature scope must be non-empty and unique")
    validate_prediction_feature_columns(columns)
    forbidden = [
        column
        for column in columns
        if any(fragment in column.lower() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"S3 feature scope contains forbidden columns: {forbidden}")
    return columns


def _comparisons(config: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "id": str(item["id"]),
            "candidate": str(item["candidate"]),
            "reference": str(item["reference"]),
        }
        for item in config["comparisons"]
    ]


def _overall_decision(
    year_summary: dict[str, Any], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, dict[str, str]] = {}
    for comparison_id, payload in year_summary["comparisons"].items():
        interval = bootstrap["paired"][comparison_id]
        comparisons[comparison_id] = {
            path: classify_s1_comparison(payload, interval, path=path)
            for path in ("probability", "ranking")
        }
    ids = tuple(comparisons)
    same_path_support = {
        path: all(comparisons[item][path] == "supported" for item in ids)
        for path in ("probability", "ranking")
    }
    if any(same_path_support.values()):
        overall = "supported"
    else:
        weak_path = any(
            "supported" in [comparisons[item][path] for item in ids]
            and all(
                comparisons[item][path] in {"supported", "weakly_supported", "inconclusive"}
                for item in ids
            )
            for path in ("probability", "ranking")
        )
        all_rejected = all(
            all(label == "rejected" for label in paths.values())
            for paths in comparisons.values()
        )
        overall = "weakly_supported" if weak_path else "rejected" if all_rejected else "inconclusive"
    if same_path_support["probability"] and same_path_support["ranking"]:
        next_recommendation = "human_review_performance_modeling_before_S2"
    elif same_path_support["ranking"] and not same_path_support["probability"]:
        next_recommendation = "S2_race_wise_probability"
    elif overall in {"rejected", "inconclusive"}:
        next_recommendation = "S2_race_wise_probability"
    else:
        next_recommendation = "human_review_S2_vs_performance_modeling"
    return {
        "schema_version": 1,
        "comparisons": comparisons,
        "same_path_supported_across_scopes": same_path_support,
        "S3": overall,
        "next_recommendation": next_recommendation,
        "production_control_change": False,
        "S2_executed": False,
        "additional_feature_experiment_executed": False,
    }


def _target_diagnostics(
    frame: pd.DataFrame, *, method: str | None = None
) -> dict[str, Any]:
    values = pd.to_numeric(frame[PERFORMANCE_TARGET_COLUMN], errors="coerce")
    finite = values.notna() & np.isfinite(values)
    payload: dict[str, Any] = {
        "runner_count": int(len(frame)),
        "race_count": int(frame["race_id"].nunique()),
        "nonmissing": int(finite.sum()),
        "missing": int((~finite).sum()),
        "coverage": float(finite.mean()),
        "clip_low_count": int(values.eq(-5.0).sum()),
        "clip_high_count": int(values.eq(5.0).sum()),
        "quantiles": {
            str(q): float(values.loc[finite].quantile(q))
            for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
    }
    if method is not None and finite.any():
        errors = pd.to_numeric(frame.loc[finite, "utility"], errors="raise") - values.loc[finite]
        correlations = []
        for _, race in frame.loc[finite].groupby("race_id", sort=False):
            if len(race) >= 2 and race[PERFORMANCE_TARGET_COLUMN].nunique() > 1:
                correlations.append(
                    race[["utility", PERFORMANCE_TARGET_COLUMN]].corr(method="spearman").iloc[0, 1]
                )
        payload.update(
            {
                "method": method,
                "mae": float(errors.abs().mean()),
                "race_wise_spearman": float(np.nanmean(correlations)) if correlations else None,
            }
        )
    return payload


def run_s3_performance_target_study(
    repo_root: str | Path,
    raw_path: str | Path,
    preregistration_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the 12 preregistered S3 fits and atomically write local artifacts."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    prereg_path = Path(preregistration_path)
    if not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S3 artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(prereg_path)
        validate_s3_preregistration(config)
        normalized, source_audit = isolate_s3_source(
            raw_path,
            maximum_outcome_year=int(config["maximum_outcome_year"]),
            expected_sha256=str(config["raw_sha256"]),
        )
        frame, all_features, base_audit = _build_base_frame(normalized, root)
        controls = _resolve_controls(root, all_features, config)
        feature_scopes = {
            "binary": _feature_scope(controls["binary"]["control_columns"], []),
            "lambdarank": _feature_scope(controls["lambdarank"]["control_columns"], []),
        }

        predictions: list[pd.DataFrame] = []
        fold_records: dict[str, Any] = {}
        target_audits: dict[str, Any] = {}
        target_diagnostics: dict[str, Any] = {}
        slice_records: list[dict[str, Any]] = []
        for fold in config["folds"]:
            targets, target_audit = build_fold_performance_targets(
                normalized, fold, ConditionAdjustedPerformanceTargetSpec()
            )
            target_audits[str(fold["id"])] = target_audit
            joined = frame.merge(
                targets,
                on=["race_id", "horse_id", "race_date"],
                how="left",
                validate="one_to_one",
            )
            joined[PERFORMANCE_TARGET_COLUMN] = pd.to_numeric(
                joined[PERFORMANCE_TARGET_COLUMN], errors="coerce"
            ).astype("float32")
            roles = assign_fold_roles(joined, fold)
            eligible = roles.notna()
            fold_frame = joined.loc[eligible].copy()
            fold_frame["rolling_role"] = roles.loc[eligible].to_numpy()
            if pd.to_datetime(fold_frame["race_date"]).dt.year.max() > 2022:
                raise AssertionError("S3 fold opened a post-2022 row")
            for role, rows in fold_frame.groupby("rolling_role", observed=True):
                target_diagnostics[f"{fold['id']}::{role}"] = _target_diagnostics(rows)

            methods = (
                ("binary_control", "binary", "binary"),
                ("lambdarank_control", "lambdarank", "lambdarank"),
                ("huber_binary_scope", "huber", "binary"),
                ("huber_lambdarank_scope", "huber", "lambdarank"),
            )
            records: dict[str, Any] = {}
            for method, objective, scope_family in methods:
                feature_columns = feature_scopes[scope_family]
                common = {
                    "frame": fold_frame,
                    "feature_columns": feature_columns,
                    "train_split": "train",
                    "model_validation_split": "model_validation",
                    "race_id_column": "race_id",
                    "split_column": "rolling_role",
                }
                if objective == "binary":
                    model_config = controls["binary"]["model_config"]
                    model = train_binary(
                        **common,
                        finish_position_column="model_finish_position",
                        params=model_config["parameters"],
                        early_stopping_rounds=model_config.get("early_stopping_rounds"),
                    )
                elif objective == "lambdarank":
                    model_config = controls["lambdarank"]["model_config"]
                    model = train_ranker(
                        **common,
                        finish_position_column="model_finish_position",
                        params=model_config["parameters"],
                        early_stopping_rounds=model_config.get("early_stopping_rounds"),
                    )
                else:
                    regression = dict(config["regression_parameters"])
                    early_stopping = int(regression.pop("early_stopping_rounds"))
                    regression.pop("sample_weight", None)
                    model = train_huber_regressor(
                        **common,
                        target_column=PERFORMANCE_TARGET_COLUMN,
                        params=regression,
                        early_stopping_rounds=early_stopping,
                    )

                scoring = fold_frame.loc[
                    fold_frame["rolling_role"].isin(("calibration", "evaluation"))
                ].copy()
                raw = predict(
                    model,
                    scoring,
                    feature_columns=feature_columns,
                    model_kind=objective,
                )
                utility = (
                    probability_logits(raw, epsilon=PROBABILITY_EPSILON)
                    if objective == "binary"
                    else raw
                )
                scoring["raw_output"] = raw
                scoring["utility"] = utility
                calibration = scoring.loc[scoring["rolling_role"].eq("calibration")]
                calibrator = fit_temperature(
                    calibration["utility"],
                    calibration["race_id"],
                    calibration["model_finish_position"],
                )
                scoring["probability_calibrated"] = apply_temperature(
                    calibrator, scoring["utility"], scoring["race_id"]
                )
                scoring["fold_id"] = str(fold["id"])
                scoring["role"] = scoring["rolling_role"]
                scoring["evaluation_year"] = int(fold["evaluation_year"])
                scoring["method"] = method
                scoring["family"] = scope_family
                scoring["arm"] = objective
                predictions.append(
                    scoring.loc[
                        :,
                        [
                            "fold_id",
                            "role",
                            "evaluation_year",
                            "race_id",
                            "race_date",
                            "method",
                            "family",
                            "arm",
                            "model_finish_position",
                            PERFORMANCE_TARGET_COLUMN,
                            "raw_output",
                            "utility",
                            "probability_calibrated",
                        ],
                    ]
                )
                evaluation = scoring.loc[scoring["rolling_role"].eq("evaluation")].copy()
                metrics = _metric_payload(evaluation)
                target_diagnostics[f"{fold['id']}::{method}::evaluation"] = _target_diagnostics(
                    evaluation, method=method
                )
                reliability = race_balanced_reliability(
                    evaluation["probability_calibrated"],
                    evaluation["model_finish_position"].astype(int),
                    evaluation["race_id"],
                    n_bins=10,
                    strategy="fixed",
                )
                evaluated = evaluate_predictions(
                    evaluation["probability_calibrated"],
                    evaluation["model_finish_position"].astype(int),
                    evaluation["race_id"],
                    ranking_scores=evaluation["utility"],
                )
                model_dir = temporary / "models" / str(fold["id"])
                model_dir.mkdir(parents=True, exist_ok=True)
                model.booster_.save_model(str(model_dir / f"{method}.txt"))
                flags = assign_s1_slice_flags(evaluation)
                for slice_name, flag in flags.items():
                    selected = evaluation.loc[flag]
                    if selected.empty:
                        continue
                    slice_records.append(
                        {
                            "fold_id": str(fold["id"]),
                            "evaluation_year": int(fold["evaluation_year"]),
                            "method": method,
                            "family": scope_family,
                            "arm": objective,
                            "slice": slice_name,
                            "race_count": int(selected["race_id"].nunique()),
                            "runner_count": int(len(selected)),
                            **_metric_payload(selected),
                        }
                    )
                records[method] = {
                    "objective": objective,
                    "feature_scope": scope_family,
                    "feature_count": len(feature_columns),
                    "feature_columns_sha256": feature_columns_checksum(feature_columns),
                    "best_iteration": int(model.best_iteration_) if getattr(model, "best_iteration_", None) else None,
                    "temperature": float(calibrator.temperature),
                    "calibration_slope_equivalent": float(1.0 / calibrator.temperature),
                    "calibration_intercept_identified": 0.0,
                    "metrics": metrics,
                    "reliability_fixed_bins": reliability,
                    "evaluation": evaluated,
                    "evaluation_choice_set_runner_count": int(len(evaluation)),
                    "evaluation_choice_set_race_count": int(evaluation["race_id"].nunique()),
                }
            fold_records[str(fold["id"])] = records

        prediction_frame = pd.concat(predictions, ignore_index=True)
        race_metrics = _race_metric_table(prediction_frame)
        comparisons = _comparisons(config)
        year_summary = _year_summary(race_metrics, comparisons)
        uncertainty = config["uncertainty"]
        bootstrap = _paired_bootstrap(
            race_metrics,
            comparisons,
            n_resamples=int(uncertainty["resamples"]),
            confidence_level=float(uncertainty["confidence_level"]),
            seed=int(uncertainty["seed"]),
            block_length_dates=int(uncertainty["block_length_dates"]),
        )
        decision = _overall_decision(year_summary, bootstrap)
        feature_schema = {
            "schema_version": 1,
            "target_is_feature": False,
            "S1_performance_feature_added": False,
            "scopes": {
                family: {
                    "feature_count": len(columns),
                    "feature_columns": list(columns),
                    "feature_columns_sha256": feature_columns_checksum(columns),
                    "market_feature_count": 0,
                    "direct_entity_id_feature_count": 0,
                    "target_or_current_outcome_feature_count": 0,
                }
                for family, columns in feature_scopes.items()
            },
        }
        metrics_payload = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "scope": {
                "evaluation_years": [2020, 2021, 2022],
                "maximum_outcome_year": 2022,
                "rows_used_2023": 0,
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "final_market_used_for_selection": False,
                "direct_entity_id_feature_count": 0,
                "fit_count": 12,
            },
            "data": {**source_audit, **base_audit},
            "folds": fold_records,
            "normalizer_audits": target_audits,
            "target_diagnostics": target_diagnostics,
            "year_summary": year_summary,
            "paired_block_bootstrap": bootstrap,
            "slices": slice_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(temporary / "config.json", config)
        write_json(temporary / "metrics.json", metrics_payload)
        write_json(temporary / "comparison.json", {"year_summary": year_summary, "bootstrap": bootstrap})
        write_json(temporary / "decision.json", decision)
        write_json(temporary / "target_diagnostics.json", target_diagnostics)
        write_json(temporary / "normalizer_audits.json", target_audits)
        write_json(temporary / "feature_schema.json", feature_schema)
        write_json(
            temporary / "run_meta.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "preregistration_sha256": sha256_file(prereg_path),
                "preregistration_hash": canonical_json_hash(config),
                "raw_sha256": source_audit["raw_sha256"],
                "git": git_state(root),
                "software": _software_versions(),
                "rows_used_2023": 0,
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
            },
        )
        prediction_dir = temporary / "predictions"
        prediction_dir.mkdir()
        prediction_frame.to_csv(prediction_dir / "scoring.csv.gz", index=False, compression="gzip")
        table_dir = temporary / "tables"
        table_dir.mkdir()
        race_metrics.to_csv(table_dir / "race_metrics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(slice_records).to_csv(table_dir / "slice_metrics.csv", index=False)
        write_artifact_manifest(temporary)
        temporary.replace(output)
        verification = verify_artifact_manifest(output)
        return {**metrics_payload, "decision": decision, "artifact_verification": verification}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
