"""Preregistered S2 rolling study for direct race-wise winner probability."""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
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
from horse_pred.evaluation import race_balanced_reliability, race_log_loss
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    predict,
    probability_logits,
    race_softmax,
    train_binary,
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.racewise_probability import (
    LinearUtilityModel,
    fit_linear_utility,
    native_race_probability,
    predict_racewise_utility,
    train_nonlinear_racewise,
)
from horse_pred.rolling_evaluation import assign_fold_roles
from horse_pred.s1_two_axis_study import (
    _build_base_frame,
    _feature_scope,
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
from horse_pred.two_axis_race_value import (
    PERFORMANCE_COLUMN,
    TwoAxisRaceValueSpec,
    build_fold_two_axis_history,
)

_METRICS = (
    "ndcg_at_3",
    "top_1_winner_mass",
    "winner_reciprocal_rank",
    "race_log_loss",
    "race_brier",
)


def _canonical_fold(fold: dict[str, Any]) -> dict[str, Any]:
    years = [int(value) for value in fold["train_years"]]
    return {
        "id": str(fold["fold_id"]),
        "train_start_year": min(years),
        "train_end_year": max(years),
        "early_stopping_year": int(fold["model_validation_year"]),
        "calibration_year": int(fold["calibration_year"]),
        "evaluation_year": int(fold["evaluation_year"]),
    }


def validate_s2_preregistration(config: dict[str, Any]) -> None:
    """Fail closed on the committed S2 design before loading outcomes."""

    required = {
        "experiment_id",
        "source_sha256",
        "max_source_date",
        "forbidden_years",
        "market_used",
        "final_odds_used",
        "arms",
        "folds",
        "linear_stage",
        "capacity_gate",
        "conditional_nonlinear_stage",
        "probability",
        "metrics",
        "bootstrap",
        "acceptance",
        "stop_after_s2",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"S2 preregistration is missing: {missing}")
    if config["max_source_date"] != "2022-12-31":
        raise ValueError("S2 source cutoff changed")
    if set(map(int, config["forbidden_years"])) != {2023, 2024, 2025}:
        raise ValueError("S2 forbidden years changed")
    if config["market_used"] is not False or config["final_odds_used"] is not False:
        raise ValueError("S2 must keep market information isolated")
    expected_arms = {
        "B0": ("lightgbm_binary", "pv01_254"),
        "B1": ("lightgbm_binary", "pv01_254_plus_s1_performance"),
        "R0": ("linear_conditional_logit", "pv01_254"),
        "R1": ("linear_conditional_logit", "pv01_254_plus_s1_performance"),
    }
    if {
        key: (value["model"], value["feature_scope"])
        for key, value in config["arms"].items()
    } != expected_arms:
        raise ValueError("S2 four-arm design changed")
    folds = [_canonical_fold(fold) for fold in config["folds"]]
    if len(folds) != 3:
        raise ValueError("S2 requires three folds")
    for fold, year in zip(folds, (2020, 2021, 2022)):
        if (
            fold["train_start_year"],
            fold["train_end_year"],
            fold["early_stopping_year"],
            fold["calibration_year"],
            fold["evaluation_year"],
        ) != (2014, year - 3, year - 2, year - 1, year):
            raise ValueError("S2 rolling fold rule changed")
    linear = config["linear_stage"]
    if linear["optimizer"] != "scipy_lbfgsb" or linear["selection_role"] != "model_validation_native_race_log_loss":
        raise ValueError("S2 linear optimizer/selection changed")
    if list(map(float, linear["l2_grid"])) != [0.0001, 0.001, 0.01]:
        raise ValueError("S2 L2 grid changed")
    gate = config["capacity_gate"]
    if gate["uses_evaluation"] is not False or float(gate["threshold"]) != 0.75:
        raise ValueError("S2 capacity gate changed")
    if tuple(config["metrics"]) != (
        "ndcg_at_3",
        "top1",
        "winner_mrr",
        "race_log_loss",
        "race_brier",
        "calibration",
    ):
        raise ValueError("S2 metrics changed")
    if config["probability"]["temperature_fit_role"] != "calibration_only":
        raise ValueError("S2 calibration scope changed")
    if config["stop_after_s2"] is not True:
        raise ValueError("S2 stop rule changed")


def isolate_s2_source(
    raw_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return isolate_s1_source(
        raw_path,
        maximum_outcome_year=2022,
        expected_sha256=expected_sha256,
    )


def _control_config() -> dict[str, Any]:
    return {
        "controls": {
            "binary": {
                "feature_config": "configs/performance/pv_001_candidate_signed_time_gap.json",
                "feature_count": 254,
            },
            "lambdarank": {
                "feature_config": "configs/performance/pv_001_control_lean.json",
                "feature_count": 253,
            },
        }
    }


def _fold_frame(
    normalized: pd.DataFrame,
    base_frame: pd.DataFrame,
    fold: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history, _observations, audit = build_fold_two_axis_history(
        normalized, fold, TwoAxisRaceValueSpec()
    )
    joined = base_frame.merge(
        history.loc[:, ["race_id", "horse_id", PERFORMANCE_COLUMN]],
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    joined[PERFORMANCE_COLUMN] = pd.to_numeric(
        joined[PERFORMANCE_COLUMN], errors="coerce"
    ).astype("float32")
    roles = assign_fold_roles(joined, fold)
    fold_frame = joined.loc[roles.notna()].copy()
    fold_frame["rolling_role"] = roles.loc[roles.notna()].to_numpy()
    fold_frame = fold_frame.sort_values(
        ["race_date", "race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)
    if pd.to_datetime(fold_frame["race_date"]).dt.year.max() > 2022:
        raise AssertionError("S2 fold opened a post-2022 row")
    return fold_frame, audit


def _fit_linear_from_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    config: dict[str, Any],
    *,
    kind: str,
) -> LinearUtilityModel:
    train = frame.loc[frame["rolling_role"].eq("train")]
    validation = frame.loc[frame["rolling_role"].eq("model_validation")]
    linear = config["linear_stage"]
    return fit_linear_utility(
        train.loc[:, list(feature_columns)],
        train["race_id"].tolist(),
        train["model_finish_position"].astype(int).tolist(),
        validation.loc[:, list(feature_columns)],
        validation["race_id"].tolist(),
        validation["model_finish_position"].astype(int).tolist(),
        feature_columns,
        kind=kind,
        l2_grid=linear["l2_grid"],
        z_clip=float(linear["z_clip"]),
        max_iterations=int(linear["max_iterations"]),
        ftol=float(linear["ftol"]),
        gtol=float(linear["gtol"]),
    )


def _native_log_loss(
    utilities: Sequence[float], frame: pd.DataFrame
) -> float:
    probabilities = race_softmax(utilities, frame["race_id"])
    return race_log_loss(
        probabilities,
        frame["model_finish_position"].astype(int),
        frame["race_id"],
    )


def _capacity_gate(records: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["capacity_gate"]["threshold"])
    required = int(config["capacity_gate"]["trigger_folds"])
    under = 0
    folds: dict[str, Any] = {}
    for fold_id, record in records.items():
        denominator = record["uniform_native_log_loss"] - record["binary_native_log_loss"]
        if denominator <= 0:
            recovery = None
            is_under = True
        else:
            recovery = (
                record["uniform_native_log_loss"] - record["linear_native_log_loss"]
            ) / denominator
            is_under = recovery < threshold
        under += int(is_under)
        folds[fold_id] = {**record, "recovery": recovery, "under_capacity": is_under}
    return {
        "role": "model_validation_only",
        "threshold": threshold,
        "under_capacity_folds": under,
        "trigger_folds": required,
        "nonlinear_stage_triggered": under >= required,
        "folds": folds,
    }


def _score_method(
    model: Any,
    model_kind: str,
    fold_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    fold: dict[str, Any],
    method: str,
    family: str,
    arm: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scoring = fold_frame.loc[
        fold_frame["rolling_role"].isin(("calibration", "evaluation"))
    ].copy()
    if model_kind == "binary":
        raw = np.asarray(
            predict(model, scoring, feature_columns=feature_columns, model_kind="binary"),
            dtype=float,
        )
        utility = np.asarray(
            probability_logits(raw, epsilon=PROBABILITY_EPSILON), dtype=float
        )
    else:
        utility = predict_racewise_utility(model, scoring, feature_columns)
        raw = utility.copy()
    scoring["raw_output"] = raw
    scoring["utility"] = utility
    scoring["probability_native"] = native_race_probability(
        utility, scoring["race_id"].tolist()
    )
    calibration = scoring.loc[scoring["rolling_role"].eq("calibration")]
    calibrator = fit_temperature(
        calibration["utility"],
        calibration["race_id"],
        calibration["model_finish_position"],
    )
    scoring["probability_calibrated"] = apply_temperature(
        calibrator, scoring["utility"], scoring["race_id"]
    )
    scoring["fold_id"] = fold["id"]
    scoring["role"] = scoring["rolling_role"]
    scoring["evaluation_year"] = int(fold["evaluation_year"])
    scoring["method"] = method
    scoring["family"] = family
    scoring["arm"] = arm
    evaluation = scoring.loc[scoring["role"].eq("evaluation")]
    native_ll = race_log_loss(
        evaluation["probability_native"],
        evaluation["model_finish_position"].astype(int),
        evaluation["race_id"],
    )
    record = {
        "model_kind": model_kind,
        "feature_count": len(feature_columns),
        "feature_columns_sha256": feature_columns_checksum(feature_columns),
        "temperature": float(calibrator.temperature),
        "calibration_slope_equivalent": float(1.0 / calibrator.temperature),
        "calibration_intercept_identified": 0.0,
        "native_race_log_loss": float(native_ll),
        "metrics": _metric_payload(evaluation),
        "reliability_fixed_bins": race_balanced_reliability(
            evaluation["probability_calibrated"],
            evaluation["model_finish_position"].astype(int),
            evaluation["race_id"],
            n_bins=10,
            strategy="fixed",
        ),
        "evaluation_runner_count": int(len(evaluation)),
        "evaluation_race_count": int(evaluation["race_id"].nunique()),
    }
    keep = [
        "fold_id",
        "role",
        "evaluation_year",
        "race_id",
        "race_date",
        "method",
        "family",
        "arm",
        "model_finish_position",
        "raw_output",
        "utility",
        "probability_native",
        "probability_calibrated",
    ]
    return scoring.loc[:, keep], record


def _classify_with_native_guard(
    payload: dict[str, Any],
    interval: dict[str, Any],
    native_improvement: float,
    *,
    path: str,
) -> str:
    label = classify_s1_comparison(payload, interval, path=path)
    if path == "probability" and native_improvement < -0.002:
        return "rejected" if label == "supported" else label
    return label


def _decision(
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
    native_summary: dict[str, Any],
    comparisons: list[dict[str, str]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    classified: dict[str, Any] = {}
    for comparison in comparisons:
        comparison_id = comparison["id"]
        native_delta = native_summary[comparison_id]["year_macro_improvement"]
        classified[comparison_id] = {
            "probability_path": _classify_with_native_guard(
                summary["comparisons"][comparison_id],
                bootstrap["paired"][comparison_id],
                native_delta,
                path="probability",
            ),
            "ranking_path": classify_s1_comparison(
                summary["comparisons"][comparison_id],
                bootstrap["paired"][comparison_id],
                path="ranking",
            ),
            "native_log_loss_improvement": native_delta,
        }
    primary_prefix = "nonlinear" if gate["nonlinear_stage_triggered"] else "linear"
    objective_ids = [
        f"{primary_prefix}_R0_vs_binary_B0",
        f"{primary_prefix}_R1_vs_binary_B1",
    ]
    labels = [classified[item]["probability_path"] for item in objective_ids]
    if labels == ["supported", "supported"]:
        objective = "supported"
    elif "supported" in labels and all(
        label in {"supported", "weakly_supported", "inconclusive"} for label in labels
    ):
        objective = "weakly_supported"
    elif set(labels) == {"supported", "rejected"}:
        objective = "inconclusive"
    elif labels == ["rejected", "rejected"]:
        objective = "rejected"
    else:
        objective = "inconclusive"
    binary_feature = classified["binary_B1_vs_B0"]["probability_path"]
    racewise_feature = classified[f"{primary_prefix}_R1_vs_R0"]["probability_path"]
    if binary_feature == racewise_feature == "supported":
        performance = "objective_robust_supported"
    elif "supported" in {binary_feature, racewise_feature}:
        performance = "objective_dependent_inconclusive"
    elif binary_feature == racewise_feature == "rejected":
        performance = "rejected"
    else:
        performance = "inconclusive"
    next_recommendation = (
        "human_review_inter_horse_race_set_model"
        if objective in {"supported", "weakly_supported"}
        else "human_review_feature_representation_before_set_model"
    )
    return {
        "schema_version": 1,
        "capacity_gate": gate,
        "primary_racewise_stage": primary_prefix,
        "comparisons": classified,
        "race_wise_objective": objective,
        "s1_performance_feature_across_objectives": performance,
        "next_recommendation": next_recommendation,
        "production_control_change": False,
        "post_s2_model_executed": False,
    }


def run_s2_racewise_probability_study(
    repo_root: str | Path,
    raw_path: str | Path,
    preregistration_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the preregistered staged S2 comparison and atomically write artifacts."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    prereg_path = Path(preregistration_path)
    if not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S2 artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(prereg_path)
        validate_s2_preregistration(config)
        normalized, source_audit = isolate_s2_source(
            raw_path, expected_sha256=str(config["source_sha256"])
        )
        base_frame, all_features, base_audit = _build_base_frame(normalized, root)
        controls = _resolve_controls(root, all_features, _control_config())
        control_columns = tuple(controls["binary"]["control_columns"])
        feature_scopes = {
            "C0": _feature_scope(control_columns, []),
            "C1": _feature_scope(control_columns, [PERFORMANCE_COLUMN]),
        }
        folds = [_canonical_fold(item) for item in config["folds"]]

        gate_models: dict[str, dict[str, Any]] = {}
        gate_records: dict[str, Any] = {}
        feature_audits: dict[str, Any] = {}
        for fold in folds:
            fold_frame, audit = _fold_frame(normalized, base_frame, fold)
            feature_audits[fold["id"]] = audit
            binary = train_binary(
                fold_frame,
                feature_columns=feature_scopes["C0"],
                train_split="train",
                model_validation_split="model_validation",
                race_id_column="race_id",
                finish_position_column="model_finish_position",
                split_column="rolling_role",
                params=controls["binary"]["model_config"]["parameters"],
                early_stopping_rounds=controls["binary"]["model_config"].get("early_stopping_rounds"),
            )
            linear = _fit_linear_from_frame(
                fold_frame, feature_scopes["C0"], config, kind="conditional_logit"
            )
            linear_binary = _fit_linear_from_frame(
                fold_frame, feature_scopes["C0"], config, kind="linear_binary"
            )
            validation = fold_frame.loc[
                fold_frame["rolling_role"].eq("model_validation")
            ]
            binary_raw = predict(
                binary,
                validation,
                feature_columns=feature_scopes["C0"],
                model_kind="binary",
            )
            binary_utility = probability_logits(binary_raw, epsilon=PROBABILITY_EPSILON)
            linear_utility = predict_racewise_utility(
                linear, validation, feature_scopes["C0"]
            )
            field_sizes = validation.groupby("race_id", sort=False).size().to_numpy()
            gate_records[fold["id"]] = {
                "uniform_native_log_loss": float(np.mean(np.log(field_sizes))),
                "binary_native_log_loss": _native_log_loss(binary_utility, validation),
                "linear_native_log_loss": _native_log_loss(linear_utility, validation),
                "linear_binary_native_log_loss": _native_log_loss(
                    predict_racewise_utility(
                        linear_binary, validation, feature_scopes["C0"]
                    ),
                    validation,
                ),
            }
            gate_models[fold["id"]] = {
                "binary_B0": binary,
                "linear_R0": linear,
                "linear_binary_LB0": linear_binary,
            }
        gate = _capacity_gate(gate_records, config)

        predictions: list[pd.DataFrame] = []
        fold_records: dict[str, Any] = {}
        slice_records: list[dict[str, Any]] = []
        diagnostic_records: dict[str, Any] = {}
        fit_count = 9
        for fold in folds:
            fold_frame, _audit = _fold_frame(normalized, base_frame, fold)
            models = gate_models[fold["id"]]
            models["binary_B1"] = train_binary(
                fold_frame,
                feature_columns=feature_scopes["C1"],
                train_split="train",
                model_validation_split="model_validation",
                race_id_column="race_id",
                finish_position_column="model_finish_position",
                split_column="rolling_role",
                params=controls["binary"]["model_config"]["parameters"],
                early_stopping_rounds=controls["binary"]["model_config"].get("early_stopping_rounds"),
            )
            models["linear_R1"] = _fit_linear_from_frame(
                fold_frame, feature_scopes["C1"], config, kind="conditional_logit"
            )
            models["linear_binary_LB1"] = _fit_linear_from_frame(
                fold_frame, feature_scopes["C1"], config, kind="linear_binary"
            )
            fit_count += 3
            if gate["nonlinear_stage_triggered"]:
                for arm, scope in (("N0", "C0"), ("N1", "C1")):
                    models[f"nonlinear_{arm}"] = train_nonlinear_racewise(
                        fold_frame,
                        feature_columns=feature_scopes[scope],
                        params=controls["binary"]["model_config"]["parameters"],
                        early_stopping_rounds=int(
                            controls["binary"]["model_config"].get("early_stopping_rounds", 50)
                        ),
                    )
                    fit_count += 1

            methods: list[tuple[str, str, str, str]] = [
                ("binary_B0", "binary", "C0", "binary"),
                ("binary_B1", "binary", "C1", "binary"),
                ("linear_R0", "linear", "C0", "racewise"),
                ("linear_R1", "linear", "C1", "racewise"),
            ]
            if gate["nonlinear_stage_triggered"]:
                methods.extend(
                    [
                        ("nonlinear_N0", "nonlinear", "C0", "racewise"),
                        ("nonlinear_N1", "nonlinear", "C1", "racewise"),
                    ]
                )
            records: dict[str, Any] = {}
            for method, kind, scope, family in methods:
                scored, record = _score_method(
                    models[method],
                    kind,
                    fold_frame,
                    feature_scopes[scope],
                    fold=fold,
                    method=method,
                    family=family,
                    arm=scope,
                )
                predictions.append(scored)
                records[method] = record
                evaluation = scored.loc[scored["role"].eq("evaluation")]
                flags = assign_s1_slice_flags(
                    fold_frame.loc[
                        fold_frame["rolling_role"].eq("evaluation")
                    ].copy()
                )
                for slice_name, flag in flags.items():
                    selected_ids = set(
                        fold_frame.loc[
                            fold_frame["rolling_role"].eq("evaluation")
                        ].loc[flag, "race_id"]
                    )
                    selected = evaluation.loc[evaluation["race_id"].isin(selected_ids)]
                    if not selected.empty:
                        slice_records.append(
                            {
                                "fold_id": fold["id"],
                                "evaluation_year": fold["evaluation_year"],
                                "method": method,
                                "slice": slice_name,
                                "race_count": int(selected["race_id"].nunique()),
                                "runner_count": int(len(selected)),
                                **_metric_payload(selected),
                            }
                        )
                model_dir = temporary / "models" / fold["id"]
                model_dir.mkdir(parents=True, exist_ok=True)
                if isinstance(models[method], LinearUtilityModel):
                    linear_model = models[method]
                    np.savez_compressed(
                        model_dir / f"{method}.npz",
                        coefficients=linear_model.coefficients,
                        medians=linear_model.transform.medians,
                        means=linear_model.transform.means,
                        scales=linear_model.transform.scales,
                        constant_mask=linear_model.transform.constant_mask,
                    )
                    diagnostic_records[f"{fold['id']}::{method}"] = linear_model.optimization
                else:
                    models[method].booster_.save_model(str(model_dir / f"{method}.txt"))
                    gains = models[method].booster_.feature_importance(importance_type="gain")
                    diagnostic_records[f"{fold['id']}::{method}"] = {
                        "best_iteration": int(models[method].best_iteration_ or 0),
                        "total_gain": float(np.sum(gains)),
                        "performance_feature_gain": (
                            float(gains[list(feature_scopes[scope]).index(PERFORMANCE_COLUMN)])
                            if PERFORMANCE_COLUMN in feature_scopes[scope]
                            else None
                        ),
                    }
            diagnostic_records[f"{fold['id']}::linear_binary"] = {
                "LB0": models["linear_binary_LB0"].optimization,
                "LB1": models["linear_binary_LB1"].optimization,
            }
            fold_records[fold["id"]] = records

        prediction_frame = pd.concat(predictions, ignore_index=True)
        race_metrics = _race_metric_table(prediction_frame)
        primary = "nonlinear" if gate["nonlinear_stage_triggered"] else "linear"
        comparisons = [
            {"id": "binary_B1_vs_B0", "candidate": "binary_B1", "reference": "binary_B0"},
            {"id": "linear_R0_vs_binary_B0", "candidate": "linear_R0", "reference": "binary_B0"},
            {"id": "linear_R1_vs_binary_B1", "candidate": "linear_R1", "reference": "binary_B1"},
            {"id": "linear_R1_vs_R0", "candidate": "linear_R1", "reference": "linear_R0"},
        ]
        if gate["nonlinear_stage_triggered"]:
            comparisons.extend(
                [
                    {"id": "nonlinear_R0_vs_binary_B0", "candidate": "nonlinear_N0", "reference": "binary_B0"},
                    {"id": "nonlinear_R1_vs_binary_B1", "candidate": "nonlinear_N1", "reference": "binary_B1"},
                    {"id": "nonlinear_R1_vs_R0", "candidate": "nonlinear_N1", "reference": "nonlinear_N0"},
                ]
            )
        summary = _year_summary(race_metrics, comparisons)
        uncertainty = config["bootstrap"]
        bootstrap = _paired_bootstrap(
            race_metrics,
            comparisons,
            n_resamples=int(uncertainty["resamples"]),
            confidence_level=0.95,
            seed=int(uncertainty["seed"]),
            block_length_dates=4,
        )
        native_years = (
            prediction_frame.loc[prediction_frame["role"].eq("evaluation")]
            .groupby(["evaluation_year", "method"], observed=True)
            .apply(
                lambda rows: race_log_loss(
                    rows["probability_native"],
                    rows["model_finish_position"].astype(int),
                    rows["race_id"],
                ),
                include_groups=False,
            )
            .rename("race_log_loss")
            .reset_index()
        )
        native_index = native_years.set_index(["evaluation_year", "method"])
        native_summary: dict[str, Any] = {}
        for comparison in comparisons:
            values = []
            for year in (2020, 2021, 2022):
                values.append(
                    float(native_index.loc[(year, comparison["reference"]), "race_log_loss"])
                    - float(native_index.loc[(year, comparison["candidate"]), "race_log_loss"])
                )
            native_summary[comparison["id"]] = {
                "per_year_improvement": dict(zip(("2020", "2021", "2022"), values)),
                "year_macro_improvement": float(np.mean(values)),
                "improved_years": int(sum(value > 0 for value in values)),
            }
        decision = _decision(summary, bootstrap, native_summary, comparisons, gate)
        scope = {
            "evaluation_years": [2020, 2021, 2022],
            "maximum_outcome_year": 2022,
            "rows_used_2023": 0,
            "rows_used_2024": 0,
            "rows_used_2025": 0,
            "odds_used": False,
            "final_market_used_for_selection": False,
            "direct_entity_id_feature_count": 0,
            "fit_count": fit_count,
            "primary_racewise_stage": primary,
        }
        metrics_payload = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "scope": scope,
            "data": {**source_audit, **base_audit},
            "feature_scopes": {
                scope_name: {
                    "count": len(columns),
                    "sha256": feature_columns_checksum(columns),
                    "performance_feature_included": PERFORMANCE_COLUMN in columns,
                    "market_feature_count": 0,
                    "direct_entity_id_feature_count": 0,
                }
                for scope_name, columns in feature_scopes.items()
            },
            "capacity_gate": gate,
            "folds": fold_records,
            "year_summary": summary,
            "native_probability_summary": native_summary,
            "paired_block_bootstrap": bootstrap,
            "slices": slice_records,
            "decision": decision,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(temporary / "config.json", config)
        write_json(temporary / "metrics.json", metrics_payload)
        write_json(
            temporary / "comparison.json",
            {"year_summary": summary, "native": native_summary, "bootstrap": bootstrap},
        )
        write_json(temporary / "decision.json", decision)
        write_json(temporary / "capacity_gate.json", gate)
        write_json(temporary / "feature_diagnostics.json", diagnostic_records)
        write_json(temporary / "feature_audits.json", feature_audits)
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
                **scope,
            },
        )
        predictions_dir = temporary / "predictions"
        predictions_dir.mkdir()
        prediction_frame.to_csv(
            predictions_dir / "scoring.csv.gz", index=False, compression="gzip"
        )
        tables_dir = temporary / "tables"
        tables_dir.mkdir()
        race_metrics.to_csv(tables_dir / "race_metrics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(slice_records).to_csv(tables_dir / "slice_metrics.csv", index=False)
        write_artifact_manifest(temporary)
        temporary.replace(output)
        verification = verify_artifact_manifest(output)
        return {**metrics_payload, "artifact_verification": verification}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
