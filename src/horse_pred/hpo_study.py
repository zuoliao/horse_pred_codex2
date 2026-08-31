"""HPO-01: bounded, temporally gated LightGBM parameter screening."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.cached_experiment import (
    feature_columns_checksum,
    resolve_semantic_feature_selection,
    validate_cached_experiment_config,
)
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import evaluate_predictions
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    predict,
    probability_logits,
    race_softmax,
    train_binary,
    train_ranker,
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.rolling_evaluation import (
    assign_fold_roles,
    isolate_rolling_source,
    paired_year_stratified_block_bootstrap,
    rolling_race_metric_table,
    summarize_year_macro,
)

_SELECTION_FOLDS = [
    {
        "id": f"roll_{year}",
        "train_start_year": 2014,
        "train_end_year": year - 3,
        "early_stopping_year": year - 2,
        "calibration_year": year - 1,
        "evaluation_year": year,
    }
    for year in range(2020, 2023)
]
_CONFIRMATION_FOLD = {
    "id": "roll_2023",
    "train_start_year": 2014,
    "train_end_year": 2020,
    "early_stopping_year": 2021,
    "calibration_year": 2022,
    "evaluation_year": 2023,
}
_EXPECTED_PROFILES: list[dict[str, Any]] = [
    {"id": "control", "parameter_overrides": {}, "complexity_rank": 0},
    {"id": "leaves_15", "parameter_overrides": {"num_leaves": 15}, "complexity_rank": 2},
    {"id": "leaves_63", "parameter_overrides": {"num_leaves": 63}, "complexity_rank": 10},
    {"id": "depth_6", "parameter_overrides": {"max_depth": 6}, "complexity_rank": 3},
    {
        "id": "min_child_50",
        "parameter_overrides": {"min_child_samples": 50},
        "complexity_rank": 9,
    },
    {
        "id": "min_child_200",
        "parameter_overrides": {"min_child_samples": 200},
        "complexity_rank": 1,
    },
    {"id": "l1_1", "parameter_overrides": {"reg_alpha": 1.0}, "complexity_rank": 4},
    {"id": "l2_5", "parameter_overrides": {"reg_lambda": 5.0}, "complexity_rank": 5},
    {
        "id": "feature_fraction_075",
        "parameter_overrides": {"colsample_bytree": 0.75},
        "complexity_rank": 6,
    },
    {
        "id": "bagging_fraction_075",
        "parameter_overrides": {"subsample": 0.75},
        "complexity_rank": 7,
    },
    {"id": "max_bin_127", "parameter_overrides": {"max_bin": 127}, "complexity_rank": 8},
    {"id": "max_bin_511", "parameter_overrides": {"max_bin": 511}, "complexity_rank": 11},
]
_PRIMARY_BY_KIND = {"binary": "race_log_loss", "lambdarank": "ndcg_at_3"}
_SECONDARY_BY_KIND = {
    "binary": ("race_brier", "ndcg_at_3", "top_1_winner_mass"),
    "lambdarank": ("top_1_winner_mass", "race_log_loss", "race_brier"),
}


def validate_hpo_config(config: dict[str, Any]) -> None:
    """Require the exact preregistered HPO-01 search surface."""

    required = {
        "schema_version",
        "experiment_id",
        "hypothesis",
        "seed",
        "model_kind",
        "maximum_outcome_year",
        "feature_config",
        "base_model_config",
        "expected_feature_count",
        "expected_columns_sha256",
        "selection_folds",
        "confirmation_fold",
        "profiles",
        "selection",
        "selection_accounting",
        "uncertainty",
    }
    if set(config) != required:
        raise ValueError(f"HPO config keys differ from preregistration: {sorted(set(config) ^ required)}")
    if config["schema_version"] != 1 or config["seed"] != 42:
        raise ValueError("HPO-01 requires schema version 1 and seed 42")
    kind = config["model_kind"]
    if kind not in _PRIMARY_BY_KIND:
        raise ValueError("model_kind must be binary or lambdarank")
    if config["maximum_outcome_year"] != 2023:
        raise ValueError("HPO-01 must not open outcomes after 2023")
    if config["selection_folds"] != _SELECTION_FOLDS:
        raise ValueError("HPO-01 selection folds must remain 2020--2022")
    if config["confirmation_fold"] != _CONFIRMATION_FOLD:
        raise ValueError("HPO-01 confirmation must remain the single 2023 fold")
    if config["profiles"] != _EXPECTED_PROFILES:
        raise ValueError("HPO-01 parameter profiles differ from preregistration")
    if int(config["expected_feature_count"]) < 1:
        raise ValueError("expected_feature_count must be positive")
    checksum = config["expected_columns_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("expected_columns_sha256 must be a SHA-256 digest")

    selection = config["selection"]
    expected_selection = {
        "primary_metric",
        "primary_improvement_min",
        "minimum_improved_selection_years",
        "primary_tie_tolerance",
        "guardrail_minimum_improvements",
    }
    if set(selection) != expected_selection:
        raise ValueError("HPO-01 selection keys differ from preregistration")
    if selection["primary_metric"] != _PRIMARY_BY_KIND[kind]:
        raise ValueError("primary metric disagrees with model kind")
    if int(selection["minimum_improved_selection_years"]) != 2:
        raise ValueError("HPO-01 requires improvement in at least two selection years")
    if float(selection["primary_tie_tolerance"]) != 0.0001:
        raise ValueError("HPO-01 primary tie tolerance must remain 1e-4")
    expected_guardrails = (
        {"race_brier": 0.0, "ndcg_at_3": -0.002, "top_1_winner_mass": -0.005}
        if kind == "binary"
        else {"race_log_loss": -0.002, "race_brier": -0.001, "top_1_winner_mass": -0.005}
    )
    if selection["guardrail_minimum_improvements"] != expected_guardrails:
        raise ValueError("HPO-01 guardrails differ from the registered family rule")
    expected_primary_min = 0.002 if kind == "binary" else 0.0
    if float(selection["primary_improvement_min"]) != expected_primary_min:
        raise ValueError("HPO-01 primary improvement minimum differs from preregistration")

    accounting = config["selection_accounting"]
    if int(accounting.get("selection_candidate_control_comparisons", -1)) != 11:
        raise ValueError("HPO-01 must record eleven selection comparisons")
    if int(accounting.get("confirmation_comparisons_maximum", -1)) != 1:
        raise ValueError("HPO-01 must expose at most one candidate to 2023")
    uncertainty = config["uncertainty"]
    if int(uncertainty.get("bootstrap_resamples", 0)) < 2:
        raise ValueError("bootstrap_resamples must be at least two")
    if int(uncertainty.get("block_length_dates", 0)) < 1:
        raise ValueError("block_length_dates must be positive")
    if not 0.0 < float(uncertainty.get("confidence_level", 0.0)) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_hpo_scope(
    *, root: Path, all_feature_columns: tuple[str, ...], config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the one frozen feature scope and base model configuration."""

    feature_path = _resolve(root, config["feature_config"])
    feature_config = load_json(feature_path)
    validate_cached_experiment_config(feature_config)
    if set(feature_config["feature_selection"]) != {"include"} or feature_config.get(
        "derived_features"
    ):
        raise ValueError("HPO-01 requires an include-only precomputed PIT feature scope")
    columns, groups, resolution = resolve_semantic_feature_selection(
        all_feature_columns, feature_config
    )
    checksum = feature_columns_checksum(columns)
    if len(columns) != config["expected_feature_count"] or checksum != config["expected_columns_sha256"]:
        raise ValueError("HPO-01 feature scope disagrees with its count or ordered hash")

    model_path = _resolve(root, config["base_model_config"])
    model_config = load_json(model_path)
    kind = config["model_kind"]
    family = "lightgbm_binary" if kind == "binary" else "lightgbm_lambdarank"
    if model_config.get("model_family") != family:
        raise ValueError("HPO-01 base model family disagrees with model_kind")
    if int(model_config.get("seed", -1)) != config["seed"]:
        raise ValueError("HPO-01 base model seed differs from experiment seed")
    params = model_config.get("parameters", {})
    if params.get("objective") != kind or int(params.get("random_state", -1)) != config["seed"]:
        raise ValueError("HPO-01 objective or random_state differs from preregistration")
    fixed_expected = {
        "learning_rate": 0.05,
        "n_estimators": 500,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "deterministic": True,
        "force_col_wise": True,
    }
    if any(params.get(name) != expected for name, expected in fixed_expected.items()):
        raise ValueError("HPO-01 base parameters differ from the frozen incumbent")
    if float(params.get("reg_alpha", 0.0)) != 0.0:
        raise ValueError("HPO-01 base reg_alpha must remain zero")
    if int(params.get("max_depth", -1)) != -1 or int(params.get("max_bin", 255)) != 255:
        raise ValueError("HPO-01 base depth and max_bin must remain at incumbent defaults")
    if int(model_config.get("early_stopping_rounds", -1)) != 50:
        raise ValueError("HPO-01 early stopping must remain 50 rounds")
    if kind == "lambdarank":
        if list(params.get("label_gain", [])) != [0, 1, 3, 7]:
            raise ValueError("HPO-01 Ranker label_gain must remain [0, 1, 3, 7]")
        if list(params.get("eval_at", [])) != [1, 3, 5]:
            raise ValueError("HPO-01 Ranker eval_at must remain [1, 3, 5]")
        if int(params.get("lambdarank_truncation_level", 6)) != 6:
            raise ValueError("Ranker truncation tuning is outside HPO-01")
    return {
        "feature_columns": columns,
        "feature_groups": groups,
        "feature_resolution": resolution,
        "feature_columns_sha256": checksum,
        "feature_config": feature_config,
        "feature_config_path": feature_path,
        "model_config": model_config,
        "model_config_path": model_path,
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None and int(value) > 0 else None


def _primary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "ndcg_at_3": float(payload["ranking"]["ndcg_at_3"]),
        "top_1_winner_mass": float(payload["ranking"]["top_1"]),
        "race_log_loss": float(payload["probability"]["race_log_loss"]),
        "race_brier": float(payload["probability"]["race_brier"]),
    }


def _fit_stage(
    *,
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    scope: dict[str, Any],
    model_kind: str,
    stage: str,
    model_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    records: dict[str, Any] = {}
    importance: list[dict[str, Any]] = []
    feature_columns = scope["feature_columns"]
    base = scope["model_config"]
    for fold in folds:
        roles = assign_fold_roles(frame, fold)
        eligible = roles.notna()
        fold_frame = frame.loc[eligible].copy()
        fold_frame["rolling_role"] = roles.loc[eligible].to_numpy()
        fold_records: dict[str, Any] = {}
        for profile in profiles:
            method = profile["id"]
            params = {**base["parameters"], **profile["parameter_overrides"]}
            common = {
                "frame": fold_frame,
                "feature_columns": feature_columns,
                "train_split": "train",
                "model_validation_split": "model_validation",
                "race_id_column": "race_id",
                "finish_position_column": "model_finish_position",
                "split_column": "rolling_role",
                "params": params,
                "early_stopping_rounds": base.get("early_stopping_rounds"),
            }
            model = train_binary(**common) if model_kind == "binary" else train_ranker(**common)
            scoring = fold_frame.loc[
                fold_frame["rolling_role"].isin(("calibration", "evaluation"))
            ].copy()
            raw = predict(model, scoring, feature_columns=feature_columns, model_kind=model_kind)
            utility = (
                probability_logits(raw, epsilon=PROBABILITY_EPSILON)
                if model_kind == "binary"
                else raw
            )
            scoring["raw_output"] = raw
            scoring["utility"] = utility
            calibration = scoring.loc[scoring["rolling_role"].eq("calibration")]
            calibrator = fit_temperature(
                calibration["utility"], calibration["race_id"], calibration["model_finish_position"]
            )
            scoring["probability_t1"] = race_softmax(scoring["utility"], scoring["race_id"])
            scoring["probability_calibrated"] = apply_temperature(
                calibrator, scoring["utility"], scoring["race_id"]
            )
            scoring["fold_id"] = fold["id"]
            scoring["role"] = scoring["rolling_role"]
            scoring["evaluation_year"] = int(fold["evaluation_year"])
            scoring["method"] = method
            scoring["model_kind"] = model_kind
            scoring["stage"] = stage
            columns = [
                "stage",
                "fold_id",
                "role",
                "evaluation_year",
                "race_id",
                "race_date",
                "method",
                "model_kind",
                "model_finish_position",
                "raw_output",
                "utility",
                "probability_t1",
                "probability_calibrated",
            ]
            for optional in ("horse_id", "horse_number", "field_size"):
                if optional in scoring:
                    columns.insert(9, optional)
            predictions.append(scoring.loc[:, columns])

            evaluation = scoring.loc[scoring["rolling_role"].eq("evaluation")]
            evaluated = evaluate_predictions(
                evaluation["probability_calibrated"],
                pd.to_numeric(evaluation["model_finish_position"], errors="raise").astype(int),
                evaluation["race_id"],
                ranking_scores=evaluation["utility"],
            )
            directory = model_root / stage / str(fold["id"])
            directory.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(directory / f"{method}.txt"))
            gains = model.booster_.feature_importance(importance_type="gain")
            splits = model.booster_.feature_importance(importance_type="split")
            gain_total = float(np.sum(gains))
            for feature, gain, split in zip(feature_columns, gains, splits):
                importance.append(
                    {
                        "stage": stage,
                        "fold_id": fold["id"],
                        "evaluation_year": fold["evaluation_year"],
                        "method": method,
                        "feature": feature,
                        "importance_gain": float(gain),
                        "importance_gain_fraction": float(gain) / gain_total if gain_total else 0.0,
                        "importance_split": int(split),
                    }
                )
            effective = model.get_params(deep=False) if hasattr(model, "get_params") else params
            fold_records[method] = {
                "best_iteration": _best_iteration(model),
                "temperature": float(calibrator.temperature),
                "effective_parameters": effective,
                "primary_metrics": _primary_metrics(evaluated),
            }
        records[str(fold["id"])] = fold_records
    return pd.concat(predictions, ignore_index=True), records, importance


def _comparisons(profiles: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "id": f"{profile['id']}_vs_control",
            "candidate": profile["id"],
            "reference": "control",
            "type": "hypothesis",
        }
        for profile in profiles
        if profile["id"] != "control"
    ]


def select_hpo_profile(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Select at most one profile using only the three selection years."""

    selection = config["selection"]
    primary = selection["primary_metric"]
    guardrails = selection["guardrail_minimum_improvements"]
    diagnostics: dict[str, Any] = {}
    eligible: list[tuple[str, dict[str, Any]]] = []
    for profile in config["profiles"]:
        if profile["id"] == "control":
            continue
        comparison = summary["comparisons"][f"{profile['id']}_vs_control"]
        metrics = comparison["metrics"]
        reasons: list[str] = []
        primary_value = float(metrics[primary]["year_macro_improvement"])
        primary_minimum = float(selection["primary_improvement_min"])
        primary_point_failed = (
            primary_value <= 0.0
            if config["model_kind"] == "lambdarank"
            else primary_value < primary_minimum
        )
        if primary_point_failed:
            reasons.append("primary_point_below_minimum")
        if metrics[primary]["improved_years"] < int(selection["minimum_improved_selection_years"]):
            reasons.append("primary_direction_below_minimum")
        for metric, minimum in guardrails.items():
            if metrics[metric]["year_macro_improvement"] < float(minimum):
                reasons.append(f"guardrail_failed:{metric}")
        payload = {"eligible": not reasons, "reasons": reasons, "metrics": metrics}
        diagnostics[profile["id"]] = payload
        if not reasons:
            eligible.append((profile["id"], metrics))
    if not eligible:
        return {"selected_profile": None, "decision": "no_change", "profiles": diagnostics}

    best_primary = max(item[1][primary]["year_macro_improvement"] for item in eligible)
    tolerance = float(selection["primary_tie_tolerance"])
    contenders = [item for item in eligible if best_primary - item[1][primary]["year_macro_improvement"] <= tolerance]
    complexity = {profile["id"]: int(profile["complexity_rank"]) for profile in config["profiles"]}
    secondary = _SECONDARY_BY_KIND[config["model_kind"]]
    contenders.sort(
        key=lambda item: (
            *(-item[1][metric]["year_macro_improvement"] for metric in secondary),
            complexity[item[0]],
            item[0],
        )
    )
    return {
        "selected_profile": contenders[0][0],
        "decision": "selected_for_confirmation",
        "profiles": diagnostics,
        "primary_best": float(best_primary),
        "primary_tie_tolerance": tolerance,
    }


def confirmation_decision(
    *, improvement: dict[str, Any], interval: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Apply the registered family-specific 2023 confirmation rule."""

    selection = config["selection"]
    primary = selection["primary_metric"]
    guardrail_failed = any(
        improvement[metric] < float(minimum)
        for metric, minimum in selection["guardrail_minimum_improvements"].items()
    )
    primary_point_passed = improvement[primary] >= float(selection["primary_improvement_min"])
    primary_interval_passed = float(interval[primary]["lower"]) > 0.0
    if primary_point_passed and primary_interval_passed and not guardrail_failed:
        decision = "accept"
    elif guardrail_failed or float(interval[primary]["upper"]) < 0.0:
        decision = "reject"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "primary_metric": primary,
        "primary_point_passed": primary_point_passed,
        "primary_interval_passed": primary_interval_passed,
        "guardrail_failed": guardrail_failed,
    }


def _software_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed-as-package"
    return result


def run_hpo_study(
    *, repo_root: str | Path, cache_path: str | Path, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run one preregistered HPO family without exposing all trials to 2023."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    cache_file = _resolve(root, cache_path)
    config_file = _resolve(root, config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite HPO artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(config_file)
        validate_hpo_config(config)
        cached, cache_meta = read_model_frame_cache(cache_file)
        frame, isolation = isolate_rolling_source(cached, maximum_outcome_year=2023)
        del cached
        scope = resolve_hpo_scope(
            root=root, all_feature_columns=tuple(cache_meta["feature_columns"]), config=config
        )
        selection_predictions, selection_models, selection_importance = _fit_stage(
            frame=frame,
            folds=config["selection_folds"],
            profiles=config["profiles"],
            scope=scope,
            model_kind=config["model_kind"],
            stage="selection",
            model_root=temporary / "models",
        )
        selection_races = rolling_race_metric_table(selection_predictions)
        comparisons = _comparisons(config["profiles"])
        selection_summary = summarize_year_macro(selection_races, comparisons)
        uncertainty = config["uncertainty"]
        selection_bootstrap = paired_year_stratified_block_bootstrap(
            selection_races,
            comparisons=comparisons,
            n_resamples=int(uncertainty["bootstrap_resamples"]),
            confidence_level=float(uncertainty["confidence_level"]),
            seed=int(uncertainty["bootstrap_seed"]),
            block_length_dates=int(uncertainty["block_length_dates"]),
        )
        selected = select_hpo_profile(selection_summary, config)
        confirmation_profiles = [config["profiles"][0]]
        if selected["selected_profile"] is not None:
            confirmation_profiles.append(
                next(
                    profile
                    for profile in config["profiles"]
                    if profile["id"] == selected["selected_profile"]
                )
            )
        confirmation_predictions, confirmation_models, confirmation_importance = _fit_stage(
            frame=frame,
            folds=[config["confirmation_fold"]],
            profiles=confirmation_profiles,
            scope=scope,
            model_kind=config["model_kind"],
            stage="confirmation",
            model_root=temporary / "models",
        )
        confirmation_races = rolling_race_metric_table(confirmation_predictions)
        confirmation: dict[str, Any] = {"decision": "no_change", "selected_profile": None}
        confirmation_summary: dict[str, Any] = {}
        confirmation_bootstrap: dict[str, Any] = {}
        if selected["selected_profile"] is not None:
            comparison = {
                "id": "selected_vs_control",
                "candidate": selected["selected_profile"],
                "reference": "control",
                "type": "hypothesis",
            }
            confirmation_summary = summarize_year_macro(confirmation_races, [comparison])
            confirmation_bootstrap = paired_year_stratified_block_bootstrap(
                confirmation_races,
                comparisons=[comparison],
                n_resamples=int(uncertainty["bootstrap_resamples"]),
                confidence_level=float(uncertainty["confidence_level"]),
                seed=int(uncertainty["bootstrap_seed"]),
                block_length_dates=int(uncertainty["block_length_dates"]),
            )
            improvements = confirmation_summary["comparisons"]["selected_vs_control"]["metrics"]
            point = {metric: float(payload["year_macro_improvement"]) for metric, payload in improvements.items()}
            decision = confirmation_decision(
                improvement=point,
                interval=confirmation_bootstrap["paired"]["selected_vs_control"],
                config=config,
            )
            confirmation = {
                **decision,
                "selected_profile": selected["selected_profile"],
                "point_improvement": point,
            }

        all_predictions = pd.concat([selection_predictions, confirmation_predictions], ignore_index=True)
        forbidden = [
            column
            for column in all_predictions.columns
            if any(token in column.lower() for token in ("odds", "popularity", "人気", "オッズ"))
        ]
        if forbidden:
            raise AssertionError(f"market columns reached HPO predictions: {forbidden}")
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "hypothesis": config["hypothesis"],
            "model_kind": config["model_kind"],
            "scope": {
                "selection_evaluation_years": [2020, 2021, 2022],
                "confirmation_evaluation_year": 2023,
                "nonselected_profiles_scored_on_2023": 0,
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
            },
            "data": {
                "fingerprint": cache_meta["data_fingerprint"],
                "cache_sha256": sha256_file(cache_file),
                **isolation,
            },
            "selection": {
                "model_records": selection_models,
                "year_summary": selection_summary,
                "paired_block_bootstrap": selection_bootstrap,
                "selection_result": selected,
            },
            "confirmation": {
                "model_records": confirmation_models,
                "year_summary": confirmation_summary,
                "paired_block_bootstrap": confirmation_bootstrap,
                "confirmation_result": confirmation,
            },
            "selection_accounting": config["selection_accounting"],
            "limitations": [
                "2020-2023 have prior project exposure and are not untouched holdouts.",
                "The single deterministic seed does not measure refit variance.",
                "Selection intervals are descriptive after eleven profile comparisons.",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "config.json", config)
        write_json(
            temporary / "resolved_configs.json",
            {
                "schema_version": 1,
                "feature_config_path": str(scope["feature_config_path"]),
                "feature_config_hash": canonical_json_hash(scope["feature_config"]),
                "feature_config": scope["feature_config"],
                "base_model_config_path": str(scope["model_config_path"]),
                "base_model_config_hash": canonical_json_hash(scope["model_config"]),
                "base_model_config": scope["model_config"],
                "profiles": config["profiles"],
            },
        )
        write_json(
            temporary / "feature_schema.json",
            {
                "schema_version": 1,
                "feature_count": len(scope["feature_columns"]),
                "feature_columns": list(scope["feature_columns"]),
                "feature_columns_sha256": scope["feature_columns_sha256"],
                "feature_groups": {
                    name: list(columns) for name, columns in scope["feature_groups"].items()
                },
                "resolution": scope["feature_resolution"],
            },
        )
        write_json(
            temporary / "run_meta.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "seed": config["seed"],
                "config_hash": canonical_json_hash(config),
                "config_sha256": sha256_file(config_file),
                "cache_sha256": metrics["data"]["cache_sha256"],
                "git": git_state(root),
                "software": _software_versions(),
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
            },
        )
        all_predictions.to_csv(temporary / "predictions_scoring.csv.gz", index=False, compression="gzip")
        pd.concat(
            [selection_races.assign(stage="selection"), confirmation_races.assign(stage="confirmation")],
            ignore_index=True,
        ).to_csv(temporary / "race_metrics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(selection_importance + confirmation_importance).to_csv(
            temporary / "feature_importance.csv", index=False
        )
        write_artifact_manifest(temporary)
        temporary.replace(output)
        return metrics
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
