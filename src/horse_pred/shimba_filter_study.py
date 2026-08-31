"""SHIMBA-FILTER-001: exclude new-horse races from Binary gradient fitting only."""

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
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.rolling_evaluation import (
    assign_fold_roles,
    isolate_rolling_source,
    paired_year_stratified_block_bootstrap,
    resolve_rolling_methods,
    rolling_race_metric_table,
    summarize_year_macro,
    validate_rolling_config,
)

CONTROL_METHOD = "binary_control"
CANDIDATE_METHOD = "binary_candidate"
COMPARISON_ID = "binary_shimba_filter_vs_control"
PV01_FEATURE_COUNT = 254
PV01_FEATURE_COLUMNS_SHA256 = (
    "e5228bb4ffd605888b7266030d5e1e9f0931e8468b6fbbf124f3cea60905e51d"
)
_SCORING_ROLES = ("calibration", "evaluation")
_EXPECTED_FIT_POPULATION = {
    "control": "all_train_races",
    "candidate": "exclude_new_horse_races_from_train_only",
    "new_horse_definition": {
        "race_class_contains_after_whitespace_normalization": "新馬",
        "validated_context_column": "context__class_tier",
        "validated_context_value": 0.0,
    },
    "early_stopping_population": "all_races",
    "calibration_population": "all_races",
    "evaluation_population": "all_races",
    "pit_cache_and_state": "unchanged",
}
_EXPECTED_DECISION_RULE = {
    "primary_metric": "race_log_loss",
    "primary_ci_lower_strictly_greater_than": 0.0,
    "minimum_primary_improved_years": 3,
    "guardrail_minimum_improvements": {
        "race_brier": -0.001,
        "ndcg_at_3": -0.002,
        "top_1_winner_mass": -0.005,
    },
}


def validate_shimba_filter_config(config: dict[str, Any]) -> None:
    """Validate the exact one-comparison SHIMBA-FILTER-001 protocol."""

    required = {
        "schema_version",
        "experiment_id",
        "hypothesis",
        "seed",
        "maximum_outcome_year",
        "folds",
        "methods",
        "comparisons",
        "selection_accounting",
        "uncertainty",
        "fit_population",
        "decision_rule",
    }
    if set(config) != required:
        raise ValueError(
            "SHIMBA-FILTER-001 config keys differ from preregistration: "
            f"{sorted(set(config) ^ required)}"
        )
    validate_rolling_config(config)
    if config["experiment_id"] != "shimba_filter_001_rolling":
        raise ValueError("SHIMBA-FILTER-001 experiment_id is frozen")
    if config["seed"] != 42:
        raise ValueError("SHIMBA-FILTER-001 seed must remain 42")
    if set(config["methods"]) != {CONTROL_METHOD, CANDIDATE_METHOD}:
        raise ValueError("SHIMBA-FILTER-001 requires exactly control and candidate Binary methods")
    control = config["methods"][CONTROL_METHOD]
    candidate = config["methods"][CANDIDATE_METHOD]
    if control != candidate:
        raise ValueError("control and candidate must use identical model and feature configs")
    if control["model_kind"] != "binary":
        raise ValueError("SHIMBA-FILTER-001 is Binary-only")
    if (
        control["expected_feature_count"] != PV01_FEATURE_COUNT
        or control["expected_columns_sha256"] != PV01_FEATURE_COLUMNS_SHA256
    ):
        raise ValueError("SHIMBA-FILTER-001 must use the frozen 254-column PV-01 scope")
    expected_comparison = [
        {
            "id": COMPARISON_ID,
            "candidate": CANDIDATE_METHOD,
            "reference": CONTROL_METHOD,
            "type": "hypothesis",
        }
    ]
    if config["comparisons"] != expected_comparison:
        raise ValueError("SHIMBA-FILTER-001 requires exactly one frozen comparison")
    if config["selection_accounting"]["candidate_comparisons_this_run"] != 1:
        raise ValueError("SHIMBA-FILTER-001 comparison count must remain one")
    if config["fit_population"] != _EXPECTED_FIT_POPULATION:
        raise ValueError("SHIMBA-FILTER-001 fit-population contract differs from preregistration")
    if config["decision_rule"] != _EXPECTED_DECISION_RULE:
        raise ValueError("SHIMBA-FILTER-001 decision rule differs from preregistration")


def new_horse_race_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the race-level new-horse mask after validating the audited proxy."""

    required = {"race_id", "race_class", "context__class_tier"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SHIMBA-FILTER-001 cache is missing: {missing}")
    normalized = (
        frame["race_class"]
        .astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    class_mask = normalized.str.contains("新馬", regex=False, na=False)
    tier = pd.to_numeric(frame["context__class_tier"], errors="coerce")
    tier_mask = tier.eq(0.0)
    mismatch = int(class_mask.ne(tier_mask).sum())
    if mismatch:
        raise ValueError(
            "race_class new-horse definition disagrees with context__class_tier==0 "
            f"for {mismatch} rows"
        )
    audit = pd.DataFrame(
        {
            "race_id": frame["race_id"].astype(str),
            "normalized_race_class": normalized,
            "is_new_horse_race": class_mask,
        }
    )
    if audit.groupby("race_id", observed=True)["normalized_race_class"].nunique(dropna=False).gt(1).any():
        raise ValueError("race_class is not constant within a race")
    if audit.groupby("race_id", observed=True)["is_new_horse_race"].nunique().gt(1).any():
        raise ValueError("new-horse classification is not constant within a race")
    return class_mask.astype(bool)


def prepare_gradient_fit_frame(
    fold_frame: pd.DataFrame, *, exclude_new_horse_races: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare model-fit rows while leaving non-training roles untouched."""

    required = {"race_id", "rolling_role", "is_new_horse_race"}
    missing = sorted(required.difference(fold_frame.columns))
    if missing:
        raise ValueError(f"fit frame is missing: {missing}")
    race_labels = fold_frame.groupby("race_id", observed=True)["is_new_horse_race"].nunique()
    if race_labels.gt(1).any():
        raise ValueError("new-horse filter would split a race")
    train = fold_frame["rolling_role"].eq("train")
    excluded = (
        train & fold_frame["is_new_horse_race"].astype(bool)
        if exclude_new_horse_races
        else pd.Series(False, index=fold_frame.index)
    )
    fit_frame = fold_frame.loc[~excluded].copy()
    train_before = fold_frame.loc[train]
    train_after = fit_frame.loc[fit_frame["rolling_role"].eq("train")]
    excluded_rows = fold_frame.loc[excluded]
    if exclude_new_horse_races and not excluded_rows.empty:
        expected_races = set(
            fold_frame.loc[train & fold_frame["is_new_horse_race"].astype(bool), "race_id"].astype(str)
        )
        if set(excluded_rows["race_id"].astype(str)) != expected_races:
            raise AssertionError("candidate did not exclude exactly the new-horse training races")
    for role in ("model_validation", "calibration", "evaluation"):
        if int(fit_frame["rolling_role"].eq(role).sum()) != int(fold_frame["rolling_role"].eq(role).sum()):
            raise AssertionError(f"SHIMBA filter changed the {role} population")
    report = {
        "filter_applied": bool(exclude_new_horse_races),
        "train_rows_before": int(len(train_before)),
        "train_races_before": int(train_before["race_id"].nunique()),
        "excluded_fit_rows": int(len(excluded_rows)),
        "excluded_fit_races": int(excluded_rows["race_id"].nunique()),
        "train_rows_after": int(len(train_after)),
        "train_races_after": int(train_after["race_id"].nunique()),
        "model_validation_rows": int(fit_frame["rolling_role"].eq("model_validation").sum()),
        "model_validation_races": int(
            fit_frame.loc[fit_frame["rolling_role"].eq("model_validation"), "race_id"].nunique()
        ),
        "calibration_rows": int(fit_frame["rolling_role"].eq("calibration").sum()),
        "calibration_races": int(
            fit_frame.loc[fit_frame["rolling_role"].eq("calibration"), "race_id"].nunique()
        ),
        "evaluation_rows": int(fit_frame["rolling_role"].eq("evaluation").sum()),
        "evaluation_races": int(
            fit_frame.loc[fit_frame["rolling_role"].eq("evaluation"), "race_id"].nunique()
        ),
    }
    if report["train_rows_before"] - report["excluded_fit_rows"] != report["train_rows_after"]:
        raise AssertionError("SHIMBA fit-row accounting does not balance")
    return fit_frame, report


def shimba_filter_decision(
    year_summary: dict[str, Any],
    paired_bootstrap: dict[str, Any],
    decision_rule: dict[str, Any],
) -> dict[str, Any]:
    """Apply the all-race preregistered acceptance and guardrail rule."""

    metrics = year_summary["comparisons"][COMPARISON_ID]["metrics"]
    interval = paired_bootstrap["paired"][COMPARISON_ID]["race_log_loss"]
    primary_interval_passed = float(interval["lower"]) > float(
        decision_rule["primary_ci_lower_strictly_greater_than"]
    )
    primary_direction_passed = int(metrics["race_log_loss"]["improved_years"]) >= int(
        decision_rule["minimum_primary_improved_years"]
    )
    guardrails = {
        metric: {
            "point_improvement": float(metrics[metric]["year_macro_improvement"]),
            "minimum": float(minimum),
            "passed": float(metrics[metric]["year_macro_improvement"]) >= float(minimum),
        }
        for metric, minimum in decision_rule["guardrail_minimum_improvements"].items()
    }
    guardrails_passed = all(payload["passed"] for payload in guardrails.values())
    accepted = primary_interval_passed and primary_direction_passed and guardrails_passed
    if accepted:
        decision = "accept"
    elif not guardrails_passed or float(interval["upper"]) < 0.0:
        decision = "reject"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "primary_population": "all_evaluation_races",
        "primary_metric": "race_log_loss",
        "primary_interval": interval,
        "primary_interval_passed": primary_interval_passed,
        "primary_improved_years": int(metrics["race_log_loss"]["improved_years"]),
        "primary_direction_passed": primary_direction_passed,
        "guardrails": guardrails,
        "guardrails_passed": guardrails_passed,
        "slice_diagnostics_used_for_decision": False,
    }


def _primary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "ndcg_at_3": float(payload["ranking"]["ndcg_at_3"]),
        "top_1_winner_mass": float(payload["ranking"]["top_1"]),
        "race_log_loss": float(payload["probability"]["race_log_loss"]),
        "race_brier": float(payload["probability"]["race_brier"]),
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None and int(value) > 0 else None


def _software_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed-as-package"
    return result


def _slice_diagnostics(
    race_metrics: pd.DataFrame, comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for race_slice in ("new_horse", "non_new_horse"):
        selected = race_metrics.loc[race_metrics["race_slice"].eq(race_slice)].copy()
        coverage = {
            "race_count": int(selected["race_id"].nunique()),
            "race_count_by_year": {
                str(int(year)): int(rows["race_id"].nunique())
                for year, rows in selected.groupby("evaluation_year", observed=True)
            },
        }
        expected_years = {2020, 2021, 2022, 2023}
        methods = {CONTROL_METHOD, CANDIDATE_METHOD}
        complete = set(selected["evaluation_year"].unique()) == expected_years and set(
            selected["method"].astype(str).unique()
        ) == methods
        diagnostics[race_slice] = {
            **coverage,
            "year_summary": summarize_year_macro(selected, comparisons) if complete else None,
        }
    return diagnostics


def run_shimba_filter_study(
    *, repo_root: str | Path, cache_path: str | Path, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run the four-fold Binary-only SHIMBA-FILTER-001 ablation."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    cache_file = Path(cache_path)
    cache_file = cache_file if cache_file.is_absolute() else root / cache_file
    config_file = Path(config_path)
    config_file = config_file if config_file.is_absolute() else root / config_file
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SHIMBA artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(config_file)
        validate_shimba_filter_config(config)
        cached, cache_meta = read_model_frame_cache(cache_file)
        frame, isolation = isolate_rolling_source(
            cached, maximum_outcome_year=int(config["maximum_outcome_year"])
        )
        del cached
        frame["is_new_horse_race"] = new_horse_race_mask(frame)
        methods = resolve_rolling_methods(
            root=root,
            all_feature_columns=tuple(cache_meta["feature_columns"]),
            config=config,
        )
        if methods[CONTROL_METHOD]["feature_columns"] != methods[CANDIDATE_METHOD]["feature_columns"]:
            raise AssertionError("resolved SHIMBA feature columns differ between methods")

        prediction_frames: list[pd.DataFrame] = []
        importance_rows: list[dict[str, Any]] = []
        model_records: dict[str, Any] = {}
        for fold in config["folds"]:
            roles = assign_fold_roles(frame, fold)
            eligible = roles.notna()
            fold_frame = frame.loc[eligible].copy()
            fold_frame["rolling_role"] = roles.loc[eligible].to_numpy()
            fold_records: dict[str, Any] = {}
            for method_id in (CONTROL_METHOD, CANDIDATE_METHOD):
                method = methods[method_id]
                fit_frame, fit_report = prepare_gradient_fit_frame(
                    fold_frame,
                    exclude_new_horse_races=method_id == CANDIDATE_METHOD,
                )
                feature_columns = method["feature_columns"]
                model_config = method["model_config"]
                model = train_binary(
                    frame=fit_frame,
                    feature_columns=feature_columns,
                    train_split="train",
                    model_validation_split="model_validation",
                    race_id_column="race_id",
                    finish_position_column="model_finish_position",
                    split_column="rolling_role",
                    params=model_config["parameters"],
                    early_stopping_rounds=model_config.get("early_stopping_rounds"),
                )
                scoring = fold_frame.loc[fold_frame["rolling_role"].isin(_SCORING_ROLES)].copy()
                raw = predict(
                    model,
                    scoring,
                    feature_columns=feature_columns,
                    model_kind="binary",
                )
                scoring["raw_output"] = raw
                scoring["utility"] = probability_logits(raw, epsilon=PROBABILITY_EPSILON)
                calibration = scoring.loc[scoring["rolling_role"].eq("calibration")]
                calibrator = fit_temperature(
                    calibration["utility"],
                    calibration["race_id"],
                    calibration["model_finish_position"],
                )
                scoring["probability_t1"] = race_softmax(scoring["utility"], scoring["race_id"])
                scoring["probability_calibrated"] = apply_temperature(
                    calibrator, scoring["utility"], scoring["race_id"]
                )
                scoring["fold_id"] = fold["id"]
                scoring["role"] = scoring["rolling_role"]
                scoring["evaluation_year"] = int(fold["evaluation_year"])
                scoring["method"] = method_id
                scoring["model_kind"] = "binary"
                scoring["race_slice"] = np.where(
                    scoring["is_new_horse_race"], "new_horse", "non_new_horse"
                )
                prediction_columns = [
                    "fold_id",
                    "role",
                    "evaluation_year",
                    "race_id",
                    "race_date",
                    "race_slice",
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
                        prediction_columns.insert(9, optional)
                prediction_frames.append(scoring.loc[:, prediction_columns])

                evaluation = scoring.loc[scoring["rolling_role"].eq("evaluation")]
                evaluated = evaluate_predictions(
                    evaluation["probability_calibrated"],
                    pd.to_numeric(
                        evaluation["model_finish_position"], errors="raise"
                    ).astype(int),
                    evaluation["race_id"],
                    ranking_scores=evaluation["utility"],
                )
                model_dir = temporary / "models" / str(fold["id"])
                model_dir.mkdir(parents=True, exist_ok=True)
                model.booster_.save_model(str(model_dir / f"{method_id}.txt"))
                gains = model.booster_.feature_importance(importance_type="gain")
                splits = model.booster_.feature_importance(importance_type="split")
                gain_total = float(np.sum(gains))
                for feature, gain, split in zip(feature_columns, gains, splits):
                    importance_rows.append(
                        {
                            "fold_id": fold["id"],
                            "evaluation_year": fold["evaluation_year"],
                            "method": method_id,
                            "feature": feature,
                            "importance_gain": float(gain),
                            "importance_gain_fraction": float(gain) / gain_total if gain_total else 0.0,
                            "importance_split": int(split),
                        }
                    )
                parameters = (
                    model.get_params(deep=False)
                    if hasattr(model, "get_params")
                    else model_config["parameters"]
                )
                fold_records[method_id] = {
                    "best_iteration": _best_iteration(model),
                    "temperature": float(calibrator.temperature),
                    "effective_parameters": parameters,
                    "feature_count": len(feature_columns),
                    "feature_columns_sha256": method["feature_columns_sha256"],
                    "fit_population": fit_report,
                    "primary_metrics": _primary_metrics(evaluated),
                    "evaluation": evaluated,
                }
            model_records[str(fold["id"])] = fold_records

        predictions = pd.concat(prediction_frames, ignore_index=True)
        forbidden = [
            column
            for column in predictions.columns
            if any(token in column.lower() for token in ("odds", "popularity", "人気", "オッズ"))
        ]
        if forbidden:
            raise AssertionError(f"market columns reached SHIMBA predictions: {forbidden}")
        race_metrics = rolling_race_metric_table(predictions)
        slice_map = predictions.loc[
            predictions["role"].eq("evaluation"),
            ["fold_id", "evaluation_year", "race_id", "race_slice"],
        ].drop_duplicates()
        if slice_map.groupby(["fold_id", "evaluation_year", "race_id"], observed=True)[
            "race_slice"
        ].nunique().gt(1).any():
            raise AssertionError("race slice changed within an evaluation race")
        race_metrics = race_metrics.merge(
            slice_map,
            on=["fold_id", "evaluation_year", "race_id"],
            how="left",
            validate="many_to_one",
        )
        year_summary = summarize_year_macro(race_metrics, config["comparisons"])
        uncertainty = config["uncertainty"]
        bootstrap = paired_year_stratified_block_bootstrap(
            race_metrics,
            comparisons=config["comparisons"],
            n_resamples=int(uncertainty["bootstrap_resamples"]),
            confidence_level=float(uncertainty["confidence_level"]),
            seed=int(uncertainty["bootstrap_seed"]),
            block_length_dates=int(uncertainty["block_length_dates"]),
        )
        decision = shimba_filter_decision(year_summary, bootstrap, config["decision_rule"])
        slice_diagnostics = _slice_diagnostics(race_metrics, config["comparisons"])
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "hypothesis": config["hypothesis"],
            "scope": {
                "evaluation_years": [2020, 2021, 2022, 2023],
                "maximum_outcome_year": 2023,
                "model_kind": "binary",
                "feature_count": PV01_FEATURE_COUNT,
                "feature_columns_sha256": PV01_FEATURE_COLUMNS_SHA256,
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "pit_cache_and_state_changed": False,
            },
            "data": {
                "fingerprint": cache_meta["data_fingerprint"],
                "cache_sha256": sha256_file(cache_file),
                **isolation,
            },
            "methods": model_records,
            "year_summary": year_summary,
            "paired_block_bootstrap": bootstrap,
            "decision": decision,
            "slice_diagnostics": slice_diagnostics,
            "selection_accounting": config["selection_accounting"],
            "limitations": [
                "2020-2023 have prior research exposure and are screening folds, not untouched holdouts.",
                "New-horse and non-new-horse slices are descriptive and cannot replace the all-race primary.",
                "The ablation changes gradient-fit membership only; cached PIT state still "
                "includes prior new-horse outcomes.",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "config.json", config)
        write_json(
            temporary / "feature_schema.json",
            {
                "schema_version": 1,
                "feature_count": PV01_FEATURE_COUNT,
                "feature_columns": list(methods[CONTROL_METHOD]["feature_columns"]),
                "feature_columns_sha256": methods[CONTROL_METHOD]["feature_columns_sha256"],
                "feature_groups": {
                    name: list(columns)
                    for name, columns in methods[CONTROL_METHOD]["feature_groups"].items()
                },
                "resolution": methods[CONTROL_METHOD]["feature_resolution"],
            },
        )
        write_json(
            temporary / "resolved_configs.json",
            {
                "schema_version": 1,
                "feature_config_path": str(methods[CONTROL_METHOD]["feature_config_path"]),
                "feature_config_hash": canonical_json_hash(
                    methods[CONTROL_METHOD]["feature_config"]
                ),
                "feature_config": methods[CONTROL_METHOD]["feature_config"],
                "model_config_path": str(methods[CONTROL_METHOD]["model_config_path"]),
                "model_config_hash": canonical_json_hash(methods[CONTROL_METHOD]["model_config"]),
                "model_config": methods[CONTROL_METHOD]["model_config"],
                "fit_population": config["fit_population"],
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
                "data_fingerprint": cache_meta["data_fingerprint"],
                "cache_sha256": metrics["data"]["cache_sha256"],
                "git": git_state(root),
                "software": _software_versions(),
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "candidate_comparisons_this_run": 1,
            },
        )
        predictions.to_csv(
            temporary / "predictions_scoring.csv.gz", index=False, compression="gzip"
        )
        race_metrics.to_csv(
            temporary / "race_metrics.csv.gz", index=False, compression="gzip"
        )
        race_metrics.to_csv(
            temporary / "slice_race_metrics.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(importance_rows).to_csv(temporary / "feature_importance.csv", index=False)
        write_artifact_manifest(temporary)
        temporary.replace(output)
        return metrics
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
