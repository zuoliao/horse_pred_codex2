"""Reusable rolling-origin screening for the no-odds LightGBM models.

The initial contract evaluates calendar years 2020--2023.  Every fold keeps
model fitting, early stopping, race-softmax temperature fitting, and evaluation
strictly chronological.  Rows from 2024 onward are removed immediately after
cache load, before feature resolution or any model operation.
"""

from __future__ import annotations

import math
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
    predict,
    probability_logits,
    race_softmax,
    train_binary,
    train_ranker,
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.uncertainty import HIGHER_IS_BETTER, PRIMARY_METRICS

_EXPECTED_EVALUATION_YEARS = (2020, 2021, 2022, 2023)
_SCORING_ROLES = ("calibration", "evaluation")
_METHOD_KINDS = ("binary", "lambdarank")


def validate_rolling_config(config: dict[str, Any]) -> None:
    """Validate the fixed first-generation EVAL-ROLL protocol."""

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
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"rolling config is missing: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("rolling config schema_version must be 1")
    if not str(config["experiment_id"]).strip() or not str(config["hypothesis"]).strip():
        raise ValueError("experiment_id and hypothesis must be non-empty")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if config["maximum_outcome_year"] != 2023:
        raise ValueError("initial EVAL-ROLL maximum_outcome_year must remain 2023")

    folds = config["folds"]
    if not isinstance(folds, list) or len(folds) != len(_EXPECTED_EVALUATION_YEARS):
        raise ValueError("rolling config must contain exactly four folds")
    fold_ids: set[str] = set()
    evaluation_years: list[int] = []
    for index, fold in enumerate(folds):
        expected_keys = {
            "id",
            "train_start_year",
            "train_end_year",
            "early_stopping_year",
            "calibration_year",
            "evaluation_year",
        }
        if not isinstance(fold, dict) or set(fold) != expected_keys:
            raise ValueError("each rolling fold must use the exact registered keys")
        fold_id = str(fold["id"])
        if not fold_id or fold_id in fold_ids:
            raise ValueError("rolling fold IDs must be unique and non-empty")
        fold_ids.add(fold_id)
        evaluation_year = int(fold["evaluation_year"])
        expected_evaluation = _EXPECTED_EVALUATION_YEARS[index]
        if evaluation_year != expected_evaluation:
            raise ValueError("rolling evaluation years must be ordered 2020 through 2023")
        if int(fold["train_start_year"]) != 2014:
            raise ValueError("rolling train_start_year must remain 2014")
        if int(fold["train_end_year"]) != evaluation_year - 3:
            raise ValueError("train_end_year must equal evaluation_year - 3")
        if int(fold["early_stopping_year"]) != evaluation_year - 2:
            raise ValueError("early_stopping_year must equal evaluation_year - 2")
        if int(fold["calibration_year"]) != evaluation_year - 1:
            raise ValueError("calibration_year must equal evaluation_year - 1")
        evaluation_years.append(evaluation_year)

    methods = config["methods"]
    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods must be a non-empty object")
    for method_id, method in methods.items():
        expected_keys = {
            "model_kind",
            "model_config",
            "feature_config",
            "expected_feature_count",
            "expected_columns_sha256",
        }
        if not str(method_id).strip() or not isinstance(method, dict) or set(method) != expected_keys:
            raise ValueError("each method must use the exact registered keys")
        if method["model_kind"] not in _METHOD_KINDS:
            raise ValueError("model_kind must be binary or lambdarank")
        if not isinstance(method["model_config"], str) or not method["model_config"]:
            raise ValueError("model_config must be a non-empty path")
        if not isinstance(method["feature_config"], str) or not method["feature_config"]:
            raise ValueError("feature_config must be a non-empty path")
        if not isinstance(method["expected_feature_count"], int) or method["expected_feature_count"] < 1:
            raise ValueError("expected_feature_count must be positive")
        checksum = method["expected_columns_sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("expected_columns_sha256 must be a SHA-256 hex digest")
        try:
            int(checksum, 16)
        except ValueError as exc:
            raise ValueError("expected_columns_sha256 must be hexadecimal") from exc

    comparisons = config["comparisons"]
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list")
    comparison_ids: set[str] = set()
    for comparison in comparisons:
        expected_keys = {"id", "candidate", "reference", "type"}
        if not isinstance(comparison, dict) or set(comparison) != expected_keys:
            raise ValueError("each comparison must use the exact registered keys")
        comparison_id = str(comparison["id"])
        if not comparison_id or comparison_id in comparison_ids:
            raise ValueError("comparison IDs must be unique and non-empty")
        comparison_ids.add(comparison_id)
        for side in ("candidate", "reference"):
            if comparison[side] not in methods:
                raise ValueError(f"comparison references unknown method: {comparison[side]}")
        if comparison["type"] not in {"hypothesis", "descriptive_cross_family"}:
            raise ValueError("comparison type must be hypothesis or descriptive_cross_family")
        candidate_kind = methods[comparison["candidate"]]["model_kind"]
        reference_kind = methods[comparison["reference"]]["model_kind"]
        if comparison["type"] == "hypothesis" and candidate_kind != reference_kind:
            raise ValueError("a hypothesis comparison must stay within one model family")

    accounting = config["selection_accounting"]
    expected_accounting = {
        "candidate_comparisons_this_run",
        "prior_selection_uses_by_evaluation_year",
        "multiple_comparison_note",
    }
    if not isinstance(accounting, dict) or set(accounting) != expected_accounting:
        raise ValueError("selection_accounting must use the exact registered keys")
    if (
        not isinstance(accounting["candidate_comparisons_this_run"], int)
        or accounting["candidate_comparisons_this_run"] < 0
    ):
        raise ValueError("candidate_comparisons_this_run must be non-negative")
    prior = accounting["prior_selection_uses_by_evaluation_year"]
    if not isinstance(prior, dict) or set(prior) != {str(year) for year in evaluation_years}:
        raise ValueError("prior selection-use counts must cover every evaluation year")
    if any(not isinstance(value, int) or value < 0 for value in prior.values()):
        raise ValueError("prior selection-use counts must be non-negative integers")
    if not str(accounting["multiple_comparison_note"]).strip():
        raise ValueError("multiple_comparison_note must be non-empty")

    uncertainty = config["uncertainty"]
    expected_uncertainty = {
        "bootstrap_resamples",
        "bootstrap_seed",
        "block_length_dates",
        "confidence_level",
    }
    if not isinstance(uncertainty, dict) or set(uncertainty) != expected_uncertainty:
        raise ValueError("uncertainty must use the exact registered keys")
    if int(uncertainty["bootstrap_resamples"]) < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    if int(uncertainty["block_length_dates"]) < 1:
        raise ValueError("block_length_dates must be positive")
    if not 0.0 < float(uncertainty["confidence_level"]) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")


def isolate_rolling_source(
    frame: pd.DataFrame, *, maximum_outcome_year: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop 2024+ before feature resolution, fitting, or scoring."""

    required = {"race_id", "race_date", "model_finish_position"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rolling cache is missing metadata: {missing}")
    if maximum_outcome_year > 2023:
        raise ValueError("rolling screening must not open 2024 or later outcomes")
    dates = pd.to_datetime(frame["race_date"], errors="raise").dt.normalize()
    if frame.assign(_date=dates).groupby("race_id", observed=True)["_date"].nunique().gt(1).any():
        raise ValueError("a race crosses calendar dates")
    years = dates.dt.year
    selected_mask = years.between(2014, maximum_outcome_year, inclusive="both")
    selected = frame.loc[selected_mask].copy()
    selected["race_date"] = dates.loc[selected_mask]
    selected["_source_order"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.sort_values(
        ["race_date", "race_id", "_source_order"], kind="stable"
    ).drop(columns="_source_order").reset_index(drop=True)
    if selected.empty:
        raise ValueError("rolling source has no eligible rows")
    if selected["race_date"].dt.year.max() > maximum_outcome_year:
        raise AssertionError("post-firewall outcomes survived isolation")
    excluded = years.loc[~selected_mask].value_counts().sort_index()
    return selected, {
        "cache_rows": int(len(frame)),
        "rolling_rows": int(len(selected)),
        "rows_excluded_by_year": {str(int(year)): int(count) for year, count in excluded.items()},
        "rows_used_2024": 0,
        "rows_used_2025": 0,
        "maximum_outcome_year": maximum_outcome_year,
    }


def assign_fold_roles(frame: pd.DataFrame, fold: dict[str, Any]) -> pd.Series:
    """Return the four chronological roles for one explicit fold."""

    years = pd.to_datetime(frame["race_date"], errors="raise").dt.year
    roles = pd.Series(pd.NA, index=frame.index, dtype="string")
    roles.loc[years.between(int(fold["train_start_year"]), int(fold["train_end_year"]))] = "train"
    roles.loc[years.eq(int(fold["early_stopping_year"]))] = "model_validation"
    roles.loc[years.eq(int(fold["calibration_year"]))] = "calibration"
    roles.loc[years.eq(int(fold["evaluation_year"]))] = "evaluation"
    eligible = years.between(int(fold["train_start_year"]), int(fold["evaluation_year"]))
    if roles.loc[eligible].isna().any():
        raise ValueError(f"fold {fold['id']} leaves an eligible year unassigned")
    if frame.loc[eligible].assign(_role=roles.loc[eligible]).groupby(
        "race_id", observed=True
    )["_role"].nunique().gt(1).any():
        raise ValueError(f"fold {fold['id']} splits a race across roles")
    counts = roles.loc[eligible].value_counts()
    missing = [role for role in ("train", "model_validation", *_SCORING_ROLES) if counts.get(role, 0) == 0]
    if missing:
        raise ValueError(f"fold {fold['id']} has empty roles: {missing}")
    return roles


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def resolve_rolling_methods(
    *,
    root: Path,
    all_feature_columns: tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve include-only method feature scopes and fixed model configs."""

    resolved: dict[str, dict[str, Any]] = {}
    for method_id, method in config["methods"].items():
        feature_path = _resolve(root, method["feature_config"])
        model_path = _resolve(root, method["model_config"])
        feature_config = load_json(feature_path)
        validate_cached_experiment_config(feature_config)
        if set(feature_config["feature_selection"]) != {"include"}:
            raise ValueError(f"rolling method {method_id} must use include-only feature selection")
        if feature_config.get("derived_features"):
            raise ValueError(f"rolling method {method_id} must use precomputed PIT features")
        columns, groups, resolution = resolve_semantic_feature_selection(
            all_feature_columns, feature_config
        )
        checksum = feature_columns_checksum(columns)
        if len(columns) != method["expected_feature_count"]:
            raise ValueError(f"rolling method {method_id} feature count disagrees with preregistration")
        if checksum != method["expected_columns_sha256"]:
            raise ValueError(f"rolling method {method_id} feature hash disagrees with preregistration")

        model_config = load_json(model_path)
        expected_family = {
            "binary": ("lightgbm_binary", "binary"),
            "lambdarank": ("lightgbm_lambdarank", "lambdarank"),
        }[method["model_kind"]]
        if model_config.get("model_family") != expected_family[0]:
            raise ValueError(f"rolling method {method_id} has the wrong model family")
        if model_config.get("parameters", {}).get("objective") != expected_family[1]:
            raise ValueError(f"rolling method {method_id} has the wrong objective")
        if int(model_config.get("seed", -1)) != config["seed"]:
            raise ValueError(f"rolling method {method_id} seed differs from experiment seed")
        if int(model_config.get("parameters", {}).get("random_state", -1)) != config["seed"]:
            raise ValueError(f"rolling method {method_id} random_state differs from experiment seed")
        resolved[method_id] = {
            "model_kind": method["model_kind"],
            "feature_columns": columns,
            "feature_groups": groups,
            "feature_resolution": resolution,
            "feature_config": feature_config,
            "feature_config_path": feature_path,
            "model_config": model_config,
            "model_config_path": model_path,
            "feature_columns_sha256": checksum,
        }
    return resolved


def _primary_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "ndcg_at_3": float(payload["ranking"]["ndcg_at_3"]),
        "top_1_winner_mass": float(payload["ranking"]["top_1"]),
        "race_log_loss": float(payload["probability"]["race_log_loss"]),
        "race_brier": float(payload["probability"]["race_brier"]),
    }


def rolling_race_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return additive per-race metrics from evaluation-role predictions."""

    required = {
        "fold_id",
        "role",
        "evaluation_year",
        "race_id",
        "race_date",
        "method",
        "model_finish_position",
        "utility",
        "probability_calibrated",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"rolling predictions are missing: {missing}")
    selected = predictions.loc[predictions["role"].eq("evaluation")].copy()
    if selected.empty:
        raise ValueError("rolling predictions contain no evaluation rows")
    rows: list[dict[str, Any]] = []
    for (fold_id, evaluation_year, method, race_id), race in selected.groupby(
        ["fold_id", "evaluation_year", "method", "race_id"], sort=False, observed=True
    ):
        ids = race["race_id"].tolist()
        positions = pd.to_numeric(race["model_finish_position"], errors="raise").astype(int).tolist()
        probabilities = pd.to_numeric(race["probability_calibrated"], errors="raise").tolist()
        scores = pd.to_numeric(race["utility"], errors="raise").tolist()
        rows.append(
            {
                "fold_id": str(fold_id),
                "evaluation_year": int(evaluation_year),
                "race_id": str(race_id),
                "race_date": pd.Timestamp(race["race_date"].iloc[0]).normalize(),
                "method": str(method),
                "ndcg_at_3": ndcg_at_k(scores, positions, ids, k=3),
                "top_1_winner_mass": top_k_winner_mass(scores, positions, ids, k=1),
                "race_log_loss": race_log_loss(probabilities, positions, ids),
                "race_brier": race_brier_score(probabilities, positions, ids),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_year", "race_date", "race_id", "method"], kind="stable"
    ).reset_index(drop=True)


def summarize_year_macro(
    race_metrics: pd.DataFrame, comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize unweighted year-macro levels and signed improvements."""

    yearly = (
        race_metrics.groupby(["evaluation_year", "method"], observed=True)[list(PRIMARY_METRICS)]
        .mean()
        .reset_index()
    )
    methods: dict[str, Any] = {}
    for method, rows in yearly.groupby("method", sort=False):
        per_year = {
            str(int(row.evaluation_year)): {
                metric: float(getattr(row, metric)) for metric in PRIMARY_METRICS
            }
            for row in rows.itertuples(index=False)
        }
        methods[str(method)] = {
            "per_year": per_year,
            "year_macro": {
                metric: float(rows[metric].mean()) for metric in PRIMARY_METRICS
            },
            "year_count": int(rows["evaluation_year"].nunique()),
        }

    comparison_payload: dict[str, Any] = {}
    pivot = yearly.set_index(["evaluation_year", "method"])
    for comparison in comparisons:
        candidate = comparison["candidate"]
        reference = comparison["reference"]
        yearly_improvements: dict[str, dict[str, float]] = {}
        for year in sorted(race_metrics["evaluation_year"].unique()):
            yearly_improvements[str(int(year))] = {}
            for metric in PRIMARY_METRICS:
                sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
                value = sign * (
                    float(pivot.loc[(year, candidate), metric])
                    - float(pivot.loc[(year, reference), metric])
                )
                yearly_improvements[str(int(year))][metric] = value
        metrics: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            values = [payload[metric] for payload in yearly_improvements.values()]
            metrics[metric] = {
                "year_macro_improvement": float(np.mean(values)),
                "improved_years": int(sum(value > 0.0 for value in values)),
                "worsened_years": int(sum(value < 0.0 for value in values)),
                "tied_years": int(sum(value == 0.0 for value in values)),
                "direction_consistency": float(sum(value > 0.0 for value in values) / len(values)),
                "worst_year_improvement": float(min(values)),
                "direction": "positive_is_candidate_improvement",
            }
        comparison_payload[comparison["id"]] = {
            "candidate": candidate,
            "reference": reference,
            "type": comparison["type"],
            "per_year_improvement": yearly_improvements,
            "metrics": metrics,
        }
    return {"methods": methods, "comparisons": comparison_payload}


def paired_year_stratified_block_bootstrap(
    race_metrics: pd.DataFrame,
    *,
    comparisons: list[dict[str, Any]],
    n_resamples: int,
    confidence_level: float,
    seed: int,
    block_length_dates: int,
) -> dict[str, Any]:
    """Bootstrap date blocks within year, then average yearly race means equally."""

    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if block_length_dates < 1:
        raise ValueError("block_length_dates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    methods = list(dict.fromkeys(race_metrics["method"].astype(str)))
    years = sorted(int(year) for year in race_metrics["evaluation_year"].unique())
    expected_rows = race_metrics["race_id"].nunique() * len(methods)
    if len(race_metrics) != expected_rows:
        raise ValueError("race metric table is not a complete race-by-method grid")

    per_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in years:
        selected = race_metrics.loc[race_metrics["evaluation_year"].eq(year)]
        dates = np.array(sorted(pd.to_datetime(selected["race_date"]).unique()))
        if block_length_dates > len(dates):
            raise ValueError("block_length_dates exceeds dates in an evaluation year")
        date_sums = np.zeros((len(dates), len(methods), len(PRIMARY_METRICS)), dtype=np.float64)
        date_counts = np.zeros(len(dates), dtype=np.int64)
        for date_index, date in enumerate(dates):
            date_rows = selected.loc[pd.to_datetime(selected["race_date"]).eq(date)]
            date_counts[date_index] = date_rows["race_id"].nunique()
            for method_index, method in enumerate(methods):
                rows = date_rows.loc[date_rows["method"].eq(method)]
                if len(rows) != date_counts[date_index]:
                    raise ValueError("a date is missing one or more method predictions")
                date_sums[date_index, method_index] = rows.loc[:, PRIMARY_METRICS].sum().to_numpy()
        per_year[year] = (date_sums, date_counts)

    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.zeros((n_resamples, len(methods), len(PRIMARY_METRICS)), dtype=np.float64)
    for draw in range(n_resamples):
        yearly_draws = []
        for year in years:
            date_sums, date_counts = per_year[year]
            blocks = math.ceil(len(date_counts) / block_length_dates)
            starts = rng.integers(0, len(date_counts), size=blocks)
            offsets = np.arange(block_length_dates)
            sampled = ((starts[:, None] + offsets) % len(date_counts)).reshape(-1)[: len(date_counts)]
            yearly_draws.append(date_sums[sampled].sum(axis=0) / date_counts[sampled].sum())
        draws[draw] = np.mean(yearly_draws, axis=0)

    alpha = (1.0 - confidence_level) / 2.0
    point_rows = (
        race_metrics.groupby(["evaluation_year", "method"], observed=True)[list(PRIMARY_METRICS)]
        .mean()
        .groupby("method", observed=True)
        .mean()
    )

    def interval(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point": float(point),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }

    marginal: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        marginal[method] = {
            metric: interval(draws[:, method_index, metric_index], float(point_rows.loc[method, metric]))
            for metric_index, metric in enumerate(PRIMARY_METRICS)
        }
    paired: dict[str, Any] = {}
    for comparison in comparisons:
        candidate_index = methods.index(comparison["candidate"])
        reference_index = methods.index(comparison["reference"])
        paired[comparison["id"]] = {}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
            values = sign * (
                draws[:, candidate_index, metric_index] - draws[:, reference_index, metric_index]
            )
            point = sign * (
                float(point_rows.loc[comparison["candidate"], metric])
                - float(point_rows.loc[comparison["reference"], metric])
            )
            payload = interval(values, point)
            payload["fraction_positive"] = float(np.mean(values > 0.0))
            payload["direction"] = "positive_is_candidate_improvement"
            paired[comparison["id"]][metric] = payload
    return {
        "scheme": "moving_date_block_within_year_then_year_macro",
        "evaluation_years": years,
        "block_length_dates": block_length_dates,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "rng": "numpy.random.PCG64",
        "quantile_method": "linear",
        "marginal": marginal,
        "paired": paired,
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None and int(value) > 0 else None


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed-as-package"
    return versions


def run_rolling_evaluation(
    *,
    repo_root: str | Path,
    cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the registered 2020--2023 rolling-origin evaluation."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    cache_file = _resolve(root, cache_path)
    config_file = _resolve(root, config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rolling artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(config_file)
        validate_rolling_config(config)
        cached, cache_meta = read_model_frame_cache(cache_file)
        frame, isolation = isolate_rolling_source(
            cached, maximum_outcome_year=int(config["maximum_outcome_year"])
        )
        del cached
        all_feature_columns = tuple(cache_meta["feature_columns"])
        methods = resolve_rolling_methods(
            root=root, all_feature_columns=all_feature_columns, config=config
        )

        prediction_frames: list[pd.DataFrame] = []
        importance_rows: list[dict[str, Any]] = []
        model_records: dict[str, Any] = {}
        for fold in config["folds"]:
            roles = assign_fold_roles(frame, fold)
            eligible = roles.notna()
            fold_frame = frame.loc[eligible].copy()
            fold_frame["rolling_role"] = roles.loc[eligible].to_numpy()
            fold_records: dict[str, Any] = {}
            for method_id, method in methods.items():
                feature_columns = method["feature_columns"]
                model_config = method["model_config"]
                common = {
                    "frame": fold_frame,
                    "feature_columns": feature_columns,
                    "train_split": "train",
                    "model_validation_split": "model_validation",
                    "race_id_column": "race_id",
                    "finish_position_column": "model_finish_position",
                    "split_column": "rolling_role",
                    "params": model_config["parameters"],
                    "early_stopping_rounds": model_config.get("early_stopping_rounds"),
                }
                if method["model_kind"] == "binary":
                    model = train_binary(**common)
                else:
                    model = train_ranker(**common)

                scoring = fold_frame.loc[fold_frame["rolling_role"].isin(_SCORING_ROLES)].copy()
                raw = predict(
                    model,
                    scoring,
                    feature_columns=feature_columns,
                    model_kind=method["model_kind"],
                )
                utility = (
                    probability_logits(raw, epsilon=PROBABILITY_EPSILON)
                    if method["model_kind"] == "binary"
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
                scoring["probability_t1"] = race_softmax(scoring["utility"], scoring["race_id"])
                scoring["probability_calibrated"] = apply_temperature(
                    calibrator, scoring["utility"], scoring["race_id"]
                )
                scoring["fold_id"] = fold["id"]
                scoring["role"] = scoring["rolling_role"]
                scoring["evaluation_year"] = int(fold["evaluation_year"])
                scoring["method"] = method_id
                scoring["model_kind"] = method["model_kind"]
                prediction_columns = [
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
                        prediction_columns.insert(8, optional)
                prediction_frames.append(scoring.loc[:, prediction_columns])

                evaluation = scoring.loc[scoring["rolling_role"].eq("evaluation")]
                evaluated = evaluate_predictions(
                    evaluation["probability_calibrated"],
                    pd.to_numeric(evaluation["model_finish_position"], errors="raise").astype(int),
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
                effective_parameters = (
                    model.get_params(deep=False) if hasattr(model, "get_params") else model_config["parameters"]
                )
                fold_records[method_id] = {
                    "best_iteration": _best_iteration(model),
                    "temperature": float(calibrator.temperature),
                    "effective_parameters": effective_parameters,
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
            raise AssertionError(f"market columns reached rolling predictions: {forbidden}")
        race_metrics = rolling_race_metric_table(predictions)
        summary = summarize_year_macro(race_metrics, config["comparisons"])
        uncertainty_config = config["uncertainty"]
        bootstrap = paired_year_stratified_block_bootstrap(
            race_metrics,
            comparisons=config["comparisons"],
            n_resamples=int(uncertainty_config["bootstrap_resamples"]),
            confidence_level=float(uncertainty_config["confidence_level"]),
            seed=int(uncertainty_config["bootstrap_seed"]),
            block_length_dates=int(uncertainty_config["block_length_dates"]),
        )
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "hypothesis": config["hypothesis"],
            "scope": {
                "evaluation_years": list(_EXPECTED_EVALUATION_YEARS),
                "maximum_outcome_year": 2023,
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "selection_role": "rolling-origin screening; not a sealed final holdout",
            },
            "data": {
                "fingerprint": cache_meta["data_fingerprint"],
                "cache_sha256": sha256_file(cache_file),
                **isolation,
            },
            "methods": model_records,
            "year_summary": summary,
            "paired_block_bootstrap": bootstrap,
            "selection_accounting": config["selection_accounting"],
            "limitations": [
                "2020-2023 have prior research exposure and are screening folds, not untouched holdouts.",
                "Block intervals condition on fitted models and do not capture model-refit uncertainty or year drift.",
                "Four yearly directions are descriptive and are not an independent significance test.",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }

        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "config.json", config)
        write_json(
            temporary / "feature_schema.json",
            {
                "schema_version": 1,
                "methods": {
                    method_id: {
                        "feature_count": len(method["feature_columns"]),
                        "feature_columns": list(method["feature_columns"]),
                        "feature_columns_sha256": method["feature_columns_sha256"],
                        "feature_groups": {
                            name: list(columns) for name, columns in method["feature_groups"].items()
                        },
                        "resolution": method["feature_resolution"],
                    }
                    for method_id, method in methods.items()
                },
            },
        )
        write_json(
            temporary / "resolved_configs.json",
            {
                "schema_version": 1,
                "methods": {
                    method_id: {
                        "feature_config_path": str(method["feature_config_path"]),
                        "feature_config_hash": canonical_json_hash(method["feature_config"]),
                        "feature_config": method["feature_config"],
                        "model_config_path": str(method["model_config_path"]),
                        "model_config_hash": canonical_json_hash(method["model_config"]),
                        "model_config": method["model_config"],
                    }
                    for method_id, method in methods.items()
                },
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
            },
        )
        predictions.to_csv(
            temporary / "predictions_scoring.csv.gz", index=False, compression="gzip"
        )
        race_metrics.to_csv(
            temporary / "race_metrics.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(importance_rows).to_csv(temporary / "feature_importance.csv", index=False)
        write_artifact_manifest(temporary)
        temporary.replace(output)
        return metrics
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
