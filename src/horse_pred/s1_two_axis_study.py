"""Preregistered S1 rolling study for two-axis historical race value.

This runner deliberately builds a fresh, outcome-cutoff source from the raw
CSV.  It never loads a model-frame cache that could physically contain 2023+
rows, and it keeps odds/popularity outside every fitted and saved table.
"""

from __future__ import annotations

import json
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
from horse_pred.data import RAW_COLUMNS, normalize_raw, sha256_file
from horse_pred.evaluation import (
    evaluate_predictions,
    ndcg_at_k,
    race_balanced_reliability,
    race_brier_score,
    race_log_loss,
    top_k_winner_mass,
)
from horse_pred.features import FeatureConfig, build_features
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    predict,
    probability_logits,
    train_binary,
    train_ranker,
    validate_prediction_feature_columns,
)
from horse_pred.pipeline import PROBABILITY_EPSILON, prepare_model_frame
from horse_pred.race_content import RACE_CONTENT_COLUMN, build_race_content_history
from horse_pred.rolling_evaluation import assign_fold_roles
from horse_pred.two_axis_race_value import (
    FIELD_QUALITY_COLUMN,
    PERFORMANCE_COLUMN,
    TwoAxisRaceValueSpec,
    build_fold_two_axis_history,
)

_FAMILIES = ("binary", "lambdarank")
_ARMS = ("C0", "C1", "C2", "C3")
_SCORING_ROLES = ("calibration", "evaluation")
_METRICS = (
    "ndcg_at_3",
    "top_1_winner_mass",
    "winner_reciprocal_rank",
    "race_log_loss",
    "race_brier",
)
_HIGHER_IS_BETTER = {
    "ndcg_at_3",
    "top_1_winner_mass",
    "winner_reciprocal_rank",
}
_FORBIDDEN_FEATURE_FRAGMENTS = (
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "race_date",
    "finish_position",
    "winner_label",
    "odds",
    "popularity",
    "payout",
    "オッズ",
    "人気",
    "払戻",
)


def validate_s1_preregistration(config: dict[str, Any]) -> None:
    """Fail closed unless the fixed S1 protocol is present."""

    required = {
        "schema_version",
        "experiment_id",
        "raw_sha256",
        "maximum_outcome_year",
        "forbidden_years",
        "market_used",
        "folds",
        "controls",
        "features",
        "arms",
        "comparisons_per_family",
        "metrics",
        "uncertainty",
        "slices",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"S1 preregistration is missing: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("S1 preregistration schema_version must be 1")
    if int(config["maximum_outcome_year"]) != 2022:
        raise ValueError("S1 maximum_outcome_year must remain 2022")
    if set(map(int, config["forbidden_years"])) != {2023, 2024, 2025}:
        raise ValueError("S1 forbidden_years must be exactly 2023, 2024, and 2025")
    if config["market_used"] is not False:
        raise ValueError("S1 must not enable market information")
    folds = config["folds"]
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("S1 must contain exactly three rolling folds")
    expected_evaluation = (2020, 2021, 2022)
    for fold, evaluation_year in zip(folds, expected_evaluation):
        if int(fold["evaluation_year"]) != evaluation_year:
            raise ValueError("S1 evaluation years must be 2020, 2021, and 2022")
        if int(fold["train_start_year"]) != 2014:
            raise ValueError("S1 train starts in 2014")
        if int(fold["train_end_year"]) != evaluation_year - 3:
            raise ValueError("S1 train end must be evaluation year minus three")
        if int(fold["early_stopping_year"]) != evaluation_year - 2:
            raise ValueError("S1 early stopping must be evaluation year minus two")
        if int(fold["calibration_year"]) != evaluation_year - 1:
            raise ValueError("S1 calibration must be evaluation year minus one")
    expected_arms = {
        "C0": [],
        "C1": [PERFORMANCE_COLUMN],
        "C2": [FIELD_QUALITY_COLUMN],
        "C3": [PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN],
    }
    if config["arms"] != expected_arms:
        raise ValueError("S1 arms differ from the registered four-arm design")
    if tuple(config["metrics"]) != _METRICS:
        raise ValueError("S1 metric list differs from preregistration")
    uncertainty = config["uncertainty"]
    if int(uncertainty["resamples"]) < 2:
        raise ValueError("S1 bootstrap requires at least two resamples")
    if int(uncertainty["block_length_dates"]) < 1:
        raise ValueError("S1 date block length must be positive")


def isolate_s1_source(
    raw_path: str | Path,
    *,
    maximum_outcome_year: int,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only raw rows through the S1 cutoff before normalization."""

    if maximum_outcome_year != 2022:
        raise ValueError("S1 source cutoff must remain 2022")
    path = Path(raw_path).resolve()
    fingerprint = sha256_file(path)
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("raw SHA-256 differs from the S1 preregistration")
    chunks: list[pd.DataFrame] = []
    source_rows = 0
    excluded_by_year: dict[str, int] = {}
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=200_000,
    ):
        if tuple(chunk.columns) != RAW_COLUMNS:
            raise ValueError("raw CSV schema differs from the frozen schema")
        source_rows += len(chunk)
        dates = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="raise")
        years = dates.dt.year
        for year, count in years.loc[years.gt(maximum_outcome_year)].value_counts().items():
            key = str(int(year))
            excluded_by_year[key] = excluded_by_year.get(key, 0) + int(count)
        chunks.append(chunk.loc[years.le(maximum_outcome_year)].copy())
    if not chunks:
        raise ValueError("raw CSV contains no chunks")
    raw = pd.concat(chunks, ignore_index=True)
    normalized = normalize_raw(raw)
    del raw, chunks
    dates = pd.to_datetime(normalized["race_date"], errors="raise")
    if dates.dt.year.max() > maximum_outcome_year:
        raise AssertionError("post-cutoff row survived S1 source isolation")
    return normalized, {
        "raw_sha256": fingerprint,
        "source_rows": int(source_rows),
        "selected_rows": int(len(normalized)),
        "selected_races": int(normalized["race_id"].nunique()),
        "selected_min_date": str(dates.min().date()),
        "selected_max_date": str(dates.max().date()),
        "excluded_rows_by_year": dict(sorted(excluded_by_year.items())),
        "maximum_outcome_year": maximum_outcome_year,
        "rows_used_2023": 0,
        "rows_used_2024": 0,
        "rows_used_2025": 0,
    }


def _previous_conditions(normalized: pd.DataFrame) -> pd.DataFrame:
    source = normalized.copy()
    source["_date"] = pd.to_datetime(source["race_date"], errors="raise").dt.normalize()
    source["_order"] = np.arange(len(source), dtype=np.int64)
    source = source.sort_values(["_date", "race_id", "_order"], kind="stable")
    state: dict[str, tuple[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for _event_date, day in source.groupby("_date", sort=True):
        starters = day.loc[day["started"].eq(True).fillna(False)]  # noqa: E712
        for _, row in starters.iterrows():
            horse = str(row["horse_id"])
            prior = state.get(horse)
            rows.append(
                {
                    "race_id": str(row["race_id"]),
                    "horse_id": horse,
                    "previous_surface": prior[0] if prior else pd.NA,
                    "previous_distance": prior[1] if prior else np.nan,
                }
            )
        for _, row in starters.iterrows():
            distance = pd.to_numeric(pd.Series([row["distance"]]), errors="coerce").iloc[0]
            if pd.notna(distance):
                state[str(row["horse_id"])] = (str(row["course_type"]), float(distance))
    return pd.DataFrame(rows)


def _build_base_frame(
    normalized: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    dataset = build_features(
        normalized,
        split_config=load_json(root / "configs/splits.json"),
        config=FeatureConfig(),
    )
    frame = prepare_model_frame(dataset)
    content = build_race_content_history(normalized, through_year=2022)
    content = content.loc[:, ["race_id", "horse_id", RACE_CONTENT_COLUMN]].copy()
    content["race_id"] = content["race_id"].astype("string")
    content["horse_id"] = content["horse_id"].astype("string")
    frame = frame.merge(
        content,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    previous = _previous_conditions(normalized)
    previous["race_id"] = previous["race_id"].astype("string")
    previous["horse_id"] = previous["horse_id"].astype("string")
    frame = frame.merge(
        previous,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    # Market fields are discarded immediately after the model population is fixed.
    market_columns = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("odds", "popularity", "オッズ", "人気"))
    ]
    frame = frame.drop(columns=market_columns)
    all_features = tuple((*dataset.feature_columns, RACE_CONTENT_COLUMN))
    validate_prediction_feature_columns(all_features)
    frame[RACE_CONTENT_COLUMN] = pd.to_numeric(
        frame[RACE_CONTENT_COLUMN], errors="coerce"
    ).astype("float32")
    return frame, all_features, {
        "base_runner_count": int(len(frame)),
        "base_race_count": int(frame["race_id"].nunique()),
        "base_feature_count_with_pv01": len(all_features),
        "market_columns_removed": market_columns,
    }


def _resolve_controls(
    root: Path, all_features: tuple[str, ...], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        control = config["controls"][family]
        feature_config = load_json(root / control["feature_config"])
        validate_cached_experiment_config(feature_config)
        columns, groups, resolution = resolve_semantic_feature_selection(
            all_features, feature_config
        )
        if len(columns) != int(control["feature_count"]):
            raise ValueError(f"{family} control feature count changed")
        model_key = "binary" if family == "binary" else "lambdarank"
        model_path = root / feature_config["model_configs"][model_key]
        model_config = load_json(model_path)
        expected_objective = "binary" if family == "binary" else "lambdarank"
        if model_config.get("parameters", {}).get("objective") != expected_objective:
            raise ValueError(f"{family} control model objective changed")
        resolved[family] = {
            "control_columns": columns,
            "feature_groups": groups,
            "feature_resolution": resolution,
            "feature_config": feature_config,
            "feature_config_path": str(Path(control["feature_config"])),
            "model_config": model_config,
            "model_config_path": str(model_path.relative_to(root)),
            "control_columns_sha256": feature_columns_checksum(columns),
        }
    return resolved


def _feature_scope(control: tuple[str, ...], additions: list[str]) -> tuple[str, ...]:
    columns = tuple((*control, *additions))
    if len(columns) != len(set(columns)):
        raise ValueError("S1 feature scope contains duplicate columns")
    validate_prediction_feature_columns(columns)
    forbidden = [
        column
        for column in columns
        if any(fragment in column.lower() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"S1 feature scope contains forbidden columns: {forbidden}")
    return columns


def _winner_reciprocal_rank(scores: Any, positions: Any, race_ids: Any) -> float:
    frame = pd.DataFrame(
        {"score": np.asarray(scores, dtype=float), "position": list(positions), "race_id": list(race_ids)}
    )
    values: list[float] = []
    for _, race in frame.groupby("race_id", sort=False, observed=True):
        ordered = race.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        winners = np.flatnonzero(pd.to_numeric(ordered["position"], errors="raise").eq(1))
        if len(winners) == 0:
            raise ValueError("race has no winner")
        values.append(float(np.mean(1.0 / (winners + 1.0))))
    return float(np.mean(values))


def _metric_payload(frame: pd.DataFrame) -> dict[str, float]:
    ids = frame["race_id"].tolist()
    positions = pd.to_numeric(frame["model_finish_position"], errors="raise").astype(int).tolist()
    scores = pd.to_numeric(frame["utility"], errors="raise").to_numpy()
    probabilities = pd.to_numeric(frame["probability_calibrated"], errors="raise").to_numpy()
    return {
        "ndcg_at_3": ndcg_at_k(scores, positions, ids, k=3),
        "top_1_winner_mass": top_k_winner_mass(scores, positions, ids, k=1),
        "winner_reciprocal_rank": _winner_reciprocal_rank(scores, positions, ids),
        "race_log_loss": race_log_loss(probabilities, positions, ids),
        "race_brier": race_brier_score(probabilities, positions, ids),
    }


def _race_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = predictions.loc[predictions["role"].eq("evaluation")]
    for keys, race in selected.groupby(
        ["fold_id", "evaluation_year", "method", "family", "arm", "race_id"],
        sort=False,
        observed=True,
    ):
        fold_id, year, method, family, arm, race_id = keys
        rows.append(
            {
                "fold_id": str(fold_id),
                "evaluation_year": int(year),
                "method": str(method),
                "family": str(family),
                "arm": str(arm),
                "race_id": str(race_id),
                "race_date": pd.Timestamp(race["race_date"].iloc[0]).normalize(),
                **_metric_payload(race),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_year", "race_date", "race_id", "method"], kind="stable"
    ).reset_index(drop=True)


def _comparisons() -> list[dict[str, str]]:
    pairs = (
        ("C1", "C0"),
        ("C2", "C0"),
        ("C3", "C0"),
        ("C3", "C2"),
        ("C3", "C1"),
    )
    return [
        {
            "id": f"{family}_{candidate}_vs_{reference}",
            "candidate": f"{family}_{candidate}",
            "reference": f"{family}_{reference}",
            "family": family,
            "candidate_arm": candidate,
            "reference_arm": reference,
        }
        for family in _FAMILIES
        for candidate, reference in pairs
    ]


def _year_summary(
    race_metrics: pd.DataFrame, comparisons: list[dict[str, str]]
) -> dict[str, Any]:
    yearly = (
        race_metrics.groupby(["evaluation_year", "method"], observed=True)[list(_METRICS)]
        .mean()
        .reset_index()
    )
    methods: dict[str, Any] = {}
    for method, rows in yearly.groupby("method", sort=False):
        methods[str(method)] = {
            "per_year": {
                str(int(row.evaluation_year)): {
                    metric: float(getattr(row, metric)) for metric in _METRICS
                }
                for row in rows.itertuples(index=False)
            },
            "year_macro": {metric: float(rows[metric].mean()) for metric in _METRICS},
        }
    indexed = yearly.set_index(["evaluation_year", "method"])
    comparison_payload: dict[str, Any] = {}
    for comparison in comparisons:
        per_year: dict[str, dict[str, float]] = {}
        for year in sorted(race_metrics["evaluation_year"].unique()):
            per_year[str(int(year))] = {}
            for metric in _METRICS:
                sign = 1.0 if metric in _HIGHER_IS_BETTER else -1.0
                per_year[str(int(year))][metric] = sign * (
                    float(indexed.loc[(year, comparison["candidate"]), metric])
                    - float(indexed.loc[(year, comparison["reference"]), metric])
                )
        comparison_payload[comparison["id"]] = {
            **comparison,
            "per_year_improvement": per_year,
            "metrics": {
                metric: {
                    "year_macro_improvement": float(
                        np.mean([row[metric] for row in per_year.values()])
                    ),
                    "improved_years": int(sum(row[metric] > 0 for row in per_year.values())),
                    "worsened_years": int(sum(row[metric] < 0 for row in per_year.values())),
                    "direction": "positive_is_candidate_improvement",
                }
                for metric in _METRICS
            },
        }
    return {"methods": methods, "comparisons": comparison_payload}


def _paired_bootstrap(
    race_metrics: pd.DataFrame,
    comparisons: list[dict[str, str]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    block_length_dates: int,
) -> dict[str, Any]:
    methods = list(dict.fromkeys(race_metrics["method"].astype(str)))
    years = sorted(int(value) for value in race_metrics["evaluation_year"].unique())
    per_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in years:
        subset = race_metrics.loc[race_metrics["evaluation_year"].eq(year)]
        dates = np.array(sorted(pd.to_datetime(subset["race_date"]).unique()))
        sums = np.zeros((len(dates), len(methods), len(_METRICS)), dtype=float)
        counts = np.zeros(len(dates), dtype=int)
        for date_index, date in enumerate(dates):
            on_date = subset.loc[pd.to_datetime(subset["race_date"]).eq(date)]
            counts[date_index] = on_date["race_id"].nunique()
            for method_index, method in enumerate(methods):
                rows = on_date.loc[on_date["method"].eq(method)]
                if len(rows) != counts[date_index]:
                    raise ValueError("S1 race metric grid is incomplete")
                sums[date_index, method_index] = rows[list(_METRICS)].sum().to_numpy()
        per_year[year] = sums, counts
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.zeros((n_resamples, len(methods), len(_METRICS)), dtype=float)
    for draw in range(n_resamples):
        yearly: list[np.ndarray] = []
        for year in years:
            sums, counts = per_year[year]
            starts = rng.integers(0, len(counts), size=math.ceil(len(counts) / block_length_dates))
            sampled = ((starts[:, None] + np.arange(block_length_dates)) % len(counts)).reshape(-1)
            sampled = sampled[: len(counts)]
            yearly.append(sums[sampled].sum(axis=0) / counts[sampled].sum())
        draws[draw] = np.mean(yearly, axis=0)
    alpha = (1.0 - confidence_level) / 2.0
    paired: dict[str, Any] = {}
    for comparison in comparisons:
        candidate = methods.index(comparison["candidate"])
        reference = methods.index(comparison["reference"])
        paired[comparison["id"]] = {}
        for metric_index, metric in enumerate(_METRICS):
            sign = 1.0 if metric in _HIGHER_IS_BETTER else -1.0
            values = sign * (draws[:, candidate, metric_index] - draws[:, reference, metric_index])
            paired[comparison["id"]][metric] = {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
                "fraction_positive": float(np.mean(values > 0)),
                "direction": "positive_is_candidate_improvement",
            }
    return {
        "scheme": "moving_date_block_within_year_then_year_macro",
        "evaluation_years": years,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "block_length_dates": block_length_dates,
        "paired": paired,
    }


def assign_s1_slice_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return preregistered race-constant diagnostic flags."""

    required = {
        "race_id",
        "model_finish_position",
        "field_size",
        "horse_history__career__starts",
        "context__class_tier",
        "course_type",
        "distance",
        "previous_surface",
        "previous_distance",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"S1 slice frame is missing: {missing}")
    winners = pd.to_numeric(frame["model_finish_position"], errors="raise").eq(1)
    prior_known = frame["previous_surface"].notna() & frame["previous_distance"].notna()
    runner = {
        "history_0_winner": winners
        & pd.to_numeric(frame["horse_history__career__starts"], errors="coerce").fillna(0).eq(0),
        "new_race": pd.to_numeric(frame["context__class_tier"], errors="coerce").eq(0),
        "open_or_graded": pd.to_numeric(frame["context__class_tier"], errors="coerce").eq(5),
        "field_size_15_plus": pd.to_numeric(frame["field_size"], errors="raise").ge(15),
        "winner_surface_switch": winners
        & prior_known
        & frame["course_type"].astype("string").ne(frame["previous_surface"].astype("string")),
        "winner_distance_change_400_plus": winners
        & prior_known
        & (
            pd.to_numeric(frame["distance"], errors="coerce")
            - pd.to_numeric(frame["previous_distance"], errors="coerce")
        ).abs().ge(400),
    }
    flags: dict[str, pd.Series] = {}
    for name, values in runner.items():
        race_flag = values.fillna(False).groupby(frame["race_id"], sort=False).transform("max").astype(bool)
        if race_flag.groupby(frame["race_id"], sort=False).nunique().gt(1).any():
            raise AssertionError(f"S1 slice {name} is not race-constant")
        flags[name] = race_flag
    return flags


def classify_s1_comparison(
    year_summary: dict[str, Any], interval: dict[str, Any], *, path: str
) -> str:
    """Apply the frozen probability or ranking acceptance path."""

    metrics = year_summary["metrics"]
    if path == "probability":
        primary = "race_log_loss"
        threshold = 0.002
        guards = (
            metrics["race_brier"]["year_macro_improvement"] >= 0,
            metrics["ndcg_at_3"]["year_macro_improvement"] >= -0.002,
            metrics["top_1_winner_mass"]["year_macro_improvement"] >= -0.005,
        )
    elif path == "ranking":
        primary = "ndcg_at_3"
        threshold = 0.0
        guards = (
            metrics["race_log_loss"]["year_macro_improvement"] >= -0.002,
            metrics["race_brier"]["year_macro_improvement"] >= -0.001,
            metrics["top_1_winner_mass"]["year_macro_improvement"] >= -0.005,
        )
    else:
        raise ValueError("S1 decision path must be probability or ranking")
    effect = float(metrics[primary]["year_macro_improvement"])
    directions = int(metrics[primary]["improved_years"])
    lower = float(interval[primary]["lower"])
    if effect >= threshold and directions >= 2 and all(guards) and lower > 0:
        return "supported"
    if effect > 0 and directions >= 2 and all(guards):
        return "weakly_supported"
    if effect < 0 and directions <= 1:
        return "rejected"
    return "inconclusive"


def _decision_payload(summary: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for comparison_id, year_result in summary["comparisons"].items():
        interval = bootstrap["paired"][comparison_id]
        comparisons[comparison_id] = {
            "probability_path": classify_s1_comparison(year_result, interval, path="probability"),
            "ranking_path": classify_s1_comparison(year_result, interval, path="ranking"),
        }

    order = {"rejected": 0, "inconclusive": 1, "weakly_supported": 2, "supported": 3}

    def aggregate(ids: list[str]) -> str:
        labels = [
            label
            for comparison_id in ids
            for label in comparisons[comparison_id].values()
        ]
        if "supported" in labels:
            return "supported"
        if "weakly_supported" in labels:
            return "weakly_supported"
        if labels and max(order[label] for label in labels) == 0:
            return "rejected"
        return "inconclusive"

    performance_ids = [f"{family}_{pair}" for family in _FAMILIES for pair in ("C1_vs_C0", "C3_vs_C2")]
    field_ids = [f"{family}_{pair}" for family in _FAMILIES for pair in ("C2_vs_C0", "C3_vs_C1")]
    joint_ids = [f"{family}_C3_vs_C0" for family in _FAMILIES]
    axes = {
        "performance_axis": aggregate(performance_ids),
        "field_quality_axis": aggregate(field_ids),
        "joint_two_axis": aggregate(joint_ids),
    }
    if axes["performance_axis"] in {"supported", "weakly_supported"} and axes[
        "field_quality_axis"
    ] in {"supported", "weakly_supported"}:
        case = "C_both_independently_supported"
        next_recommendation = "human_choice_between_S2_and_S3"
    elif axes["performance_axis"] in {"supported", "weakly_supported"}:
        case = "A_performance_supported"
        next_recommendation = "S3_condition_adjusted_performance_target"
    elif axes["field_quality_axis"] in {"supported", "weakly_supported"}:
        case = "B_field_quality_supported"
        next_recommendation = "field_quality_follow_up_human_review"
    elif axes["joint_two_axis"] in {"supported", "weakly_supported"}:
        case = "D_only_joint_supported"
        next_recommendation = "human_review_of_complementarity"
    else:
        case = "E_all_inconclusive_or_rejected"
        next_recommendation = "S2_supervised_race_wise_probability"
    return {
        "schema_version": 1,
        "comparisons": comparisons,
        "axes": axes,
        "case": case,
        "next_recommendation": next_recommendation,
        "production_control_change": False,
        "S2_executed": False,
        "S3_executed": False,
    }


def _software_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed-as-package"
    return result


def verify_artifact_manifest(directory: str | Path) -> dict[str, Any]:
    """Verify every hash and size in a completed artifact manifest."""

    root = Path(directory)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise ValueError(f"artifact manifest file is missing: {item['path']}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"artifact size mismatch: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact hash mismatch: {item['path']}")
    return {"verified": True, "file_count": len(manifest["files"])}


def run_s1_two_axis_study(
    repo_root: str | Path,
    raw_path: str | Path,
    preregistration_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run all 24 preregistered S1 fits and write a local atomic artifact."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    prereg_path = Path(preregistration_path)
    if not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S1 artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(prereg_path)
        validate_s1_preregistration(config)
        normalized, source_audit = isolate_s1_source(
            raw_path,
            maximum_outcome_year=int(config["maximum_outcome_year"]),
            expected_sha256=str(config["raw_sha256"]),
        )
        frame, all_features, base_audit = _build_base_frame(normalized, root)
        controls = _resolve_controls(root, all_features, config)

        predictions: list[pd.DataFrame] = []
        importance_rows: list[dict[str, Any]] = []
        shap_rows: list[dict[str, Any]] = []
        permutation_rows: list[dict[str, Any]] = []
        model_records: dict[str, Any] = {}
        fold_feature_audits: dict[str, Any] = {}
        slice_records: list[dict[str, Any]] = []

        for fold in config["folds"]:
            history, observations, feature_audit = build_fold_two_axis_history(
                normalized, fold, TwoAxisRaceValueSpec()
            )
            fold_feature_audits[str(fold["id"])] = feature_audit
            joined = frame.merge(
                history.loc[:, ["race_id", "horse_id", PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN]],
                on=["race_id", "horse_id"],
                how="left",
                validate="one_to_one",
            )
            for column in (PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN):
                joined[column] = pd.to_numeric(joined[column], errors="coerce").astype("float32")
            roles = assign_fold_roles(joined, fold)
            eligible = roles.notna()
            fold_frame = joined.loc[eligible].copy()
            fold_frame["rolling_role"] = roles.loc[eligible].to_numpy()
            if pd.to_datetime(fold_frame["race_date"]).dt.year.max() > 2022:
                raise AssertionError("S1 fold opened a post-2022 row")
            fold_records: dict[str, Any] = {}

            for family in _FAMILIES:
                control = controls[family]
                for arm in _ARMS:
                    method = f"{family}_{arm}"
                    additions = list(config["arms"][arm])
                    feature_columns = _feature_scope(control["control_columns"], additions)
                    model_config = control["model_config"]
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
                    model = train_binary(**common) if family == "binary" else train_ranker(**common)
                    scoring = fold_frame.loc[
                        fold_frame["rolling_role"].isin(_SCORING_ROLES)
                    ].copy()
                    raw = predict(model, scoring, feature_columns=feature_columns, model_kind=family)
                    utility = (
                        probability_logits(raw, epsilon=PROBABILITY_EPSILON)
                        if family == "binary"
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
                    scoring["fold_id"] = fold["id"]
                    scoring["role"] = scoring["rolling_role"]
                    scoring["evaluation_year"] = int(fold["evaluation_year"])
                    scoring["method"] = method
                    scoring["family"] = family
                    scoring["arm"] = arm
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
                        "probability_calibrated",
                    ]
                    predictions.append(scoring.loc[:, keep])
                    evaluation = scoring.loc[scoring["rolling_role"].eq("evaluation")].copy()
                    evaluated = evaluate_predictions(
                        evaluation["probability_calibrated"],
                        evaluation["model_finish_position"].astype(int),
                        evaluation["race_id"],
                        ranking_scores=evaluation["utility"],
                    )
                    metrics = _metric_payload(evaluation)
                    reliability = race_balanced_reliability(
                        evaluation["probability_calibrated"],
                        evaluation["model_finish_position"].astype(int),
                        evaluation["race_id"],
                        n_bins=10,
                        strategy="fixed",
                    )
                    model_dir = temporary / "models" / str(fold["id"])
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model.booster_.save_model(str(model_dir / f"{method}.txt"))
                    gains = model.booster_.feature_importance(importance_type="gain")
                    splits = model.booster_.feature_importance(importance_type="split")
                    total_gain = float(np.sum(gains))
                    for feature, gain, split in zip(feature_columns, gains, splits):
                        importance_rows.append(
                            {
                                "fold_id": fold["id"],
                                "evaluation_year": fold["evaluation_year"],
                                "method": method,
                                "feature": feature,
                                "importance_gain": float(gain),
                                "importance_gain_fraction": float(gain) / total_gain if total_gain else 0.0,
                                "importance_split": int(split),
                            }
                        )

                    if additions:
                        matrix = evaluation.loc[:, feature_columns]
                        contributions = model.booster_.predict(matrix, pred_contrib=True)
                        for feature in additions:
                            feature_index = feature_columns.index(feature)
                            feature_values = pd.to_numeric(evaluation[feature], errors="coerce")
                            shap_values = np.asarray(contributions)[:, feature_index]
                            shap_rows.append(
                                {
                                    "fold_id": fold["id"],
                                    "evaluation_year": fold["evaluation_year"],
                                    "method": method,
                                    "feature": feature,
                                    "runner_count": int(len(evaluation)),
                                    "coverage": float(feature_values.notna().mean()),
                                    "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                                    "mean_shap": float(np.mean(shap_values)),
                                }
                            )
                            rng = np.random.Generator(
                                np.random.PCG64(42 + int(fold["evaluation_year"]) + feature_index)
                            )
                            permuted = evaluation.copy()
                            permutation = {
                                race_id: rng.permutation(len(values))
                                for race_id, values in evaluation.groupby(
                                    "race_id", sort=False
                                )[feature]
                            }

                            def permute_group(
                                values: pd.Series,
                                choices: dict[Any, np.ndarray] = permutation,
                            ) -> np.ndarray:
                                return values.iloc[choices[values.name]].to_numpy()

                            permuted[feature] = evaluation.groupby(
                                "race_id", sort=False
                            )[feature].transform(permute_group)
                            perm_raw = predict(
                                model, permuted, feature_columns=feature_columns, model_kind=family
                            )
                            perm_utility = (
                                probability_logits(perm_raw, epsilon=PROBABILITY_EPSILON)
                                if family == "binary"
                                else perm_raw
                            )
                            permuted["utility"] = perm_utility
                            permuted["probability_calibrated"] = apply_temperature(
                                calibrator, permuted["utility"], permuted["race_id"]
                            )
                            perm_metrics = _metric_payload(permuted)
                            permutation_rows.append(
                                {
                                    "fold_id": fold["id"],
                                    "evaluation_year": fold["evaluation_year"],
                                    "method": method,
                                    "feature": feature,
                                    **{
                                        f"{metric}_degradation": (
                                            metrics[metric] - perm_metrics[metric]
                                            if metric in _HIGHER_IS_BETTER
                                            else perm_metrics[metric] - metrics[metric]
                                        )
                                        for metric in _METRICS
                                    },
                                }
                            )

                    flags = assign_s1_slice_flags(evaluation)
                    for slice_name, flag in flags.items():
                        selected = evaluation.loc[flag]
                        if selected.empty:
                            continue
                        slice_records.append(
                            {
                                "fold_id": fold["id"],
                                "evaluation_year": fold["evaluation_year"],
                                "method": method,
                                "family": family,
                                "arm": arm,
                                "slice": slice_name,
                                "race_count": int(selected["race_id"].nunique()),
                                "runner_count": int(len(selected)),
                                **_metric_payload(selected),
                            }
                        )
                    fold_records[method] = {
                        "family": family,
                        "arm": arm,
                        "feature_count": len(feature_columns),
                        "feature_columns_sha256": feature_columns_checksum(feature_columns),
                        "best_iteration": (
                            int(model.best_iteration_)
                            if getattr(model, "best_iteration_", None)
                            else None
                        ),
                        "temperature": float(calibrator.temperature),
                        "calibration_slope_equivalent": float(1.0 / calibrator.temperature),
                        "calibration_intercept_identified": 0.0,
                        "metrics": metrics,
                        "reliability_fixed_bins": reliability,
                        "evaluation": evaluated,
                    }
            model_records[str(fold["id"])] = fold_records
            del history, observations, joined, fold_frame

        prediction_frame = pd.concat(predictions, ignore_index=True)
        race_metrics = _race_metric_table(prediction_frame)
        comparisons = _comparisons()
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
        decision = _decision_payload(year_summary, bootstrap)

        feature_distribution: dict[str, Any] = {}
        for fold in config["folds"]:
            history, _, _ = build_fold_two_axis_history(normalized, fold, TwoAxisRaceValueSpec())
            for feature in (PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN):
                values = pd.to_numeric(history[feature], errors="coerce")
                key = f"{fold['id']}::{feature}"
                feature_distribution[key] = {
                    "runner_count": int(len(values)),
                    "nonmissing": int(values.notna().sum()),
                    "missing": int(values.isna().sum()),
                    "coverage": float(values.notna().mean()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "quantiles": {
                        str(q): float(values.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
                    },
                }

        feature_diagnostics = {
            "schema_version": 1,
            "distribution": feature_distribution,
            "gain_importance": pd.DataFrame(importance_rows).loc[
                lambda x: x["feature"].isin([PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN])
            ].to_dict(orient="records"),
            "shap_dependence_summary": shap_rows,
            "race_aware_permutation": permutation_rows,
            "fold_feature_audits": fold_feature_audits,
            "importance_is_not_acceptance_evidence": True,
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
            },
            "data": {**source_audit, **base_audit},
            "folds": model_records,
            "year_summary": year_summary,
            "paired_block_bootstrap": bootstrap,
            "slices": slice_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        feature_schema = {
            "schema_version": 1,
            "controls": {
                family: {
                    "feature_count": len(payload["control_columns"]),
                    "feature_columns": list(payload["control_columns"]),
                    "feature_columns_sha256": payload["control_columns_sha256"],
                }
                for family, payload in controls.items()
            },
            "arms": {
                family: {
                    arm: {
                        "feature_count": len(_feature_scope(controls[family]["control_columns"], config["arms"][arm])),
                        "feature_columns_sha256": feature_columns_checksum(
                            _feature_scope(controls[family]["control_columns"], config["arms"][arm])
                        ),
                    }
                    for arm in _ARMS
                }
                for family in _FAMILIES
            },
        }
        write_json(temporary / "config.json", config)
        write_json(temporary / "metrics.json", metrics_payload)
        write_json(temporary / "comparison.json", {"year_summary": year_summary, "bootstrap": bootstrap})
        write_json(temporary / "decision.json", decision)
        write_json(temporary / "feature_diagnostics.json", feature_diagnostics)
        write_json(temporary / "feature_schema.json", feature_schema)
        write_json(temporary / "fold_feature_audits.json", fold_feature_audits)
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
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
            },
        )
        prediction_dir = temporary / "predictions"
        prediction_dir.mkdir()
        prediction_frame.to_csv(prediction_dir / "scoring.csv.gz", index=False, compression="gzip")
        table_dir = temporary / "feature_tables"
        table_dir.mkdir()
        race_metrics.to_csv(table_dir / "race_metrics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(slice_records).to_csv(table_dir / "slice_metrics.csv", index=False)
        pd.DataFrame(importance_rows).to_csv(table_dir / "feature_importance.csv", index=False)
        pd.DataFrame(shap_rows).to_csv(table_dir / "shap_summary.csv", index=False)
        pd.DataFrame(permutation_rows).to_csv(table_dir / "permutation_summary.csv", index=False)
        write_artifact_manifest(temporary)
        temporary.replace(output)
        verification = verify_artifact_manifest(output)
        return {**metrics_payload, "decision": decision, "artifact_verification": verification}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
