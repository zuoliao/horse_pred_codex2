"""2024-only diagnostics for the frozen LightGBM baseline.

This module deliberately separates model diagnostics from model selection.  It
loads the already fitted baseline, keeps 2025 outside every calculation, and
writes feature-attribution, race-aware permutation, and conditional-error
tables.  Final odds are joined only as a post-event oracle diagnostic.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.cached_experiment import isolate_pre_2025_frame
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import (
    conditional_race_metrics,
    evaluate_predictions,
    winner_mass_targets,
)
from horse_pred.features import semantic_feature_groups_v2
from horse_pred.modeling import race_softmax, validate_grouped_rows

_MODEL_SPECS = {
    "binary": {
        "model_file": "models/binary.txt",
        "temperature_key": "binary",
        "artifact_raw": "pred_binary_raw",
        "artifact_probability": "prob_binary_logit_softmax_temperature_2023",
    },
    "lambdarank": {
        "model_file": "models/lambdarank.txt",
        "temperature_key": "lambdarank",
        "artifact_raw": "score_lambdarank",
        "artifact_probability": "prob_lambdarank_softmax_temperature_2023",
    },
}

_VENUE_LABELS = {
    1: "01_sapporo",
    2: "02_hakodate",
    3: "03_fukushima",
    4: "04_niigata",
    5: "05_tokyo",
    6: "06_nakayama",
    7: "07_chukyo",
    8: "08_kyoto",
    9: "09_hanshin",
    10: "10_kokura",
}
_CLASS_LABELS = {
    0: "newcomer",
    1: "maiden",
    2: "one_win",
    3: "two_win",
    4: "three_win",
    5: "open_or_graded",
}
_FORBIDDEN_NAME_FRAGMENTS = (
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "race_date",
    "final_odds",
    "odds",
    "popularity",
    "finish_position",
    "winner_label",
    "payout",
)


def isolate_development_2024(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop retrospective rows first, then return an exact 2024 development view."""

    pre_2025, isolation = isolate_pre_2025_frame(frame)
    development = pre_2025.loc[pre_2025["split"].eq("development")].copy()
    dates = pd.to_datetime(development["race_date"], errors="raise")
    if development.empty or not dates.dt.year.eq(2024).all():
        raise ValueError("diagnostics require development rows from calendar year 2024 only")
    development["race_date"] = dates
    validate_grouped_rows(development["race_id"])
    isolation = dict(isolation)
    isolation.update(
        {
            "development_2024_rows": int(len(development)),
            "development_2024_races": int(development["race_id"].nunique()),
            "development_2024_dates": int(development["race_date"].nunique()),
        }
    )
    return development, isolation


def previous_condition_metadata(pre_2025: pd.DataFrame) -> pd.DataFrame:
    """Build prior eligible-race condition metadata with a strict past-date check.

    The cache contains only the model population.  Consequently these fields
    describe the previous *eligible cached flat race*, not necessarily the
    horse's previous official start when an intervening race was unavailable or
    excluded by the PIT-C population rule.
    """

    required = {
        "race_id",
        "race_date",
        "horse_id",
        "course_type",
        "distance",
        "context__venue_code",
    }
    missing = sorted(required.difference(pre_2025.columns))
    if missing:
        raise ValueError(f"condition history is missing columns: {missing}")
    history = pre_2025.loc[:, sorted(required)].copy()
    history["race_date"] = pd.to_datetime(history["race_date"], errors="raise")
    if history["race_date"].dt.year.ge(2025).any():
        raise ValueError("previous-condition reconstruction received 2025+ rows")
    history["race_id"] = history["race_id"].astype("string")
    history["horse_id"] = history["horse_id"].astype("string")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("condition history has duplicate (race_id, horse_id) rows")
    history = history.sort_values(
        ["horse_id", "race_date", "race_id"], kind="stable"
    )
    grouped = history.groupby("horse_id", sort=False, dropna=False)
    for source, target in (
        ("race_date", "previous_eligible_race_date"),
        ("course_type", "previous_eligible_surface"),
        ("distance", "previous_eligible_distance"),
        ("context__venue_code", "previous_eligible_venue_code"),
    ):
        history[target] = grouped[source].shift(1)
    known = history["previous_eligible_race_date"].notna()
    if not (
        history.loc[known, "previous_eligible_race_date"]
        < history.loc[known, "race_date"]
    ).all():
        raise ValueError("previous-condition metadata is not strictly before target date")
    return history[
        [
            "race_id",
            "horse_id",
            "previous_eligible_race_date",
            "previous_eligible_surface",
            "previous_eligible_distance",
            "previous_eligible_venue_code",
        ]
    ]


def _race_slices(race_ids: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    structure = validate_grouped_rows(race_ids)
    starts = np.asarray([start for start, _ in structure.row_slices], dtype=np.int64)
    ends = np.asarray([end for _, end in structure.row_slices], dtype=np.int64)
    return starts, ends


def empirically_race_constant_columns(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> dict[str, bool]:
    """Identify features that are constant within every development race."""

    return {
        column: bool(
            frame.groupby("race_id", sort=False, observed=True)[column]
            .nunique(dropna=False)
            .le(1)
            .all()
        )
        for column in feature_columns
    }


def race_aware_group_permutation(
    matrix: np.ndarray,
    race_ids: Sequence[Any],
    *,
    varying_indices: Sequence[int],
    constant_indices: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Jointly permute one semantic group while retaining race structure.

    Runner-varying columns use one common row shuffle within each race, which
    preserves their within-race multiset and joint column relationships.
    Empirically race-constant columns are moved jointly as complete race-level
    vectors.  The two components are permuted separately for mixed groups, an
    unavoidable off-manifold limitation recorded in the output metadata.
    """

    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    starts, ends = _race_slices(race_ids)
    if len(race_ids) != len(values):
        raise ValueError("race_ids length disagrees with matrix")
    result = values.copy()

    varying = np.asarray(varying_indices, dtype=np.int64)
    if varying.size:
        row_source = np.arange(len(values), dtype=np.int64)
        for start, end in zip(starts, ends):
            row_source[start:end] = rng.permutation(row_source[start:end])
        result[:, varying] = values[row_source][:, varying]

    constant = np.asarray(constant_indices, dtype=np.int64)
    if constant.size:
        race_source = rng.permutation(len(starts))
        target_sizes = ends - starts
        source_rows = np.repeat(starts[race_source], target_sizes)
        result[:, constant] = values[source_rows][:, constant]
    return result


def _raw_scores(booster: Any, matrix: Any, model: str) -> np.ndarray:
    if model == "binary":
        return np.asarray(booster.predict(matrix, raw_score=True), dtype=float)
    if model == "lambdarank":
        return np.asarray(booster.predict(matrix), dtype=float)
    raise ValueError(f"unknown model: {model}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _flat_metrics(
    probabilities: Sequence[float],
    scores: Sequence[float],
    finish_positions: Sequence[int],
    race_ids: Sequence[Any],
) -> dict[str, float]:
    result = evaluate_predictions(
        probabilities,
        finish_positions,
        race_ids,
        ranking_scores=scores,
    )
    return {
        "ndcg_at_3": float(result["ranking"]["ndcg_at_3"]),
        "top_1": float(result["ranking"]["top_1"]),
        "race_log_loss": float(result["probability"]["race_log_loss"]),
        "race_brier": float(result["probability"]["race_brier"]),
        "race_balanced_ece": float(result["reliability"]["ece"]),
    }


def permutation_importance(
    *,
    boosters: Mapping[str, Any],
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    race_constant: Mapping[str, bool],
    race_ids: Sequence[Any],
    finish_positions: Sequence[int],
    temperatures: Mapping[str, float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Compute paired, race-aware semantic-group permutation importance."""

    if repeats < 1:
        raise ValueError("permutation repeats must be positive")
    column_index = {column: index for index, column in enumerate(feature_columns)}
    baseline: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for model, booster in boosters.items():
        raw = _raw_scores(booster, matrix, model)
        probabilities = race_softmax(
            raw, race_ids, temperature=float(temperatures[model])
        )
        baseline[model] = _flat_metrics(
            probabilities, raw, finish_positions, race_ids
        )

    for group_index, (group, columns) in enumerate(feature_groups.items()):
        varying = [column_index[c] for c in columns if not race_constant[c]]
        constant = [column_index[c] for c in columns if race_constant[c]]
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, group_index, repeat]))
            permuted = race_aware_group_permutation(
                matrix,
                race_ids,
                varying_indices=varying,
                constant_indices=constant,
                rng=rng,
            )
            for model, booster in boosters.items():
                raw = _raw_scores(booster, permuted, model)
                probabilities = race_softmax(
                    raw, race_ids, temperature=float(temperatures[model])
                )
                measured = _flat_metrics(
                    probabilities, raw, finish_positions, race_ids
                )
                for metric, permuted_value in measured.items():
                    baseline_value = baseline[model][metric]
                    higher_is_better = metric in {"ndcg_at_3", "top_1"}
                    degradation = (
                        baseline_value - permuted_value
                        if higher_is_better
                        else permuted_value - baseline_value
                    )
                    rows.append(
                        {
                            "model": model,
                            "group": group,
                            "repeat": repeat,
                            "metric": metric,
                            "higher_is_better": higher_is_better,
                            "baseline": baseline_value,
                            "permuted": permuted_value,
                            "degradation": degradation,
                            "runner_varying_feature_count": len(varying),
                            "race_constant_feature_count": len(constant),
                        }
                    )
    raw_table = pd.DataFrame(rows)
    summary = (
        raw_table.groupby(["model", "group", "metric"], sort=False, observed=True)
        .agg(
            repeats=("repeat", "count"),
            baseline=("baseline", "first"),
            mean_permuted=("permuted", "mean"),
            mean_degradation=("degradation", "mean"),
            std_degradation=("degradation", "std"),
            min_degradation=("degradation", "min"),
            max_degradation=("degradation", "max"),
            runner_varying_feature_count=("runner_varying_feature_count", "first"),
            race_constant_feature_count=("race_constant_feature_count", "first"),
        )
        .reset_index()
    )
    summary["std_degradation"] = summary["std_degradation"].fillna(0.0)
    return raw_table, summary, baseline


def _tree_importance(
    *,
    booster: Any,
    matrix: pd.DataFrame,
    raw_scores: np.ndarray,
    feature_columns: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    race_ids: Sequence[Any],
    race_constant: Mapping[str, bool],
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    contributions = booster.predict(matrix, pred_contrib=True)
    if hasattr(contributions, "toarray"):
        contributions = contributions.toarray()
    contributions = np.asarray(contributions, dtype=float)
    expected_shape = (len(matrix), len(feature_columns) + 1)
    if contributions.shape != expected_shape:
        raise ValueError(
            f"unexpected TreeSHAP shape {contributions.shape}; expected {expected_shape}"
        )
    feature_shap = contributions[:, :-1]
    reconstructed = contributions.sum(axis=1)
    additivity_error = float(np.max(np.abs(reconstructed - raw_scores)))
    if additivity_error > 1e-7:
        raise ValueError(f"TreeSHAP additivity error is too large: {additivity_error}")

    mean_abs = np.mean(np.abs(feature_shap), axis=0)
    mean_signed = np.mean(feature_shap, axis=0)
    centered_abs_sum = np.zeros(len(feature_columns), dtype=float)
    starts, ends = _race_slices(race_ids)
    for start, end in zip(starts, ends):
        block = feature_shap[start:end]
        centered_abs_sum += np.abs(block - block.mean(axis=0)).sum(axis=0)
    centered_mean_abs = centered_abs_sum / len(feature_shap)

    group_for = {
        column: group for group, columns in feature_groups.items() for column in columns
    }
    split_total = float(split.sum())
    gain_total = float(gain.sum())
    shap_total = float(mean_abs.sum())
    centered_total = float(centered_mean_abs.sum())
    feature_rows = []
    for index, column in enumerate(feature_columns):
        feature_rows.append(
            {
                "model": model,
                "feature": column,
                "group": group_for[column],
                "empirically_race_constant": race_constant[column],
                "built_in_split": split[index],
                "built_in_split_share": split[index] / split_total if split_total else 0.0,
                "built_in_gain": gain[index],
                "built_in_gain_share": gain[index] / gain_total if gain_total else 0.0,
                "shap_mean_signed_raw": mean_signed[index],
                "shap_mean_abs_raw": mean_abs[index],
                "shap_mean_abs_raw_share": mean_abs[index] / shap_total if shap_total else 0.0,
                "shap_race_centered_mean_abs_raw": centered_mean_abs[index],
                "shap_race_centered_share": (
                    centered_mean_abs[index] / centered_total if centered_total else 0.0
                ),
            }
        )
    feature_table = pd.DataFrame(feature_rows)

    group_rows = []
    column_index = {column: index for index, column in enumerate(feature_columns)}
    for group, columns in feature_groups.items():
        indices = np.asarray([column_index[column] for column in columns], dtype=int)
        joint = feature_shap[:, indices].sum(axis=1)
        joint_centered_sum = 0.0
        for start, end in zip(starts, ends):
            block = joint[start:end]
            joint_centered_sum += float(np.abs(block - block.mean()).sum())
        selected = feature_table.loc[feature_table["group"].eq(group)]
        group_rows.append(
            {
                "model": model,
                "group": group,
                "feature_count": len(columns),
                "race_constant_feature_count": int(
                    sum(bool(race_constant[column]) for column in columns)
                ),
                "built_in_split": float(selected["built_in_split"].sum()),
                "built_in_split_share": float(selected["built_in_split_share"].sum()),
                "built_in_gain": float(selected["built_in_gain"].sum()),
                "built_in_gain_share": float(selected["built_in_gain_share"].sum()),
                "shap_feature_mean_abs_sum": float(selected["shap_mean_abs_raw"].sum()),
                "shap_feature_mean_abs_share": float(
                    selected["shap_mean_abs_raw_share"].sum()
                ),
                "shap_joint_mean_abs_raw": float(np.mean(np.abs(joint))),
                "shap_race_centered_feature_mean_abs_sum": float(
                    selected["shap_race_centered_mean_abs_raw"].sum()
                ),
                "shap_race_centered_feature_share": float(
                    selected["shap_race_centered_share"].sum()
                ),
                "shap_joint_race_centered_mean_abs_raw": joint_centered_sum / len(joint),
            }
        )
    return (
        feature_table,
        pd.DataFrame(group_rows),
        {
            "max_abs_additivity_error": additivity_error,
            "expected_value_raw": float(np.mean(contributions[:, -1])),
            "row_count": int(len(matrix)),
        },
    )


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    ranked_left = left.loc[mask].rank(method="average").to_numpy(dtype=float)
    ranked_right = right.loc[mask].rank(method="average").to_numpy(dtype=float)
    if np.std(ranked_left) == 0.0 or np.std(ranked_right) == 0.0:
        return float("nan")
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def _pearson(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    x = left.loc[mask].to_numpy(dtype=float)
    y = right.loc[mask].to_numpy(dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _add_proxy_diagnostics(
    table: pd.DataFrame, development: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    date_ordinal = (
        pd.to_datetime(development["race_date"]) - pd.Timestamp("2024-01-01")
    ).dt.days.astype(float)
    winner = development["winner_label"].astype(float)
    proxy = pd.DataFrame(
        {
            "feature": feature_columns,
            "development_date_spearman": [
                _rank_correlation(development[column].astype(float), date_ordinal)
                for column in feature_columns
            ],
            "winner_label_pearson": [
                _pearson(development[column].astype(float), winner)
                for column in feature_columns
            ],
        }
    )
    return table.merge(proxy, on="feature", how="left", validate="many_to_one")


def _market_probabilities(odds: pd.Series, race_ids: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(odds, errors="raise").astype(float)
    if not np.isfinite(numeric).all() or numeric.lt(1.0).any():
        raise ValueError("final market odds must be finite decimal odds >= 1")
    inverse = 1.0 / numeric
    totals = inverse.groupby(race_ids, sort=False).transform("sum")
    return (inverse / totals).to_numpy(dtype=float)


def _condition_labels(frame: pd.DataFrame) -> dict[str, pd.Series]:
    venue = pd.to_numeric(frame["context__venue_code"], errors="coerce")
    class_tier = pd.to_numeric(frame["context__class_tier"], errors="coerce")
    dates = pd.to_datetime(frame["race_date"], errors="raise")
    return {
        "surface": frame["course_type"].astype("string").fillna("unknown"),
        "distance_band": frame["distance_band"].astype("string").fillna("unknown"),
        "race_class_tier": class_tier.map(
            lambda value: _CLASS_LABELS.get(int(value), "unknown")
            if pd.notna(value)
            else "unknown"
        ),
        "venue": venue.map(
            lambda value: _VENUE_LABELS.get(int(value), "unknown")
            if pd.notna(value)
            else "unknown"
        ),
        "field_size_band": frame["field_size_band"].astype("string").fillna("unknown"),
        "calendar_quarter": dates.dt.quarter.map(lambda value: f"Q{value}"),
    }


def conditional_race_table(
    frame: pd.DataFrame, model_columns: Mapping[str, tuple[str, str]]
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Evaluate only race-constant slices with race-macro metrics."""

    labels = _condition_labels(frame)
    rows: list[dict[str, Any]] = []
    overall: dict[str, dict[str, float]] = {}
    for model, (probability_column, score_column) in model_columns.items():
        overall[model] = _flat_metrics(
            frame[probability_column],
            frame[score_column],
            frame["model_finish_position"],
            frame["race_id"],
        )
        result = conditional_race_metrics(
            frame[probability_column],
            frame["model_finish_position"],
            frame["race_id"],
            labels,
            ranking_scores=frame[score_column],
        )
        for condition, levels in result.items():
            for level, metrics in levels.items():
                selected = labels[condition].astype(str).eq(level)
                row = {
                    "model": model,
                    "condition": condition,
                    "level": level,
                    "runner_count": int(selected.sum()),
                    "oracle_only": model == "final_odds_market",
                    **metrics,
                }
                row["race_count"] = int(row["race_count"])
                rows.append(row)
    return pd.DataFrame(rows), overall


def _runner_slice_labels(frame: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.to_numeric(frame["context__age"], errors="coerce")
    starts = pd.to_numeric(
        frame["horse_history__career__starts"], errors="coerce"
    ).fillna(0.0)
    rest = pd.to_numeric(
        frame["horse_history__days_since_last_start"], errors="coerce"
    )
    odds = pd.to_numeric(frame["final_win_odds_oracle"], errors="coerce")
    popularity = pd.to_numeric(frame["final_popularity_oracle"], errors="coerce")
    previous_known = frame["previous_eligible_race_date"].notna()
    distance_change = (
        pd.to_numeric(frame["distance"], errors="coerce")
        - pd.to_numeric(frame["previous_eligible_distance"], errors="coerce")
    ).abs()
    surface_change = (
        frame["course_type"].astype("string")
        != frame["previous_eligible_surface"].astype("string")
    ) & previous_known
    venue_change = (
        pd.to_numeric(frame["context__venue_code"], errors="coerce")
        != pd.to_numeric(frame["previous_eligible_venue_code"], errors="coerce")
    ) & previous_known
    distance_any_change = distance_change.gt(0) & previous_known
    change_count = (
        surface_change.astype(int)
        + venue_change.astype(int)
        + distance_any_change.astype(int)
    )

    def age_label(value: float) -> str:
        if pd.isna(value):
            return "unknown"
        if value <= 2:
            return "age_2"
        if value == 3:
            return "age_3"
        if value == 4:
            return "age_4"
        return "age_5_plus"

    def starts_label(value: float) -> str:
        if value <= 0:
            return "0_debut"
        if value <= 2:
            return "1_2"
        if value <= 5:
            return "3_5"
        if value <= 10:
            return "6_10"
        return "11_plus"

    def rest_label(value: float) -> str:
        if pd.isna(value):
            return "no_prior_history"
        if value <= 30:
            return "5_30_days"
        if value <= 90:
            return "31_90_days"
        if value <= 180:
            return "91_180_days"
        return "181_plus_days"

    def odds_label(value: float) -> str:
        if pd.isna(value):
            return "unknown"
        if value < 2:
            return "1_lt2"
        if value < 4:
            return "2_2to4"
        if value < 10:
            return "3_4to10"
        if value < 30:
            return "4_10to30"
        return "5_30plus"

    def popularity_label(value: float) -> str:
        if pd.isna(value):
            return "unknown"
        if value == 1:
            return "favorite_rank1"
        if value <= 5:
            return "middle_rank2_5"
        return "longshot_rank6_plus"

    distance_change_label = pd.Series("no_prior_eligible_race", index=frame.index)
    distance_change_label.loc[previous_known & distance_change.eq(0)] = "same_distance"
    distance_change_label.loc[previous_known & distance_change.gt(0) & distance_change.le(200)] = (
        "change_1_200m"
    )
    distance_change_label.loc[previous_known & distance_change.gt(200) & distance_change.lt(400)] = (
        "change_201_399m"
    )
    distance_change_label.loc[previous_known & distance_change.ge(400)] = "change_400m_plus"

    surface_change_label = pd.Series("no_prior_eligible_race", index=frame.index)
    surface_change_label.loc[previous_known & ~surface_change] = "same_surface"
    surface_change_label.loc[previous_known & surface_change] = "surface_changed"
    venue_change_label = pd.Series("no_prior_eligible_race", index=frame.index)
    venue_change_label.loc[previous_known & ~venue_change] = "same_venue"
    venue_change_label.loc[previous_known & venue_change] = "venue_changed"
    combined = pd.Series("no_prior_eligible_race", index=frame.index)
    combined.loc[previous_known & change_count.eq(0)] = "no_condition_change"
    combined.loc[previous_known & change_count.eq(1)] = "one_condition_change"
    combined.loc[previous_known & change_count.ge(2)] = "multiple_condition_changes"

    return {
        "runner_age": age.map(age_label),
        "final_odds_band": odds.map(odds_label),
        "final_popularity_tier": popularity.map(popularity_label),
        "career_start_band": starts.map(starts_label),
        "layoff_band": rest.map(rest_label),
        "surface_change_from_previous_eligible": surface_change_label,
        "distance_change_from_previous_eligible": distance_change_label,
        "venue_change_from_previous_eligible": venue_change_label,
        "combined_condition_change": combined,
    }


def runner_slice_table(
    frame: pd.DataFrame, model_columns: Mapping[str, tuple[str, str]]
) -> pd.DataFrame:
    """Compute runner-micro calibration/error and top-selection composition.

    Race log loss is intentionally absent: runner-specific categories split a
    race and therefore cannot be interpreted as coherent race populations.
    """

    labels = _runner_slice_labels(frame)
    targets = np.asarray(
        winner_mass_targets(frame["model_finish_position"], frame["race_id"]),
        dtype=float,
    )
    starts, ends = _race_slices(frame["race_id"])
    total_winner_mass = float(targets.sum())
    rows: list[dict[str, Any]] = []
    for model, (probability_column, score_column) in model_columns.items():
        probabilities = frame[probability_column].to_numpy(dtype=float)
        scores = frame[score_column].to_numpy(dtype=float)
        top_selected = np.zeros(len(frame), dtype=bool)
        for start, end in zip(starts, ends):
            top_selected[start + int(np.argmax(scores[start:end]))] = True
        for dimension, categories in labels.items():
            oracle_dimension = dimension in {"final_odds_band", "final_popularity_tier"}
            for category in sorted(categories.astype(str).unique()):
                selected = categories.astype(str).eq(category).to_numpy()
                count = int(selected.sum())
                if count == 0:
                    continue
                p = np.clip(probabilities[selected], 1e-15, 1.0 - 1e-15)
                y = targets[selected]
                top = selected & top_selected
                top_count = int(top.sum())
                rows.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "category": category,
                        "runner_count": count,
                        "race_count_represented": int(frame.loc[selected, "race_id"].nunique()),
                        "runner_share": count / len(frame),
                        "winner_mass": float(y.sum()),
                        "winner_mass_share": float(y.sum()) / total_winner_mass,
                        "observed_win_rate": float(y.mean()),
                        "mean_probability": float(p.mean()),
                        "calibration_gap_observed_minus_predicted": float(y.mean() - p.mean()),
                        "runner_micro_log_loss": float(
                            np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
                        ),
                        "runner_micro_brier": float(np.mean((p - y) ** 2)),
                        "top_selected_count": top_count,
                        "top_selected_share": top_count / len(starts),
                        "top_selected_winner_mass": float(targets[top].sum()),
                        "top_selected_hit_rate": (
                            float(targets[top].sum()) / top_count if top_count else float("nan")
                        ),
                        "oracle_only": bool(model == "final_odds_market" or oracle_dimension),
                    }
                )
    return pd.DataFrame(rows)


def _join_inputs(
    development: pd.DataFrame,
    predictions_path: Path,
    market_path: Path,
    prior: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["race_id", "horse_id"]
    development = development.copy()
    for column in keys:
        development[column] = development[column].astype("string")

    predictions = pd.read_csv(
        predictions_path, dtype={"race_id": "string", "horse_id": "string"}
    )
    prediction_dates = pd.to_datetime(predictions["race_date"], errors="raise")
    mislabeled = predictions["split"].eq("development") & ~prediction_dates.dt.year.eq(2024)
    if mislabeled.any():
        raise ValueError("prediction artifact has non-2024 development rows")
    predictions = predictions.loc[predictions["split"].eq("development")].copy()
    prediction_columns = keys + [
        "pred_binary_raw",
        "score_lambdarank",
        "prob_binary_logit_softmax_temperature_2023",
        "prob_lambdarank_softmax_temperature_2023",
    ]
    if predictions.duplicated(keys).any():
        raise ValueError("prediction artifact has duplicate runner keys")
    development = development.merge(
        predictions[prediction_columns], on=keys, how="left", sort=False, validate="one_to_one"
    )

    market = pd.read_csv(market_path, dtype={"race_id": "string", "horse_id": "string"})
    market_dates = pd.to_datetime(market["race_date"], errors="raise")
    mislabeled_market = market["split"].eq("development") & ~market_dates.dt.year.eq(2024)
    if mislabeled_market.any():
        raise ValueError("market artifact has non-2024 development rows")
    market = market.loc[market["split"].eq("development")].copy()
    if market.duplicated(keys).any():
        raise ValueError("market artifact has duplicate runner keys")
    market = market.rename(
        columns={
            "final_win_odds": "final_win_odds_oracle",
            "final_popularity": "final_popularity_oracle",
        }
    )
    development = development.merge(
        market[keys + ["final_win_odds_oracle", "final_popularity_oracle"]],
        on=keys,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    prior = prior.copy()
    for column in keys:
        prior[column] = prior[column].astype("string")
    development = development.merge(
        prior, on=keys, how="left", sort=False, validate="one_to_one"
    )
    required_joined = prediction_columns[2:] + [
        "final_win_odds_oracle",
        "final_popularity_oracle",
    ]
    if development[required_joined].isna().any().any():
        missing = development[required_joined].isna().sum()
        raise ValueError(f"diagnostic input join has missing values: {missing[missing.gt(0)].to_dict()}")
    validate_grouped_rows(development["race_id"])
    return development


def run_baseline_diagnostics(
    *,
    repo_root: str | Path,
    cache_path: str | Path,
    baseline_dir: str | Path,
    output_dir: str | Path,
    permutation_repeats: int = 5,
    seed: int = 20240830,
) -> dict[str, Any]:
    """Run the frozen 2024 baseline diagnostic suite and atomically save it."""

    root = Path(repo_root).resolve()
    cache = _resolve(root, cache_path)
    baseline = _resolve(root, baseline_dir)
    final_output = Path(output_dir).resolve()
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_output.with_name(f".{final_output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()

    try:
        run_git_state = git_state(root)
        cached_frame, cache_meta = read_model_frame_cache(cache)
        pre_2025, cache_isolation = isolate_pre_2025_frame(cached_frame)
        development, development_isolation = isolate_development_2024(cached_frame)
        del cached_frame
        prior = previous_condition_metadata(pre_2025)
        prior = prior.loc[
            prior["race_id"].isin(set(development["race_id"].astype("string")))
        ].copy()
        del pre_2025

        predictions_path = baseline / "predictions.csv.gz"
        market_path = baseline / "final_market_oracle.csv.gz"
        development = _join_inputs(development, predictions_path, market_path, prior)
        feature_columns = tuple(cache_meta["feature_columns"])
        taxonomy = semantic_feature_groups_v2(feature_columns)
        race_constant = empirically_race_constant_columns(development, feature_columns)
        matrix_frame = development.loc[:, feature_columns]
        matrix = matrix_frame.to_numpy(dtype=np.float32, copy=True)

        import lightgbm as lgb

        boosters = {
            model: lgb.Booster(model_file=str(baseline / spec["model_file"]))
            for model, spec in _MODEL_SPECS.items()
        }
        for model, booster in boosters.items():
            if tuple(booster.feature_name()) != feature_columns:
                raise ValueError(f"{model} feature names disagree with cache schema")
        baseline_metrics = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
        temperatures = {
            model: float(baseline_metrics["models"][spec["temperature_key"]]["temperature"])
            for model, spec in _MODEL_SPECS.items()
        }

        raw_scores: dict[str, np.ndarray] = {}
        probability_columns: dict[str, tuple[str, str]] = {}
        reproduction: dict[str, dict[str, float]] = {}
        for model, spec in _MODEL_SPECS.items():
            raw = _raw_scores(boosters[model], matrix_frame, model)
            probability = np.asarray(
                race_softmax(raw, development["race_id"], temperature=temperatures[model]),
                dtype=float,
            )
            raw_scores[model] = raw
            development[f"diagnostic_score_{model}"] = raw
            development[f"diagnostic_probability_{model}"] = probability
            artifact_raw = development[str(spec["artifact_raw"])].to_numpy(dtype=float)
            comparable_raw = _sigmoid(raw) if model == "binary" else raw
            artifact_probability = development[str(spec["artifact_probability"])].to_numpy(
                dtype=float
            )
            reproduction[model] = {
                "max_abs_raw_difference": float(np.max(np.abs(comparable_raw - artifact_raw))),
                "max_abs_calibrated_probability_difference": float(
                    np.max(np.abs(probability - artifact_probability))
                ),
            }
            if max(reproduction[model].values()) > 1e-9:
                raise ValueError(f"{model} predictions do not reproduce frozen artifact")
            probability_columns[model] = (
                f"diagnostic_probability_{model}",
                f"diagnostic_score_{model}",
            )

        market_probability = _market_probabilities(
            development["final_win_odds_oracle"], development["race_id"]
        )
        development["diagnostic_probability_final_market"] = market_probability
        development["diagnostic_score_final_market"] = market_probability
        probability_columns["final_odds_market"] = (
            "diagnostic_probability_final_market",
            "diagnostic_score_final_market",
        )

        feature_tables: list[pd.DataFrame] = []
        group_tables: list[pd.DataFrame] = []
        shap_checks: dict[str, dict[str, float]] = {}
        for model in _MODEL_SPECS:
            feature_table, group_table, shap_check = _tree_importance(
                booster=boosters[model],
                matrix=matrix_frame,
                raw_scores=raw_scores[model],
                feature_columns=feature_columns,
                feature_groups=taxonomy,
                race_ids=development["race_id"],
                race_constant=race_constant,
                model=model,
            )
            feature_tables.append(feature_table)
            group_tables.append(group_table)
            shap_checks[model] = shap_check
        feature_importance = _add_proxy_diagnostics(
            pd.concat(feature_tables, ignore_index=True), development, feature_columns
        )
        group_importance = pd.concat(group_tables, ignore_index=True)

        permutation_raw, permutation_summary, permutation_baseline = permutation_importance(
            boosters=boosters,
            matrix=matrix,
            feature_columns=feature_columns,
            feature_groups=taxonomy,
            race_constant=race_constant,
            race_ids=development["race_id"].tolist(),
            finish_positions=development["model_finish_position"].astype(int).tolist(),
            temperatures=temperatures,
            repeats=permutation_repeats,
            seed=seed,
        )
        conditional, overall = conditional_race_table(development, probability_columns)
        runner_slices = runner_slice_table(development, probability_columns)

        feature_importance.to_csv(temporary / "feature_importance.csv", index=False)
        group_importance.to_csv(temporary / "group_importance.csv", index=False)
        permutation_raw.to_csv(temporary / "permutation_repeats.csv", index=False)
        permutation_summary.to_csv(temporary / "permutation_summary.csv", index=False)
        conditional.to_csv(temporary / "conditional_race_metrics.csv", index=False)
        runner_slices.to_csv(temporary / "runner_slice_metrics.csv", index=False)

        forbidden_hits = sorted(
            column
            for column in feature_columns
            if any(fragment in column.lower() for fragment in _FORBIDDEN_NAME_FRAGMENTS)
        )
        top_date_proxy = (
            feature_importance[["feature", "development_date_spearman"]]
            .drop_duplicates("feature")
            .assign(
                absolute=lambda table: table["development_date_spearman"].abs()
            )
            .sort_values("absolute", ascending=False)
            .head(20)
            .drop(columns="absolute")
            .to_dict("records")
        )
        top_target_correlation = (
            feature_importance[["feature", "winner_label_pearson"]]
            .drop_duplicates("feature")
            .assign(absolute=lambda table: table["winner_label_pearson"].abs())
            .sort_values("absolute", ascending=False)
            .head(20)
            .drop(columns="absolute")
            .to_dict("records")
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": run_git_state,
            "scope": {
                "evaluation": "2024 development only",
                "development_rows": int(len(development)),
                "development_races": int(development["race_id"].nunique()),
                "development_dates": int(development["race_date"].nunique()),
                "date_start": development["race_date"].min().date().isoformat(),
                "date_end": development["race_date"].max().date().isoformat(),
                "retrospective_test_used": False,
                "retrospective_2025_rows_used": 0,
                "odds_in_prediction_features": False,
                "final_odds_usage": "post-event oracle diagnosis only",
            },
            "inputs": {
                "cache": str(cache),
                "baseline_artifact": str(baseline),
                "data_fingerprint": cache_meta["data_fingerprint"],
                "feature_count": len(feature_columns),
                "cache_isolation": cache_isolation,
                "development_isolation": development_isolation,
                "prediction_reproduction": reproduction,
            },
            "tree_importance": {
                "built_in": ["split", "gain"],
                "tree_shap_model_output": "raw score",
                "tree_shap_rows": int(len(development)),
                "race_centering": (
                    "subtract each feature's within-race mean SHAP before mean absolute aggregation"
                ),
                "additivity_checks": shap_checks,
            },
            "permutation": {
                "seed": seed,
                "repeats": permutation_repeats,
                "method": (
                    "joint within-race row shuffle for runner-varying columns; joint race-block "
                    "shuffle for empirically race-constant columns"
                ),
                "importance_sign": (
                    "positive means worse after permutation: baseline-permuted for ranking; "
                    "permuted-baseline for losses/ECE"
                ),
                "limitations": [
                    "Permutation importance measures dependence of this fitted model, not causal feature value.",
                    "Mixed groups permute race-constant and runner-varying components "
                    "separately and can create off-manifold combinations.",
                    "Correlated groups can substitute for one another, understating conditional importance.",
                    "Race-constant features can still moderate runner effects through tree interactions.",
                ],
                "baseline_metrics": permutation_baseline,
            },
            "conditional_error_analysis": {
                "race_slices": (
                    "race-constant conditions use coherent race-macro ranking, probability, and calibration metrics"
                ),
                "runner_slices": (
                    "runner-specific conditions use runner-micro log loss/Brier, calibration gap, "
                    "and model top-selection composition; race log loss is intentionally not reported"
                ),
                "previous_condition_semantics": (
                    "previous eligible cached flat race strictly before the target date; may skip "
                    "unavailable or PIT-C-excluded official starts"
                ),
                "final_odds_semantics": (
                    "all final-odds/popularity slices and the market model are post-event oracle-only"
                ),
                "overall_metrics": overall,
            },
            "leak_proxy_audit": {
                "forbidden_feature_name_hits": forbidden_hits,
                "top_development_date_correlations": top_date_proxy,
                "top_winner_label_correlations": top_target_correlation,
                "interpretation": (
                    "Correlation is a screening diagnostic, not proof of leakage; calendar-correlated "
                    "race context can reflect legitimate JRA seasonality. Suspicious candidates require knockout tests."
                ),
            },
            "files": {
                "feature_importance": "feature_importance.csv",
                "group_importance": "group_importance.csv",
                "permutation_repeats": "permutation_repeats.csv",
                "permutation_summary": "permutation_summary.csv",
                "conditional_race_metrics": "conditional_race_metrics.csv",
                "runner_slice_metrics": "runner_slice_metrics.csv",
            },
        }
        write_json(temporary / "diagnostics.json", payload)
        write_artifact_manifest(temporary)
        temporary.replace(final_output)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
