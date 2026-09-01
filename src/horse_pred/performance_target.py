"""Fold-scoped condition-adjusted continuous targets for the S3 study.

The target normalizer is fitted once on pooled clean races from the model
training years and then frozen.  Outcome rows from early stopping,
calibration, and evaluation may be transformed into labels or diagnostics,
but they can never update the normalizer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.config import canonical_json_hash
from horse_pred.features import FeatureConfig, is_flat_race
from horse_pred.speed_figure import (
    _clean_race_times,
    _solve_ridge,
    condition_design_vector,
)

PERFORMANCE_TARGET_COLUMN = "target__condition_adjusted_performance"

_CONDITION_DIMENSION = 51
_MAXIMUM_OUTCOME_YEAR = 2022


@dataclass(frozen=True)
class PerformanceTargetSpec:
    """Frozen S3 target-transformation parameters."""

    ridge_alpha: float = 1.0
    observation_clip: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if not np.isfinite(self.observation_clip) or self.observation_clip <= 0:
            raise ValueError("observation_clip must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_column": PERFORMANCE_TARGET_COLUMN,
            "raw_value": (
                "expected_winner_seconds_per_1000m_minus_"
                "runner_seconds_per_1000m"
            ),
            "higher_is_better": True,
            "ridge_alpha": self.ridge_alpha,
            "observation_clip": [-self.observation_clip, self.observation_clip],
            "normalizer": "pooled_fold_train_2014_to_train_end_then_frozen",
            "condition_effects": [
                "course_surface",
                "exact_distance",
                "going",
                "class_tier",
                "age_restriction",
            ],
            "condition_dimension": _CONDITION_DIMENSION,
            "season_included": False,
            "field_size_included": False,
            "track_day_variant_included": False,
            "missing_target_fit_policy": "exclude_without_zero_imputation",
        }


# Descriptive public name used by the S3 runner and tests.  Keep the shorter
# alias for callers that already imported the implementation draft.
ConditionAdjustedPerformanceTargetSpec = PerformanceTargetSpec


def _validated_fold(fold: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id",
        "train_start_year",
        "train_end_year",
        "early_stopping_year",
        "calibration_year",
        "evaluation_year",
    }
    missing = sorted(required.difference(fold))
    if missing:
        raise ValueError(f"S3 fold is missing fields: {missing}")
    result = {"id": str(fold["id"])}
    for name in sorted(required.difference({"id"})):
        result[name] = int(fold[name])
    if result["train_start_year"] != 2014:
        raise ValueError("S3 model train must start in 2014")
    ordered = (
        result["train_end_year"],
        result["early_stopping_year"],
        result["calibration_year"],
        result["evaluation_year"],
    )
    if not all(left < right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("S3 fold roles must be strictly chronological")
    if result["train_start_year"] > result["train_end_year"]:
        raise ValueError("S3 fold train interval is empty")
    if result["evaluation_year"] > _MAXIMUM_OUTCOME_YEAR:
        raise ValueError("S3 evaluation year must not exceed 2022")
    return result


def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "venue_code",
        "course_type",
        "distance",
        "ground_state",
        "race_class",
        "status",
        "started",
        "finish_position",
        "time_raw",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"S3 normalized input is missing columns: {missing}")


def _beta_hash(beta: np.ndarray) -> str:
    canonical = np.asarray(beta, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _role_for_year(year: int, fold: Mapping[str, Any]) -> str | None:
    if int(fold["train_start_year"]) <= year <= int(fold["train_end_year"]):
        return "train"
    if year == int(fold["early_stopping_year"]):
        return "model_validation"
    if year == int(fold["calibration_year"]):
        return "calibration"
    if year == int(fold["evaluation_year"]):
        return "evaluation"
    return None


def _race_failure_reason(race: pd.DataFrame, design: np.ndarray | None) -> str:
    if race["status"].astype("string").isin(("demoted", "disqualified")).any():
        return "demotion_or_disqualification"
    if design is None:
        return "unknown_condition"
    return "invalid_clock_distance_or_winner"


def build_fold_performance_targets(
    normalized: pd.DataFrame,
    fold: Mapping[str, Any],
    spec: PerformanceTargetSpec | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build S3 targets using a pooled, fold-train-only condition normalizer.

    The returned table contains only join metadata and the target.  Horse IDs
    remain join keys and are never exposed as model features.
    """

    spec = spec or PerformanceTargetSpec()
    fold_spec = _validated_fold(fold)
    _required_columns(normalized)

    dates = pd.to_datetime(normalized["race_date"], errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("S3 input contains a missing race date")
    if dates.dt.year.ge(2023).any():
        raise ValueError("S3 input must not contain 2023+ rows")

    evaluation_year = int(fold_spec["evaluation_year"])
    source = normalized.copy()
    source["_event_date"] = dates
    source["_source_position"] = np.arange(len(source), dtype=np.int64)
    rows_after_evaluation = int(source["_event_date"].dt.year.gt(evaluation_year).sum())
    source = source.loc[source["_event_date"].dt.year.le(evaluation_year)].copy()
    source["race_id"] = source["race_id"].astype("string")
    source["horse_id"] = source["horse_id"].astype("string")

    feature_config = FeatureConfig()
    flat_ids = [
        str(race_id)
        for race_id, race in source.groupby("race_id", sort=True)
        if is_flat_race(race, feature_config)
    ]
    flat = source.loc[source["race_id"].astype(str).isin(flat_ids)].copy()
    flat = flat.sort_values(
        ["_event_date", "race_id", "_source_position"], kind="stable"
    )
    starters = flat.loc[flat["started"].eq(True).fillna(False)].copy()  # noqa: E712
    if starters["horse_id"].isna().any():
        raise ValueError("S3 starter horse_id must not be missing")
    if starters.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("S3 input contains duplicate starter keys")

    train_start = int(fold_spec["train_start_year"])
    train_end = int(fold_spec["train_end_year"])
    xtx = np.zeros((_CONDITION_DIMENSION, _CONDITION_DIMENSION), dtype=np.float64)
    xty = np.zeros(_CONDITION_DIMENSION, dtype=np.float64)
    fit_count = 0
    fit_year_counts: dict[int, int] = {}
    fit_dates: list[pd.Timestamp] = []

    for (event_date, _), race in flat.groupby(
        ["_event_date", "race_id"], sort=True, observed=True
    ):
        year = int(pd.Timestamp(event_date).year)
        if year < train_start or year > train_end:
            continue
        design = condition_design_vector(race)
        clean = _clean_race_times(race)
        if design is None or clean is None:
            continue
        xtx += np.outer(design, design)
        xty += design * float(clean[0])
        fit_count += 1
        fit_year_counts[year] = fit_year_counts.get(year, 0) + 1
        fit_dates.append(pd.Timestamp(event_date).normalize())

    if fit_count == 0:
        raise ValueError("S3 fold train contains no clean races for the normalizer")
    beta = _solve_ridge(xtx, xty, spec.ridge_alpha)
    if beta.shape != (_CONDITION_DIMENSION,) or not np.isfinite(beta).all():
        raise AssertionError("S3 condition normalizer produced invalid coefficients")

    emitted: list[dict[str, Any]] = []
    missing_reasons: dict[str, int] = {}
    invalid_race_reasons: dict[str, int] = {}
    finite_by_year: dict[int, int] = {}
    missing_by_year: dict[int, int] = {}
    finite_by_role: dict[str, int] = {}
    missing_by_role: dict[str, int] = {}
    lower_clipped = 0
    upper_clipped = 0
    raw_min = np.inf
    raw_max = -np.inf
    finite_values: list[float] = []

    for (event_date, race_id), race in flat.groupby(
        ["_event_date", "race_id"], sort=True, observed=True
    ):
        event_date = pd.Timestamp(event_date).normalize()
        year = int(event_date.year)
        role = _role_for_year(year, fold_spec)
        race_starters = race.loc[race["started"].eq(True).fillna(False)].copy()  # noqa: E712
        if race_starters.empty:
            continue
        design = condition_design_vector(race)
        clean = _clean_race_times(race)
        target_by_index: dict[int, float] = {}
        if design is not None and clean is not None:
            expected = float(design @ beta)
            for index, runner_clock in clean[1].items():
                raw = expected - float(runner_clock)
                raw_min = min(raw_min, raw)
                raw_max = max(raw_max, raw)
                if raw < -spec.observation_clip:
                    lower_clipped += 1
                elif raw > spec.observation_clip:
                    upper_clipped += 1
                target_by_index[int(index)] = float(
                    np.clip(raw, -spec.observation_clip, spec.observation_clip)
                )
        else:
            reason = _race_failure_reason(race, design)
            invalid_race_reasons[reason] = invalid_race_reasons.get(reason, 0) + 1

        for index, runner in race_starters.iterrows():
            target = target_by_index.get(int(index), np.nan)
            emitted.append(
                {
                    "_source_position": int(runner["_source_position"]),
                    "race_id": str(race_id),
                    "horse_id": str(runner["horse_id"]),
                    "race_date": event_date,
                    PERFORMANCE_TARGET_COLUMN: target,
                }
            )
            bucket = role or "outside_fold_roles"
            if np.isfinite(target):
                finite_values.append(float(target))
                finite_by_year[year] = finite_by_year.get(year, 0) + 1
                finite_by_role[bucket] = finite_by_role.get(bucket, 0) + 1
            else:
                missing_by_year[year] = missing_by_year.get(year, 0) + 1
                missing_by_role[bucket] = missing_by_role.get(bucket, 0) + 1
                if design is None or clean is None:
                    reason = _race_failure_reason(race, design)
                else:
                    status = str(runner["status"])
                    reason = (
                        "did_not_finish"
                        if status == "did_not_finish"
                        else "runner_clock_or_finish_missing"
                    )
                missing_reasons[reason] = missing_reasons.get(reason, 0) + 1

    columns = [
        "race_id",
        "horse_id",
        "race_date",
        PERFORMANCE_TARGET_COLUMN,
    ]
    if emitted:
        targets = pd.DataFrame(emitted).sort_values("_source_position", kind="stable")
        targets = targets.loc[:, ["_source_position", *columns]].drop(
            columns="_source_position"
        )
        targets[PERFORMANCE_TARGET_COLUMN] = pd.to_numeric(
            targets[PERFORMANCE_TARGET_COLUMN], errors="coerce"
        ).astype("float32")
        target_dates = pd.to_datetime(targets["race_date"], errors="raise")
    else:
        targets = pd.DataFrame(columns=columns)
        target_dates = pd.Series(dtype="datetime64[ns]")

    finite_count = len(finite_values)
    missing_count = int(len(targets) - finite_count)
    total_count = int(len(targets))
    clipped_count = lower_clipped + upper_clipped
    normalizer_fit = {
        "scope": "pooled 2014 through fold train end; frozen for all roles",
        "clean_race_count": fit_count,
        "fit_race_count_by_year": {
            str(year): count for year, count in sorted(fit_year_counts.items())
        },
        "min_fit_date": str(min(fit_dates).date()) if fit_dates else None,
        "max_fit_date": str(max(fit_dates).date()) if fit_dates else None,
        "warmup_2013_rows_used": 0,
        "early_stopping_rows_used": 0,
        "calibration_rows_used": 0,
        "evaluation_rows_used": 0,
        "frozen_beta": [float(value) for value in beta],
        "frozen_beta_sha256": _beta_hash(beta),
    }
    audit = {
        "schema_version": 1,
        "fold": fold_spec,
        "spec": spec.as_dict(),
        "transformation_hash": canonical_json_hash(
            {"fold": fold_spec, "spec": spec.as_dict()}
        ),
        "scope": {
            "input_rows": int(len(normalized)),
            "input_max_date": str(dates.max().date()) if len(dates) else None,
            "rows_after_fold_evaluation_ignored": rows_after_evaluation,
            "processed_starter_rows": total_count,
            "processed_races": int(targets["race_id"].nunique()),
            "processed_min_date": (
                str(target_dates.min().date()) if len(target_dates) else None
            ),
            "processed_max_date": (
                str(target_dates.max().date()) if len(target_dates) else None
            ),
            "rows_used_2023": 0,
            "rows_used_2024": 0,
            "rows_used_2025": 0,
            "rows_used_2023_plus": 0,
            "odds_used": False,
            "direct_entity_id_feature_count": 0,
            "direct_entity_id_features": 0,
        },
        "normalizer_fit": normalizer_fit,
        "condition_fit": normalizer_fit,
        "target": {
            "nonmissing": finite_count,
            "missing": missing_count,
            "coverage": float(finite_count / total_count) if total_count else 0.0,
            "missing_imputed_to_zero": False,
            "nonmissing_by_year": {
                str(year): count for year, count in sorted(finite_by_year.items())
            },
            "missing_by_year": {
                str(year): count for year, count in sorted(missing_by_year.items())
            },
            "nonmissing_by_role": dict(sorted(finite_by_role.items())),
            "missing_by_role": dict(sorted(missing_by_role.items())),
            "missing_reasons": dict(sorted(missing_reasons.items())),
            "invalid_race_reasons": dict(sorted(invalid_race_reasons.items())),
            "raw_min": float(raw_min) if finite_count else None,
            "raw_max": float(raw_max) if finite_count else None,
            "clipped_lower": lower_clipped,
            "clipped_upper": upper_clipped,
            "clipped_total": clipped_count,
            "clip_rate": float(clipped_count / finite_count) if finite_count else 0.0,
            "minimum": float(np.min(finite_values)) if finite_count else None,
            "maximum": float(np.max(finite_values)) if finite_count else None,
            "mean": float(np.mean(finite_values)) if finite_count else None,
            "standard_deviation": (
                float(np.std(finite_values, ddof=0)) if finite_count else None
            ),
        },
    }
    return targets.reset_index(drop=True), audit
