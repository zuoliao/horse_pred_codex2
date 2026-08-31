"""Fold-scoped PIT histories for the S1 two-axis past-race value study.

The performance axis measures how fast a horse ran relative to a condition
model.  The field-quality axis is the arithmetic mean of every starter's
pre-race global Elo.  Both observations become visible only on later dates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.config import canonical_json_hash
from horse_pred.features import FeatureConfig, is_flat_race
from horse_pred.rating import RatingEvent, RatingSpec, build_rating_history_from_events
from horse_pred.speed_figure import (
    _clean_race_times,
    _solve_ridge,
    condition_design_vector,
)

PERFORMANCE_COLUMN = "race_value__decay_90d__performance_residual"
FIELD_QUALITY_COLUMN = "race_value__decay_90d__field_quality"

_MAXIMUM_OUTCOME_YEAR = 2022
_CONDITION_DIMENSION = 51


@dataclass(frozen=True)
class TwoAxisRaceValueSpec:
    """Frozen S1 transformation parameters."""

    ridge_alpha: float = 1.0
    min_prior_clean_races: int = 510
    observation_clip: float = 5.0
    decay_half_life_days: int = 90
    initial_rating: float = 1500.0
    elo_k: float = 24.0
    elo_scale: float = 400.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if self.min_prior_clean_races <= 0:
            raise ValueError("min_prior_clean_races must be positive")
        if not np.isfinite(self.observation_clip) or self.observation_clip <= 0:
            raise ValueError("observation_clip must be finite and positive")
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")
        if not np.isfinite(self.initial_rating):
            raise ValueError("initial_rating must be finite")
        if not np.isfinite(self.elo_k) or self.elo_k <= 0:
            raise ValueError("elo_k must be finite and positive")
        if not np.isfinite(self.elo_scale) or self.elo_scale <= 0:
            raise ValueError("elo_scale must be finite and positive")

    def rating_spec(self) -> RatingSpec:
        return RatingSpec(
            family="pairwise_elo",
            initial_rating=self.initial_rating,
            k=self.elo_k,
            scale=self.elo_scale,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ridge_alpha": self.ridge_alpha,
            "min_prior_clean_races": self.min_prior_clean_races,
            "observation_clip": self.observation_clip,
            "decay_half_life_days": self.decay_half_life_days,
            "condition_effects": [
                "course_surface",
                "exact_distance",
                "going",
                "class_tier",
                "age_restriction",
            ],
            "season_included": False,
            "field_size_included": False,
            "rating": {
                "family": "pairwise_elo",
                "initial_rating": self.initial_rating,
                "k": self.elo_k,
                "scale": self.elo_scale,
                "pairwise_actual": "ordinal",
            },
            "performance_column": PERFORMANCE_COLUMN,
            "field_quality_column": FIELD_QUALITY_COLUMN,
        }


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
        raise ValueError(f"S1 fold is missing fields: {missing}")
    result = {"id": str(fold["id"])}
    for name in sorted(required.difference({"id"})):
        result[name] = int(fold[name])
    if result["train_start_year"] < 2014:
        raise ValueError("S1 model train must start in 2014 or later")
    ordered = (
        result["train_end_year"],
        result["early_stopping_year"],
        result["calibration_year"],
        result["evaluation_year"],
    )
    if not all(left < right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("S1 fold roles must be strictly chronological")
    if result["train_start_year"] > result["train_end_year"]:
        raise ValueError("S1 fold train interval is empty")
    if result["evaluation_year"] > _MAXIMUM_OUTCOME_YEAR:
        raise ValueError("S1 evaluation year must not exceed 2022")
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
        raise ValueError(f"S1 normalized input is missing columns: {missing}")


def _rating_events(flat: pd.DataFrame) -> tuple[RatingEvent, ...]:
    events: list[RatingEvent] = []
    for (event_date, race_id), race in flat.groupby(
        ["_event_date", "race_id"], sort=True
    ):
        starters = race.loc[race["started"].eq(True).fillna(False)].copy()  # noqa: E712
        if starters.empty:
            continue
        if starters["horse_id"].isna().any():
            raise ValueError("S1 starter horse_id must not be missing")
        if starters["horse_id"].duplicated().any():
            raise ValueError("S1 race contains duplicate starter horse IDs")
        finishes = pd.to_numeric(starters["finish_position"], errors="coerce")
        surfaces = starters["course_type"].dropna().astype(str).unique()
        surface_key = surfaces[0] if len(surfaces) == 1 else None
        events.append(
            RatingEvent(
                race_id=str(race_id),
                race_date=pd.Timestamp(event_date).normalize(),
                surface_key=surface_key,
                horse_ids=tuple(starters["horse_id"].astype(str)),
                finishes=tuple(
                    float(value) if pd.notna(value) else np.nan
                    for value in finishes
                ),
                source_positions=tuple(
                    int(value) for value in starters["_source_position"]
                ),
            )
        )
    return tuple(events)


def _history_value(
    state: dict[str, tuple[float, float, pd.Timestamp]],
    horse_id: str,
    event_date: pd.Timestamp,
    decay_rate: float,
) -> float:
    record = state.get(horse_id)
    if record is None:
        return np.nan
    total, weight, state_date = record
    factor = exp(-decay_rate * (event_date - state_date).days)
    decayed_weight = weight * factor
    return total * factor / decayed_weight if decayed_weight > 0 else np.nan


def _update_history(
    state: dict[str, tuple[float, float, pd.Timestamp]],
    horse_id: str,
    observation: float,
    event_date: pd.Timestamp,
    decay_rate: float,
) -> None:
    record = state.get(horse_id)
    if record is None:
        total = weight = 0.0
    else:
        total, weight, state_date = record
        factor = exp(-decay_rate * (event_date - state_date).days)
        total *= factor
        weight *= factor
    state[horse_id] = (total + observation, weight + 1.0, event_date)


def _beta_hash(beta: np.ndarray | None) -> str | None:
    if beta is None:
        return None
    canonical = np.asarray(beta, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_fold_two_axis_history(
    normalized: pd.DataFrame,
    fold: Mapping[str, Any],
    spec: TwoAxisRaceValueSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one fold's two PIT-safe histories and private observation audit.

    The condition model is prequential through the fold's train end and then
    frozen.  The ordinal rating remains a fixed causal state algorithm through
    evaluation.  Rows after the evaluation year are ignored, while any input
    from 2023 or later is rejected before processing.
    """

    spec = spec or TwoAxisRaceValueSpec()
    fold_spec = _validated_fold(fold)
    _required_columns(normalized)

    dates = pd.to_datetime(normalized["race_date"], errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("S1 input contains a missing race date")
    if dates.dt.year.ge(2023).any():
        raise ValueError("S1 input must not contain 2023+ rows")

    evaluation_year = int(fold_spec["evaluation_year"])
    source = normalized.loc[dates.dt.year.le(evaluation_year)].copy()
    source = source.reset_index(drop=True)
    source["_event_date"] = pd.to_datetime(
        source["race_date"], errors="raise"
    ).dt.normalize()
    source["_source_position"] = np.arange(len(source), dtype=np.int64)
    source["race_id"] = source["race_id"].astype("string")
    source["horse_id"] = source["horse_id"].astype("string")
    if source.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("S1 input contains duplicate runner keys")
    source = source.sort_values(
        ["_event_date", "race_id", "_source_position"], kind="stable"
    )

    feature_config = FeatureConfig()
    flat_ids = [
        str(race_id)
        for race_id, race in source.groupby("race_id", sort=False)
        if is_flat_race(race, feature_config)
    ]
    flat = source.loc[source["race_id"].astype(str).isin(flat_ids)].copy()
    starters = flat.loc[flat["started"].eq(True).fillna(False)].copy()  # noqa: E712

    history_columns = [
        "race_id",
        "horse_id",
        "race_date",
        PERFORMANCE_COLUMN,
        FIELD_QUALITY_COLUMN,
    ]
    observation_columns = [
        "source_position",
        "race_id",
        "horse_id",
        "race_date",
        "finish_position",
        "status",
        "global_rating_pre",
        "field_quality_observation",
        "field_starter_count",
        "expected_winner_seconds_per_1000m",
        "runner_seconds_per_1000m",
        "performance_residual_observation",
        "condition_model_phase",
        "condition_fit_races_before",
    ]
    if starters.empty:
        empty_history = pd.DataFrame(columns=history_columns)
        empty_observations = pd.DataFrame(columns=observation_columns)
        return empty_history, empty_observations, {
            "schema_version": 1,
            "fold": fold_spec,
            "spec": spec.as_dict(),
            "scope": {
                "input_rows": int(len(normalized)),
                "processed_rows": 0,
                "processed_races": 0,
                "rows_used_2023_plus": 0,
            },
            "condition_fit": {
                "clean_race_count": 0,
                "max_fit_date": None,
                "frozen_beta": None,
                "frozen_beta_sha256": None,
            },
            "field_quality": {"race_constant_violations": 0},
        }

    events = _rating_events(flat)
    ratings = build_rating_history_from_events(events, spec.rating_spec())
    if ratings.empty or len(ratings) != len(starters):
        raise AssertionError("S1 rating history does not match flat starters")
    rating_by_position = ratings.set_index("source_position", verify_integrity=True)

    xtx = np.zeros((_CONDITION_DIMENSION, _CONDITION_DIMENSION), dtype=np.float64)
    xty = np.zeros(_CONDITION_DIMENSION, dtype=np.float64)
    clean_race_count = 0
    frozen_beta: np.ndarray | None = None
    max_fit_date: pd.Timestamp | None = None
    fit_year_counts: dict[int, int] = {}
    decay_rate = log(2.0) / spec.decay_half_life_days
    performance_state: dict[str, tuple[float, float, pd.Timestamp]] = {}
    field_state: dict[str, tuple[float, float, pd.Timestamp]] = {}
    emitted: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    train_start = int(fold_spec["train_start_year"])
    train_end = int(fold_spec["train_end_year"])
    for event_date, day in flat.groupby("_event_date", sort=True):
        event_date = pd.Timestamp(event_date).normalize()
        day_starters = day.loc[day["started"].eq(True).fillna(False)].copy()  # noqa: E712
        for _, row in day_starters.iterrows():
            horse_id = str(row["horse_id"])
            emitted.append(
                {
                    "_source_position": int(row["_source_position"]),
                    "race_id": str(row["race_id"]),
                    "horse_id": horse_id,
                    "race_date": event_date,
                    PERFORMANCE_COLUMN: _history_value(
                        performance_state, horse_id, event_date, decay_rate
                    ),
                    FIELD_QUALITY_COLUMN: _history_value(
                        field_state, horse_id, event_date, decay_rate
                    ),
                }
            )

        year = int(event_date.year)
        if year <= train_end:
            beta_for_date = (
                _solve_ridge(xtx, xty, spec.ridge_alpha)
                if clean_race_count >= spec.min_prior_clean_races
                else None
            )
            phase = "warmup_prequential" if year < train_start else "train_prequential"
        else:
            beta_for_date = frozen_beta
            phase = "frozen_after_train"

        pending_performance: list[tuple[str, float]] = []
        pending_field: list[tuple[str, float]] = []
        condition_updates: list[tuple[np.ndarray, float]] = []
        for race_id, race in day.groupby("race_id", sort=False):
            race_starters = race.loc[
                race["started"].eq(True).fillna(False)  # noqa: E712
            ].copy()
            if race_starters.empty:
                continue
            positions = race_starters["_source_position"].astype(int).to_numpy()
            race_ratings = rating_by_position.loc[positions]
            field_quality = float(
                pd.to_numeric(
                    race_ratings["global_state_pre"], errors="raise"
                ).mean()
            )
            field_count = int(len(race_starters))

            design = condition_design_vector(race)
            clean = _clean_race_times(race)
            expected = (
                float(design @ beta_for_date)
                if design is not None and clean is not None and beta_for_date is not None
                else np.nan
            )
            runner_clocks = clean[1] if clean is not None else pd.Series(dtype=float)
            performance_by_index: dict[int, float] = {}
            if np.isfinite(expected):
                for index, runner_clock in runner_clocks.items():
                    performance_by_index[int(index)] = float(
                        np.clip(
                            expected - float(runner_clock),
                            -spec.observation_clip,
                            spec.observation_clip,
                        )
                    )

            for index, runner in race_starters.iterrows():
                source_position = int(runner["_source_position"])
                rating_pre = float(
                    rating_by_position.at[source_position, "global_state_pre"]
                )
                horse_id = str(runner["horse_id"])
                performance = performance_by_index.get(int(index), np.nan)
                runner_clock = (
                    float(runner_clocks.at[index]) if index in runner_clocks.index else np.nan
                )
                observations.append(
                    {
                        "source_position": source_position,
                        "race_id": str(race_id),
                        "horse_id": horse_id,
                        "race_date": event_date,
                        "finish_position": runner["finish_position"],
                        "status": str(runner["status"]),
                        "global_rating_pre": rating_pre,
                        "field_quality_observation": field_quality,
                        "field_starter_count": field_count,
                        "expected_winner_seconds_per_1000m": expected,
                        "runner_seconds_per_1000m": runner_clock,
                        "performance_residual_observation": performance,
                        "condition_model_phase": phase,
                        "condition_fit_races_before": clean_race_count,
                    }
                )
                pending_field.append((horse_id, field_quality))
                if np.isfinite(performance):
                    pending_performance.append((horse_id, performance))

            if year <= train_end and design is not None and clean is not None:
                condition_updates.append((design, float(clean[0])))

        for horse_id, performance in pending_performance:
            _update_history(
                performance_state,
                horse_id,
                performance,
                event_date,
                decay_rate,
            )
        for horse_id, field_quality in pending_field:
            _update_history(
                field_state,
                horse_id,
                field_quality,
                event_date,
                decay_rate,
            )

        if year <= train_end:
            for design, winner_clock in condition_updates:
                xtx += np.outer(design, design)
                xty += design * winner_clock
                clean_race_count += 1
                fit_year_counts[year] = fit_year_counts.get(year, 0) + 1
            if condition_updates:
                max_fit_date = event_date
            frozen_beta = (
                _solve_ridge(xtx, xty, spec.ridge_alpha)
                if clean_race_count >= spec.min_prior_clean_races
                else None
            )

    history = pd.DataFrame(emitted).sort_values(
        "_source_position", kind="stable"
    )
    history = history.loc[:, ["_source_position", *history_columns]].drop(
        columns="_source_position"
    )
    history[PERFORMANCE_COLUMN] = pd.to_numeric(
        history[PERFORMANCE_COLUMN], errors="coerce"
    ).astype("float32")
    history[FIELD_QUALITY_COLUMN] = pd.to_numeric(
        history[FIELD_QUALITY_COLUMN], errors="coerce"
    ).astype("float32")

    source_observations = pd.DataFrame(observations).sort_values(
        "source_position", kind="stable"
    )
    source_observations = source_observations.loc[:, observation_columns].reset_index(
        drop=True
    )
    race_unique = source_observations.groupby("race_id", observed=True)[
        "field_quality_observation"
    ].nunique(dropna=False)
    constant_violations = int(race_unique.gt(1).sum())
    if constant_violations:
        raise AssertionError("S1 field quality is not race-constant")
    recalculated = source_observations.groupby("race_id", observed=True)[
        "global_rating_pre"
    ].transform("mean")
    field_error = (
        source_observations["field_quality_observation"] - recalculated
    ).abs()
    max_field_error = float(field_error.max()) if len(field_error) else 0.0
    if max_field_error > 1e-10:
        raise AssertionError("S1 field quality differs from full-starter rating mean")

    history_dates = pd.to_datetime(history["race_date"], errors="raise")
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
            "rows_after_fold_evaluation_ignored": int(
                dates.dt.year.gt(evaluation_year).sum()
            ),
            "processed_starter_rows": int(len(history)),
            "processed_races": int(history["race_id"].nunique()),
            "processed_min_date": str(history_dates.min().date()),
            "processed_max_date": str(history_dates.max().date()),
            "rows_used_2023_plus": 0,
            "odds_used": False,
            "direct_entity_id_features": 0,
        },
        "condition_fit": {
            "scope": "2013 warm-up plus fold train; frozen after train end",
            "clean_race_count": clean_race_count,
            "fit_race_count_by_year": {
                str(year): count for year, count in sorted(fit_year_counts.items())
            },
            "max_fit_date": str(max_fit_date.date()) if max_fit_date is not None else None,
            "evaluation_rows_used": 0,
            "calibration_rows_used": 0,
            "early_stopping_rows_used": 0,
            "frozen_beta": (
                [float(value) for value in frozen_beta] if frozen_beta is not None else None
            ),
            "frozen_beta_sha256": _beta_hash(frozen_beta),
        },
        "field_quality": {
            "definition": "full-starter arithmetic mean of pre-race global Elo",
            "race_constant_violations": constant_violations,
            "max_abs_error_vs_recalculated_full_starter_mean": max_field_error,
            "leave_one_out_used": False,
            "future_opponent_results_used": False,
        },
        "history": {
            "performance_nonmissing": int(history[PERFORMANCE_COLUMN].notna().sum()),
            "performance_missing": int(history[PERFORMANCE_COLUMN].isna().sum()),
            "field_quality_nonmissing": int(history[FIELD_QUALITY_COLUMN].notna().sum()),
            "field_quality_missing": int(history[FIELD_QUALITY_COLUMN].isna().sum()),
            "same_date_emit_before_update": True,
        },
    }
    return history.reset_index(drop=True), source_observations, audit
