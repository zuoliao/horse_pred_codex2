"""Forward-only standalone horse rating algorithms.

The algorithm specification is immutable during a run; horse states update
only after every race on a date has emitted its pre-race prediction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from math import exp, isfinite, log, sqrt
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.evaluation import evaluate_predictions
from horse_pred.features import (
    FeatureConfig,
    _numeric_finish,
    _race_elo_deltas,
    _race_surface_key,
    _starter_flags,
    is_flat_race,
)
from horse_pred.race_content import parse_result_time_seconds


@dataclass(frozen=True)
class RatingSpec:
    """Versionable online rating algorithm configuration."""

    family: str
    initial_rating: float = 1500.0
    k: float = 24.0
    scale: float = 400.0
    learning_rate: float = 0.1
    surface_blend_weight: float = 0.0
    pairwise_actual: str = "ordinal"
    time_margin_tau_seconds_per_1000m: float | None = None

    def __post_init__(self) -> None:
        if self.family not in {"pairwise_elo", "online_top1_pl"}:
            raise ValueError(f"unsupported rating family: {self.family}")
        if self.k <= 0 or self.scale <= 0 or self.learning_rate <= 0:
            raise ValueError("rating update parameters must be positive")
        if not 0.0 <= self.surface_blend_weight <= 1.0:
            raise ValueError("surface_blend_weight must be in [0, 1]")
        if self.pairwise_actual not in {"ordinal", "time_margin_logistic"}:
            raise ValueError(f"unsupported pairwise actual: {self.pairwise_actual}")
        if self.family != "pairwise_elo" and self.pairwise_actual != "ordinal":
            raise ValueError("non-ordinal pairwise actual requires pairwise_elo")
        if self.pairwise_actual == "time_margin_logistic":
            tau = self.time_margin_tau_seconds_per_1000m
            if tau is None or not isfinite(tau) or tau <= 0:
                raise ValueError("time-margin pairwise actual requires a finite positive tau")
        elif self.time_margin_tau_seconds_per_1000m is not None:
            raise ValueError("time-margin tau is only valid for time_margin_logistic")

    @property
    def feature_initial_score(self) -> float:
        return self.initial_rating if self.family == "pairwise_elo" else 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "initial_rating": self.initial_rating,
            "k": self.k,
            "scale": self.scale,
            "learning_rate": self.learning_rate,
            "surface_blend_weight": self.surface_blend_weight,
        }
        if self.pairwise_actual != "ordinal":
            payload["pairwise_actual"] = self.pairwise_actual
            payload["time_margin_tau_seconds_per_1000m"] = (
                self.time_margin_tau_seconds_per_1000m
            )
        return payload


@dataclass(frozen=True)
class RatingEvent:
    """One flat race containing starters only, in deterministic source order."""

    race_id: str
    race_date: pd.Timestamp
    surface_key: str | None
    horse_ids: tuple[object, ...]
    finishes: tuple[float, ...]
    source_positions: tuple[int, ...]
    result_times_seconds: tuple[float, ...] = ()
    finish_statuses: tuple[str, ...] = ()
    distance_m: float = np.nan

    def __post_init__(self) -> None:
        size = len(self.horse_ids)
        if len(self.finishes) != size or len(self.source_positions) != size:
            raise ValueError("rating event runner fields must have equal length")
        for values in (self.result_times_seconds, self.finish_statuses):
            if values and len(values) != size:
                raise ValueError("rating event content fields must match runner count")


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _model_score(raw_state: float, spec: RatingSpec) -> float:
    if spec.family == "pairwise_elo":
        return (raw_state - spec.initial_rating) * log(10.0) / spec.scale
    return raw_state


def _top1_pl_deltas(
    horse_ids: Sequence[object],
    finishes: Sequence[float],
    pre_states: Mapping[object, float],
    learning_rate: float,
) -> dict[object, float]:
    """Gradient step for the top-choice stage of a Plackett-Luce model."""

    if len(horse_ids) < 2:
        return {}
    winner_indices = [
        index for index, finish in enumerate(finishes) if np.isfinite(finish) and finish == 1.0
    ]
    if not winner_indices:
        return {}
    probabilities = _softmax([pre_states[horse_id] for horse_id in horse_ids])
    target = np.zeros(len(horse_ids), dtype=np.float64)
    target[winner_indices] = 1.0 / len(winner_indices)
    return {
        horse_id: learning_rate * float(target[index] - probabilities[index])
        for index, horse_id in enumerate(horse_ids)
    }


def _ordinal_pair_actual(finish_i: float, finish_j: float) -> float:
    if np.isfinite(finish_i) and np.isfinite(finish_j):
        return 0.5 if finish_i == finish_j else float(finish_i < finish_j)
    if np.isfinite(finish_i):
        return 1.0
    if np.isfinite(finish_j):
        return 0.0
    return 0.5


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    weight = exp(value)
    return weight / (1.0 + weight)


def _time_margin_elo_deltas(
    event: RatingEvent,
    pre_states: Mapping[object, float],
    spec: RatingSpec,
) -> dict[object, float]:
    """Return zero-sum Elo deltas using a continuous clock-margin actual."""

    horse_ids = event.horse_ids
    if len(horse_ids) < 2:
        return {}
    if not event.result_times_seconds or not event.finish_statuses:
        raise ValueError("time-margin rating requires event times and statuses")
    tau = spec.time_margin_tau_seconds_per_1000m
    if tau is None:
        raise AssertionError("validated time-margin spec has no tau")
    force_ordinal = any(
        status in {"demoted", "disqualified", "unknown"}
        for status in event.finish_statuses
    )
    valid_distance = np.isfinite(event.distance_m) and event.distance_m > 0
    divisor = float(len(horse_ids) - 1)
    deltas: dict[object, float] = defaultdict(float)
    for index, horse_i in enumerate(horse_ids):
        if pd.isna(horse_i):
            continue
        finish_i = event.finishes[index]
        time_i = event.result_times_seconds[index]
        for other_index in range(index + 1, len(horse_ids)):
            horse_j = horse_ids[other_index]
            if pd.isna(horse_j) or horse_i == horse_j:
                continue
            finish_j = event.finishes[other_index]
            time_j = event.result_times_seconds[other_index]
            actual_i = _ordinal_pair_actual(finish_i, finish_j)
            continuous_eligible = bool(
                not force_ordinal
                and valid_distance
                and np.isfinite(finish_i)
                and np.isfinite(finish_j)
                and finish_i != finish_j
                and np.isfinite(time_i)
                and np.isfinite(time_j)
            )
            if continuous_eligible:
                margin = (time_j - time_i) * 1000.0 / event.distance_m
                official_direction = 1.0 if finish_i < finish_j else -1.0
                if margin == 0.0 or margin * official_direction > 0.0:
                    actual_i = _logistic(margin / tau)
            expected_i = 1.0 / (
                1.0
                + 10.0
                ** ((pre_states[horse_j] - pre_states[horse_i]) / spec.scale)
            )
            delta = spec.k * (actual_i - expected_i) / divisor
            deltas[horse_i] += delta
            deltas[horse_j] -= delta
    return dict(deltas)


def _race_updates(
    horse_ids: Sequence[object],
    finishes: Sequence[float],
    pre_states: Mapping[object, float],
    spec: RatingSpec,
    event: RatingEvent | None = None,
) -> dict[object, float]:
    if spec.family == "pairwise_elo":
        if spec.pairwise_actual == "time_margin_logistic":
            if event is None:
                raise ValueError("time-margin pairwise actual requires a rating event")
            return _time_margin_elo_deltas(event, pre_states, spec)
        config = FeatureConfig(
            initial_elo=spec.initial_rating,
            elo_k=spec.k,
            elo_scale=spec.scale,
        )
        outcomes = [
            (horse_id, finish, bool(np.isfinite(finish) and finish == 1.0))
            for horse_id, finish in zip(horse_ids, finishes)
        ]
        return dict(_race_elo_deltas(outcomes, pre_states, config))
    return _top1_pl_deltas(
        horse_ids, finishes, pre_states, spec.learning_rate
    )


def _build_rating_history_legacy(
    normalized: pd.DataFrame,
    spec: RatingSpec,
    *,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Generate pre-race rating outputs for every flat-race starter.

    The input may include later data, but rows after ``through_year`` are
    discarded before any state update.  Results on a date become visible only
    after all predictions for that date have been emitted.
    """

    required = {
        "raceid",
        "date",
        "horse_id",
        "着順",
        "started",
        "course_type",
        "race_class",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"rating input is missing columns: {missing}")
    work = normalized.loc[
        :,
        [
            "raceid",
            "date",
            "horse_id",
            "着順",
            "started",
            "course_type",
            "race_class",
        ],
    ].copy(deep=False)
    work = work.assign(_input_order=np.arange(len(work)))
    work["_event_date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    work = work.loc[work["_event_date"].dt.year.le(through_year)].copy()
    work = work.sort_values(["_event_date", "raceid", "_input_order"], kind="stable")

    feature_config = FeatureConfig()
    global_states: dict[object, float] = {}
    global_starts: dict[object, int] = defaultdict(int)
    surface_states: dict[tuple[object, str], float] = {}
    surface_starts: dict[tuple[object, str], int] = defaultdict(int)
    emitted: list[dict[str, Any]] = []

    for event_date, date_rows in work.groupby("_event_date", sort=True):
        pending: list[
            tuple[
                list[object],
                list[float],
                dict[object, float],
                dict[object, float],
                str | None,
            ]
        ] = []
        for race_id, race in date_rows.groupby("raceid", sort=True):
            if not is_flat_race(race, feature_config):
                continue
            finish_numeric = race["着順"].map(_numeric_finish).astype(float)
            starter_flags = _starter_flags(race, race["着順"], feature_config)
            active_positions = [
                index
                for index, flag in enumerate(starter_flags)
                if flag is not pd.NA and bool(flag)
            ]
            if not active_positions:
                continue
            surface_key = _race_surface_key(race, feature_config)
            horse_ids = [race.iloc[index]["horse_id"] for index in active_positions]
            finishes = [float(finish_numeric.iloc[index]) for index in active_positions]
            global_pre = {
                horse_id: global_states.get(horse_id, spec.feature_initial_score)
                for horse_id in horse_ids
            }
            surface_pre = {
                horse_id: surface_states.get(
                    (horse_id, surface_key), spec.feature_initial_score
                )
                for horse_id in horse_ids
            }
            global_scores = [_model_score(global_pre[horse_id], spec) for horse_id in horse_ids]
            surface_scores = [_model_score(surface_pre[horse_id], spec) for horse_id in horse_ids]
            blended_scores = [
                global_score
                + spec.surface_blend_weight * (surface_score - global_score)
                for global_score, surface_score in zip(global_scores, surface_scores)
            ]
            probabilities = _softmax(blended_scores)
            for local_index, position in enumerate(active_positions):
                horse_id = horse_ids[local_index]
                condition_starts = (
                    surface_starts[(horse_id, surface_key)]
                    if surface_key is not None
                    else 0
                )
                emitted.append(
                    {
                        "source_position": int(race.iloc[position]["_input_order"]),
                        "race_id": str(race_id),
                        "horse_id": str(horse_id),
                        "race_date": event_date,
                        "finish_position": finishes[local_index],
                        "surface_key": surface_key,
                        "global_state_pre": global_pre[horse_id],
                        "condition_state_pre": surface_pre[horse_id],
                        "modular_rating__score_pre": blended_scores[local_index],
                        "modular_rating__raw_win_probability_pre": float(
                            probabilities[local_index]
                        ),
                        "modular_rating__global_starts_pre": float(
                            global_starts[horse_id]
                        ),
                        "modular_rating__condition_starts_pre": float(condition_starts),
                        "modular_rating__uncertainty_proxy_pre": 1.0
                        / sqrt(global_starts[horse_id] + 1.0),
                    }
                )
            pending.append((horse_ids, finishes, global_pre, surface_pre, surface_key))

        for horse_ids, finishes, global_pre, surface_pre, surface_key in pending:
            global_deltas = _race_updates(
                horse_ids, finishes, global_pre, spec
            )
            for horse_id in horse_ids:
                if horse_id in global_deltas:
                    global_states[horse_id] = global_pre[horse_id] + global_deltas[horse_id]
                global_starts[horse_id] += 1
            if surface_key is not None:
                surface_deltas = _race_updates(
                    horse_ids, finishes, surface_pre, spec
                )
                for horse_id in horse_ids:
                    if horse_id in surface_deltas:
                        surface_states[(horse_id, surface_key)] = (
                            surface_pre[horse_id] + surface_deltas[horse_id]
                        )
                    surface_starts[(horse_id, surface_key)] += 1

    result = pd.DataFrame(emitted)
    if result.empty:
        return result
    return result.sort_values("source_position", kind="stable").reset_index(drop=True)


def prepare_rating_events(
    normalized: pd.DataFrame, *, through_year: int = 2024
) -> tuple[RatingEvent, ...]:
    """Parse the raw runner frame once for reuse across rating candidates."""

    required = {
        "raceid",
        "date",
        "horse_id",
        "着順",
        "started",
        "course_type",
        "race_class",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"rating input is missing columns: {missing}")
    columns = [
        "raceid",
        "date",
        "horse_id",
        "着順",
        "started",
        "course_type",
        "race_class",
    ]
    for optional in ("time_raw", "status", "distance_m", "distance"):
        if optional in normalized.columns and optional not in columns:
            columns.append(optional)
    work = normalized.loc[:, columns].copy(deep=False)
    work = work.assign(_input_order=np.arange(len(work)))
    work["_event_date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    work = work.loc[work["_event_date"].dt.year.le(through_year)].copy()
    work = work.sort_values(["_event_date", "raceid", "_input_order"], kind="stable")
    feature_config = FeatureConfig()
    events: list[RatingEvent] = []
    for (event_date, race_id), race in work.groupby(
        ["_event_date", "raceid"], sort=True
    ):
        if not is_flat_race(race, feature_config):
            continue
        finishes = race["着順"].map(_numeric_finish).astype(float)
        starters = _starter_flags(race, race["着順"], feature_config)
        active = [
            index
            for index, flag in enumerate(starters)
            if flag is not pd.NA and bool(flag)
        ]
        if not active:
            continue
        if "time_raw" in race.columns:
            times = race["time_raw"].map(parse_result_time_seconds)
        else:
            times = pd.Series(np.nan, index=race.index, dtype="float64")
        if "status" in race.columns:
            statuses = race["status"].astype("string")
        else:
            finish_text = race["着順"].astype("string").str.strip()
            statuses = pd.Series("unknown", index=race.index, dtype="string")
            statuses.loc[finish_text.str.fullmatch(r"[0-9]+", na=False)] = "finished"
            statuses.loc[finish_text.isin(["中", "中止", "競走中止"])] = (
                "did_not_finish"
            )
            statuses.loc[finish_text.str.contains("降", na=False)] = "demoted"
            statuses.loc[finish_text.isin(["失", "失格"])] = "disqualified"
        distance_column = "distance_m" if "distance_m" in race.columns else "distance"
        distance_values = (
            pd.to_numeric(race[distance_column], errors="coerce").dropna().unique()
            if distance_column in race.columns
            else np.array([])
        )
        distance_m = float(distance_values[0]) if len(distance_values) == 1 else np.nan
        events.append(
            RatingEvent(
                race_id=str(race_id),
                race_date=event_date,
                surface_key=_race_surface_key(race, feature_config),
                horse_ids=tuple(race.iloc[index]["horse_id"] for index in active),
                finishes=tuple(float(finishes.iloc[index]) for index in active),
                source_positions=tuple(
                    int(race.iloc[index]["_input_order"]) for index in active
                ),
                result_times_seconds=tuple(
                    float(times.iloc[index]) for index in active
                ),
                finish_statuses=tuple(str(statuses.iloc[index]) for index in active),
                distance_m=distance_m,
            )
        )
    return tuple(events)


def build_rating_history_from_events(
    events: Sequence[RatingEvent], spec: RatingSpec
) -> pd.DataFrame:
    """Run one rating candidate over a pre-parsed chronological event stream."""

    global_states: dict[object, float] = {}
    global_starts: dict[object, int] = defaultdict(int)
    surface_states: dict[tuple[object, str], float] = {}
    surface_starts: dict[tuple[object, str], int] = defaultdict(int)
    emitted: list[dict[str, Any]] = []
    for _, date_events_iterator in groupby(events, key=lambda event: event.race_date):
        date_events = list(date_events_iterator)
        pending: list[
            tuple[
                RatingEvent,
                dict[object, float],
                dict[object, float],
            ]
        ] = []
        for event in date_events:
            global_pre = {
                horse_id: global_states.get(horse_id, spec.feature_initial_score)
                for horse_id in event.horse_ids
            }
            surface_pre = {
                horse_id: surface_states.get(
                    (horse_id, event.surface_key), spec.feature_initial_score
                )
                for horse_id in event.horse_ids
            }
            global_scores = [
                _model_score(global_pre[horse_id], spec)
                for horse_id in event.horse_ids
            ]
            surface_scores = [
                _model_score(surface_pre[horse_id], spec)
                for horse_id in event.horse_ids
            ]
            blended_scores = [
                global_score
                + spec.surface_blend_weight * (surface_score - global_score)
                for global_score, surface_score in zip(global_scores, surface_scores)
            ]
            probabilities = _softmax(blended_scores)
            for index, horse_id in enumerate(event.horse_ids):
                condition_starts = (
                    surface_starts[(horse_id, event.surface_key)]
                    if event.surface_key is not None
                    else 0
                )
                emitted.append(
                    {
                        "source_position": event.source_positions[index],
                        "race_id": event.race_id,
                        "horse_id": str(horse_id),
                        "race_date": event.race_date,
                        "finish_position": event.finishes[index],
                        "surface_key": event.surface_key,
                        "global_state_pre": global_pre[horse_id],
                        "condition_state_pre": surface_pre[horse_id],
                        "modular_rating__score_pre": blended_scores[index],
                        "modular_rating__raw_win_probability_pre": float(
                            probabilities[index]
                        ),
                        "modular_rating__global_starts_pre": float(
                            global_starts[horse_id]
                        ),
                        "modular_rating__condition_starts_pre": float(
                            condition_starts
                        ),
                        "modular_rating__uncertainty_proxy_pre": 1.0
                        / sqrt(global_starts[horse_id] + 1.0),
                    }
                )
            pending.append((event, global_pre, surface_pre))

        for event, global_pre, surface_pre in pending:
            global_deltas = _race_updates(
                event.horse_ids, event.finishes, global_pre, spec, event
            )
            for horse_id in event.horse_ids:
                if horse_id in global_deltas:
                    global_states[horse_id] = (
                        global_pre[horse_id] + global_deltas[horse_id]
                    )
                global_starts[horse_id] += 1
            if event.surface_key is not None:
                surface_deltas = _race_updates(
                    event.horse_ids, event.finishes, surface_pre, spec, event
                )
                for horse_id in event.horse_ids:
                    if horse_id in surface_deltas:
                        surface_states[(horse_id, event.surface_key)] = (
                            surface_pre[horse_id] + surface_deltas[horse_id]
                        )
                    surface_starts[(horse_id, event.surface_key)] += 1

    result = pd.DataFrame(emitted)
    if result.empty:
        return result
    return result.sort_values("source_position", kind="stable").reset_index(drop=True)


def build_rating_history(
    normalized: pd.DataFrame,
    spec: RatingSpec,
    *,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Parse and run a standalone rating history in one call."""

    events = prepare_rating_events(normalized, through_year=through_year)
    return build_rating_history_from_events(events, spec)


def attach_scoring_population(
    rating_history: pd.DataFrame, model_frame: pd.DataFrame
) -> pd.DataFrame:
    """Restrict rating output to the exact accepted model scoring population."""

    metadata = [
        "race_id",
        "horse_id",
        "race_date",
        "split",
        "model_finish_position",
        "course_type",
        "distance",
        "race_class",
        "field_size",
    ]
    missing = sorted(set(metadata).difference(model_frame.columns))
    if missing:
        raise ValueError(f"model frame is missing rating metadata: {missing}")
    accepted = model_frame.loc[
        ~model_frame["split"].eq("retrospective_test"), metadata
    ].copy()
    accepted["race_id"] = accepted["race_id"].astype("string")
    accepted["horse_id"] = accepted["horse_id"].astype("string")
    history = rating_history.copy()
    history["race_id"] = history["race_id"].astype("string")
    history["horse_id"] = history["horse_id"].astype("string")
    joined = accepted.merge(
        history.drop(columns=["race_date", "finish_position"]),
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["modular_rating__score_pre"].isna().any():
        raise ValueError("rating history does not cover the accepted scoring population")
    return joined.sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    ).reset_index(drop=True)


def evaluate_rating_rows(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    """Evaluate one standalone rating split with the common EVAL-01 metrics."""

    selected = frame.loc[frame["split"].eq(split)].copy()
    if selected.empty:
        raise ValueError(f"rating split has no rows: {split}")
    selected = selected.sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    )
    distance = pd.to_numeric(selected["distance"], errors="coerce")
    selected["distance_band"] = pd.cut(
        distance,
        bins=[0, 1400, 1800, 2200, np.inf],
        labels=["sprint", "mile", "middle", "long"],
        include_lowest=True,
    ).astype("string")
    selected["field_size_band"] = pd.cut(
        selected["field_size"],
        bins=[0, 9, 13, 16, np.inf],
        labels=["small", "medium", "large", "very_large"],
        include_lowest=True,
    ).astype("string")
    return evaluate_predictions(
        selected["modular_rating__raw_win_probability_pre"],
        selected["model_finish_position"],
        selected["race_id"],
        ranking_scores=selected["modular_rating__score_pre"],
        conditions={
            column: selected[column]
            for column in (
                "course_type",
                "distance_band",
                "race_class",
                "field_size_band",
            )
        },
    )
