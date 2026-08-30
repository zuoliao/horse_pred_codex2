# ruff: noqa: UP045
"""Point-in-time feature construction for the initial JRA tabular baseline.

The source data used by the MVP has a race date but no publication/start
timestamp.  Consequently, this module deliberately uses the conservative
PIT-C rule ``history_date < target_date``: features for *all* races on a date
are emitted before any result from that date updates state.

Raw identifiers and outcome/market columns are retained only as ``meta__``
join keys.  The model allowlist is built exclusively from documented feature
group prefixes.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from math import exp, log
from typing import Optional

import numpy as np
import pandas as pd

JRA_VENUE_CODES: Mapping[str, str] = {
    "01": "sapporo",
    "02": "hakodate",
    "03": "fukushima",
    "04": "niigata",
    "05": "tokyo",
    "06": "nakayama",
    "07": "chukyo",
    "08": "kyoto",
    "09": "hanshin",
    "10": "kokura",
}

_GENERATED_PREFIXES: tuple[str, ...] = (
    "context__",
    "horse_history__",
    "jockey_history__",
    "trainer_history__",
    "field_relative__",
    "rating__",
    "surface_rating__",
    "race_value__",
)

FEATURE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "race_context": ("context__",),
    "horse_history_basic": ("horse_history__",),
    "form_workload": ("horse_history__",),
    "connections_pit": ("jockey_history__", "trainer_history__"),
    "field_relative": ("field_relative__",),
    "rating_strength": ("rating__",),
    "surface_conditioned_rating": ("surface_rating__",),
    "race_value_expected_actual": ("race_value__",),
}

# This is documentation and a defensive check.  Safety does not depend on
# enumerating every spelling: model_feature_allowlist() only admits generated
# feature prefixes and therefore rejects every unrecognised raw column.
FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "raceid",
        "race_id",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "trainer",
        "着順",
        "finish_position",
        "タイム",
        "着差",
        "通過順位",
        "上がり3F",
        "ground_state",
        "weather",
        "単勝",
        "win_odds",
        "人気",
        "popularity",
        "払戻",
        "payout",
    }
)


@dataclass(frozen=True)
class FeatureDataset:
    """Integrated feature output consumed by future model runners.

    ``frame`` intentionally retains source metadata/labels/final-market fields
    for joins and evaluation.  Only ``feature_columns`` may be passed to a
    prediction model; it contains generated numeric columns only.
    """

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_groups: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class FeatureConfig:
    """Column contract and intentionally small baseline feature settings."""

    race_id_col: str = "raceid"
    date_col: str = "date"
    horse_id_col: str = "horse_id"
    jockey_id_col: str = "jockey_id"
    # The accepted raw has trainer name but no trainer ID.  It is a state key,
    # never a model feature; DATA-01 should replace it if a canonical ID arrives.
    trainer_id_col: str = "trainer"
    finish_col: str = "着順"
    started_col: Optional[str] = "started"
    distance_col: str = "distance"
    race_class_col: str = "race_class"
    surface_col: str = "course_type"
    ground_col: str = "ground_state"
    direction_col: str = "around"
    weather_col: str = "weather"
    sex_col: str = "sex"
    age_col: str = "age"
    gate_col: str = "枠番"
    horse_number_col: str = "馬番"
    venue_col: Optional[str] = None

    count_windows: tuple[int, ...] = (1, 3, 5, 10)
    day_windows: tuple[int, ...] = (14, 30, 90, 180, 365)
    decay_half_lives: tuple[int, ...] = (30, 90, 180)

    initial_elo: float = 1500.0
    elo_k: float = 24.0
    elo_scale: float = 400.0
    # Experimental opt-in.  Keeping this false preserves the accepted
    # 268-column baseline contract byte-for-byte while allowing a separately
    # ablatable turf/dirt-specific Elo state to be materialized.
    surface_conditioned_elo: bool = False
    # Experimental opt-in.  This exposes the 90-day exponentially decayed
    # mean of actual pairwise performance minus global-Elo expectation.  The
    # observation for a race is exactly its global Elo delta divided by K.
    expected_actual_race_value: bool = False

    def __post_init__(self) -> None:
        for name, values in (
            ("count_windows", self.count_windows),
            ("day_windows", self.day_windows),
            ("decay_half_lives", self.decay_half_lives),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive values")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing and unique")
        if self.elo_k <= 0 or self.elo_scale <= 0:
            raise ValueError("elo_k and elo_scale must be positive")
        if self.expected_actual_race_value and 90 not in self.decay_half_lives:
            raise ValueError(
                "expected_actual_race_value requires a 90-day decay half-life"
            )


@dataclass(frozen=True)
class _Performance:
    date: pd.Timestamp
    win: float
    finish: float
    distance: float
    surface: object
    distance_band: object
    venue: object
    opponent_mean_elo: float
    performance_value: float
    elo_surprise: float


@dataclass
class _Aggregate:
    starts: float = 0.0
    wins: float = 0.0
    completed: float = 0.0
    finish_sum: float = 0.0
    distance_sum: float = 0.0
    opponent_count: float = 0.0
    opponent_sum: float = 0.0
    value_count: float = 0.0
    value_sum: float = 0.0
    surprise_count: float = 0.0
    surprise_sum: float = 0.0

    def add(self, record: _Performance, weight: float = 1.0) -> None:
        self.starts += weight
        self.wins += weight * record.win
        if np.isfinite(record.finish):
            self.completed += weight
            self.finish_sum += weight * record.finish
        if np.isfinite(record.distance):
            self.distance_sum += weight * record.distance
        if np.isfinite(record.opponent_mean_elo):
            self.opponent_count += weight
            self.opponent_sum += weight * record.opponent_mean_elo
        if np.isfinite(record.performance_value):
            self.value_count += weight
            self.value_sum += weight * record.performance_value
        if np.isfinite(record.elo_surprise):
            self.surprise_count += weight
            self.surprise_sum += weight * record.elo_surprise

    def multiply(self, factor: float) -> None:
        for name in (
            "starts",
            "wins",
            "completed",
            "finish_sum",
            "distance_sum",
            "opponent_count",
            "opponent_sum",
            "value_count",
            "value_sum",
            "surprise_count",
            "surprise_sum",
        ):
            setattr(self, name, getattr(self, name) * factor)

    def as_features(self, prefix: str, include_distance: bool = True) -> dict[str, float]:
        values = {
            f"{prefix}starts": self.starts,
            f"{prefix}wins": self.wins,
            f"{prefix}win_rate": _safe_ratio(self.wins, self.starts),
            f"{prefix}completed": self.completed,
            f"{prefix}mean_finish": _safe_ratio(self.finish_sum, self.completed),
        }
        if include_distance:
            values[f"{prefix}distance_sum"] = self.distance_sum
        return values


@dataclass
class _EntityState:
    max_count: int
    max_days: int
    half_lives: tuple[int, ...]
    career: _Aggregate = field(default_factory=_Aggregate)
    recent_count: deque[_Performance] = field(default_factory=deque)
    recent_days: deque[_Performance] = field(default_factory=deque)
    conditioned: MutableMapping[tuple[str, object], _Aggregate] = field(
        default_factory=lambda: defaultdict(_Aggregate)
    )
    decayed: MutableMapping[int, _Aggregate] = field(default_factory=dict)
    decay_date: Optional[pd.Timestamp] = None
    last_start_date: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        self.recent_count = deque(maxlen=self.max_count)
        self.decayed = {half_life: _Aggregate() for half_life in self.half_lives}

    def _advance_decay(self, target_date: pd.Timestamp) -> None:
        if self.decay_date is None:
            self.decay_date = target_date
            return
        elapsed_days = (target_date - self.decay_date).total_seconds() / 86_400.0
        if elapsed_days < 0:
            raise ValueError("entity state cannot move backward in time")
        for half_life, aggregate in self.decayed.items():
            aggregate.multiply(exp(-log(2.0) * elapsed_days / half_life))
        self.decay_date = target_date

    def snapshot(
        self,
        target_date: pd.Timestamp,
        prefix: str,
        config: FeatureConfig,
        include_workload: bool,
        conditions: Optional[Mapping[str, object]] = None,
    ) -> dict[str, float]:
        self._advance_decay(target_date)
        cutoff = target_date - pd.Timedelta(self.max_days, unit="D")
        while self.recent_days and self.recent_days[0].date < cutoff:
            self.recent_days.popleft()

        result = self.career.as_features(f"{prefix}career__", include_distance=include_workload)
        if include_workload:
            result[f"{prefix}days_since_last_start"] = (
                float((target_date - self.last_start_date).days) if self.last_start_date is not None else np.nan
            )

        count_records = list(self.recent_count)
        day_records = list(self.recent_days)
        for window in config.count_windows:
            aggregate = _aggregate_records(count_records[-window:])
            result.update(aggregate.as_features(f"{prefix}last_{window}__", include_distance=include_workload))
        for window in config.day_windows:
            window_start = target_date - pd.Timedelta(window, unit="D")
            aggregate = _aggregate_records(record for record in day_records if record.date >= window_start)
            result.update(aggregate.as_features(f"{prefix}days_{window}__", include_distance=include_workload))
        for half_life, aggregate in self.decayed.items():
            result.update(
                aggregate.as_features(f"{prefix}decay_{half_life}d__", include_distance=include_workload)
            )

        if conditions is not None:
            for condition_name, condition_value in conditions.items():
                aggregate = self.conditioned.get((condition_name, condition_value), _Aggregate())
                result.update(
                    aggregate.as_features(
                        f"{prefix}same_{condition_name}__",
                        include_distance=False,
                    )
                )
            result[f"{prefix}career__mean_opponent_elo"] = _safe_ratio(
                self.career.opponent_sum, self.career.opponent_count
            )
            result[f"{prefix}career__mean_performance_value"] = _safe_ratio(
                self.career.value_sum, self.career.value_count
            )
            if config.expected_actual_race_value:
                decay_90d = self.decayed[90]
                result["race_value__decay_90d__mean_global_elo_surprise"] = (
                    _safe_ratio(decay_90d.surprise_sum, decay_90d.surprise_count)
                )
        return result

    def update(self, record: _Performance) -> None:
        self._advance_decay(record.date)
        self.career.add(record)
        self.recent_count.append(record)
        self.recent_days.append(record)
        for condition_name, condition_value in (
            ("surface", record.surface),
            ("distance_band", record.distance_band),
            ("venue", record.venue),
        ):
            if not _is_missing_key(condition_value):
                self.conditioned[(condition_name, condition_value)].add(record)
        for aggregate in self.decayed.values():
            aggregate.add(record)
        self.last_start_date = record.date


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _aggregate_records(records: Iterable[_Performance]) -> _Aggregate:
    aggregate = _Aggregate()
    for record in records:
        aggregate.add(record)
    return aggregate


def _is_missing_key(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _state_for(
    states: MutableMapping[object, _EntityState], key: object, config: FeatureConfig
) -> Optional[_EntityState]:
    if _is_missing_key(key):
        return None
    if key not in states:
        states[key] = _EntityState(
            max_count=max(config.count_windows),
            max_days=max(config.day_windows),
            half_lives=config.decay_half_lives,
        )
    return states[key]


def _empty_snapshot(
    target_date: pd.Timestamp,
    prefix: str,
    config: FeatureConfig,
    include_workload: bool,
    conditions: Optional[Mapping[str, object]] = None,
) -> dict[str, float]:
    return _EntityState(
        max_count=max(config.count_windows),
        max_days=max(config.day_windows),
        half_lives=config.decay_half_lives,
    ).snapshot(target_date, prefix, config, include_workload, conditions)


def _distance_band(value: float) -> object:
    if not np.isfinite(value):
        return np.nan
    if value < 1400:
        return "sprint"
    if value < 1800:
        return "mile"
    if value < 2200:
        return "middle"
    return "staying"


def _venue_from_race_id(race_id: object) -> object:
    if _is_missing_key(race_id):
        return np.nan
    text = str(race_id).strip()
    # pandas may have parsed an otherwise integral ID as a float.
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) < 6:
        return np.nan
    code = text[4:6]
    return int(code) if code in JRA_VENUE_CODES else np.nan


def _venue_code(value: object) -> float:
    if _is_missing_key(value):
        return np.nan
    numeric = _numeric_scalar(value)
    if np.isfinite(numeric) and 1 <= numeric <= 10:
        return numeric
    normalized = str(value).strip().lower()
    aliases = {
        **{name: int(code) for code, name in JRA_VENUE_CODES.items()},
        "札幌": 1,
        "函館": 2,
        "福島": 3,
        "新潟": 4,
        "東京": 5,
        "中山": 6,
        "中京": 7,
        "京都": 8,
        "阪神": 9,
        "小倉": 10,
    }
    return float(aliases.get(normalized, np.nan))


def _class_context(value: object) -> dict[str, float]:
    text = "" if _is_missing_key(value) else re.sub(r"\s+", "", str(value).replace("\u00a0", ""))
    if "新馬" in text:
        tier = 0.0
    elif "未勝利" in text:
        tier = 1.0
    elif "1勝" in text or "500万" in text:
        tier = 2.0
    elif "2勝" in text or "1000万" in text:
        tier = 3.0
    elif "3勝" in text or "1600万" in text:
        tier = 4.0
    elif (
        any(token in text.upper() for token in ("OPEN", "OP", "G1", "G2", "G3"))
        or "オープン" in text
        or "重賞" in text
    ):
        tier = 5.0
    else:
        tier = np.nan
    age_match = re.search(r"([234])歳", text)
    return {
        "context__class_tier": tier,
        "context__class_female_only": float("牝" in text),
        "context__class_age_min": float(age_match.group(1)) if age_match else np.nan,
        "context__class_age_and_older": float(bool(age_match and "以上" in text)),
    }


def _one_hot(group: str, value: object, aliases: Mapping[str, Sequence[str]]) -> dict[str, float]:
    text = "" if _is_missing_key(value) else str(value).strip().lower()
    matched = False
    result: dict[str, float] = {}
    for name, candidates in aliases.items():
        present = text in candidates
        result[name] = float(present)
        matched |= present
    result[f"context__{group}_unknown"] = float(not matched)
    return result


def _numeric_finish(value: object) -> float:
    if _is_missing_key(value):
        return np.nan
    extracted = pd.Series([str(value)]).str.extract(r"^\s*(\d+)", expand=False).iloc[0]
    return float(extracted) if not pd.isna(extracted) else np.nan


def _numeric_scalar(value: object) -> float:
    if _is_missing_key(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_cancelled_or_excluded(value: object) -> bool:
    if _is_missing_key(value):
        return False
    text = str(value)
    return "取" in text or "除" in text


def _starter_from_finish(value: object) -> Optional[bool]:
    if _is_missing_key(value):
        return None
    text = str(value).strip()
    if _is_cancelled_or_excluded(text):
        return False
    if re.match(r"^\d+", text) or "中" in text or "失" in text:
        return True
    return None


def _starter_flags(race: pd.DataFrame, finish_raw: pd.Series, config: FeatureConfig) -> pd.Series:
    if config.started_col is None or config.started_col not in race:
        return finish_raw.map(_starter_from_finish).astype("boolean")
    normalized = race[config.started_col]
    invalid = normalized.notna() & ~normalized.isin([True, False, 1, 0])
    if invalid.any():
        raise ValueError(f"{config.started_col!r} must contain only boolean or missing values")
    return normalized.astype("boolean")


def _flat_surface_key(value: object) -> Optional[str]:
    if _is_missing_key(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"芝", "turf"}:
        return "turf"
    if normalized in {"ダート", "dirt"}:
        return "dirt"
    return None


def _is_flat_surface(value: object) -> bool:
    return _flat_surface_key(value) is not None


def _race_surface_key(race: pd.DataFrame, config: FeatureConfig) -> Optional[str]:
    """Return a single normalized flat surface, or None for inconsistent data."""

    if config.surface_col not in race:
        return None
    keys = {
        key
        for key in race[config.surface_col].map(_flat_surface_key)
        if key is not None
    }
    return next(iter(keys)) if len(keys) == 1 else None


def _is_obstacle_class(value: object) -> bool:
    if _is_missing_key(value):
        return False
    return "障害" in str(value)


def is_flat_race(race: pd.DataFrame, config: Optional[FeatureConfig] = None) -> bool:
    """Return whether every row belongs to the flat-racing population.

    The source's ``course_type`` is not a reliable flat/jump discriminator:
    some jump races retain the underlying turf or dirt course label.  A race
    whose class contains ``障害`` is therefore non-flat regardless of its
    surface label.
    """

    config = config or FeatureConfig()
    if config.surface_col not in race:
        return False
    surfaces_are_flat = bool(race[config.surface_col].map(_is_flat_surface).all())
    if not surfaces_are_flat:
        return False
    if config.race_class_col not in race:
        return False
    return not bool(race[config.race_class_col].map(_is_obstacle_class).any())


def _expected_elo(rating_i: float, rating_j: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_j - rating_i) / scale))


def _race_elo_deltas(
    outcomes: Sequence[tuple[object, float, bool]],
    pre_ratings: Mapping[object, float],
    config: FeatureConfig,
) -> dict[object, float]:
    """Return simultaneous pairwise Elo deltas; non-starters are absent.

    Numeric finishes outrank missing finishes (中止/失格); two missing
    finishes and equal numeric ranks are ties.  The K factor is divided by the
    number of opponents, preventing field size from mechanically scaling the
    total update.
    """

    deltas: dict[object, float] = defaultdict(float)
    if len(outcomes) < 2:
        return deltas
    divisor = float(len(outcomes) - 1)
    for index, (horse_i, finish_i, _) in enumerate(outcomes):
        if _is_missing_key(horse_i):
            continue
        for horse_j, finish_j, _ in outcomes[index + 1 :]:
            if _is_missing_key(horse_j) or horse_i == horse_j:
                continue
            if np.isfinite(finish_i) and np.isfinite(finish_j):
                actual_i = 0.5 if finish_i == finish_j else float(finish_i < finish_j)
            elif np.isfinite(finish_i):
                actual_i = 1.0
            elif np.isfinite(finish_j):
                actual_i = 0.0
            else:
                actual_i = 0.5
            expected_i = _expected_elo(pre_ratings[horse_i], pre_ratings[horse_j], config.elo_scale)
            delta = config.elo_k * (actual_i - expected_i) / divisor
            deltas[horse_i] += delta
            deltas[horse_j] -= delta
    return deltas


def _require_columns(raw: pd.DataFrame, config: FeatureConfig) -> None:
    required = {
        config.race_id_col,
        config.date_col,
        config.horse_id_col,
        config.jockey_id_col,
        config.trainer_id_col,
        config.finish_col,
        config.distance_col,
        config.race_class_col,
        config.surface_col,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"missing required raw columns: {missing}")


def _copy_context(row: pd.Series, config: FeatureConfig, venue: object, field_size: int) -> dict[str, object]:
    def value(column: Optional[str]) -> object:
        return row[column] if column is not None and column in row.index else np.nan

    context: dict[str, object] = {
        "context__venue_code": _venue_code(venue),
        "context__distance": _numeric_scalar(value(config.distance_col)),
        "context__age": _numeric_scalar(value(config.age_col)),
        "context__gate": _numeric_scalar(value(config.gate_col)),
        "context__horse_number": _numeric_scalar(value(config.horse_number_col)),
        "context__field_size_rows": float(field_size),
    }
    context.update(
        _one_hot(
            "surface",
            value(config.surface_col),
            {
                "context__surface_turf": ("芝", "turf"),
                "context__surface_dirt": ("ダート", "dirt"),
            },
        )
    )
    context.update(
        _one_hot(
            "direction",
            value(config.direction_col),
            {
                "context__direction_right": ("右", "right"),
                "context__direction_left": ("左", "left"),
                "context__direction_straight": ("直線", "straight"),
            },
        )
    )
    context.update(
        _one_hot(
            "sex",
            value(config.sex_col),
            {
                "context__sex_male": ("牡", "male", "m"),
                "context__sex_female": ("牝", "female", "f"),
                "context__sex_gelding": ("セ", "騸", "gelding", "g"),
            },
        )
    )
    context.update(_class_context(value(config.race_class_col)))
    return context


def _add_field_relative(race_features: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "horse_win_rate": "horse_history__career__win_rate",
        "horse_recent_win_rate": "horse_history__last_5__win_rate",
        "jockey_win_rate": "jockey_history__career__win_rate",
        "trainer_win_rate": "trainer_history__career__win_rate",
        "horse_rest_days": "horse_history__days_since_last_start",
    }
    for short_name, source in candidates.items():
        if source not in race_features:
            continue
        values = pd.to_numeric(race_features[source], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0)
        race_features[f"field_relative__{short_name}__minus_mean"] = values - mean
        race_features[f"field_relative__{short_name}__zscore"] = (
            (values - mean) / std if pd.notna(std) and std > 0 else values.where(values.isna(), 0.0)
        )
        race_features[f"field_relative__{short_name}__percentile"] = values.rank(pct=True, method="average")
    return race_features


def build_pit_features(raw: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Build deterministic, forward-only features from runner-level raw data.

    The returned frame has one row per input row and preserves input order and
    index.  ``meta__`` columns are join/audit fields, not model inputs.  Use
    :func:`model_feature_allowlist` rather than selecting numeric columns.
    """

    config = config or FeatureConfig()
    _require_columns(raw, config)
    if raw.empty:
        return pd.DataFrame(index=raw.index)

    work_columns = [
        column
        for column in (
            config.race_id_col,
            config.date_col,
            config.horse_id_col,
            config.jockey_id_col,
            config.trainer_id_col,
            config.finish_col,
            config.started_col,
            config.distance_col,
            config.race_class_col,
            config.surface_col,
            config.direction_col,
            config.sex_col,
            config.age_col,
            config.gate_col,
            config.horse_number_col,
            config.venue_col,
        )
        if column is not None and column in raw.columns
    ]
    work = raw.loc[:, list(dict.fromkeys(work_columns))].copy(deep=False)
    work = work.assign(_input_order=np.arange(len(raw)))
    work["_event_date"] = pd.to_datetime(work[config.date_col], errors="raise").dt.normalize()
    if work["_event_date"].isna().any():
        raise ValueError("date column contains missing values")
    if work[config.race_id_col].isna().any():
        raise ValueError("race ID contains missing values")
    if work.duplicated([config.race_id_col, config.horse_id_col]).any():
        raise ValueError("duplicate (race_id, horse_id) rows are not allowed")
    race_date_counts = work.groupby(config.race_id_col, dropna=False)["_event_date"].nunique()
    if (race_date_counts != 1).any():
        raise ValueError("each race ID must belong to exactly one date")

    work = work.sort_values(["_event_date", config.race_id_col, "_input_order"], kind="stable")
    horse_states: dict[object, _EntityState] = {}
    jockey_states: dict[object, _EntityState] = {}
    trainer_states: dict[object, _EntityState] = {}
    elo_ratings: dict[object, float] = {}
    surface_elo_ratings: dict[tuple[object, str], float] = {}
    emitted: list[pd.DataFrame] = []

    for event_date, date_rows in work.groupby("_event_date", sort=True):
        date_emitted: list[pd.DataFrame] = []
        connection_snapshot_cache: dict[tuple[str, object], dict[str, float]] = {}
        empty_connection_snapshots: dict[str, dict[str, float]] = {}
        pending_updates: list[
            tuple[
                pd.DataFrame,
                pd.DataFrame,
                dict[object, float],
                dict[object, float],
                object,
                object,
                bool,
                Optional[str],
            ]
        ] = []

        # Emit every race on this date before applying any result from the date.
        for race_id, race in date_rows.groupby(config.race_id_col, sort=False):
            race = race.copy()
            finish_raw = race[config.finish_col]
            finish_numeric = pd.to_numeric(
                finish_raw.astype("string").str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
            ).astype(float)
            starter_flags = _starter_flags(race, finish_raw, config)
            is_nonstarter = starter_flags.eq(False).fillna(False).astype(bool)
            has_unknown_start = bool(starter_flags.isna().any())
            race_is_flat = is_flat_race(race, config)
            is_scored_race = race_is_flat and not bool(is_nonstarter.any()) and not has_unknown_start
            field_size = len(race)
            venue_values = (
                race[config.venue_col]
                if config.venue_col is not None and config.venue_col in race
                else race[config.race_id_col].map(_venue_from_race_id)
            )
            distances = pd.to_numeric(race[config.distance_col], errors="coerce")

            pre_ratings = {
                horse_id: elo_ratings.get(horse_id, config.initial_elo)
                for horse_id in race[config.horse_id_col]
                if not _is_missing_key(horse_id)
            }
            active_pre = [
                pre_ratings.get(horse_id, config.initial_elo)
                for horse_id, started in zip(race[config.horse_id_col], starter_flags)
                if started is not pd.NA and bool(started)
            ]
            field_mean_elo = float(np.mean(active_pre)) if active_pre else config.initial_elo
            field_max_elo = float(np.max(active_pre)) if active_pre else config.initial_elo
            field_std_elo = float(np.std(active_pre)) if active_pre else 0.0

            surface_key = _race_surface_key(race, config)
            surface_pre_ratings = {
                horse_id: surface_elo_ratings.get(
                    (horse_id, surface_key), config.initial_elo
                )
                for horse_id in race[config.horse_id_col]
                if config.surface_conditioned_elo
                and surface_key is not None
                and not _is_missing_key(horse_id)
            }
            surface_active_pre = [
                surface_pre_ratings.get(horse_id, config.initial_elo)
                for horse_id, started in zip(race[config.horse_id_col], starter_flags)
                if started is not pd.NA and bool(started)
            ]
            surface_field_mean_elo = (
                float(np.mean(surface_active_pre))
                if surface_active_pre
                else config.initial_elo
            )

            rows: list[dict[str, object]] = []
            for position, (_, row) in enumerate(race.iterrows()):
                horse_id = row[config.horse_id_col]
                jockey_id = row[config.jockey_id_col]
                trainer_id = row[config.trainer_id_col]
                venue = venue_values.iloc[position]
                distance = float(distances.iloc[position]) if pd.notna(distances.iloc[position]) else np.nan
                conditions = {
                    "surface": row[config.surface_col] if config.surface_col in row.index else np.nan,
                    "distance_band": _distance_band(distance),
                    "venue": venue,
                }

                features: dict[str, object] = {
                    "meta__source_position": int(row["_input_order"]),
                    "meta__race_id": race_id,
                    "meta__horse_id": horse_id,
                    "meta__date": event_date,
                    "meta__venue": venue,
                    "meta__is_runner": starter_flags.iloc[position],
                    "meta__is_flat_race": race_is_flat,
                    "meta__is_scored_race": is_scored_race,
                }
                features.update(_copy_context(row, config, venue, field_size))

                horse_state = _state_for(horse_states, horse_id, config)
                features.update(
                    horse_state.snapshot(
                        event_date, "horse_history__", config, include_workload=True, conditions=conditions
                    )
                    if horse_state is not None
                    else _empty_snapshot(
                        event_date,
                        "horse_history__",
                        config,
                        include_workload=True,
                        conditions=conditions,
                    )
                )
                for state_map, key, prefix in (
                    (jockey_states, jockey_id, "jockey_history__"),
                    (trainer_states, trainer_id, "trainer_history__"),
                ):
                    state = _state_for(state_map, key, config)
                    if state is None:
                        snapshot = empty_connection_snapshots.get(prefix)
                        if snapshot is None:
                            snapshot = _empty_snapshot(
                                event_date, prefix, config, include_workload=False
                            )
                            empty_connection_snapshots[prefix] = snapshot
                    else:
                        cache_key = (prefix, key)
                        snapshot = connection_snapshot_cache.get(cache_key)
                        if snapshot is None:
                            snapshot = state.snapshot(
                                event_date, prefix, config, include_workload=False
                            )
                            connection_snapshot_cache[cache_key] = snapshot
                    features.update(snapshot)

                horse_elo = pre_ratings.get(horse_id, config.initial_elo)
                features.update(
                    {
                        "rating__horse_elo_pre": horse_elo,
                        "rating__field_mean_elo_pre": field_mean_elo,
                        "rating__field_max_elo_pre": field_max_elo,
                        "rating__field_std_elo_pre": field_std_elo,
                        "rating__horse_minus_field_mean_elo": horse_elo - field_mean_elo,
                    }
                )
                if config.surface_conditioned_elo:
                    surface_horse_elo = surface_pre_ratings.get(
                        horse_id, config.initial_elo
                    )
                    features.update(
                        {
                            "surface_rating__horse_elo_pre": surface_horse_elo,
                            "surface_rating__horse_minus_field_mean_elo": (
                                surface_horse_elo - surface_field_mean_elo
                            ),
                        }
                    )
                rows.append(features)

            race_features = _add_field_relative(pd.DataFrame(rows))
            rating_values = race_features["rating__horse_elo_pre"]
            race_features["rating__horse_elo_percentile"] = rating_values.rank(pct=True, method="average")
            if config.surface_conditioned_elo:
                surface_rating_values = race_features[
                    "surface_rating__horse_elo_pre"
                ]
                race_features["surface_rating__horse_elo_percentile"] = (
                    surface_rating_values.rank(pct=True, method="average")
                )
            date_emitted.append(race_features)
            pending_updates.append(
                (
                    race,
                    race_features,
                    pre_ratings,
                    surface_pre_ratings,
                    finish_numeric,
                    starter_flags,
                    race_is_flat,
                    surface_key,
                )
            )

        # Results become state only after features for the complete date exist.
        for (
            race,
            race_features,
            pre_ratings,
            surface_pre_ratings,
            finish_numeric,
            starter_flags,
            race_is_flat,
            surface_key,
        ) in pending_updates:
            if not race_is_flat:
                continue
            active_positions = [
                index for index, flag in enumerate(starter_flags) if flag is not pd.NA and bool(flag)
            ]
            active_finishes = finish_numeric.iloc[active_positions]
            finite_finishes = active_finishes[np.isfinite(active_finishes)]
            worst_finite = float(finite_finishes.max()) if len(finite_finishes) else np.nan
            outcomes = [
                (
                    race.iloc[position][config.horse_id_col],
                    float(finish_numeric.iloc[position]),
                    bool(
                        np.isfinite(float(finish_numeric.iloc[position]))
                        and float(finish_numeric.iloc[position]) == 1.0
                    ),
                )
                for position in active_positions
            ]
            deltas = _race_elo_deltas(outcomes, pre_ratings, config)

            for position in active_positions:
                row = race.iloc[position]
                horse_id = row[config.horse_id_col]
                finish = float(finish_numeric.iloc[position])
                is_win = float(np.isfinite(finish) and finish == 1.0)
                # Both dead-heat winners are official wins.  A downstream binary
                # loss should assign each winner race weight 1/m; this state is a
                # historical official-win count, so it remains one per winner.
                if np.isfinite(finish):
                    if len(finite_finishes) <= 1 or worst_finite <= 1:
                        finish_value = 1.0
                    else:
                        finish_value = 1.0 - (finish - 1.0) / (worst_finite - 1.0)
                else:
                    finish_value = 0.0
                distance = _numeric_scalar(row[config.distance_col])
                venue = race_features.iloc[position]["meta__venue"]
                record = _Performance(
                    date=event_date,
                    win=is_win,
                    finish=finish,
                    distance=float(distance) if pd.notna(distance) else np.nan,
                    surface=row[config.surface_col] if config.surface_col in row.index else np.nan,
                    distance_band=_distance_band(float(distance)) if pd.notna(distance) else np.nan,
                    venue=venue,
                    opponent_mean_elo=float(race_features.iloc[position]["rating__field_mean_elo_pre"]),
                    performance_value=finish_value
                    + (float(race_features.iloc[position]["rating__field_mean_elo_pre"]) - config.initial_elo)
                    / config.elo_scale,
                    elo_surprise=deltas.get(horse_id, np.nan) / config.elo_k,
                )
                for state_map, key in (
                    (horse_states, horse_id),
                    (jockey_states, row[config.jockey_id_col]),
                    (trainer_states, row[config.trainer_id_col]),
                ):
                    state = _state_for(state_map, key, config)
                    if state is not None:
                        state.update(record)
            for horse_id, delta in deltas.items():
                elo_ratings[horse_id] = pre_ratings[horse_id] + delta
            if config.surface_conditioned_elo and surface_key is not None:
                surface_deltas = _race_elo_deltas(
                    outcomes, surface_pre_ratings, config
                )
                for horse_id, delta in surface_deltas.items():
                    surface_elo_ratings[(horse_id, surface_key)] = (
                        surface_pre_ratings[horse_id] + delta
                    )

        date_frame = pd.concat(date_emitted, ignore_index=True)
        date_numeric = [
            column
            for column in model_feature_allowlist(date_frame)
            if pd.api.types.is_numeric_dtype(date_frame[column].dtype)
        ]
        date_frame[date_numeric] = date_frame[date_numeric].astype("float32")
        emitted.append(date_frame)

    result = pd.concat(emitted, ignore_index=True).sort_values("meta__source_position", kind="stable")
    result.index = raw.index
    return result


def feature_groups(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Return generated model columns partitioned by stable feature group."""

    horse_workload_tokens = ("days_since_last_start", "__days_", "__decay_", "distance_sum")
    horse_columns = tuple(column for column in frame if column.startswith("horse_history__"))
    groups = {
        "race_context": tuple(column for column in frame if column.startswith("context__")),
        "horse_history_basic": tuple(
            column for column in horse_columns if not any(token in column for token in horse_workload_tokens)
        ),
        "form_workload": tuple(
            column for column in horse_columns if any(token in column for token in horse_workload_tokens)
        ),
        "connections_pit": tuple(
            column
            for column in frame
            if column.startswith(("jockey_history__", "trainer_history__"))
        ),
        "field_relative": tuple(column for column in frame if column.startswith("field_relative__")),
        "rating_strength": tuple(column for column in frame if column.startswith("rating__")),
        "surface_conditioned_rating": tuple(
            column for column in frame if column.startswith("surface_rating__")
        ),
        "race_value_expected_actual": tuple(
            column for column in frame if column.startswith("race_value__")
        ),
    }
    return groups


def model_feature_allowlist(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the only columns permitted as model input.

    This is intentionally prefix-based and closed-world.  New raw numeric
    columns (including odds, popularity, or outcomes) cannot enter merely
    because their dtype happens to be numeric.
    """

    groups = feature_groups(frame)
    return tuple(column for group in FEATURE_PREFIXES for column in groups[group])


def validate_model_feature_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise when requested model columns are outside the generated allowlist."""

    allowed = set(model_feature_allowlist(frame))
    invalid = sorted(set(columns).difference(allowed))
    if invalid:
        raise ValueError(f"columns outside model feature allowlist: {invalid}")


def semantic_feature_groups_v2(
    feature_columns: Sequence[str],
) -> Mapping[str, tuple[str, ...]]:
    """Partition the MVP allowlist into interpretable ablation families.

    Unlike the original construction-oriented groups, this taxonomy separates
    horse suitability and Elo-derived race value from general horse history.
    Every supplied column must match exactly one family.
    """

    groups: dict[str, list[str]] = {
        "current_context": [],
        "horse_performance": [],
        "form_workload": [],
        "suitability": [],
        "connections": [],
        "field_relative": [],
        "rating_value": [],
    }
    if any(column.startswith("surface_rating__") for column in feature_columns):
        groups["surface_conditioned_rating"] = []
    if any(column.startswith("race_value__") for column in feature_columns):
        groups["race_value_expected_actual"] = []
    for column in feature_columns:
        if column.startswith("context__"):
            group = "current_context"
        elif column.startswith(("jockey_history__", "trainer_history__")):
            group = "connections"
        elif column.startswith("field_relative__"):
            group = "field_relative"
        elif column.startswith("rating__") or column in {
            "horse_history__career__mean_opponent_elo",
            "horse_history__career__mean_performance_value",
        }:
            group = "rating_value"
        elif column.startswith("surface_rating__"):
            group = "surface_conditioned_rating"
        elif column.startswith("race_value__"):
            group = "race_value_expected_actual"
        elif column.startswith("horse_history__same_"):
            group = "suitability"
        elif column.startswith("horse_history__") and any(
            token in column for token in ("__days_", "__decay_")
        ):
            group = "form_workload"
        elif column in {
            "horse_history__career__distance_sum",
            "horse_history__days_since_last_start",
            "horse_history__last_1__distance_sum",
            "horse_history__last_3__distance_sum",
            "horse_history__last_5__distance_sum",
            "horse_history__last_10__distance_sum",
        }:
            group = "form_workload"
        elif column.startswith("horse_history__"):
            group = "horse_performance"
        else:
            raise ValueError(f"feature column does not match taxonomy v2: {column}")
        groups[group].append(column)

    flattened = [column for columns in groups.values() for column in columns]
    if len(flattened) != len(feature_columns) or set(flattened) != set(feature_columns):
        raise AssertionError("taxonomy v2 must partition feature columns exactly once")
    return {name: tuple(columns) for name, columns in groups.items()}


def source_family_knockout_columns(
    feature_groups: Mapping[str, Sequence[str]], family: str
) -> tuple[str, ...]:
    """Include deterministic field-relative descendants in source knockouts."""

    if family not in feature_groups:
        raise ValueError(f"unknown semantic feature family: {family}")
    descendants = {
        "horse_performance": ("field_relative__horse_win_rate__", "field_relative__horse_recent_win_rate__"),
        "form_workload": ("field_relative__horse_rest_days__",),
        "connections": ("field_relative__jockey_win_rate__", "field_relative__trainer_win_rate__"),
    }.get(family, ())
    selected = list(feature_groups[family])
    for column in feature_groups.get("field_relative", ()):
        if column.startswith(descendants):
            selected.append(column)
    return tuple(selected)


def _assign_split_labels(dates: pd.Series, split_config: Mapping[str, object]) -> pd.Series:
    """Assign non-overlapping absolute-date splits, including model_validation.

    The accepted shape is either ``{"splits": {name: interval}}`` or the
    interval mapping itself.  Each interval has ``start`` and optional ``end``
    (inclusive).  This intentionally does not depend on a fixed list of names.
    """

    raw_intervals = split_config.get("splits", split_config)
    if not isinstance(raw_intervals, Mapping):
        raise ValueError("split_config must contain a mapping of date intervals")
    intervals: list[tuple[pd.Timestamp, Optional[pd.Timestamp], str]] = []
    for name, raw_interval in raw_intervals.items():
        if not isinstance(raw_interval, Mapping) or "start" not in raw_interval:
            # Permit unrelated top-level metadata next to interval definitions.
            continue
        start = pd.Timestamp(raw_interval["start"]).normalize()
        end_raw = raw_interval.get("end")
        end = pd.Timestamp(end_raw).normalize() if end_raw is not None else None
        if end is not None and end < start:
            raise ValueError(f"split {name!r} ends before it starts")
        intervals.append((start, end, str(name)))
    if not intervals:
        raise ValueError("split_config has no date intervals")
    intervals.sort(key=lambda interval: interval[0])
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] is None or current[0] <= previous[1]:
            raise ValueError(f"split intervals overlap: {previous[2]!r}, {current[2]!r}")

    parsed = pd.to_datetime(dates, errors="raise").dt.normalize()
    labels = pd.Series(pd.NA, index=dates.index, dtype="string")
    for start, end, name in intervals:
        mask = parsed.ge(start)
        if end is not None:
            mask &= parsed.le(end)
        labels.loc[mask] = name
    return labels


def build_features(
    normalized: pd.DataFrame,
    split_config: Optional[Mapping[str, object]] = None,
    config: Optional[FeatureConfig] = None,
) -> FeatureDataset:
    """Return model-safe features plus complete audit/evaluation metadata.

    ``normalized`` is not destructively reduced: source identifiers, targets,
    final odds, popularity and result fields remain in ``frame``.  The closed
    numeric ``feature_columns`` tuple is the sole model-input contract.
    """

    config = config or FeatureConfig()
    generated = build_pit_features(normalized, config)
    collisions = sorted(set(normalized.columns).intersection(generated.columns))
    if collisions:
        raise ValueError(f"normalized columns collide with generated columns: {collisions}")
    frame = pd.concat([normalized, generated], axis=1, copy=False)
    if split_config is not None:
        frame["split"] = _assign_split_labels(frame[config.date_col], split_config)

    generated_allowed = model_feature_allowlist(generated)
    numeric_columns = tuple(
        column for column in generated_allowed if pd.api.types.is_numeric_dtype(frame[column].dtype)
    )
    numeric_set = set(numeric_columns)
    groups = {
        group: tuple(column for column in columns if column in numeric_set)
        for group, columns in feature_groups(generated).items()
    }
    validate_model_feature_columns(generated, numeric_columns)
    return FeatureDataset(frame=frame, feature_columns=numeric_columns, feature_groups=groups)
