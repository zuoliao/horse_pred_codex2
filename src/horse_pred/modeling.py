"""Modeling primitives for race-grouped horse-racing experiments.

The module deliberately keeps odds out of every prediction-model API.  It is
usable without third-party packages except for the functions that instantiate
LightGBM estimators; those import ``lightgbm`` lazily.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from math import exp, isfinite, log
from typing import Any

_EPSILON = 1e-15

STANDARD_SPLIT_YEARS: dict[str, tuple[int, int]] = {
    "train": (2014, 2021),
    "model_validation": (2022, 2022),
    "calibration": (2023, 2023),
    "development": (2024, 2024),
    "retrospective_test": (2025, 2025),
}

_FORBIDDEN_PRIMARY_FEATURE_FRAGMENTS = (
    "odds",
    "popularity",
    "payout",
    "払戻",
    "オッズ",
    "人気",
)


def _as_list(values: Iterable[Any], name: str) -> list[Any]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _same_length(expected: int, values: Sequence[Any], name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has {len(values)} rows; expected {expected}")


@dataclass(frozen=True)
class GroupStructure:
    """Contiguous race-query structure required by LambdaRank."""

    race_ids: tuple[Any, ...]
    group_sizes: tuple[int, ...]
    row_slices: tuple[tuple[int, int], ...]

    @property
    def row_count(self) -> int:
        return sum(self.group_sizes)


def validate_grouped_rows(race_ids: Iterable[Any]) -> GroupStructure:
    """Validate that every race occupies one non-empty contiguous row block.

    LightGBM's ``group`` argument is positional.  Accidentally returning to a
    race ID after another race would silently describe the wrong queries, so it
    is rejected rather than sorted implicitly.
    """

    ids = _as_list(race_ids, "race_ids")
    seen: set[Any] = set()
    race_order: list[Any] = []
    sizes: list[int] = []
    slices: list[tuple[int, int]] = []
    start = 0
    current = ids[0]

    for index, race_id in enumerate(ids[1:], start=1):
        if race_id == current:
            continue
        if current in seen:
            raise ValueError(f"race_id {current!r} is not contiguous")
        seen.add(current)
        race_order.append(current)
        sizes.append(index - start)
        slices.append((start, index))
        if race_id in seen:
            raise ValueError(f"race_id {race_id!r} is not contiguous")
        current = race_id
        start = index

    if current in seen:
        raise ValueError(f"race_id {current!r} is not contiguous")
    race_order.append(current)
    sizes.append(len(ids) - start)
    slices.append((start, len(ids)))
    return GroupStructure(tuple(race_order), tuple(sizes), tuple(slices))


def validate_race_splits(
    race_ids: Iterable[Any], split_labels: Iterable[Any]
) -> dict[Any, Any]:
    """Require all runners from one race to remain in one temporal split."""

    ids = _as_list(race_ids, "race_ids")
    splits = _as_list(split_labels, "split_labels")
    _same_length(len(ids), splits, "split_labels")
    mapping: dict[Any, Any] = {}
    for race_id, split in zip(ids, splits):
        previous = mapping.setdefault(race_id, split)
        if previous != split:
            raise ValueError(
                f"race_id {race_id!r} spans split labels {previous!r} and {split!r}"
            )
    return mapping


def _year_of(value: Any) -> int:
    if hasattr(value, "year"):
        return int(value.year)
    text = str(value)
    if len(text) < 4 or not text[:4].isdigit():
        raise ValueError(f"cannot extract four-digit year from {value!r}")
    return int(text[:4])


def validate_standard_split_partition(
    race_ids: Iterable[Any],
    split_labels: Iterable[str],
    race_dates: Iterable[Any],
    *,
    require_all_splits: bool = True,
) -> dict[str, int]:
    """Validate the preregistered 2014--2025 chronological split contract."""

    ids = _as_list(race_ids, "race_ids")
    splits = _as_list(split_labels, "split_labels")
    dates = _as_list(race_dates, "race_dates")
    _same_length(len(ids), splits, "split_labels")
    _same_length(len(ids), dates, "race_dates")
    validate_race_splits(ids, splits)
    counts = {name: 0 for name in STANDARD_SPLIT_YEARS}
    for split, date in zip(splits, dates):
        if split not in STANDARD_SPLIT_YEARS:
            raise ValueError(f"unknown standard split label {split!r}")
        year = _year_of(date)
        lower, upper = STANDARD_SPLIT_YEARS[split]
        if not lower <= year <= upper:
            raise ValueError(
                f"year {year} is outside {split!r} range {lower}..{upper}"
            )
        counts[split] += 1
    if require_all_splits:
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise ValueError(f"standard split partition is missing {missing}")
    return counts


def validate_feature_matrix(
    features: Iterable[Sequence[Any]], feature_names: Sequence[str] | None = None
) -> list[list[Any]]:
    """Return a materialized, rectangular feature matrix."""

    rows = [list(row) for row in features]
    if not rows:
        raise ValueError("features must not be empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("features must contain at least one column")
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"features row {index} has width {len(row)}; expected {width}"
            )
    if feature_names is not None and len(feature_names) != width:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries; expected {width}"
        )
    return rows


def validate_prediction_feature_columns(feature_columns: Sequence[str]) -> None:
    """Fail closed on obvious market columns in the no-odds primary model."""

    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    forbidden = [
        column
        for column in feature_columns
        if any(
            fragment in column.lower()
            for fragment in _FORBIDDEN_PRIMARY_FEATURE_FRAGMENTS
        )
    ]
    if forbidden:
        raise ValueError(
            "primary prediction features must exclude market/final-odds columns: "
            f"{forbidden}"
        )


@dataclass(frozen=True)
class ComparisonDataset:
    """One immutable row scope shared by Binary and LambdaRank.

    ``finish_positions`` uses 1 for first place, 2 for second, and so on.
    Dead-heated winners may both have position 1.  Non-finishers should be
    resolved by the upstream outcome contract before constructing this object.
    """

    features: tuple[tuple[Any, ...], ...]
    feature_names: tuple[str, ...]
    race_ids: tuple[Any, ...]
    finish_positions: tuple[int, ...]
    split_labels: tuple[Any, ...]

    @classmethod
    def from_rows(
        cls,
        *,
        features: Iterable[Sequence[Any]],
        feature_names: Sequence[str],
        race_ids: Iterable[Any],
        finish_positions: Iterable[int],
        split_labels: Iterable[Any],
    ) -> ComparisonDataset:
        validate_prediction_feature_columns(feature_names)
        matrix = validate_feature_matrix(features, feature_names)
        ids = _as_list(race_ids, "race_ids")
        positions = _as_list(finish_positions, "finish_positions")
        splits = _as_list(split_labels, "split_labels")
        row_count = len(matrix)
        _same_length(row_count, ids, "race_ids")
        _same_length(row_count, positions, "finish_positions")
        _same_length(row_count, splits, "split_labels")
        validate_grouped_rows(ids)
        validate_race_splits(ids, splits)
        for position in positions:
            if not isinstance(position, int) or position < 1:
                raise ValueError("finish_positions must contain positive integers")
        return cls(
            tuple(tuple(row) for row in matrix),
            tuple(feature_names),
            tuple(ids),
            tuple(positions),
            tuple(splits),
        )

    def select_split(self, split_label: Any) -> ComparisonDataset:
        indices = [
            index
            for index, value in enumerate(self.split_labels)
            if value == split_label
        ]
        if not indices:
            raise ValueError(f"split label {split_label!r} has no rows")
        return ComparisonDataset.from_rows(
            features=[self.features[index] for index in indices],
            feature_names=self.feature_names,
            race_ids=[self.race_ids[index] for index in indices],
            finish_positions=[self.finish_positions[index] for index in indices],
            split_labels=[self.split_labels[index] for index in indices],
        )


def binary_win_targets(finish_positions: Iterable[int]) -> list[int]:
    """Return runner-level marginal win targets (all tied winners are 1)."""

    positions = _as_list(finish_positions, "finish_positions")
    return [1 if position == 1 else 0 for position in positions]


def ranking_relevance_targets(finish_positions: Iterable[int]) -> list[int]:
    """Map finish positions to the fixed LambdaRank relevance 3/2/1/0."""

    positions = _as_list(finish_positions, "finish_positions")
    relevance: list[int] = []
    for position in positions:
        if position < 1:
            raise ValueError("finish_positions must contain positive integers")
        relevance.append({1: 3, 2: 2, 3: 1}.get(position, 0))
    return relevance


def grouped_ranking_relevance_targets(
    finish_positions: Iterable[int],
    group_sizes: Iterable[int],
    *,
    relevance_scheme: str = "top3",
) -> list[int]:
    """Map grouped finish positions under an explicit LambdaRank label scheme.

    ``top3`` is the existing fixed 1st/2nd/3rd relevance mapping.  The
    ``graded_field_half`` candidate retains first place as relevance 3, merges
    second and third at relevance 2, assigns relevance 1 from fourth through
    ``ceil(field_size / 2)``, and assigns zero to the remainder.  The upstream
    DNF sentinel ``field_size + 1`` is consequently always relevance zero.
    """

    positions = _as_list(finish_positions, "finish_positions")
    sizes = _as_list(group_sizes, "group_sizes")
    if relevance_scheme not in {"top3", "graded_field_half"}:
        raise ValueError(
            "relevance_scheme must be 'top3' or 'graded_field_half'"
        )
    for group_size in sizes:
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size < 1
        ):
            raise ValueError("group_sizes must contain positive integers")
    _same_length(sum(sizes), positions, "finish_positions")

    cursor = 0
    for group_size in sizes:
        for position in positions[cursor : cursor + group_size]:
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 1 <= position <= group_size + 1
            ):
                raise ValueError(
                    "finish_positions must contain positive integers no greater "
                    "than their group size plus the DNF sentinel"
                )
        cursor += group_size

    if relevance_scheme == "top3":
        return ranking_relevance_targets(positions)

    relevance: list[int] = []
    cursor = 0
    for group_size in sizes:
        upper_half_cutoff = (group_size + 1) // 2
        for position in positions[cursor : cursor + group_size]:
            if position > group_size:
                value = 0
            elif position == 1:
                value = 3
            elif position in (2, 3):
                value = 2
            elif position <= upper_half_cutoff:
                value = 1
            else:
                value = 0
            relevance.append(value)
        cursor += group_size
    return relevance


def race_balanced_weights(race_ids: Iterable[Any]) -> list[float]:
    """Give each race total training weight one."""

    structure = validate_grouped_rows(race_ids)
    weights: list[float] = []
    for group_size in structure.group_sizes:
        weights.extend([1.0 / group_size] * group_size)
    return weights


def _validate_numeric(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        if not isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
    return result


def normalize_race_probabilities(
    probabilities: Iterable[float], race_ids: Iterable[Any]
) -> list[float]:
    """Normalize non-negative runner values to a coherent vector per race.

    A race with all-zero inputs falls back to the uniform distribution.  This
    behavior is deterministic and should be recorded as part of the experiment
    mapping rather than silently treated as calibrated output.
    """

    values = _validate_numeric(probabilities, "probabilities")
    ids = _as_list(race_ids, "race_ids")
    _same_length(len(ids), values, "probabilities")
    structure = validate_grouped_rows(ids)
    result = [0.0] * len(values)
    for start, end in structure.row_slices:
        group = values[start:end]
        if any(value < 0.0 for value in group):
            raise ValueError("probabilities must be non-negative")
        total = sum(group)
        if total <= 0.0:
            uniform = 1.0 / len(group)
            result[start:end] = [uniform] * len(group)
        else:
            result[start:end] = [value / total for value in group]
    return result


def uniform_probabilities(race_ids: Iterable[Any]) -> list[float]:
    """BASE-01 uniform win-probability baseline."""

    structure = validate_grouped_rows(race_ids)
    result: list[float] = []
    for group_size in structure.group_sizes:
        result.extend([1.0 / group_size] * group_size)
    return result


def history_rate_probabilities(
    history_wins: Iterable[float],
    history_starts: Iterable[float],
    race_ids: Iterable[Any],
    *,
    prior_strength: float = 2.0,
) -> list[float]:
    """BASE-01 smoothed history-win-rate baseline, normalized within race.

    The prior mean is the current race's uniform probability.  Inputs must have
    been computed strictly before the target race; this function cannot verify
    their point-in-time provenance.
    """

    wins = _validate_numeric(history_wins, "history_wins")
    starts = _validate_numeric(history_starts, "history_starts")
    ids = _as_list(race_ids, "race_ids")
    _same_length(len(ids), wins, "history_wins")
    _same_length(len(ids), starts, "history_starts")
    if prior_strength < 0 or not isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and non-negative")
    structure = validate_grouped_rows(ids)
    raw = [0.0] * len(ids)
    for start, end in structure.row_slices:
        prior_mean = 1.0 / (end - start)
        for index in range(start, end):
            if wins[index] < 0 or starts[index] < 0 or wins[index] > starts[index]:
                raise ValueError("history counts require 0 <= wins <= starts")
            denominator = starts[index] + prior_strength
            raw[index] = (
                (wins[index] + prior_strength * prior_mean) / denominator
                if denominator > 0
                else prior_mean
            )
    return normalize_race_probabilities(raw, ids)


def race_softmax(
    scores: Iterable[float], race_ids: Iterable[Any], *, temperature: float = 1.0
) -> list[float]:
    """Map arbitrary runner scores to coherent race probabilities."""

    values = _validate_numeric(scores, "scores")
    ids = _as_list(race_ids, "race_ids")
    _same_length(len(ids), values, "scores")
    if temperature <= 0.0 or not isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    structure = validate_grouped_rows(ids)
    result = [0.0] * len(values)
    for start, end in structure.row_slices:
        scaled = [value / temperature for value in values[start:end]]
        maximum = max(scaled)
        exponentials = [exp(value - maximum) for value in scaled]
        total = sum(exponentials)
        result[start:end] = [value / total for value in exponentials]
    return result


def winner_mass_targets(
    finish_positions: Iterable[int], race_ids: Iterable[Any]
) -> list[float]:
    """Return per-race unit target mass, split evenly across dead-heated winners."""

    positions = _as_list(finish_positions, "finish_positions")
    ids = _as_list(race_ids, "race_ids")
    _same_length(len(ids), positions, "finish_positions")
    structure = validate_grouped_rows(ids)
    targets = [0.0] * len(ids)
    for start, end in structure.row_slices:
        winners = [index for index in range(start, end) if positions[index] == 1]
        if not winners:
            raise ValueError(f"race_id {ids[start]!r} has no winner")
        mass = 1.0 / len(winners)
        for index in winners:
            targets[index] = mass
    return targets


def _race_log_loss(
    probabilities: Sequence[float],
    targets: Sequence[float],
    structure: GroupStructure,
) -> float:
    total = 0.0
    for start, end in structure.row_slices:
        total -= sum(
            targets[index]
            * log(min(max(probabilities[index], _EPSILON), 1.0 - _EPSILON))
            for index in range(start, end)
            if targets[index] > 0.0
        )
    return total / len(structure.group_sizes)


@dataclass
class TemperatureCalibrator:
    """One-parameter race-softmax calibration fitted on an OOT slice."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self,
        scores: Iterable[float],
        race_ids: Iterable[Any],
        finish_positions: Iterable[int],
        *,
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
        iterations: int = 80,
    ) -> TemperatureCalibrator:
        values = _validate_numeric(scores, "scores")
        ids = _as_list(race_ids, "race_ids")
        positions = _as_list(finish_positions, "finish_positions")
        _same_length(len(ids), values, "scores")
        _same_length(len(ids), positions, "finish_positions")
        structure = validate_grouped_rows(ids)
        targets = winner_mass_targets(positions, ids)
        if min_temperature <= 0 or max_temperature <= min_temperature:
            raise ValueError("temperature bounds must satisfy 0 < min < max")

        def objective(log_temperature: float) -> float:
            probabilities = race_softmax(
                values, ids, temperature=exp(log_temperature)
            )
            return _race_log_loss(probabilities, targets, structure)

        left = log(min_temperature)
        right = log(max_temperature)
        ratio = (5.0**0.5 - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = objective(x1)
        f2 = objective(x2)
        for _ in range(iterations):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = objective(x2)
        self.temperature = exp((left + right) / 2.0)
        self.fitted = True
        return self

    def transform(
        self, scores: Iterable[float], race_ids: Iterable[Any]
    ) -> list[float]:
        if not self.fitted:
            raise RuntimeError("TemperatureCalibrator must be fitted before transform")
        return race_softmax(scores, race_ids, temperature=self.temperature)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass
class PlattCalibrator:
    """Simple two-parameter sigmoid calibrator for binary margins/logits."""

    slope: float = 1.0
    intercept: float = 0.0
    l2: float = 1e-6
    fitted: bool = False

    def fit(
        self,
        scores: Iterable[float],
        targets: Iterable[float],
        *,
        sample_weights: Iterable[float] | None = None,
        iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> PlattCalibrator:
        x = _validate_numeric(scores, "scores")
        y = _validate_numeric(targets, "targets")
        _same_length(len(x), y, "targets")
        if any(value < 0.0 or value > 1.0 for value in y):
            raise ValueError("targets must be in [0, 1]")
        if sample_weights is None:
            weights = [1.0] * len(x)
        else:
            weights = _validate_numeric(sample_weights, "sample_weights")
            _same_length(len(x), weights, "sample_weights")
            if any(weight <= 0.0 for weight in weights):
                raise ValueError("sample_weights must be positive")
        if self.l2 < 0.0 or not isfinite(self.l2):
            raise ValueError("l2 must be finite and non-negative")

        slope = self.slope
        positive_weight = sum(w * target for w, target in zip(weights, y))
        negative_weight = sum(w * (1.0 - target) for w, target in zip(weights, y))
        intercept = log((positive_weight + 0.5) / (negative_weight + 0.5))

        def objective(a: float, b: float) -> float:
            loss = 0.5 * self.l2 * a * a
            for value, target, weight in zip(x, y, weights):
                linear = a * value + b
                if linear >= 0.0:
                    loss += weight * (
                        (1.0 - target) * linear + log(1.0 + exp(-linear))
                    )
                else:
                    loss += weight * (
                        -target * linear + log(1.0 + exp(linear))
                    )
            return loss

        current = objective(slope, intercept)
        for _ in range(iterations):
            gradient_a = self.l2 * slope
            gradient_b = 0.0
            hessian_aa = self.l2
            hessian_ab = 0.0
            hessian_bb = 0.0
            for value, target, weight in zip(x, y, weights):
                probability = _sigmoid(slope * value + intercept)
                residual = weight * (probability - target)
                curvature = weight * probability * (1.0 - probability)
                gradient_a += residual * value
                gradient_b += residual
                hessian_aa += curvature * value * value
                hessian_ab += curvature * value
                hessian_bb += curvature
            determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
            if determinant <= _EPSILON:
                break
            step_a = (gradient_a * hessian_bb - gradient_b * hessian_ab) / determinant
            step_b = (gradient_b * hessian_aa - gradient_a * hessian_ab) / determinant
            if max(abs(step_a), abs(step_b)) < tolerance:
                break
            step_scale = 1.0
            accepted = False
            while step_scale >= 1e-8:
                candidate_a = slope - step_scale * step_a
                candidate_b = intercept - step_scale * step_b
                candidate = objective(candidate_a, candidate_b)
                if candidate <= current:
                    slope, intercept, current = candidate_a, candidate_b, candidate
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break
        self.slope = slope
        self.intercept = intercept
        self.fitted = True
        return self

    def transform(self, scores: Iterable[float]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("PlattCalibrator must be fitted before transform")
        values = _validate_numeric(scores, "scores")
        return [_sigmoid(self.slope * value + self.intercept) for value in values]


def probability_logits(
    probabilities: Iterable[float], *, epsilon: float = 1e-6
) -> list[float]:
    """Convert binary probabilities to clipped logits for Platt scaling."""

    values = _validate_numeric(probabilities, "probabilities")
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    result: list[float] = []
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        clipped = min(max(value, epsilon), 1.0 - epsilon)
        result.append(log(clipped / (1.0 - clipped)))
    return result


def coherent_binary_probabilities(
    raw_probabilities: Iterable[float],
    race_ids: Iterable[Any],
    *,
    calibrator: PlattCalibrator | None = None,
    epsilon: float = 1e-6,
) -> list[float]:
    """Optionally Platt-calibrate binary probabilities, then race-normalize."""

    raw = _validate_numeric(raw_probabilities, "raw_probabilities")
    if calibrator is not None:
        mapped = calibrator.transform(probability_logits(raw, epsilon=epsilon))
    else:
        mapped = raw
    return normalize_race_probabilities(mapped, race_ids)


def validate_chronological_calibration(
    model_fit_times: Iterable[Any],
    calibration_times: Iterable[Any],
    evaluation_times: Iterable[Any],
) -> None:
    """Require strictly ordered, non-overlapping model/calibration/eval windows."""

    fit = _as_list(model_fit_times, "model_fit_times")
    calibration = _as_list(calibration_times, "calibration_times")
    evaluation = _as_list(evaluation_times, "evaluation_times")
    if max(fit) >= min(calibration):
        raise ValueError("model-fitting period must end before calibration starts")
    if max(calibration) >= min(evaluation):
        raise ValueError("calibration period must end before evaluation starts")


_BINARY_DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "class_weight": None,
}

_RANKING_DEFAULTS: dict[str, Any] = {
    "objective": "lambdarank",
    "label_gain": [0, 1, 3, 7],
    "lambdarank_truncation_level": 6,
    "lambdarank_norm": True,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
}

_HUBER_DEFAULTS: dict[str, Any] = {
    "objective": "huber",
    "metric": "huber",
    "alpha": 0.9,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
}


def build_lightgbm_estimator(
    model_kind: str, *, params: Mapping[str, Any] | None = None
) -> Any:
    """Create a fixed Binary, LambdaRank, or diagnostic Huber estimator."""

    try:
        import lightgbm as lgb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM execution requires the optional 'lightgbm' dependency"
        ) from exc
    overrides = dict(params or {})
    if model_kind == "binary":
        merged = {**_BINARY_DEFAULTS, **overrides}
        if merged.get("objective") != "binary":
            raise ValueError("binary estimator objective must remain 'binary'")
        if (
            merged.get("class_weight") is not None
            or merged.get("is_unbalance", False)
            or float(merged.get("scale_pos_weight", 1.0)) != 1.0
        ):
            raise ValueError("primary Binary baseline must not use class balancing")
        return lgb.LGBMClassifier(**merged)
    if model_kind == "lambdarank":
        merged = {**_RANKING_DEFAULTS, **overrides}
        if merged.get("objective") != "lambdarank":
            raise ValueError("ranking estimator objective must remain 'lambdarank'")
        if list(merged.get("label_gain", [])) != [0, 1, 3, 7]:
            raise ValueError("primary LambdaRank label_gain must remain [0, 1, 3, 7]")
        return lgb.LGBMRanker(**merged)
    if model_kind == "huber":
        merged = {**_HUBER_DEFAULTS, **overrides}
        if merged.get("objective") != "huber":
            raise ValueError("Huber estimator objective must remain 'huber'")
        alpha = float(merged.get("alpha", 0.9))
        if not 0.0 < alpha < 1.0:
            raise ValueError("Huber alpha must be in (0, 1)")
        return lgb.LGBMRegressor(**merged)
    raise ValueError("model_kind must be 'binary', 'lambdarank', or 'huber'")


def _lightgbm_validation_data(model: Any, features: Any, targets: Any) -> dict[str, Any]:
    """Use LightGBM 4.7's eval_X/eval_y without dropping 4.3 compatibility."""

    if "eval_X" in signature(model.fit).parameters:
        return {"eval_X": features, "eval_y": targets}
    return {"eval_set": [(features, targets)]}


@dataclass(frozen=True)
class LightGBMComparison:
    """Fitted estimators trained on exactly the same row and feature scope."""

    binary_model: Any
    ranking_model: Any
    feature_names: tuple[str, ...]
    race_ids: tuple[Any, ...]
    group_sizes: tuple[int, ...]


def _frame_column(frame: Any, column: str) -> list[Any]:
    """Read a column from a pandas-like frame without importing pandas."""

    try:
        values = frame[column]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"frame is missing required column {column!r}") from exc
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _frame_feature_rows(frame: Any, feature_columns: Sequence[str]) -> list[list[Any]]:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    columns = [_frame_column(frame, column) for column in feature_columns]
    row_count = len(columns[0])
    for column_name, values in zip(feature_columns, columns):
        _same_length(row_count, values, column_name)
    return [list(row) for row in zip(*columns)]


def _frame_feature_matrix(frame: Any, feature_columns: Sequence[str]) -> Any:
    """Return a float32 pandas view/copy when available, else materialize rows."""

    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    if hasattr(frame, "loc"):
        try:
            return frame.loc[:, list(feature_columns)].astype("float32", copy=False)
        except KeyError as exc:
            raise ValueError("frame is missing one or more feature columns") from exc
    return _frame_feature_rows(frame, feature_columns)


def _frame_split_mask(frame: Any, split_column: str, split_label: Any) -> Any:
    """Build a pandas boolean mask without copying unrelated frame columns."""

    if not hasattr(frame, "loc"):
        raise TypeError("large-frame training API requires a pandas-like .loc indexer")
    try:
        mask = frame[split_column] == split_label
    except KeyError as exc:
        raise ValueError(f"frame is missing required column {split_column!r}") from exc
    if int(mask.sum()) == 0:
        raise ValueError(f"split label {split_label!r} has no rows")
    return mask


def _frame_masked_column(frame: Any, mask: Any, column: str) -> list[Any]:
    try:
        values = frame.loc[mask, column]
    except KeyError as exc:
        raise ValueError(f"frame is missing required column {column!r}") from exc
    return list(values.tolist())


def _frame_masked_features(
    frame: Any, mask: Any, feature_columns: Sequence[str]
) -> Any:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    try:
        return frame.loc[mask, list(feature_columns)].astype("float32", copy=False)
    except KeyError as exc:
        raise ValueError("frame is missing one or more feature columns") from exc


def comparison_dataset_from_frame(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    race_id_column: str = "race_id",
    finish_position_column: str = "finish_position",
    split_column: str = "split",
) -> ComparisonDataset:
    """Build the common Binary/LambdaRank scope from a pandas-like frame."""

    validate_prediction_feature_columns(feature_columns)
    return ComparisonDataset.from_rows(
        features=_frame_feature_rows(frame, feature_columns),
        feature_names=feature_columns,
        race_ids=_frame_column(frame, race_id_column),
        finish_positions=[
            int(value) for value in _frame_column(frame, finish_position_column)
        ],
        split_labels=_frame_column(frame, split_column),
    )


def train_binary(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    train_split: Any = "train",
    model_validation_split: Any = "model_validation",
    race_id_column: str = "race_id",
    finish_position_column: str = "finish_position",
    split_column: str = "split",
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int | None = 50,
) -> Any:
    """Fit Binary on 2014--2021, using 2022 only for early stopping."""

    validate_prediction_feature_columns(feature_columns)
    training_mask = _frame_split_mask(frame, split_column, train_split)
    validation_mask = _frame_split_mask(frame, split_column, model_validation_split)
    training_ids = _frame_masked_column(frame, training_mask, race_id_column)
    validation_ids = _frame_masked_column(frame, validation_mask, race_id_column)
    validate_grouped_rows(training_ids)
    validate_grouped_rows(validation_ids)
    training_positions = [
        int(value)
        for value in _frame_masked_column(frame, training_mask, finish_position_column)
    ]
    validation_positions = [
        int(value)
        for value in _frame_masked_column(frame, validation_mask, finish_position_column)
    ]
    training_features = _frame_masked_features(frame, training_mask, feature_columns)
    validation_features = _frame_masked_features(frame, validation_mask, feature_columns)
    model = build_lightgbm_estimator("binary", params=params)
    fit_options: dict[str, Any] = {
        "sample_weight": race_balanced_weights(training_ids),
        "feature_name": list(feature_columns),
        "eval_sample_weight": [race_balanced_weights(validation_ids)],
        "eval_metric": "binary_logloss",
        **_lightgbm_validation_data(
            model, validation_features, binary_win_targets(validation_positions)
        ),
    }
    if early_stopping_rounds is not None:
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive or None")
        import lightgbm as lgb  # type: ignore[import-not-found]

        fit_options["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False)
        ]
    model.fit(
        training_features,
        binary_win_targets(training_positions),
        **fit_options,
    )
    return model


def train_ranker(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    train_split: Any = "train",
    model_validation_split: Any = "model_validation",
    race_id_column: str = "race_id",
    finish_position_column: str = "finish_position",
    split_column: str = "split",
    relevance_scheme: str = "top3",
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int | None = 50,
) -> Any:
    """Fit LambdaRank on 2014--2021, using 2022 only for early stopping."""

    validate_prediction_feature_columns(feature_columns)
    training_mask = _frame_split_mask(frame, split_column, train_split)
    validation_mask = _frame_split_mask(frame, split_column, model_validation_split)
    training_ids = _frame_masked_column(frame, training_mask, race_id_column)
    validation_ids = _frame_masked_column(frame, validation_mask, race_id_column)
    groups = validate_grouped_rows(training_ids)
    validation_groups = validate_grouped_rows(validation_ids)
    training_positions = [
        int(value)
        for value in _frame_masked_column(frame, training_mask, finish_position_column)
    ]
    validation_positions = [
        int(value)
        for value in _frame_masked_column(frame, validation_mask, finish_position_column)
    ]
    training_features = _frame_masked_features(frame, training_mask, feature_columns)
    validation_features = _frame_masked_features(frame, validation_mask, feature_columns)
    estimator_params = dict(params or {})
    eval_at = list(estimator_params.pop("eval_at", [1, 3, 5]))
    model = build_lightgbm_estimator("lambdarank", params=estimator_params)
    fit_options: dict[str, Any] = {
        "group": list(groups.group_sizes),
        "feature_name": list(feature_columns),
        "eval_group": [list(validation_groups.group_sizes)],
        "eval_at": eval_at,
        "eval_metric": "ndcg",
        **_lightgbm_validation_data(
            model,
            validation_features,
            grouped_ranking_relevance_targets(
                validation_positions,
                validation_groups.group_sizes,
                relevance_scheme=relevance_scheme,
            ),
        ),
    }
    if early_stopping_rounds is not None:
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive or None")
        import lightgbm as lgb  # type: ignore[import-not-found]

        fit_options["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False)
        ]
    model.fit(
        training_features,
        grouped_ranking_relevance_targets(
            training_positions,
            groups.group_sizes,
            relevance_scheme=relevance_scheme,
        ),
        **fit_options,
    )
    return model


def train_huber_regressor(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    train_split: Any = "train",
    model_validation_split: Any = "model_validation",
    race_id_column: str = "race_id",
    split_column: str = "split",
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int | None = 50,
) -> Any:
    """Fit the preregistered continuous-performance Huber model.

    Missing performance targets are excluded from model fitting only.  Callers
    retain the complete race choice sets when scoring calibration/evaluation.
    """

    validate_prediction_feature_columns(feature_columns)
    training_role = _frame_split_mask(frame, split_column, train_split)
    validation_role = _frame_split_mask(frame, split_column, model_validation_split)
    try:
        training_target = frame[target_column].astype(float)
        validation_target = frame[target_column].astype(float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"frame is missing numeric target column {target_column!r}") from exc
    training_mask = training_role & training_target.map(isfinite)
    validation_mask = validation_role & validation_target.map(isfinite)
    if int(training_mask.sum()) == 0:
        raise ValueError("Huber train split has no finite performance targets")
    if int(validation_mask.sum()) == 0:
        raise ValueError("Huber validation split has no finite performance targets")
    training_ids = _frame_masked_column(frame, training_mask, race_id_column)
    validation_ids = _frame_masked_column(frame, validation_mask, race_id_column)
    validate_grouped_rows(training_ids)
    validate_grouped_rows(validation_ids)
    training_features = _frame_masked_features(frame, training_mask, feature_columns)
    validation_features = _frame_masked_features(frame, validation_mask, feature_columns)
    training_values = [
        float(value) for value in _frame_masked_column(frame, training_mask, target_column)
    ]
    validation_values = [
        float(value) for value in _frame_masked_column(frame, validation_mask, target_column)
    ]
    model = build_lightgbm_estimator("huber", params=params)
    fit_options: dict[str, Any] = {
        "sample_weight": race_balanced_weights(training_ids),
        "feature_name": list(feature_columns),
        "eval_sample_weight": [race_balanced_weights(validation_ids)],
        "eval_metric": "huber",
        **_lightgbm_validation_data(model, validation_features, validation_values),
    }
    if early_stopping_rounds is not None:
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive or None")
        import lightgbm as lgb  # type: ignore[import-not-found]

        fit_options["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False)
        ]
    model.fit(training_features, training_values, **fit_options)
    return model


def predict(
    model: Any,
    frame: Any,
    *,
    feature_columns: Sequence[str],
    model_kind: str,
) -> list[float]:
    """Predict raw binary probabilities, ranking scores, or Huber utilities."""

    validate_prediction_feature_columns(feature_columns)
    features = _frame_feature_matrix(frame, feature_columns)
    if model_kind == "binary":
        rows = model.predict_proba(features)
        return [float(row[1]) for row in rows]
    if model_kind == "lambdarank":
        return [float(value) for value in model.predict(features)]
    if model_kind == "huber":
        return [float(value) for value in model.predict(features)]
    raise ValueError("model_kind must be 'binary', 'lambdarank', or 'huber'")


def fit_temperature(
    scores: Iterable[float],
    race_ids: Iterable[Any],
    finish_positions: Iterable[int],
    **fit_options: Any,
) -> TemperatureCalibrator:
    """Fit and return a race-softmax temperature calibrator."""

    return TemperatureCalibrator().fit(
        scores, race_ids, finish_positions, **fit_options
    )


def fit_temperature_from_frame(
    frame: Any,
    *,
    score_column: str,
    race_id_column: str = "race_id",
    finish_position_column: str = "finish_position",
    split_column: str = "split",
    calibration_split: Any = "calibration",
    **fit_options: Any,
) -> TemperatureCalibrator:
    """Fit temperature using only the preregistered 2023 calibration rows."""

    mask = _frame_split_mask(frame, split_column, calibration_split)
    scores = _frame_masked_column(frame, mask, score_column)
    race_ids = _frame_masked_column(frame, mask, race_id_column)
    positions = _frame_masked_column(frame, mask, finish_position_column)
    return fit_temperature(
        scores,
        race_ids,
        [int(position) for position in positions],
        **fit_options,
    )


def fit_platt_from_frame(
    frame: Any,
    *,
    score_column: str,
    race_id_column: str = "race_id",
    finish_position_column: str = "finish_position",
    split_column: str = "split",
    calibration_split: Any = "calibration",
    input_kind: str = "probability",
    epsilon: float = 1e-6,
    l2: float = 1e-6,
    iterations: int = 100,
    tolerance: float = 1e-9,
) -> PlattCalibrator:
    """Fit race-balanced Platt scaling on only the 2023 calibration split."""

    mask = _frame_split_mask(frame, split_column, calibration_split)
    scores = [float(value) for value in _frame_masked_column(frame, mask, score_column)]
    race_ids = _frame_masked_column(frame, mask, race_id_column)
    positions = [
        int(value)
        for value in _frame_masked_column(frame, mask, finish_position_column)
    ]
    validate_grouped_rows(race_ids)
    if input_kind == "probability":
        inputs = probability_logits(scores, epsilon=epsilon)
    elif input_kind == "margin":
        inputs = scores
    else:
        raise ValueError("input_kind must be 'probability' or 'margin'")
    return PlattCalibrator(l2=l2).fit(
        inputs,
        binary_win_targets(positions),
        sample_weights=race_balanced_weights(race_ids),
        iterations=iterations,
        tolerance=tolerance,
    )


def apply_temperature(
    calibrator: TemperatureCalibrator,
    scores: Iterable[float],
    race_ids: Iterable[Any],
) -> list[float]:
    """Apply a fitted temperature map to arbitrary race-grouped scores."""

    return calibrator.transform(scores, race_ids)


def uniform_baseline(
    frame_or_race_ids: Any, *, race_id_column: str = "race_id"
) -> list[float]:
    """DataFrame-friendly alias for the uniform BASE-01 baseline."""

    if hasattr(frame_or_race_ids, "columns") or isinstance(frame_or_race_ids, Mapping):
        race_ids = _frame_column(frame_or_race_ids, race_id_column)
    else:
        race_ids = list(frame_or_race_ids)
    return uniform_probabilities(race_ids)


def fit_lightgbm_pair(
    dataset: ComparisonDataset,
    *,
    train_split: Any = "train",
    model_validation_split: Any = "model_validation",
    binary_params: Mapping[str, Any] | None = None,
    ranking_params: Mapping[str, Any] | None = None,
) -> LightGBMComparison:
    """Fit Binary and LambdaRank on one shared feature/split selection.

    Odds are intentionally absent from this API.  Evaluation and calibration
    data are also kept out so callers cannot accidentally fit on them.
    """

    training = dataset.select_split(train_split)
    validation = dataset.select_split(model_validation_split)
    group = validate_grouped_rows(training.race_ids)
    validation_group = validate_grouped_rows(validation.race_ids)
    features = [list(row) for row in training.features]
    binary_model = build_lightgbm_estimator("binary", params=binary_params)
    ranking_model = build_lightgbm_estimator("lambdarank", params=ranking_params)
    import lightgbm as lgb  # type: ignore[import-not-found]

    binary_model.fit(
        features,
        binary_win_targets(training.finish_positions),
        sample_weight=race_balanced_weights(training.race_ids),
        feature_name=list(training.feature_names),
        eval_sample_weight=[race_balanced_weights(validation.race_ids)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
        **_lightgbm_validation_data(
            binary_model,
            [list(row) for row in validation.features],
            binary_win_targets(validation.finish_positions),
        ),
    )
    ranking_model.fit(
        features,
        ranking_relevance_targets(training.finish_positions),
        group=list(group.group_sizes),
        feature_name=list(training.feature_names),
        eval_group=[list(validation_group.group_sizes)],
        eval_at=[1, 3, 5],
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(50, verbose=False)],
        **_lightgbm_validation_data(
            ranking_model,
            [list(row) for row in validation.features],
            ranking_relevance_targets(validation.finish_positions),
        ),
    )
    return LightGBMComparison(
        binary_model=binary_model,
        ranking_model=ranking_model,
        feature_names=training.feature_names,
        race_ids=training.race_ids,
        group_sizes=group.group_sizes,
    )
