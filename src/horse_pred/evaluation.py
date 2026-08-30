"""Race-aware evaluation for prediction models.

Final odds are accepted only by :func:`final_odds_oracle_diagnostic`.  That
function contains no selection, threshold, staking, or ROI interface by design.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite, log, log2
from typing import Any

_EPSILON = 1e-15


def _materialize(values: Iterable[Any], name: str) -> list[Any]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _group_slices(race_ids: Iterable[Any]) -> tuple[list[Any], list[tuple[int, int]]]:
    ids = _materialize(race_ids, "race_ids")
    seen: set[Any] = set()
    slices: list[tuple[int, int]] = []
    start = 0
    current = ids[0]
    for index, race_id in enumerate(ids[1:], start=1):
        if race_id == current:
            continue
        if current in seen or race_id in seen:
            raise ValueError("race_ids must form contiguous race groups")
        seen.add(current)
        slices.append((start, index))
        start = index
        current = race_id
    if current in seen:
        raise ValueError("race_ids must form contiguous race groups")
    slices.append((start, len(ids)))
    return ids, slices


def _numeric(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _check_length(expected: int, values: Sequence[Any], name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has {len(values)} rows; expected {expected}")


def winner_mass_targets(
    finish_positions: Iterable[int], race_ids: Iterable[Any]
) -> list[float]:
    """Allocate one unit of target mass per race, including dead heats."""

    positions = _materialize(finish_positions, "finish_positions")
    ids, slices = _group_slices(race_ids)
    _check_length(len(ids), positions, "finish_positions")
    target = [0.0] * len(ids)
    for start, end in slices:
        winners = [index for index in range(start, end) if positions[index] == 1]
        if not winners:
            raise ValueError(f"race_id {ids[start]!r} has no winner")
        mass = 1.0 / len(winners)
        for index in winners:
            target[index] = mass
    return target


def probability_coherence(
    probabilities: Iterable[float], race_ids: Iterable[Any], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Summarize and validate the race-wise probability-sum constraint."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    _check_length(len(ids), values, "probabilities")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be in [0, 1]")
    sums = [sum(values[start:end]) for start, end in slices]
    errors = [abs(value - 1.0) for value in sums]
    return {
        "race_count": len(slices),
        "min_sum": min(sums),
        "max_sum": max(sums),
        "mean_sum": sum(sums) / len(sums),
        "max_abs_error": max(errors),
        "coherent": all(error <= tolerance for error in errors),
        "tolerance": tolerance,
    }


def _require_coherent(probabilities: Sequence[float], race_ids: Sequence[Any]) -> None:
    summary = probability_coherence(probabilities, race_ids)
    if not summary["coherent"]:
        raise ValueError(
            "probabilities must sum to one within every race; "
            f"max error={summary['max_abs_error']:.6g}"
        )


def race_log_loss(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    epsilon: float = _EPSILON,
) -> float:
    """Macro-average winner-mass cross-entropy over races."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    _require_coherent(values, ids)
    target = winner_mass_targets(positions, ids)
    losses = []
    for start, end in slices:
        losses.append(
            -sum(
                target[index]
                * log(min(max(values[index], epsilon), 1.0 - epsilon))
                for index in range(start, end)
                if target[index] > 0.0
            )
        )
    return sum(losses) / len(losses)


def race_brier_score(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
) -> float:
    """Macro-average multiclass Brier score without a one-half factor."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    _require_coherent(values, ids)
    target = winner_mass_targets(positions, ids)
    per_race = [
        sum((values[index] - target[index]) ** 2 for index in range(start, end))
        for start, end in slices
    ]
    return sum(per_race) / len(per_race)


def runner_binary_scores(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    epsilon: float = _EPSILON,
) -> dict[str, float]:
    """Return runner-micro and race-macro binary Log Loss/Brier."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be in [0, 1]")
    target = [1.0 if position == 1 else 0.0 for position in positions]

    def row_log(index: int) -> float:
        probability = min(max(values[index], epsilon), 1.0 - epsilon)
        return -(
            target[index] * log(probability)
            + (1.0 - target[index]) * log(1.0 - probability)
        )

    row_logs = [row_log(index) for index in range(len(ids))]
    row_briers = [(values[index] - target[index]) ** 2 for index in range(len(ids))]
    race_logs = [
        sum(row_logs[start:end]) / (end - start) for start, end in slices
    ]
    race_briers = [
        sum(row_briers[start:end]) / (end - start) for start, end in slices
    ]
    return {
        "runner_micro_log_loss": sum(row_logs) / len(row_logs),
        "runner_micro_brier": sum(row_briers) / len(row_briers),
        "race_macro_binary_log_loss": sum(race_logs) / len(race_logs),
        "race_macro_binary_brier": sum(race_briers) / len(race_briers),
    }


def _relevance(position: int) -> int:
    return {1: 3, 2: 2, 3: 1}.get(position, 0)


def ndcg_at_k(
    scores: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    k: int,
) -> float:
    """Macro-average NDCG using relevance 3/2/1/0 and gains 7/3/1/0."""

    if k < 1:
        raise ValueError("k must be positive")
    values = _numeric(scores, "scores")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "scores")
    _check_length(len(ids), positions, "finish_positions")

    def dcg(relevances: Sequence[int]) -> float:
        return sum(
            ((2**relevance) - 1.0) / log2(rank + 2.0)
            for rank, relevance in enumerate(relevances[:k])
        )

    race_values: list[float] = []
    for start, end in slices:
        order = sorted(range(start, end), key=lambda index: (-values[index], index))
        actual = [_relevance(positions[index]) for index in order]
        ideal = sorted(
            (_relevance(positions[index]) for index in range(start, end)),
            reverse=True,
        )
        denominator = dcg(ideal)
        race_values.append(dcg(actual) / denominator if denominator > 0.0 else 0.0)
    return sum(race_values) / len(race_values)


def top_k_winner_mass(
    scores: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    k: int,
) -> float:
    """Average winner target mass captured by the top-k ranking."""

    if k < 1:
        raise ValueError("k must be positive")
    values = _numeric(scores, "scores")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "scores")
    _check_length(len(ids), positions, "finish_positions")
    target = winner_mass_targets(positions, ids)
    captured = []
    for start, end in slices:
        order = sorted(range(start, end), key=lambda index: (-values[index], index))
        captured.append(sum(target[index] for index in order[:k]))
    return sum(captured) / len(captured)


def reliability_table(
    probabilities: Iterable[float],
    outcomes: Iterable[float],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    sample_weights: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute a versioned reliability table and descriptive ECE."""

    predicted = _numeric(probabilities, "probabilities")
    observed = _numeric(outcomes, "outcomes")
    _check_length(len(predicted), observed, "outcomes")
    if any(value < 0.0 or value > 1.0 for value in predicted + observed):
        raise ValueError("probabilities and outcomes must be in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if sample_weights is None:
        weights = [1.0] * len(predicted)
    else:
        weights = _numeric(sample_weights, "sample_weights")
        _check_length(len(predicted), weights, "sample_weights")
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("sample_weights must be positive")

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    if strategy == "fixed":
        for index, value in enumerate(predicted):
            bin_index = min(int(value * n_bins), n_bins - 1)
            bins[bin_index].append(index)
    elif strategy == "quantile":
        order = sorted(range(len(predicted)), key=lambda index: (predicted[index], index))
        for rank, index in enumerate(order):
            bin_index = min(rank * n_bins // len(order), n_bins - 1)
            bins[bin_index].append(index)
    else:
        raise ValueError("strategy must be 'quantile' or 'fixed'")

    rows: list[dict[str, Any]] = []
    total_weight = sum(weights)
    ece = 0.0
    for bin_index, indices in enumerate(bins):
        if not indices:
            continue
        bin_weight = sum(weights[index] for index in indices)
        mean_prediction = sum(
            weights[index] * predicted[index] for index in indices
        ) / bin_weight
        observed_rate = sum(
            weights[index] * observed[index] for index in indices
        ) / bin_weight
        gap = observed_rate - mean_prediction
        ece += (bin_weight / total_weight) * abs(gap)
        rows.append(
            {
                "bin": bin_index,
                "count": len(indices),
                "weight": bin_weight,
                "lower_prediction": min(predicted[index] for index in indices),
                "upper_prediction": max(predicted[index] for index in indices),
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "gap": gap,
            }
        )
    return {
        "strategy": strategy,
        "requested_bins": n_bins,
        "nonempty_bins": len(rows),
        "ece": ece,
        "bins": rows,
    }


def race_balanced_reliability(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, Any]:
    """Reliability where every race contributes total sample weight one."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    target = winner_mass_targets(positions, ids)
    weights = [0.0] * len(ids)
    for start, end in slices:
        weights[start:end] = [1.0 / (end - start)] * (end - start)
    return reliability_table(
        values,
        target,
        n_bins=n_bins,
        strategy=strategy,
        sample_weights=weights,
    )


def conditional_race_metrics(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    conditions: Mapping[str, Iterable[Any]],
    *,
    ranking_scores: Iterable[float] | None = None,
    min_races: int = 1,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate race-constant conditions such as surface, class, and field band."""

    values = _numeric(probabilities, "probabilities")
    ids, slices = _group_slices(race_ids)
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), values, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    ranks = values if ranking_scores is None else _numeric(ranking_scores, "ranking_scores")
    _check_length(len(ids), ranks, "ranking_scores")
    if min_races < 1:
        raise ValueError("min_races must be positive")

    result: dict[str, dict[str, dict[str, float]]] = {}
    for condition_name, raw_labels in conditions.items():
        labels = list(raw_labels)
        _check_length(len(ids), labels, condition_name)
        groups: dict[str, list[tuple[int, int]]] = {}
        for start, end in slices:
            race_labels = labels[start:end]
            if any(label != race_labels[0] for label in race_labels[1:]):
                raise ValueError(
                    f"condition {condition_name!r} must be constant within a race"
                )
            groups.setdefault(str(race_labels[0]), []).append((start, end))
        condition_result: dict[str, dict[str, float]] = {}
        for label, selected_slices in groups.items():
            if len(selected_slices) < min_races:
                continue
            indices = [
                index
                for start, end in selected_slices
                for index in range(start, end)
            ]
            subset_ids = [ids[index] for index in indices]
            subset_probabilities = [values[index] for index in indices]
            subset_positions = [positions[index] for index in indices]
            subset_ranks = [ranks[index] for index in indices]
            condition_result[label] = {
                "race_count": float(len(selected_slices)),
                "log_loss": race_log_loss(
                    subset_probabilities, subset_positions, subset_ids
                ),
                "brier": race_brier_score(
                    subset_probabilities, subset_positions, subset_ids
                ),
                "ndcg_at_3": ndcg_at_k(
                    subset_ranks, subset_positions, subset_ids, k=3
                ),
                "top_1": top_k_winner_mass(
                    subset_ranks, subset_positions, subset_ids, k=1
                ),
            }
        result[condition_name] = condition_result
    return result


def _normalize_implied_probabilities(
    final_odds: Sequence[float], slices: Sequence[tuple[int, int]]
) -> tuple[list[float], list[float]]:
    market = [0.0] * len(final_odds)
    overrounds: list[float] = []
    for start, end in slices:
        inverse = [1.0 / final_odds[index] for index in range(start, end)]
        total = sum(inverse)
        overrounds.append(total)
        market[start:end] = [value / total for value in inverse]
    return market, overrounds


def final_odds_oracle_diagnostic(
    probabilities: Iterable[float],
    final_odds: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    odds_band_edges: Sequence[float] = (1.0, 3.0, 10.0, 30.0, float("inf")),
) -> dict[str, Any]:
    """Compare a model with normalized final-odds probabilities after the race.

    This diagnostic intentionally exposes no bet selection, expected-value
    threshold, stake, profit, or ROI.  Final odds were unavailable at the
    historical decision time and must never select the evaluated runners.
    """

    model = _numeric(probabilities, "probabilities")
    odds = _numeric(final_odds, "final_odds")
    positions = _materialize(finish_positions, "finish_positions")
    ids, slices = _group_slices(race_ids)
    _check_length(len(ids), model, "probabilities")
    _check_length(len(ids), odds, "final_odds")
    _check_length(len(ids), positions, "finish_positions")
    _require_coherent(model, ids)
    if any(value < 1.0 for value in odds):
        raise ValueError("final_odds must be finite decimal odds at least 1")
    if len(odds_band_edges) < 2 or odds_band_edges[0] > min(odds):
        raise ValueError("odds_band_edges must cover all supplied final odds")
    if any(
        odds_band_edges[index] >= odds_band_edges[index + 1]
        for index in range(len(odds_band_edges) - 1)
    ):
        raise ValueError("odds_band_edges must be strictly increasing")

    market, overrounds = _normalize_implied_probabilities(odds, slices)
    target = winner_mass_targets(positions, ids)
    bands: list[dict[str, Any]] = []
    for lower, upper in zip(odds_band_edges[:-1], odds_band_edges[1:]):
        indices = [
            index
            for index, value in enumerate(odds)
            if lower <= value < upper
        ]
        if not indices:
            continue
        bands.append(
            {
                "lower": lower,
                "upper": upper,
                "runner_count": len(indices),
                "winner_mass": sum(target[index] for index in indices),
                "mean_model_probability": sum(model[index] for index in indices)
                / len(indices),
                "mean_market_probability": sum(market[index] for index in indices)
                / len(indices),
                "empirical_win_rate": sum(target[index] for index in indices)
                / len(indices),
            }
        )
    model_log_loss = race_log_loss(model, positions, ids)
    market_log_loss = race_log_loss(market, positions, ids)
    model_brier = race_brier_score(model, positions, ids)
    market_brier = race_brier_score(market, positions, ids)
    return {
        "usage": (
            "post-event final-odds oracle diagnostic only; final odds MUST NOT "
            "be used for runner selection, EV thresholds, staking, or executable ROI"
        ),
        "race_count": len(slices),
        "mean_inverse_odds_sum": sum(overrounds) / len(overrounds),
        "model_log_loss": model_log_loss,
        "market_log_loss": market_log_loss,
        "model_minus_market_log_loss": model_log_loss - market_log_loss,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_minus_market_brier": model_brier - market_brier,
        "odds_bands": bands,
    }


def evaluate_predictions(
    probabilities: Iterable[float],
    finish_positions: Iterable[int],
    race_ids: Iterable[Any],
    *,
    ranking_scores: Iterable[float] | None = None,
    conditions: Mapping[str, Iterable[Any]] | None = None,
    reliability_bins: int = 10,
) -> dict[str, Any]:
    """Create the core EVAL-01 metric payload from one prediction vector."""

    model = _numeric(probabilities, "probabilities")
    ids = _materialize(race_ids, "race_ids")
    positions = _materialize(finish_positions, "finish_positions")
    _check_length(len(ids), model, "probabilities")
    _check_length(len(ids), positions, "finish_positions")
    ranks = model if ranking_scores is None else _numeric(ranking_scores, "ranking_scores")
    _check_length(len(ids), ranks, "ranking_scores")
    coherence = probability_coherence(model, ids)
    if not coherence["coherent"]:
        raise ValueError("EVAL-01 requires coherent race probabilities")
    payload: dict[str, Any] = {
        "coherence": coherence,
        "ranking": {
            "ndcg_at_1": ndcg_at_k(ranks, positions, ids, k=1),
            "ndcg_at_3": ndcg_at_k(ranks, positions, ids, k=3),
            "ndcg_at_5": ndcg_at_k(ranks, positions, ids, k=5),
            "top_1": top_k_winner_mass(ranks, positions, ids, k=1),
            "top_3": top_k_winner_mass(ranks, positions, ids, k=3),
        },
        "probability": {
            "race_log_loss": race_log_loss(model, positions, ids),
            "race_brier": race_brier_score(model, positions, ids),
            **runner_binary_scores(model, positions, ids),
        },
        "reliability": race_balanced_reliability(
            model,
            positions,
            ids,
            n_bins=reliability_bins,
            strategy="quantile",
        ),
    }
    if conditions:
        payload["conditional"] = conditional_race_metrics(
            model,
            positions,
            ids,
            conditions,
            ranking_scores=ranks,
        )
    return payload


def _frame_column(frame: Any, column: str) -> list[Any]:
    try:
        values = frame[column]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"frame is missing required column {column!r}") from exc
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def evaluate_prediction_frame(
    frame: Any,
    *,
    probability_column: str,
    finish_position_column: str = "finish_position",
    race_id_column: str = "race_id",
    ranking_score_column: str | None = None,
    condition_columns: Sequence[str] = (),
    final_odds_column: str | None = None,
    split_column: str = "split",
    evaluation_split: str = "development",
    reliability_bins: int = 10,
) -> dict[str, Any]:
    """DataFrame-friendly EVAL-01 entry point.

    Supplying ``final_odds_column`` only appends the post-event oracle
    diagnostic; it never changes runner selection or the core metrics.
    """

    if evaluation_split not in {"development", "retrospective_test"}:
        raise ValueError(
            "evaluation_split must be 'development' or 'retrospective_test'"
        )
    if hasattr(frame, "loc"):
        try:
            mask = frame[split_column] == evaluation_split
        except KeyError as exc:
            raise ValueError(f"frame is missing required column {split_column!r}") from exc
        if int(mask.sum()) == 0:
            raise ValueError(f"evaluation split {evaluation_split!r} has no rows")

        def selected(column: str) -> list[Any]:
            try:
                return list(frame.loc[mask, column].tolist())
            except KeyError as exc:
                raise ValueError(f"frame is missing required column {column!r}") from exc

    else:
        split_values = _frame_column(frame, split_column)
        indices = [
            index for index, split in enumerate(split_values) if split == evaluation_split
        ]
        if not indices:
            raise ValueError(f"evaluation split {evaluation_split!r} has no rows")

        def selected(column: str) -> list[Any]:
            values = _frame_column(frame, column)
            return [values[index] for index in indices]

    probabilities = selected(probability_column)
    positions = [int(value) for value in selected(finish_position_column)]
    race_ids = selected(race_id_column)
    ranking_scores = None if ranking_score_column is None else selected(ranking_score_column)
    conditions = {column: selected(column) for column in condition_columns}
    payload = evaluate_predictions(
        probabilities,
        positions,
        race_ids,
        ranking_scores=ranking_scores,
        conditions=conditions,
        reliability_bins=reliability_bins,
    )
    if final_odds_column is not None:
        payload["final_odds_oracle"] = final_odds_oracle_diagnostic(
            probabilities,
            selected(final_odds_column),
            positions,
            race_ids,
        )
    return payload
