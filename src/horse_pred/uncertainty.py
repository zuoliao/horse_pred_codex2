"""2024-only, race-aware uncertainty analysis for baseline predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.evaluation import (
    ndcg_at_k,
    race_brier_score,
    race_log_loss,
    top_k_winner_mass,
)

PRIMARY_METRICS = ("ndcg_at_3", "top_1_winner_mass", "race_log_loss", "race_brier")
HIGHER_IS_BETTER = frozenset({"ndcg_at_3", "top_1_winner_mass"})


@dataclass(frozen=True)
class ModelSpec:
    """Columns used for the probability and ranking sides of one method."""

    probability: str
    ranking: str


DEFAULT_MODEL_SPECS: Mapping[str, ModelSpec] = {
    "uniform": ModelSpec("prob_uniform", "prob_uniform"),
    "history_rate": ModelSpec("prob_history_rate", "prob_history_rate"),
    "binary": ModelSpec(
        "prob_binary_logit_softmax_temperature_2023", "pred_binary_raw"
    ),
    "lambdarank": ModelSpec(
        "prob_lambdarank_softmax_temperature_2023", "score_lambdarank"
    ),
}


def development_race_metric_table(
    frame: pd.DataFrame,
    *,
    model_specs: Mapping[str, ModelSpec] = DEFAULT_MODEL_SPECS,
) -> pd.DataFrame:
    """Return additive per-race metrics after enforcing the 2024 firewall."""

    required = {
        "race_id",
        "race_date",
        "split",
        "model_finish_position",
        "course_type",
        "distance",
        "race_class",
        "field_size",
    }
    required.update(spec.probability for spec in model_specs.values())
    required.update(spec.ranking for spec in model_specs.values())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")

    source = frame.copy()
    source["race_id"] = source["race_id"].astype("string")
    source["race_date"] = pd.to_datetime(source["race_date"], errors="raise").dt.normalize()
    if source.groupby("race_id", sort=False)["split"].nunique(dropna=False).gt(1).any():
        raise ValueError("a race crosses prediction splits")
    if source.groupby("race_id", sort=False)["race_date"].nunique(dropna=False).gt(1).any():
        raise ValueError("a race crosses dates")

    selected = source.loc[source["split"].eq("development")].copy()
    if selected.empty:
        raise ValueError("prediction frame has no development rows")
    first = pd.Timestamp("2024-01-01")
    last = pd.Timestamp("2024-12-31")
    if not selected["race_date"].between(first, last, inclusive="both").all():
        raise ValueError("development contains a date outside calendar year 2024")
    if selected["race_id"].str[:4].ne("2024").any():
        raise ValueError("development contains a race ID outside year 2024")

    rows: list[dict[str, Any]] = []
    for race_id, race in selected.groupby("race_id", sort=False):
        dates = race["race_date"].unique()
        if len(dates) != 1:
            raise ValueError(f"race {race_id} has nonconstant race_date")
        for column in ("course_type", "distance", "race_class", "field_size"):
            if race[column].nunique(dropna=False) != 1:
                raise ValueError(f"race {race_id} has nonconstant {column}")
        positions = pd.to_numeric(race["model_finish_position"], errors="raise").astype(int)
        ids = race["race_id"].tolist()
        base = _race_metadata(race.iloc[0])
        base["race_id"] = str(race_id)
        for model, spec in model_specs.items():
            probabilities = pd.to_numeric(race[spec.probability], errors="raise")
            scores = pd.to_numeric(race[spec.ranking], errors="raise")
            if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
                raise ValueError(f"race {race_id} has non-finite predictions for {model}")
            if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"race {race_id} probabilities are incoherent for {model}")
            rows.append(
                {
                    **base,
                    "model": model,
                    "ndcg_at_3": ndcg_at_k(scores, positions, ids, k=3),
                    "top_1_winner_mass": top_k_winner_mass(scores, positions, ids, k=1),
                    "race_log_loss": race_log_loss(probabilities, positions, ids),
                    "race_brier": race_brier_score(probabilities, positions, ids),
                }
            )
    result = pd.DataFrame(rows)
    expected = selected["race_id"].nunique() * len(model_specs)
    if len(result) != expected:
        raise AssertionError("per-race metric table is incomplete")
    return result.sort_values(["race_date", "race_id", "model"], kind="stable").reset_index(drop=True)


def paired_block_bootstrap(
    race_metrics: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]] | None = None,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20240830,
    block_length_dates: int = 4,
    scheme: str = "moving_date_block",
) -> dict[str, Any]:
    """Paired bootstrap of race-macro metrics with date blocks as clusters."""

    _validate_race_metric_table(race_metrics)
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if block_length_dates < 1:
        raise ValueError("block_length_dates must be positive")
    if scheme not in {"moving_date_block", "race_iid"}:
        raise ValueError("scheme must be 'moving_date_block' or 'race_iid'")

    models = list(dict.fromkeys(race_metrics["model"].astype(str)))
    comparisons = tuple(comparisons or _default_comparisons(models))
    unknown = sorted({item for pair in comparisons for item in pair}.difference(models))
    if unknown:
        raise ValueError(f"comparison references unknown models: {unknown}")

    dates = np.array(sorted(pd.to_datetime(race_metrics["race_date"]).unique()))
    races = np.array(sorted(race_metrics["race_id"].astype(str).unique()))
    if scheme == "moving_date_block" and block_length_dates > len(dates):
        raise ValueError("block_length_dates cannot exceed the number of dates")

    cube = _metric_cube(race_metrics, models, races)
    date_index = pd.Series(
        pd.to_datetime(
            race_metrics.drop_duplicates("race_id").set_index("race_id").loc[races, "race_date"]
        ).values,
        index=races,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((n_resamples, len(models), len(PRIMARY_METRICS)), dtype=np.float64)

    if scheme == "race_iid":
        for draw in range(n_resamples):
            sampled = rng.integers(0, len(races), size=len(races))
            draws[draw] = cube[sampled].mean(axis=0)
    else:
        per_date_sums = np.zeros((len(dates), len(models), len(PRIMARY_METRICS)))
        per_date_counts = np.zeros(len(dates), dtype=np.int64)
        for date_position, date in enumerate(dates):
            race_positions = np.flatnonzero(date_index.values == date)
            per_date_sums[date_position] = cube[race_positions].sum(axis=0)
            per_date_counts[date_position] = len(race_positions)
        blocks_per_draw = math.ceil(len(dates) / block_length_dates)
        offsets = np.arange(block_length_dates)
        for draw in range(n_resamples):
            starts = rng.integers(0, len(dates), size=blocks_per_draw)
            sampled_dates = ((starts[:, None] + offsets) % len(dates)).reshape(-1)[: len(dates)]
            total_races = per_date_counts[sampled_dates].sum()
            draws[draw] = per_date_sums[sampled_dates].sum(axis=0) / total_races

    alpha = (1.0 - confidence_level) / 2.0
    point = cube.mean(axis=0)
    marginal: dict[str, Any] = {}
    for model_index, model in enumerate(models):
        marginal[model] = {}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            values = draws[:, model_index, metric_index]
            marginal[model][metric] = _interval(values, point[model_index, metric_index], alpha)

    paired: dict[str, Any] = {}
    for candidate, reference in comparisons:
        candidate_index = models.index(candidate)
        reference_index = models.index(reference)
        name = f"{candidate}_vs_{reference}"
        paired[name] = {}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
            values = sign * (draws[:, candidate_index, metric_index] - draws[:, reference_index, metric_index])
            point_difference = sign * (
                point[candidate_index, metric_index] - point[reference_index, metric_index]
            )
            payload = _interval(values, point_difference, alpha)
            payload["fraction_positive"] = float(np.mean(values > 0.0))
            payload["direction"] = "positive_is_candidate_improvement"
            paired[name][metric] = payload

    return {
        "scheme": scheme,
        "block_length_dates": block_length_dates if scheme == "moving_date_block" else None,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "rng": "numpy.random.PCG64",
        "quantile_method": "linear",
        "race_count": len(races),
        "date_count": len(dates),
        "marginal": marginal,
        "paired": paired,
    }


def development_stability_table(race_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return 2024-only point metrics over predeclared temporal/condition slices."""

    _validate_race_metric_table(race_metrics)
    dimensions = (
        "month",
        "quarter",
        "course_type",
        "distance_band",
        "field_size_band",
        "class_group",
        "venue_code",
    )
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for level, group in race_metrics.groupby(dimension, observed=True, dropna=False):
            for model, model_rows in group.groupby("model", sort=False):
                row: dict[str, Any] = {
                    "dimension": dimension,
                    "level": str(level),
                    "model": model,
                    "race_count": int(model_rows["race_id"].nunique()),
                    "date_count": int(model_rows["race_date"].nunique()),
                }
                for metric in PRIMARY_METRICS:
                    row[metric] = float(model_rows[metric].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _race_metadata(row: pd.Series) -> dict[str, Any]:
    date = pd.Timestamp(row["race_date"])
    distance = float(row["distance"])
    field_size = int(row["field_size"])
    return {
        "race_date": date,
        "month": int(date.month),
        "quarter": f"Q{date.quarter}",
        "venue_code": str(row["race_id"])[4:6],
        "course_type": str(row["course_type"]),
        "distance_band": _distance_band(distance),
        "field_size_band": _field_size_band(field_size),
        "class_group": _class_group(row["race_class"]),
        "field_size": field_size,
    }


def _distance_band(distance: float) -> str:
    if distance <= 1400:
        return "sprint"
    if distance <= 1800:
        return "mile"
    if distance <= 2200:
        return "middle"
    return "long"


def _field_size_band(field_size: int) -> str:
    if field_size <= 9:
        return "small"
    if field_size <= 13:
        return "medium"
    if field_size <= 16:
        return "large"
    return "very_large"


def _class_group(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).replace(" ", "")
    if "障害" in text:
        return "jump"
    if "新馬" in text:
        return "maiden_debut"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text or "500万" in text:
        return "class_1"
    if "2勝" in text or "1000万" in text:
        return "class_2"
    if "3勝" in text or "1600万" in text:
        return "class_3"
    if "ＧⅠ" in text or "GⅠ" in text or "GI" in text:
        return "grade_1"
    if "ＧⅡ" in text or "GⅡ" in text or "GII" in text:
        return "grade_2"
    if "ＧⅢ" in text or "GⅢ" in text or "GIII" in text:
        return "grade_3"
    if "（Ｌ）" in text or "(L)" in text:
        return "listed"
    return "open_or_special"


def _validate_race_metric_table(frame: pd.DataFrame) -> None:
    required = {"race_id", "race_date", "model", *PRIMARY_METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"race metric table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("race metric table must not be empty")
    if frame.duplicated(["race_id", "model"]).any():
        raise ValueError("race metric table has duplicate (race_id, model) rows")
    support = frame.groupby("model", sort=False)["race_id"].agg(lambda values: frozenset(values))
    if support.nunique() != 1:
        raise ValueError("models do not share an identical race population")
    if not np.isfinite(frame.loc[:, PRIMARY_METRICS].to_numpy(dtype=float)).all():
        raise ValueError("race metric table contains non-finite metrics")


def _metric_cube(frame: pd.DataFrame, models: Sequence[str], races: np.ndarray) -> np.ndarray:
    cube = np.empty((len(races), len(models), len(PRIMARY_METRICS)), dtype=np.float64)
    indexed = frame.set_index([frame["race_id"].astype(str), frame["model"].astype(str)])
    for model_index, model in enumerate(models):
        cube[:, model_index, :] = indexed.loc[
            [(race, model) for race in races], list(PRIMARY_METRICS)
        ].to_numpy(dtype=float)
    return cube


def _default_comparisons(models: Sequence[str]) -> tuple[tuple[str, str], ...]:
    desired = (
        ("binary", "uniform"),
        ("binary", "history_rate"),
        ("lambdarank", "uniform"),
        ("lambdarank", "history_rate"),
        ("lambdarank", "binary"),
    )
    available = set(models)
    return tuple(pair for pair in desired if set(pair).issubset(available))


def _interval(values: np.ndarray, point: float, alpha: float) -> dict[str, float]:
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return {
        "point": float(point),
        "bootstrap_mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1)),
        "lower": float(lower),
        "upper": float(upper),
    }
