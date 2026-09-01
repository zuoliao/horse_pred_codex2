"""Phase 5C non-selective historical market-oracle diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import load_json
from horse_pred.data import sha256_file
from horse_pred.evaluation import (
    race_balanced_reliability,
    race_brier_score,
    race_log_loss,
)


def _verify_input(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"input fingerprint mismatch for {path}: {actual}")


def _normalize(values: pd.Series, race_ids: pd.Series) -> pd.Series:
    totals = values.groupby(race_ids, sort=False).transform("sum")
    if not np.isfinite(totals).all() or totals.le(0.0).any():
        raise ValueError("race normalization totals must be finite and positive")
    return values / totals


def _method_metrics(frame: pd.DataFrame, probability: str) -> dict[str, Any]:
    values = frame[probability].to_numpy(dtype=float)
    positions = frame["model_finish_position"].astype(int).tolist()
    race_ids = frame["race_id"].tolist()
    return {
        "race_log_loss": float(race_log_loss(values, positions, race_ids)),
        "race_brier": float(race_brier_score(values, positions, race_ids)),
        "calibration": race_balanced_reliability(
            values, positions, race_ids, n_bins=10, strategy="fixed"
        ),
    }


def _race_losses(frame: pd.DataFrame, probability: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_id, group in frame.groupby("race_id", sort=False):
        probabilities = group[probability].to_numpy(dtype=float)
        winners = group["model_finish_position"].astype(int).eq(1).to_numpy()
        target = winners.astype(float) / winners.sum()
        rows.append(
            {
                "race_id": race_id,
                "race_date": group["race_date"].iloc[0],
                "race_log_loss": float(
                    -np.sum(target[winners] * np.log(probabilities[winners]))
                ),
                "race_brier": float(np.sum((probabilities - target) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def _block_interval(
    race_contrast: pd.DataFrame,
    *,
    metric: str,
    resamples: int,
    seed: int,
    block_length: int,
) -> list[float]:
    dates = np.array(sorted(race_contrast["race_date"].unique()))
    if len(dates) < block_length:
        raise ValueError("not enough race dates for the registered block length")
    grouped = {
        date: race_contrast.loc[race_contrast["race_date"].eq(date), metric].to_numpy()
        for date in dates
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(resamples, dtype=float)
    blocks = math.ceil(len(dates) / block_length)
    offsets = np.arange(block_length)
    for index in range(resamples):
        starts = rng.integers(0, len(dates), size=blocks)
        selected = ((starts[:, None] + offsets) % len(dates)).reshape(-1)[: len(dates)]
        draws[index] = float(
            np.concatenate([grouped[dates[position]] for position in selected]).mean()
        )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def run_historical_oracle_diagnostic(
    *,
    repo_root: str | Path,
    preregistration_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the single preregistered final-market descriptive comparison."""

    root = Path(repo_root).resolve()
    prereg_path = Path(preregistration_path)
    if not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    prereg = load_json(prereg_path)
    if prereg.get("selection_or_adoption_use") is not False:
        raise ValueError("historical oracle must be non-selective")
    if prereg.get("profit_or_roi_use") is not False:
        raise ValueError("historical oracle must not evaluate profit or ROI")

    inputs = prereg["inputs"]
    predictions_path = root / inputs["fundamental_predictions"]
    market_path = root / inputs["market_oracle"]
    _verify_input(predictions_path, inputs["fundamental_predictions_sha256"])
    _verify_input(market_path, inputs["market_oracle_sha256"])

    probability_column = inputs["fundamental_probability_column"]
    predictions = pd.read_csv(predictions_path, parse_dates=["race_date"])
    predictions = predictions.loc[
        predictions["split"].eq(prereg["population"]["prediction_split"])
        & predictions["race_date"].dt.year.eq(int(prereg["population"]["year"])),
        [
            "race_id",
            "race_date",
            "horse_id",
            "horse_number",
            "model_finish_position",
            probability_column,
        ],
    ].copy()
    market = pd.read_csv(market_path, parse_dates=["race_date"])
    market = market.loc[
        market["race_date"].dt.year.eq(int(prereg["population"]["year"])),
        ["race_id", "race_date", "horse_id", "final_win_odds"],
    ].copy()
    keys = list(inputs["join_keys"])
    if predictions.duplicated(keys).any() or market.duplicated(keys).any():
        raise ValueError("oracle inputs must be unique by registered join keys")
    merged = predictions.merge(
        market.drop(columns="race_date"), on=keys, how="inner", validate="one_to_one"
    )

    prediction_sets = predictions.groupby("race_id")["horse_id"].agg(frozenset)
    market_sets = market.groupby("race_id")["horse_id"].agg(frozenset)
    common = prediction_sets.index.intersection(market_sets.index)
    eligible = [
        race_id
        for race_id in common
        if prediction_sets.loc[race_id] == market_sets.loc[race_id]
    ]
    merged = merged.loc[merged["race_id"].isin(eligible)].copy()
    merged["final_win_odds"] = pd.to_numeric(
        merged["final_win_odds"], errors="coerce"
    )
    valid_race = merged.groupby("race_id")["final_win_odds"].transform(
        lambda values: bool(np.isfinite(values).all() and values.ge(1.0).all())
    )
    merged = merged.loc[valid_race].copy()
    merged.sort_values(["race_date", "race_id", "horse_number"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    if merged.empty:
        raise ValueError("registered oracle population is empty")
    if not merged.groupby("race_id")["model_finish_position"].apply(
        lambda values: values.astype(int).eq(1).any()
    ).all():
        raise ValueError("every eligible race must have an official winner")

    merged["fundamental_only"] = pd.to_numeric(
        merged[probability_column], errors="raise"
    )
    if not np.isfinite(merged["fundamental_only"]).all() or merged[
        "fundamental_only"
    ].le(0.0).any():
        raise ValueError("fundamental probabilities must be finite and positive")
    merged["fundamental_only"] = _normalize(
        merged["fundamental_only"], merged["race_id"]
    )
    inverse_odds = 1.0 / merged["final_win_odds"]
    merged["final_market_only"] = _normalize(inverse_odds, merged["race_id"])
    log_pool = np.sqrt(
        merged["fundamental_only"] * merged["final_market_only"]
    )
    merged["combined_fixed_50_50_log"] = _normalize(log_pool, merged["race_id"])

    methods = {
        method: _method_metrics(merged, method)
        for method in (
            "fundamental_only",
            "final_market_only",
            "combined_fixed_50_50_log",
        )
    }
    bootstrap = prereg["metrics"]["uncertainty"]
    race_losses = {
        method: _race_losses(merged, method).set_index("race_id")
        for method in methods
    }
    comparisons: dict[str, Any] = {}
    for candidate in ("fundamental_only", "combined_fixed_50_50_log"):
        contrast = race_losses["final_market_only"].copy()
        for metric in ("race_log_loss", "race_brier"):
            contrast[metric] = (
                race_losses["final_market_only"][metric]
                - race_losses[candidate][metric]
            )
        comparisons[f"{candidate}_vs_final_market"] = {
            metric: {
                "improvement": float(contrast[metric].mean()),
                "interval_95": _block_interval(
                    contrast.reset_index(),
                    metric=metric,
                    resamples=int(bootstrap["resamples"]),
                    seed=int(bootstrap["seed"]),
                    block_length=int(bootstrap["block_length_dates"]),
                ),
                "positive_means_candidate_better": True,
            }
            for metric in ("race_log_loss", "race_brier")
        }

    payload = {
        "schema_version": 1,
        "diagnostic_id": prereg["diagnostic_id"],
        "status": "completed_descriptive_oracle_only",
        "preregistration_sha256": sha256_file(prereg_path),
        "scope": {
            "year": int(prereg["population"]["year"]),
            "runner_count": int(len(merged)),
            "race_count": int(merged["race_id"].nunique()),
            "date_count": int(merged["race_date"].nunique()),
            "final_odds_used": True,
            "cutoff_odds_used": False,
            "selection_or_adoption_use": False,
            "profit_or_roi_use": False,
        },
        "inputs": {
            "fundamental_predictions_sha256": sha256_file(predictions_path),
            "market_oracle_sha256": sha256_file(market_path),
            "fundamental_feature_count": int(inputs["fundamental_feature_count"]),
            "fundamental_feature_columns_sha256": inputs[
                "fundamental_feature_columns_sha256"
            ],
            "fundamental_model_sha256": inputs["fundamental_model_sha256"],
        },
        "methods": methods,
        "comparisons": comparisons,
        "interpretation_constraint": prereg["interpretation"]["always"],
        "next_model_executed": False,
    }
    write_json(output_path, payload)
    return payload
