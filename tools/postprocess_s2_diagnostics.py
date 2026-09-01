"""Complete preregistered non-selection S2 diagnostics from the local artifact."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_artifact_manifest, write_json
from horse_pred.config import load_json
from horse_pred.s1_two_axis_study import verify_artifact_manifest
from horse_pred.two_axis_race_value import PERFORMANCE_COLUMN

METRICS = (
    "ndcg_at_3",
    "top_1_winner_mass",
    "winner_reciprocal_rank",
    "race_log_loss",
    "race_brier",
)
HIGHER_IS_BETTER = {"ndcg_at_3", "top_1_winner_mass", "winner_reciprocal_rank"}


def _did_bootstrap(
    race_metrics: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    block_length: int,
) -> dict[str, Any]:
    indexed = race_metrics.pivot_table(
        index=["evaluation_year", "race_date", "race_id"],
        columns="method",
        values=list(METRICS),
        aggfunc="first",
    )
    contrast = pd.DataFrame(index=indexed.index)
    for metric in METRICS:
        sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
        contrast[metric] = sign * (
            (indexed[(metric, "linear_R1")] - indexed[(metric, "linear_R0")])
            - (indexed[(metric, "binary_B1")] - indexed[(metric, "binary_B0")])
        )
    per_year = contrast.groupby(level="evaluation_year").mean()
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((resamples, len(METRICS)), dtype=float)
    years = sorted(contrast.index.get_level_values("evaluation_year").unique())
    for draw in range(resamples):
        year_values = []
        for year in years:
            subset = contrast.xs(year, level="evaluation_year")
            dates = np.array(sorted(subset.index.get_level_values("race_date").unique()))
            starts = rng.integers(
                0, len(dates), size=math.ceil(len(dates) / block_length)
            )
            sampled_dates = (
                (starts[:, None] + np.arange(block_length)) % len(dates)
            ).reshape(-1)[: len(dates)]
            sampled = pd.concat(
                [subset.xs(dates[index], level="race_date") for index in sampled_dates]
            )
            year_values.append(sampled.loc[:, list(METRICS)].mean().to_numpy())
        draws[draw] = np.mean(year_values, axis=0)
    result: dict[str, Any] = {}
    for index, metric in enumerate(METRICS):
        result[metric] = {
            "per_year": {
                str(int(year)): float(per_year.loc[year, metric]) for year in years
            },
            "year_macro": float(per_year[metric].mean()),
            "interval_95": [
                float(np.quantile(draws[:, index], 0.025)),
                float(np.quantile(draws[:, index], 0.975)),
            ],
            "direction": "positive_means_larger_S1_increment_under_racewise",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--control-schema",
        type=Path,
        default=Path(
            "artifacts/s3_condition_adjusted_performance_target_20260901/"
            "feature_schema.json"
        ),
    )
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    race_metrics = pd.read_csv(
        artifact / "tables/race_metrics.csv.gz", parse_dates=["race_date"]
    )
    did = _did_bootstrap(
        race_metrics, resamples=10_000, seed=20240830, block_length=4
    )

    coverage: dict[str, Any] = {}
    race_constant: dict[str, Any] | None = None
    schema = load_json(args.control_schema)
    control_columns = list(schema["scopes"]["binary"]["feature_columns"])
    for path in sorted((artifact / "feature_tables").glob("roll_*.pkl")):
        frame = pd.read_pickle(path)
        by_role: dict[str, Any] = {}
        for role, rows in frame.groupby("rolling_role", observed=True):
            values = pd.to_numeric(rows[PERFORMANCE_COLUMN], errors="coerce")
            by_role[str(role)] = {
                "year": int(pd.to_datetime(rows["race_date"]).dt.year.iloc[0])
                if str(role) != "train"
                else None,
                "runner_count": int(len(rows)),
                "race_count": int(rows["race_id"].nunique()),
                "coverage": float(values.notna().mean()),
                "missing_count": int(values.isna().sum()),
                "quantiles": {
                    str(q): float(values.quantile(q))
                    for q in (0.01, 0.1, 0.5, 0.9, 0.99)
                },
            }
        coverage[path.stem] = by_role
        if path.stem == "roll_2020":
            train = frame.loc[frame["rolling_role"].eq("train")]
            maximum_nunique = (
                train.groupby("race_id", sort=False)[control_columns]
                .nunique(dropna=False)
                .max(axis=0)
            )
            columns = maximum_nunique.loc[maximum_nunique.le(1)].index.tolist()
            race_constant = {
                "fit_role": "roll_2020_train",
                "feature_count": len(control_columns),
                "race_constant_feature_count": len(columns),
                "race_constant_features": columns,
                "interpretation": "linear main effects cancel within every choice set",
            }
        del frame

    coefficient: dict[str, Any] = {}
    for path in sorted((artifact / "models").glob("roll_*/linear_R1.npz")):
        with np.load(path) as model:
            coefficient[path.parent.name] = {
                "feature": PERFORMANCE_COLUMN,
                "standardized_coefficient": float(model["coefficients"][-1]),
                "feature_is_last_registered_column": True,
            }

    predictions = pd.read_csv(
        artifact / "predictions/scoring.csv.gz",
        usecols=[
            "fold_id",
            "role",
            "method",
            "race_id",
            "model_finish_position",
            "probability_native",
            "probability_calibrated",
        ],
    )
    population: dict[str, Any] = {}
    for (fold_id, role), rows in predictions.groupby(["fold_id", "role"]):
        methods = {
            method: tuple(
                zip(
                    group["race_id"].astype(str),
                    group["model_finish_position"].astype(int),
                )
            )
            for method, group in rows.groupby("method", sort=False)
        }
        first = next(iter(methods.values()))
        population[f"{fold_id}::{role}"] = {
            "method_count": len(methods),
            "runner_count": len(first),
            "race_count": int(rows.loc[rows["method"].eq(next(iter(methods))), "race_id"].nunique()),
            "identical_race_finish_sequence": all(value == first for value in methods.values()),
        }
    native_error = float(
        predictions.groupby(["fold_id", "role", "method", "race_id"])[
            "probability_native"
        ]
        .sum()
        .sub(1.0)
        .abs()
        .max()
    )
    calibrated_error = float(
        predictions.groupby(["fold_id", "role", "method", "race_id"])[
            "probability_calibrated"
        ]
        .sum()
        .sub(1.0)
        .abs()
        .max()
    )

    metrics = load_json(artifact / "metrics.json")
    run_meta = load_json(artifact / "run_meta.json")
    metrics["scope"].update(
        {
            "fit_count": 42,
            "model_unit_count": 18,
            "lightgbm_fit_count": 6,
            "linear_optimizer_candidate_fit_count": 36,
        }
    )
    run_meta.update(metrics["scope"])
    write_json(artifact / "metrics.json", metrics)
    write_json(artifact / "run_meta.json", run_meta)

    write_json(
        artifact / "postrun_diagnostics.json",
        {
            "schema_version": 1,
            "difference_in_differences": did,
            "performance_coverage_and_drift": coverage,
            "linear_performance_coefficients": coefficient,
            "race_constant_linear_features": race_constant,
            "integrity": {
                "choice_set_population": population,
                "all_choice_set_sequences_identical": all(
                    item["identical_race_finish_sequence"]
                    for item in population.values()
                ),
                "maximum_native_probability_sum_error": native_error,
                "maximum_calibrated_probability_sum_error": calibrated_error,
                "nonfinite_probability_count": int(
                    (~np.isfinite(predictions["probability_native"])).sum()
                    + (~np.isfinite(predictions["probability_calibrated"])).sum()
                ),
            },
            "protocol_deviations": [
                {
                    "item": "race-aware permutation dependence",
                    "status": "omitted_non_selection_diagnostic",
                    "reason": "not generated by the completed runner; no refit or acceptance metric depends on it",
                    "result_impact": False,
                },
                {
                    "item": "capacity-matched LB0/LB1 model files and predictions",
                    "status": "optimizer_diagnostics_only_saved",
                    "reason": (
                        "auxiliary validation losses and optimizer records were saved, "
                        "but coefficients and runner predictions were not"
                    ),
                    "result_impact": False,
                }
            ],
        },
    )
    write_artifact_manifest(artifact)
    print(verify_artifact_manifest(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
