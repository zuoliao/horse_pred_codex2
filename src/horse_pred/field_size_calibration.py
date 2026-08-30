"""Field-size-conditioned temperature calibration experiment.

The experiment reuses frozen baseline scores.  It fits calibration parameters
on 2023 only and evaluates them on 2024 only; it never refits or reranks either
prediction model.  A source file may physically contain retrospective rows,
but they are removed immediately after loading and before any fitting or
metric computation.
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
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.evaluation import evaluate_predictions
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    probability_logits,
    validate_grouped_rows,
)
from horse_pred.pipeline import PROBABILITY_EPSILON
from horse_pred.uncertainty import (
    ModelSpec,
    development_race_metric_table,
    paired_block_bootstrap,
)

_MODEL_SPECS: Mapping[str, Mapping[str, str]] = {
    "binary": {
        "raw_source": "pred_binary_raw",
        "baseline_probability": "prob_binary_logit_softmax_temperature_2023",
        "calibration_score": "score_binary_logit",
        "ranking_score": "pred_binary_raw",
        "candidate_probability": "prob_binary_field_size_temperature_2023",
    },
    "lambdarank": {
        "raw_source": "score_lambdarank",
        "baseline_probability": "prob_lambdarank_softmax_temperature_2023",
        "calibration_score": "score_lambdarank",
        "ranking_score": "score_lambdarank",
        "candidate_probability": "prob_lambdarank_field_size_temperature_2023",
    },
}
_GLOBAL_TEMPERATURE_TOLERANCE = 1e-8


def validate_field_size_calibration_config(config: Mapping[str, Any]) -> None:
    """Validate the intentionally narrow one-hypothesis experiment config."""

    required = {
        "schema_version",
        "experiment_id",
        "parent_experiment_id",
        "hypothesis",
        "seed",
        "field_size_bands",
        "minimum_calibration_races_per_band",
        "bootstrap",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"field-size calibration config is missing: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("field-size calibration schema_version must be 1")
    for key in ("experiment_id", "parent_experiment_id", "hypothesis"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    bands = config["field_size_bands"]
    if not isinstance(bands, Mapping):
        raise ValueError("field_size_bands must be an object")
    labels = bands.get("labels")
    upper_bounds = bands.get("upper_bounds")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("field_size_bands.labels must contain unique non-empty strings")
    if (
        not isinstance(upper_bounds, list)
        or any(not isinstance(value, int) or value < 1 for value in upper_bounds)
        or upper_bounds != sorted(set(upper_bounds))
        or len(labels) != len(upper_bounds) + 1
    ):
        raise ValueError(
            "field_size_bands.upper_bounds must be increasing positive integers "
            "with one fewer item than labels"
        )

    minimum = config["minimum_calibration_races_per_band"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_calibration_races_per_band must be a positive integer")
    bootstrap = config["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("bootstrap must be an object")
    if bootstrap.get("scheme") != "moving_date_block":
        raise ValueError("bootstrap.scheme must be moving_date_block")
    if not isinstance(bootstrap.get("resamples"), int) or bootstrap["resamples"] < 2:
        raise ValueError("bootstrap.resamples must be at least 2")
    if not isinstance(bootstrap.get("block_length_dates"), int) or bootstrap[
        "block_length_dates"
    ] < 1:
        raise ValueError("bootstrap.block_length_dates must be positive")


def field_size_bands(field_sizes: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    """Return the preregistered field-size band for every runner."""

    validate_field_size_calibration_config(config)
    numeric = pd.to_numeric(field_sizes, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("field_size must be finite")
    if not np.equal(numeric, np.floor(numeric)).all() or numeric.lt(1).any():
        raise ValueError("field_size must contain positive integers")
    settings = config["field_size_bands"]
    bins = [0, *settings["upper_bounds"], np.inf]
    result = pd.cut(
        numeric,
        bins=bins,
        labels=settings["labels"],
        include_lowest=True,
        ordered=True,
    )
    if result.isna().any():
        raise AssertionError("field-size band assignment is incomplete")
    return result.astype("string")


def isolate_calibration_and_development(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Immediately discard 2025+, then require exact 2023/2024 split semantics."""

    required = {
        "race_id",
        "race_date",
        "split",
        "field_size",
        "model_finish_position",
    }
    for spec in _MODEL_SPECS.values():
        required.update((spec["raw_source"], spec["baseline_probability"]))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"baseline predictions are missing: {missing}")

    source_rows = len(frame)
    dates = pd.to_datetime(frame["race_date"], errors="raise").dt.normalize()
    retrospective = frame["split"].eq("retrospective_test") | dates.dt.year.ge(2025)
    mislabeled_2025 = dates.dt.year.ge(2025) & ~frame["split"].eq("retrospective_test")
    if mislabeled_2025.any():
        raise ValueError("2025+ rows must be labeled retrospective_test")
    wrongly_dated_retrospective = frame["split"].eq("retrospective_test") & dates.dt.year.lt(2025)
    if wrongly_dated_retrospective.any():
        raise ValueError("retrospective_test contains a pre-2025 row")

    # The retrospective firewall deliberately occurs before numeric conversion,
    # score transformation, band derivation, fitting, or evaluation.
    selected = frame.loc[~retrospective].copy()
    selected["race_date"] = dates.loc[~retrospective]
    del frame
    selected = selected.loc[selected["split"].isin(("calibration", "development"))].copy()
    expected_year = selected["split"].map({"calibration": 2023, "development": 2024})
    if selected.empty or not selected["race_date"].dt.year.eq(expected_year).all():
        raise ValueError("calibration/development rows must be from 2023/2024 respectively")
    if set(selected["split"].unique()) != {"calibration", "development"}:
        raise ValueError("source must contain both calibration and development rows")

    selected["race_id"] = selected["race_id"].astype("string")
    selected = selected.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    validate_grouped_rows(selected["race_id"])
    for column in ("race_date", "split", "field_size"):
        if selected.groupby("race_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"{column} must be constant within a race")
    declared_field_size = pd.to_numeric(selected["field_size"], errors="raise")
    actual_field_size = selected.groupby("race_id", sort=False)["race_id"].transform("size")
    if not declared_field_size.eq(actual_field_size).all():
        raise ValueError("field_size must equal the number of scored runners in every race")
    return selected, {
        "source_rows": int(source_rows),
        "retrospective_2025_plus_rows_in_source": int(retrospective.sum()),
        "retrospective_2025_plus_rows_used": 0,
        "calibration_2023_rows": int(selected["split"].eq("calibration").sum()),
        "development_2024_rows": int(selected["split"].eq("development").sum()),
    }


def fit_band_temperatures(
    calibration: pd.DataFrame,
    *,
    score_column: str,
    band_column: str,
    labels: Sequence[str],
    minimum_races: int,
) -> tuple[dict[str, Any], Any]:
    """Fit one temperature per adequately sized band, with global fallback."""

    global_calibrator = fit_temperature(
        calibration[score_column],
        calibration["race_id"],
        calibration["model_finish_position"],
    )
    fitted: dict[str, Any] = {}
    for label in labels:
        subset = calibration.loc[calibration[band_column].eq(label)]
        race_count = int(subset["race_id"].nunique())
        if race_count >= minimum_races:
            calibrator = fit_temperature(
                subset[score_column],
                subset["race_id"],
                subset["model_finish_position"],
            )
            fallback = False
        else:
            calibrator = global_calibrator
            fallback = True
        fitted[label] = {
            "calibrator": calibrator,
            "temperature": float(calibrator.temperature),
            "calibration_race_count": race_count,
            "used_global_fallback": fallback,
        }
    return fitted, global_calibrator


def apply_band_temperatures(
    frame: pd.DataFrame,
    *,
    score_column: str,
    band_column: str,
    fitted: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Apply one race-constant band calibrator and preserve row order."""

    result = np.full(len(frame), np.nan, dtype=np.float64)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for label, payload in fitted.items():
        mask = frame[band_column].eq(label)
        if not mask.any():
            continue
        subset = frame.loc[mask]
        transformed = apply_temperature(
            payload["calibrator"], subset[score_column], subset["race_id"]
        )
        result[positions.loc[mask].to_numpy()] = transformed
    if not np.isfinite(result).all():
        raise ValueError("a development field-size band has no fitted calibrator")
    return result


def _model_metrics(frame: pd.DataFrame, probability: str, score: str) -> dict[str, Any]:
    return evaluate_predictions(
        frame[probability],
        frame["model_finish_position"],
        frame["race_id"],
        ranking_scores=frame[score],
        conditions={"field_size_band": frame["field_size_band"]},
    )


def paired_bootstrap_by_field_size_band(
    race_metrics: pd.DataFrame,
    *,
    labels: Sequence[str],
    n_resamples: int,
    seed: int,
    block_length_dates: int,
) -> dict[str, Any]:
    """Run the same paired date-block comparison within each field-size band."""

    comparisons = (
        ("candidate_binary", "baseline_binary"),
        ("candidate_lambdarank", "baseline_lambdarank"),
    )
    bands: dict[str, Any] = {}
    for offset, label in enumerate(labels, start=1):
        subset = race_metrics.loc[race_metrics["field_size_band"].eq(label)].copy()
        race_count = int(subset["race_id"].nunique())
        date_count = int(subset["race_date"].nunique())
        support = {"race_count": race_count, "date_count": date_count}
        if race_count == 0:
            bands[label] = {
                "status": "insufficient_support",
                **support,
                "reason": "no development races in this field-size band",
            }
            continue
        if date_count < block_length_dates:
            bands[label] = {
                "status": "insufficient_support",
                **support,
                "reason": (
                    f"requires at least {block_length_dates} distinct race dates for "
                    "the preregistered moving-date block"
                ),
            }
            continue
        bands[label] = {
            "status": "available",
            **support,
            "bootstrap": paired_block_bootstrap(
                subset,
                comparisons=comparisons,
                n_resamples=n_resamples,
                seed=seed + offset,
                block_length_dates=block_length_dates,
            ),
        }
    return {
        "scheme": "moving_date_block",
        "block_length_dates": block_length_dates,
        "n_resamples": n_resamples,
        "seed_policy": "experiment seed plus one-based field-size-band position",
        "bands": bands,
    }


def run_field_size_calibration_experiment(
    *,
    repo_root: str | Path,
    baseline_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit the registered 2023 calibrators and write a 2024-only artifact."""

    root = Path(repo_root).resolve()
    source_dir = Path(baseline_dir).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    final_output = Path(output_dir).resolve()
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite experiment artifact: {final_output}")
    temporary = final_output.with_name(f".{final_output.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True)

    try:
        config = load_json(config_file)
        validate_field_size_calibration_config(config)
        predictions_path = source_dir / "predictions.csv.gz"
        baseline_metrics_path = source_dir / "metrics.json"
        if not predictions_path.is_file() or not baseline_metrics_path.is_file():
            raise FileNotFoundError("baseline artifact requires predictions.csv.gz and metrics.json")

        loaded = pd.read_csv(predictions_path, dtype={"race_id": "string"})
        selected, isolation = isolate_calibration_and_development(loaded)
        del loaded
        selected["field_size_band"] = field_size_bands(selected["field_size"], config)
        selected["score_binary_logit"] = probability_logits(
            selected["pred_binary_raw"], epsilon=PROBABILITY_EPSILON
        )
        calibration = selected.loc[selected["split"].eq("calibration")].copy()
        development = selected.loc[selected["split"].eq("development")].copy()

        labels = list(config["field_size_bands"]["labels"])
        minimum = int(config["minimum_calibration_races_per_band"])
        calibrator_payload: dict[str, Any] = {}
        for model, spec in _MODEL_SPECS.items():
            fitted, global_calibrator = fit_band_temperatures(
                calibration,
                score_column=spec["calibration_score"],
                band_column="field_size_band",
                labels=labels,
                minimum_races=minimum,
            )
            development[spec["candidate_probability"]] = apply_band_temperatures(
                development,
                score_column=spec["calibration_score"],
                band_column="field_size_band",
                fitted=fitted,
            )
            calibrator_payload[model] = {
                "global_temperature_refit_2023": float(global_calibrator.temperature),
                "bands": {
                    label: {
                        key: value
                        for key, value in fitted[label].items()
                        if key != "calibrator"
                    }
                    for label in labels
                },
            }

        evaluations: dict[str, Any] = {}
        for model, spec in _MODEL_SPECS.items():
            baseline = _model_metrics(
                development, spec["baseline_probability"], spec["ranking_score"]
            )
            candidate = _model_metrics(
                development, spec["candidate_probability"], spec["ranking_score"]
            )
            for metric in baseline["ranking"]:
                if candidate["ranking"][metric] != baseline["ranking"][metric]:
                    raise AssertionError("temperature calibration changed ranking metrics")
            evaluations[model] = {"baseline_global": baseline, "candidate_field_size": candidate}

        uncertainty_specs = {
            "baseline_binary": ModelSpec(
                _MODEL_SPECS["binary"]["baseline_probability"],
                _MODEL_SPECS["binary"]["ranking_score"],
            ),
            "candidate_binary": ModelSpec(
                _MODEL_SPECS["binary"]["candidate_probability"],
                _MODEL_SPECS["binary"]["ranking_score"],
            ),
            "baseline_lambdarank": ModelSpec(
                _MODEL_SPECS["lambdarank"]["baseline_probability"],
                _MODEL_SPECS["lambdarank"]["ranking_score"],
            ),
            "candidate_lambdarank": ModelSpec(
                _MODEL_SPECS["lambdarank"]["candidate_probability"],
                _MODEL_SPECS["lambdarank"]["ranking_score"],
            ),
        }
        race_metrics = development_race_metric_table(
            development, model_specs=uncertainty_specs
        )
        bootstrap_config = config["bootstrap"]
        bootstrap = paired_block_bootstrap(
            race_metrics,
            comparisons=(
                ("candidate_binary", "baseline_binary"),
                ("candidate_lambdarank", "baseline_lambdarank"),
            ),
            n_resamples=int(bootstrap_config["resamples"]),
            seed=int(config["seed"]),
            block_length_dates=int(bootstrap_config["block_length_dates"]),
        )

        paired_by_band = paired_bootstrap_by_field_size_band(
            race_metrics,
            labels=labels,
            n_resamples=int(bootstrap_config["resamples"]),
            seed=int(config["seed"]),
            block_length_dates=int(bootstrap_config["block_length_dates"]),
        )

        with baseline_metrics_path.open(encoding="utf-8") as handle:
            baseline_metrics = json.load(handle)
        try:
            saved_temperatures = {
                model: float(baseline_metrics["models"][model]["temperature"])
                for model in _MODEL_SPECS
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "baseline metrics must contain models.<model>.temperature"
            ) from exc
        for model, saved_temperature in saved_temperatures.items():
            refit_temperature = calibrator_payload[model]["global_temperature_refit_2023"]
            absolute_difference = abs(refit_temperature - saved_temperature)
            if absolute_difference > _GLOBAL_TEMPERATURE_TOLERANCE:
                raise AssertionError(
                    f"{model} 2023 global temperature does not reproduce the baseline: "
                    f"absolute difference {absolute_difference:.12g}"
                )
            calibrator_payload[model]["baseline_global_temperature_reproduction"] = {
                "saved_temperature": saved_temperature,
                "refit_temperature": refit_temperature,
                "absolute_difference": absolute_difference,
                "absolute_tolerance": _GLOBAL_TEMPERATURE_TOLERANCE,
                "within_tolerance": True,
            }
        data_fingerprint = baseline_metrics.get("data", {}).get("fingerprint")
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "parent_experiment_id": config["parent_experiment_id"],
            "hypothesis": config["hypothesis"],
            "scope": {
                "model_scores": "frozen corrected baseline",
                "calibrator_fit": "2023 calibration only",
                "evaluation": "2024 development only",
                "ranking_scores_changed": False,
                "odds_used": False,
                "retrospective_used": False,
                **isolation,
            },
            "data": {
                "fingerprint": data_fingerprint,
                "source_predictions_sha256": sha256_file(predictions_path),
                "calibration_races": int(calibration["race_id"].nunique()),
                "development_races": int(development["race_id"].nunique()),
            },
            "config_hash": canonical_json_hash(config),
            "calibrators": calibrator_payload,
            "development": evaluations,
            "paired_block_bootstrap": bootstrap,
            "paired_by_field_size_band": paired_by_band,
            "limitations": [
                "Field-size bands and the minimum cell size were preregistered; no band boundary was tuned on 2024.",
                "ECE is descriptive and is not used as the sole acceptance criterion.",
                "This experiment changes probability calibration only, so ranking metrics "
                "must remain exactly unchanged.",
            ],
        }
        output_columns = [
            "race_id",
            "race_date",
            "horse_id",
            "split",
            "field_size",
            "field_size_band",
            "model_finish_position",
            "pred_binary_raw",
            "score_binary_logit",
            "score_lambdarank",
            "prob_binary_logit_softmax_temperature_2023",
            "prob_binary_field_size_temperature_2023",
            "prob_lambdarank_softmax_temperature_2023",
            "prob_lambdarank_field_size_temperature_2023",
        ]
        available = [column for column in output_columns if column in development]
        development.loc[:, available].to_csv(
            temporary / "predictions_2024.csv.gz", index=False, compression="gzip"
        )
        race_metrics.to_csv(
            temporary / "race_metrics_2024.csv.gz", index=False, compression="gzip"
        )
        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "config.json", config)
        write_json(
            temporary / "run_meta.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "git": git_state(root),
                "config_hash": canonical_json_hash(config),
                "source_predictions_sha256": sha256_file(predictions_path),
            },
        )
        write_artifact_manifest(temporary)
        temporary.rename(final_output)
        return metrics
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
