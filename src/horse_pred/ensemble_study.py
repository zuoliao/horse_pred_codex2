"""ENS-01: fixed coherent-probability ensemble on stored rolling predictions."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.modeling import apply_temperature, fit_temperature, race_softmax
from horse_pred.rolling_evaluation import (
    paired_year_stratified_block_bootstrap,
    rolling_race_metric_table,
    summarize_year_macro,
)

_ENSEMBLE_METHOD = "ensemble_fixed_5050"
_SCREEN_YEARS = [2020, 2021, 2022]
_CONFIRMATION_YEAR = 2023
_REQUIRED_PREDICTION_COLUMNS = {
    "fold_id",
    "role",
    "evaluation_year",
    "race_id",
    "race_date",
    "method",
    "model_kind",
    "model_finish_position",
    "utility",
    "probability_t1",
    "probability_calibrated",
}


def validate_ensemble_config(config: dict[str, Any]) -> None:
    """Validate the single fixed ENS-01 candidate and its temporal gate."""

    required = {
        "schema_version",
        "experiment_id",
        "hypothesis",
        "seed",
        "source",
        "methods",
        "blend",
        "screen_evaluation_years",
        "confirmation_evaluation_year",
        "selection",
        "selection_accounting",
        "uncertainty",
    }
    if set(config) != required:
        raise ValueError(
            f"ENS-01 config keys differ from preregistration: {sorted(set(config) ^ required)}"
        )
    if config["schema_version"] != 1 or config["seed"] != 42:
        raise ValueError("ENS-01 requires schema version 1 and seed 42")
    if config["screen_evaluation_years"] != _SCREEN_YEARS:
        raise ValueError("ENS-01 screen years must remain 2020--2022")
    if config["confirmation_evaluation_year"] != _CONFIRMATION_YEAR:
        raise ValueError("ENS-01 confirmation year must remain 2023")

    source = config["source"]
    expected_source = {
        "artifact_directory",
        "predictions_file",
        "predictions_sha256",
        "manifest_file",
        "manifest_sha256",
        "experiment_id",
        "run_commit",
        "run_dirty",
    }
    if set(source) != expected_source:
        raise ValueError("ENS-01 source keys differ from preregistration")
    for name in ("predictions_sha256", "manifest_sha256"):
        value = source[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"ENS-01 source {name} must be a full hexadecimal digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"ENS-01 source {name} must be hexadecimal") from exc
    run_commit = source["run_commit"]
    if not isinstance(run_commit, str) or len(run_commit) != 40:
        raise ValueError("ENS-01 source run_commit must be a full Git SHA-1")
    try:
        int(run_commit, 16)
    except ValueError as exc:
        raise ValueError("ENS-01 source run_commit must be hexadecimal") from exc
    if source["run_dirty"] is not False:
        raise ValueError("ENS-01 requires a clean source run")

    methods = config["methods"]
    if methods != {
        "binary_reference": "binary_control",
        "lambdarank_reference": "lambdarank_candidate",
        "ensemble_candidate": _ENSEMBLE_METHOD,
    }:
        raise ValueError("ENS-01 method IDs differ from the frozen SEC-3F source")
    blend = config["blend"]
    if blend != {
        "binary_weight": 0.5,
        "lambdarank_weight": 0.5,
        "input_probability": "probability_t1",
        "utility": "log_clipped_arithmetic_probability",
        "clip_epsilon": 1e-15,
        "temperature": "independent_per_fold_calibration_role",
    }:
        raise ValueError("ENS-01 blend must remain the fixed 50:50 arithmetic candidate")

    selection = config["selection"]
    if selection != {
        "primary_metric": "race_log_loss",
        "primary_improvement_min": 0.002,
        "minimum_improved_screen_years": 2,
        "guardrail_minimum_improvements": {
            "race_brier": 0.0,
            "ndcg_at_3": -0.002,
            "top_1_winner_mass": -0.005,
        },
        "must_pass_against_both_references": True,
    }:
        raise ValueError("ENS-01 selection rule differs from preregistration")
    accounting = config["selection_accounting"]
    if int(accounting.get("candidate_reference_comparisons", -1)) != 2:
        raise ValueError("ENS-01 must record two candidate-reference comparisons")
    if int(accounting.get("weight_search_candidates", -1)) != 0:
        raise ValueError("ENS-01 must not search blend weights")
    uncertainty = config["uncertainty"]
    if int(uncertainty.get("bootstrap_resamples", 0)) < 2:
        raise ValueError("bootstrap_resamples must be at least two")
    if int(uncertainty.get("block_length_dates", 0)) < 1:
        raise ValueError("block_length_dates must be positive")
    if not 0.0 < float(uncertainty.get("confidence_level", 0.0)) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _manifest_hash(manifest: dict[str, Any], relative_path: str) -> str | None:
    for entry in manifest.get("files", []):
        if entry.get("path") == relative_path:
            return str(entry.get("sha256"))
    return None


def load_and_validate_source(*, root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the frozen SEC-3F artifact before reading its predictions."""

    source = config["source"]
    artifact = _resolve(root, source["artifact_directory"])
    predictions_path = artifact / source["predictions_file"]
    manifest_path = artifact / source["manifest_file"]
    if sha256_file(predictions_path) != source["predictions_sha256"]:
        raise ValueError("ENS-01 source prediction SHA-256 mismatch")
    if sha256_file(manifest_path) != source["manifest_sha256"]:
        raise ValueError("ENS-01 source artifact manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    if _manifest_hash(manifest, source["predictions_file"]) != source["predictions_sha256"]:
        raise ValueError("ENS-01 source manifest does not authenticate the prediction file")

    run_meta = load_json(artifact / "run_meta.json")
    metrics = load_json(artifact / "metrics.json")
    if run_meta.get("experiment_id") != source["experiment_id"]:
        raise ValueError("ENS-01 source experiment ID mismatch")
    if run_meta.get("git") != {"commit": source["run_commit"], "dirty": source["run_dirty"]}:
        raise ValueError("ENS-01 source git state mismatch")
    for payload in (run_meta, metrics.get("scope", {})):
        if payload.get("rows_used_2024") != 0 or payload.get("rows_used_2025") != 0:
            raise ValueError("ENS-01 source used forbidden 2024/2025 outcomes")
        if payload.get("odds_used") is not False:
            raise ValueError("ENS-01 source must be odds-free")

    frame = pd.read_csv(predictions_path, parse_dates=["race_date"])
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ENS-01 source predictions are missing: {missing}")
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("odds", "popularity", "人気", "オッズ"))
    ]
    if forbidden:
        raise ValueError(f"ENS-01 source predictions contain market columns: {forbidden}")
    method_ids = [config["methods"]["binary_reference"], config["methods"]["lambdarank_reference"]]
    selected = frame.loc[frame["method"].isin(method_ids)].copy()
    if set(selected["method"].unique()) != set(method_ids):
        raise ValueError("ENS-01 source does not contain both frozen references")
    if not selected["evaluation_year"].isin([*_SCREEN_YEARS, _CONFIRMATION_YEAR]).all():
        raise ValueError("ENS-01 source contains an unregistered evaluation year")
    if set(selected["role"].unique()) != {"calibration", "evaluation"}:
        raise ValueError("ENS-01 source must contain calibration and evaluation roles")
    expected_kind = {
        config["methods"]["binary_reference"]: "binary",
        config["methods"]["lambdarank_reference"]: "lambdarank",
    }
    for method, kind in expected_kind.items():
        if set(selected.loc[selected["method"].eq(method), "model_kind"].unique()) != {kind}:
            raise ValueError(f"ENS-01 source model kind mismatch for {method}")
    return selected, {
        "artifact_directory": str(artifact),
        "predictions_path": str(predictions_path),
        "predictions_sha256": source["predictions_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "source_run_meta": run_meta,
        "source_scope": metrics["scope"],
    }


def align_reference_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Align Binary and Ranker rows one-to-one and verify outcome metadata."""

    binary_id = config["methods"]["binary_reference"]
    rank_id = config["methods"]["lambdarank_reference"]
    identity = ["fold_id", "role", "evaluation_year", "race_id"]
    if "horse_id" in frame:
        identity.append("horse_id")
    elif "horse_number" in frame:
        identity.append("horse_number")
    else:
        identity.append("model_finish_position")
    if frame.duplicated(["method", *identity]).any():
        raise ValueError("ENS-01 source contains duplicate method-runner keys")

    metadata = ["race_date", "model_finish_position"]
    for optional in ("horse_number", "field_size"):
        if optional in frame and optional not in identity:
            metadata.append(optional)
    values = ["utility", "probability_t1", "probability_calibrated"]
    binary = frame.loc[frame["method"].eq(binary_id), [*identity, *metadata, *values]].copy()
    rank = frame.loc[frame["method"].eq(rank_id), [*identity, *metadata, *values]].copy()
    aligned = binary.merge(
        rank,
        on=identity,
        how="outer",
        suffixes=("_binary", "_lambdarank"),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("ENS-01 Binary and Ranker runner keys do not match")
    aligned = aligned.drop(columns="_merge")
    for column in metadata:
        left = aligned[f"{column}_binary"]
        right = aligned[f"{column}_lambdarank"]
        if column == "race_date":
            equal = pd.to_datetime(left).eq(pd.to_datetime(right))
        else:
            equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(equal.all()):
            raise ValueError(f"ENS-01 source metadata mismatch: {column}")
        aligned[column] = left
        aligned = aligned.drop(columns=[f"{column}_binary", f"{column}_lambdarank"])

    probability_columns = [
        "probability_t1_binary",
        "probability_t1_lambdarank",
        "probability_calibrated_binary",
        "probability_calibrated_lambdarank",
    ]
    if not np.isfinite(aligned[probability_columns].to_numpy(dtype=float)).all():
        raise ValueError("ENS-01 source probabilities must be finite")
    if not ((aligned[probability_columns] >= 0.0) & (aligned[probability_columns] <= 1.0)).all().all():
        raise ValueError("ENS-01 source probabilities must be in [0, 1]")
    groups = ["fold_id", "role", "race_id"]
    for column in probability_columns:
        sums = aligned.groupby(groups, observed=True)[column].sum()
        if not np.allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError(f"ENS-01 source probability is not race-coherent: {column}")
    return aligned.sort_values(
        ["evaluation_year", "fold_id", "role", "race_date", "race_id", identity[-1]],
        kind="stable",
    ).reset_index(drop=True)


def build_ensemble_predictions(
    aligned: pd.DataFrame, *, evaluation_years: list[int], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build q0 and fit one independent ensemble temperature per fold."""

    selected = aligned.loc[aligned["evaluation_year"].isin(evaluation_years)].copy()
    if selected.empty:
        raise ValueError("ENS-01 has no rows for requested evaluation years")
    blend = config["blend"]
    q0 = (
        float(blend["binary_weight"]) * selected["probability_t1_binary"].to_numpy(dtype=float)
        + float(blend["lambdarank_weight"])
        * selected["probability_t1_lambdarank"].to_numpy(dtype=float)
    )
    utility = np.log(np.clip(q0, float(blend["clip_epsilon"]), 1.0))
    selected["raw_output"] = q0
    selected["utility"] = utility
    selected["probability_t1"] = np.nan
    selected["probability_calibrated"] = np.nan
    selected["ensemble_temperature"] = np.nan
    temperatures: dict[str, float] = {}
    for fold_id, index in selected.groupby("fold_id", sort=False, observed=True).groups.items():
        fold = selected.loc[index].sort_values(
            ["role", "race_date", "race_id", "horse_number" if "horse_number" in selected else "model_finish_position"],
            kind="stable",
        )
        calibration = fold.loc[fold["role"].eq("calibration")]
        evaluation = fold.loc[fold["role"].eq("evaluation")]
        if calibration.empty or evaluation.empty:
            raise ValueError(f"ENS-01 fold {fold_id} lacks calibration or evaluation rows")
        calibrator = fit_temperature(
            calibration["utility"], calibration["race_id"], calibration["model_finish_position"]
        )
        t1 = race_softmax(fold["utility"], fold["race_id"])
        calibrated = apply_temperature(calibrator, fold["utility"], fold["race_id"])
        selected.loc[fold.index, "probability_t1"] = t1
        selected.loc[fold.index, "probability_calibrated"] = calibrated
        selected.loc[fold.index, "ensemble_temperature"] = float(calibrator.temperature)
        temperatures[str(fold_id)] = float(calibrator.temperature)
    if not np.allclose(selected["probability_t1"], selected["raw_output"], rtol=0.0, atol=1e-12):
        raise AssertionError("ENS-01 T=1 mapping does not reproduce the coherent arithmetic blend")

    selected["method"] = config["methods"]["ensemble_candidate"]
    selected["model_kind"] = "fixed_probability_ensemble"
    selected["race_date"] = pd.to_datetime(selected["race_date"]).dt.normalize()
    output_columns = [
        "fold_id",
        "role",
        "evaluation_year",
        "race_id",
        "race_date",
        "method",
        "model_kind",
        "model_finish_position",
    ]
    for optional in ("field_size", "horse_number", "horse_id"):
        if optional in selected:
            output_columns.append(optional)
    output_columns.extend(
        [
            "raw_output",
            "utility",
            "probability_t1",
            "probability_calibrated",
            "ensemble_temperature",
        ]
    )
    return selected.loc[:, output_columns].copy(), temperatures


def _reference_predictions(
    source: pd.DataFrame, *, evaluation_years: list[int], config: dict[str, Any]
) -> pd.DataFrame:
    methods = {config["methods"]["binary_reference"], config["methods"]["lambdarank_reference"]}
    selected = source.loc[
        source["evaluation_year"].isin(evaluation_years) & source["method"].isin(methods)
    ].copy()
    selected["race_date"] = pd.to_datetime(selected["race_date"]).dt.normalize()
    return selected


def _comparisons(config: dict[str, Any]) -> list[dict[str, str]]:
    candidate = config["methods"]["ensemble_candidate"]
    return [
        {
            "id": "ensemble_vs_binary",
            "candidate": candidate,
            "reference": config["methods"]["binary_reference"],
            "type": "descriptive_cross_family",
        },
        {
            "id": "ensemble_vs_lambdarank",
            "candidate": candidate,
            "reference": config["methods"]["lambdarank_reference"],
            "type": "descriptive_cross_family",
        },
    ]


def screen_ensemble(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Require the fixed candidate to clear the same screen against both references."""

    selection = config["selection"]
    diagnostics: dict[str, Any] = {}
    all_passed = True
    for comparison_id in ("ensemble_vs_binary", "ensemble_vs_lambdarank"):
        metrics = summary["comparisons"][comparison_id]["metrics"]
        reasons: list[str] = []
        primary = metrics["race_log_loss"]
        if primary["year_macro_improvement"] < float(selection["primary_improvement_min"]):
            reasons.append("primary_point_below_minimum")
        if primary["improved_years"] < int(selection["minimum_improved_screen_years"]):
            reasons.append("primary_direction_below_minimum")
        for metric, minimum in selection["guardrail_minimum_improvements"].items():
            if metrics[metric]["year_macro_improvement"] < float(minimum):
                reasons.append(f"guardrail_failed:{metric}")
        diagnostics[comparison_id] = {"passed": not reasons, "reasons": reasons, "metrics": metrics}
        all_passed &= not reasons
    return {
        "passed": all_passed,
        "decision": "screen_passed" if all_passed else "reject",
        "comparisons": diagnostics,
    }


def confirmation_decision(
    *, summary: dict[str, Any], bootstrap: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Accept only when both singleton comparisons pass the same LL path."""

    selection = config["selection"]
    diagnostics: dict[str, Any] = {}
    all_passed = True
    any_reject = False
    for comparison_id in ("ensemble_vs_binary", "ensemble_vs_lambdarank"):
        metrics = summary["comparisons"][comparison_id]["metrics"]
        point = {
            metric: float(payload["year_macro_improvement"]) for metric, payload in metrics.items()
        }
        interval = bootstrap["paired"][comparison_id]
        guardrail_failed = any(
            point[metric] < float(minimum)
            for metric, minimum in selection["guardrail_minimum_improvements"].items()
        )
        point_passed = point["race_log_loss"] >= float(selection["primary_improvement_min"])
        interval_passed = float(interval["race_log_loss"]["lower"]) > 0.0
        passed = point_passed and interval_passed and not guardrail_failed
        all_passed &= passed
        any_reject |= guardrail_failed or float(interval["race_log_loss"]["upper"]) < 0.0
        diagnostics[comparison_id] = {
            "passed": passed,
            "primary_point_passed": point_passed,
            "primary_interval_passed": interval_passed,
            "guardrail_failed": guardrail_failed,
            "point_improvement": point,
        }
    decision = "accept" if all_passed else ("reject" if any_reject else "inconclusive")
    return {"decision": decision, "comparisons": diagnostics}


def _uncertainty(
    race_metrics: pd.DataFrame, comparisons: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, Any]:
    uncertainty = config["uncertainty"]
    return paired_year_stratified_block_bootstrap(
        race_metrics,
        comparisons=comparisons,
        n_resamples=int(uncertainty["bootstrap_resamples"]),
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        block_length_dates=int(uncertainty["block_length_dates"]),
    )


def run_ensemble_study(
    *, repo_root: str | Path, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run ENS-01 without fitting models or consulting 2024/2025 outcomes."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    config_file = _resolve(root, config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ensemble artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        config = load_json(config_file)
        validate_ensemble_config(config)
        source, source_validation = load_and_validate_source(root=root, config=config)
        aligned = align_reference_predictions(source, config)
        comparisons = _comparisons(config)

        screen_candidate, screen_temperatures = build_ensemble_predictions(
            aligned, evaluation_years=_SCREEN_YEARS, config=config
        )
        screen_predictions = pd.concat(
            [_reference_predictions(source, evaluation_years=_SCREEN_YEARS, config=config), screen_candidate],
            ignore_index=True,
        )
        screen_races = rolling_race_metric_table(screen_predictions)
        screen_summary = summarize_year_macro(screen_races, comparisons)
        screen_bootstrap = _uncertainty(screen_races, comparisons, config)
        screen_result = screen_ensemble(screen_summary, config)

        confirmation_opened = bool(screen_result["passed"])
        confirmation_temperatures: dict[str, float] = {}
        confirmation_summary: dict[str, Any] = {}
        confirmation_bootstrap: dict[str, Any] = {}
        confirmation_result: dict[str, Any] = {
            "decision": "not_opened",
            "reason": "fixed ensemble failed the 2020-2022 screen",
        }
        predictions = screen_predictions
        race_metrics = screen_races.assign(stage="screen")
        if confirmation_opened:
            candidate, confirmation_temperatures = build_ensemble_predictions(
                aligned, evaluation_years=[_CONFIRMATION_YEAR], config=config
            )
            confirmation_predictions = pd.concat(
                [
                    _reference_predictions(
                        source, evaluation_years=[_CONFIRMATION_YEAR], config=config
                    ),
                    candidate,
                ],
                ignore_index=True,
            )
            confirmation_races = rolling_race_metric_table(confirmation_predictions)
            confirmation_summary = summarize_year_macro(confirmation_races, comparisons)
            confirmation_bootstrap = _uncertainty(
                confirmation_races, comparisons, config
            )
            confirmation_result = confirmation_decision(
                summary=confirmation_summary,
                bootstrap=confirmation_bootstrap,
                config=config,
            )
            predictions = pd.concat(
                [screen_predictions, confirmation_predictions], ignore_index=True
            )
            race_metrics = pd.concat(
                [
                    screen_races.assign(stage="screen"),
                    confirmation_races.assign(stage="confirmation"),
                ],
                ignore_index=True,
            )
        final_decision = confirmation_result["decision"] if confirmation_opened else "reject"
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "hypothesis": config["hypothesis"],
            "scope": {
                "screen_evaluation_years": _SCREEN_YEARS,
                "confirmation_evaluation_year": _CONFIRMATION_YEAR,
                "confirmation_opened": confirmation_opened,
                "screen_passed": bool(screen_result["passed"]),
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "model_refit": False,
                "weight_search_candidates": 0,
            },
            "source": source_validation,
            "blend": config["blend"],
            "screen": {
                "temperatures": screen_temperatures,
                "year_summary": screen_summary,
                "paired_block_bootstrap": screen_bootstrap,
                "screen_result": screen_result,
            },
            "confirmation": {
                "temperatures": confirmation_temperatures,
                "year_summary": confirmation_summary,
                "paired_block_bootstrap": confirmation_bootstrap,
                "confirmation_result": confirmation_result,
            },
            "decision": final_decision,
            "selection_accounting": config["selection_accounting"],
            "limitations": [
                "The source singleton models were selected in earlier rolling experiments.",
                "2020-2023 have prior project exposure and are not untouched holdouts.",
                "Block intervals condition on stored model predictions and do not include refit uncertainty.",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(temporary / "metrics.json", metrics)
        write_json(temporary / "config.json", config)
        write_json(temporary / "source_validation.json", source_validation)
        write_json(
            temporary / "calibration.json",
            {"screen": screen_temperatures, "confirmation": confirmation_temperatures},
        )
        write_json(
            temporary / "run_meta.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "seed": config["seed"],
                "config_hash": canonical_json_hash(config),
                "config_sha256": sha256_file(config_file),
                "source_predictions_sha256": config["source"]["predictions_sha256"],
                "source_manifest_sha256": config["source"]["manifest_sha256"],
                "git": git_state(root),
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
                "model_refit": False,
                "weight_search_candidates": 0,
            },
        )
        predictions.to_csv(
            temporary / "predictions_scoring.csv.gz", index=False, compression="gzip"
        )
        race_metrics.to_csv(
            temporary / "race_metrics.csv.gz", index=False, compression="gzip"
        )
        write_artifact_manifest(temporary)
        temporary.replace(output)
        return metrics
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
