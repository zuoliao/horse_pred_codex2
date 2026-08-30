"""Fast, reproducible experiments from the ignored point-in-time frame cache.

The cache may physically contain the preregistered 2025 retrospective split,
but this runner drops it before feature selection, fitting, calibration, or
prediction.  Cached experiments therefore have exactly one selection surface:
the 2024 development split.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.config import canonical_json_hash, load_json
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.evaluation import evaluate_prediction_frame
from horse_pred.features import (
    semantic_feature_groups_v2,
    source_family_knockout_columns,
)
from horse_pred.modeling import (
    apply_temperature,
    fit_temperature,
    history_rate_probabilities,
    predict,
    probability_logits,
    race_softmax,
    train_binary,
    train_ranker,
    uniform_probabilities,
    validate_standard_split_partition,
)
from horse_pred.pipeline import PROBABILITY_EPSILON

_RUN_SPLITS = ("train", "model_validation", "calibration", "development")
_MODEL_NAMES = ("binary", "lambdarank")
_DERIVED_OPERATION = "within_race_percentile"


def feature_columns_checksum(columns: tuple[str, ...] | list[str]) -> str:
    """Hash an ordered feature list without depending on JSON whitespace."""

    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_cached_experiment_config(config: dict[str, Any]) -> None:
    """Validate the deliberately small one-hypothesis config schema."""

    required = {
        "schema_version",
        "experiment_id",
        "hypothesis",
        "seed",
        "model_configs",
        "feature_selection",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"cached experiment config is missing: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("cached experiment config schema_version must be 1")
    if not str(config["experiment_id"]).strip() or not str(config["hypothesis"]).strip():
        raise ValueError("experiment_id and hypothesis must be non-empty")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    model_configs = config["model_configs"]
    if not isinstance(model_configs, dict) or set(model_configs) != set(_MODEL_NAMES):
        raise ValueError("model_configs must contain exactly binary and lambdarank")
    if any(not isinstance(value, str) or not value for value in model_configs.values()):
        raise ValueError("model config paths must be non-empty strings")

    selection = config["feature_selection"]
    if not isinstance(selection, dict):
        raise ValueError("feature_selection must be an object")
    operations = set(selection).intersection({"include", "drop"})
    if len(operations) != 1:
        raise ValueError("feature_selection must declare exactly one of include or drop")
    operation = next(iter(operations))
    groups = selection[operation]
    if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
        raise ValueError(f"feature_selection.{operation} must be a list of strings")
    if operation == "include" and not groups:
        raise ValueError("feature_selection.include must not be empty")
    if len(groups) != len(set(groups)):
        raise ValueError(f"feature_selection.{operation} contains duplicates")

    derived = config.get("derived_features", [])
    if not isinstance(derived, list):
        raise ValueError("derived_features must be a list")
    outputs: set[str] = set()
    for item in derived:
        if not isinstance(item, dict) or set(item) != {"operation", "source", "output"}:
            raise ValueError(
                "each derived feature must contain exactly operation, source, and output"
            )
        if item["operation"] != _DERIVED_OPERATION:
            raise ValueError(f"unsupported derived feature operation: {item['operation']}")
        if not isinstance(item["source"], str) or not item["source"]:
            raise ValueError("derived feature source must be a non-empty string")
        output = item["output"]
        if not isinstance(output, str) or not output.startswith("experimental__"):
            raise ValueError("derived feature output must start with experimental__")
        if output in outputs:
            raise ValueError(f"duplicate derived feature output: {output}")
        outputs.add(output)


def resolve_semantic_feature_selection(
    feature_columns: tuple[str, ...], config: dict[str, Any]
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Resolve v2 semantic include/drop selection with source dependencies."""

    validate_cached_experiment_config(config)
    taxonomy = semantic_feature_groups_v2(feature_columns)
    selection = config["feature_selection"]
    operation = "include" if "include" in selection else "drop"
    named_groups = tuple(selection[operation])
    unknown = sorted(set(named_groups).difference(taxonomy))
    if unknown:
        raise ValueError(f"unknown semantic feature groups: {unknown}")

    included_groups = (
        set(named_groups)
        if operation == "include"
        else set(taxonomy).difference(named_groups)
    )
    selected = {
        column
        for group in included_groups
        for column in taxonomy[group]
    }

    # A field-relative feature derived from an explicitly dropped source would
    # leak that source back into a knockout.  An include experiment, however,
    # means exactly the named semantic families (including field_relative).
    dependency_removed: set[str] = set()
    for source in ("horse_performance", "form_workload", "connections"):
        if operation == "drop" and source in named_groups:
            descendants = set(source_family_knockout_columns(taxonomy, source)).difference(
                taxonomy[source]
            )
            dependency_removed.update(selected.intersection(descendants))
            selected.difference_update(descendants)

    resolved_columns = tuple(column for column in feature_columns if column in selected)
    if not resolved_columns:
        raise ValueError("feature selection resolved to zero columns")
    resolved_groups = {
        name: tuple(column for column in columns if column in selected)
        for name, columns in taxonomy.items()
        if any(column in selected for column in columns)
    }
    resolution = {
        "taxonomy": "semantic_feature_groups_v2",
        "operation": operation,
        "configured_groups": list(named_groups),
        "included_groups": [name for name in taxonomy if name in included_groups],
        "dependency_removed_columns": sorted(dependency_removed),
    }
    return resolved_columns, resolved_groups, resolution


def isolate_pre_2025_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove 2025 before any experiment operation and verify split semantics."""

    required = {"race_id", "race_date", "split", "model_finish_position"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"model-frame cache is missing experiment metadata: {missing}")
    dates = pd.to_datetime(frame["race_date"], errors="raise")
    retrospective = frame["split"].eq("retrospective_test")
    mislabeled_2025 = dates.dt.year.eq(2025) & ~retrospective
    if mislabeled_2025.any():
        raise ValueError("2025 rows must be labeled retrospective_test")
    wrongly_dated_retrospective = retrospective & ~dates.dt.year.eq(2025)
    if wrongly_dated_retrospective.any():
        raise ValueError("retrospective_test contains non-2025 rows")

    isolated = frame.loc[~retrospective].copy()
    isolated["race_date"] = dates.loc[~retrospective]
    if isolated["race_date"].dt.year.ge(2025).any():
        raise AssertionError("2025+ rows survived retrospective isolation")
    unexpected = sorted(set(isolated["split"].dropna().astype(str)).difference(_RUN_SPLITS))
    if unexpected:
        raise ValueError(f"cache contains unsupported pre-2025 splits: {unexpected}")
    split_counts = validate_standard_split_partition(
        isolated["race_id"],
        isolated["split"],
        isolated["race_date"],
        require_all_splits=False,
    )
    missing_run_splits = [split for split in _RUN_SPLITS if split_counts[split] == 0]
    if missing_run_splits:
        raise ValueError(f"pre-2025 experiment frame is missing {missing_run_splits}")
    return isolated, {
        "cache_rows": int(len(frame)),
        "retrospective_2025_rows_in_cache": int(retrospective.sum()),
        "retrospective_2025_rows_used": 0,
        "experiment_rows": int(len(isolated)),
    }


def add_registered_derived_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    feature_groups: dict[str, tuple[str, ...]],
    resolution: dict[str, Any],
    config: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Add narrowly registered, race-local transforms after source selection.

    The operation uses only values already available in the same PIT feature
    row.  It cannot access labels, dates, market columns, or raw identifiers as
    sources because the source must already belong to the model allowlist.
    """

    derived = config.get("derived_features", [])
    if not derived:
        return feature_columns, feature_groups, resolution

    selected = set(feature_columns)
    outputs: list[str] = []
    records: list[dict[str, str]] = []
    for item in derived:
        source = item["source"]
        output = item["output"]
        if source not in selected:
            raise ValueError(
                f"derived feature source is not in the selected allowlist: {source}"
            )
        if output in frame.columns:
            raise ValueError(f"derived feature output already exists: {output}")
        frame[output] = frame.groupby("race_id", sort=False, observed=True)[
            source
        ].rank(pct=True, method="average")
        outputs.append(output)
        records.append(
            {
                "operation": _DERIVED_OPERATION,
                "source": source,
                "output": output,
                "timestamp_semantics": "same-row PIT source, target-race cross-section",
            }
        )

    updated_groups = dict(feature_groups)
    updated_groups["experimental_derived"] = tuple(outputs)
    updated_resolution = dict(resolution)
    updated_resolution["derived_features"] = records
    return feature_columns + tuple(outputs), updated_groups, updated_resolution


def run_cached_experiment(
    *,
    repo_root: str | Path,
    cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit Binary and LambdaRank from a cache and evaluate only 2024."""

    started_at = time.monotonic()
    root = Path(repo_root).resolve()
    cache = _resolve(root, cache_path)
    config_file = _resolve(root, config_path)
    final_output = Path(output_dir).resolve()
    if final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing experiment artifact: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = final_output.with_name(
        f".{final_output.name}.tmp-{uuid4().hex}"
    )
    temporary_output.mkdir()

    try:
        config = load_json(config_file)
        validate_cached_experiment_config(config)
        binary_config = load_json(_resolve(root, config["model_configs"]["binary"]))
        ranker_config = load_json(_resolve(root, config["model_configs"]["lambdarank"]))
        _validate_model_configs(config, binary_config, ranker_config)

        cached_frame, cache_meta = read_model_frame_cache(cache)
        frame, isolation = isolate_pre_2025_frame(cached_frame)
        del cached_frame
        all_features = tuple(cache_meta["feature_columns"])
        feature_columns, feature_groups, feature_resolution = (
            resolve_semantic_feature_selection(all_features, config)
        )
        feature_columns, feature_groups, feature_resolution = (
            add_registered_derived_features(
                frame,
                feature_columns,
                feature_groups,
                feature_resolution,
                config,
            )
        )

        common_kwargs = {
            "feature_columns": feature_columns,
            "race_id_column": "race_id",
            "finish_position_column": "model_finish_position",
            "split_column": "split",
        }
        binary_model = train_binary(
            frame,
            params=binary_config["parameters"],
            early_stopping_rounds=binary_config.get("early_stopping_rounds"),
            **common_kwargs,
        )
        ranker_model = train_ranker(
            frame,
            params=ranker_config["parameters"],
            early_stopping_rounds=ranker_config.get("early_stopping_rounds"),
            **common_kwargs,
        )

        scoring = frame.loc[frame["split"].isin(("calibration", "development"))].copy()
        scoring["pred_binary_raw"] = predict(
            binary_model, scoring, feature_columns=feature_columns, model_kind="binary"
        )
        scoring["score_binary_logit"] = probability_logits(
            scoring["pred_binary_raw"], epsilon=PROBABILITY_EPSILON
        )
        scoring["score_lambdarank"] = predict(
            ranker_model,
            scoring,
            feature_columns=feature_columns,
            model_kind="lambdarank",
        )
        calibration = scoring.loc[scoring["split"].eq("calibration")]
        binary_temperature = fit_temperature(
            calibration["score_binary_logit"],
            calibration["race_id"],
            calibration["model_finish_position"],
        )
        ranker_temperature = fit_temperature(
            calibration["score_lambdarank"],
            calibration["race_id"],
            calibration["model_finish_position"],
        )

        development = scoring.loc[scoring["split"].eq("development")].copy()
        race_ids = development["race_id"].tolist()
        development["prob_uniform"] = uniform_probabilities(race_ids)
        development["prob_history_rate"] = history_rate_probabilities(
            development["horse_history__career__wins"].fillna(0.0),
            development["horse_history__career__starts"].fillna(0.0),
            race_ids,
        )
        development["prob_binary_logit_softmax_temperature_2023"] = apply_temperature(
            binary_temperature, development["score_binary_logit"], race_ids
        )
        development["prob_lambdarank_softmax_temperature_2023"] = apply_temperature(
            ranker_temperature, development["score_lambdarank"], race_ids
        )
        development["prob_binary_logit_softmax_t1"] = race_softmax(
            development["score_binary_logit"], race_ids
        )
        development["prob_lambdarank_softmax_t1"] = race_softmax(
            development["score_lambdarank"], race_ids
        )

        data_fingerprint = str(cache_meta["data_fingerprint"])
        selected_checksum = feature_columns_checksum(feature_columns)
        cache_feature_checksum = feature_columns_checksum(list(all_features))
        split_rows = {
            split: int(count) for split, count in frame["split"].value_counts().items()
        }
        split_races = {
            str(split): int(count)
            for split, count in frame.groupby("split", observed=True)["race_id"]
            .nunique()
            .items()
        }
        condition_columns = [
            column
            for column in ("course_type", "race_class", "distance_band", "field_size_band")
            if column in development
        ]
        methods = (
            ("uniform", "prob_uniform", None),
            ("history_rate", "prob_history_rate", None),
            (
                "binary_logit_softmax_temperature_2023",
                "prob_binary_logit_softmax_temperature_2023",
                "pred_binary_raw",
            ),
            (
                "lambdarank_softmax_temperature_2023",
                "prob_lambdarank_softmax_temperature_2023",
                "score_lambdarank",
            ),
            ("binary_logit_softmax_t1", "prob_binary_logit_softmax_t1", "pred_binary_raw"),
            ("lambdarank_softmax_t1", "prob_lambdarank_softmax_t1", "score_lambdarank"),
        )
        evaluated = {
            name: evaluate_prediction_frame(
                development,
                probability_column=probability,
                ranking_score_column=ranking,
                finish_position_column="model_finish_position",
                race_id_column="race_id",
                condition_columns=condition_columns,
                final_odds_column=None,
                split_column="split",
                evaluation_split="development",
            )
            for name, probability, ranking in methods
        }
        metrics = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "hypothesis": config["hypothesis"],
            "scope": {
                "model_fit": "2014-2021",
                "early_stopping": "2022",
                "temperature_calibration": "2023",
                "evaluation": "2024 development only",
                "retrospective_used": False,
                "odds_used": False,
                "market_outputs_written": False,
            },
            "data": {
                "fingerprint": data_fingerprint,
                "cache_sha256": sha256_file(cache),
                **isolation,
                "split_rows": split_rows,
                "split_races": split_races,
            },
            "features": {
                "count": len(feature_columns),
                "columns_sha256": selected_checksum,
                "cache_columns_sha256": cache_feature_checksum,
                "groups": {name: list(columns) for name, columns in feature_groups.items()},
                "resolution": feature_resolution,
            },
            "models": {
                "binary": {
                    "best_iteration": _best_iteration(binary_model),
                    "temperature": binary_temperature.temperature,
                },
                "lambdarank": {
                    "best_iteration": _best_iteration(ranker_model),
                    "temperature": ranker_temperature.temperature,
                },
            },
            "development": evaluated,
            "elapsed_seconds": time.monotonic() - started_at,
        }

        write_json(temporary_output / "metrics.json", metrics)
        write_json(
            temporary_output / "feature_schema.json",
            {
                "schema_version": 1,
                "feature_count": len(feature_columns),
                "feature_columns": list(feature_columns),
                "feature_columns_sha256": selected_checksum,
                "feature_groups": {
                    name: list(columns) for name, columns in feature_groups.items()
                },
                "resolution": feature_resolution,
            },
        )
        write_json(temporary_output / "config.json", config)
        write_json(temporary_output / "model_configs" / "binary.json", binary_config)
        write_json(temporary_output / "model_configs" / "lambdarank.json", ranker_config)
        write_json(
            temporary_output / "run_meta.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "hypothesis": config["hypothesis"],
                "seed": config["seed"],
                "experiment_config_hash": canonical_json_hash(config),
                "data_fingerprint": data_fingerprint,
                "cache_sha256": metrics["data"]["cache_sha256"],
                "feature_columns_sha256": selected_checksum,
                "git": git_state(root),
                "retrospective_used": False,
                "odds_used": False,
                "software": _software_versions(),
            },
        )
        _write_models_and_importance(
            temporary_output, binary_model, ranker_model, feature_columns
        )
        _write_development_predictions(temporary_output, development)
        write_artifact_manifest(temporary_output)
        temporary_output.replace(final_output)
        return metrics
    except BaseException:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


def _validate_model_configs(
    experiment: dict[str, Any], binary: dict[str, Any], ranker: dict[str, Any]
) -> None:
    expected = {
        "binary": (binary, "lightgbm_binary", "binary"),
        "lambdarank": (ranker, "lightgbm_lambdarank", "lambdarank"),
    }
    for name, (config, family, objective) in expected.items():
        if config.get("model_family") != family:
            raise ValueError(f"{name} base config has wrong model_family")
        if config.get("parameters", {}).get("objective") != objective:
            raise ValueError(f"{name} base config has wrong objective")
        if int(config.get("seed", -1)) != experiment["seed"]:
            raise ValueError(f"{name} base config seed differs from experiment seed")
        if int(config.get("parameters", {}).get("random_state", -1)) != experiment["seed"]:
            raise ValueError(f"{name} random_state differs from experiment seed")


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else None


def _write_models_and_importance(
    output: Path,
    binary_model: Any,
    ranker_model: Any,
    feature_columns: tuple[str, ...],
) -> None:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    binary_model.booster_.save_model(str(model_dir / "binary.txt"))
    ranker_model.booster_.save_model(str(model_dir / "lambdarank.txt"))
    rows: list[dict[str, Any]] = []
    for model_name, model in (("binary", binary_model), ("lambdarank", ranker_model)):
        split = model.booster_.feature_importance(importance_type="split")
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_total = float(np.sum(gain))
        for feature, split_value, gain_value in zip(feature_columns, split, gain):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance_split": int(split_value),
                    "importance_gain": float(gain_value),
                    "importance_gain_fraction": (
                        float(gain_value) / gain_total if gain_total > 0.0 else 0.0
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output / "feature_importance.csv", index=False)


def _write_development_predictions(output: Path, frame: pd.DataFrame) -> None:
    columns = (
        "race_id",
        "race_date",
        "horse_id",
        "horse_number",
        "split",
        "course_type",
        "distance",
        "race_class",
        "field_size",
        "finish_position",
        "model_finish_position",
        "winner_label",
        "pred_binary_raw",
        "score_lambdarank",
        "prob_uniform",
        "prob_history_rate",
        "prob_binary_logit_softmax_t1",
        "prob_binary_logit_softmax_temperature_2023",
        "prob_lambdarank_softmax_t1",
        "prob_lambdarank_softmax_temperature_2023",
    )
    available = [column for column in columns if column in frame]
    forbidden = {"final_win_odds", "final_popularity"}.intersection(available)
    if forbidden:
        raise AssertionError(f"market columns reached cached predictions: {sorted(forbidden)}")
    frame.loc[:, available].to_csv(
        output / "predictions_2024.csv.gz", index=False, compression="gzip"
    )


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed-as-package"
    return versions
