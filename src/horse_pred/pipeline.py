"""End-to-end MVP baseline experiment orchestration."""

from __future__ import annotations

import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from horse_pred.artifacts import build_run_meta, write_json
from horse_pred.config import load_json
from horse_pred.data import (
    audit_csv,
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_audit_against_manifest,
    verify_raw_file,
)
from horse_pred.dataset_cache import write_model_frame_cache
from horse_pred.evaluation import (
    evaluate_prediction_frame,
    final_odds_oracle_diagnostic,
    runner_binary_scores,
)
from horse_pred.features import FeatureDataset, build_features
from horse_pred.modeling import (
    STANDARD_SPLIT_YEARS,
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

PROBABILITY_EPSILON = 1e-6


def prepare_model_frame(dataset: FeatureDataset) -> pd.DataFrame:
    """Select the conservative flat-race scoring population."""

    frame = dataset.frame
    required = {
        "course_type",
        "race_class",
        "started",
        "pit_c_scoring_eligible",
        "meta__is_flat_race",
        "winner_label",
        "finish_position",
        "race_id",
        "race_date",
        "final_win_odds",
        "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"feature frame is missing model metadata: {missing}")

    metadata_columns = (
        "course_type",
        "started",
        "pit_c_scoring_eligible",
        "meta__is_flat_race",
        "meta__is_scored_race",
        "winner_label",
        "finish_position",
        "race_id",
        "race_date",
        "final_win_odds",
        "final_popularity",
        "split",
        "distance",
        "race_class",
        "horse_number",
        "馬番",
        "horse_id",
        "馬名",
    )
    selected_columns = list(
        dict.fromkeys(
            [column for column in metadata_columns if column in frame]
            + list(dataset.feature_columns)
        )
    )
    frame = frame.loc[:, selected_columns]

    eligible = frame["pit_c_scoring_eligible"].eq(True)  # noqa: E712
    if "meta__is_scored_race" in frame:
        eligible &= frame["meta__is_scored_race"].eq(True)  # noqa: E712
    race_has_obstacle_class = (
        frame["race_class"]
        .astype("string")
        .str.contains("障害", na=False)
        .groupby(frame["race_id"], sort=False)
        .transform("max")
    )
    selected = frame.loc[
        frame["course_type"].isin(["芝", "ダート"])
        & ~race_has_obstacle_class
        & frame["meta__is_flat_race"].eq(True)  # noqa: E712
        & frame["started"].eq(True)  # noqa: E712 - pandas nullable boolean comparison
        & eligible
        & frame["split"].isin(STANDARD_SPLIT_YEARS)
    ].copy()
    selected["race_id"] = selected["race_id"].astype("string")
    selected["race_date"] = pd.to_datetime(selected["race_date"], errors="raise")
    selected["winner_label"] = pd.to_numeric(selected["winner_label"], errors="coerce")
    winner_count = selected.groupby("race_id", sort=False)["winner_label"].transform("sum")
    selected = selected.loc[winner_count.gt(0)].copy()
    selected["field_size"] = selected.groupby("race_id", sort=False)["race_id"].transform("size").astype(int)
    numeric_finish = pd.to_numeric(selected["finish_position"], errors="coerce")
    selected["model_finish_position"] = numeric_finish.fillna(selected["field_size"] + 1).astype(int)
    selected["final_win_odds"] = pd.to_numeric(selected["final_win_odds"], errors="coerce")
    selected["distance_band"] = pd.cut(
        pd.to_numeric(selected["distance"], errors="coerce"),
        bins=[0, 1400, 1800, 2200, math.inf],
        labels=["sprint", "mile", "middle", "long"],
        include_lowest=True,
    ).astype("string")
    selected["field_size_band"] = pd.cut(
        selected["field_size"],
        bins=[0, 9, 13, 16, math.inf],
        labels=["small", "medium", "large", "very_large"],
        include_lowest=True,
    ).astype("string")

    sort_columns = ["race_date", "race_id"]
    if "horse_number" in selected:
        sort_columns.append("horse_number")
    elif "馬番" in selected:
        sort_columns.append("馬番")
    selected = selected.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    for column in dataset.feature_columns:
        if selected[column].dtype != np.dtype("float32"):
            selected[column] = pd.to_numeric(selected[column], errors="coerce").astype(
                "float32"
            )
    if not selected.groupby("race_id", sort=False)["split"].nunique().eq(1).all():
        raise ValueError("a race crosses chronological splits")
    validate_standard_split_partition(
        selected["race_id"],
        selected["split"],
        selected["race_date"],
        require_all_splits=False,
    )
    return selected


def resolve_experiment_features(
    dataset: FeatureDataset,
    binary_config: dict[str, Any],
    ranker_config: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Resolve configured feature groups and enforce a fair model comparison."""

    binary_groups = tuple(binary_config.get("feature_groups", ()))
    ranker_groups = tuple(ranker_config.get("feature_groups", ()))
    if not binary_groups or not ranker_groups:
        raise ValueError("both experiment configs must declare feature_groups")
    if binary_groups != ranker_groups:
        raise ValueError("Binary and LambdaRank must use identical ordered feature_groups")
    unknown = sorted(set(binary_groups).difference(dataset.feature_groups))
    if unknown:
        raise ValueError(f"experiment config references unknown feature groups: {unknown}")

    selected_groups = {
        group: tuple(dataset.feature_groups[group]) for group in binary_groups
    }
    columns = tuple(column for group in binary_groups for column in selected_groups[group])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("configured feature groups must resolve to unique feature columns")
    return columns, selected_groups


def validate_experiment_seeds(*configs: dict[str, Any]) -> None:
    """Require the artifact seed and effective LightGBM seed to agree."""

    for config in configs:
        seed = config.get("seed")
        random_state = config.get("parameters", {}).get("random_state")
        if seed is None or random_state is None or int(seed) != int(random_state):
            raise ValueError(
                f"{config.get('experiment_id', 'experiment')} seed must equal parameters.random_state"
            )


def run_mvp(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path = "configs/data_manifest.json",
    split_path: str | Path = "configs/splits.json",
    binary_config_path: str | Path = "configs/exp_001_binary.json",
    ranker_config_path: str | Path = "configs/exp_002_lambdarank.json",
    include_retrospective_test: bool = False,
    model_frame_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run audit, PIT features, both LightGBM models, calibration, and EVAL-01."""

    started_at = time.monotonic()
    root = Path(repo_root).resolve()
    final_output = Path(output_dir).resolve()
    if final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing experiment artifact: {final_output}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = final_output.with_name(f".{final_output.name}.tmp-{uuid4().hex}")
    output.mkdir()

    manifest = load_manifest(_resolve(root, manifest_path))
    split_config = load_json(_resolve(root, split_path))
    binary_config = load_json(_resolve(root, binary_config_path))
    ranker_config = load_json(_resolve(root, ranker_config_path))
    validate_experiment_seeds(binary_config, ranker_config)
    expected_hash = manifest["raw_file"]["sha256"]

    fingerprint = verify_raw_file(raw_path, manifest)
    audit = audit_csv(raw_path, manifest)
    verify_audit_against_manifest(audit, manifest)
    write_json(output / "data_audit.json", {"fingerprint": fingerprint, "audit": audit})

    raw = load_raw(raw_path, expected_sha256=expected_hash)
    normalized = normalize_raw(raw)
    del raw
    # Preserve every normalized race through PIT feature construction.  The
    # feature builder owns the race-level flat/jump decision so a jump race
    # with a turf/dirt course label cannot update flat historical state.
    features = build_features(normalized, split_config=split_config)
    del normalized
    model_frame = prepare_model_frame(features)
    if model_frame_cache_path is not None:
        write_model_frame_cache(
            _resolve(root, model_frame_cache_path),
            model_frame,
            features,
            data_fingerprint=expected_hash,
        )
    feature_columns, feature_groups = resolve_experiment_features(
        features, binary_config, ranker_config
    )
    del features

    validate_standard_split_partition(
        model_frame["race_id"],
        model_frame["split"],
        model_frame["race_date"],
        require_all_splits=True,
    )

    feature_schema = {
        "schema_version": 1,
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "feature_groups": {key: list(value) for key, value in feature_groups.items()},
        "row_count": len(model_frame),
        "race_count": int(model_frame["race_id"].nunique()),
        "split_rows": {str(key): int(value) for key, value in model_frame["split"].value_counts().items()},
        "split_races": {
            str(key): int(value)
            for key, value in model_frame.groupby("split", observed=True)["race_id"].nunique().items()
        },
        "group_size": _distribution_summary(
            model_frame.groupby("race_id", sort=False).size()
        ),
        "dead_heat_race_count": int(
            model_frame.groupby("race_id", sort=False)["winner_label"].sum().gt(1).sum()
        ),
    }
    write_json(output / "feature_schema.json", feature_schema)

    common_kwargs = {
        "feature_columns": feature_columns,
        "race_id_column": "race_id",
        "finish_position_column": "model_finish_position",
        "split_column": "split",
    }
    binary_model = train_binary(
        model_frame,
        params=binary_config["parameters"],
        early_stopping_rounds=binary_config.get("early_stopping_rounds"),
        **common_kwargs,
    )
    ranker_model = train_ranker(
        model_frame,
        params=ranker_config["parameters"],
        early_stopping_rounds=ranker_config.get("early_stopping_rounds"),
        **common_kwargs,
    )

    scoring_splits = ["calibration", "development"]
    if include_retrospective_test:
        scoring_splits.append("retrospective_test")
    scoring_mask = model_frame["split"].isin(scoring_splits)
    prediction_frame = model_frame.loc[scoring_mask].copy()
    del model_frame

    prediction_frame["pred_binary_raw"] = predict(
        binary_model, prediction_frame, feature_columns=feature_columns, model_kind="binary"
    )
    prediction_frame["score_binary_logit"] = probability_logits(
        prediction_frame["pred_binary_raw"], epsilon=PROBABILITY_EPSILON
    )
    prediction_frame["score_lambdarank"] = predict(
        ranker_model, prediction_frame, feature_columns=feature_columns, model_kind="lambdarank"
    )

    calibration_mask = prediction_frame["split"].eq("calibration")
    calibration = prediction_frame.loc[calibration_mask]
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
    race_ids = prediction_frame["race_id"].tolist()
    prediction_frame["prob_binary_logit_softmax_t1"] = race_softmax(
        prediction_frame["score_binary_logit"], race_ids
    )
    prediction_frame["prob_binary_logit_softmax_temperature_2023"] = apply_temperature(
        binary_temperature, prediction_frame["score_binary_logit"], race_ids
    )
    prediction_frame["prob_lambdarank_softmax_t1"] = race_softmax(
        prediction_frame["score_lambdarank"], race_ids
    )
    prediction_frame["prob_lambdarank_softmax_temperature_2023"] = apply_temperature(
        ranker_temperature, prediction_frame["score_lambdarank"], race_ids
    )
    prediction_frame["prob_uniform"] = uniform_probabilities(race_ids)
    prediction_frame["prob_history_rate"] = history_rate_probabilities(
        prediction_frame["horse_history__career__wins"].fillna(0.0),
        prediction_frame["horse_history__career__starts"].fillna(0.0),
        race_ids,
    )

    condition_columns = ["course_type", "race_class", "distance_band", "field_size_band"]
    evaluation_splits = ["development"]
    if include_retrospective_test:
        evaluation_splits.append("retrospective_test")
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "scope": {
            "pit_quality": "PIT-C event reconstruction",
            "same_day_history_allowed": False,
            "flat_only": True,
            "scratch_or_exclusion_races_scored": False,
            "final_odds_usage": "post-event oracle diagnostic only",
            "odds_in_features": False,
            "odds_in_calibration": False,
            "odds_in_model_selection": False,
            "retrospective_test_opt_in": include_retrospective_test,
            "retrospective_test_evaluated": include_retrospective_test,
            "retrospective_test_used_for_reselection": False,
            "selection_bias_note": (
                "PIT-C scoring excludes entire races containing scratches/exclusions; "
                "this retrospective subset is not an executable live population"
            ),
        },
        "data": {
            "fingerprint": expected_hash,
            "model_row_count": feature_schema["row_count"],
            "model_race_count": feature_schema["race_count"],
            "known_official_race_shortfall": 146,
            "development_2024_shortfall": 108,
            "dead_heat_race_count": feature_schema["dead_heat_race_count"],
            "group_size": feature_schema["group_size"],
        },
        "features": feature_schema,
        "models": {
            "binary": {
                "best_iteration": _best_iteration(binary_model),
                "temperature": binary_temperature.temperature,
                "effective_parameters": _model_parameters(binary_model),
                "probability_mapping": {
                    "name": "race_softmax_clipped_logit_temperature",
                    "formula": "softmax(logit(clip(raw_binary_probability, epsilon, 1-epsilon)) / T)",
                    "epsilon": PROBABILITY_EPSILON,
                    "calibration_split": "2023-01-01/2023-12-31",
                    "objective": "race-level winner-mass Log Loss",
                },
            },
            "lambdarank": {
                "best_iteration": _best_iteration(ranker_model),
                "temperature": ranker_temperature.temperature,
                "effective_parameters": _model_parameters(ranker_model),
                "probability_mapping": {
                    "name": "race_softmax_temperature",
                    "formula": "softmax(raw_lambdarank_score / T)",
                    "calibration_split": "2023-01-01/2023-12-31",
                    "objective": "race-level winner-mass Log Loss",
                },
            },
        },
        "software": _software_versions(),
        "splits": {},
    }
    for split in evaluation_splits:
        split_payload: dict[str, Any] = {}
        for name, probability, ranking in (
            ("uniform", "prob_uniform", None),
            ("history_rate", "prob_history_rate", None),
            ("binary_logit_softmax_t1", "prob_binary_logit_softmax_t1", "pred_binary_raw"),
            (
                "binary_logit_softmax_temperature_2023",
                "prob_binary_logit_softmax_temperature_2023",
                "pred_binary_raw",
            ),
            ("lambdarank_softmax_t1", "prob_lambdarank_softmax_t1", "score_lambdarank"),
            (
                "lambdarank_softmax_temperature_2023",
                "prob_lambdarank_softmax_temperature_2023",
                "score_lambdarank",
            ),
        ):
            split_payload[name] = evaluate_prediction_frame(
                prediction_frame,
                probability_column=probability,
                ranking_score_column=ranking,
                finish_position_column="model_finish_position",
                race_id_column="race_id",
                condition_columns=condition_columns,
                final_odds_column=None,
                split_column="split",
                evaluation_split=split,
            )
            split_payload[name]["final_odds_oracle"] = _final_odds_oracle_for_complete_races(
                prediction_frame.loc[prediction_frame["split"].eq(split)], probability
            )
        raw_split = prediction_frame.loc[prediction_frame["split"].eq(split)]
        split_payload["binary_raw_probability"] = {
            "race_probability_sum": _race_sum_summary(raw_split, "pred_binary_raw"),
            "runner_probability": runner_binary_scores(
                raw_split["pred_binary_raw"],
                raw_split["model_finish_position"],
                raw_split["race_id"],
            ),
        }
        metrics["splits"][split] = split_payload

    metrics["elapsed_seconds"] = time.monotonic() - started_at
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "run_meta_binary.json",
        build_run_meta(
            repo_root=root,
            experiment_config=binary_config,
            split_config=split_config,
            data_manifest={"sha256": expected_hash},
        ),
    )
    write_json(
        output / "run_meta_lambdarank.json",
        build_run_meta(
            repo_root=root,
            experiment_config=ranker_config,
            split_config=split_config,
            data_manifest={"sha256": expected_hash},
        ),
    )
    _write_model_artifacts(output, binary_model, ranker_model, feature_columns)
    _write_predictions(output, prediction_frame)
    _write_market_oracle_inputs(output, prediction_frame)
    write_json(output / "configs" / "split.json", split_config)
    write_json(output / "configs" / "binary.json", binary_config)
    write_json(output / "configs" / "lambdarank.json", ranker_config)
    _write_artifact_manifest(output)
    output.rename(final_output)
    return metrics


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else None


def _race_sum_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    sums = frame.groupby("race_id", sort=False)[column].sum()
    return {
        "min": float(sums.min()),
        "mean": float(sums.mean()),
        "max": float(sums.max()),
        "mean_abs_error_from_one": float((sums - 1.0).abs().mean()),
    }


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="raise")
    return {
        "count": int(numeric.size),
        "min": int(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "max": int(numeric.max()),
    }


def _model_parameters(model: Any) -> dict[str, Any]:
    parameters = dict(model.get_params(deep=False))
    return {
        str(key): value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in parameters.items()
    }


def _software_versions() -> dict[str, str]:
    packages = ("horse-pred", "lightgbm", "numpy", "pandas", "scikit-learn")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed-as-package"
    return versions


def _final_odds_oracle_for_complete_races(
    frame: pd.DataFrame, probability_column: str
) -> dict[str, Any]:
    """Evaluate final odds only on races whose runners all have valid odds."""

    odds = pd.to_numeric(frame["final_win_odds"], errors="coerce")
    valid_row = odds.notna() & np.isfinite(odds) & odds.ge(1.0)
    complete_race = valid_row.groupby(frame["race_id"], sort=False).transform("all")
    eligible = frame.loc[complete_race].copy()
    total_races = int(frame["race_id"].nunique())
    eligible_races = int(eligible["race_id"].nunique())
    coverage = {
        "total_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "total_races": total_races,
        "eligible_races": eligible_races,
        "race_coverage": eligible_races / total_races if total_races else 0.0,
        "exclusion_rule": "exclude an entire race unless every scored runner has finite final odds >= 1",
    }
    if eligible.empty:
        return {"status": "unavailable", "coverage": coverage}
    diagnostic = final_odds_oracle_diagnostic(
        eligible[probability_column],
        eligible["final_win_odds"],
        eligible["model_finish_position"],
        eligible["race_id"],
    )
    return {"status": "available", "coverage": coverage, "diagnostic": diagnostic}


def _write_model_artifacts(
    output: Path, binary_model: Any, ranker_model: Any, feature_columns: tuple[str, ...]
) -> None:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    binary_model.booster_.save_model(str(model_dir / "binary.txt"))
    ranker_model.booster_.save_model(str(model_dir / "lambdarank.txt"))
    rows = []
    for name, model in (("binary", binary_model), ("lambdarank", ranker_model)):
        for feature, importance in zip(feature_columns, model.feature_importances_):
            rows.append({"model": name, "feature": feature, "importance": int(importance)})
    pd.DataFrame(rows).to_csv(output / "feature_importance.csv", index=False)


def _write_predictions(output: Path, frame: pd.DataFrame) -> None:
    columns = [
        "race_id",
        "race_date",
        "horse_id",
        "馬名",
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
    ]
    available = [column for column in columns if column in frame]
    frame.loc[:, available].to_csv(output / "predictions.csv.gz", index=False, compression="gzip")


def _write_market_oracle_inputs(output: Path, frame: pd.DataFrame) -> None:
    columns = [
        "race_id",
        "race_date",
        "horse_id",
        "split",
        "final_win_odds",
        "final_popularity",
    ]
    available = [column for column in columns if column in frame]
    frame.loc[:, available].to_csv(
        output / "final_market_oracle.csv.gz", index=False, compression="gzip"
    )


def _write_artifact_manifest(output: Path) -> None:
    files = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output / "artifact_manifest.json",
        {"schema_version": 1, "files": files},
    )
