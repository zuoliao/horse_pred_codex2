from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import horse_pred.rolling_evaluation as rolling
from horse_pred.artifacts import write_json
from horse_pred.cached_experiment import feature_columns_checksum
from horse_pred.cli import parser
from horse_pred.modeling import TemperatureCalibrator


def _folds() -> list[dict[str, int | str]]:
    return [
        {
            "id": f"roll_{year}",
            "train_start_year": 2014,
            "train_end_year": year - 3,
            "early_stopping_year": year - 2,
            "calibration_year": year - 1,
            "evaluation_year": year,
        }
        for year in range(2020, 2024)
    ]


def _config(
    *,
    feature_config: str = "features.json",
    model_config: str = "model.json",
    feature_count: int = 1,
    feature_hash: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "eval_roll_test",
        "hypothesis": "registered rolling test",
        "seed": 42,
        "maximum_outcome_year": 2023,
        "folds": _folds(),
        "methods": {
            "binary_test": {
                "model_kind": "binary",
                "model_config": model_config,
                "feature_config": feature_config,
                "expected_feature_count": feature_count,
                "expected_columns_sha256": feature_hash,
            }
        },
        "comparisons": [],
        "selection_accounting": {
            "candidate_comparisons_this_run": 0,
            "prior_selection_uses_by_evaluation_year": {
                "2020": 0,
                "2021": 0,
                "2022": 0,
                "2023": 0,
            },
            "multiple_comparison_note": "fixture has no candidate comparison",
        },
        "uncertainty": {
            "bootstrap_resamples": 20,
            "bootstrap_seed": 17,
            "block_length_dates": 1,
            "confidence_level": 0.95,
        },
    }


def test_config_freezes_four_pre_2024_expanding_folds() -> None:
    config = _config()
    rolling.validate_rolling_config(config)

    bad = json.loads(json.dumps(config))
    bad["maximum_outcome_year"] = 2024
    with pytest.raises(ValueError, match="maximum_outcome_year"):
        rolling.validate_rolling_config(bad)

    bad = json.loads(json.dumps(config))
    bad["folds"][2]["train_end_year"] = 2020
    with pytest.raises(ValueError, match="train_end_year"):
        rolling.validate_rolling_config(bad)


def test_source_firewall_and_fold_roles_remove_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "race_id": [f"r{year}" for year in range(2014, 2026)],
            "race_date": [f"{year}-01-01" for year in range(2014, 2026)],
            "model_finish_position": [1] * 12,
        }
    )
    isolated, report = rolling.isolate_rolling_source(frame, maximum_outcome_year=2023)
    assert isolated["race_date"].dt.year.max() == 2023
    assert report["rows_excluded_by_year"] == {"2024": 1, "2025": 1}
    assert report["rows_used_2024"] == 0
    assert report["rows_used_2025"] == 0

    roles = rolling.assign_fold_roles(isolated, _folds()[2])
    by_year = dict(zip(isolated["race_date"].dt.year, roles))
    assert by_year[2019] == "train"
    assert by_year[2020] == "model_validation"
    assert by_year[2021] == "calibration"
    assert by_year[2022] == "evaluation"
    assert pd.isna(by_year[2023])


def _race_metrics() -> pd.DataFrame:
    rows = []
    for year in range(2020, 2024):
        for race_index, date in enumerate((f"{year}-01-04", f"{year}-01-05"), start=1):
            for method, offset in (("control", 0.0), ("candidate", 0.1)):
                rows.append(
                    {
                        "fold_id": f"roll_{year}",
                        "evaluation_year": year,
                        "race_id": f"{year}r{race_index}",
                        "race_date": pd.Timestamp(date),
                        "method": method,
                        "ndcg_at_3": 0.5 + offset,
                        "top_1_winner_mass": 0.3 + offset,
                        "race_log_loss": 2.0 - offset,
                        "race_brier": 0.8 - offset,
                    }
                )
    return pd.DataFrame(rows)


def _comparison() -> list[dict[str, str]]:
    return [
        {
            "id": "candidate_vs_control",
            "candidate": "candidate",
            "reference": "control",
            "type": "hypothesis",
        }
    ]


def test_year_macro_direction_and_stratified_bootstrap_are_deterministic() -> None:
    metrics = _race_metrics()
    summary = rolling.summarize_year_macro(metrics, _comparison())
    comparison = summary["comparisons"]["candidate_vs_control"]
    assert comparison["metrics"]["ndcg_at_3"]["improved_years"] == 4
    assert comparison["metrics"]["race_log_loss"]["improved_years"] == 4
    assert comparison["metrics"]["race_brier"]["year_macro_improvement"] == pytest.approx(0.1)

    options = {
        "comparisons": _comparison(),
        "n_resamples": 30,
        "confidence_level": 0.95,
        "seed": 9,
        "block_length_dates": 1,
    }
    left = rolling.paired_year_stratified_block_bootstrap(metrics, **options)
    right = rolling.paired_year_stratified_block_bootstrap(metrics, **options)
    assert left == right
    for metric in rolling.PRIMARY_METRICS:
        interval = left["paired"]["candidate_vs_control"][metric]
        assert interval["point"] == pytest.approx(0.1)
        assert interval["lower"] == pytest.approx(0.1)
        assert interval["upper"] == pytest.approx(0.1)


class _FakeBooster:
    def save_model(self, path: str) -> None:
        Path(path).write_text("fake rolling model\n", encoding="utf-8")

    def feature_importance(self, *, importance_type: str) -> np.ndarray:
        return np.array([2.0 if importance_type == "gain" else 1.0])


class _FakeModel:
    best_iteration_ = 7
    booster_ = _FakeBooster()

    def get_params(self, *, deep: bool = False) -> dict[str, object]:
        return {"objective": "binary", "deep": deep}


def _cache_frame() -> pd.DataFrame:
    rows = []
    for year in range(2014, 2026):
        for position in (1, 2):
            rows.append(
                {
                    "race_id": f"{year}01010101",
                    "race_date": f"{year}-01-01",
                    "horse_id": f"h{position}",
                    "horse_number": position,
                    "field_size": 2,
                    "model_finish_position": position,
                    "context__distance": 1600.0 + position,
                    "final_win_odds": float(position + 1),
                    "final_popularity": position,
                }
            )
    return pd.DataFrame(rows)


def test_runner_writes_atomic_reusable_odds_free_artifact(tmp_path, monkeypatch) -> None:
    frame = _cache_frame()
    cache_path = tmp_path / "cache.pkl"
    frame.to_pickle(cache_path)
    write_json(
        cache_path.with_suffix(".pkl.meta.json"),
        {
            "schema_version": 1,
            "data_fingerprint": "fixture-data",
            "row_count": len(frame),
            "race_count": frame["race_id"].nunique(),
            "feature_columns": ["context__distance"],
        },
    )
    feature_config_path = tmp_path / "features.json"
    write_json(
        feature_config_path,
        {
            "schema_version": 1,
            "experiment_id": "fixture_features",
            "hypothesis": "fixture include-only scope",
            "seed": 42,
            "model_configs": {"binary": "unused", "lambdarank": "unused"},
            "feature_selection": {"include": ["current_context"]},
        },
    )
    model_config_path = tmp_path / "model.json"
    write_json(
        model_config_path,
        {
            "experiment_id": "fixture_binary",
            "hypothesis": "fixture model",
            "model_family": "lightgbm_binary",
            "seed": 42,
            "parameters": {"objective": "binary", "random_state": 42},
            "early_stopping_rounds": 2,
        },
    )
    checksum = feature_columns_checksum(("context__distance",))
    config = _config(
        feature_config=str(feature_config_path),
        model_config=str(model_config_path),
        feature_hash=checksum,
    )
    config_path = tmp_path / "rolling.json"
    write_json(config_path, config)

    seen_max_years: list[int] = []

    def fake_train(*, frame, **kwargs):
        seen_max_years.append(pd.to_datetime(frame["race_date"]).dt.year.max())
        assert "final_win_odds" in frame  # metadata may remain outside the model allowlist
        return _FakeModel()

    def fake_predict(model, selected, *, feature_columns, model_kind):
        assert feature_columns == ("context__distance",)
        assert model_kind == "binary"
        return [0.8 if position == 1 else 0.2 for position in selected["model_finish_position"]]

    monkeypatch.setattr(rolling, "train_binary", fake_train)
    monkeypatch.setattr(rolling, "predict", fake_predict)
    monkeypatch.setattr(
        rolling,
        "fit_temperature",
        lambda *args, **kwargs: TemperatureCalibrator(temperature=1.0, fitted=True),
    )

    output = tmp_path / "artifact"
    metrics = rolling.run_rolling_evaluation(
        repo_root=Path.cwd(),
        cache_path=cache_path,
        config_path=config_path,
        output_dir=output,
    )

    assert seen_max_years == [2020, 2021, 2022, 2023]
    assert metrics["scope"]["rows_used_2024"] == 0
    assert metrics["scope"]["rows_used_2025"] == 0
    predictions = pd.read_csv(output / "predictions_scoring.csv.gz")
    assert set(predictions["role"]) == {"calibration", "evaluation"}
    assert set(predictions["evaluation_year"]) == {2020, 2021, 2022, 2023}
    assert "final_win_odds" not in predictions
    assert "final_popularity" not in predictions
    assert (output / "models" / "roll_2023" / "binary_test.txt").is_file()
    assert (output / "artifact_manifest.json").is_file()
    assert not any(path.name.startswith(f".{output.name}.tmp") for path in tmp_path.iterdir())

    with pytest.raises(FileExistsError):
        rolling.run_rolling_evaluation(
            repo_root=Path.cwd(),
            cache_path=cache_path,
            config_path=config_path,
            output_dir=output,
        )


def test_feature_resolution_rejects_drop_selection(tmp_path) -> None:
    feature_path = tmp_path / "drop.json"
    write_json(
        feature_path,
        {
            "schema_version": 1,
            "experiment_id": "drop_fixture",
            "hypothesis": "unsafe augmented-cache drop",
            "seed": 42,
            "model_configs": {"binary": "unused", "lambdarank": "unused"},
            "feature_selection": {"drop": []},
        },
    )
    model_path = tmp_path / "model.json"
    write_json(
        model_path,
        {
            "model_family": "lightgbm_binary",
            "seed": 42,
            "parameters": {"objective": "binary", "random_state": 42},
        },
    )
    config = _config(
        feature_config=str(feature_path),
        model_config=str(model_path),
        feature_hash=feature_columns_checksum(("context__distance",)),
    )
    with pytest.raises(ValueError, match="include-only"):
        rolling.resolve_rolling_methods(
            root=Path.cwd(),
            all_feature_columns=("context__distance",),
            config=config,
        )


def test_cli_registers_rolling_command() -> None:
    args = parser().parse_args(
        [
            "run-rolling-evaluation",
            "--cache",
            "cache.pkl",
            "--output",
            "artifact",
        ]
    )
    assert args.command == "run-rolling-evaluation"
    assert args.config == Path("configs/evaluation/eval_roll_001_current_best.json")
