from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import horse_pred.shimba_filter_study as shimba
from horse_pred.artifacts import write_json
from horse_pred.cli import parser
from horse_pred.modeling import TemperatureCalibrator


def _registered_config() -> dict[str, object]:
    return json.loads(
        Path("configs/evaluation/shimba_filter_001_rolling.json").read_text(encoding="utf-8")
    )


def test_config_freezes_binary_pv01_scope_and_one_comparison() -> None:
    config = _registered_config()
    shimba.validate_shimba_filter_config(config)

    bad = copy.deepcopy(config)
    bad["maximum_outcome_year"] = 2024
    with pytest.raises(ValueError, match="maximum_outcome_year"):
        shimba.validate_shimba_filter_config(bad)

    bad = copy.deepcopy(config)
    bad["methods"]["binary_candidate"]["expected_columns_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="identical model and feature configs"):
        shimba.validate_shimba_filter_config(bad)

    bad = copy.deepcopy(config)
    bad["selection_accounting"]["candidate_comparisons_this_run"] = 2
    with pytest.raises(ValueError, match="comparison count"):
        shimba.validate_shimba_filter_config(bad)


def test_new_horse_definition_validates_context_tier_and_race_consistency() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["new", "new", "old", "old"],
            "race_class": ["2歳\u00a0新馬", "2歳 新馬", "1勝クラス", "1勝クラス"],
            "context__class_tier": [0.0, 0.0, 2.0, 2.0],
        }
    )
    assert shimba.new_horse_race_mask(frame).tolist() == [True, True, False, False]

    mismatch = frame.copy()
    mismatch.loc[0, "context__class_tier"] = 2.0
    with pytest.raises(ValueError, match="disagrees"):
        shimba.new_horse_race_mask(mismatch)

    inconsistent = frame.copy()
    inconsistent.loc[1, "race_class"] = "1勝クラス"
    inconsistent.loc[1, "context__class_tier"] = 2.0
    with pytest.raises(ValueError, match="not constant within a race"):
        shimba.new_horse_race_mask(inconsistent)


def test_candidate_excludes_only_whole_new_horse_training_races() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["train_new"] * 2
            + ["train_old"] * 2
            + ["validation_new"] * 2
            + ["calibration_new"] * 2
            + ["evaluation_new"] * 2,
            "rolling_role": ["train"] * 4
            + ["model_validation"] * 2
            + ["calibration"] * 2
            + ["evaluation"] * 2,
            "is_new_horse_race": [True] * 2
            + [False] * 2
            + [True] * 6,
        }
    )
    control, control_report = shimba.prepare_gradient_fit_frame(
        frame, exclude_new_horse_races=False
    )
    candidate, candidate_report = shimba.prepare_gradient_fit_frame(
        frame, exclude_new_horse_races=True
    )

    assert len(control) == len(frame)
    assert control_report["excluded_fit_rows"] == 0
    assert candidate_report["excluded_fit_rows"] == 2
    assert candidate_report["excluded_fit_races"] == 1
    assert set(candidate.loc[candidate["rolling_role"].eq("train"), "race_id"]) == {
        "train_old"
    }
    assert set(candidate.loc[candidate["rolling_role"].ne("train"), "race_id"]) == {
        "validation_new",
        "calibration_new",
        "evaluation_new",
    }


def _decision_payload(
    *, improved_years: int = 4, brier: float = 0.0
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = {
        "race_log_loss": {
            "year_macro_improvement": 0.003,
            "improved_years": improved_years,
        },
        "race_brier": {"year_macro_improvement": brier},
        "ndcg_at_3": {"year_macro_improvement": 0.0},
        "top_1_winner_mass": {"year_macro_improvement": 0.0},
    }
    summary = {"comparisons": {shimba.COMPARISON_ID: {"metrics": metrics}}}
    bootstrap = {
        "paired": {
            shimba.COMPARISON_ID: {
                "race_log_loss": {"point": 0.003, "lower": 0.001, "upper": 0.005}
            }
        }
    }
    return summary, bootstrap


def test_decision_uses_all_race_ll_ci_directions_and_guardrails() -> None:
    config = _registered_config()
    summary, bootstrap = _decision_payload()
    accepted = shimba.shimba_filter_decision(
        summary, bootstrap, config["decision_rule"]
    )
    assert accepted["decision"] == "accept"
    assert accepted["slice_diagnostics_used_for_decision"] is False

    summary, bootstrap = _decision_payload(improved_years=2)
    assert (
        shimba.shimba_filter_decision(summary, bootstrap, config["decision_rule"])[
            "decision"
        ]
        == "inconclusive"
    )

    summary, bootstrap = _decision_payload(brier=-0.0011)
    assert (
        shimba.shimba_filter_decision(summary, bootstrap, config["decision_rule"])[
            "decision"
        ]
        == "reject"
    )


class _FakeBooster:
    def __init__(self, feature_count: int) -> None:
        self.feature_count = feature_count

    def save_model(self, path: str) -> None:
        Path(path).write_text("fake shimba model\n", encoding="utf-8")

    def feature_importance(self, *, importance_type: str) -> np.ndarray:
        value = 2.0 if importance_type == "gain" else 1.0
        return np.full(self.feature_count, value)


class _FakeModel:
    best_iteration_ = 9

    def __init__(self, *, candidate: bool, feature_count: int) -> None:
        self.candidate = candidate
        self.booster_ = _FakeBooster(feature_count)

    def get_params(self, *, deep: bool = False) -> dict[str, object]:
        return {"objective": "binary", "candidate": self.candidate, "deep": deep}


def _fixture_cache() -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = ("context__class_tier",) + tuple(
        f"fixture__feature_{index:03d}" for index in range(253)
    )
    rows: list[dict[str, object]] = []
    for year in range(2014, 2026):
        for race_type, day, race_class, tier in (
            ("new", 1, "2歳 新馬", 0.0),
            ("old", 2, "1勝クラス", 2.0),
        ):
            for position in (1, 2):
                row: dict[str, object] = {
                    "race_id": f"{year}_{race_type}",
                    "race_date": f"{year}-01-{day:02d}",
                    "race_class": race_class,
                    "horse_id": f"{year}_{race_type}_h{position}",
                    "horse_number": position,
                    "field_size": 2,
                    "model_finish_position": position,
                    "final_win_odds": float(position + 1),
                    "final_popularity": position,
                }
                for feature in features:
                    row[feature] = tier if feature == "context__class_tier" else float(position)
                rows.append(row)
    return pd.DataFrame(rows), features


def test_runner_preserves_full_nonfit_populations_and_writes_diagnostics(
    tmp_path, monkeypatch
) -> None:
    frame, features = _fixture_cache()
    cache_path = tmp_path / "cache.pkl"
    frame.to_pickle(cache_path)
    write_json(
        cache_path.with_suffix(".pkl.meta.json"),
        {
            "schema_version": 1,
            "data_fingerprint": "fixture-data",
            "row_count": len(frame),
            "race_count": frame["race_id"].nunique(),
            "feature_columns": list(features),
        },
    )
    config = _registered_config()
    config["uncertainty"]["bootstrap_resamples"] = 20
    config["uncertainty"]["block_length_dates"] = 1
    config_path = tmp_path / "shimba.json"
    write_json(config_path, config)

    resolved = {
        method: {
            "model_kind": "binary",
            "feature_columns": features,
            "feature_groups": {"fixture": features},
            "feature_resolution": {"fixture": list(features)},
            "feature_config": {"fixture": True},
            "feature_config_path": tmp_path / "features.json",
            "model_config": {
                "parameters": {"objective": "binary", "random_state": 42},
                "early_stopping_rounds": 2,
            },
            "model_config_path": tmp_path / "model.json",
            "feature_columns_sha256": shimba.PV01_FEATURE_COLUMNS_SHA256,
        }
        for method in (shimba.CONTROL_METHOD, shimba.CANDIDATE_METHOD)
    }
    monkeypatch.setattr(shimba, "resolve_rolling_methods", lambda **kwargs: resolved)

    calls: list[dict[str, int | bool]] = []

    def fake_train(*, frame, feature_columns, **kwargs):
        train = frame.loc[frame["rolling_role"].eq("train")]
        validation = frame.loc[frame["rolling_role"].eq("model_validation")]
        candidate = not bool(train["is_new_horse_race"].any())
        calls.append(
            {
                "candidate": candidate,
                "max_year": int(pd.to_datetime(frame["race_date"]).dt.year.max()),
                "validation_new_rows": int(validation["is_new_horse_race"].sum()),
            }
        )
        return _FakeModel(candidate=candidate, feature_count=len(feature_columns))

    def fake_predict(model, selected, *, feature_columns, model_kind):
        assert feature_columns == features
        assert model_kind == "binary"
        winner = selected["model_finish_position"].eq(1).to_numpy()
        probability = 0.80 if model.candidate else 0.75
        return np.where(winner, probability, 1.0 - probability)

    monkeypatch.setattr(shimba, "train_binary", fake_train)
    monkeypatch.setattr(shimba, "predict", fake_predict)
    monkeypatch.setattr(
        shimba,
        "fit_temperature",
        lambda *args, **kwargs: TemperatureCalibrator(temperature=1.0, fitted=True),
    )

    output = tmp_path / "artifact"
    metrics = shimba.run_shimba_filter_study(
        repo_root=Path.cwd(),
        cache_path=cache_path,
        config_path=config_path,
        output_dir=output,
    )

    assert len(calls) == 8
    assert [call["candidate"] for call in calls] == [False, True] * 4
    assert all(call["max_year"] <= 2023 for call in calls)
    assert all(call["validation_new_rows"] == 2 for call in calls)
    assert metrics["scope"]["rows_used_2024"] == 0
    assert metrics["scope"]["rows_used_2025"] == 0
    assert metrics["scope"]["feature_columns_sha256"] == shimba.PV01_FEATURE_COLUMNS_SHA256
    for fold in metrics["methods"].values():
        assert fold[shimba.CONTROL_METHOD]["fit_population"]["excluded_fit_races"] == 0
        assert fold[shimba.CANDIDATE_METHOD]["fit_population"]["excluded_fit_races"] > 0
    predictions = pd.read_csv(output / "predictions_scoring.csv.gz")
    assert set(predictions["race_slice"]) == {"new_horse", "non_new_horse"}
    assert "final_win_odds" not in predictions
    assert "final_popularity" not in predictions
    assert metrics["decision"]["primary_population"] == "all_evaluation_races"
    assert metrics["decision"]["slice_diagnostics_used_for_decision"] is False
    assert set(metrics["slice_diagnostics"]) == {"new_horse", "non_new_horse"}
    assert (output / "slice_race_metrics.csv.gz").is_file()
    assert (output / "models" / "roll_2023" / "binary_candidate.txt").is_file()
    assert (output / "artifact_manifest.json").is_file()


def test_cli_registers_shimba_filter_without_replacing_existing_commands() -> None:
    args = parser().parse_args(
        ["run-shimba-filter-study", "--cache", "cache.pkl", "--output", "artifact"]
    )
    assert args.command == "run-shimba-filter-study"
    assert args.config == Path("configs/evaluation/shimba_filter_001_rolling.json")
    hpo = parser().parse_args(
        ["run-hpo-study", "--cache", "c", "--config", "x", "--output", "o"]
    )
    opponent = parser().parse_args(
        [
            "build-opponent-recent-cache",
            "--raw-path",
            "r",
            "--baseline-cache",
            "c",
            "--output",
            "o",
        ]
    )
    sectional = parser().parse_args(
        [
            "build-sectional-recent-cache",
            "--raw-path",
            "r",
            "--baseline-cache",
            "c",
            "--output",
            "o",
        ]
    )
    assert hpo.command == "run-hpo-study"
    assert opponent.command == "build-opponent-recent-cache"
    assert sectional.command == "build-sectional-recent-cache"
