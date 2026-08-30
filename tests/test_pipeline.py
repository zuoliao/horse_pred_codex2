import pandas as pd
import pytest

from horse_pred.features import FeatureDataset
from horse_pred.pipeline import (
    _final_odds_oracle_for_complete_races,
    prepare_model_frame,
    resolve_experiment_features,
    run_mvp,
    validate_experiment_seeds,
)


def test_prepare_model_frame_keeps_dnf_and_excludes_ambiguous_or_jump_races() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1", "r2", "r2", "r3", "r3"],
            "race_date": ["2024-01-01"] * 3 + ["2024-01-02"] * 2 + ["2024-01-03"] * 2,
            "course_type": ["芝"] * 5 + ["障害"] * 2,
            "started": [True, True, True, True, False, True, True],
            "pit_c_scoring_eligible": [True, True, True, False, False, True, True],
            "winner_label": [1, 0, 0, 1, None, 1, 0],
            "finish_position": [1, 2, None, 1, None, 1, 2],
            "final_win_odds": [2.0, 3.0, 4.0, 2.0, None, 2.0, 3.0],
            "split": ["development"] * 7,
            "distance": [1600] * 7,
            "horse_number": [1, 2, 3, 1, 2, 1, 2],
            "feature__value": [0.0] * 7,
        }
    )
    dataset = FeatureDataset(
        frame=frame,
        feature_columns=("feature__value",),
        feature_groups={"test": ("feature__value",)},
    )
    result = prepare_model_frame(dataset)
    assert result["race_id"].tolist() == ["r1", "r1", "r1"]
    assert result["field_size"].tolist() == [3, 3, 3]
    assert result["model_finish_position"].tolist() == [1, 2, 4]


def test_prepare_model_frame_rejects_split_year_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "race_date": ["2025-01-01", "2025-01-01"],
            "course_type": ["芝", "芝"],
            "started": [True, True],
            "pit_c_scoring_eligible": [True, True],
            "winner_label": [1, 0],
            "finish_position": [1, 2],
            "final_win_odds": [2.0, 3.0],
            "split": ["development", "development"],
            "distance": [1600, 1600],
            "horse_number": [1, 2],
            "feature__value": [0.0, 0.0],
        }
    )
    dataset = FeatureDataset(frame, ("feature__value",), {"test": ("feature__value",)})
    with pytest.raises(ValueError, match="outside 'development'"):
        prepare_model_frame(dataset)


def test_resolve_experiment_features_uses_config_and_requires_same_groups() -> None:
    dataset = FeatureDataset(
        pd.DataFrame(),
        ("a", "b"),
        {"first": ("a",), "second": ("b",)},
    )
    binary = {"feature_groups": ["second"]}
    ranker = {"feature_groups": ["second"]}
    columns, groups = resolve_experiment_features(dataset, binary, ranker)
    assert columns == ("b",)
    assert groups == {"second": ("b",)}

    ranker["feature_groups"] = ["first"]
    with pytest.raises(ValueError, match="identical"):
        resolve_experiment_features(dataset, binary, ranker)


def test_validate_experiment_seeds_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="random_state"):
        validate_experiment_seeds(
            {"experiment_id": "bad", "seed": 42, "parameters": {"random_state": 7}}
        )


def test_final_odds_oracle_drops_incomplete_race_without_breaking_core() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2", "r2"],
            "final_win_odds": [2.0, 3.0, 2.0, None],
            "prob": [0.6, 0.4, 0.7, 0.3],
            "model_finish_position": [1, 2, 1, 2],
        }
    )
    result = _final_odds_oracle_for_complete_races(frame, "prob")
    assert result["status"] == "available"
    assert result["coverage"]["eligible_races"] == 1
    assert result["coverage"]["total_races"] == 2


def test_run_mvp_refuses_existing_output_before_reading_data(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_mvp(repo_root=tmp_path, raw_path=tmp_path / "missing.csv", output_dir=output)
