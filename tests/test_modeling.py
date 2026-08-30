from __future__ import annotations

from math import isclose

import pandas as pd
import pytest

from horse_pred.modeling import (
    ComparisonDataset,
    PlattCalibrator,
    apply_temperature,
    coherent_binary_probabilities,
    fit_platt_from_frame,
    fit_temperature,
    fit_temperature_from_frame,
    history_rate_probabilities,
    predict,
    race_balanced_weights,
    race_softmax,
    train_binary,
    train_ranker,
    uniform_baseline,
    validate_chronological_calibration,
    validate_grouped_rows,
    validate_prediction_feature_columns,
    validate_race_splits,
    validate_standard_split_partition,
)


def test_group_sizes_require_contiguous_complete_races() -> None:
    structure = validate_grouped_rows(["r1", "r1", "r2", "r2", "r2"])
    assert structure.race_ids == ("r1", "r2")
    assert structure.group_sizes == (2, 3)
    assert race_balanced_weights(["r1", "r1", "r2", "r2", "r2"]) == [
        0.5,
        0.5,
        1 / 3,
        1 / 3,
        1 / 3,
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        validate_grouped_rows(["r1", "r2", "r1"])


def test_one_race_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="spans split"):
        validate_race_splits(["r1", "r1"], ["train", "model_validation"])


def test_preregistered_split_years_are_enforced() -> None:
    labels = [
        "train",
        "model_validation",
        "calibration",
        "development",
        "retrospective_test",
    ]
    counts = validate_standard_split_partition(
        ["r1", "r2", "r3", "r4", "r5"],
        labels,
        ["2021-12-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
    )
    assert counts == {label: 1 for label in labels}
    with pytest.raises(ValueError, match="outside"):
        validate_standard_split_partition(
            ["r1"], ["calibration"], ["2024-01-01"], require_all_splits=False
        )


def test_shared_dataset_rejects_non_rectangular_or_split_races() -> None:
    dataset = ComparisonDataset.from_rows(
        features=[[1.0], [2.0], [3.0], [4.0]],
        feature_names=["history_rate"],
        race_ids=["r1", "r1", "r2", "r2"],
        finish_positions=[1, 2, 2, 1],
        split_labels=["train", "train", "model_validation", "model_validation"],
    )
    assert dataset.select_split("train").race_ids == ("r1", "r1")


def test_primary_model_rejects_obvious_market_features() -> None:
    validate_prediction_feature_columns(["history_win_rate", "distance"])
    with pytest.raises(ValueError, match="exclude market"):
        validate_prediction_feature_columns(["history_win_rate", "final_odds"])


def test_uniform_and_history_baselines_are_coherent() -> None:
    ids = ["r1", "r1", "r1", "r2", "r2"]
    uniform = uniform_baseline(ids)
    assert uniform == [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]
    history = history_rate_probabilities(
        [3, 0, 0, 0, 0], [4, 4, 0, 0, 0], ids, prior_strength=2.0
    )
    assert isclose(sum(history[:3]), 1.0)
    assert isclose(sum(history[3:]), 1.0)
    # An unstarted runner stays at the prior, above a 0-for-4 runner.
    assert history[0] > history[2] > history[1]


def test_platt_mapping_and_race_normalization() -> None:
    calibrator = PlattCalibrator().fit(
        [-3, -2, -1, 1, 2, 3], [0, 0, 0, 1, 1, 1]
    )
    mapped = calibrator.transform([-2, 0, 2])
    assert mapped[0] < mapped[1] < mapped[2]
    coherent = coherent_binary_probabilities(
        [0.1, 0.3, 0.8, 0.4, 0.6], ["r1", "r1", "r1", "r2", "r2"]
    )
    assert isclose(sum(coherent[:3]), 1.0)
    assert isclose(sum(coherent[3:]), 1.0)


def test_temperature_fit_uses_coherent_race_softmax() -> None:
    ids = ["r1", "r1", "r1", "r2", "r2", "r2"]
    positions = [1, 2, 3, 2, 1, 3]
    scores = [2.0, 0.0, -1.0, 0.0, 2.0, -1.0]
    calibrator = fit_temperature(scores, ids, positions)
    probabilities = apply_temperature(calibrator, scores, ids)
    assert calibrator.temperature > 0
    assert isclose(sum(probabilities[:3]), 1.0)
    assert isclose(sum(probabilities[3:]), 1.0)
    assert probabilities[0] > probabilities[1]
    assert probabilities[4] > probabilities[3]


def test_temperature_frame_wrapper_reads_only_2023_calibration() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["d1", "d1", "c1", "c1", "c2", "c2"],
            "finish_position": [2, 1, 1, 2, 2, 1],
            "score": [100.0, -100.0, 2.0, 0.0, 0.0, 2.0],
            "split": [
                "development",
                "development",
                "calibration",
                "calibration",
                "calibration",
                "calibration",
            ],
        }
    )
    calibrator = fit_temperature_from_frame(frame, score_column="score")
    calibrated = calibrator.transform([2.0, 0.0], ["x", "x"])
    assert calibrated[0] > calibrated[1]


def test_platt_frame_wrapper_reads_only_2023_calibration() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["d1", "d1", "c1", "c1", "c2", "c2"],
            "finish_position": [2, 1, 1, 2, 2, 1],
            "binary_probability": [0.99, 0.01, 0.8, 0.2, 0.2, 0.8],
            "split": [
                "development",
                "development",
                "calibration",
                "calibration",
                "calibration",
                "calibration",
            ],
        }
    )
    calibrator = fit_platt_from_frame(
        frame, score_column="binary_probability", input_kind="probability"
    )
    mapped = calibrator.transform([-2.0, 2.0])
    assert mapped[0] < mapped[1]


def test_calibration_windows_are_strictly_chronological() -> None:
    validate_chronological_calibration(
        ["2021-12-31"], ["2023-01-01"], ["2024-01-01"]
    )
    with pytest.raises(ValueError, match="model-fitting"):
        validate_chronological_calibration(
            ["2023-01-01"], ["2023-01-01"], ["2024-01-01"]
        )


def _lightgbm_fixture() -> pd.DataFrame:
    rows = []
    for split, race_offset, race_count in [
        ("train", 0, 8),
        ("model_validation", 100, 3),
    ]:
        for race_index in range(race_count):
            race_id = f"r{race_offset + race_index}"
            for runner in range(3):
                rows.append(
                    {
                        "race_id": race_id,
                        "finish_position": runner + 1,
                        "split": split,
                        "form": float(3 - runner) + race_index * 0.01,
                        "load": float(runner),
                    }
                )
    return pd.DataFrame(rows)


def test_lightgbm_binary_and_ranker_use_same_frame_contract() -> None:
    frame = _lightgbm_fixture()
    params = {
        "n_estimators": 12,
        "min_child_samples": 1,
        "verbosity": -1,
        "random_state": 7,
    }
    binary = train_binary(
        frame,
        feature_columns=["form", "load"],
        params=params,
        early_stopping_rounds=3,
    )
    ranker = train_ranker(
        frame,
        feature_columns=["form", "load"],
        params=params,
        early_stopping_rounds=3,
    )
    binary_predictions = predict(
        binary, frame.iloc[:6], feature_columns=["form", "load"], model_kind="binary"
    )
    ranking_scores = predict(
        ranker,
        frame.iloc[:6],
        feature_columns=["form", "load"],
        model_kind="lambdarank",
    )
    assert len(binary_predictions) == len(ranking_scores) == 6
    assert all(0.0 <= value <= 1.0 for value in binary_predictions)
    assert all(isinstance(value, float) for value in ranking_scores)


def test_race_softmax_rejects_noncontiguous_groups() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        race_softmax([1, 2, 3], ["r1", "r2", "r1"])
