from __future__ import annotations

import warnings
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
    grouped_ranking_relevance_targets,
    history_rate_probabilities,
    predict,
    race_balanced_weights,
    race_softmax,
    ranking_relevance_targets,
    train_binary,
    train_huber_regressor,
    train_ranker,
    uniform_baseline,
    validate_chronological_calibration,
    validate_grouped_rows,
    validate_prediction_feature_columns,
    validate_race_splits,
    validate_standard_split_partition,
)


def test_graded_field_half_relevance_handles_even_odd_and_small_fields() -> None:
    positions = [
        *range(1, 9),
        *range(1, 10),
        1,
        1,
        2,
        1,
        2,
        3,
    ]
    assert grouped_ranking_relevance_targets(
        positions,
        [8, 9, 1, 2, 3],
        relevance_scheme="graded_field_half",
    ) == [
        3,
        2,
        2,
        1,
        0,
        0,
        0,
        0,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        3,
        3,
        2,
        3,
        2,
        2,
    ]


def test_graded_field_half_relevance_handles_dead_heats_and_dnf() -> None:
    assert grouped_ranking_relevance_targets(
        [1, 1, 3, 4, 5, 6, 7, 9],
        [8],
        relevance_scheme="graded_field_half",
    ) == [3, 3, 2, 1, 0, 0, 0, 0]
    # In a two-runner group, position 3 is the DNF sentinel, not a top-three finish.
    assert grouped_ranking_relevance_targets(
        [1, 3],
        [2],
        relevance_scheme="graded_field_half",
    ) == [3, 0]


def test_grouped_relevance_default_is_existing_top3_mapping() -> None:
    positions = [1, 2, 3, 4, 1, 1, 3, 5, 6]
    assert grouped_ranking_relevance_targets(positions, [4, 5]) == (
        ranking_relevance_targets(positions)
    )


@pytest.mark.parametrize(
    ("positions", "group_sizes", "message"),
    [
        ([1, 2], [3], "expected 3"),
        ([1], [0], "positive integers"),
        ([0], [1], "finish_positions"),
        ([3], [1], "finish_positions"),
    ],
)
def test_grouped_relevance_validates_group_sizes_and_positions(
    positions: list[int], group_sizes: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        grouped_ranking_relevance_targets(positions, group_sizes)


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


def test_train_ranker_propagates_relevance_scheme_to_train_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingRanker:
        def __init__(self) -> None:
            self.training_targets: list[int] | None = None
            self.validation_targets: list[int] | None = None

        def fit(
            self,
            features: object,
            targets: list[int],
            *,
            eval_X: object = None,
            eval_y: list[int] | None = None,
            **fit_options: object,
        ) -> CapturingRanker:
            del features, eval_X, fit_options
            self.training_targets = targets
            self.validation_targets = eval_y
            return self

    capturing_ranker = CapturingRanker()
    monkeypatch.setattr(
        "horse_pred.modeling.build_lightgbm_estimator",
        lambda model_kind, *, params=None: capturing_ranker,
    )
    rows = []
    for split, race_id, field_size in [
        ("train", "train-race", 8),
        ("model_validation", "validation-race", 9),
    ]:
        for position in range(1, field_size + 1):
            rows.append(
                {
                    "race_id": race_id,
                    "finish_position": position,
                    "split": split,
                    "form": float(field_size - position),
                }
            )

    fitted = train_ranker(
        pd.DataFrame(rows),
        feature_columns=["form"],
        relevance_scheme="graded_field_half",
        early_stopping_rounds=None,
    )

    assert fitted is capturing_ranker
    assert capturing_ranker.training_targets == [3, 2, 2, 1, 0, 0, 0, 0]
    assert capturing_ranker.validation_targets == [3, 2, 2, 1, 1, 0, 0, 0, 0]


def test_train_huber_filters_missing_targets_and_uses_race_balanced_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingRegressor:
        def __init__(self) -> None:
            self.targets: list[float] | None = None
            self.validation_targets: list[float] | None = None
            self.weights: list[float] | None = None

        def fit(
            self,
            features: object,
            targets: list[float],
            *,
            sample_weight: list[float],
            eval_X: object = None,
            eval_y: list[float] | None = None,
            **fit_options: object,
        ) -> CapturingRegressor:
            del features, eval_X, fit_options
            self.targets = targets
            self.validation_targets = eval_y
            self.weights = sample_weight
            return self

    model = CapturingRegressor()
    monkeypatch.setattr(
        "horse_pred.modeling.build_lightgbm_estimator",
        lambda model_kind, *, params=None: model,
    )
    frame = pd.DataFrame(
        {
            "race_id": ["t1", "t1", "t1", "t2", "t2", "v1", "v1"],
            "split": ["train"] * 5 + ["model_validation"] * 2,
            "form": [3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "performance": [0.2, -0.1, float("nan"), 0.3, -0.4, 0.1, float("nan")],
        }
    )

    fitted = train_huber_regressor(
        frame,
        feature_columns=["form"],
        target_column="performance",
        early_stopping_rounds=None,
    )

    assert fitted is model
    assert model.targets == [0.2, -0.1, 0.3, -0.4]
    assert model.validation_targets == [0.1]
    assert model.weights == [0.5, 0.5, 0.5, 0.5]


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ranker = train_ranker(
            frame,
            feature_columns=["form", "load"],
            params={**params, "eval_at": [1, 3, 5]},
            early_stopping_rounds=3,
        )
    assert not any("Found 'eval_at' in params" in str(item.message) for item in caught)
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
