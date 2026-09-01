from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from horse_pred.modeling import validate_grouped_rows, winner_mass_targets
from horse_pred.racewise_probability import (
    FrozenNumericTransform,
    conditional_logit_loss_gradient,
    fit_linear_utility,
    grouped_softmax_numpy,
    native_race_probability,
)
from horse_pred.s1_two_axis_study import _feature_scope
from horse_pred.s2_racewise_probability_study import (
    _capacity_gate,
    validate_s2_preregistration,
)
from horse_pred.two_axis_race_value import PERFORMANCE_COLUMN

PREREGISTRATION = Path(
    "experiments/s2_supervised_racewise_probability_20260901/preregistration.json"
)


def _config() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_s2_preregistration_freezes_firewall_folds_arms_and_gate() -> None:
    config = _config()
    validate_s2_preregistration(config)

    assert config["forbidden_years"] == [2023, 2024, 2025]
    assert config["market_used"] is False
    assert config["final_odds_used"] is False
    assert list(config["arms"]) == ["B0", "B1", "R0", "R1"]
    assert [fold["evaluation_year"] for fold in config["folds"]] == [2020, 2021, 2022]
    assert config["capacity_gate"]["uses_evaluation"] is False
    assert config["capacity_gate"]["threshold"] == 0.75


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("max_source_date",), "2023-12-31"),
        (("forbidden_years",), [2024, 2025]),
        (("market_used",), True),
        (("final_odds_used",), True),
        (("capacity_gate", "uses_evaluation"), True),
        (("capacity_gate", "threshold"), 0.5),
        (("probability", "temperature_fit_role"), "evaluation"),
        (("stop_after_s2",), False),
    ],
)
def test_s2_preregistration_rejects_protocol_mutations(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(_config())
    destination = config
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(ValueError):
        validate_s2_preregistration(config)


def test_conditional_logit_matches_hand_loss_and_finite_difference_gradient() -> None:
    race_ids = ["a", "a", "b", "b", "b"]
    structure = validate_grouped_rows(race_ids)
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0]]
    )
    targets = np.array(winner_mass_targets([1, 2, 2, 1, 3], race_ids))
    coefficients = np.array([0.2, -0.35])
    loss, gradient = conditional_logit_loss_gradient(
        coefficients, features, targets, structure, l2=0.001
    )
    probabilities = grouped_softmax_numpy(features @ coefficients, structure)
    expected = -(np.log(probabilities[0]) + np.log(probabilities[3])) / 2
    expected += 0.5 * 0.001 * float(coefficients @ coefficients)
    assert loss == pytest.approx(expected)

    epsilon = 1e-6
    numeric = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        step = np.zeros_like(coefficients)
        step[index] = epsilon
        upper = conditional_logit_loss_gradient(
            coefficients + step, features, targets, structure, l2=0.001
        )[0]
        lower = conditional_logit_loss_gradient(
            coefficients - step, features, targets, structure, l2=0.001
        )[0]
        numeric[index] = (upper - lower) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numeric, atol=1e-7)


def test_dead_heat_mass_and_native_probabilities_sum_to_one() -> None:
    race_ids = ["dead", "dead", "dead", "normal", "normal"]
    targets = winner_mass_targets([1, 1, 3, 1, 2], race_ids)
    assert targets == [0.5, 0.5, 0.0, 1.0, 0.0]
    probabilities = native_race_probability([1.0, 2.0, -1.0, 0.0, 0.5], race_ids)
    np.testing.assert_allclose(probabilities[:3].sum(), 1.0)
    np.testing.assert_allclose(probabilities[3:].sum(), 1.0)


def test_race_common_shift_and_race_permutation_leave_choice_loss_unchanged() -> None:
    race_ids = ["a", "a", "b", "b"]
    structure = validate_grouped_rows(race_ids)
    utility = np.array([0.2, -0.1, 1.0, -2.0])
    shifted = utility + np.array([7.0, 7.0, -4.0, -4.0])
    np.testing.assert_allclose(
        grouped_softmax_numpy(utility, structure),
        grouped_softmax_numpy(shifted, structure),
    )

    permuted_ids = ["b", "b", "a", "a"]
    permuted = np.array([1.0, -2.0, 0.2, -0.1])
    expected = grouped_softmax_numpy(utility, structure)
    actual = grouped_softmax_numpy(permuted, validate_grouped_rows(permuted_ids))
    np.testing.assert_allclose(actual, expected[[2, 3, 0, 1]])


def test_train_only_transform_is_invariant_to_future_append_and_eval_mutation() -> None:
    train = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0]])
    transform = FrozenNumericTransform.fit(train, ["a", "b"])
    baseline = transform.transform([[4.0, np.nan]])
    appended = np.vstack([train, [[10000.0, -9999.0]]])
    # The frozen transform, not a refit transform, must be used later.
    np.testing.assert_allclose(transform.transform([[4.0, np.nan]]), baseline)
    assert FrozenNumericTransform.fit(appended, ["a", "b"]).audit() != transform.audit()


def test_linear_fit_is_deterministic_and_validation_outcome_cannot_change_transform() -> None:
    train_x = np.array([[0.0], [1.0], [1.0], [0.0]])
    train_ids = ["a", "a", "b", "b"]
    train_finish = [2, 1, 1, 2]
    validation_x = np.array([[0.2], [0.8], [0.9], [0.1]])
    validation_ids = ["c", "c", "d", "d"]
    first = fit_linear_utility(
        train_x,
        train_ids,
        train_finish,
        validation_x,
        validation_ids,
        [2, 1, 1, 2],
        ["signal"],
    )
    second = fit_linear_utility(
        train_x,
        train_ids,
        train_finish,
        validation_x,
        validation_ids,
        [1, 2, 2, 1],
        ["signal"],
    )
    assert first.transform.audit() == second.transform.audit()
    repeated = fit_linear_utility(
        train_x,
        train_ids,
        train_finish,
        validation_x,
        validation_ids,
        [2, 1, 1, 2],
        ["signal"],
    )
    np.testing.assert_array_equal(first.coefficients, repeated.coefficients)


def test_feature_scopes_are_exact_and_forbid_market_or_direct_ids() -> None:
    control = ("context__distance", "horse_history__career__starts")
    assert _feature_scope(control, []) == control
    assert _feature_scope(control, [PERFORMANCE_COLUMN]) == (*control, PERFORMANCE_COLUMN)
    for forbidden in ("final_odds", "popularity", "horse_id", "jockey_id", "trainer_id"):
        with pytest.raises(ValueError, match="forbidden|market"):
            _feature_scope((*control, forbidden), [])


def test_capacity_gate_uses_validation_records_and_global_two_fold_rule() -> None:
    config = _config()
    records = {
        "a": {"uniform_native_log_loss": 2.0, "binary_native_log_loss": 1.0, "linear_native_log_loss": 1.4},
        "b": {"uniform_native_log_loss": 2.0, "binary_native_log_loss": 1.0, "linear_native_log_loss": 1.3},
        "c": {"uniform_native_log_loss": 2.0, "binary_native_log_loss": 1.0, "linear_native_log_loss": 1.2},
    }
    gate = _capacity_gate(records, config)
    assert gate["role"] == "model_validation_only"
    assert gate["under_capacity_folds"] == 2
    assert gate["nonlinear_stage_triggered"] is True


def test_group_validation_rejects_noncontiguous_and_missing_winner() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        validate_grouped_rows(["a", "b", "a"])
    with pytest.raises(ValueError, match="no winner"):
        winner_mass_targets([2, 3], ["a", "a"])
