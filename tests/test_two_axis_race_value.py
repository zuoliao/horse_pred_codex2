from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from horse_pred.two_axis_race_value import (
    FIELD_QUALITY_COLUMN,
    PERFORMANCE_COLUMN,
    TwoAxisRaceValueSpec,
    build_fold_two_axis_history,
)

FOLD = {
    "id": "roll_2022",
    "train_start_year": 2014,
    "train_end_year": 2019,
    "early_stopping_year": 2020,
    "calibration_year": 2021,
    "evaluation_year": 2022,
}
SPEC = TwoAxisRaceValueSpec(min_prior_clean_races=1)


def _runner(
    race_id: str,
    date: str,
    horse_id: str,
    finish: object,
    clock: object,
    *,
    status: str = "finished",
    started: bool = True,
) -> dict[str, object]:
    return {
        "race_id": race_id,
        "race_date": pd.Timestamp(date),
        "horse_id": horse_id,
        "venue_code": "05",
        "course_type": "芝",
        "distance": 1000,
        "ground_state": "良",
        "race_class": "3歳未勝利",
        "status": status,
        "started": started,
        "finish_position": finish,
        "time_raw": clock,
    }


def _race(
    race_id: str,
    date: str,
    first: str,
    second: str,
    *,
    loser_clock: str = "1:01.0",
) -> list[dict[str, object]]:
    return [
        _runner(race_id, date, first, 1, "1:00.0"),
        _runner(race_id, date, second, 2, loser_clock),
    ]


def _base_history() -> pd.DataFrame:
    rows = [
        *_race("warmup", "2013-01-01", "warm-w", "warm-l"),
        *_race("observe", "2014-01-01", "h-win", "horse-x", loser_clock="1:02.0"),
        *_race("target", "2020-01-01", "horse-x", "target-other"),
    ]
    return pd.DataFrame(rows)


def _ordered_features(frame: pd.DataFrame) -> pd.DataFrame:
    history, _, _ = build_fold_two_axis_history(frame, FOLD, SPEC)
    return history.sort_values(["race_id", "horse_id"]).reset_index(drop=True)


def _feature_row(history: pd.DataFrame, race_id: str, horse_id: str) -> pd.Series:
    return history.loc[
        history["race_id"].eq(race_id) & history["horse_id"].eq(horse_id)
    ].iloc[0]


def test_future_append_does_not_change_existing_s1_features() -> None:
    frame = _base_history()
    baseline = _ordered_features(frame)
    appended = pd.concat(
        [frame, pd.DataFrame(_race("future", "2022-01-01", "future-a", "future-b"))],
        ignore_index=True,
    )
    actual = _ordered_features(appended)
    actual = actual.loc[actual["race_id"].isin(baseline["race_id"])].reset_index(drop=True)

    pdt.assert_frame_equal(baseline, actual)


def test_target_outcome_mutation_does_not_change_pre_race_features() -> None:
    frame = _base_history()
    baseline = _ordered_features(frame)
    changed = frame.copy()
    target = changed["race_id"].eq("target")
    changed.loc[target, "finish_position"] = [2, 1]
    changed.loc[target, "time_raw"] = ["1:40.0", "0:59.0"]
    actual = _ordered_features(changed)

    columns = [PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN]
    pdt.assert_frame_equal(
        baseline.loc[baseline["race_id"].eq("target"), columns].reset_index(drop=True),
        actual.loc[actual["race_id"].eq("target"), columns].reset_index(drop=True),
    )


def test_same_date_order_and_result_changes_do_not_change_same_date_features() -> None:
    frame = pd.DataFrame(
        [
            *_race("warmup", "2013-01-01", "warm-w", "warm-l"),
            *_race("prior", "2014-01-01", "shared", "prior-other"),
            *_race("same-a", "2020-01-01", "shared", "same-a-other"),
            *_race("same-b", "2020-01-01", "shared", "same-b-other"),
        ]
    )
    baseline = _ordered_features(frame)
    changed = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    same_a = changed["race_id"].eq("same-a")
    changed.loc[same_a, "finish_position"] = [2, 1]
    changed.loc[same_a, "time_raw"] = ["1:20.0", "1:00.0"]
    actual = _ordered_features(changed)

    columns = ["race_id", "horse_id", PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN]
    expected = baseline.loc[baseline["race_id"].isin(["same-a", "same-b"]), columns]
    observed = actual.loc[actual["race_id"].isin(["same-a", "same-b"]), columns]
    pdt.assert_frame_equal(expected.reset_index(drop=True), observed.reset_index(drop=True))


def test_field_quality_is_constant_and_equals_full_starter_pre_race_mean() -> None:
    _, observations, audit = build_fold_two_axis_history(_base_history(), FOLD, SPEC)
    by_race = observations.groupby("race_id", observed=True)

    assert by_race["field_quality_observation"].nunique(dropna=False).eq(1).all()
    expected = by_race["global_rating_pre"].transform("mean")
    np.testing.assert_allclose(observations["field_quality_observation"], expected)
    assert audit["field_quality"]["race_constant_violations"] == 0
    assert audit["field_quality"]["leave_one_out_used"] is False
    assert audit["field_quality"]["future_opponent_results_used"] is False

    # The second runner of a non-tied field proves this is not leave-one-out.
    target = observations.loc[observations["race_id"].eq("target")]
    assert target["global_rating_pre"].nunique() == 2
    assert target["field_quality_observation"].nunique() == 1


def test_completed_performance_updates_history_only_on_a_later_date() -> None:
    frame = pd.DataFrame(
        [
            *_race("warmup", "2013-01-01", "warm-w", "warm-l"),
            *_race("observe", "2014-01-01", "observe-w", "horse-x", loser_clock="1:02.0"),
            *_race("same-date", "2014-01-01", "horse-x", "same-other"),
            *_race("later", "2014-01-02", "horse-x", "later-other"),
        ]
    )
    history, observations, _ = build_fold_two_axis_history(frame, FOLD, SPEC)

    same = _feature_row(history, "same-date", "horse-x")
    later = _feature_row(history, "later", "horse-x")
    completed = observations.loc[
        observations["race_id"].isin(["observe", "same-date"])
        & observations["horse_id"].eq("horse-x"),
        "performance_residual_observation",
    ]
    assert np.isnan(same[PERFORMANCE_COLUMN])
    assert completed.notna().all()
    assert later[PERFORMANCE_COLUMN] == pytest.approx(completed.mean())


def test_dnf_and_disqualified_races_do_not_add_performance_observations() -> None:
    frame = pd.DataFrame(
        [
            *_race("warmup", "2013-01-01", "warm-w", "warm-l"),
            _runner("dnf", "2014-01-01", "dnf-w", 1, "1:00.0"),
            _runner("dnf", "2014-01-01", "horse-dnf", np.nan, None, status="did_not_finish"),
            _runner("dq", "2014-01-02", "dq-w", 1, "1:00.0"),
            _runner("dq", "2014-01-02", "horse-dq", 2, "1:01.0", status="disqualified"),
            *_race("later-dnf", "2014-01-03", "horse-dnf", "other-dnf"),
            *_race("later-dq", "2014-01-03", "horse-dq", "other-dq"),
        ]
    )
    history, observations, _ = build_fold_two_axis_history(frame, FOLD, SPEC)

    assert observations.loc[
        observations["race_id"].eq("dnf") & observations["horse_id"].eq("horse-dnf"),
        "performance_residual_observation",
    ].isna().all()
    assert observations.loc[
        observations["race_id"].eq("dq"), "performance_residual_observation"
    ].isna().all()
    assert np.isnan(_feature_row(history, "later-dnf", "horse-dnf")[PERFORMANCE_COLUMN])
    assert np.isnan(_feature_row(history, "later-dq", "horse-dq")[PERFORMANCE_COLUMN])


def test_nullable_integer_finish_accepts_real_normalized_dnf_missingness() -> None:
    frame = pd.DataFrame(
        [
            *_race("warmup", "2013-01-01", "warm-w", "warm-l"),
            _runner("dnf", "2014-01-01", "dnf-w", 1, "1:00.0"),
            _runner(
                "dnf",
                "2014-01-01",
                "horse-dnf",
                pd.NA,
                None,
                status="did_not_finish",
            ),
        ]
    )
    frame["finish_position"] = pd.array(frame["finish_position"], dtype="Int64")

    history, observations, _ = build_fold_two_axis_history(frame, FOLD, SPEC)

    assert len(history) == 4
    assert observations.loc[
        observations["horse_id"].eq("horse-dnf"),
        "performance_residual_observation",
    ].isna().all()


def test_condition_normalizer_is_train_only_and_frozen_after_train() -> None:
    frame = _base_history()
    _, _, baseline = build_fold_two_axis_history(frame, FOLD, SPEC)
    changed = frame.copy()
    target = changed["race_id"].eq("target")
    changed.loc[target, "time_raw"] = ["0:50.0", "1:50.0"]
    changed = pd.concat(
        [changed, pd.DataFrame(_race("calibration", "2021-01-01", "cal-a", "cal-b"))],
        ignore_index=True,
    )
    _, _, actual = build_fold_two_axis_history(changed, FOLD, SPEC)

    assert actual["condition_fit"]["frozen_beta_sha256"] == baseline["condition_fit"][
        "frozen_beta_sha256"
    ]
    assert actual["condition_fit"]["max_fit_date"] <= "2019-12-31"
    assert actual["condition_fit"]["evaluation_rows_used"] == 0
    assert actual["condition_fit"]["calibration_rows_used"] == 0
    assert actual["condition_fit"]["early_stopping_rows_used"] == 0


@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_s1_loader_rejects_every_post_2022_outcome_year(year: int) -> None:
    frame = pd.concat(
        [
            _base_history(),
            pd.DataFrame(_race(f"forbidden-{year}", f"{year}-01-01", "fa", "fb")),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match=r"2023\+"):
        build_fold_two_axis_history(frame, FOLD, SPEC)


def test_s1_feature_view_has_only_join_keys_and_the_two_registered_features() -> None:
    history, _, audit = build_fold_two_axis_history(_base_history(), FOLD, SPEC)

    assert list(history.columns) == [
        "race_id",
        "horse_id",
        "race_date",
        PERFORMANCE_COLUMN,
        FIELD_QUALITY_COLUMN,
    ]
    assert audit["scope"]["odds_used"] is False
    assert audit["scope"]["direct_entity_id_features"] == 0
    assert not any(
        token in column.lower()
        for column in (PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN)
        for token in ("odds", "popularity", "horse_id", "jockey_id", "trainer")
    )
