from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from horse_pred.performance_target import (
    PERFORMANCE_TARGET_COLUMN,
    ConditionAdjustedPerformanceTargetSpec,
    build_fold_performance_targets,
)

FOLD = {
    "id": "roll_2022",
    "train_start_year": 2014,
    "train_end_year": 2019,
    "early_stopping_year": 2020,
    "calibration_year": 2021,
    "evaluation_year": 2022,
}
SPEC = ConditionAdjustedPerformanceTargetSpec()


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
    *,
    winner: str | None = None,
    loser: str | None = None,
    winner_clock: str = "1:00.0",
    loser_clock: str = "1:01.0",
) -> list[dict[str, object]]:
    winner = winner or f"winner-{race_id}"
    loser = loser or f"loser-{race_id}"
    return [
        _runner(race_id, date, winner, 1, winner_clock),
        _runner(race_id, date, loser, 2, loser_clock),
    ]


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            *_race("train-a", "2014-01-01"),
            *_race(
                "train-b",
                "2015-01-01",
                winner_clock="1:02.0",
                loser_clock="1:03.0",
            ),
            *_race("early-stop", "2020-01-01"),
            *_race("calibration", "2021-01-01"),
        ]
    )


def _ordered_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    targets, audit = build_fold_performance_targets(frame, FOLD, SPEC)
    return (
        targets.sort_values(["race_id", "horse_id"]).reset_index(drop=True),
        audit,
    )


def test_normalizer_is_fit_only_on_fold_train_and_frozen_after_train() -> None:
    baseline_targets, baseline_audit = _ordered_targets(_base_frame())
    changed = _base_frame()
    post_train = changed["race_id"].isin(["early-stop", "calibration"])
    changed.loc[post_train & changed["finish_position"].eq(1), "time_raw"] = "0:50.0"
    changed.loc[post_train & changed["finish_position"].eq(2), "time_raw"] = "1:50.0"
    changed_targets, changed_audit = _ordered_targets(changed)

    baseline_fit = baseline_audit["normalizer_fit"]
    changed_fit = changed_audit["normalizer_fit"]
    assert baseline_fit["frozen_beta_sha256"] == changed_fit["frozen_beta_sha256"]
    assert baseline_fit["max_fit_date"] == "2015-01-01"
    assert baseline_fit["fit_race_count_by_year"] == {"2014": 1, "2015": 1}
    assert baseline_fit["early_stopping_rows_used"] == 0
    assert baseline_fit["calibration_rows_used"] == 0
    assert baseline_fit["evaluation_rows_used"] == 0

    # Post-train outcomes are labels, so changing them changes their targets,
    # while the train-fitted definition above remains byte-for-byte frozen.
    role_races = ["early-stop", "calibration"]
    left = baseline_targets.loc[
        baseline_targets["race_id"].isin(role_races), PERFORMANCE_TARGET_COLUMN
    ].reset_index(drop=True)
    right = changed_targets.loc[
        changed_targets["race_id"].isin(role_races), PERFORMANCE_TARGET_COLUMN
    ].reset_index(drop=True)
    assert not left.equals(right)


def test_post_train_future_append_preserves_beta_and_existing_targets() -> None:
    baseline, baseline_audit = _ordered_targets(_base_frame())
    appended_frame = pd.concat(
        [
            _base_frame(),
            pd.DataFrame(
                _race(
                    "evaluation",
                    "2022-01-01",
                    winner_clock="0:59.5",
                    loser_clock="1:05.0",
                )
            ),
        ],
        ignore_index=True,
    )
    appended, appended_audit = _ordered_targets(appended_frame)
    existing = appended.loc[appended["race_id"].isin(baseline["race_id"])]
    existing = existing.reset_index(drop=True)

    pdt.assert_frame_equal(baseline, existing)
    assert baseline_audit["normalizer_fit"]["frozen_beta_sha256"] == (
        appended_audit["normalizer_fit"]["frozen_beta_sha256"]
    )


def test_same_date_race_and_runner_order_do_not_change_targets() -> None:
    frame = pd.concat(
        [
            _base_frame(),
            pd.DataFrame(_race("same-a", "2022-01-01")),
            pd.DataFrame(
                _race(
                    "same-b",
                    "2022-01-01",
                    winner="shared-horse",
                    loser="same-b-other",
                )
            ),
        ],
        ignore_index=True,
    )
    baseline, baseline_audit = _ordered_targets(frame)
    shuffled, shuffled_audit = _ordered_targets(
        frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    )

    pdt.assert_frame_equal(baseline, shuffled)
    assert baseline_audit["normalizer_fit"]["frozen_beta_sha256"] == (
        shuffled_audit["normalizer_fit"]["frozen_beta_sha256"]
    )


@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_performance_target_builder_rejects_every_post_2022_year(year: int) -> None:
    frame = pd.concat(
        [
            _base_frame(),
            pd.DataFrame(_race(f"forbidden-{year}", f"{year}-01-01")),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"2023\+"):
        build_fold_performance_targets(frame, FOLD, SPEC)


def test_status_and_target_missingness_semantics_are_fail_closed() -> None:
    rows = [*_base_frame().to_dict("records")]
    rows.extend(
        [
            _runner("dnf", "2022-01-01", "dnf-winner", 1, "1:00.0"),
            _runner(
                "dnf",
                "2022-01-01",
                "dnf-horse",
                np.nan,
                None,
                status="did_not_finish",
            ),
            _runner("dq", "2022-01-02", "dq-winner", 1, "1:00.0"),
            _runner(
                "dq",
                "2022-01-02",
                "dq-horse",
                2,
                "1:01.0",
                status="disqualified",
            ),
            _runner("demoted", "2022-01-03", "demoted-winner", 1, "1:00.0"),
            _runner(
                "demoted",
                "2022-01-03",
                "demoted-horse",
                2,
                "1:01.0",
                status="demoted",
            ),
            _runner("dead-heat", "2022-01-04", "dead-a", 1, "1:00.0"),
            _runner("dead-heat", "2022-01-04", "dead-b", 1, "1:00.0"),
            _runner("bad-dead-heat", "2022-01-05", "bad-dead-a", 1, "1:00.0"),
            _runner("bad-dead-heat", "2022-01-05", "bad-dead-b", 1, "1:00.1"),
            _runner("scratch", "2022-01-06", "scratch-winner", 1, "1:00.0"),
            _runner(
                "scratch",
                "2022-01-06",
                "scratched-horse",
                np.nan,
                None,
                status="scratched",
                started=False,
            ),
        ]
    )
    targets, _ = _ordered_targets(pd.DataFrame(rows))

    def race_values(race_id: str) -> pd.Series:
        return targets.loc[
            targets["race_id"].eq(race_id), PERFORMANCE_TARGET_COLUMN
        ]

    dnf = targets.loc[targets["race_id"].eq("dnf")].set_index("horse_id")
    assert np.isfinite(dnf.at["dnf-winner", PERFORMANCE_TARGET_COLUMN])
    assert np.isnan(dnf.at["dnf-horse", PERFORMANCE_TARGET_COLUMN])
    assert race_values("dq").isna().all()
    assert race_values("demoted").isna().all()
    assert race_values("dead-heat").notna().all()
    assert race_values("dead-heat").nunique() == 1
    assert race_values("bad-dead-heat").isna().all()
    assert "scratched-horse" not in set(targets["horse_id"])
    assert race_values("scratch").notna().all()
    finite = targets[PERFORMANCE_TARGET_COLUMN].dropna()
    assert finite.between(-5.0, 5.0).all()


def test_target_table_is_keyed_and_contains_no_model_features() -> None:
    targets, audit = _ordered_targets(_base_frame())

    assert list(targets.columns) == [
        "race_id",
        "horse_id",
        "race_date",
        PERFORMANCE_TARGET_COLUMN,
    ]
    assert not targets.duplicated(["race_id", "horse_id"]).any()
    assert audit["scope"]["rows_used_2023"] == 0
    assert audit["scope"]["rows_used_2024"] == 0
    assert audit["scope"]["rows_used_2025"] == 0
    assert audit["scope"]["odds_used"] is False
    assert audit["scope"]["direct_entity_id_feature_count"] == 0
