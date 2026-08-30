from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_pred.diagnostics import (
    isolate_development_2024,
    previous_condition_metadata,
    race_aware_group_permutation,
    runner_slice_table,
)


def _split_frame() -> pd.DataFrame:
    rows = []
    for split, year in (
        ("train", 2014),
        ("model_validation", 2022),
        ("calibration", 2023),
        ("development", 2024),
        ("retrospective_test", 2025),
    ):
        for finish in (1, 2):
            rows.append(
                {
                    "race_id": f"{year}01010101",
                    "race_date": f"{year}-01-01",
                    "split": split,
                    "model_finish_position": finish,
                }
            )
    return pd.DataFrame(rows)


def test_isolate_development_is_exactly_2024_and_reports_closed_2025() -> None:
    development, report = isolate_development_2024(_split_frame())

    assert development["split"].eq("development").all()
    assert development["race_date"].dt.year.eq(2024).all()
    assert report["retrospective_2025_rows_in_cache"] == 2
    assert report["retrospective_2025_rows_used"] == 0


def test_isolate_development_rejects_mislabeled_2025() -> None:
    frame = _split_frame()
    frame.loc[frame["split"].eq("development"), "race_date"] = "2025-02-01"
    with pytest.raises(ValueError, match="labeled retrospective_test"):
        isolate_development_2024(frame)


def test_previous_condition_metadata_is_strictly_past() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r2", "r3"],
            "race_date": ["2022-01-01", "2023-01-01", "2024-01-01"],
            "horse_id": ["h1", "h1", "h1"],
            "course_type": ["芝", "ダート", "芝"],
            "distance": [1600, 1800, 2000],
            "context__venue_code": [5.0, 6.0, 8.0],
        }
    )

    result = previous_condition_metadata(frame).set_index("race_id")

    assert pd.isna(result.loc["r1", "previous_eligible_race_date"])
    assert result.loc["r3", "previous_eligible_surface"] == "ダート"
    assert result.loc["r3", "previous_eligible_distance"] == 1800
    assert result.loc["r3", "previous_eligible_race_date"] < pd.Timestamp("2024-01-01")


def test_previous_condition_metadata_rejects_2025_before_shifting() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1"],
            "race_date": ["2025-01-01"],
            "horse_id": ["h1"],
            "course_type": ["芝"],
            "distance": [1600],
            "context__venue_code": [5.0],
        }
    )
    with pytest.raises(ValueError, match=r"2025\+"):
        previous_condition_metadata(frame)


def test_race_aware_group_permutation_preserves_declared_structures() -> None:
    matrix = np.asarray(
        [
            [10.0, 1.0, 101.0],
            [10.0, 2.0, 102.0],
            [10.0, 3.0, 103.0],
            [20.0, 4.0, 104.0],
            [20.0, 5.0, 105.0],
        ]
    )
    race_ids = ["a", "a", "a", "b", "b"]
    actual = race_aware_group_permutation(
        matrix,
        race_ids,
        varying_indices=[1],
        constant_indices=[0],
        rng=np.random.default_rng(17),
    )
    repeated = race_aware_group_permutation(
        matrix,
        race_ids,
        varying_indices=[1],
        constant_indices=[0],
        rng=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(actual, repeated)
    assert len(set(actual[:3, 0])) == 1
    assert len(set(actual[3:, 0])) == 1
    assert {actual[0, 0], actual[3, 0]} == {10.0, 20.0}
    assert sorted(actual[:3, 1]) == [1.0, 2.0, 3.0]
    assert sorted(actual[3:, 1]) == [4.0, 5.0]
    np.testing.assert_array_equal(actual[:, 2], matrix[:, 2])


def test_runner_slices_do_not_report_incoherent_race_log_loss() -> None:
    rows = []
    for race, winner in (("r1", 1), ("r2", 2)):
        for horse in (1, 2):
            finish = 1 if horse == winner else 2
            rows.append(
                {
                    "race_id": race,
                    "model_finish_position": finish,
                    "context__age": float(horse + 2),
                    "horse_history__career__starts": float(horse - 1),
                    "horse_history__days_since_last_start": np.nan if horse == 1 else 40.0,
                    "final_win_odds_oracle": 2.0 if horse == winner else 5.0,
                    "final_popularity_oracle": 1 if horse == winner else 2,
                    "previous_eligible_race_date": pd.NaT if horse == 1 else pd.Timestamp("2023-01-01"),
                    "course_type": "芝",
                    "previous_eligible_surface": np.nan if horse == 1 else "芝",
                    "distance": 1600,
                    "previous_eligible_distance": np.nan if horse == 1 else 1600,
                    "context__venue_code": 5.0,
                    "previous_eligible_venue_code": np.nan if horse == 1 else 5.0,
                    "prob_model": 0.8 if horse == winner else 0.2,
                    "score_model": 0.8 if horse == winner else 0.2,
                }
            )
    frame = pd.DataFrame(rows)

    result = runner_slice_table(frame, {"model": ("prob_model", "score_model")})

    assert "race_log_loss" not in result.columns
    assert "runner_micro_log_loss" in result.columns
    assert set(result.loc[result["dimension"].eq("career_start_band"), "category"]) == {
        "0_debut",
        "1_2",
    }
    assert not result["oracle_only"].all()
