from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_race_content_cache_control
from horse_pred.features import semantic_feature_groups_v2
from horse_pred.race_content import (
    RACE_CONTENT_COLUMN,
    RaceContentSpec,
    build_race_content_augmented_cache,
    build_race_content_history,
    parse_result_time_seconds,
    signed_time_content_scores,
)


def _race(
    race_id: str,
    date: str,
    horses: list[str],
    finishes: list[object],
    times: list[object],
    *,
    statuses: list[str] | None = None,
    distance: int = 1000,
) -> pd.DataFrame:
    if statuses is None:
        statuses = ["finished"] * len(horses)
    return pd.DataFrame(
        {
            "race_id": race_id,
            "race_date": pd.Timestamp(date),
            "horse_id": horses,
            "finish_position": finishes,
            "status": statuses,
            "started": [True] * len(horses),
            "time_raw": times,
            "distance": distance,
            "course_type": "芝",
            "race_class": "3歳未勝利",
        }
    )


def test_result_time_parser_accepts_only_frozen_format() -> None:
    assert parse_result_time_seconds("1:34.5") == pytest.approx(94.5)
    assert np.isnan(parse_result_time_seconds(None))
    assert np.isnan(parse_result_time_seconds("--"))
    assert np.isnan(parse_result_time_seconds("1:64.5"))
    assert np.isnan(parse_result_time_seconds("94.5"))


def test_signed_content_rewards_big_win_and_distinguishes_close_fourth() -> None:
    big_win = _race(
        "r1",
        "2023-01-01",
        ["winner", "second", "third"],
        [1, 2, 3],
        ["1:40.0", "1:41.0", "1:42.0"],
    )
    big_scores = signed_time_content_scores(big_win)
    assert big_scores.iloc[0] == pytest.approx(1.0)

    close_loss = _race(
        "r2",
        "2023-01-02",
        ["winner", "second", "third", "close_fourth", "far"],
        [1, 2, 3, 4, 5],
        ["1:40.0", "1:40.1", "1:40.1", "1:40.2", "2:20.0"],
    )
    scores = signed_time_content_scores(close_loss)

    assert scores.iloc[3] == pytest.approx(-0.2)
    assert scores.iloc[3] > scores.iloc[4]
    assert scores.iloc[4] == pytest.approx(-5.0)


def test_dead_heat_winners_are_neutral_and_anomalous_races_are_missing() -> None:
    dead_heat = _race(
        "r1",
        "2023-01-01",
        ["h1", "h2", "h3"],
        [1, 1, 3],
        ["1:40.0", "1:40.0", "1:41.0"],
    )
    scores = signed_time_content_scores(dead_heat)
    assert scores.iloc[:2].eq(0.0).all()
    assert scores.iloc[2] == pytest.approx(-1.0)

    demoted = dead_heat.copy()
    demoted.loc[2, "status"] = "demoted"
    assert signed_time_content_scores(demoted).isna().all()


def test_history_is_same_date_batched_and_2025_is_not_emitted() -> None:
    day_one_a = _race(
        "r1", "2023-01-01", ["h1", "h2"], [1, 2], ["1:40.0", "1:41.0"]
    )
    day_one_b = _race(
        "r2", "2023-01-01", ["h1", "h3"], [2, 1], ["1:41.0", "1:40.0"]
    )
    day_two = _race(
        "r3", "2023-01-02", ["h1", "h4"], [1, 2], ["1:40.0", "1:40.5"]
    )
    future = _race(
        "r4", "2025-01-01", ["h1", "h5"], [1, 2], ["1:40.0", "1:41.0"]
    )
    history = build_race_content_history(
        pd.concat([day_one_a, day_one_b, day_two, future], ignore_index=True)
    )

    assert history["race_id"].tolist() == ["r1", "r1", "r2", "r2", "r3", "r3"]
    assert history.loc[history["race_id"].isin(["r1", "r2"]), RACE_CONTENT_COLUMN].isna().all()
    h1_day_two = history.loc[
        history["race_id"].eq("r3") & history["horse_id"].eq("h1"),
        RACE_CONTENT_COLUMN,
    ].iloc[0]
    # h1's +1 and -1 same-date observations cancel; neither could affect the other.
    assert h1_day_two == pytest.approx(0.0)
    assert not history["race_id"].eq("r4").any()


def test_cache_join_preserves_cold_missingness_and_firewalls_2025(tmp_path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
            "split": ["development", "development", "retrospective_test"],
            "horse_number": [1, 2, 1],
            "feature_a": np.array([1.0, 2.0, 3.0], dtype="float32"),
        }
    )
    baseline_path = tmp_path / "baseline.pkl"
    baseline.to_pickle(baseline_path)
    baseline_path.with_suffix(".pkl.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "pandas_pickle",
                "data_fingerprint": "raw",
                "row_count": 3,
                "race_count": 2,
                "feature_columns": ["feature_a"],
            }
        ),
        encoding="utf-8",
    )
    history = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["h1", "h2"],
            RACE_CONTENT_COLUMN: [np.nan, -0.25],
            "_race_content_history_row_present": [True, True],
        }
    )
    output = tmp_path / "candidate.pkl"
    result = build_race_content_augmented_cache(
        baseline_path,
        history,
        output,
        config={"race_content": {"absolute_cap": 5.0}},
    )
    augmented = pd.read_pickle(output)

    assert result["candidate_feature_count"] == 2
    assert np.isnan(augmented.loc[0, RACE_CONTENT_COLUMN])
    assert augmented.loc[1, RACE_CONTENT_COLUMN] == pytest.approx(-0.25)
    assert np.isnan(augmented.loc[2, RACE_CONTENT_COLUMN])
    control = compare_race_content_cache_control(
        baseline_path, output, expected_baseline_feature_count=1
    )
    assert control["passed"]


def test_semantic_taxonomy_places_race_content_in_its_own_group() -> None:
    groups = semantic_feature_groups_v2(
        ("context__distance", RACE_CONTENT_COLUMN)
    )
    assert groups["race_content_time"] == (RACE_CONTENT_COLUMN,)


def test_spec_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="absolute_cap"):
        RaceContentSpec(absolute_cap=0.0)
