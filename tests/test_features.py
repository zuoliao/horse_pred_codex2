from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from horse_pred.features import (
    FORBIDDEN_SOURCE_COLUMNS,
    FeatureConfig,
    build_features,
    build_pit_features,
    feature_groups,
    model_feature_allowlist,
    semantic_feature_groups_v2,
    source_family_knockout_columns,
    validate_model_feature_columns,
)


def _row(
    raceid: str,
    date: str,
    horse_id: str,
    jockey_id: str,
    trainer: str,
    finish: object,
    *,
    distance: int = 1600,
    odds: float = 5.0,
    popularity: int = 3,
) -> dict[str, object]:
    return {
        "raceid": raceid,
        "date": date,
        "horse_id": horse_id,
        "jockey_id": jockey_id,
        "trainer": trainer,
        "着順": finish,
        "distance": distance,
        "race_class": "3歳未勝利",
        "course_type": "芝",
        "ground_state": "良",
        "around": "右",
        "weather": "晴",
        "sex": "牡",
        "age": 3,
        "枠番": 1,
        "馬番": 1,
        "単勝": odds,
        "人気": popularity,
    }


@pytest.fixture
def raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # All four rows must be emitted from the same cold-start date state,
            # including trainer t_shared in a different race.
            _row("202305010101", "2023-01-01", "h1", "j1", "t_shared", 1),
            _row("202305010101", "2023-01-01", "h2", "j2", "t_shared", 2),
            _row("202305010102", "2023-01-01", "h3", "j3", "t_shared", 1),
            _row("202305010102", "2023-01-01", "h4", "j4", "t4", "中止", distance=1800),
            _row("202305010201", "2023-01-02", "h1", "j1", "t_shared", 2),
            _row("202305010201", "2023-01-02", "h2", "j2", "t_shared", 1),
            _row("202305010202", "2023-01-02", "h4", "j4", "t4", "失格", distance=2000),
            _row("202305010202", "2023-01-02", "h3", "j3", "t_shared", 1),
        ]
    )


def _by_horse(features: pd.DataFrame, raceid: str, horse_id: str) -> pd.Series:
    selected = features[
        features["meta__race_id"].eq(raceid) & features["meta__horse_id"].eq(horse_id)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_complete_date_is_emitted_before_any_same_date_update(raw_fixture: pd.DataFrame) -> None:
    features = build_pit_features(raw_fixture)

    first_day = features[features["meta__date"].eq(pd.Timestamp("2023-01-01"))]
    assert (first_day["horse_history__career__starts"] == 0).all()
    assert (first_day["trainer_history__career__starts"] == 0).all()
    assert (first_day["rating__horse_elo_pre"] == 1500).all()

    next_day = _by_horse(features, "202305010201", "h1")
    assert next_day["horse_history__career__starts"] == 1
    # Three t_shared runners started across two races on the prior date.  None
    # could leak into another race on that same date.
    assert next_day["trainer_history__career__starts"] == 3
    assert next_day["rating__horse_elo_pre"] > 1500


def test_input_row_order_within_date_does_not_change_features(raw_fixture: pd.DataFrame) -> None:
    original = build_pit_features(raw_fixture)
    permuted_raw = raw_fixture.sample(frac=1.0, random_state=19).reset_index(drop=True)
    permuted = build_pit_features(permuted_raw)

    key = ["meta__race_id", "meta__horse_id"]
    columns = list(model_feature_allowlist(original))
    left = original.sort_values(key).reset_index(drop=True)[key + columns]
    right = permuted.sort_values(key).reset_index(drop=True)[key + columns]
    pdt.assert_frame_equal(left, right, check_dtype=False)


def test_appending_future_rows_cannot_change_existing_features(raw_fixture: pd.DataFrame) -> None:
    prefix = raw_fixture.iloc[:6].copy()
    future = pd.DataFrame(
        [
            _row("202305020101", "2024-01-01", "h1", "j9", "t9", 1),
            _row("202305020101", "2024-01-01", "h9", "j1", "t_shared", 2),
        ]
    )
    prefix_features = build_pit_features(prefix)
    appended_features = build_pit_features(pd.concat([prefix, future], ignore_index=True)).iloc[: len(prefix)]
    pdt.assert_frame_equal(prefix_features.reset_index(drop=True), appended_features.reset_index(drop=True))


def test_day_window_boundary_is_inclusive_but_target_date_is_not_visible() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", "j1", "t1", 1, distance=1200),
            _row("202305010101", "2023-01-01", "h2", "j2", "t2", 2),
            _row("202305020101", "2023-01-31", "h1", "j1", "t1", 2, distance=1400),
            # Same date, later race number: it still cannot see the earlier race.
            _row("202305020102", "2023-01-31", "h1", "j1", "t1", 1, distance=1600),
            _row("202305020102", "2023-01-31", "h3", "j3", "t3", 2),
            _row("202305030101", "2023-02-01", "h1", "j1", "t1", 1),
            _row("202305030101", "2023-02-01", "h4", "j4", "t4", 2),
        ]
    )
    config = FeatureConfig(count_windows=(1, 3), day_windows=(14, 30), decay_half_lives=(30,))
    features = build_pit_features(raw, config)
    jan31_a = _by_horse(features, "202305020101", "h1")
    jan31_b = _by_horse(features, "202305020102", "h1")
    assert jan31_a["horse_history__days_30__starts"] == 1
    assert jan31_a["horse_history__days_30__distance_sum"] == 1200
    assert jan31_a["horse_history__days_14__starts"] == 0
    state_columns = [
        column
        for column in model_feature_allowlist(features)
        if column.startswith(("horse_history__", "jockey_history__", "trainer_history__"))
    ] + ["rating__horse_elo_pre"]
    pdt.assert_series_equal(jan31_a[state_columns], jan31_b[state_columns], check_names=False)
    feb1 = _by_horse(features, "202305030101", "h1")
    assert feb1["horse_history__career__starts"] == 3


def test_nonfinish_is_retained_and_updates_start_but_not_mean_finish(raw_fixture: pd.DataFrame) -> None:
    features = build_pit_features(raw_fixture)
    dnf = _by_horse(features, "202305010102", "h4")
    assert bool(dnf["meta__is_runner"])
    assert dnf["context__field_size_rows"] == 2

    next_start = _by_horse(features, "202305010202", "h4")
    assert next_start["horse_history__career__starts"] == 1
    assert next_start["horse_history__career__wins"] == 0
    assert next_start["horse_history__career__completed"] == 0
    assert np.isnan(next_start["horse_history__career__mean_finish"])
    assert next_start["horse_history__career__distance_sum"] == 1800
    assert next_start["rating__horse_elo_pre"] < 1500


def test_cancelled_or_excluded_rows_are_retained_but_never_update_state() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "hc", "jc", "tc", "取消"),
            _row("202305010101", "2023-01-01", "hw", "jw", "tw", 1),
            _row("202305010201", "2023-01-02", "hc", "jc", "tc", 1),
            _row("202305010201", "2023-01-02", "hx", "jx", "tx", 2),
            _row("202305010202", "2023-01-02", "hw", "jw", "tw", 1),
            _row("202305010202", "2023-01-02", "hy", "jy", "ty", 2),
        ]
    )
    features = build_pit_features(raw)
    cancelled = _by_horse(features, "202305010101", "hc")
    assert not bool(cancelled["meta__is_runner"])
    assert not bool(cancelled["meta__is_scored_race"])
    assert not bool(_by_horse(features, "202305010101", "hw")["meta__is_scored_race"])
    assert _by_horse(features, "202305010201", "hc")["horse_history__career__starts"] == 0
    # The actual winner remains a valid historical start even though that race
    # is excluded from scored datasets because its as-of field is unknown.
    assert _by_horse(features, "202305010202", "hw")["horse_history__career__starts"] == 1


def test_unknown_start_status_is_not_guessed_or_added_to_history() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "hu", "ju", "tu", "不明"),
            _row("202305010101", "2023-01-01", "hw", "jw", "tw", 1),
            _row("202305010201", "2023-01-02", "hu", "ju", "tu", 1),
            _row("202305010201", "2023-01-02", "hx", "jx", "tx", 2),
        ]
    )
    features = build_pit_features(raw)
    unknown = _by_horse(features, "202305010101", "hu")
    assert pd.isna(unknown["meta__is_runner"])
    assert not bool(unknown["meta__is_scored_race"])
    assert _by_horse(features, "202305010201", "hu")["horse_history__career__starts"] == 0


def test_dead_heat_winners_each_count_as_an_official_historical_win() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", "j1", "t1", "1(同)"),
            _row("202305010101", "2023-01-01", "h2", "j2", "t2", "1(同)"),
            _row("202305010201", "2023-01-02", "h1", "j1", "t1", 1),
            _row("202305010201", "2023-01-02", "h3", "j3", "t3", 2),
            _row("202305010202", "2023-01-02", "h2", "j2", "t2", 1),
            _row("202305010202", "2023-01-02", "h4", "j4", "t4", 2),
        ]
    )
    features = build_pit_features(raw)
    assert _by_horse(features, "202305010201", "h1")["horse_history__career__wins"] == 1
    assert _by_horse(features, "202305010202", "h2")["horse_history__career__wins"] == 1
    assert _by_horse(features, "202305010201", "h1")["rating__horse_elo_pre"] == 1500
    assert _by_horse(features, "202305010202", "h2")["rating__horse_elo_pre"] == 1500


def test_non_flat_race_is_retained_but_not_scored_or_added_to_flat_history() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", "j1", "t1", 1),
            _row("202305010101", "2023-01-01", "h2", "j2", "t2", 2),
            _row("202305010201", "2023-01-02", "h1", "j1", "t1", 1),
            _row("202305010201", "2023-01-02", "h3", "j3", "t3", 2),
            _row("202305010301", "2023-01-03", "h1", "j1", "t1", 1),
            _row("202305010301", "2023-01-03", "h4", "j4", "t4", 2),
        ]
    )
    raw.loc[raw["date"].eq("2023-01-02"), "course_type"] = "障害"
    features = build_pit_features(raw)
    obstacle = _by_horse(features, "202305010201", "h1")
    assert not bool(obstacle["meta__is_scored_race"])
    assert bool(obstacle["meta__is_runner"])
    # The obstacle result does not enter the flat-only state.
    assert _by_horse(features, "202305010301", "h1")["horse_history__career__starts"] == 1


def test_dirt_coded_obstacle_race_never_updates_flat_entity_or_elo_state() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", "j1", "t1", 1),
            _row("202305010101", "2023-01-01", "h2", "j2", "t2", 2),
            _row("202305010201", "2023-01-02", "h1", "j1", "t1", 2),
            _row("202305010201", "2023-01-02", "h3", "j3", "t3", 1),
            _row("202305010301", "2023-01-03", "h1", "j1", "t1", 1),
            _row("202305010301", "2023-01-03", "h4", "j4", "t4", 2),
        ]
    )
    obstacle_mask = raw["date"].eq("2023-01-02")
    raw.loc[obstacle_mask, "course_type"] = "ダート"
    raw.loc[obstacle_mask, "race_class"] = "障害4歳以上未勝利"

    features = build_pit_features(raw)
    obstacle = _by_horse(features, "202305010201", "h1")
    following = _by_horse(features, "202305010301", "h1")

    assert not bool(obstacle["meta__is_flat_race"])
    assert not bool(obstacle["meta__is_scored_race"])
    assert following["horse_history__career__starts"] == 1
    assert following["jockey_history__career__starts"] == 1
    assert following["trainer_history__career__starts"] == 1
    assert following["rating__horse_elo_pre"] == obstacle["rating__horse_elo_pre"]


def test_integrated_api_keeps_metadata_but_exposes_numeric_closed_allowlist(raw_fixture: pd.DataFrame) -> None:
    split_config = json.loads((Path(__file__).parents[1] / "configs" / "splits.json").read_text())
    dataset = build_features(raw_fixture, split_config=split_config)

    for column in ("raceid", "date", "horse_id", "着順", "単勝", "人気", "split"):
        assert column in dataset.frame
    assert (dataset.frame["split"] == "calibration").all()
    assert dataset.feature_columns
    assert all(pd.api.types.is_numeric_dtype(dataset.frame[column]) for column in dataset.feature_columns)
    assert set(FORBIDDEN_SOURCE_COLUMNS).isdisjoint(dataset.feature_columns)
    assert all(not column.startswith("meta__") for column in dataset.feature_columns)
    assert set(dataset.feature_columns) == {
        column for columns in dataset.feature_groups.values() for column in columns
    }
    assert set(dataset.feature_groups) == set(feature_groups(build_pit_features(raw_fixture)))
    assert {
        "context__venue_code",
        "context__surface_turf",
        "context__surface_dirt",
        "context__direction_right",
        "context__sex_male",
        "context__class_tier",
        "context__class_age_min",
    }.issubset(dataset.feature_columns)
    assert dataset.frame.loc[0, "context__venue_code"] == 5
    assert dataset.frame.loc[0, "context__surface_turf"] == 1
    assert dataset.frame.loc[0, "context__class_tier"] == 1
    assert dataset.frame.loc[0, "context__class_age_min"] == 3
    assert not any("ground" in column or "weather" in column for column in dataset.feature_columns)

    validate_model_feature_columns(dataset.frame, dataset.feature_columns)
    with pytest.raises(ValueError, match="outside model feature allowlist"):
        validate_model_feature_columns(dataset.frame, ["単勝"])


def test_split_config_accepts_model_validation_period(raw_fixture: pd.DataFrame) -> None:
    raw = raw_fixture.copy()
    raw["date"] = "2022-06-01"
    config = {
        "train": {"start": "2014-01-01", "end": "2021-12-31"},
        "model_validation": {"start": "2022-01-01", "end": "2022-12-31"},
        "calibration": {"start": "2023-01-01", "end": "2023-12-31"},
    }
    assert (build_features(raw, split_config=config).frame["split"] == "model_validation").all()


def test_duplicate_runner_key_is_rejected(raw_fixture: pd.DataFrame) -> None:
    duplicate = pd.concat([raw_fixture, raw_fixture.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_pit_features(duplicate)


def test_semantic_taxonomy_v2_is_exact_and_dependency_aware(
    raw_fixture: pd.DataFrame,
) -> None:
    dataset = build_features(raw_fixture)
    groups = semantic_feature_groups_v2(dataset.feature_columns)

    assert {name: len(columns) for name, columns in groups.items()} == {
        "current_context": 21,
        "horse_performance": 25,
        "form_workload": 54,
        "suitability": 15,
        "connections": 130,
        "field_relative": 15,
        "rating_value": 8,
    }
    flattened = [column for columns in groups.values() for column in columns]
    assert len(flattened) == len(set(flattened)) == 268
    assert set(flattened) == set(dataset.feature_columns)
    assert len(source_family_knockout_columns(groups, "horse_performance")) == 31
    assert len(source_family_knockout_columns(groups, "form_workload")) == 57
    assert len(source_family_knockout_columns(groups, "connections")) == 136
