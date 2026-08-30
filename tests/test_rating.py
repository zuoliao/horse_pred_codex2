from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_pred.features import build_pit_features
from horse_pred.rating import (
    RatingEvent,
    RatingSpec,
    _time_margin_elo_deltas,
    build_rating_history,
)


def _row(
    raceid: str,
    date: str,
    horse_id: str,
    finish: object,
    *,
    surface: str = "芝",
    race_class: str = "3歳未勝利",
) -> dict[str, object]:
    return {
        "raceid": raceid,
        "date": date,
        "horse_id": horse_id,
        "jockey_id": "101",
        "trainer": "trainer",
        "着順": finish,
        "started": False if str(finish) in {"取消", "除外"} else True,
        "distance": 1600,
        "race_class": race_class,
        "course_type": surface,
        "ground_state": "良",
        "around": "右",
        "weather": "晴",
        "sex": "牡",
        "age": 3,
        "枠番": 1,
        "馬番": 1,
    }


def test_r0_pairwise_elo_exactly_matches_feature_builder_and_same_day_batch() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", 1),
            _row("202305010101", "2023-01-01", "h2", 2),
            _row("202305010102", "2023-01-01", "h1", 2),
            _row("202305010102", "2023-01-01", "h3", 1),
            _row("202305010201", "2023-01-02", "h1", 1),
            _row("202305010201", "2023-01-02", "h4", 2),
        ]
    )
    expected = build_pit_features(raw)
    actual = build_rating_history(raw, RatingSpec(family="pairwise_elo"))
    joined = expected.merge(
        actual,
        left_on=["meta__race_id", "meta__horse_id"],
        right_on=["race_id", "horse_id"],
        validate="one_to_one",
    )

    np.testing.assert_allclose(
        joined["rating__horse_elo_pre"], joined["global_state_pre"]
    )
    same_day = joined.loc[joined["meta__race_id"].eq("202305010102") & joined["horse_id"].eq("h1")].iloc[0]
    assert same_day["global_state_pre"] == 1500.0
    next_day = joined.loc[joined["meta__race_id"].eq("202305010201") & joined["horse_id"].eq("h1")].iloc[0]
    assert next_day["global_state_pre"] != 1500.0


def test_rating_ignores_jump_race_and_nonstarter() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", 1),
            _row("202305010101", "2023-01-01", "h2", 2),
            _row(
                "202305010201",
                "2023-01-02",
                "h1",
                2,
                race_class="障害4歳以上未勝利",
            ),
            _row(
                "202305010201",
                "2023-01-02",
                "h3",
                1,
                race_class="障害4歳以上未勝利",
            ),
            _row("202305010301", "2023-01-03", "h1", 1),
            _row("202305010301", "2023-01-03", "hc", "取消"),
            _row("202305010301", "2023-01-03", "h4", 2),
        ]
    )
    history = build_rating_history(raw, RatingSpec(family="pairwise_elo"))

    assert not history["race_id"].eq("202305010201").any()
    assert not history["horse_id"].eq("hc").any()
    h1 = history.loc[
        history["race_id"].eq("202305010301") & history["horse_id"].eq("h1")
    ].iloc[0]
    assert h1["modular_rating__global_starts_pre"] == 1


def test_online_top1_pl_emits_coherent_probabilities_and_updates_after_date() -> None:
    raw = pd.DataFrame(
        [
            _row("202305010101", "2023-01-01", "h1", 1),
            _row("202305010101", "2023-01-01", "h2", 2),
            _row("202305010201", "2023-01-02", "h1", 1),
            _row("202305010201", "2023-01-02", "h3", 2),
        ]
    )
    history = build_rating_history(
        raw, RatingSpec(family="online_top1_pl", learning_rate=0.2)
    )

    sums = history.groupby("race_id")[
        "modular_rating__raw_win_probability_pre"
    ].sum()
    np.testing.assert_allclose(sums, 1.0)
    first = history.loc[history["race_id"].eq("202305010101")]
    np.testing.assert_allclose(
        first["modular_rating__raw_win_probability_pre"], 0.5
    )
    second_h1 = history.loc[
        history["race_id"].eq("202305010201") & history["horse_id"].eq("h1")
    ].iloc[0]
    assert second_h1["modular_rating__raw_win_probability_pre"] > 0.5


def test_rating_spec_rejects_invalid_family_or_surface_weight() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        RatingSpec(family="other")
    with pytest.raises(ValueError, match="surface_blend_weight"):
        RatingSpec(family="pairwise_elo", surface_blend_weight=1.1)


def test_default_ordinal_spec_keeps_legacy_serialization() -> None:
    default = RatingSpec(family="pairwise_elo", k=48.0, scale=200.0)
    explicit = RatingSpec(
        family="pairwise_elo", k=48.0, scale=200.0, pairwise_actual="ordinal"
    )
    assert default == explicit
    assert default.as_dict() == {
        "family": "pairwise_elo",
        "initial_rating": 1500.0,
        "k": 48.0,
        "scale": 200.0,
        "learning_rate": 0.1,
        "surface_blend_weight": 0.0,
    }


def test_time_margin_actual_is_continuous_antisymmetric_and_zero_sum() -> None:
    tau = 0.125
    event = RatingEvent(
        race_id="202205010101",
        race_date=pd.Timestamp("2022-01-01"),
        surface_key="turf",
        horse_ids=("h1", "h2", "h3"),
        finishes=(1.0, 2.0, 3.0),
        source_positions=(0, 1, 2),
        result_times_seconds=(100.0, 100.2, 101.0),
        finish_statuses=("finished", "finished", "finished"),
        distance_m=1600.0,
    )
    spec = RatingSpec(
        family="pairwise_elo",
        k=48.0,
        scale=200.0,
        pairwise_actual="time_margin_logistic",
        time_margin_tau_seconds_per_1000m=tau,
    )
    deltas = _time_margin_elo_deltas(
        event, {horse_id: 1500.0 for horse_id in event.horse_ids}, spec
    )

    assert abs(sum(deltas.values())) < 1e-12
    assert deltas["h1"] > deltas["h2"] > deltas["h3"]
    expected_h1 = 24.0 * ((1.0 / (1.0 + np.exp(-1.0)) - 0.5) + (
        1.0 / (1.0 + np.exp(-5.0)) - 0.5
    ))
    assert deltas["h1"] == pytest.approx(expected_h1)


def test_time_margin_equal_clock_is_neutral_and_dnf_pair_falls_back() -> None:
    spec = RatingSpec(
        family="pairwise_elo",
        k=48.0,
        scale=200.0,
        pairwise_actual="time_margin_logistic",
        time_margin_tau_seconds_per_1000m=0.125,
    )
    equal_clock = RatingEvent(
        race_id="202205010101",
        race_date=pd.Timestamp("2022-01-01"),
        surface_key="turf",
        horse_ids=("h1", "h2"),
        finishes=(1.0, 2.0),
        source_positions=(0, 1),
        result_times_seconds=(100.0, 100.0),
        finish_statuses=("finished", "finished"),
        distance_m=1600.0,
    )
    assert _time_margin_elo_deltas(
        equal_clock, {"h1": 1500.0, "h2": 1500.0}, spec
    ) == {"h1": 0.0, "h2": 0.0}

    dnf = RatingEvent(
        race_id="202205010102",
        race_date=pd.Timestamp("2022-01-01"),
        surface_key="turf",
        horse_ids=("h1", "h2", "h3"),
        finishes=(1.0, 2.0, np.nan),
        source_positions=(0, 1, 2),
        result_times_seconds=(100.0, 100.2, np.nan),
        finish_statuses=("finished", "finished", "did_not_finish"),
        distance_m=1600.0,
    )
    deltas = _time_margin_elo_deltas(
        dnf, {"h1": 1500.0, "h2": 1500.0, "h3": 1500.0}, spec
    )
    assert deltas["h3"] == pytest.approx(-24.0)
    assert abs(sum(deltas.values())) < 1e-12


def test_demotion_forces_whole_race_ordinal_fallback() -> None:
    spec = RatingSpec(
        family="pairwise_elo",
        k=48.0,
        scale=200.0,
        pairwise_actual="time_margin_logistic",
        time_margin_tau_seconds_per_1000m=0.125,
    )
    event = RatingEvent(
        race_id="202205010101",
        race_date=pd.Timestamp("2022-01-01"),
        surface_key="turf",
        horse_ids=("h1", "h2", "h3"),
        finishes=(1.0, 2.0, 3.0),
        source_positions=(0, 1, 2),
        result_times_seconds=(100.0, 100.1, 105.0),
        finish_statuses=("finished", "demoted", "finished"),
        distance_m=1600.0,
    )
    deltas = _time_margin_elo_deltas(
        event, {"h1": 1500.0, "h2": 1500.0, "h3": 1500.0}, spec
    )
    assert deltas == pytest.approx({"h1": 24.0, "h2": 0.0, "h3": -24.0})


def test_time_margin_history_uses_same_day_batch_and_updates_next_day() -> None:
    raw = pd.DataFrame(
        [
            {**_row("202205010101", "2022-01-01", "h1", 1), "time_raw": "1:40.0"},
            {**_row("202205010101", "2022-01-01", "h2", 2), "time_raw": "1:40.2"},
            {**_row("202205010102", "2022-01-01", "h1", 2), "time_raw": "1:40.2"},
            {**_row("202205010102", "2022-01-01", "h3", 1), "time_raw": "1:40.0"},
            {**_row("202205010201", "2022-01-02", "h1", 1), "time_raw": "1:40.0"},
            {**_row("202205010201", "2022-01-02", "h4", 2), "time_raw": "1:40.2"},
        ]
    )
    history = build_rating_history(
        raw,
        RatingSpec(
            family="pairwise_elo",
            k=48.0,
            scale=200.0,
            pairwise_actual="time_margin_logistic",
            time_margin_tau_seconds_per_1000m=0.125,
        ),
    )
    same_day = history.loc[
        history["race_id"].eq("202205010102") & history["horse_id"].eq("h1")
    ].iloc[0]
    next_day = history.loc[
        history["race_id"].eq("202205010201") & history["horse_id"].eq("h1")
    ].iloc[0]
    assert same_day["global_state_pre"] == 1500.0
    assert next_day["global_state_pre"] != 1500.0


def test_time_margin_spec_requires_pairwise_family_and_positive_tau() -> None:
    with pytest.raises(ValueError, match="positive tau"):
        RatingSpec(
            family="pairwise_elo",
            pairwise_actual="time_margin_logistic",
            time_margin_tau_seconds_per_1000m=0.0,
        )
    with pytest.raises(ValueError, match="requires pairwise_elo"):
        RatingSpec(
            family="online_top1_pl",
            pairwise_actual="time_margin_logistic",
            time_margin_tau_seconds_per_1000m=0.125,
        )
