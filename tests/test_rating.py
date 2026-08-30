from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_pred.features import build_pit_features
from horse_pred.rating import RatingSpec, build_rating_history


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
