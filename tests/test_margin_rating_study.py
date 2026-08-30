from __future__ import annotations

import pandas as pd
import pytest

from horse_pred.margin_rating_study import _decision, derive_adjacent_margin_scale
from horse_pred.rating import RatingEvent


def _event(
    race_id: str,
    times: tuple[float, ...],
    statuses: tuple[str, ...],
) -> RatingEvent:
    return RatingEvent(
        race_id=race_id,
        race_date=pd.Timestamp(f"{race_id[:4]}-01-01"),
        surface_key="turf",
        horse_ids=tuple(f"h{index}" for index in range(len(times))),
        finishes=tuple(float(index + 1) for index in range(len(times))),
        source_positions=tuple(range(len(times))),
        result_times_seconds=times,
        finish_statuses=statuses,
        distance_m=1000.0,
    )


def test_adjacent_margin_scale_is_train_year_only_and_excludes_status_races() -> None:
    eligible = _event(
        "201401010101",
        (100.0, 100.1, 100.3),
        ("finished", "finished", "finished"),
    )
    excluded = _event(
        "201501010101",
        (100.0, 101.0),
        ("finished", "demoted"),
    )
    future = _event(
        "202201010101",
        (100.0, 110.0),
        ("finished", "finished"),
    )

    result = derive_adjacent_margin_scale(
        (eligible, excluded, future), {2014, 2015}
    )

    assert result["tau"] == pytest.approx(0.15)
    assert result["eligible_adjacent_pair_count"] == 2
    assert result["strictly_positive_pair_count"] == 2
    assert result["clean_race_count"] == 1
    assert result["excluded_status_race_count"] == 1


def test_decision_requires_positive_log_loss_interval_and_guardrails() -> None:
    config = {
        "selection": {
            "race_log_loss_improvement_ci_lower_exclusive": 0.0,
            "guardrails": {
                "race_brier_improvement_min": -0.001,
                "ndcg_at_3_improvement_min": -0.002,
                "top_1_improvement_min": -0.005,
            },
        }
    }
    improvement = {
        "race_log_loss": 0.01,
        "race_brier": 0.0,
        "ndcg_at_3": 0.0,
        "top_1": 0.0,
    }

    def bootstrap(lower: float, upper: float) -> dict:
        return {
            "paired": {
                "candidate_vs_control": {
                    "race_log_loss": {"lower": lower, "upper": upper}
                }
            }
        }

    assert _decision(improvement, bootstrap(0.001, 0.02), config) == "go"
    assert _decision(improvement, bootstrap(-0.001, 0.02), config) == "inconclusive"
    assert _decision(improvement, bootstrap(-0.02, -0.001), config) == "reject"
    failed_guardrail = {**improvement, "race_brier": -0.002}
    assert _decision(failed_guardrail, bootstrap(0.001, 0.02), config) == "reject"
