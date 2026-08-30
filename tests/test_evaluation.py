from __future__ import annotations

from math import isclose, log

import pandas as pd
import pytest

from horse_pred.evaluation import (
    conditional_race_metrics,
    evaluate_prediction_frame,
    evaluate_predictions,
    final_odds_oracle_diagnostic,
    ndcg_at_k,
    probability_coherence,
    race_brier_score,
    race_log_loss,
    reliability_table,
    top_k_winner_mass,
)

RACE_IDS = ["r1", "r1", "r1", "r2", "r2"]
POSITIONS = [1, 2, 3, 2, 1]
GOOD_PROBABILITIES = [0.8, 0.15, 0.05, 0.2, 0.8]


def test_uniform_log_loss_and_brier_have_known_values() -> None:
    uniform = [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]
    assert isclose(
        race_log_loss(uniform, POSITIONS, RACE_IDS),
        (log(3) + log(2)) / 2,
    )
    assert isclose(race_brier_score(uniform, POSITIONS, RACE_IDS), 7 / 12)


def test_perfect_ranking_scores_ndcg_and_top_k() -> None:
    assert ndcg_at_k(GOOD_PROBABILITIES, POSITIONS, RACE_IDS, k=1) == 1.0
    assert ndcg_at_k(GOOD_PROBABILITIES, POSITIONS, RACE_IDS, k=3) == 1.0
    assert top_k_winner_mass(GOOD_PROBABILITIES, POSITIONS, RACE_IDS, k=1) == 1.0


def test_dead_heat_winner_mass_is_scored_without_double_counting() -> None:
    ids = ["r1", "r1", "r1"]
    positions = [1, 1, 3]
    probabilities = [0.45, 0.45, 0.1]
    assert isclose(race_log_loss(probabilities, positions, ids), -log(0.45))
    assert top_k_winner_mass(probabilities, positions, ids, k=2) == 1.0


def test_noncoherent_probabilities_are_rejected_by_probability_scores() -> None:
    summary = probability_coherence([0.4, 0.4], ["r", "r"])
    assert not summary["coherent"]
    with pytest.raises(ValueError, match="sum to one"):
        race_log_loss([0.4, 0.4], [1, 2], ["r", "r"])


def test_reliability_table_records_binning_and_counts() -> None:
    table = reliability_table(
        [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2, strategy="quantile"
    )
    assert table["strategy"] == "quantile"
    assert table["requested_bins"] == 2
    assert sum(row["count"] for row in table["bins"]) == 4
    assert table["ece"] >= 0.0


def test_conditional_metrics_require_race_constant_labels() -> None:
    result = conditional_race_metrics(
        GOOD_PROBABILITIES,
        POSITIONS,
        RACE_IDS,
        {"surface": ["turf", "turf", "turf", "dirt", "dirt"]},
    )
    assert set(result["surface"]) == {"turf", "dirt"}
    with pytest.raises(ValueError, match="constant within a race"):
        conditional_race_metrics(
            GOOD_PROBABILITIES,
            POSITIONS,
            RACE_IDS,
            {"bad": ["a", "b", "a", "c", "c"]},
        )


def test_final_odds_are_oracle_diagnostic_only_and_have_no_roi_output() -> None:
    diagnostic = final_odds_oracle_diagnostic(
        GOOD_PROBABILITIES,
        [2.0, 5.0, 12.0, 4.0, 1.8],
        POSITIONS,
        RACE_IDS,
    )
    assert "MUST NOT" in diagnostic["usage"]
    assert "roi" not in diagnostic
    assert "profit" not in diagnostic
    assert "model_minus_market_log_loss" in diagnostic
    assert diagnostic["odds_bands"]


def test_final_odds_oracle_accepts_even_money_refund_like_value() -> None:
    diagnostic = final_odds_oracle_diagnostic(
        [0.5, 0.5], [1.0, 3.0], [1, 2], ["r", "r"]
    )
    assert diagnostic["race_count"] == 1


def test_integrated_evaluation_contains_all_required_families() -> None:
    payload = evaluate_predictions(
        GOOD_PROBABILITIES,
        POSITIONS,
        RACE_IDS,
        conditions={"surface": ["turf", "turf", "turf", "dirt", "dirt"]},
        reliability_bins=2,
    )
    assert set(payload) == {
        "coherence",
        "ranking",
        "probability",
        "reliability",
        "conditional",
    }
    assert payload["ranking"]["top_1"] == 1.0
    assert payload["probability"]["race_log_loss"] < 0.3


def test_frame_evaluation_uses_only_requested_evaluation_split() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["d1", "d1", "t1", "t1"],
            "finish_position": [1, 2, 2, 1],
            "probability": [0.8, 0.2, 0.1, 0.9],
            "rank_score": [2.0, 0.0, 0.0, 2.0],
            "surface": ["turf", "turf", "dirt", "dirt"],
            "final_odds": [2.0, 5.0, 6.0, 1.5],
            "split": [
                "development",
                "development",
                "retrospective_test",
                "retrospective_test",
            ],
        }
    )
    development = evaluate_prediction_frame(
        frame,
        probability_column="probability",
        ranking_score_column="rank_score",
        condition_columns=["surface"],
        final_odds_column="final_odds",
        evaluation_split="development",
        reliability_bins=2,
    )
    assert development["coherence"]["race_count"] == 1
    assert set(development["conditional"]["surface"]) == {"turf"}
    assert "final_odds_oracle" in development
    with pytest.raises(ValueError, match="development.*retrospective_test"):
        evaluate_prediction_frame(
            frame,
            probability_column="probability",
            evaluation_split="calibration",
        )
