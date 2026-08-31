from __future__ import annotations

import pandas as pd

from horse_pred.graded_rank_study import (
    _study_split,
    _year_frame,
    graded_rank_decision,
)


def _frame() -> pd.DataFrame:
    rows = []
    for year in range(2014, 2024):
        for position in (1, 2):
            rows.append(
                {
                    "race_id": f"{year}-r",
                    "horse_id": f"{year}-h{position}",
                    "race_date": f"{year}-01-01",
                    "field_size": 2,
                    "model_finish_position": position,
                    "sentinel": 999 if year >= 2023 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_year_firewall_drops_post_2022_before_study_use() -> None:
    isolated = _year_frame(_frame(), 2022)

    assert isolated["race_date"].dt.year.max() == 2022
    assert not isolated["sentinel"].eq(999).any()


def test_study_split_is_strictly_chronological() -> None:
    isolated = _year_frame(_frame(), 2022)
    config = {
        "model_fit_years": list(range(2014, 2020)),
        "early_stopping_year": 2020,
        "temperature_calibration_year": 2021,
        "validation_year": 2022,
    }

    labels = _study_split(isolated, config)
    by_year = pd.DataFrame(
        {"year": isolated["race_date"].dt.year, "label": labels}
    ).groupby("year")["label"].first()

    assert set(by_year.loc[2014:2019]) == {"train"}
    assert by_year.loc[2020] == "model_validation"
    assert by_year.loc[2021] == "calibration"
    assert by_year.loc[2022] == "evaluation"


def test_decision_accepts_only_complete_path_and_rejects_guardrail() -> None:
    config = {
        "acceptance": {
            "probability_path": {
                "log_loss_improvement_min": 0.002,
                "brier_improvement_min": 0.0,
                "ndcg_at_3_improvement_min": -0.002,
                "top_1_improvement_min": -0.005,
            },
            "ranking_path": {
                "log_loss_improvement_min": -0.002,
                "brier_improvement_min": -0.001,
                "top_1_improvement_min": -0.005,
            },
        }
    }
    improvement = {
        "race_log_loss": 0.0,
        "race_brier": 0.0,
        "ndcg_at_3": 0.01,
        "top_1": 0.0,
    }
    bootstrap = {
        "paired": {
            "candidate_vs_control": {
                "ndcg_at_3": {"lower": 0.001, "upper": 0.02},
                "race_log_loss": {"lower": -0.001, "upper": 0.001},
            }
        }
    }

    accepted = graded_rank_decision(improvement, bootstrap, config)
    assert accepted["decision"] == "accept"
    assert accepted["ranking_path_passed"]
    failed = graded_rank_decision(
        {**improvement, "race_brier": -0.002}, bootstrap, config
    )
    assert failed["decision"] == "reject"
