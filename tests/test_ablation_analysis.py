from __future__ import annotations

import pandas as pd

from horse_pred.ablation_analysis import compare_ablation_predictions


def _frame() -> pd.DataFrame:
    rows = []
    for day in range(1, 9):
        race_id = f"20240501{day:02d}01"
        for position in (1, 2):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": f"h{position}",
                    "race_date": f"2024-01-{day:02d}",
                    "split": "development",
                    "model_finish_position": position,
                    "course_type": "芝",
                    "distance": 1600,
                    "race_class": "3歳未勝利",
                    "field_size": 2,
                    "pred_binary_raw": 0.8 if position == 1 else 0.2,
                    "score_lambdarank": -float(position),
                    "prob_binary_logit_softmax_temperature_2023": (
                        0.8 if position == 1 else 0.2
                    ),
                    "prob_lambdarank_softmax_temperature_2023": (
                        0.8 if position == 1 else 0.2
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_identical_ablation_has_zero_paired_interval() -> None:
    frame = _frame()
    result = compare_ablation_predictions(frame, frame, n_resamples=20, seed=4)

    for comparison in (
        "candidate_binary_vs_baseline_binary",
        "candidate_lambdarank_vs_baseline_lambdarank",
    ):
        for interval in result["bootstrap"]["paired"][comparison].values():
            assert interval["lower"] == 0.0
            assert interval["upper"] == 0.0
    assert result["scope"]["retrospective_2025_used"] is False
