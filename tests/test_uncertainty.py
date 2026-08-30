from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_pred.uncertainty import (
    PRIMARY_METRICS,
    ModelSpec,
    development_race_metric_table,
    paired_block_bootstrap,
    run_uncertainty_analysis,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for race_id, date, winner in (
        ("202405010101", "2024-01-06", 1),
        ("202406010101", "2024-01-06", 2),
        ("202405020101", "2024-01-07", 1),
    ):
        for horse in (1, 2):
            good = 0.8 if horse == winner else 0.2
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": date,
                    "split": "development",
                    "model_finish_position": 1 if horse == winner else 2,
                    "course_type": "芝",
                    "distance": 1600,
                    "race_class": "3歳未勝利",
                    "field_size": 2,
                    "prob_a": good,
                    "score_a": good,
                    "prob_b": 0.5,
                    "score_b": 0.5,
                }
            )
    return pd.DataFrame(rows)


SPECS = {
    "a": ModelSpec("prob_a", "score_a"),
    "b": ModelSpec("prob_b", "score_b"),
}


def test_development_metric_table_is_2024_only_and_ignores_retrospective() -> None:
    base = _predictions()
    retrospective = base.iloc[:2].copy()
    retrospective["race_id"] = "202505010101"
    retrospective["race_date"] = "2025-01-01"
    retrospective["split"] = "retrospective_test"
    retrospective["prob_a"] = [1.0, 0.0]
    actual = development_race_metric_table(
        pd.concat([base, retrospective], ignore_index=True), model_specs=SPECS
    )
    expected = development_race_metric_table(base, model_specs=SPECS)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["race_id"].str.startswith("2024").all()


def test_development_metric_table_rejects_mislabeled_2025() -> None:
    template = _predictions().iloc[:2]
    frames = []
    for day in range(1, 9):
        race = template.copy()
        race["race_id"] = f"20240501{day:02d}01"
        race["race_date"] = f"2024-01-{day:02d}"
        frames.append(race)
    frame = pd.concat(frames, ignore_index=True)
    frame.loc[:1, "race_id"] = "202505010101"
    frame.loc[:1, "race_date"] = "2025-01-01"
    with pytest.raises(ValueError, match="outside calendar year 2024"):
        development_race_metric_table(frame, model_specs=SPECS)


def test_metric_table_matches_expected_per_race_values() -> None:
    result = development_race_metric_table(_predictions(), model_specs=SPECS)
    good = result.loc[result["model"].eq("a")]
    assert good["ndcg_at_3"].eq(1.0).all()
    assert good["top_1_winner_mass"].eq(1.0).all()
    assert np.allclose(good["race_log_loss"], 0.2231435513142097)
    assert np.allclose(good["race_brier"], 0.08)


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_models() -> None:
    metrics = development_race_metric_table(_predictions(), model_specs=SPECS)
    left = paired_block_bootstrap(
        metrics,
        comparisons=(("b", "b"),),
        n_resamples=100,
        seed=17,
        block_length_dates=1,
    )
    right = paired_block_bootstrap(
        metrics,
        comparisons=(("b", "b"),),
        n_resamples=100,
        seed=17,
        block_length_dates=1,
    )
    assert left == right
    for metric in PRIMARY_METRICS:
        interval = left["paired"]["b_vs_b"][metric]
        assert interval["lower"] == 0.0
        assert interval["upper"] == 0.0


def test_date_bootstrap_uses_race_weighted_ratio() -> None:
    metrics = development_race_metric_table(_predictions(), model_specs=SPECS)
    result = paired_block_bootstrap(
        metrics,
        comparisons=(("a", "b"),),
        n_resamples=50,
        seed=1,
        block_length_dates=1,
    )
    assert result["race_count"] == 3
    assert result["date_count"] == 2
    assert result["marginal"]["a"]["top_1_winner_mass"]["point"] == 1.0


@pytest.mark.parametrize("block_length", [0, 3])
def test_bootstrap_rejects_invalid_block_length(block_length: int) -> None:
    metrics = development_race_metric_table(_predictions(), model_specs=SPECS)
    with pytest.raises(ValueError, match="block_length_dates"):
        paired_block_bootstrap(metrics, n_resamples=10, block_length_dates=block_length)


def test_uncertainty_artifact_excludes_2025(tmp_path) -> None:
    source = tmp_path / "predictions.csv"
    output = tmp_path / "uncertainty"
    template = _predictions().iloc[:2]
    frames = []
    for day in range(1, 9):
        race = template.copy()
        race["race_id"] = f"20240502{day:02d}01"
        race["race_date"] = f"2024-02-{day:02d}"
        frames.append(race)
    frame = pd.concat(frames, ignore_index=True)
    frame["prob_uniform"] = frame["prob_b"]
    frame["prob_history_rate"] = frame["prob_b"]
    frame["pred_binary_raw"] = frame["score_a"]
    frame["prob_binary_logit_softmax_temperature_2023"] = frame["prob_a"]
    frame["score_lambdarank"] = frame["score_a"]
    frame["prob_lambdarank_softmax_temperature_2023"] = frame["prob_a"]
    frame.to_csv(source, index=False)

    result = run_uncertainty_analysis(source, output, n_resamples=10, seed=9)

    assert result["scope"]["retrospective_test_used"] is False
    assert result["scope"]["date_end"].startswith("2024")
    assert (output / "uncertainty.json").is_file()
    assert (output / "artifact_manifest.json").is_file()
