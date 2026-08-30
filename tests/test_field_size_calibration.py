from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.artifacts import write_json
from horse_pred.field_size_calibration import (
    field_size_bands,
    isolate_calibration_and_development,
    run_field_size_calibration_experiment,
    validate_field_size_calibration_config,
)
from horse_pred.modeling import fit_temperature, probability_logits
from horse_pred.pipeline import PROBABILITY_EPSILON


def _config() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test_field_size_temperature",
        "parent_experiment_id": "baseline",
        "hypothesis": "test hypothesis",
        "seed": 7,
        "field_size_bands": {
            "labels": ["small", "medium", "large", "very_large"],
            "upper_bounds": [9, 13, 16],
        },
        "minimum_calibration_races_per_band": 1,
        "bootstrap": {"scheme": "moving_date_block", "block_length_dates": 4, "resamples": 20},
    }


def _predictions() -> pd.DataFrame:
    rows = []
    for split, year, date_count in (
        ("calibration", 2023, 4),
        ("development", 2024, 4),
        ("retrospective_test", 2025, 1),
    ):
        for day in range(1, date_count + 1):
            for race_number, field_size in enumerate((2, 10), start=1):
                race_id = f"{year}0501{day:02d}{race_number:02d}"
                for position in range(1, field_size + 1):
                    winner = position == 1
                    binary_raw = 0.7 if winner else 0.3 / (field_size - 1)
                    rank_score = -float(position)
                    rows.append(
                        {
                            "race_id": race_id,
                            "race_date": f"{year}-01-{day:02d}",
                            "horse_id": f"h{position}",
                            "split": split,
                            "course_type": "芝",
                            "distance": 1600,
                            "race_class": "3歳未勝利",
                            "field_size": field_size,
                            "model_finish_position": position,
                            "pred_binary_raw": binary_raw,
                            "score_lambdarank": rank_score,
                            "prob_binary_logit_softmax_temperature_2023": (
                                0.7 if winner else 0.3 / (field_size - 1)
                            ),
                            "prob_lambdarank_softmax_temperature_2023": (
                                0.7 if winner else 0.3 / (field_size - 1)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_config_and_field_size_band_boundaries() -> None:
    config = _config()
    validate_field_size_calibration_config(config)
    actual = field_size_bands(pd.Series([1, 9, 10, 13, 14, 16, 17, 18]), config)
    assert actual.tolist() == [
        "small",
        "small",
        "medium",
        "medium",
        "large",
        "large",
        "very_large",
        "very_large",
    ]


def test_isolation_drops_retrospective_before_experiment() -> None:
    selected, scope = isolate_calibration_and_development(_predictions())
    assert selected["race_date"].dt.year.max() == 2024
    assert not selected["split"].eq("retrospective_test").any()
    assert scope["retrospective_2025_plus_rows_in_source"] > 0
    assert scope["retrospective_2025_plus_rows_used"] == 0


def test_isolation_rejects_mislabeled_2025() -> None:
    frame = _predictions()
    frame.loc[frame["split"].eq("retrospective_test"), "split"] = "development"
    with pytest.raises(ValueError, match="must be labeled retrospective_test"):
        isolate_calibration_and_development(frame)


def test_isolation_rejects_field_size_that_disagrees_with_group_rows() -> None:
    frame = _predictions()
    race_id = frame.loc[0, "race_id"]
    frame.loc[frame["race_id"].eq(race_id), "field_size"] = 3
    with pytest.raises(ValueError, match="equal the number of scored runners"):
        isolate_calibration_and_development(frame)


def test_run_writes_2024_only_probability_artifact_without_reranking(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    predictions = _predictions()
    predictions.to_csv(baseline / "predictions.csv.gz", index=False, compression="gzip")
    calibration = predictions.loc[predictions["split"].eq("calibration")].copy()
    binary_temperature = fit_temperature(
        probability_logits(
            calibration["pred_binary_raw"], epsilon=PROBABILITY_EPSILON
        ),
        calibration["race_id"],
        calibration["model_finish_position"],
    ).temperature
    rank_temperature = fit_temperature(
        calibration["score_lambdarank"],
        calibration["race_id"],
        calibration["model_finish_position"],
    ).temperature
    write_json(
        baseline / "metrics.json",
        {
            "data": {"fingerprint": "raw-sha"},
            "models": {
                "binary": {"temperature": binary_temperature},
                "lambdarank": {"temperature": rank_temperature},
            },
        },
    )
    config = _config()
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    output = tmp_path / "result"

    result = run_field_size_calibration_experiment(
        repo_root=Path.cwd(),
        baseline_dir=baseline,
        config_path=config_path,
        output_dir=output,
    )

    saved = pd.read_csv(output / "predictions_2024.csv.gz")
    assert saved["race_date"].str.startswith("2024").all()
    assert np.allclose(
        saved.groupby("race_id")["prob_binary_field_size_temperature_2023"].sum(), 1.0
    )
    for model in ("binary", "lambdarank"):
        baseline_ranking = result["development"][model]["baseline_global"]["ranking"]
        candidate_ranking = result["development"][model]["candidate_field_size"]["ranking"]
        assert baseline_ranking == candidate_ranking
    assert result["scope"]["retrospective_used"] is False
    assert result["scope"]["retrospective_2025_plus_rows_used"] == 0
    for model in ("binary", "lambdarank"):
        reproduction = result["calibrators"][model][
            "baseline_global_temperature_reproduction"
        ]
        assert reproduction["within_tolerance"] is True
        assert reproduction["absolute_difference"] <= reproduction["absolute_tolerance"]
    assert result["paired_by_field_size_band"]["bands"]["small"]["status"] == "available"
    assert result["paired_by_field_size_band"]["bands"]["medium"]["status"] == "available"
    assert (
        result["paired_by_field_size_band"]["bands"]["large"]["status"]
        == "insufficient_support"
    )
    assert (output / "race_metrics_2024.csv.gz").is_file()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert any(item["path"] == "metrics.json" for item in manifest["files"])
    assert not list(tmp_path.glob(".result.tmp-*"))
