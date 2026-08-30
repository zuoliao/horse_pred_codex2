from __future__ import annotations

import json

import numpy as np
import pandas as pd

from horse_pred.cache_control import compare_surface_elo_cache_control


def _write_cache(path, frame: pd.DataFrame, features: list[str]) -> None:
    frame.to_pickle(path)
    path.with_suffix(".pkl.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "pandas_pickle",
                "data_fingerprint": "raw-sha",
                "row_count": len(frame),
                "race_count": frame["race_id"].nunique(),
                "feature_columns": features,
            }
        ),
        encoding="utf-8",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "split": ["development"] * 3,
            "horse_number": [1, 2, 1],
            "feature_a": np.array([1.0, np.nan, 3.0], dtype="float32"),
            "feature_b": np.array([4.0, 5.0, 6.0], dtype="float32"),
        }
    )


def test_surface_cache_control_passes_for_exact_control_values(tmp_path) -> None:
    baseline = _frame()
    candidate = baseline.copy()
    candidate["surface_rating__horse_elo_pre"] = 1500.0
    candidate["surface_rating__horse_minus_field_mean_elo"] = 0.0
    candidate["surface_rating__horse_elo_percentile"] = 0.5
    baseline_path = tmp_path / "baseline.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["feature_a", "feature_b"])
    _write_cache(
        candidate_path,
        candidate,
        [
            "feature_a",
            "feature_b",
            "surface_rating__horse_elo_pre",
            "surface_rating__horse_minus_field_mean_elo",
            "surface_rating__horse_elo_percentile",
        ],
    )
    output = tmp_path / "control.json"

    result = compare_surface_elo_cache_control(
        baseline_path,
        candidate_path,
        output_path=output,
        chunk_size=2,
        expected_baseline_feature_count=2,
    )

    assert result["passed"]
    assert result["mismatch_count"] == 0
    assert result["feature_values"]["max_abs_diff"] == 0.0
    assert result["feature_values"]["nan_position_mismatch_count"] == 0
    assert result["feature_counts"]["candidate_surface_rating"] == 3
    assert len(result["cache_sha256"]["baseline"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_surface_cache_control_reports_identity_value_and_nan_mismatches(
    tmp_path,
) -> None:
    baseline = _frame()
    candidate = baseline.copy()
    candidate.loc[0, "horse_id"] = "other"
    candidate.loc[0, "feature_a"] = 1.5
    candidate.loc[1, "feature_a"] = 2.0
    for column in (
        "surface_rating__horse_elo_pre",
        "surface_rating__horse_minus_field_mean_elo",
        "surface_rating__horse_elo_percentile",
    ):
        candidate[column] = 0.0
    baseline_path = tmp_path / "baseline.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["feature_a", "feature_b"])
    _write_cache(
        candidate_path,
        candidate,
        [
            "feature_a",
            "feature_b",
            "surface_rating__horse_elo_pre",
            "surface_rating__horse_minus_field_mean_elo",
            "surface_rating__horse_elo_percentile",
        ],
    )

    result = compare_surface_elo_cache_control(
        baseline_path,
        candidate_path,
        chunk_size=2,
        expected_baseline_feature_count=2,
    )

    assert not result["passed"]
    assert result["runner_identity"]["mismatch_cell_count"] == 1
    assert result["feature_schema"]["control_positional_mismatch_count"] == 0
    assert result["feature_values"]["mismatch_count"] == 2
    assert result["feature_values"]["nan_position_mismatch_count"] == 1
    assert result["feature_values"]["max_abs_diff"] == 0.5
    assert result["mismatch_count"] >= 3


def test_surface_cache_control_rejects_changed_control_feature_order(tmp_path) -> None:
    baseline = _frame()
    candidate = baseline.copy()
    for column in (
        "surface_rating__horse_elo_pre",
        "surface_rating__horse_minus_field_mean_elo",
        "surface_rating__horse_elo_percentile",
    ):
        candidate[column] = 0.0
    baseline_path = tmp_path / "baseline.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["feature_a", "feature_b"])
    _write_cache(
        candidate_path,
        candidate,
        [
            "feature_b",
            "feature_a",
            "surface_rating__horse_elo_pre",
            "surface_rating__horse_minus_field_mean_elo",
            "surface_rating__horse_elo_percentile",
        ],
    )

    result = compare_surface_elo_cache_control(
        baseline_path,
        candidate_path,
        expected_baseline_feature_count=2,
    )

    assert not result["passed"]
    assert result["feature_schema"]["control_positional_mismatch_count"] == 2
    # Value comparison is intentionally withheld when the column contract differs.
    assert result["feature_values"]["max_abs_diff"] is None
