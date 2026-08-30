from __future__ import annotations

import json

import numpy as np
import pandas as pd

from horse_pred.rating_integration import build_rating_augmented_cache


def test_rating_cache_join_covers_pre2025_and_leaves_2025_unread(tmp_path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
            "split": ["development", "development", "retrospective_test"],
            "horse_number": [1, 2, 1],
            "feature_a": np.array([1.0, 2.0, 3.0], dtype="float32"),
        }
    )
    baseline_path = tmp_path / "baseline.pkl"
    baseline.to_pickle(baseline_path)
    baseline_path.with_suffix(".pkl.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "pandas_pickle",
                "data_fingerprint": "raw",
                "row_count": 3,
                "race_count": 2,
                "feature_columns": ["feature_a"],
            }
        ),
        encoding="utf-8",
    )
    columns = [
        "modular_rating__score_pre",
        "modular_rating__raw_win_probability_pre",
        "modular_rating__global_starts_pre",
        "modular_rating__condition_starts_pre",
        "modular_rating__uncertainty_proxy_pre",
    ]
    ratings = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["h1", "h2"],
            "split": ["development", "development"],
            **{column: [0.1, 0.2] for column in columns},
        }
    )
    ratings_path = tmp_path / "ratings.pkl"
    ratings.to_pickle(ratings_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text('{"schema_version": 1}', encoding="utf-8")
    output = tmp_path / "augmented.pkl"

    result = build_rating_augmented_cache(
        baseline_path, ratings_path, spec_path, output
    )
    augmented = pd.read_pickle(output)

    assert result["candidate_feature_count"] == 6
    assert augmented.loc[augmented["split"].eq("development"), columns].notna().all().all()
    assert augmented.loc[
        augmented["split"].eq("retrospective_test"), columns
    ].isna().all().all()
