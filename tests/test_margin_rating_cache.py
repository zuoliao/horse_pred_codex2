from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from horse_pred.features import semantic_feature_groups_v2
from horse_pred.margin_rating_cache import (
    MARGIN_RATING_COLUMN,
    build_margin_rating_score_augmented_cache,
)


def test_margin_rating_cache_preserves_old_feature_and_blocks_2025(tmp_path) -> None:
    baseline = tmp_path / "baseline.pkl"
    output = tmp_path / "candidate.pkl"
    frame = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101", "202501010101"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
            "split": ["development", "development", "retrospective_test"],
            "model_finish_position": [1, 2, 1],
            "course_type": ["芝", "芝", "芝"],
            "distance": [1600, 1600, 1600],
            "race_class": ["open", "open", "open"],
            "field_size": [2, 2, 1],
            "context__distance": np.array([1600.0, np.nan, 1600.0], dtype="float32"),
        }
    )
    frame.to_pickle(baseline)
    baseline.with_suffix(".pkl.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_count": 3,
                "race_count": 2,
                "feature_columns": ["context__distance"],
                "feature_groups_v1": {"race_context": ["context__distance"]},
            }
        ),
        encoding="utf-8",
    )
    history = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101"],
            "horse_id": ["h1", "h2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "finish_position": [1.0, 2.0],
            "source_position": [0, 1],
            "surface_key": ["turf", "turf"],
            "global_state_pre": [1500.0, 1500.0],
            "condition_state_pre": [1500.0, 1500.0],
            "modular_rating__score_pre": [0.1, -0.1],
            "modular_rating__raw_win_probability_pre": [0.55, 0.45],
            "modular_rating__global_starts_pre": [1.0, 1.0],
            "modular_rating__condition_starts_pre": [1.0, 1.0],
            "modular_rating__uncertainty_proxy_pre": [0.5, 0.5],
        }
    )
    config = {
        "rating_spec": {"family": "pairwise_elo"},
        "candidate_column": MARGIN_RATING_COLUMN,
    }

    result = build_margin_rating_score_augmented_cache(
        baseline, history, output, config=config
    )
    augmented = pd.read_pickle(output)

    assert result["old_feature_exact"] is True
    assert result["candidate_feature_count"] == 2
    assert augmented["context__distance"].equals(frame["context__distance"])
    assert augmented.loc[:1, MARGIN_RATING_COLUMN].tolist() == pytest.approx([0.1, -0.1])
    assert np.isnan(augmented.loc[2, MARGIN_RATING_COLUMN])
    metadata = json.loads(output.with_suffix(".pkl.meta.json").read_text())
    assert metadata["feature_columns"] == ["context__distance", MARGIN_RATING_COLUMN]


def test_margin_rating_has_a_separate_semantic_group() -> None:
    groups = semantic_feature_groups_v2(
        ("context__distance", "race_content__x", MARGIN_RATING_COLUMN)
    )
    assert groups["margin_rating"] == (MARGIN_RATING_COLUMN,)
