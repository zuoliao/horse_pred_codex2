import pandas as pd
import pytest

from horse_pred.config import assign_splits, configured_intervals


def test_assign_splits_uses_closed_non_overlapping_intervals() -> None:
    config = {
        "warmup": {"start": "2013-01-01", "end": "2013-12-31"},
        "train": {"start": "2014-01-01", "end": "2021-12-31"},
        "model_validation": {"start": "2022-01-01", "end": "2022-12-31"},
        "calibration": {"start": "2023-01-01", "end": "2023-12-31"},
        "development": {"start": "2024-01-01", "end": "2024-12-31"},
        "retrospective_test": {"start": "2025-01-01", "end": "2025-12-31"},
        "prospective_final": {"start": "2026-01-01", "end": None},
    }
    dates = pd.Series(["2013-12-31", "2014-01-01", "2023-05-01", "2026-01-01", None])
    assert assign_splits(dates, config).tolist() == [
        "warmup",
        "train",
        "calibration",
        "prospective_final",
        pd.NA,
    ]


def test_configured_intervals_rejects_overlap() -> None:
    config = {
        "warmup": {"start": "2013-01-01", "end": "2013-12-31"},
        "train": {"start": "2013-12-31", "end": "2021-12-31"},
        "model_validation": {"start": "2022-01-01", "end": "2022-12-31"},
        "calibration": {"start": "2023-01-01", "end": "2023-12-31"},
        "development": {"start": "2024-01-01", "end": "2024-12-31"},
        "retrospective_test": {"start": "2025-01-01", "end": "2025-12-31"},
        "prospective_final": {"start": "2026-01-01", "end": None},
    }
    with pytest.raises(ValueError, match="overlap"):
        configured_intervals(config)
