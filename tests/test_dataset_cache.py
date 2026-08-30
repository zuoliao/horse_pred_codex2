from __future__ import annotations

import pandas as pd

from horse_pred.dataset_cache import read_model_frame_cache, write_model_frame_cache
from horse_pred.features import FeatureDataset


def test_model_frame_cache_round_trip(tmp_path) -> None:
    frame = pd.DataFrame({"race_id": ["r1", "r1"], "feature__x": [1.0, 2.0]})
    dataset = FeatureDataset(frame, ("feature__x",), {"test": ("feature__x",)})
    path = tmp_path / "cache.pkl"

    write_model_frame_cache(path, frame, dataset, data_fingerprint="abc")
    actual, metadata = read_model_frame_cache(path)

    pd.testing.assert_frame_equal(actual, frame)
    assert metadata["data_fingerprint"] == "abc"
    assert metadata["feature_columns"] == ["feature__x"]
