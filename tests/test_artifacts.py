import json

import numpy as np

from horse_pred.artifacts import write_json


def test_write_json_emits_strict_json_for_nonfinite_values(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    write_json(
        target,
        {
            "positive_infinity": float("inf"),
            "negative_infinity": np.float64("-inf"),
            "not_a_number": float("nan"),
            "finite": np.float32(1.25),
        },
    )
    text = target.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    assert json.loads(text) == {
        "finite": 1.25,
        "negative_infinity": None,
        "not_a_number": None,
        "positive_infinity": None,
    }
