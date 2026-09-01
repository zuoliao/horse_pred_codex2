from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from horse_pred.program_audit import run_historical_oracle_diagnostic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    predictions = tmp_path / "predictions.csv"
    market = tmp_path / "market.csv"
    prereg = tmp_path / "prereg.json"
    pd.DataFrame(
        {
            "race_id": [1, 1, 2, 2],
            "race_date": ["2024-01-01"] * 2 + ["2024-01-02"] * 2,
            "horse_id": [11, 12, 21, 22],
            "horse_number": [1, 2, 1, 2],
            "split": ["development"] * 4,
            "model_finish_position": [1, 2, 2, 1],
            "probability": [0.7, 0.3, 0.4, 0.6],
        }
    ).to_csv(predictions, index=False)
    pd.DataFrame(
        {
            "race_id": [1, 1, 2, 2],
            "race_date": ["2024-01-01"] * 2 + ["2024-01-02"] * 2,
            "horse_id": [11, 12, 21, 22],
            "final_win_odds": [2.0, 4.0, 3.0, 2.0],
        }
    ).to_csv(market, index=False)
    prereg.write_text(
        json.dumps(
            {
                "diagnostic_id": "fixture",
                "selection_or_adoption_use": False,
                "profit_or_roi_use": False,
                "inputs": {
                    "fundamental_predictions": str(predictions.relative_to(tmp_path)),
                    "fundamental_predictions_sha256": _sha(predictions),
                    "fundamental_probability_column": "probability",
                    "fundamental_feature_count": 2,
                    "fundamental_feature_columns_sha256": "features",
                    "fundamental_model_sha256": "model",
                    "market_oracle": str(market.relative_to(tmp_path)),
                    "market_oracle_sha256": _sha(market),
                    "join_keys": ["race_id", "horse_id"],
                },
                "population": {
                    "year": 2024,
                    "prediction_split": "development",
                },
                "metrics": {
                    "uncertainty": {
                        "resamples": 20,
                        "seed": 7,
                        "block_length_dates": 2,
                    }
                },
                "interpretation": {"always": "descriptive only"},
            }
        ),
        encoding="utf-8",
    )
    return predictions, market, prereg


def test_historical_oracle_uses_fixed_three_methods(tmp_path: Path) -> None:
    _, _, prereg = _fixture(tmp_path)
    result = run_historical_oracle_diagnostic(
        repo_root=tmp_path,
        preregistration_path=prereg,
        output_path=tmp_path / "summary.json",
    )
    assert result["scope"]["race_count"] == 2
    assert set(result["methods"]) == {
        "fundamental_only",
        "final_market_only",
        "combined_fixed_50_50_log",
    }
    assert result["scope"]["selection_or_adoption_use"] is False
    assert result["scope"]["profit_or_roi_use"] is False
    assert result["next_model_executed"] is False


def test_historical_oracle_rejects_changed_input(tmp_path: Path) -> None:
    predictions, _, prereg = _fixture(tmp_path)
    predictions.write_text(predictions.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_historical_oracle_diagnostic(
            repo_root=tmp_path,
            preregistration_path=prereg,
            output_path=tmp_path / "summary.json",
        )
