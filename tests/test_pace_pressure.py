from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_pace_pressure_cache_control
from horse_pred.cached_experiment import resolve_semantic_feature_selection
from horse_pred.cli import parser
from horse_pred.config import canonical_json_hash
from horse_pred.features import model_feature_allowlist, semantic_feature_groups_v2
from horse_pred.pace_pressure import (
    PACE_PRESSURE_COLUMN,
    PACE_PRESSURE_TRANSFORMATION_HASH,
    PacePressureSpec,
    build_pace_pressure_cache,
    load_pace_pressure_config,
    rival_front_excess_sum,
)
from horse_pred.pace_recent import PACE_RECENT_COLUMN
from horse_pred.rolling_evaluation import validate_rolling_config


def _frame(values: list[float], *, race_id: str = "r1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [race_id] * len(values),
            "horse_id": [f"h{i}" for i in range(len(values))],
            "context__field_size_rows": [float(len(values))] * len(values),
            PACE_RECENT_COLUMN: values,
        }
    )


def _write_cache(path: Path, frame: pd.DataFrame, features: list[str]) -> None:
    frame.to_pickle(path)
    path.with_suffix(f"{path.suffix}.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "pandas_pickle",
                "data_fingerprint": "raw-sha",
                "row_count": len(frame),
                "race_count": int(frame["race_id"].nunique()),
                "feature_columns": features,
                "feature_groups_v1": {},
            }
        ),
        encoding="utf-8",
    )


def _registered_config() -> dict[str, object]:
    spec = PacePressureSpec()
    return {
        "candidate_column": PACE_PRESSURE_COLUMN,
        "transformation_hash": PACE_PRESSURE_TRANSFORMATION_HASH,
        "transformation": spec.transformation_dict(),
    }


def test_rival_pressure_excludes_target_and_distinguishes_missing() -> None:
    result = rival_front_excess_sum(_frame([1.0, 0.8, 0.4, np.nan]))
    assert result.tolist() == pytest.approx([0.3, 0.5, 0.8, 0.8])

    all_missing = rival_front_excess_sum(_frame([np.nan, np.nan]))
    assert all_missing.isna().all()
    target_only = rival_front_excess_sum(_frame([1.0, np.nan]))
    assert np.isnan(target_only.iloc[0])
    assert target_only.iloc[1] == pytest.approx(0.5)
    known_slow = rival_front_excess_sum(_frame([0.5, 0.4, np.nan]))
    assert known_slow.fillna(-1).tolist() == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("value", [-0.01, 1.01, np.inf, -np.inf])
def test_rival_pressure_rejects_invalid_source(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        rival_front_excess_sum(_frame([value, 0.5]))


def test_rival_pressure_is_key_stable_race_local_and_outcome_independent() -> None:
    left = pd.concat([_frame([1.0, 0.6, 0.2], race_id="r1"), _frame([0.9, 0.4], race_id="r2")])
    left["finish"] = [1, 2, 3, 1, 2]
    left["odds"] = [2.0, 3.0, 4.0, 5.0, 6.0]
    expected = left.assign(value=rival_front_excess_sum(left)).set_index(["race_id", "horse_id"])["value"]
    right = left.sample(frac=1, random_state=42).copy()
    right["finish"] = 99
    right["odds"] = 999.0
    actual = right.assign(value=rival_front_excess_sum(right)).set_index(["race_id", "horse_id"])["value"]
    pd.testing.assert_series_equal(expected.sort_index(), actual.sort_index())


def test_rival_pressure_rejects_duplicate_keys_and_field_size_mismatch() -> None:
    duplicate = _frame([1.0, 0.5])
    duplicate.loc[1, "horse_id"] = duplicate.loc[0, "horse_id"]
    with pytest.raises(ValueError, match="duplicate"):
        rival_front_excess_sum(duplicate)
    mismatch = _frame([1.0, 0.5])
    mismatch["context__field_size_rows"] = 3.0
    with pytest.raises(ValueError, match="field size"):
        rival_front_excess_sum(mismatch)


def test_cache_preserves_control_and_keeps_2025_missing(tmp_path: Path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101", "202501010101", "202501010101"],
            "horse_id": ["h1", "h2", "h3", "h4"],
            "race_date": pd.to_datetime(["2024-01-01"] * 2 + ["2025-01-01"] * 2),
            "split": ["development"] * 2 + ["retrospective_test"] * 2,
            "horse_number": [1, 2, 1, 2],
            "context__field_size_rows": np.array([2.0] * 4, dtype="float32"),
            PACE_RECENT_COLUMN: np.array([1.0, 0.5, np.nan, np.nan], dtype="float32"),
        }
    )
    features = ["context__field_size_rows", PACE_RECENT_COLUMN]
    input_path = tmp_path / "input.pkl"
    output_path = tmp_path / "output.pkl"
    _write_cache(input_path, baseline, features)

    result = build_pace_pressure_cache(
        input_path, output_path, config=_registered_config()
    )
    output = pd.read_pickle(output_path)
    assert output.loc[:, features].equals(baseline.loc[:, features])
    assert output.loc[:1, PACE_PRESSURE_COLUMN].tolist() == pytest.approx([0.0, 0.5])
    assert output.loc[2:, PACE_PRESSURE_COLUMN].isna().all()
    assert result["retrospective_2025_feature_nonmissing"] == 0
    control = compare_pace_pressure_cache_control(
        input_path, output_path, expected_baseline_feature_count=2
    )
    assert control["passed"]


def test_config_taxonomy_and_rolling_keep_control_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    config, spec = load_pace_pressure_config(root / "configs/features/pace_02_field_pressure.json")
    assert canonical_json_hash(spec.transformation_dict()) == PACE_PRESSURE_TRANSFORMATION_HASH
    assert config["transformation_hash"] == PACE_PRESSURE_TRANSFORMATION_HASH
    columns = (
        "context__distance",
        "race_content__time",
        PACE_RECENT_COLUMN,
        PACE_PRESSURE_COLUMN,
    )
    groups = semantic_feature_groups_v2(columns)
    assert groups["pace"] == (PACE_RECENT_COLUMN,)
    assert groups["pace_pressure"] == (PACE_PRESSURE_COLUMN,)
    frame = pd.DataFrame({column: [0.5] for column in columns} | {"通過順位": ["1-1"]})
    assert PACE_PRESSURE_COLUMN in model_feature_allowlist(frame)
    assert "通過順位" not in model_feature_allowlist(frame)

    control_cfg = json.loads((root / "configs/performance/pace_01_binary_candidate.json").read_text())
    candidate_cfg = json.loads((root / "configs/performance/pace_02_binary_candidate.json").read_text())
    control, _, _ = resolve_semantic_feature_selection(columns, control_cfg)
    candidate, _, _ = resolve_semantic_feature_selection(columns, candidate_cfg)
    assert PACE_PRESSURE_COLUMN not in control
    assert PACE_PRESSURE_COLUMN in candidate

    rolling = json.loads((root / "configs/evaluation/pace_02_rolling.json").read_text())
    validate_rolling_config(rolling)
    assert rolling["maximum_outcome_year"] == 2023


def test_cli_registers_pace_pressure_builder() -> None:
    args = parser().parse_args(
        [
            "build-pace-pressure-cache",
            "--input-cache",
            "pace.pkl",
            "--output",
            "pressure.pkl",
        ]
    )
    assert args.command == "build-pace-pressure-cache"
    assert args.config == Path("configs/features/pace_02_field_pressure.json")
