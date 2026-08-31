from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_speed_cache_control
from horse_pred.cached_experiment import feature_columns_checksum
from horse_pred.cli import parser
from horse_pred.config import canonical_json_hash
from horse_pred.features import model_feature_allowlist, semantic_feature_groups_v2
from horse_pred.rolling_evaluation import validate_rolling_config
from horse_pred.speed_figure import (
    SPEED_COLUMN,
    SPEED_TRANSFORMATION_HASH,
    SpeedFigureSpec,
    build_speed_augmented_cache,
    build_speed_history,
    condition_design_vector,
    load_speed_config,
)


def _runner(
    race: str,
    date: str,
    horse: str,
    finish: int,
    clock: str,
    *,
    venue: str = "05",
    race_class: str = "3歳未勝利",
    status: str = "finished",
) -> dict[str, object]:
    return {
        "race_id": race,
        "race_date": pd.Timestamp(date),
        "horse_id": horse,
        "venue_code": venue,
        "course_type": "芝",
        "distance": 1000,
        "ground_state": "良",
        "race_class": race_class,
        "status": status,
        "started": True,
        "finish_position": finish,
        "time_raw": clock,
    }


def _race(race: str, date: str, *, loser: str, loser_clock: str = "1:01.0") -> list[dict[str, object]]:
    return [
        _runner(race, date, f"winner-{race}", 1, "1:00.0"),
        _runner(race, date, loser, 2, loser_clock),
    ]


def _training_history(*, loser_clock: str = "1:20.0") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(510):
        rows.extend(_race(f"warm-{index:03d}", "2013-01-01", loser=f"warm-loser-{index:03d}"))
    rows.extend(_race("observe", "2013-01-02", loser="horse-x", loser_clock=loser_clock))
    rows.extend(_race("target", "2013-01-03", loser="horse-x"))
    return pd.DataFrame(rows)


def _value(history: pd.DataFrame, race: str, horse: str) -> float:
    return float(history.loc[history["race_id"].eq(race) & history["horse_id"].eq(horse), SPEED_COLUMN].iloc[0])


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


def _config() -> dict[str, object]:
    spec = SpeedFigureSpec()
    return {
        "candidate_column": SPEED_COLUMN,
        "transformation_hash": SPEED_TRANSFORMATION_HASH,
        "transformation": spec.transformation_dict(),
    }


def test_condition_design_is_frozen_51d_and_rejects_unknown() -> None:
    race = pd.DataFrame(_race("r1", "2014-01-01", loser="h2"))
    vector = condition_design_vector(race)
    assert vector is not None
    assert vector.shape == (51,)
    assert vector[0] == 1.0
    assert vector.sum() == 2.0
    race["ground_state"] = "unknown"
    assert condition_design_vector(race) is None


def test_prequential_cold_gate_clip_and_target_outcome_invariance() -> None:
    frame = _training_history()
    history = build_speed_history(frame)
    assert np.isnan(_value(history, "observe", "horse-x"))
    assert _value(history, "target", "horse-x") == pytest.approx(-5.0)

    changed = frame.copy()
    changed.loc[changed["race_id"].eq("target") & changed["horse_id"].eq("horse-x"), "time_raw"] = "1:40.0"
    changed_history = build_speed_history(changed)
    assert _value(changed_history, "target", "horse-x") == pytest.approx(-5.0)


def test_future_append_and_same_date_order_do_not_change_past() -> None:
    frame = _training_history(loser_clock="1:02.0")
    baseline = build_speed_history(frame).sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    appended = pd.concat(
        [frame, pd.DataFrame(_race("future", "2013-01-04", loser="future-horse"))],
        ignore_index=True,
    )
    with_future = build_speed_history(appended)
    with_future = with_future.loc[~with_future["race_id"].eq("future")]
    with_future = with_future.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, with_future)

    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_history = build_speed_history(shuffled).sort_values(["race_id", "horse_id"])
    shuffled_history = shuffled_history.reset_index(drop=True)
    pd.testing.assert_series_equal(baseline[SPEED_COLUMN], shuffled_history[SPEED_COLUMN], check_names=False)


def test_same_date_result_cannot_change_same_date_features() -> None:
    frame = _training_history(loser_clock="1:02.0")
    extra = pd.DataFrame(_race("same-day", "2013-01-02", loser="same-horse"))
    frame = pd.concat([frame, extra], ignore_index=True)
    baseline = build_speed_history(frame)
    changed = frame.copy()
    changed.loc[changed["race_id"].eq("observe"), "time_raw"] = ["0:59.0", "1:30.0"]
    changed_history = build_speed_history(changed)
    left = baseline.loc[baseline["race_id"].eq("same-day"), SPEED_COLUMN].reset_index(drop=True)
    right = changed_history.loc[changed_history["race_id"].eq("same-day"), SPEED_COLUMN].reset_index(drop=True)
    pd.testing.assert_series_equal(left, right)


def test_2025_is_rejected() -> None:
    frame = pd.DataFrame(_race("future", "2025-01-01", loser="h2"))
    assert build_speed_history(frame).empty
    with pytest.raises(ValueError, match="2025"):
        build_speed_history(frame, through_year=2025)


def test_cache_control_and_2025_firewall(tmp_path: Path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101", "202501010101"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
            "split": ["development", "development", "retrospective_test"],
            "horse_number": [1, 2, 1],
            "context__distance": np.array([1600.0, 1600.0, 1200.0], dtype="float32"),
            "pace__existing": np.array([0.2, 0.3, np.nan], dtype="float32"),
        }
    )
    features = ["context__distance", "pace__existing"]
    baseline_path = tmp_path / "base.pkl"
    output_path = tmp_path / "out.pkl"
    _write_cache(baseline_path, baseline, features)
    history = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101"],
            "horse_id": ["h1", "h2"],
            SPEED_COLUMN: [np.nan, -0.5],
            "_speed_history_row_present": [True, True],
        }
    )
    result = build_speed_augmented_cache(baseline_path, history, output_path, config=_config())
    output = pd.read_pickle(output_path)
    assert output.loc[:, features].equals(baseline.loc[:, features])
    assert np.isnan(output.loc[2, SPEED_COLUMN])
    assert result["retrospective_2025_feature_nonmissing"] == 0
    assert compare_speed_cache_control(baseline_path, output_path, expected_baseline_feature_count=2)["passed"]


def test_config_taxonomy_hashes_rolling_and_cli_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    config, spec = load_speed_config(root / "configs/features/speed_01_condition_adjusted.json")
    assert canonical_json_hash(spec.transformation_dict()) == SPEED_TRANSFORMATION_HASH
    assert config["transformation_hash"] == SPEED_TRANSFORMATION_HASH
    groups = semantic_feature_groups_v2(("context__distance", SPEED_COLUMN))
    assert groups["speed"] == (SPEED_COLUMN,)
    frame = pd.DataFrame({"context__distance": [1600.0], SPEED_COLUMN: [-0.5], "タイム": ["1:34.0"]})
    assert SPEED_COLUMN in model_feature_allowlist(frame)
    assert "タイム" not in model_feature_allowlist(frame)
    rolling = json.loads((root / "configs/evaluation/speed_01_rolling.json").read_text())
    validate_rolling_config(rolling)
    assert rolling["maximum_outcome_year"] == 2023

    schema = json.loads((root / "artifacts/pace_01_rolling_20260831/feature_schema.json").read_text())["methods"]
    assert (
        feature_columns_checksum(schema["binary_candidate"]["feature_columns"] + [SPEED_COLUMN])
        == (rolling["methods"]["binary_candidate"]["expected_columns_sha256"])
    )
    assert (
        feature_columns_checksum(schema["lambdarank_candidate"]["feature_columns"] + [SPEED_COLUMN])
        == rolling["methods"]["lambdarank_candidate"]["expected_columns_sha256"]
    )

    args = parser().parse_args(
        [
            "build-speed-cache",
            "--raw-path",
            "raw.csv",
            "--baseline-cache",
            "base.pkl",
            "--output",
            "speed.pkl",
        ]
    )
    assert args.command == "build-speed-cache"
