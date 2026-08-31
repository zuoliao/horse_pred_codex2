from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_pace_final_cache_control
from horse_pred.cli import parser
from horse_pred.config import canonical_json_hash
from horse_pred.features import model_feature_allowlist, semantic_feature_groups_v2
from horse_pred.pace_final import (
    PACE_FINAL_COLUMN,
    PACE_FINAL_TRANSFORMATION_HASH,
    PaceFinalSpec,
    build_pace_final_augmented_cache,
    build_pace_final_cache_from_raw,
    build_pace_final_history,
    final_position_percentiles,
    load_pace_final_config,
)
from horse_pred.rolling_evaluation import validate_rolling_config


def _row(race: str, date: str, horse: str, passing: object, *, race_class: str = "3歳未勝利") -> dict[str, object]:
    return {
        "race_id": race,
        "race_date": pd.Timestamp(date),
        "horse_id": horse,
        "started": True,
        "passing_order_raw": passing,
        "course_type": "芝",
        "race_class": race_class,
    }


def _value(history: pd.DataFrame, race: str, horse: str) -> float:
    return float(
        history.loc[
            history["race_id"].eq(race) & history["horse_id"].eq(horse),
            PACE_FINAL_COLUMN,
        ].iloc[0]
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


def _config() -> dict[str, object]:
    spec = PaceFinalSpec()
    return {
        "candidate_column": PACE_FINAL_COLUMN,
        "transformation_hash": PACE_FINAL_TRANSFORMATION_HASH,
        "transformation": spec.transformation_dict(),
    }


def test_final_percentile_uses_last_token_and_average_ties() -> None:
    race = pd.DataFrame(
        {
            "started": [True, True, True, True, False],
            "passing_order_raw": ["4-1", "1-3", "2-3", "1", "3-2"],
        }
    )
    result = final_position_percentiles(race)
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(0.25)
    assert result.iloc[2] == pytest.approx(0.25)
    assert result.iloc[3:].isna().all()


def test_history_uses_prior_dates_batches_same_day_and_skips_one_segment() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", "4-1"),
        _row("r1", "2023-01-01", "h2", "1-2"),
        _row("r2", "2023-01-01", "h1", "1-2"),
        _row("r2", "2023-01-01", "h3", "2-1"),
        _row("r3", "2023-01-02", "h1", "1"),
        _row("r3", "2023-01-02", "h4", "2"),
        _row("r4", "2023-01-03", "h1", "2-2"),
        _row("r4", "2023-01-03", "h5", "1-1"),
    ]
    history = build_pace_final_history(pd.DataFrame(rows))
    assert np.isnan(_value(history, "r1", "h1"))
    assert np.isnan(_value(history, "r2", "h1"))
    assert _value(history, "r3", "h1") == pytest.approx(0.5)
    assert _value(history, "r4", "h1") == pytest.approx(0.5)


def test_history_ignores_obstacle_and_future() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", "1-1", race_class="障害4歳以上未勝利"),
        _row("r1", "2023-01-01", "h2", "2-2", race_class="障害4歳以上未勝利"),
        _row("r2", "2023-01-02", "h1", "1-1"),
        _row("r2", "2023-01-02", "h3", "2-2"),
        _row("r3", "2025-01-01", "h1", "1-1"),
        _row("r3", "2025-01-01", "h4", "2-2"),
    ]
    history = build_pace_final_history(pd.DataFrame(rows))
    assert not history["race_id"].eq("r1").any()
    assert np.isnan(_value(history, "r2", "h1"))
    assert not history["race_id"].eq("r3").any()
    with pytest.raises(ValueError, match="2025"):
        build_pace_final_history(pd.DataFrame(rows), through_year=2025)


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
            PACE_FINAL_COLUMN: [np.nan, 0.75],
            "_pace_final_history_row_present": [True, True],
        }
    )
    result = build_pace_final_augmented_cache(
        baseline_path, history, output_path, config=_config()
    )
    output = pd.read_pickle(output_path)
    assert output.loc[:, features].equals(baseline.loc[:, features])
    assert output.loc[2, PACE_FINAL_COLUMN] is np.nan or np.isnan(output.loc[2, PACE_FINAL_COLUMN])
    assert result["retrospective_2025_feature_nonmissing"] == 0
    assert compare_pace_final_cache_control(
        baseline_path, output_path, expected_baseline_feature_count=2
    )["passed"]


def test_raw_builder_removes_2025_before_normalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["202401010101"],
            "horse_id": ["h1"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "split": ["development"],
            "horse_number": [1],
            "context__distance": np.array([1600.0], dtype="float32"),
        }
    )
    base = tmp_path / "base.pkl"
    out = tmp_path / "out.pkl"
    _write_cache(base, baseline, ["context__distance"])
    raw = pd.DataFrame({"raceid": ["202401010101", "202501010101"], "sentinel": [2024, 2025]})
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setattr("horse_pred.pace_final.load_manifest", lambda _: {"raw_file": {"sha256": "raw-sha"}})
    monkeypatch.setattr("horse_pred.pace_final.verify_raw_file", lambda *_: None)
    monkeypatch.setattr("horse_pred.pace_final.load_raw", lambda *_args, **_kwargs: raw)

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        assert frame["sentinel"].tolist() == [2024]
        return frame

    monkeypatch.setattr("horse_pred.pace_final.normalize_raw", normalize)
    history = pd.DataFrame(
        {
            "race_id": ["202401010101"],
            "horse_id": ["h1"],
            PACE_FINAL_COLUMN: [np.nan],
            "_pace_final_history_row_present": [True],
        }
    )
    monkeypatch.setattr("horse_pred.pace_final.build_pace_final_history", lambda *_args, **_kwargs: history)
    result = build_pace_final_cache_from_raw(
        repo_root=tmp_path,
        raw_path=tmp_path / "raw.csv",
        baseline_cache_path=base,
        output_path=out,
        config_path=cfg,
    )
    assert result["raw_rows_after_pre_normalization_cutoff"] == 1


def test_config_taxonomy_rolling_and_cli_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    config, spec = load_pace_final_config(root / "configs/features/pace_03_final_position.json")
    assert canonical_json_hash(spec.transformation_dict()) == PACE_FINAL_TRANSFORMATION_HASH
    assert config["transformation_hash"] == PACE_FINAL_TRANSFORMATION_HASH
    groups = semantic_feature_groups_v2(("context__distance", PACE_FINAL_COLUMN))
    assert groups["pace_final"] == (PACE_FINAL_COLUMN,)
    frame = pd.DataFrame({"context__distance": [1600.0], PACE_FINAL_COLUMN: [0.5], "通過順位": ["1-1"]})
    assert PACE_FINAL_COLUMN in model_feature_allowlist(frame)
    assert "通過順位" not in model_feature_allowlist(frame)
    rolling = json.loads((root / "configs/evaluation/pace_03_rolling.json").read_text())
    validate_rolling_config(rolling)
    assert rolling["maximum_outcome_year"] == 2023

    args = parser().parse_args(
        [
            "build-pace-final-cache",
            "--raw-path",
            "raw.csv",
            "--baseline-cache",
            "base.pkl",
            "--output",
            "final.pkl",
        ]
    )
    assert args.command == "build-pace-final-cache"
