from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_opponent_recent_cache_control
from horse_pred.features import semantic_feature_groups_v2
from horse_pred.opponent_recent import (
    OPPONENT_RECENT_COLUMN,
    OPPONENT_RECENT_TRANSFORMATION_HASH,
    OpponentRecentSpec,
    build_opponent_recent_augmented_cache,
    build_opponent_recent_cache_from_raw,
    build_opponent_recent_history,
    load_opponent_recent_config,
)


def _row(
    race_id: str,
    date: str,
    horse_id: str,
    finish: object,
    *,
    race_class: str = "3歳未勝利",
) -> dict[str, object]:
    return {
        "raceid": race_id,
        "date": date,
        "horse_id": horse_id,
        "着順": finish,
        "started": True,
        "course_type": "芝",
        "race_class": race_class,
    }


def _value(history: pd.DataFrame, race_id: str, horse_id: str) -> float:
    return float(
        history.loc[
            history["race_id"].eq(race_id) & history["horse_id"].eq(horse_id),
            OPPONENT_RECENT_COLUMN,
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
                "feature_groups_v1": {"race_context": features},
            }
        ),
        encoding="utf-8",
    )


def _registered_config() -> dict[str, object]:
    spec = OpponentRecentSpec()
    return {
        "candidate_column": OPPONENT_RECENT_COLUMN,
        "transformation_hash": OPPONENT_RECENT_TRANSFORMATION_HASH,
        "transformation": spec.transformation_dict(),
    }


def test_history_excludes_self_rating_and_uses_90_day_decay() -> None:
    rows = [
        _row("202301010101", "2023-01-01", "h1", 1),
        _row("202301010101", "2023-01-01", "h2", 2),
        _row("202301010201", "2023-01-02", "h1", 1),
        _row("202301010201", "2023-01-02", "h2", 2),
        _row("202301010201", "2023-01-02", "h3", 3),
        _row("202301010301", "2023-01-03", "h1", 1),
        _row("202301010301", "2023-01-03", "h2", 2),
        _row("202301010301", "2023-01-03", "h3", 3),
    ]
    history = build_opponent_recent_history(pd.DataFrame(rows))

    assert np.isnan(_value(history, "202301010101", "h1"))
    assert _value(history, "202301010201", "h1") == pytest.approx(1500.0)
    assert _value(history, "202301010201", "h2") == pytest.approx(1500.0)
    assert np.isnan(_value(history, "202301010201", "h3"))

    # Day-one pre-ratings were tied at 1500.  After h1 beat h2 they are
    # 1512/1488, so day-two opponent-only observations are 1494/1506/1500.
    daily_decay = exp(-log(2.0) / 90.0)
    h1_expected = (1500.0 * daily_decay + 1494.0) / (daily_decay + 1.0)
    h2_expected = (1500.0 * daily_decay + 1506.0) / (daily_decay + 1.0)
    assert _value(history, "202301010301", "h1") == pytest.approx(h1_expected)
    assert _value(history, "202301010301", "h2") == pytest.approx(h2_expected)
    assert _value(history, "202301010301", "h3") == pytest.approx(1500.0)


def test_history_batches_every_race_on_same_date_and_singleton_adds_nothing() -> None:
    rows = [
        _row("202301010101", "2023-01-01", "h1", 1),
        _row("202301010101", "2023-01-01", "h2", 2),
        _row("202301010201", "2023-01-02", "h1", 1),
        _row("202301010201", "2023-01-02", "h3", 2),
        _row("202301010202", "2023-01-02", "h1", 2),
        _row("202301010202", "2023-01-02", "h4", 1),
        _row("202301010203", "2023-01-02", "solo", 1),
        _row("202301010301", "2023-01-03", "solo", 1),
    ]
    history = build_opponent_recent_history(pd.DataFrame(rows))

    assert _value(history, "202301010201", "h1") == pytest.approx(1500.0)
    assert _value(history, "202301010202", "h1") == pytest.approx(1500.0)
    assert np.isnan(_value(history, "202301010203", "solo"))
    assert np.isnan(_value(history, "202301010301", "solo"))


def test_history_ignores_obstacle_races_and_rejects_2025_cutoff() -> None:
    rows = [
        _row(
            "202301010101",
            "2023-01-01",
            "h1",
            1,
            race_class="障害4歳以上未勝利",
        ),
        _row(
            "202301010101",
            "2023-01-01",
            "h2",
            2,
            race_class="障害4歳以上未勝利",
        ),
        _row("202301010201", "2023-01-02", "h1", 1),
        _row("202301010201", "2023-01-02", "h3", 2),
        _row("202501010101", "2025-01-01", "h1", 1),
        _row("202501010101", "2025-01-01", "h4", 2),
    ]
    history = build_opponent_recent_history(pd.DataFrame(rows))

    assert not history["race_id"].eq("202301010101").any()
    assert np.isnan(_value(history, "202301010201", "h1"))
    assert not history["race_id"].eq("202501010101").any()
    with pytest.raises(ValueError, match="2025"):
        build_opponent_recent_history(pd.DataFrame(rows), through_year=2025)


def test_cache_preserves_control_cold_nan_and_2025_firewall(tmp_path: Path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101", "202501010101"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2025-01-01"]
            ),
            "split": ["development", "development", "retrospective_test"],
            "horse_number": [1, 2, 1],
            "context__distance": np.array([1600.0, np.nan, 1200.0], dtype="float32"),
        }
    )
    baseline_path = tmp_path / "baseline.pkl"
    output_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["context__distance"])
    history = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101"],
            "horse_id": ["h1", "h2"],
            OPPONENT_RECENT_COLUMN: [np.nan, 1510.0],
            "_opponent_recent_history_row_present": [True, True],
        }
    )

    result = build_opponent_recent_augmented_cache(
        baseline_path,
        history,
        output_path,
        config=_registered_config(),
    )
    augmented = pd.read_pickle(output_path)

    assert result["old_feature_exact"] is True
    assert result["candidate_feature_count"] == 2
    assert augmented["context__distance"].equals(baseline["context__distance"])
    assert np.isnan(augmented.loc[0, OPPONENT_RECENT_COLUMN])
    assert augmented.loc[1, OPPONENT_RECENT_COLUMN] == pytest.approx(1510.0)
    assert np.isnan(augmented.loc[2, OPPONENT_RECENT_COLUMN])
    metadata = json.loads(
        output_path.with_suffix(".pkl.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["feature_columns"] == ["context__distance", OPPONENT_RECENT_COLUMN]
    assert metadata["opponent_recent"]["transformation_hash"] == (
        OPPONENT_RECENT_TRANSFORMATION_HASH
    )
    control = compare_opponent_recent_cache_control(
        baseline_path,
        output_path,
        expected_baseline_feature_count=1,
    )
    assert control["passed"]


def test_raw_builder_removes_2025_before_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    baseline_path = tmp_path / "baseline.pkl"
    output_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["context__distance"])
    raw = pd.DataFrame(
        {"raceid": ["202401010101", "202501010101"], "sentinel": [2024, 2025]}
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_registered_config()), encoding="utf-8")

    monkeypatch.setattr(
        "horse_pred.opponent_recent.load_manifest",
        lambda _: {"raw_file": {"sha256": "raw-sha"}},
    )
    monkeypatch.setattr("horse_pred.opponent_recent.verify_raw_file", lambda *_: None)
    monkeypatch.setattr("horse_pred.opponent_recent.load_raw", lambda *_args, **_kwargs: raw)

    def normalize_without_future(frame: pd.DataFrame) -> pd.DataFrame:
        assert frame["sentinel"].tolist() == [2024]
        return frame

    monkeypatch.setattr("horse_pred.opponent_recent.normalize_raw", normalize_without_future)
    history = pd.DataFrame(
        {
            "race_id": ["202401010101"],
            "horse_id": ["h1"],
            OPPONENT_RECENT_COLUMN: [np.nan],
            "_opponent_recent_history_row_present": [True],
        }
    )
    monkeypatch.setattr(
        "horse_pred.opponent_recent.build_opponent_recent_history",
        lambda *_args, **_kwargs: history,
    )

    result = build_opponent_recent_cache_from_raw(
        repo_root=tmp_path,
        raw_path=tmp_path / "raw.csv",
        baseline_cache_path=baseline_path,
        output_path=output_path,
        config_path=config_path,
    )

    assert result["raw_rows_after_pre_normalization_cutoff"] == 1
    assert result["retrospective_2025_feature_nonmissing"] == 0


def test_frozen_config_hash_and_semantic_group() -> None:
    root = Path(__file__).resolve().parents[1]
    config, spec = load_opponent_recent_config(
        root / "configs/features/opp_recent_001.json"
    )

    assert config["transformation_hash"] == OPPONENT_RECENT_TRANSFORMATION_HASH
    assert spec.transformation_dict() == config["transformation"]
    groups = semantic_feature_groups_v2(
        ("context__distance", OPPONENT_RECENT_COLUMN)
    )
    assert groups["opponent_recent"] == (OPPONENT_RECENT_COLUMN,)
