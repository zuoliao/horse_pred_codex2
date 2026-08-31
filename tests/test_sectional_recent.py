from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horse_pred.cache_control import compare_sectional_recent_cache_control
from horse_pred.config import canonical_json_hash
from horse_pred.features import (
    model_feature_allowlist,
    semantic_feature_groups_v2,
)
from horse_pred.rolling_evaluation import validate_rolling_config
from horse_pred.sectional_recent import (
    SECTIONAL_RECENT_COLUMN,
    SECTIONAL_RECENT_TRANSFORMATION_HASH,
    SectionalRecentSpec,
    build_sectional_recent_augmented_cache,
    build_sectional_recent_cache_from_raw,
    build_sectional_recent_history,
    last_3f_speed_percentiles,
    load_sectional_recent_config,
)


def _row(
    race_id: str,
    date: str,
    horse_id: str,
    last_3f: object,
    *,
    started: object = True,
    race_class: str = "3歳未勝利",
) -> dict[str, object]:
    return {
        "race_id": race_id,
        "race_date": pd.Timestamp(date),
        "horse_id": horse_id,
        "started": started,
        "last_3f_seconds": last_3f,
        "course_type": "芝",
        "race_class": race_class,
    }


def _value(history: pd.DataFrame, race_id: str, horse_id: str) -> float:
    return float(
        history.loc[
            history["race_id"].eq(race_id) & history["horse_id"].eq(horse_id),
            SECTIONAL_RECENT_COLUMN,
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
    spec = SectionalRecentSpec()
    return {
        "candidate_column": SECTIONAL_RECENT_COLUMN,
        "transformation_hash": SECTIONAL_RECENT_TRANSFORMATION_HASH,
        "transformation": spec.transformation_dict(),
    }


def test_percentile_uses_ascending_average_tie_rank_and_finite_starters() -> None:
    race = pd.DataFrame(
        {
            "started": [True, True, True, True, False],
            "last_3f_seconds": [34.0, 35.0, 35.0, np.nan, 33.0],
        }
    )
    percentile = last_3f_speed_percentiles(race)

    assert percentile.iloc[0] == pytest.approx(1.0)
    assert percentile.iloc[1] == pytest.approx(0.25)
    assert percentile.iloc[2] == pytest.approx(0.25)
    assert np.isnan(percentile.iloc[3])
    assert np.isnan(percentile.iloc[4])


def test_percentile_requires_two_valid_values_and_ignores_infinity() -> None:
    race = pd.DataFrame(
        {
            "started": [True, True, True, False],
            "last_3f_seconds": [34.0, np.nan, np.inf, 33.0],
        }
    )
    assert last_3f_speed_percentiles(race).isna().all()


def test_history_uses_own_observation_and_90_day_decay() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", 34.0),
        _row("r1", "2023-01-01", "h2", 35.0),
        _row("r1", "2023-01-01", "h3", 36.0),
        _row("r2", "2023-01-02", "h1", 36.0),
        _row("r2", "2023-01-02", "h4", 34.0),
        _row("r3", "2023-01-03", "h1", 35.0),
        _row("r3", "2023-01-03", "h5", 34.0),
    ]
    history = build_sectional_recent_history(pd.DataFrame(rows))

    assert np.isnan(_value(history, "r1", "h1"))
    assert _value(history, "r2", "h1") == pytest.approx(1.0)
    daily_decay = exp(-log(2.0) / 90.0)
    expected = (1.0 * daily_decay**2 + 0.0 * daily_decay) / (daily_decay**2 + daily_decay)
    assert _value(history, "r3", "h1") == pytest.approx(expected)


def test_history_batches_all_same_date_races() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", 34.0),
        _row("r1", "2023-01-01", "h2", 35.0),
        _row("r2", "2023-01-01", "h1", 36.0),
        _row("r2", "2023-01-01", "h3", 34.0),
        _row("r3", "2023-01-02", "h1", 35.0),
        _row("r3", "2023-01-02", "h4", 34.0),
    ]
    history = build_sectional_recent_history(pd.DataFrame(rows))

    assert np.isnan(_value(history, "r1", "h1"))
    assert np.isnan(_value(history, "r2", "h1"))
    assert _value(history, "r3", "h1") == pytest.approx(0.5)


def test_missing_sectional_does_not_update_but_prior_history_is_emitted() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", 34.0),
        _row("r1", "2023-01-01", "h2", 35.0),
        _row("r2", "2023-01-02", "h1", np.nan),
        _row("r2", "2023-01-02", "h3", 34.0),
        _row("r3", "2023-01-03", "h1", 36.0),
        _row("r3", "2023-01-03", "h4", 34.0),
    ]
    history = build_sectional_recent_history(pd.DataFrame(rows))

    assert _value(history, "r2", "h1") == pytest.approx(1.0)
    assert _value(history, "r3", "h1") == pytest.approx(1.0)


def test_history_ignores_obstacle_and_2025_and_rejects_future_cutoff() -> None:
    rows = [
        _row("r1", "2023-01-01", "h1", 34.0, race_class="障害4歳以上未勝利"),
        _row("r1", "2023-01-01", "h2", 35.0, race_class="障害4歳以上未勝利"),
        _row("r2", "2023-01-02", "h1", 34.0),
        _row("r2", "2023-01-02", "h3", 35.0),
        _row("r3", "2025-01-01", "h1", 34.0),
        _row("r3", "2025-01-01", "h4", 35.0),
    ]
    history = build_sectional_recent_history(pd.DataFrame(rows))

    assert not history["race_id"].eq("r1").any()
    assert np.isnan(_value(history, "r2", "h1"))
    assert not history["race_id"].eq("r3").any()
    with pytest.raises(ValueError, match="2025"):
        build_sectional_recent_history(pd.DataFrame(rows), through_year=2025)


def test_cache_preserves_arbitrary_control_cold_nan_and_2025_firewall(tmp_path: Path) -> None:
    baseline = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101", "202501010101"],
            "horse_id": ["h1", "h2", "h3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
            "split": ["development", "development", "retrospective_test"],
            "horse_number": [1, 2, 1],
            "context__distance": np.array([1600.0, np.nan, 1200.0], dtype="float32"),
            "race_content__arbitrary_existing": np.array([0.1, 0.2, 0.3], dtype="float32"),
        }
    )
    old_features = ["context__distance", "race_content__arbitrary_existing"]
    baseline_path = tmp_path / "baseline.pkl"
    output_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, old_features)
    history = pd.DataFrame(
        {
            "race_id": ["202401010101", "202401010101"],
            "horse_id": ["h1", "h2"],
            SECTIONAL_RECENT_COLUMN: [np.nan, 0.75],
            "_sectional_recent_history_row_present": [True, True],
        }
    )

    result = build_sectional_recent_augmented_cache(
        baseline_path,
        history,
        output_path,
        config=_registered_config(),
    )
    augmented = pd.read_pickle(output_path)

    assert result["old_feature_exact"] is True
    assert result["candidate_feature_count"] == 3
    assert augmented.loc[:, old_features].equals(baseline.loc[:, old_features])
    assert np.isnan(augmented.loc[0, SECTIONAL_RECENT_COLUMN])
    assert augmented.loc[1, SECTIONAL_RECENT_COLUMN] == pytest.approx(0.75)
    assert np.isnan(augmented.loc[2, SECTIONAL_RECENT_COLUMN])
    metadata = json.loads(output_path.with_suffix(".pkl.meta.json").read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == [*old_features, SECTIONAL_RECENT_COLUMN]
    assert metadata["sectional_recent"]["transformation_hash"] == (SECTIONAL_RECENT_TRANSFORMATION_HASH)
    control = compare_sectional_recent_cache_control(
        baseline_path,
        output_path,
        expected_baseline_feature_count=2,
    )
    assert control["passed"]


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
    baseline_path = tmp_path / "baseline.pkl"
    output_path = tmp_path / "candidate.pkl"
    _write_cache(baseline_path, baseline, ["context__distance"])
    raw = pd.DataFrame({"raceid": ["202401010101", "202501010101"], "sentinel": [2024, 2025]})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_registered_config()), encoding="utf-8")

    monkeypatch.setattr(
        "horse_pred.sectional_recent.load_manifest",
        lambda _: {"raw_file": {"sha256": "raw-sha"}},
    )
    monkeypatch.setattr("horse_pred.sectional_recent.verify_raw_file", lambda *_: None)
    monkeypatch.setattr("horse_pred.sectional_recent.load_raw", lambda *_args, **_kwargs: raw)

    def normalize_without_future(frame: pd.DataFrame) -> pd.DataFrame:
        assert frame["sentinel"].tolist() == [2024]
        return frame

    monkeypatch.setattr("horse_pred.sectional_recent.normalize_raw", normalize_without_future)
    history = pd.DataFrame(
        {
            "race_id": ["202401010101"],
            "horse_id": ["h1"],
            SECTIONAL_RECENT_COLUMN: [np.nan],
            "_sectional_recent_history_row_present": [True],
        }
    )
    monkeypatch.setattr(
        "horse_pred.sectional_recent.build_sectional_recent_history",
        lambda *_args, **_kwargs: history,
    )

    result = build_sectional_recent_cache_from_raw(
        repo_root=tmp_path,
        raw_path=tmp_path / "raw.csv",
        baseline_cache_path=baseline_path,
        output_path=output_path,
        config_path=config_path,
    )

    assert result["raw_rows_after_pre_normalization_cutoff"] == 1
    assert result["retrospective_2025_feature_nonmissing"] == 0


def test_frozen_config_hash_feature_group_and_allowlist() -> None:
    root = Path(__file__).resolve().parents[1]
    config, spec = load_sectional_recent_config(root / "configs/features/sec_3f_001.json")

    assert config["transformation_hash"] == SECTIONAL_RECENT_TRANSFORMATION_HASH
    assert canonical_json_hash(spec.transformation_dict()) == (SECTIONAL_RECENT_TRANSFORMATION_HASH)
    groups = semantic_feature_groups_v2(("context__distance", SECTIONAL_RECENT_COLUMN))
    assert groups["sectional"] == (SECTIONAL_RECENT_COLUMN,)
    frame = pd.DataFrame({"context__distance": [1600.0], SECTIONAL_RECENT_COLUMN: [0.5], "上がり3F": [34.0]})
    allowed = model_feature_allowlist(frame)
    assert SECTIONAL_RECENT_COLUMN in allowed
    assert "上がり3F" not in allowed


def test_rolling_config_keeps_2024_closed_and_freezes_candidate_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/evaluation/sec_3f_001_rolling.json").read_text(encoding="utf-8"))
    assert config["maximum_outcome_year"] == 2023
    assert [fold["evaluation_year"] for fold in config["folds"]] == [
        2020,
        2021,
        2022,
        2023,
    ]
    assert config["methods"]["binary_candidate"]["expected_columns_sha256"] == (
        "7dc5fe968a454a8415723995ae5fe73cdebb195b4da97477f9e9965afad3eaa6"
    )
    assert config["methods"]["lambdarank_candidate"]["expected_columns_sha256"] == (
        "fff37fcb9c1b40611df1a1813908be8d4b9eba2fd5dcd9a575858cc1f4942183"
    )
    validate_rolling_config(config)
