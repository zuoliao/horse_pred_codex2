from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from horse_pred.artifacts import write_artifact_manifest
from horse_pred.data import RAW_COLUMNS
from horse_pred.s1_two_axis_study import (
    _decision_payload,
    assign_s1_slice_flags,
    classify_s1_comparison,
    isolate_s1_source,
    validate_s1_preregistration,
    verify_artifact_manifest,
)
from horse_pred.two_axis_race_value import FIELD_QUALITY_COLUMN, PERFORMANCE_COLUMN

PREREGISTRATION = Path(
    "experiments/s1_two_axis_race_value_20260901/preregistration.json"
)


def _preregistration() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def _raw_row(race_id: str, horse_number: int, finish: str, date: str) -> dict[str, str]:
    return {
        "raceid": race_id,
        "race_class": "3歳未勝利",
        "course_type": "芝",
        "distance": "1600",
        "ground_state": "良",
        "around": "右",
        "weather": "晴",
        "着順": finish,
        "枠番": str(horse_number),
        "馬番": str(horse_number),
        "馬名": f"horse-{horse_number}",
        "horse_id": f"{horse_number:010d}",
        "sex": "牡",
        "age": "3",
        "騎手": f"jockey-{horse_number}",
        "jockey_id": f"{100 + horse_number:04d}",
        "trainer": f"trainer-{horse_number}",
        "タイム": "1:34.5" if finish == "1" else "1:34.8",
        "着差": "" if finish == "1" else "2",
        "通過順位": "1-1" if finish == "1" else "2-2",
        "上がり3F": "34.5" if finish == "1" else "34.8",
        "単勝": "2.5" if finish == "1" else "4.0",
        "人気": "1" if finish == "1" else "2",
        "馬体重": "480",
        "馬体重増減": "0",
        "date": date,
    }


def _write_raw(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_preregistration_freezes_three_folds_four_arms_and_eight_methods() -> None:
    config = _preregistration()
    validate_s1_preregistration(config)

    assert config["maximum_outcome_year"] == 2022
    assert config["forbidden_years"] == [2023, 2024, 2025]
    assert [fold["evaluation_year"] for fold in config["folds"]] == [2020, 2021, 2022]
    assert list(config["arms"]) == ["C0", "C1", "C2", "C3"]
    model_controls = {
        family: value
        for family, value in config["controls"].items()
        if isinstance(value, dict)
    }
    assert len(model_controls) * len(config["arms"]) == 8
    assert config["candidate_comparison_count"] == 10


def test_preregistered_arm_columns_and_feature_counts_are_exact() -> None:
    config = _preregistration()
    assert config["arms"] == {
        "C0": [],
        "C1": [PERFORMANCE_COLUMN],
        "C2": [FIELD_QUALITY_COLUMN],
        "C3": [PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN],
    }

    counts = {
        family: {
            arm: int(control["feature_count"]) + len(additions)
            for arm, additions in config["arms"].items()
        }
        for family, control in config["controls"].items()
        if isinstance(control, dict)
    }
    assert counts == {
        "binary": {"C0": 254, "C1": 255, "C2": 255, "C3": 256},
        "lambdarank": {"C0": 253, "C1": 254, "C2": 254, "C3": 255},
    }


def test_preregistration_rejects_post_2022_or_market_enabled_variants() -> None:
    config = _preregistration()
    for maximum in (2023, 2024, 2025):
        bad = copy.deepcopy(config)
        bad["maximum_outcome_year"] = maximum
        with pytest.raises(ValueError):
            validate_s1_preregistration(bad)

    bad = copy.deepcopy(config)
    bad["market_used"] = True
    with pytest.raises(ValueError):
        validate_s1_preregistration(bad)


def test_preregistered_features_contain_no_market_or_direct_entity_id() -> None:
    config = _preregistration()
    additions = {
        column for columns in config["arms"].values() for column in columns
    }
    forbidden_tokens = (
        "odds",
        "popularity",
        "人気",
        "オッズ",
        "horse_id",
        "jockey_id",
        "trainer_id",
    )
    assert additions == {PERFORMANCE_COLUMN, FIELD_QUALITY_COLUMN}
    assert not any(
        token in column.lower()
        for column in additions
        for token in forbidden_tokens
    )


def test_preregistration_freezes_metrics_slices_and_comparison_paths() -> None:
    config = _preregistration()
    assert config["metrics"] == [
        "ndcg_at_3",
        "top_1_winner_mass",
        "winner_reciprocal_rank",
        "race_log_loss",
        "race_brier",
    ]
    assert config["slices"] == [
        "history_0_winner",
        "new_race",
        "open_or_graded",
        "field_size_15_plus",
        "winner_surface_switch",
        "winner_distance_change_400_plus",
    ]
    assert config["comparisons_per_family"] == [
        "C1_vs_C0",
        "C2_vs_C0",
        "C3_vs_C0",
        "C3_vs_C2",
        "C3_vs_C1",
    ]


def test_s1_source_isolates_before_normalization_and_reports_zero_use(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.csv"
    _write_raw(
        path,
        [
            _raw_row("202205010101", 1, "1", "2022-01-01"),
            _raw_row("202205010101", 2, "2", "2022-01-01"),
            _raw_row("202305010101", 1, "1", "2023-01-01"),
            _raw_row("202405010101", 1, "1", "2024-01-01"),
            _raw_row("202505010101", 1, "1", "2025-01-01"),
        ],
    )

    frame, audit = isolate_s1_source(path, maximum_outcome_year=2022)

    assert set(frame["race_id"].astype(str)) == {"202205010101"}
    assert pd.to_datetime(frame["race_date"]).dt.year.max() == 2022
    assert audit["excluded_rows_by_year"] == {"2023": 1, "2024": 1, "2025": 1}
    assert audit["rows_used_2023"] == 0
    assert audit["rows_used_2024"] == 0
    assert audit["rows_used_2025"] == 0
    for forbidden_cutoff in (2023, 2024, 2025):
        with pytest.raises(ValueError, match="2022"):
            isolate_s1_source(path, maximum_outcome_year=forbidden_cutoff)


def test_slice_flags_are_race_constant_and_never_break_choice_sets() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["new"] * 2 + ["open-large"] * 2,
            "model_finish_position": [1, 2, 1, 2],
            "field_size": [2, 2, 15, 15],
            "horse_history__career__starts": [0, 0, 3, 4],
            "context__class_tier": [0, 0, 5, 5],
            "course_type": ["芝", "芝", "芝", "芝"],
            "distance": [1600, 1600, 1600, 1600],
            "previous_surface": [pd.NA, pd.NA, "ダート", "芝"],
            "previous_distance": [float("nan"), float("nan"), 1000, 1600],
        }
    )

    flags = assign_s1_slice_flags(frame)

    assert set(flags) == set(_preregistration()["slices"])
    for flag in flags.values():
        assert flag.groupby(frame["race_id"], observed=True).nunique().eq(1).all()
        selected_races = set(frame.loc[flag, "race_id"])
        for race_id in selected_races:
            assert int(flag.loc[frame["race_id"].eq(race_id)].sum()) == int(
                frame["race_id"].eq(race_id).sum()
            )
    assert flags["history_0_winner"].tolist() == [True, True, False, False]
    assert flags["new_race"].tolist() == [True, True, False, False]
    assert flags["open_or_graded"].tolist() == [False, False, True, True]
    assert flags["field_size_15_plus"].tolist() == [False, False, True, True]
    assert flags["winner_surface_switch"].tolist() == [False, False, True, True]
    assert flags["winner_distance_change_400_plus"].tolist() == [False, False, True, True]


def _comparison_summary(
    *, log_loss: float = 0.003, ndcg: float = 0.001, improved_years: int = 3
) -> dict[str, object]:
    return {
        "metrics": {
            "race_log_loss": {
                "year_macro_improvement": log_loss,
                "improved_years": improved_years,
            },
            "race_brier": {"year_macro_improvement": 0.0005},
            "ndcg_at_3": {
                "year_macro_improvement": ndcg,
                "improved_years": improved_years,
            },
            "top_1_winner_mass": {"year_macro_improvement": 0.0},
        }
    }


def test_comparison_decision_distinguishes_supported_weak_and_rejected() -> None:
    interval = {
        "race_log_loss": {"lower": 0.001, "upper": 0.005},
        "ndcg_at_3": {"lower": 0.0001, "upper": 0.002},
    }
    assert (
        classify_s1_comparison(_comparison_summary(), interval, path="probability")
        == "supported"
    )
    crossing = copy.deepcopy(interval)
    crossing["race_log_loss"]["lower"] = -0.001
    assert (
        classify_s1_comparison(_comparison_summary(), crossing, path="probability")
        == "weakly_supported"
    )
    rejected = _comparison_summary(log_loss=-0.003, ndcg=-0.003, improved_years=0)
    assert classify_s1_comparison(rejected, interval, path="probability") == "rejected"
    assert classify_s1_comparison(rejected, interval, path="ranking") == "rejected"
    with pytest.raises(ValueError, match="probability or ranking"):
        classify_s1_comparison(_comparison_summary(), interval, path="other")


def test_conditional_field_increment_does_not_promote_standalone_field_axis() -> None:
    pairs = ("C1_vs_C0", "C2_vs_C0", "C3_vs_C0", "C3_vs_C2", "C3_vs_C1")
    comparisons: dict[str, object] = {}
    intervals: dict[str, object] = {}
    for family in ("binary", "lambdarank"):
        for pair in pairs:
            comparison_id = f"{family}_{pair}"
            direct_field = pair == "C2_vs_C0"
            conditional_field = pair == "C3_vs_C1"
            comparisons[comparison_id] = _comparison_summary(
                log_loss=-0.003 if direct_field else 0.003,
                ndcg=-0.003 if direct_field else 0.001,
                improved_years=0 if direct_field else 3,
            )
            lower = -0.001 if conditional_field else 0.001
            intervals[comparison_id] = {
                "race_log_loss": {"lower": lower, "upper": 0.005},
                "ndcg_at_3": {"lower": lower, "upper": 0.002},
            }

    decision = _decision_payload(
        {"comparisons": comparisons}, {"paired": intervals}
    )

    assert decision["axes"]["performance_axis"] == "supported"
    assert decision["axes"]["field_quality_axis"] == "rejected"
    assert decision["axes"]["joint_two_axis"] == "supported"
    assert decision["conditional_increment"]["field_quality_given_performance"] == (
        "weakly_supported"
    )
    assert decision["case"] == "A_performance_supported"
    assert decision["next_recommendation"] == (
        "S3_condition_adjusted_performance_target"
    )


def test_artifact_manifest_verification_checks_every_hash_and_detects_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")
    nested = artifact / "feature_tables"
    nested.mkdir()
    (nested / "summary.csv").write_text("feature,value\na,1\n", encoding="utf-8")
    write_artifact_manifest(artifact)

    report = verify_artifact_manifest(artifact)
    assert report == {"verified": True, "file_count": 2}

    (artifact / "metrics.json").write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_artifact_manifest(artifact)
