from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from horse_pred.artifacts import write_artifact_manifest
from horse_pred.data import RAW_COLUMNS
from horse_pred.s3_performance_target_study import (
    _feature_scope,
    isolate_s3_source,
    validate_s3_preregistration,
    verify_artifact_manifest,
)

PREREGISTRATION = Path(
    "experiments/s3_condition_adjusted_performance_target_20260901/"
    "preregistration.json"
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


def test_preregistration_freezes_target_only_methods_folds_and_comparisons() -> None:
    config = _preregistration()

    validate_s3_preregistration(config)

    assert config["maximum_outcome_year"] == 2022
    assert config["forbidden_years"] == [2023, 2024, 2025]
    assert [fold["evaluation_year"] for fold in config["folds"]] == [
        2020,
        2021,
        2022,
    ]
    assert config["target"] == {
        "column": "target__condition_adjusted_performance",
        "raw_value": (
            "expected_winner_seconds_per_1000m_minus_runner_seconds_per_1000m"
        ),
        "higher_is_better": True,
        "ridge_alpha": 1.0,
        "clip": [-5.0, 5.0],
        "normalizer": "pooled_fold_train_2014_to_train_end_then_frozen",
        "effects": [
            "course_surface",
            "exact_distance",
            "going",
            "class_tier",
            "age_restriction",
        ],
        "season_included": False,
        "field_size_included": False,
        "track_day_variant_included": False,
        "missing_target_fit_policy": "exclude_without_zero_imputation",
    }
    assert list(config["methods"]) == [
        "binary_control",
        "lambdarank_control",
        "huber_binary_scope",
        "huber_lambdarank_scope",
    ]
    assert [item["id"] for item in config["comparisons"]] == [
        "huber_binary_scope_vs_binary_control",
        "huber_lambdarank_scope_vs_lambdarank_control",
    ]
    assert config["candidate_comparison_count"] == 2
    assert config["s1_performance_feature_added"] is False
    assert config["production_changed_by_preregistration"] is False


def test_preregistration_rejects_firewall_and_target_definition_mutations() -> None:
    config = _preregistration()
    mutations: list[tuple[tuple[str, ...], object]] = [
        (("maximum_outcome_year",), 2023),
        (("forbidden_years",), [2024, 2025]),
        (("market_used",), True),
        (("s1_performance_feature_added",), True),
        (("target", "column"), "target__finish_position"),
        (("target", "normalizer"), "full_period"),
        (("target", "clip"), [-10.0, 10.0]),
        (("regression_parameters", "objective"), "regression"),
        (("regression_parameters", "alpha"), 0.5),
    ]
    for path, value in mutations:
        bad = copy.deepcopy(config)
        destination = bad
        for key in path[:-1]:
            destination = destination[key]
        destination[path[-1]] = value
        with pytest.raises(ValueError):
            validate_s3_preregistration(bad)


def test_s3_source_isolates_before_normalization_and_reports_zero_use(
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

    frame, audit = isolate_s3_source(path, maximum_outcome_year=2022)

    assert set(frame["race_id"].astype(str)) == {"202205010101"}
    assert pd.to_datetime(frame["race_date"]).dt.year.max() == 2022
    assert audit["excluded_rows_by_year"] == {"2023": 1, "2024": 1, "2025": 1}
    assert audit["rows_used_2023"] == 0
    assert audit["rows_used_2024"] == 0
    assert audit["rows_used_2025"] == 0
    for forbidden_cutoff in (2023, 2024, 2025):
        with pytest.raises(ValueError, match="2022"):
            isolate_s3_source(path, maximum_outcome_year=forbidden_cutoff)


@pytest.mark.parametrize(
    "forbidden",
    [
        "target__condition_adjusted_performance",
        "finish_position",
        "winner_label",
        "final_odds",
        "popularity",
        "horse_id",
        "jockey_id",
        "trainer_id",
    ],
)
def test_feature_scope_rejects_target_outcome_market_and_entity_ids(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden|market"):
        _feature_scope(("context__distance", forbidden), [])


def test_feature_scope_is_target_only_and_rejects_duplicate_columns() -> None:
    control = ("context__distance", "horse_history__career__starts")

    assert _feature_scope(control, []) == control
    with pytest.raises(ValueError, match="duplicate"):
        _feature_scope(control, [control[0]])


def test_artifact_manifest_verification_checks_all_hashes_and_detects_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")
    nested = artifact / "target_diagnostics"
    nested.mkdir()
    (nested / "coverage.csv").write_text(
        "fold,coverage\nroll_2020,0.99\n", encoding="utf-8"
    )
    write_artifact_manifest(artifact)

    report = verify_artifact_manifest(artifact)
    assert report == {"verified": True, "file_count": 2}

    (artifact / "metrics.json").write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        verify_artifact_manifest(artifact)
