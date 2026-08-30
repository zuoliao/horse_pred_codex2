from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from horse_pred.data import (
    RAW_COLUMNS,
    DataContractError,
    FingerprintMismatchError,
    SchemaMismatchError,
    audit_csv,
    audit_raw,
    expand_race_id_ranges,
    load_manifest,
    load_raw,
    normalize_raw,
    resolve_raw_path,
    verify_audit_against_manifest,
)
from horse_pred.data_health import build_race_population_table, population_selection_audit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _raw_row(
    race_id: str,
    finish: str,
    horse_number: int,
    *,
    course_type: str = "芝",
    margin: str = "",
    race_date: str = "2024-01-06",
) -> dict[str, str]:
    return {
        "raceid": race_id,
        "race_class": "3歳以上\u00a01勝クラス",
        "course_type": course_type,
        "distance": "1600",
        "ground_state": "良",
        "around": "左",
        "weather": "晴",
        "着順": finish,
        "枠番": str(min(horse_number, 8)),
        "馬番": str(horse_number),
        "馬名": f"fixture-{horse_number}",
        "horse_id": f"{horse_number:010d}",
        "sex": "牡",
        "age": "3",
        "騎手": "fixture-jockey",
        "jockey_id": "0123",
        "trainer": "fixture-trainer",
        "タイム": "1:34.5" if finish.isdigit() else "",
        "着差": margin,
        "通過順位": "1-1" if finish.isdigit() else "",
        "上がり3F": "34.5" if finish.isdigit() else "",
        "単勝": "2.5" if finish not in {"取", "除"} else "---",
        "人気": "1" if finish not in {"取", "除"} else "",
        "馬体重": "480",
        "馬体重増減": "+2",
        "date": race_date,
    }


def _write_csv(path: Path, rows: list[dict[str, str]], columns=RAW_COLUMNS) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_load_raw_is_bom_safe_and_preserves_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    _write_csv(path, [_raw_row("202405010101", "1", 1)])
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    raw = load_raw(path, expected_hash)

    assert tuple(raw.columns) == RAW_COLUMNS
    assert raw.loc[0, "raceid"] == "202405010101"
    assert raw.loc[0, "horse_id"] == "0000000001"
    assert raw.loc[0, "jockey_id"] == "0123"
    assert raw.loc[0, "race_class"] == "3歳以上\u00a01勝クラス"


def test_audit_raw_reports_actual_hash_even_without_expected_hash(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    _write_csv(path, [_raw_row("202405010101", "1", 1)])

    report = audit_raw(path)

    assert report["fingerprint"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_raw_rejects_fingerprint_and_ordered_schema_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    _write_csv(path, [_raw_row("202405010101", "1", 1)])

    with pytest.raises(FingerprintMismatchError):
        load_raw(path, "0" * 64)

    reordered = tmp_path / "reordered.csv"
    columns = list(RAW_COLUMNS)
    columns[0], columns[1] = columns[1], columns[0]
    _write_csv(reordered, [_raw_row("202405010101", "1", 1)], columns)
    with pytest.raises(SchemaMismatchError):
        load_raw(reordered)


def test_normalize_raw_retains_exceptional_outcomes_and_dead_heat() -> None:
    race_id = "202405010101"
    raw = pd.DataFrame(
        [
            _raw_row(race_id, "1", 1),
            _raw_row(race_id, "1", 2, margin="同着"),
            _raw_row(race_id, "中", 3),
            _raw_row(race_id, "失", 4),
            _raw_row(race_id, "2(降)", 5),
            _raw_row(race_id, "取", 6),
            _raw_row(race_id, "除", 7),
        ],
        columns=RAW_COLUMNS,
    )

    normalized = normalize_raw(raw)

    required = {
        "raceid",
        "date",
        "horse_id",
        "finish_position",
        "status",
        "started",
        "winner_label",
        "coherent_win_target",
        "pit_c_scoring_eligible",
        "venue",
    }
    assert required <= set(normalized.columns)
    assert normalized["status"].tolist() == [
        "finished",
        "finished",
        "did_not_finish",
        "disqualified",
        "demoted",
        "scratched",
        "excluded",
    ]
    assert normalized["finish_status"].equals(normalized["status"])

    winners = normalized[normalized["finish_position"] == 1]
    assert winners["winner_label"].tolist() == [1.0, 1.0]
    assert winners["coherent_win_target"].tolist() == [0.5, 0.5]
    assert winners["is_dead_heat"].tolist() == [True, True]
    assert normalized.loc[4, "finish_position"] == 2
    assert bool(normalized.loc[2, "started"]) is True
    assert bool(normalized.loc[3, "started"]) is True
    assert bool(normalized.loc[5, "started"]) is False
    assert bool(normalized.loc[6, "started"]) is False
    assert bool(normalized.loc[5, "history_update_eligible"]) is False
    assert normalized["pit_c_scoring_eligible"].tolist() == [False] * 7
    assert normalized.loc[0, "venue"] == "tokyo"


def test_unknown_surface_and_venue_are_preserved_not_dropped() -> None:
    raw = pd.DataFrame(
        [_raw_row("202499010101", "1", 1, course_type="砂")],
        columns=RAW_COLUMNS,
    )

    normalized = normalize_raw(raw)

    assert len(normalized) == 1
    assert normalized.loc[0, "venue"] == "unknown"
    assert normalized.loc[0, "venue_code"] == "99"
    assert normalized.loc[0, "surface"] == "unknown"
    assert normalized.loc[0, "surface_raw"] == "砂"


def test_unknown_finish_is_retained_without_guessing_start_status() -> None:
    raw = pd.DataFrame(
        [_raw_row("202405010101", "判定不能", 1)],
        columns=RAW_COLUMNS,
    )

    normalized = normalize_raw(raw)

    assert len(normalized) == 1
    assert normalized.loc[0, "finish_raw"] == "判定不能"
    assert normalized.loc[0, "status"] == "unknown"
    assert pd.isna(normalized.loc[0, "started"])
    assert pd.isna(normalized.loc[0, "winner_label"])


def test_normalize_raw_validates_source_native_id_types() -> None:
    invalid = _raw_row("202405010101", "1", 1)
    invalid["horse_id"] = "1"
    raw = pd.DataFrame([invalid], columns=RAW_COLUMNS)

    with pytest.raises(DataContractError, match="horse_id must be 10 ASCII digits"):
        normalize_raw(raw)


def test_normalize_raw_rejects_conflicting_race_metadata() -> None:
    race_id = "202405010101"
    raw = pd.DataFrame(
        [
            _raw_row(race_id, "1", 1, course_type="芝"),
            _raw_row(race_id, "2", 2, course_type="ダート"),
        ],
        columns=RAW_COLUMNS,
    )

    with pytest.raises(DataContractError, match="surface_raw conflicts"):
        normalize_raw(raw)


def test_normalize_raw_rejects_noncontiguous_race_groups() -> None:
    raw = pd.DataFrame(
        [
            _raw_row("202405010101", "1", 1),
            _raw_row("202405010102", "1", 1),
            _raw_row("202405010101", "2", 2),
        ],
        columns=RAW_COLUMNS,
    )

    with pytest.raises(DataContractError, match="not stored contiguously"):
        normalize_raw(raw)


def test_audit_apis_report_coverage_quality_and_verify_hash(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    rows = [
        _raw_row("202405010101", "1", 1),
        _raw_row("202405010101", "取", 2),
        _raw_row("202405010102", "1", 1, course_type="ダート"),
    ]
    _write_csv(path, rows)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    report = audit_raw(path, expected_hash)

    assert report["fingerprint"]["sha256"] == expected_hash
    assert report["fingerprint"]["has_utf8_bom"] is True
    assert report["row_count"] == 3
    assert report["race_count"] == 2
    assert report["coverage"]["race_count_by_year"] == {"2024": 2}
    assert report["coverage"]["runner_count_by_surface"] == {"ダート": 1, "芝": 2}
    assert report["outcomes"]["normalized_status_counts"] == {
        "finished": 2,
        "scratched": 1,
    }
    assert report["outcomes"]["declared_runner_count"] == 3
    assert report["outcomes"]["starter_count"] == 2
    assert report["outcomes"]["nonstarter_count"] == 1
    assert report["outcomes"]["unknown_start_count"] == 0
    assert report["outcomes"]["pit_c_scoring_ineligible_race_count"] == 1
    assert report["quality"]["duplicate_race_horse_id_count"] == 0


def test_manifest_has_frozen_26_column_contract_missing_ids_and_split() -> None:
    manifest_path = REPOSITORY_ROOT / "configs" / "data_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    schema_names = tuple(item["name"] for item in manifest["raw_file"]["schema"])
    missing_ids = expand_race_id_ranges(manifest["coverage"]["known_missing_race_id_ranges"])

    assert manifest["raw_file"]["column_count"] == 26
    assert schema_names == RAW_COLUMNS
    assert manifest["path_policy"]["absolute_path_in_manifest"] is False
    assert "/Users/" not in manifest_text
    assert len(missing_ids) == 146
    assert len(set(missing_ids)) == 146
    assert "201508010707" in missing_ids
    assert "201508010711" in missing_ids
    assert "201708040206" in missing_ids
    assert "202408070912" in missing_ids
    assert manifest["coverage"]["confirmed_shortfalls"] == {
        "2015": 2,
        "2017": 36,
        "2024": 108,
        "total": 146,
    }
    assert manifest["split"] == {
        "frozen_before_model_metrics": True,
        "warmup": {"start": "2013-01-01", "end": "2013-12-31"},
        "train": {"start": "2014-01-01", "end": "2021-12-31"},
        "model_validation": {"start": "2022-01-01", "end": "2022-12-31"},
        "calibration": {"start": "2023-01-01", "end": "2023-12-31"},
        "development": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "coverage_shortfall": 108,
        },
        "retrospective_test": {
            "start": "2025-01-01",
            "end": "2025-12-31",
            "untouched_claim_allowed": False,
        },
        "prospective_final": {"start": "2026-01-01", "end": None},
    }


def test_manifest_aware_audit_checks_declared_absence(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    _write_csv(path, [_raw_row("202405010101", "1", 1)])
    manifest = {
        "raw_file": {
            "row_count": 1,
            "race_count": 1,
            "column_count": 26,
            "date_min": "2024-01-06",
            "date_max": "2024-01-06",
        },
        "coverage": {
            "jra_official_race_counts": {"2024": 2},
            "known_missing_race_id_ranges": [
                {
                    "day_prefix": "2024050101",
                    "race_number_start": 2,
                    "race_number_end": 2,
                }
            ],
        },
    }

    report = audit_csv(path, manifest)
    verify_audit_against_manifest(report, manifest)

    comparison = report["coverage"]["jra_official_comparison"]["2024"]
    assert comparison == {
        "raw_race_count": 1,
        "jra_official_race_count": 2,
        "shortfall": 1,
    }
    assert report["coverage"]["known_missing_race_ids"] == {
        "expected_count": 1,
        "absent_count": 1,
        "unexpectedly_present": [],
    }


def test_resolve_raw_path_requires_injection_and_accepts_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_variable = "HORSE_PRED_TEST_RAW_CSV"
    monkeypatch.delenv(environment_variable, raising=False)
    with pytest.raises(DataContractError, match="raw CSV path is required"):
        resolve_raw_path(environment_variable=environment_variable)

    path = tmp_path / "fixture.csv"
    _write_csv(path, [_raw_row("202405010101", "1", 1)])
    monkeypatch.setenv(environment_variable, str(path))
    assert resolve_raw_path(environment_variable=environment_variable) == path


def test_manifest_json_is_valid_json() -> None:
    manifest_path = REPOSITORY_ROOT / "configs" / "data_manifest.json"
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["manifest_version"] == 1


def test_population_audit_uses_class_and_declared_nonstarter_scope() -> None:
    rows = [
        _raw_row("202405010101", "1", 1),
        _raw_row("202405010101", "2", 2),
        _raw_row("202405010102", "1", 1, course_type="ダート"),
        _raw_row("202405010102", "2", 2, course_type="ダート"),
        _raw_row("202405010103", "1", 1),
        _raw_row("202405010103", "取消", 2),
    ]
    raw = pd.DataFrame(rows)
    raw.loc[raw["raceid"].eq("202405010102"), "race_class"] = "障害4歳以上未勝利"
    races = build_race_population_table(normalize_raw(raw))
    audit = population_selection_audit(races)

    assert len(races) == 3
    assert not bool(races.set_index("race_id").loc["202405010102", "is_flat"])
    assert audit["development_2024"]["flat_races"] == 2
    assert audit["development_2024"]["scoring_eligible_races"] == 1
    assert audit["development_2024"]["nonstarter_races"] == 1
