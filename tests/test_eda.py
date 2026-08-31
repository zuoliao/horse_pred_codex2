from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from horse_pred.data import RAW_COLUMNS
from horse_pred.eda import (
    EDA_MAX_DATE,
    MARKET_COLUMNS,
    build_historical_performance,
    build_market_oracle,
    build_race_table,
    load_eda_population,
    run_eda,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _row(race_id: str, horse_number: int, finish: str, race_date: str) -> dict[str, str]:
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
        "date": race_date,
    }


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_eda_loader_physically_excludes_rows_after_2022(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    rows = [
        _row("202205010101", 1, "1", "2022-01-01"),
        _row("202205010101", 2, "2", "2022-01-01"),
        _row("202305010101", 1, "1", "2023-01-01"),
        _row("202305010101", 2, "2", "2023-01-01"),
    ]
    _write(path, rows)

    frame = load_eda_population(path, max_date="2022-12-31")

    assert pd.to_datetime(frame["race_date"]).max() <= EDA_MAX_DATE
    assert set(frame["race_id"]) == {"202205010101"}


def test_eda_loader_rejects_any_alternative_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    _write(path, [_row("202205010101", 1, "1", "2022-01-01")])

    with pytest.raises(ValueError, match="requires max_date"):
        load_eda_population(path, max_date="2023-01-01")


def test_market_is_separate_from_performance_and_race_views(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    _write(
        path,
        [
            _row("202205010101", 1, "1", "2022-01-01"),
            _row("202205010101", 2, "2", "2022-01-01"),
        ],
    )
    normalized = load_eda_population(path, max_date="2022-12-31")

    market = build_market_oracle(normalized)
    historical = build_historical_performance(normalized)
    race = build_race_table(normalized)

    assert MARKET_COLUMNS.intersection(market.columns) == {"final_win_odds", "final_popularity"}
    assert not MARKET_COLUMNS.intersection(historical.columns)
    assert not MARKET_COLUMNS.intersection(race.columns)
    assert (historical["available_from"] > historical["race_date"]).all()


def test_completed_eda_artifact_can_resume_without_reloading_raw(tmp_path: Path) -> None:
    output = tmp_path / "eda"
    output.mkdir()
    expected = {"analysis_id": "fixture", "max_target_date": "2022-12-31"}
    (output / "manifest.json").write_text(json.dumps(expected), encoding="utf-8")

    result = run_eda(
        repo_root=REPOSITORY_ROOT,
        raw_path=tmp_path / "does-not-exist.csv",
        output_dir=output,
        max_date="2022-12-31",
        resume=True,
    )

    assert result == expected
