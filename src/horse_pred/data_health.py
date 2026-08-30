"""Race-population audits for flat eligibility and non-starter selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_race_population_table(normalized: pd.DataFrame) -> pd.DataFrame:
    """Collapse normalized runner rows to auditable race-level selection data."""

    required = {
        "race_id",
        "race_date",
        "venue",
        "course_type",
        "race_class",
        "distance",
        "status",
        "started",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"normalized frame is missing columns: {missing}")
    work = normalized.copy(deep=False)
    group = work.groupby("race_id", sort=False, observed=True)
    for column in ("race_date", "venue", "course_type", "race_class", "distance"):
        if group[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"race-level column is nonconstant: {column}")
    races = group.agg(
        race_date=("race_date", "first"),
        venue=("venue", "first"),
        course_type=("course_type", "first"),
        race_class=("race_class", "first"),
        distance=("distance", "first"),
        declared_field_size=("race_id", "size"),
        starter_count=("started", lambda values: int(values.eq(True).sum())),  # noqa: E712
        scratch_count=("status", lambda values: int(values.eq("scratched").sum())),
        exclusion_count=("status", lambda values: int(values.eq("excluded").sum())),
        unknown_start_count=("started", lambda values: int(values.isna().sum())),
    ).reset_index()
    races["race_date"] = pd.to_datetime(races["race_date"], errors="raise")
    races["year"] = races["race_date"].dt.year.astype(int)
    races["month"] = races["race_date"].dt.month.astype(int)
    races["venue_code"] = races["race_id"].astype("string").str[4:6]
    races["is_flat"] = (
        races["course_type"].isin(["芝", "ダート"])
        & ~races["race_class"].astype("string").str.contains("障害", na=False)
    )
    races["has_nonstarter"] = races[
        ["scratch_count", "exclusion_count", "unknown_start_count"]
    ].sum(axis=1).gt(0)
    races["pit_c_scoring_eligible"] = races["is_flat"] & ~races["has_nonstarter"]
    races["distance_band"] = pd.cut(
        pd.to_numeric(races["distance"], errors="coerce"),
        bins=[0, 1399, 1799, 2199, np.inf],
        labels=["le_1399", "1400_1799", "1800_2199", "ge_2200"],
        include_lowest=True,
    ).astype("string")
    races["declared_field_size_band"] = pd.cut(
        races["declared_field_size"],
        bins=[0, 9, 12, 15, np.inf],
        labels=["le_9", "10_12", "13_15", "ge_16"],
        include_lowest=True,
    ).astype("string")
    races["class_group"] = races["race_class"].map(_class_group).astype("string")
    return races


def population_selection_audit(races: pd.DataFrame) -> dict[str, Any]:
    """Summarize the conservative no-nonstarter selection, especially 2024."""

    required = {"is_flat", "pit_c_scoring_eligible", "year", "has_nonstarter"}
    missing = sorted(required.difference(races.columns))
    if missing:
        raise ValueError(f"race table is missing columns: {missing}")
    flat = races.loc[races["is_flat"]].copy()
    development = flat.loc[flat["year"].eq(2024)].copy()
    return {
        "schema_version": 1,
        "flat_definition": "course_type in {芝,ダート} and race_class does not contain 障害",
        "all_years": _population_counts(flat),
        "development_2024": {
            **_population_counts(development),
            "selection_by_condition": {
                dimension: _selection_rates(development, dimension)
                for dimension in (
                    "venue",
                    "month",
                    "course_type",
                    "distance_band",
                    "class_group",
                    "declared_field_size_band",
                )
            },
        },
    }


def _population_counts(frame: pd.DataFrame) -> dict[str, Any]:
    ineligible = frame.loc[frame["has_nonstarter"]]
    return {
        "flat_races": len(frame),
        "scoring_eligible_races": int(frame["pit_c_scoring_eligible"].sum()),
        "nonstarter_races": len(ineligible),
        "nonstarter_race_rate": float(frame["has_nonstarter"].mean()),
        "scratch_races": int(ineligible["scratch_count"].gt(0).sum()),
        "exclusion_races": int(ineligible["exclusion_count"].gt(0).sum()),
        "both_scratch_and_exclusion_races": int(
            (ineligible["scratch_count"].gt(0) & ineligible["exclusion_count"].gt(0)).sum()
        ),
        "scratch_runners": int(frame["scratch_count"].sum()),
        "excluded_runners": int(frame["exclusion_count"].sum()),
        "declared_field_size_mean_scored": float(
            frame.loc[frame["pit_c_scoring_eligible"], "declared_field_size"].mean()
        ),
        "declared_field_size_mean_ineligible": float(ineligible["declared_field_size"].mean()),
        "starter_count_mean_ineligible": float(ineligible["starter_count"].mean()),
    }


def _selection_rates(frame: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    rows = []
    for level, group in frame.groupby(dimension, observed=True, dropna=False):
        ineligible = int(group["has_nonstarter"].sum())
        rows.append(
            {
                "level": str(level),
                "race_count": len(group),
                "nonstarter_races": ineligible,
                "nonstarter_race_rate": ineligible / len(group),
            }
        )
    return sorted(rows, key=lambda row: row["level"])


def _class_group(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).replace(" ", "")
    if "障害" in text:
        return "jump"
    if "新馬" in text:
        return "newcomer"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text or "500万" in text:
        return "class_1"
    if "2勝" in text or "1000万" in text:
        return "class_2"
    if "3勝" in text or "1600万" in text:
        return "class_3"
    return "open"
