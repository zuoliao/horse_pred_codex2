"""Configuration and chronological split helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DateInterval:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None

    def contains(self, values: pd.Series) -> pd.Series:
        mask = values.ge(self.start)
        if self.end is not None:
            mask &= values.le(self.end)
        return mask


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configured_intervals(split_config: Mapping[str, Any]) -> list[DateInterval]:
    intervals: list[DateInterval] = []
    for name in (
        "warmup",
        "train",
        "model_validation",
        "calibration",
        "development",
        "retrospective_test",
        "prospective_final",
    ):
        item = split_config[name]
        end = pd.Timestamp(item["end"]) if item.get("end") else None
        intervals.append(DateInterval(name=name, start=pd.Timestamp(item["start"]), end=end))
    _validate_intervals(intervals)
    return intervals


def assign_splits(dates: pd.Series, split_config: Mapping[str, Any]) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    output = pd.Series(pd.NA, index=dates.index, dtype="string")
    for interval in configured_intervals(split_config):
        mask = interval.contains(parsed)
        if output.loc[mask].notna().any():
            raise ValueError(f"Overlapping split interval: {interval.name}")
        output.loc[mask] = interval.name
    return output


def _validate_intervals(intervals: list[DateInterval]) -> None:
    previous_end: pd.Timestamp | None = None
    for interval in intervals:
        if interval.end is not None and interval.end < interval.start:
            raise ValueError(f"Split ends before it starts: {interval.name}")
        if previous_end is not None and interval.start <= previous_end:
            raise ValueError(f"Split intervals overlap at {interval.name}")
        previous_end = interval.end
