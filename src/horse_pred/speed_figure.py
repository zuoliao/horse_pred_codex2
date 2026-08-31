"""PIT-safe prequential condition-adjusted speed history for SPEED-01."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import load_manifest, load_raw, normalize_raw, sha256_file, verify_raw_file
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.features import FeatureConfig, is_flat_race
from horse_pred.race_content import parse_result_time_seconds

SPEED_COLUMN = "speed__decay_90d__mean_condition_adjusted_time_residual"
SPEED_TRANSFORMATION_HASH = "55f5e4a5eb9c7bb12368bed9faa0a617b35ade5cbae918ffa8046be33b2d5c43"
_PRESENCE_COLUMN = "_speed_history_row_present"

_COURSE_SURFACE_LEVELS = tuple(
    f"{venue_code}_{surface}"
    for venue_code in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    for surface in ("芝", "ダート")
)
_DISTANCE_LEVELS = (
    1000,
    1150,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
    2300,
    2400,
    2500,
    2600,
    3000,
    3200,
    3400,
    3600,
)
_GOING_LEVELS = ("良", "稍重", "重", "不良")
_CLASS_LEVELS = ("new", "maiden", "1win", "2win", "3win", "open")
_AGE_LEVELS = ("2yo", "3yo", "3yo+", "4yo+")
_REFERENCES: dict[str, object] = {
    "course_surface": "05_芝",
    "distance_m": 1600,
    "going": "良",
    "class_tier": "maiden",
    "race_age_restriction": "3yo",
}


@dataclass(frozen=True)
class SpeedFigureSpec:
    ridge_alpha: float = 1.0
    min_prior_clean_races: int = 510
    observation_clip: float = 5.0
    decay_half_life_days: int = 90

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if self.min_prior_clean_races <= 0:
            raise ValueError("min_prior_clean_races must be positive")
        if not np.isfinite(self.observation_clip) or self.observation_clip <= 0:
            raise ValueError("observation_clip must be finite and positive")
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")

    def transformation_dict(self) -> dict[str, Any]:
        return {
            "feature_id": "SPEED-01",
            "source": "time_raw and completed-race conditions",
            "eligible_race": (
                "JRA flat with consistent known conditions, positive distance, a finite "
                "official winner clock, no demotion or disqualification, and no timed "
                "nonwinner faster than the winner"
            ),
            "expected_winner_clock": {
                "response": "winner elapsed seconds per 1000m",
                "model": "prequential ridge main effects",
                "ridge_alpha": self.ridge_alpha,
                "intercept_penalized": False,
                "scaling": "none; all explanatory columns are reference-coded dummies",
                "effects": {
                    "course_surface": {
                        "levels": "10 JRA venue codes crossed with turf/dirt",
                        "reference": "05_芝",
                    },
                    "distance_m": {
                        "levels": list(_DISTANCE_LEVELS),
                        "reference": 1600,
                    },
                    "going": {"levels": list(_GOING_LEVELS), "reference": "良"},
                    "class_tier": {
                        "levels": list(_CLASS_LEVELS),
                        "reference": "maiden",
                    },
                    "race_age_restriction": {
                        "levels": list(_AGE_LEVELS),
                        "reference": "3yo",
                    },
                },
                "design_dimension_including_intercept": 51,
                "cold_start_min_prior_clean_races": self.min_prior_clean_races,
                "same_date_batch": True,
                "unknown_condition": "no observation or expectation update",
                "forbidden": [
                    "full-period fit",
                    "fold future fit",
                    "same-date later-race result",
                    "day/course track variant",
                ],
            },
            "runner_observation": {
                "value": ("expected winner seconds per 1000m - runner elapsed seconds per 1000m"),
                "meaning": "positive is faster than the condition expectation",
                "eligible_runner": "started with finite parseable M:SS.t clock",
                "clip_seconds_per_1000m": [
                    -self.observation_clip,
                    self.observation_clip,
                ],
            },
            "history": {
                "half_life_days": self.decay_half_life_days,
                "same_date_batch": True,
                "missing_history": "NaN",
            },
            "output_column": SPEED_COLUMN,
        }


def _validate_frozen_transformation(spec: SpeedFigureSpec) -> str:
    actual = canonical_json_hash(spec.transformation_dict())
    if actual != SPEED_TRANSFORMATION_HASH:
        raise ValueError(f"SPEED-01 transformation differs from preregistration: {actual}")
    return actual


def _clean_class_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).replace("\u00a0", ""))


def _class_tier(value: object) -> str | None:
    text = _clean_class_text(value)
    if "新馬" in text:
        return "new"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text or "500万" in text:
        return "1win"
    if "2勝" in text or "1000万" in text:
        return "2win"
    if "3勝" in text or "1600万" in text:
        return "3win"
    if any(token in text.upper() for token in ("OPEN", "OP", "G1", "G2", "G3")) or "オープン" in text or "重賞" in text:
        return "open"
    return None


def _age_restriction(value: object) -> str | None:
    text = _clean_class_text(value)
    match = re.search(r"([234])歳", text)
    if not match:
        return None
    age = int(match.group(1))
    older = "以上" in text
    if age == 2 and not older:
        return "2yo"
    if age == 3:
        return "3yo+" if older else "3yo"
    if age == 4 and older:
        return "4yo+"
    return None


def _constant_value(race: pd.DataFrame, column: str) -> object | None:
    if column not in race:
        return None
    values = race[column].dropna().unique()
    return values[0] if len(values) == 1 else None


def condition_design_vector(race: pd.DataFrame) -> np.ndarray | None:
    """Encode one race's frozen 51-dimensional reference-coded main effects."""

    venue = _constant_value(race, "venue_code")
    surface = _constant_value(race, "course_type")
    distance_raw = _constant_value(race, "distance")
    going = _constant_value(race, "ground_state")
    race_class = _constant_value(race, "race_class")
    try:
        distance = int(float(distance_raw)) if distance_raw is not None else None
    except (TypeError, ValueError):
        return None
    conditions: tuple[tuple[str, object, tuple[object, ...]], ...] = (
        ("course_surface", f"{venue}_{surface}", _COURSE_SURFACE_LEVELS),
        ("distance_m", distance, _DISTANCE_LEVELS),
        ("going", going, _GOING_LEVELS),
        ("class_tier", _class_tier(race_class), _CLASS_LEVELS),
        ("race_age_restriction", _age_restriction(race_class), _AGE_LEVELS),
    )
    if any(value not in levels for _, value, levels in conditions):
        return None
    encoded = [1.0]
    for name, value, levels in conditions:
        encoded.extend(float(value == level) for level in levels if level != _REFERENCES[name])
    result = np.asarray(encoded, dtype=np.float64)
    if result.shape != (51,):
        raise AssertionError(f"SPEED-01 design dimension changed: {result.shape}")
    return result


def _clean_race_times(
    race: pd.DataFrame,
) -> tuple[float, pd.Series] | None:
    """Return winner sec/km and timed starter sec/km for a clean race."""

    required = {"status", "started", "finish_position", "time_raw", "distance"}
    if required.difference(race.columns):
        return None
    if race["status"].astype("string").isin(("demoted", "disqualified")).any():
        return None
    distance_raw = _constant_value(race, "distance")
    try:
        distance = float(distance_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(distance) or distance <= 0:
        return None
    times = race["time_raw"].map(parse_result_time_seconds)
    started = race["started"].eq(True).fillna(False)  # noqa: E712
    finish = pd.to_numeric(race["finish_position"], errors="coerce")
    timed = started & times.notna() & finish.notna()
    winners = timed & finish.eq(1)
    if not winners.any():
        return None
    winner_times = times.loc[winners].astype(float)
    if winner_times.max() - winner_times.min() > 1e-12:
        return None
    winner_time = float(winner_times.min())
    nonwinners = timed & finish.gt(1)
    if nonwinners.any() and times.loc[nonwinners].astype(float).lt(winner_time).any():
        return None
    return winner_time * 1000.0 / distance, times.loc[timed].astype(float) * 1000.0 / distance


def _solve_ridge(xtx: np.ndarray, xty: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(len(xty), dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def build_speed_history(
    normalized: pd.DataFrame,
    *,
    spec: SpeedFigureSpec | None = None,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Emit SPEED-01 history for every flat starter through 2024."""

    if through_year > 2024:
        raise ValueError("SPEED-01 must not generate or inspect 2025+ history")
    spec = spec or SpeedFigureSpec()
    _validate_frozen_transformation(spec)
    required = {
        "race_id",
        "race_date",
        "horse_id",
        "venue_code",
        "course_type",
        "distance",
        "ground_state",
        "race_class",
        "status",
        "started",
        "finish_position",
        "time_raw",
    }
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"normalized SPEED-01 input is missing columns: {missing}")
    frame = normalized.copy()
    dates = pd.to_datetime(frame["race_date"], errors="raise").dt.normalize()
    frame = frame.loc[dates.dt.year.le(through_year)].copy()
    frame["_speed_date"] = dates.loc[frame.index]
    frame["_source_position"] = np.arange(len(frame), dtype=np.int64)
    feature_config = FeatureConfig()
    flat_races: list[pd.DataFrame] = []
    for _, race in frame.groupby("race_id", sort=False):
        if is_flat_race(race, feature_config):
            flat_races.append(race)
    output_columns = ("race_id", "horse_id", "race_date", SPEED_COLUMN, _PRESENCE_COLUMN)
    if not flat_races:
        return pd.DataFrame(columns=output_columns)
    ordered = pd.concat(flat_races, ignore_index=True).sort_values(
        ["_speed_date", "race_id", "_source_position"], kind="stable"
    )
    if (
        ordered.loc[ordered["started"].eq(True).fillna(False)]
        .duplicated(  # noqa: E712
            ["race_id", "horse_id"]
        )
        .any()
    ):
        raise ValueError("SPEED-01 history contains duplicate starter keys")

    dimension = 51
    xtx = np.zeros((dimension, dimension), dtype=np.float64)
    xty = np.zeros(dimension, dtype=np.float64)
    clean_race_count = 0
    horse_state: dict[object, tuple[float, float, pd.Timestamp]] = {}
    decay_rate = log(2.0) / spec.decay_half_life_days
    emitted_parts: list[pd.DataFrame] = []

    for event_date, day in ordered.groupby("_speed_date", sort=True):
        event_date = pd.Timestamp(event_date).normalize()
        starters = day.loc[day["started"].eq(True).fillna(False)].copy()  # noqa: E712
        values: list[float] = []
        for horse_id in starters["horse_id"]:
            record = horse_state.get(horse_id)
            if record is None:
                values.append(np.nan)
                continue
            total, weight, state_date = record
            factor = exp(-decay_rate * (event_date - state_date).days)
            values.append(total * factor / (weight * factor) if weight > 0 else np.nan)
        starters[SPEED_COLUMN] = np.asarray(values, dtype=np.float64)
        starters[_PRESENCE_COLUMN] = True
        emitted_parts.append(starters)

        beta = _solve_ridge(xtx, xty, spec.ridge_alpha) if clean_race_count >= spec.min_prior_clean_races else None
        daily_observations: list[tuple[object, float]] = []
        daily_updates: list[tuple[np.ndarray, float]] = []
        for _, race in day.groupby("race_id", sort=False):
            design = condition_design_vector(race)
            clean = _clean_race_times(race)
            if design is None or clean is None:
                continue
            winner_clock, runner_clocks = clean
            if beta is not None:
                expected = float(design @ beta)
                for index, runner_clock in runner_clocks.items():
                    observation = float(
                        np.clip(expected - float(runner_clock), -spec.observation_clip, spec.observation_clip)
                    )
                    daily_observations.append((race.at[index, "horse_id"], observation))
            daily_updates.append((design, winner_clock))

        for horse_id, observation in daily_observations:
            record = horse_state.get(horse_id)
            if record is None:
                total = weight = 0.0
            else:
                total, weight, state_date = record
                factor = exp(-decay_rate * (event_date - state_date).days)
                total *= factor
                weight *= factor
            horse_state[horse_id] = (total + observation, weight + 1.0, event_date)
        for design, winner_clock in daily_updates:
            xtx += np.outer(design, design)
            xty += design * winner_clock
            clean_race_count += 1

    emitted = pd.concat(emitted_parts, ignore_index=True).sort_values("_source_position", kind="stable")
    return emitted.loc[:, list(output_columns)].reset_index(drop=True)


def build_speed_augmented_cache(
    baseline_cache_path: str | Path,
    history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append SPEED-01 while preserving the PACE-01 incumbent cache."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite SPEED-01 cache: {target}")
    frame, metadata = read_model_frame_cache(baseline_path)
    if any(column.startswith("speed__") for column in frame.columns):
        raise ValueError("baseline cache already contains SPEED-01")
    required = {"race_id", "horse_id", SPEED_COLUMN, _PRESENCE_COLUMN}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"SPEED-01 history is missing columns: {missing}")
    if history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("SPEED-01 history contains duplicate runner keys")

    base = frame.copy()
    base["_cache_order"] = np.arange(len(base), dtype=np.int64)
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = history.loc[:, ["race_id", "horse_id", SPEED_COLUMN, _PRESENCE_COLUMN]].copy()
    selected["race_id"] = selected["race_id"].astype("string")
    selected["horse_id"] = selected["horse_id"].astype("string")
    augmented = base.merge(
        selected, on=["race_id", "horse_id"], how="left", validate="one_to_one", sort=False
    ).sort_values("_cache_order", kind="stable")
    augmented = augmented.drop(columns="_cache_order").reset_index(drop=True)
    years = pd.to_datetime(augmented["race_date"], errors="raise").dt.year
    pre2025 = years.le(2024)
    if not augmented.loc[pre2025, _PRESENCE_COLUMN].eq(True).all():  # noqa: E712
        raise ValueError("SPEED-01 history does not cover every pre-2025 row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025+ SPEED-01 history rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[SPEED_COLUMN] = pd.to_numeric(augmented[SPEED_COLUMN], errors="coerce").astype("float32")
    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].reset_index(drop=True).equals(augmented.loc[:, old_features]):
        raise ValueError("an existing feature changed during SPEED-01 augmentation")
    feature_columns = [*old_features, SPEED_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["speed"] = [SPEED_COLUMN]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    write_json(
        sidecar_path,
        {
            **metadata,
            "row_count": len(augmented),
            "race_count": int(augmented["race_id"].nunique()),
            "feature_columns": feature_columns,
            "feature_groups_v1": groups,
            "speed": {
                "baseline_cache_sha256": sha256_file(baseline_path),
                "experiment_config_hash": canonical_json_hash(config),
                "transformation_hash": SPEED_TRANSFORMATION_HASH,
                "transformation": config["transformation"],
                "stored_dtype": "float32",
                "retrospective_2025_used": False,
            },
        },
    )
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(old_features),
        "candidate_feature_count": len(feature_columns),
        "old_feature_exact": True,
        "pre2025_nonmissing": int(augmented.loc[pre2025, SPEED_COLUMN].notna().sum()),
        "pre2025_missing": int(augmented.loc[pre2025, SPEED_COLUMN].isna().sum()),
        "retrospective_2025_feature_nonmissing": int(augmented.loc[~pre2025, SPEED_COLUMN].notna().sum()),
    }


def load_speed_config(path: str | Path) -> tuple[dict[str, Any], SpeedFigureSpec]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("candidate_column") != SPEED_COLUMN:
        raise ValueError("registered SPEED-01 column differs from code contract")
    raw = config.get("transformation")
    if not isinstance(raw, dict):
        raise ValueError("SPEED-01 config is missing transformation")
    expected = raw["expected_winner_clock"]
    runner = raw["runner_observation"]
    history = raw["history"]
    spec = SpeedFigureSpec(
        ridge_alpha=float(expected["ridge_alpha"]),
        min_prior_clean_races=int(expected["cold_start_min_prior_clean_races"]),
        observation_clip=float(runner["clip_seconds_per_1000m"][1]),
        decay_half_life_days=int(history["half_life_days"]),
    )
    if raw != spec.transformation_dict():
        raise ValueError("SPEED-01 config transformation differs from code contract")
    actual = _validate_frozen_transformation(spec)
    if config.get("transformation_hash") != actual:
        raise ValueError("SPEED-01 config transformation_hash is invalid")
    return config, spec


def build_speed_cache_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    baseline_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config, spec = load_speed_config(config_file)
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(2024)].copy()
    history = build_speed_history(normalize_raw(raw), spec=spec, through_year=2024)
    result = build_speed_augmented_cache(baseline_cache_path, history, output_path, config=config)
    result["raw_rows_after_pre_normalization_cutoff"] = len(raw)
    result["transformation_hash"] = SPEED_TRANSFORMATION_HASH
    return result
