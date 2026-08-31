"""PIT-safe recent opponent-only field-strength history.

The historical observation for horse ``i`` is the mean *pre-race* global Elo
of the other starters in that completed race.  It deliberately excludes the
horse itself and never uses an opponent's later result.  All rows on a date
are emitted before observations from that date update the 90-day state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import (
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_raw_file,
)
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.rating import RatingSpec, build_rating_history

OPPONENT_RECENT_COLUMN = (
    "opponent_recent__decay_90d__mean_opponent_only_pre_elo"
)
OPPONENT_RECENT_TRANSFORMATION_HASH = (
    "801e2a282629d4951af5d4c770a78f56f6e10c1311b76f1e5d6ba68cbdf7d940"
)
_PRESENCE_COLUMN = "_opponent_recent_history_row_present"


@dataclass(frozen=True)
class OpponentRecentSpec:
    """Frozen OPP-RECENT transformation parameters."""

    initial_rating: float = 1500.0
    k: float = 24.0
    scale: float = 400.0
    decay_half_life_days: int = 90

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_rating):
            raise ValueError("initial_rating must be finite")
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be finite and positive")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")

    def transformation_dict(self) -> dict[str, Any]:
        """Return the exact preregistered payload whose hash is frozen."""

        return {
            "feature_id": "OPP-RECENT",
            "rating": {
                "family": "pairwise_elo",
                "initial_rating": self.initial_rating,
                "k": self.k,
                "scale": self.scale,
                "pairwise_actual": "ordinal",
                "same_date_batch": True,
            },
            "eligible_race": "JRA flat",
            "eligible_runner": "started",
            "past_race_transform": {
                "value": (
                    "(sum_pre_race_rating - own_pre_race_rating)/(n_started-1)"
                ),
                "min_started": 2,
                "own_result_used": False,
            },
            "history": {
                "half_life_days": self.decay_half_life_days,
                "same_date_batch": True,
                "missing_history": "NaN",
            },
            "output_column": OPPONENT_RECENT_COLUMN,
        }

    def rating_spec(self) -> RatingSpec:
        """Return the exact global Elo already used by the corrected baseline."""

        return RatingSpec(
            family="pairwise_elo",
            initial_rating=self.initial_rating,
            k=self.k,
            scale=self.scale,
        )


def _validate_frozen_transformation(spec: OpponentRecentSpec) -> str:
    actual = canonical_json_hash(spec.transformation_dict())
    if actual != OPPONENT_RECENT_TRANSFORMATION_HASH:
        raise ValueError(
            "OPP-RECENT transformation differs from the preregistered hash: "
            f"{actual}"
        )
    return actual


def build_opponent_recent_history(
    normalized: pd.DataFrame,
    *,
    spec: OpponentRecentSpec | None = None,
    through_year: int = 2024,
) -> pd.DataFrame:
    """Emit the one-column opponent history for every flat-race starter.

    ``through_year`` is a hard research firewall.  The public cache builder
    additionally removes 2025+ raw rows before normalization.
    """

    if through_year > 2024:
        raise ValueError("OPP-RECENT must not generate or inspect 2025+ history")
    spec = spec or OpponentRecentSpec()
    _validate_frozen_transformation(spec)

    ratings = build_rating_history(
        normalized,
        spec.rating_spec(),
        through_year=through_year,
    )
    if ratings.empty:
        return pd.DataFrame(
            columns=(
                "race_id",
                "horse_id",
                "race_date",
                OPPONENT_RECENT_COLUMN,
                _PRESENCE_COLUMN,
            )
        )
    if ratings.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("rating history contains duplicate runner keys")

    ordered = ratings.sort_values(
        ["race_date", "race_id", "source_position"], kind="stable"
    ).copy()
    race_group = ordered.groupby("race_id", sort=False)["global_state_pre"]
    field_count = race_group.transform("size").astype("int64")
    field_sum = race_group.transform("sum").astype("float64")
    opponent_observation = (
        field_sum - ordered["global_state_pre"].astype("float64")
    ) / (field_count - 1)
    opponent_observation = opponent_observation.where(field_count.ge(2))

    # horse -> (decayed total, decayed weight, state date)
    state: dict[object, tuple[float, float, pd.Timestamp]] = {}
    emitted_values: list[float] = []
    decay_rate = log(2.0) / spec.decay_half_life_days

    for event_date, day in ordered.groupby("race_date", sort=True):
        event_date = pd.Timestamp(event_date).normalize()
        for horse_id in day["horse_id"]:
            record = state.get(horse_id)
            if record is None:
                emitted_values.append(np.nan)
                continue
            total, weight, state_date = record
            factor = exp(-decay_rate * (event_date - state_date).days)
            total *= factor
            weight *= factor
            emitted_values.append(total / weight if weight > 0.0 else np.nan)

        # No observation from event_date is visible to another race that day.
        for index in day.index:
            observation = float(opponent_observation.loc[index])
            if not np.isfinite(observation):
                continue
            horse_id = ordered.at[index, "horse_id"]
            record = state.get(horse_id)
            if record is None:
                total = weight = 0.0
            else:
                total, weight, state_date = record
                factor = exp(-decay_rate * (event_date - state_date).days)
                total *= factor
                weight *= factor
            state[horse_id] = (total + observation, weight + 1.0, event_date)

    ordered[OPPONENT_RECENT_COLUMN] = np.asarray(
        emitted_values, dtype=np.float64
    )
    ordered[_PRESENCE_COLUMN] = True
    result = ordered.sort_values("source_position", kind="stable")
    return result.loc[
        :,
        [
            "race_id",
            "horse_id",
            "race_date",
            OPPONENT_RECENT_COLUMN,
            _PRESENCE_COLUMN,
        ],
    ].reset_index(drop=True)


def build_opponent_recent_augmented_cache(
    baseline_cache_path: str | Path,
    opponent_history: pd.DataFrame,
    output_path: str | Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Join one frozen opponent feature while preserving the baseline exactly."""

    baseline_path = Path(baseline_cache_path).resolve()
    target = Path(output_path).resolve()
    sidecar_path = target.with_suffix(f"{target.suffix}.meta.json")
    if target.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite opponent cache: {target}")

    frame, metadata = read_model_frame_cache(baseline_path)
    if any(column.startswith("opponent_recent__") for column in frame.columns):
        raise ValueError("baseline cache already contains an opponent-recent column")
    required = {
        "race_id",
        "horse_id",
        OPPONENT_RECENT_COLUMN,
        _PRESENCE_COLUMN,
    }
    missing = sorted(required.difference(opponent_history.columns))
    if missing:
        raise ValueError(f"opponent history is missing columns: {missing}")
    if opponent_history.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("opponent history contains duplicate runner keys")
    if "race_date" not in frame:
        raise ValueError("baseline cache is missing race_date")

    base = frame.copy()
    base["_cache_order"] = np.arange(len(base), dtype=np.int64)
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    selected = opponent_history.loc[
        :,
        ["race_id", "horse_id", OPPONENT_RECENT_COLUMN, _PRESENCE_COLUMN],
    ].copy()
    selected["race_id"] = selected["race_id"].astype("string")
    selected["horse_id"] = selected["horse_id"].astype("string")
    augmented = base.merge(
        selected,
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_cache_order", kind="stable")
    augmented = augmented.drop(columns="_cache_order").reset_index(drop=True)

    years = pd.to_datetime(augmented["race_date"], errors="raise").dt.year
    pre2025 = years.le(2024)
    if not augmented.loc[pre2025, _PRESENCE_COLUMN].eq(True).all():  # noqa: E712
        raise ValueError("opponent history does not cover every pre-2025 model row")
    if augmented.loc[~pre2025, _PRESENCE_COLUMN].notna().any():
        raise ValueError("2025+ opponent-history rows must remain unavailable")
    augmented = augmented.drop(columns=_PRESENCE_COLUMN)
    augmented[OPPONENT_RECENT_COLUMN] = pd.to_numeric(
        augmented[OPPONENT_RECENT_COLUMN], errors="coerce"
    ).astype("float32")

    old_features = list(metadata["feature_columns"])
    if not frame.loc[:, old_features].reset_index(drop=True).equals(
        augmented.loc[:, old_features]
    ):
        raise ValueError("an existing baseline feature changed during cache augmentation")
    feature_columns = [*old_features, OPPONENT_RECENT_COLUMN]
    groups = dict(metadata.get("feature_groups_v1", {}))
    groups["opponent_recent"] = [OPPONENT_RECENT_COLUMN]

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    augmented.to_pickle(temporary)
    temporary.replace(target)
    sidecar = {
        **metadata,
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "feature_columns": feature_columns,
        "feature_groups_v1": groups,
        "opponent_recent": {
            "baseline_cache_sha256": sha256_file(baseline_path),
            "experiment_config_hash": canonical_json_hash(config),
            "transformation_hash": OPPONENT_RECENT_TRANSFORMATION_HASH,
            "transformation": config["transformation"],
            "stored_dtype": "float32",
            "retrospective_2025_used": False,
        },
    }
    write_json(sidecar_path, sidecar)
    return {
        "schema_version": 1,
        "output": str(target),
        "row_count": len(augmented),
        "race_count": int(augmented["race_id"].nunique()),
        "baseline_feature_count": len(old_features),
        "candidate_feature_count": len(feature_columns),
        "old_feature_exact": True,
        "pre2025_nonmissing": int(
            augmented.loc[pre2025, OPPONENT_RECENT_COLUMN].notna().sum()
        ),
        "pre2025_missing": int(
            augmented.loc[pre2025, OPPONENT_RECENT_COLUMN].isna().sum()
        ),
        "retrospective_2025_feature_nonmissing": int(
            augmented.loc[~pre2025, OPPONENT_RECENT_COLUMN].notna().sum()
        ),
    }


def load_opponent_recent_config(
    path: str | Path,
) -> tuple[dict[str, Any], OpponentRecentSpec]:
    """Load and validate the frozen OPP-RECENT preregistration."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("candidate_column") != OPPONENT_RECENT_COLUMN:
        raise ValueError("registered opponent-recent column differs from code contract")
    raw = config.get("transformation")
    if not isinstance(raw, dict):
        raise ValueError("OPP-RECENT config is missing transformation")
    rating = raw.get("rating", {})
    history = raw.get("history", {})
    spec = OpponentRecentSpec(
        initial_rating=float(rating["initial_rating"]),
        k=float(rating["k"]),
        scale=float(rating["scale"]),
        decay_half_life_days=int(history["half_life_days"]),
    )
    if raw != spec.transformation_dict():
        raise ValueError("OPP-RECENT config transformation differs from code contract")
    actual = _validate_frozen_transformation(spec)
    if config.get("transformation_hash") != actual:
        raise ValueError("OPP-RECENT config transformation_hash is invalid")
    return config, spec


def build_opponent_recent_cache_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    baseline_cache_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build OPP-RECENT after removing 2025+ before normalization."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config, spec = load_opponent_recent_config(config_file)
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(2024)].copy()
    history = build_opponent_recent_history(
        normalize_raw(raw), spec=spec, through_year=2024
    )
    result = build_opponent_recent_augmented_cache(
        baseline_cache_path,
        history,
        output_path,
        config=config,
    )
    result["raw_rows_after_pre_normalization_cutoff"] = len(raw)
    result["transformation_hash"] = OPPONENT_RECENT_TRANSFORMATION_HASH
    return result
