"""Exact control comparison for an opt-in feature cache.

Pandas pickle files must be deserialized as complete frames, but all potentially
large cell-wise comparisons below are performed in bounded row chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.data import sha256_file
from horse_pred.dataset_cache import read_model_frame_cache

RUNNER_IDENTITY_COLUMNS: tuple[str, ...] = (
    "race_id",
    "horse_id",
    "race_date",
    "split",
    "horse_number",
)
SURFACE_RATING_COLUMNS: tuple[str, ...] = (
    "surface_rating__horse_elo_pre",
    "surface_rating__horse_minus_field_mean_elo",
    "surface_rating__horse_elo_percentile",
)
RACE_VALUE_COLUMNS: tuple[str, ...] = (
    "race_value__decay_90d__mean_global_elo_surprise",
)
MODULAR_RATING_COLUMNS: tuple[str, ...] = (
    "modular_rating__score_pre",
    "modular_rating__raw_win_probability_pre",
    "modular_rating__global_starts_pre",
    "modular_rating__condition_starts_pre",
    "modular_rating__uncertainty_proxy_pre",
)
RACE_CONTENT_COLUMNS: tuple[str, ...] = (
    "race_content__decay_90d__mean_signed_time_gap_per_1000m",
)
OPPONENT_RECENT_COLUMNS: tuple[str, ...] = (
    "opponent_recent__decay_90d__mean_opponent_only_pre_elo",
)
SECTIONAL_RECENT_COLUMNS: tuple[str, ...] = (
    "sectional__decay_90d__mean_last_3f_speed_percentile",
)
PACE_RECENT_COLUMNS: tuple[str, ...] = (
    "pace__decay_90d__mean_early_position_percentile",
)


def _positional_mismatch_count(left: Sequence[str], right: Sequence[str]) -> int:
    missing = object()
    return sum(
        left_value != right_value
        for left_value, right_value in zip_longest(left, right, fillvalue=missing)
    )


def _identity_mismatches(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Sequence[str],
    *,
    chunk_size: int,
) -> tuple[int, int]:
    mismatch_cells = 0
    mismatch_rows = 0
    compared_rows = min(len(baseline), len(candidate))
    for start in range(0, compared_rows, chunk_size):
        stop = min(start + chunk_size, compared_rows)
        chunk_row_mismatch = np.zeros(stop - start, dtype=bool)
        for column in columns:
            left = baseline[column].iloc[start:stop].reset_index(drop=True)
            right = candidate[column].iloc[start:stop].reset_index(drop=True)
            both_missing = left.isna().to_numpy() & right.isna().to_numpy()
            equal = left.eq(right).fillna(False).to_numpy(dtype=bool) | both_missing
            mismatch = ~equal
            mismatch_cells += int(mismatch.sum())
            chunk_row_mismatch |= mismatch
        mismatch_rows += int(chunk_row_mismatch.sum())
    return mismatch_cells, mismatch_rows


def _feature_value_mismatches(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Sequence[str],
    *,
    chunk_size: int,
) -> dict[str, int | float | bool | None]:
    mismatch_count = 0
    mismatch_rows = 0
    nan_mismatch_count = 0
    infinite_mismatch_count = 0
    max_abs_diff = 0.0
    compared_rows = min(len(baseline), len(candidate))
    for start in range(0, compared_rows, chunk_size):
        stop = min(start + chunk_size, compared_rows)
        left = baseline.iloc[start:stop].loc[:, columns].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        right = candidate.iloc[start:stop].loc[:, columns].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        nan_mismatch = left_nan ^ right_nan
        equal = (left == right) | (left_nan & right_nan)
        mismatch = ~equal
        mismatch_count += int(mismatch.sum())
        mismatch_rows += int(mismatch.any(axis=1).sum())
        nan_mismatch_count += int(nan_mismatch.sum())

        finite_pair = np.isfinite(left) & np.isfinite(right)
        if finite_pair.any():
            max_abs_diff = max(
                max_abs_diff,
                float(np.max(np.abs(left[finite_pair] - right[finite_pair]))),
            )
        infinite_mismatch_count += int(
            (mismatch & ~nan_mismatch & ~finite_pair).sum()
        )

    return {
        "mismatch_count": mismatch_count,
        "mismatch_row_count": mismatch_rows,
        "nan_position_mismatch_count": nan_mismatch_count,
        "infinite_value_mismatch_count": infinite_mismatch_count,
        "max_abs_diff": None if infinite_mismatch_count else max_abs_diff,
        "max_abs_diff_is_infinite": bool(infinite_mismatch_count),
    }


def _compare_opt_in_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    expected_experimental_columns: Sequence[str],
    experimental_prefix: str,
    comparison_name: str,
    count_key: str,
    contract_key: str,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a cache differs only by a declared opt-in column contract."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    baseline_path = Path(baseline_cache_path).resolve()
    candidate_path = Path(candidate_cache_path).resolve()
    baseline, baseline_meta = read_model_frame_cache(baseline_path)
    candidate, candidate_meta = read_model_frame_cache(candidate_path)

    missing_identity = sorted(
        set(RUNNER_IDENTITY_COLUMNS).difference(baseline.columns)
        | set(RUNNER_IDENTITY_COLUMNS).difference(candidate.columns)
    )
    if missing_identity:
        raise ValueError(f"cache is missing runner identity columns: {missing_identity}")

    baseline_features = tuple(baseline_meta["feature_columns"])
    candidate_features = tuple(candidate_meta["feature_columns"])
    candidate_experimental_features = tuple(
        column for column in candidate_features if column.startswith(experimental_prefix)
    )
    candidate_control_features = tuple(
        column for column in candidate_features if not column.startswith(experimental_prefix)
    )
    schema_mismatch_count = _positional_mismatch_count(
        baseline_features, candidate_control_features
    )
    experimental_schema_mismatch_count = _positional_mismatch_count(
        expected_experimental_columns, candidate_experimental_features
    )
    baseline_count_mismatch = int(
        len(baseline_features) != expected_baseline_feature_count
    )
    fingerprint_mismatch = int(
        baseline_meta.get("data_fingerprint")
        != candidate_meta.get("data_fingerprint")
    )
    row_count_delta = abs(len(baseline) - len(candidate))

    identity_cell_mismatches, identity_row_mismatches = _identity_mismatches(
        baseline,
        candidate,
        RUNNER_IDENTITY_COLUMNS,
        chunk_size=chunk_size,
    )
    comparable_features = tuple(
        column
        for column in baseline_features
        if column in baseline.columns and column in candidate.columns
    )
    missing_value_columns = sorted(
        set(baseline_features).difference(comparable_features)
    )
    if schema_mismatch_count == 0 and not missing_value_columns:
        value_comparison = _feature_value_mismatches(
            baseline,
            candidate,
            comparable_features,
            chunk_size=chunk_size,
        )
    else:
        value_comparison = {
            "mismatch_count": 0,
            "mismatch_row_count": 0,
            "nan_position_mismatch_count": 0,
            "infinite_value_mismatch_count": 0,
            "max_abs_diff": None,
            "max_abs_diff_is_infinite": False,
        }

    mismatch_count = (
        row_count_delta
        + baseline_count_mismatch
        + fingerprint_mismatch
        + schema_mismatch_count
        + experimental_schema_mismatch_count
        + len(missing_value_columns)
        + identity_cell_mismatches
        + int(value_comparison["mismatch_count"])
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": comparison_name,
        "passed": mismatch_count == 0,
        "chunk_size": chunk_size,
        "cache_sha256": {
            "baseline": sha256_file(baseline_path),
            "candidate": sha256_file(candidate_path),
        },
        "data_fingerprint": {
            "baseline": baseline_meta.get("data_fingerprint"),
            "candidate": candidate_meta.get("data_fingerprint"),
            "matches": fingerprint_mismatch == 0,
        },
        "row_counts": {
            "baseline": len(baseline),
            "candidate": len(candidate),
            "compared": min(len(baseline), len(candidate)),
            "delta": row_count_delta,
        },
        "feature_counts": {
            "expected_baseline": expected_baseline_feature_count,
            "baseline": len(baseline_features),
            "candidate": len(candidate_features),
            "candidate_control": len(candidate_control_features),
            count_key: len(candidate_experimental_features),
        },
        "feature_schema": {
            "control_order_matches": schema_mismatch_count == 0,
            "control_positional_mismatch_count": schema_mismatch_count,
            f"{contract_key}_contract_matches": experimental_schema_mismatch_count == 0,
            f"{contract_key}_positional_mismatch_count": experimental_schema_mismatch_count,
            f"{contract_key}_columns": list(candidate_experimental_features),
            "missing_value_comparison_columns": missing_value_columns,
        },
        "runner_identity": {
            "columns": list(RUNNER_IDENTITY_COLUMNS),
            "mismatch_cell_count": identity_cell_mismatches,
            "mismatch_row_count": identity_row_mismatches,
        },
        "feature_values": value_comparison,
        "mismatch_count": mismatch_count,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def compare_surface_elo_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a surface-Elo cache differs only by its three opt-in columns."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=SURFACE_RATING_COLUMNS,
        experimental_prefix="surface_rating__",
        comparison_name="surface_conditioned_elo_cache_control",
        count_key="candidate_surface_rating",
        contract_key="surface",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_race_value_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that an expected-actual cache differs only by its one opt-in column."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=RACE_VALUE_COLUMNS,
        experimental_prefix="race_value__",
        comparison_name="expected_actual_race_value_cache_control",
        count_key="candidate_race_value",
        contract_key="race_value",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_modular_rating_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a rating-module cache differs only by its five frozen columns."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=MODULAR_RATING_COLUMNS,
        experimental_prefix="modular_rating__",
        comparison_name="modular_rating_cache_control",
        count_key="candidate_modular_rating",
        contract_key="modular_rating",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_race_content_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a PV-01 cache differs only by its frozen time-content column."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=RACE_CONTENT_COLUMNS,
        experimental_prefix="race_content__",
        comparison_name="race_content_time_cache_control",
        count_key="candidate_race_content_time",
        contract_key="race_content_time",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_opponent_recent_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that an OPP-RECENT cache adds only its frozen column."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=OPPONENT_RECENT_COLUMNS,
        experimental_prefix="opponent_recent__",
        comparison_name="opponent_recent_cache_control",
        count_key="candidate_opponent_recent",
        contract_key="opponent_recent",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_sectional_recent_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a SEC-3F cache adds only its frozen column."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=SECTIONAL_RECENT_COLUMNS,
        experimental_prefix="sectional__",
        comparison_name="sectional_recent_cache_control",
        count_key="candidate_sectional",
        contract_key="sectional",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )


def compare_pace_recent_cache_control(
    baseline_cache_path: str | Path,
    candidate_cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 10_000,
    expected_baseline_feature_count: int = 268,
) -> dict[str, Any]:
    """Verify that a PACE-01 cache adds only its frozen column."""

    return _compare_opt_in_cache_control(
        baseline_cache_path,
        candidate_cache_path,
        expected_experimental_columns=PACE_RECENT_COLUMNS,
        experimental_prefix="pace__",
        comparison_name="pace_recent_cache_control",
        count_key="candidate_pace",
        contract_key="pace",
        output_path=output_path,
        chunk_size=chunk_size,
        expected_baseline_feature_count=expected_baseline_feature_count,
    )
