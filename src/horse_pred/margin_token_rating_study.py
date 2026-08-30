"""PV-06: train-only raw-margin-token audit and 2022 rating gate.

The raw ``着差`` token is used only to dequantize pairs of distinct official
finish groups that share the same displayed 0.1-second clock.  The mapping is
frozen from 2014--2021 structure before any 2022 performance metric is read.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.artifacts import git_state, write_artifact_manifest, write_json
from horse_pred.config import canonical_json_hash
from horse_pred.data import (
    load_manifest,
    load_raw,
    normalize_raw,
    sha256_file,
    verify_raw_file,
)
from horse_pred.dataset_cache import read_model_frame_cache
from horse_pred.margin_rating_calibration_study import _calibrate_for_year
from horse_pred.margin_rating_study import _decision, _race_metric_table, _score
from horse_pred.rating import RatingEvent, RatingSpec, prepare_rating_events
from horse_pred.rating_study import _candidate_improvement, _primary_metrics
from horse_pred.uncertainty import paired_block_bootstrap


def _token(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _rating_spec(payload: dict[str, Any]) -> RatingSpec:
    values = dict(payload)
    mapping = values.get("time_margin_token_seconds")
    if isinstance(mapping, dict):
        values["time_margin_token_seconds"] = tuple(
            sorted((str(token), float(seconds)) for token, seconds in mapping.items())
        )
    return RatingSpec(**values)


def _rank_groups(event: RatingEvent) -> tuple[list[float], dict[float, list[int]]]:
    groups: dict[float, list[int]] = {}
    for index, (finish, result_time, status) in enumerate(
        zip(event.finishes, event.result_times_seconds, event.finish_statuses)
    ):
        if status != "finished" or not np.isfinite(finish) or not np.isfinite(result_time):
            continue
        groups.setdefault(float(finish), []).append(index)
    return sorted(groups), groups


def _boundary_token(event: RatingEvent, indices: Iterable[int]) -> tuple[str | None, bool]:
    tokens = [_token(event.margin_tokens[index]) for index in indices]
    carriers = [token for token in tokens if token not in {"", "同着"}]
    structure_valid = bool(
        len(carriers) == 1
        and all(token == "同着" for token in tokens if token != carriers[0])
    )
    return (carriers[0] if structure_valid else None), structure_valid


def derive_margin_token_audit(
    events: Iterable[RatingEvent],
    *,
    years: set[int],
    ordered_tokens: list[str],
    tie_token_seconds: dict[str, float],
) -> dict[str, Any]:
    """Audit raw token semantics without consulting any model metric."""

    token_order = {token: index for index, token in enumerate(ordered_tokens)}
    vocabulary: Counter[str] = Counter()
    edge_gaps: dict[str, list[float]] = {token: [] for token in ordered_tokens}
    edge_counts: Counter[str] = Counter()
    equal_clock_tokens: Counter[str] = Counter()
    equal_block_sizes: Counter[int] = Counter()
    race_count = 0
    clean_race_count = 0
    excluded_status_race_count = 0
    eligible_edge_count = 0
    recognized_edge_count = 0
    structure_error_count = 0
    group_clock_conflict_count = 0
    clock_inversion_count = 0
    equal_clock_edge_count = 0
    observed_orders: list[float] = []
    observed_gaps: list[float] = []

    for event in events:
        if event.race_date.year not in years:
            continue
        race_count += 1
        vocabulary.update(_token(value) for value in event.margin_tokens)
        if any(
            status in {"demoted", "disqualified", "unknown"}
            for status in event.finish_statuses
        ):
            excluded_status_race_count += 1
            continue
        clean_race_count += 1
        if not np.isfinite(event.distance_m) or event.distance_m <= 0:
            continue
        ranks, groups = _rank_groups(event)
        zero_run = 0
        for upper_rank, lower_rank in zip(ranks, ranks[1:]):
            eligible_edge_count += 1
            upper_times = {
                float(event.result_times_seconds[index]) for index in groups[upper_rank]
            }
            lower_times = {
                float(event.result_times_seconds[index]) for index in groups[lower_rank]
            }
            if len(upper_times) != 1 or len(lower_times) != 1:
                group_clock_conflict_count += 1
                zero_run = 0
                continue
            token, structure_valid = _boundary_token(event, groups[lower_rank])
            if not structure_valid or token is None:
                structure_error_count += 1
                zero_run = 0
                continue
            gap_seconds = next(iter(lower_times)) - next(iter(upper_times))
            if gap_seconds < 0:
                clock_inversion_count += 1
                zero_run = 0
                continue
            edge_counts[token] += 1
            if token in token_order:
                recognized_edge_count += 1
                edge_gaps[token].append(float(gap_seconds))
                observed_orders.append(float(token_order[token]))
                observed_gaps.append(float(gap_seconds))
            if gap_seconds == 0.0:
                equal_clock_edge_count += 1
                equal_clock_tokens[token] += 1
                zero_run += 1
            else:
                if zero_run:
                    equal_block_sizes[zero_run + 1] += 1
                    zero_run = 0
        if zero_run:
            equal_block_sizes[zero_run + 1] += 1

    token_statistics: dict[str, Any] = {}
    token_means: list[float] = []
    for token in ordered_tokens:
        gaps = np.asarray(edge_gaps[token], dtype=np.float64)
        count = int(len(gaps))
        mean = float(np.mean(gaps)) if count else np.nan
        if count:
            token_means.append(mean)
        token_statistics[token] = {
            "count": count,
            "zero_count": int(np.sum(gaps == 0.0)),
            "positive_count": int(np.sum(gaps > 0.0)),
            "negative_count": int(np.sum(gaps < 0.0)),
            "zero_fraction": float(np.mean(gaps == 0.0)) if count else None,
            "mean_displayed_gap_seconds": mean if count else None,
            "median_displayed_gap_seconds": float(np.median(gaps)) if count else None,
        }
    if observed_orders:
        order_ranks = pd.Series(observed_orders).rank(method="average")
        gap_ranks = pd.Series(observed_gaps).rank(method="average")
        edge_spearman = float(order_ranks.corr(gap_ranks))
    else:
        edge_spearman = np.nan
    adjacent_mean_reversals = sum(
        later <= earlier for earlier, later in zip(token_means, token_means[1:])
    )
    equal_clock_recognized = sum(equal_clock_tokens[token] for token in tie_token_seconds)
    mapping_payload = {
        "time_margin_token_seconds": dict(tie_token_seconds),
        "time_margin_equal_clock_block_cap_seconds": 0.08,
    }
    return {
        "years": sorted(years),
        "race_count": race_count,
        "clean_race_count": clean_race_count,
        "excluded_status_race_count": excluded_status_race_count,
        "vocabulary": dict(sorted(vocabulary.items())),
        "ordered_tokens": ordered_tokens,
        "token_statistics": token_statistics,
        "eligible_adjacent_distinct_rank_edges": eligible_edge_count,
        "recognized_edge_count": recognized_edge_count,
        "recognized_edge_fraction": (
            recognized_edge_count / eligible_edge_count if eligible_edge_count else 0.0
        ),
        "boundary_structure_error_count": structure_error_count,
        "group_clock_conflict_count": group_clock_conflict_count,
        "clock_inversion_count": clock_inversion_count,
        "token_order_displayed_gap_spearman": edge_spearman,
        "token_mean_adjacent_order_reversal_count": adjacent_mean_reversals,
        "equal_clock": {
            "edge_count": equal_clock_edge_count,
            "token_counts": dict(sorted(equal_clock_tokens.items())),
            "recognized_mapping_edge_count": equal_clock_recognized,
            "recognized_mapping_fraction": (
                equal_clock_recognized / equal_clock_edge_count
                if equal_clock_edge_count
                else 0.0
            ),
            "block_size_counts": {
                str(size): count for size, count in sorted(equal_block_sizes.items())
            },
        },
        "frozen_refinement": {
            "description": (
                "equal-clock adjacent edges use four ordered interior fifths of the "
                "unresolved 0.1-second tick; blocks are proportionally capped at 0.08s"
            ),
            **mapping_payload,
            "mapping_hash": canonical_json_hash(mapping_payload),
        },
    }


def _mapping_gate(audit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["mapping_gate"]
    expected_tie_tokens = set(config["token_refinement"]["time_margin_token_seconds"])
    observed_tie_tokens = set(audit["equal_clock"]["token_counts"])
    checks = {
        "recognized_edge_fraction": bool(
            audit["recognized_edge_fraction"]
            >= float(gate["recognized_edge_fraction_min"])
        ),
        "boundary_structure": audit["boundary_structure_error_count"] == 0,
        "group_clock_consistency": audit["group_clock_conflict_count"] == 0,
        "no_clean_clock_inversions": audit["clock_inversion_count"] == 0,
        "strict_token_mean_order": (
            audit["token_mean_adjacent_order_reversal_count"] == 0
        ),
        "equal_clock_token_set": observed_tie_tokens == expected_tie_tokens,
        "equal_clock_mapping_coverage": bool(
            audit["equal_clock"]["recognized_mapping_fraction"]
            >= float(gate["equal_clock_mapping_fraction_min"])
        ),
        "mapping_hash": (
            audit["frozen_refinement"]["mapping_hash"]
            == config["token_refinement"]["mapping_hash"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _load_events_through(
    raw_path: str | Path,
    *,
    expected_sha256: str,
    through_year: int,
) -> tuple[RatingEvent, ...]:
    raw = load_raw(raw_path, expected_sha256=expected_sha256)
    years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
    raw = raw.loc[years.le(through_year)].copy()
    return prepare_rating_events(normalize_raw(raw), through_year=through_year)


def audit_margin_tokens_from_raw(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the train-only audit without loading any post-train outcome fields."""

    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PV-06 token audit: {output}")
    output.mkdir(parents=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = json.loads(config_file.read_text(encoding="utf-8"))
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)
    years = set(int(year) for year in config["data"]["mapping_years"])
    events = _load_events_through(
        raw_path,
        expected_sha256=manifest["raw_file"]["sha256"],
        through_year=max(years),
    )
    token_config = config["token_refinement"]
    audit = derive_margin_token_audit(
        events,
        years=years,
        ordered_tokens=list(token_config["ordered_tokens"]),
        tie_token_seconds={
            str(key): float(value)
            for key, value in token_config["time_margin_token_seconds"].items()
        },
    )
    audit["mapping_gate"] = _mapping_gate(audit, config)
    audit["scope"] = {
        "maximum_normalized_outcome_year": max(years),
        "performance_metrics_opened": False,
        "retrospective_2025_used": False,
        "odds_used": False,
    }
    audit["reproducibility"] = {
        "git": git_state(root),
        "config_sha256": sha256_file(config_file),
        "data_fingerprint": manifest["raw_file"]["sha256"],
    }
    write_json(output / "token_audit_train.json", audit)
    write_json(output / "frozen_token_mapping.json", audit["frozen_refinement"])
    write_json(output / "config.json", config)
    write_artifact_manifest(output)
    return audit


def run_margin_token_rating_study(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    model_cache_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen PV-06 comparison and stop after the 2022 gate."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PV-06 output: {output}")
    output.mkdir(parents=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    cache_file = Path(model_cache_path)
    if not cache_file.is_absolute():
        cache_file = root / cache_file
    config = json.loads(config_file.read_text(encoding="utf-8"))
    manifest = load_manifest(root / "configs/data_manifest.json")
    verify_raw_file(raw_path, manifest)

    validation_year = int(config["data"]["validation_year"])
    calibration_year = int(config["data"]["calibration_year"])
    if validation_year != 2022 or calibration_year != 2021:
        raise ValueError("PV-06 must calibrate on 2021 and stop at the 2022 gate")
    events = _load_events_through(
        raw_path,
        expected_sha256=manifest["raw_file"]["sha256"],
        through_year=validation_year,
    )
    token_config = config["token_refinement"]
    audit = derive_margin_token_audit(
        events,
        years=set(int(year) for year in config["data"]["mapping_years"]),
        ordered_tokens=list(token_config["ordered_tokens"]),
        tie_token_seconds={
            str(key): float(value)
            for key, value in token_config["time_margin_token_seconds"].items()
        },
    )
    gate = _mapping_gate(audit, config)
    audit["mapping_gate"] = gate
    write_json(output / "token_audit_train.json", audit)
    write_json(output / "frozen_token_mapping.json", audit["frozen_refinement"])
    if not gate["passed"]:
        result = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "scope": {
                "maximum_normalized_outcome_year": validation_year,
                "performance_metrics_opened": False,
                "development_2024_opened": False,
                "retrospective_2025_used": False,
                "odds_used": False,
            },
            "mapping_gate": gate,
            "decision": "abort_mapping_not_defensible",
        }
        write_json(output / "metrics.json", result)
        write_json(output / "config.json", config)
        write_artifact_manifest(output)
        return result

    model_frame, _ = read_model_frame_cache(cache_file)
    model_frame["race_date"] = pd.to_datetime(
        model_frame["race_date"], errors="raise"
    )
    stage_frame = model_frame.loc[
        model_frame["race_date"].dt.year.le(validation_year)
    ].copy()
    control_spec = _rating_spec(config["control"])
    candidate_spec = _rating_spec(config["candidate"])
    frozen_base = (1500.0, 48.0, 200.0, 0.125)
    for spec in (control_spec, candidate_spec):
        actual = (
            spec.initial_rating,
            spec.k,
            spec.scale,
            spec.time_margin_tau_seconds_per_1000m,
        )
        if actual != frozen_base:
            raise ValueError("PV-06 must preserve the frozen PV-03 rating parameters")
    arms = {
        "control": _score(events, control_spec, stage_frame),
        "candidate": _score(events, candidate_spec, stage_frame),
    }
    raw_metrics = {
        arm: _primary_metrics(frame, [validation_year])
        for arm, frame in arms.items()
    }
    calibrated_frames: dict[str, pd.DataFrame] = {}
    temperatures: dict[str, float] = {}
    calibrated_metrics: dict[str, dict[str, float | int]] = {}
    for arm, frame in arms.items():
        calibrated, temperature, metrics = _calibrate_for_year(
            frame, calibration_year, validation_year
        )
        calibrated_frames[arm] = calibrated
        temperatures[arm] = temperature
        calibrated_metrics[arm] = metrics
    improvement = _candidate_improvement(
        calibrated_metrics["control"], calibrated_metrics["candidate"]
    )
    race_metrics = _race_metric_table(
        calibrated_frames, validation_year, "calibrated_probability"
    )
    uncertainty = config["uncertainty"]
    bootstrap = paired_block_bootstrap(
        race_metrics,
        comparisons=(("candidate", "control"),),
        n_resamples=int(uncertainty["bootstrap_resamples"]),
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        block_length_dates=int(uncertainty["block_length_dates"]),
    )
    decision = _decision(improvement, bootstrap, config)
    validation_audit = derive_margin_token_audit(
        events,
        years={validation_year},
        ordered_tokens=list(token_config["ordered_tokens"]),
        tie_token_seconds={
            str(key): float(value)
            for key, value in token_config["time_margin_token_seconds"].items()
        },
    )
    race_metrics.to_csv(
        output / "race_metrics_2022.csv.gz", index=False, compression="gzip"
    )
    for arm, frame in calibrated_frames.items():
        frame.to_pickle(output / f"predictions_2022_{arm}.pkl")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "scope": {
            "mapping_years": config["data"]["mapping_years"],
            "calibration_year": calibration_year,
            "validation_year": validation_year,
            "maximum_normalized_outcome_year": validation_year,
            "outcome_rows_used_2023": 0,
            "outcome_rows_used_2024": 0,
            "outcome_rows_used_2025": 0,
            "development_2024_opened": False,
            "retrospective_2025_used": False,
            "odds_used": False,
        },
        "mapping_gate": gate,
        "specs": {
            "control": control_spec.as_dict(),
            "candidate": candidate_spec.as_dict(),
        },
        "validation_2022": {
            "temperature_2021": temperatures,
            "raw_metrics_descriptive": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "candidate_improvement": improvement,
            "paired_bootstrap": bootstrap,
            "token_coverage": validation_audit,
            "decision": decision,
        },
        "reproducibility": {
            "git": git_state(root),
            "config_sha256": sha256_file(config_file),
            "model_cache_sha256": sha256_file(cache_file),
            "data_fingerprint": manifest["raw_file"]["sha256"],
            "seed": config["seed"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "config.json", config)
    write_artifact_manifest(output)
    return result
