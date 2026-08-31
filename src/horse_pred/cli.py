"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from horse_pred.artifacts import write_json
from horse_pred.cached_experiment import run_cached_experiment
from horse_pred.data import (
    audit_csv,
    load_manifest,
    load_raw,
    normalize_raw,
    resolve_raw_path,
    verify_audit_against_manifest,
    verify_raw_file,
)
from horse_pred.data_health import (
    build_race_population_table,
    population_selection_audit,
)
from horse_pred.diagnostics import run_baseline_diagnostics
from horse_pred.ensemble_study import run_ensemble_study
from horse_pred.features import FeatureConfig
from horse_pred.graded_rank_study import run_graded_rank_study
from horse_pred.hpo_study import run_hpo_study
from horse_pred.live_data_archive import (
    archive_jvlink_batch,
    load_live_data_archive_config,
)
from horse_pred.margin_rating_cache import (
    build_margin_rating_cache_from_raw,
    build_margin_rating_delta_cache,
)
from horse_pred.margin_rating_calibration_study import (
    run_margin_rating_calibration_study,
)
from horse_pred.margin_rating_study import run_margin_rating_study
from horse_pred.margin_token_rating_study import (
    audit_margin_tokens_from_raw,
    run_margin_token_rating_study,
)
from horse_pred.opponent_recent import build_opponent_recent_cache_from_raw
from horse_pred.pace_pressure import build_pace_pressure_cache_from_config
from horse_pred.pace_recent import build_pace_recent_cache_from_raw
from horse_pred.pipeline import run_mvp
from horse_pred.race_content import (
    build_race_content_augmented_cache,
    build_race_content_history,
    load_race_content_config,
)
from horse_pred.rating_study import run_rating_study
from horse_pred.rolling_evaluation import run_rolling_evaluation
from horse_pred.sectional_recent import build_sectional_recent_cache_from_raw
from horse_pred.shimba_filter_study import run_shimba_filter_study
from horse_pred.uncertainty import run_uncertainty_analysis


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="horse-pred")
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="verify and summarize the approved raw CSV")
    audit.add_argument("--raw-path", type=Path)
    audit.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.json"))
    audit.add_argument("--skip-sha256", action="store_true")

    population = commands.add_parser(
        "audit-population", help="audit flat/jump and non-starter race selection"
    )
    population.add_argument("--raw-path", type=Path)
    population.add_argument(
        "--manifest", type=Path, default=Path("configs/data_manifest.json")
    )
    population.add_argument("--output", type=Path, required=True)

    live_archive = commands.add_parser(
        "archive-jvlink-batch",
        help="validate and append a private normalized JV-Link batch without network access",
    )
    live_archive.add_argument("--input", type=Path, required=True)
    live_archive.add_argument("--archive-root", type=Path, required=True)
    live_archive.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_data/jvlink_archive.json"),
    )
    live_archive.add_argument("--repo-root", type=Path, default=Path.cwd())

    run = commands.add_parser(
        "run-mvp", help="run PIT features, both LightGBM models, calibration, and evaluation"
    )
    run.add_argument("--raw-path", type=Path)
    run.add_argument("--output", type=Path, default=Path("artifacts/mvp_baseline"))
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.json"))
    run.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    run.add_argument(
        "--binary-config", type=Path, default=Path("configs/exp_001_binary.json")
    )
    run.add_argument(
        "--ranker-config", type=Path, default=Path("configs/exp_002_lambdarank.json")
    )
    run.add_argument(
        "--include-retrospective-test",
        action="store_true",
        help="explicitly inspect the non-sealed 2025 retrospective split",
    )
    run.add_argument(
        "--model-frame-cache",
        type=Path,
        help="optional ignored local pickle cache for later ablation/diagnostic runs",
    )
    run.add_argument(
        "--surface-conditioned-elo",
        action="store_true",
        help="materialize the opt-in turf/dirt-specific Elo feature group",
    )
    run.add_argument(
        "--expected-actual-race-value",
        action="store_true",
        help="materialize the opt-in 90-day Elo expected-vs-actual race-value feature",
    )
    uncertainty = commands.add_parser(
        "analyze-uncertainty", help="run the fixed 2024 paired block bootstrap"
    )
    uncertainty.add_argument("--predictions", type=Path, required=True)
    uncertainty.add_argument("--output", type=Path, required=True)
    uncertainty.add_argument("--resamples", type=int, default=10_000)
    uncertainty.add_argument("--seed", type=int, default=20240830)

    cached = commands.add_parser(
        "run-cached-experiment",
        help="fit and evaluate a registered 2024-only experiment from a PIT frame cache",
    )
    cached.add_argument("--cache", type=Path, required=True)
    cached.add_argument("--config", type=Path, required=True)
    cached.add_argument("--output", type=Path, required=True)
    cached.add_argument("--repo-root", type=Path, default=Path.cwd())
    rolling = commands.add_parser(
        "run-rolling-evaluation",
        help="run the registered 2020-2023 rolling-origin no-odds evaluation",
    )
    rolling.add_argument("--cache", type=Path, required=True)
    rolling.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/eval_roll_001_current_best.json"),
    )
    rolling.add_argument("--output", type=Path, required=True)
    rolling.add_argument("--repo-root", type=Path, default=Path.cwd())
    hpo = commands.add_parser(
        "run-hpo-study",
        help="run one preregistered 2020-2022 LightGBM HPO selection and 2023 confirmation",
    )
    hpo.add_argument("--cache", type=Path, required=True)
    hpo.add_argument("--config", type=Path, required=True)
    hpo.add_argument("--output", type=Path, required=True)
    hpo.add_argument("--repo-root", type=Path, default=Path.cwd())
    ensemble = commands.add_parser(
        "run-ensemble-study",
        help="run the preregistered fixed 50:50 rolling probability ensemble",
    )
    ensemble.add_argument("--config", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--repo-root", type=Path, default=Path.cwd())
    shimba_filter = commands.add_parser(
        "run-shimba-filter-study",
        help="run the preregistered Binary-only new-horse fit-population ablation",
    )
    shimba_filter.add_argument("--cache", type=Path, required=True)
    shimba_filter.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/shimba_filter_001_rolling.json"),
    )
    shimba_filter.add_argument("--output", type=Path, required=True)
    shimba_filter.add_argument("--repo-root", type=Path, default=Path.cwd())
    diagnostics = commands.add_parser(
        "diagnose-baseline",
        help="run 2024-only importance, permutation, and conditional diagnostics",
    )
    diagnostics.add_argument("--cache", type=Path, required=True)
    diagnostics.add_argument("--baseline", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--repo-root", type=Path, default=Path.cwd())
    diagnostics.add_argument("--permutation-repeats", type=int, default=5)
    diagnostics.add_argument("--seed", type=int, default=20240830)
    rating = commands.add_parser(
        "run-rating-study", help="run the preregistered standalone rating R0-R5 study"
    )
    rating.add_argument("--raw-path", type=Path, required=True)
    rating.add_argument("--cache", type=Path, required=True)
    rating.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rating/rating_module_r0_r6.json"),
    )
    rating.add_argument("--output", type=Path, required=True)
    rating.add_argument("--repo-root", type=Path, default=Path.cwd())
    race_content = commands.add_parser(
        "build-race-content-cache",
        help="build the preregistered PV-01 time-content cache without 2025",
    )
    race_content.add_argument("--raw-path", type=Path, required=True)
    race_content.add_argument("--baseline-cache", type=Path, required=True)
    race_content.add_argument("--output", type=Path, required=True)
    race_content.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_001_race_content_time.json"),
    )
    race_content.add_argument("--repo-root", type=Path, default=Path.cwd())
    opponent_recent = commands.add_parser(
        "build-opponent-recent-cache",
        help="build the preregistered one-column OPP-RECENT cache without 2025",
    )
    opponent_recent.add_argument("--raw-path", type=Path, required=True)
    opponent_recent.add_argument("--baseline-cache", type=Path, required=True)
    opponent_recent.add_argument("--output", type=Path, required=True)
    opponent_recent.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features/opp_recent_001.json"),
    )
    opponent_recent.add_argument("--repo-root", type=Path, default=Path.cwd())
    sectional_recent = commands.add_parser(
        "build-sectional-recent-cache",
        help="build the preregistered one-column SEC-3F cache without 2025",
    )
    sectional_recent.add_argument("--raw-path", type=Path, required=True)
    sectional_recent.add_argument("--baseline-cache", type=Path, required=True)
    sectional_recent.add_argument("--output", type=Path, required=True)
    sectional_recent.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features/sec_3f_001.json"),
    )
    sectional_recent.add_argument("--repo-root", type=Path, default=Path.cwd())
    pace_recent = commands.add_parser(
        "build-pace-recent-cache",
        help="build the preregistered one-column PACE-01 cache without 2025",
    )
    pace_recent.add_argument("--raw-path", type=Path, required=True)
    pace_recent.add_argument("--baseline-cache", type=Path, required=True)
    pace_recent.add_argument("--output", type=Path, required=True)
    pace_recent.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features/pace_01_early_position.json"),
    )
    pace_recent.add_argument("--repo-root", type=Path, default=Path.cwd())
    pace_pressure = commands.add_parser(
        "build-pace-pressure-cache",
        help="build the preregistered one-column PACE-02 field-pressure cache",
    )
    pace_pressure.add_argument("--input-cache", type=Path, required=True)
    pace_pressure.add_argument("--output", type=Path, required=True)
    pace_pressure.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features/pace_02_field_pressure.json"),
    )
    pace_pressure.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_rating = commands.add_parser(
        "run-margin-rating-study",
        help="run the preregistered PV-02 time-margin standalone rating study",
    )
    margin_rating.add_argument("--raw-path", type=Path, required=True)
    margin_rating.add_argument("--cache", type=Path, required=True)
    margin_rating.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_002_margin_aware_rating.json"),
    )
    margin_rating.add_argument("--output", type=Path, required=True)
    margin_rating.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_calibration = commands.add_parser(
        "run-margin-rating-calibration-study",
        help="run the preregistered PV-03 temporal rating calibration study",
    )
    margin_calibration.add_argument("--raw-path", type=Path, required=True)
    margin_calibration.add_argument("--cache", type=Path, required=True)
    margin_calibration.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/performance/pv_003_margin_rating_temporal_calibration.json"
        ),
    )
    margin_calibration.add_argument("--output", type=Path, required=True)
    margin_calibration.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_token_audit = commands.add_parser(
        "audit-margin-tokens",
        help="run the preregistered PV-06 train-only raw margin-token audit",
    )
    margin_token_audit.add_argument("--raw-path", type=Path, required=True)
    margin_token_audit.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_006_margin_token_refinement.json"),
    )
    margin_token_audit.add_argument("--output", type=Path, required=True)
    margin_token_audit.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_token_study = commands.add_parser(
        "run-margin-token-rating-study",
        help="run the preregistered PV-06 equal-clock token rating gate",
    )
    margin_token_study.add_argument("--raw-path", type=Path, required=True)
    margin_token_study.add_argument("--cache", type=Path, required=True)
    margin_token_study.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_006_margin_token_refinement.json"),
    )
    margin_token_study.add_argument("--output", type=Path, required=True)
    margin_token_study.add_argument("--repo-root", type=Path, default=Path.cwd())
    graded_rank = commands.add_parser(
        "run-graded-rank-study",
        help="run the preregistered GR-001 field-aware LambdaRank label gate",
    )
    graded_rank.add_argument("--cache", type=Path, required=True)
    graded_rank.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/gr_001_graded_lambdarank.json"),
    )
    graded_rank.add_argument("--output", type=Path, required=True)
    graded_rank.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_cache = commands.add_parser(
        "build-margin-rating-cache",
        help="build the preregistered PV-04 cache with one frozen margin-rating score",
    )
    margin_cache.add_argument("--raw-path", type=Path, required=True)
    margin_cache.add_argument("--baseline-cache", type=Path, required=True)
    margin_cache.add_argument("--output", type=Path, required=True)
    margin_cache.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_004_margin_rating_integration.json"),
    )
    margin_cache.add_argument(
        "--pv03-predictions",
        type=Path,
        help="optional PV-03 2024 candidate predictions for exact score reproduction",
    )
    margin_cache.add_argument("--repo-root", type=Path, default=Path.cwd())
    margin_delta_cache = commands.add_parser(
        "build-margin-rating-delta-cache",
        help="build the preregistered PV-05 same-spec margin-minus-ordinal cache",
    )
    margin_delta_cache.add_argument("--baseline-cache", type=Path, required=True)
    margin_delta_cache.add_argument("--margin-cache", type=Path, required=True)
    margin_delta_cache.add_argument("--ordinal-predictions", type=Path, required=True)
    margin_delta_cache.add_argument("--output", type=Path, required=True)
    margin_delta_cache.add_argument(
        "--config",
        type=Path,
        default=Path("configs/performance/pv_005_margin_rating_delta.json"),
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "audit":
        manifest = load_manifest(args.manifest)
        raw_path = resolve_raw_path(args.raw_path, environment_variable=manifest["path_policy"]["environment_variable"])
        fingerprint = verify_raw_file(raw_path, manifest, verify_hash=not args.skip_sha256)
        report = audit_csv(raw_path, manifest)
        verify_audit_against_manifest(report, manifest)
        print(json.dumps({"fingerprint": fingerprint, "audit": report}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-population":
        manifest = load_manifest(args.manifest)
        raw_path = resolve_raw_path(
            args.raw_path,
            environment_variable=manifest["path_policy"]["environment_variable"],
        )
        fingerprint = verify_raw_file(raw_path, manifest)
        raw = load_raw(raw_path, expected_sha256=manifest["raw_file"]["sha256"])
        races = build_race_population_table(normalize_raw(raw))
        report = population_selection_audit(races)
        report["fingerprint"] = fingerprint
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "archive-jvlink-batch":
        repo_root = args.repo_root.resolve()
        config_path = args.config if args.config.is_absolute() else repo_root / args.config
        config = load_live_data_archive_config(config_path)
        with args.input.open(encoding="utf-8") as handle:
            envelope = json.load(handle)
        result = archive_jvlink_batch(
            envelope,
            archive_root=args.archive_root,
            repo_root=repo_root,
            config=config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-mvp":
        manifest_path = args.manifest
        if not manifest_path.is_absolute():
            manifest_path = args.repo_root / manifest_path
        manifest = load_manifest(manifest_path)
        raw_path = resolve_raw_path(
            args.raw_path,
            environment_variable=manifest["path_policy"]["environment_variable"],
        )
        metrics = run_mvp(
            repo_root=args.repo_root,
            raw_path=raw_path,
            output_dir=args.output,
            manifest_path=args.manifest,
            split_path=args.splits,
            binary_config_path=args.binary_config,
            ranker_config_path=args.ranker_config,
            include_retrospective_test=args.include_retrospective_test,
            model_frame_cache_path=args.model_frame_cache,
            feature_config=FeatureConfig(
                surface_conditioned_elo=args.surface_conditioned_elo,
                expected_actual_race_value=args.expected_actual_race_value,
            ),
        )
        summary = {
            "output": str(args.output.resolve()),
            "elapsed_seconds": metrics["elapsed_seconds"],
            "data": metrics["data"],
            "models": {
                name: {
                    "best_iteration": payload["best_iteration"],
                    "temperature": payload["temperature"],
                }
                for name, payload in metrics["models"].items()
            },
            "evaluated_splits": list(metrics["splits"]),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze-uncertainty":
        result = run_uncertainty_analysis(
            args.predictions,
            args.output,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        print(json.dumps(result["scope"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-cached-experiment":
        metrics = run_cached_experiment(
            repo_root=args.repo_root,
            cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "experiment_id": metrics["experiment_id"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "feature_count": metrics["features"]["count"],
            "evaluated_split": "development",
            "retrospective_used": metrics["scope"]["retrospective_used"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-rolling-evaluation":
        metrics = run_rolling_evaluation(
            repo_root=args.repo_root,
            cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "experiment_id": metrics["experiment_id"],
            "evaluation_years": metrics["scope"]["evaluation_years"],
            "rows_used_2024": metrics["scope"]["rows_used_2024"],
            "rows_used_2025": metrics["scope"]["rows_used_2025"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-hpo-study":
        metrics = run_hpo_study(
            repo_root=args.repo_root,
            cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "experiment_id": metrics["experiment_id"],
            "model_kind": metrics["model_kind"],
            "selected_profile": metrics["selection"]["selection_result"]["selected_profile"],
            "decision": metrics["confirmation"]["confirmation_result"]["decision"],
            "rows_used_2024": metrics["scope"]["rows_used_2024"],
            "rows_used_2025": metrics["scope"]["rows_used_2025"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-ensemble-study":
        metrics = run_ensemble_study(
            repo_root=args.repo_root,
            config_path=args.config,
            output_dir=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "experiment_id": metrics["experiment_id"],
            "decision": metrics["decision"],
            "confirmation_opened": metrics["scope"]["confirmation_opened"],
            "rows_used_2024": metrics["scope"]["rows_used_2024"],
            "rows_used_2025": metrics["scope"]["rows_used_2025"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-shimba-filter-study":
        metrics = run_shimba_filter_study(
            repo_root=args.repo_root,
            cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "experiment_id": metrics["experiment_id"],
            "decision": metrics["decision"]["decision"],
            "evaluation_years": metrics["scope"]["evaluation_years"],
            "rows_used_2024": metrics["scope"]["rows_used_2024"],
            "rows_used_2025": metrics["scope"]["rows_used_2025"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "diagnose-baseline":
        result = run_baseline_diagnostics(
            repo_root=args.repo_root,
            cache_path=args.cache,
            baseline_dir=args.baseline,
            output_dir=args.output,
            permutation_repeats=args.permutation_repeats,
            seed=args.seed,
        )
        print(json.dumps(result["scope"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-rating-study":
        result = run_rating_study(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            model_cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "study_id": result["study_id"],
                    "scope": result["scope"],
                    "r0_passed": result["r0"]["passed"],
                    "final_spec": result["r5"]["final_spec"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "build-race-content-cache":
        repo_root = args.repo_root.resolve()
        config_path = args.config
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config, spec = load_race_content_config(config_path)
        manifest = load_manifest(repo_root / "configs/data_manifest.json")
        verify_raw_file(args.raw_path, manifest)
        raw = load_raw(
            args.raw_path, expected_sha256=manifest["raw_file"]["sha256"]
        )
        # Remove 2025 before normalized outcome fields are parsed or inspected.
        years = pd.to_numeric(raw["raceid"].str.slice(0, 4), errors="raise")
        raw = raw.loc[years.le(2024)].copy()
        history = build_race_content_history(normalize_raw(raw), spec=spec)
        result = build_race_content_augmented_cache(
            args.baseline_cache,
            history,
            args.output,
            config=config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-opponent-recent-cache":
        result = build_opponent_recent_cache_from_raw(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            baseline_cache_path=args.baseline_cache,
            output_path=args.output,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-sectional-recent-cache":
        result = build_sectional_recent_cache_from_raw(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            baseline_cache_path=args.baseline_cache,
            output_path=args.output,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-pace-recent-cache":
        result = build_pace_recent_cache_from_raw(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            baseline_cache_path=args.baseline_cache,
            output_path=args.output,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-pace-pressure-cache":
        result = build_pace_pressure_cache_from_config(
            repo_root=args.repo_root,
            input_cache_path=args.input_cache,
            output_path=args.output,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-margin-rating-study":
        result = run_margin_rating_study(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            model_cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "validation_2022": result["validation_2022"],
                    "development_2024_opened": result["scope"][
                        "development_2024_opened"
                    ],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "run-margin-rating-calibration-study":
        result = run_margin_rating_calibration_study(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            model_cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "rolling_gate": result["rolling_gate"],
                    "development_2024_opened": result["scope"][
                        "development_2024_opened"
                    ],
                    "development_2024": result.get("development_2024"),
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "audit-margin-tokens":
        result = audit_margin_tokens_from_raw(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "years": result["years"],
                    "mapping_gate": result["mapping_gate"],
                    "equal_clock": result["equal_clock"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "run-margin-token-rating-study":
        result = run_margin_token_rating_study(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            model_cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "mapping_gate": result["mapping_gate"],
                    "validation_2022": result.get("validation_2022"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "run-graded-rank-study":
        result = run_graded_rank_study(
            repo_root=args.repo_root,
            cache_path=args.cache,
            config_path=args.config,
            output_dir=args.output,
        )
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "scope": result["scope"],
                    "models": result["models"],
                    "validation_2022": {
                        key: result["validation_2022"][key]
                        for key in (
                            "primary_metrics",
                            "candidate_improvement",
                            "decision",
                            "probability_path_passed",
                            "ranking_path_passed",
                            "guardrail_failed",
                        )
                    },
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "build-margin-rating-cache":
        result = build_margin_rating_cache_from_raw(
            repo_root=args.repo_root,
            raw_path=args.raw_path,
            baseline_cache_path=args.baseline_cache,
            output_path=args.output,
            config_path=args.config,
            pv03_predictions_path=args.pv03_predictions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-margin-rating-delta-cache":
        result = build_margin_rating_delta_cache(
            baseline_cache_path=args.baseline_cache,
            margin_cache_path=args.margin_cache,
            ordinal_predictions_path=args.ordinal_predictions,
            output_path=args.output,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
