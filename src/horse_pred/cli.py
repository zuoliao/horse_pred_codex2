"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

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
from horse_pred.features import FeatureConfig
from horse_pred.pipeline import run_mvp
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
