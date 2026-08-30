"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from horse_pred.data import (
    audit_csv,
    load_manifest,
    resolve_raw_path,
    verify_audit_against_manifest,
    verify_raw_file,
)
from horse_pred.pipeline import run_mvp


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="horse-pred")
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="verify and summarize the approved raw CSV")
    audit.add_argument("--raw-path", type=Path)
    audit.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.json"))
    audit.add_argument("--skip-sha256", action="store_true")

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
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
