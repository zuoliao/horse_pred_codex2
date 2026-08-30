#!/usr/bin/env python3
"""Verify the frozen raw file and print coverage/quality audit JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from horse_pred.data import (  # noqa: E402
    audit_csv,
    load_manifest,
    resolve_raw_path,
    verify_audit_against_manifest,
    verify_raw_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "data_manifest.json",
        help="versioned data manifest (default: configs/data_manifest.json)",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        help="external raw CSV; otherwise use the manifest path environment variable",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="skip the expensive content hash check; size/BOM/schema are still checked",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    environment_variable = manifest["path_policy"]["environment_variable"]
    raw_path = resolve_raw_path(
        args.raw_path,
        environment_variable=environment_variable,
    )
    fingerprint = verify_raw_file(raw_path, manifest, verify_hash=not args.skip_sha256)
    report = audit_csv(raw_path, manifest)
    verify_audit_against_manifest(report, manifest)
    output = {
        "manifest": str(args.manifest),
        "raw_path": str(raw_path),
        "fingerprint": fingerprint,
        "audit": report,
        "verification": "passed",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

