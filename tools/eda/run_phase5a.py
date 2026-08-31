"""Run the registered Phase 5A EDA pipeline and aggregate workstreams.

This command is intentionally diagnostic.  It never trains or promotes a
production candidate, and the central loader rejects any target-aware cutoff
other than 2022-12-31.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from horse_pred.eda import run_eda


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_workstream(script: Path, env: dict[str, str], log: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"workstream failed: {script}; see {log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-date", default="2022-12-31")
    parser.add_argument(
        "--rolling-predictions",
        default="artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz",
        type=Path,
    )
    parser.add_argument("--resume-common", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    run_eda(
        repo_root=repo,
        raw_path=args.raw_path.resolve(),
        output_dir=output,
        max_date=args.max_date,
        resume=args.resume_common,
    )

    env = os.environ.copy()
    env.update(
        {
            "HORSE_EDA_RAW_PATH": str(args.raw_path.resolve()),
            "HORSE_EDA_RUNNER_VIEW": str(output / "views/runner_pre_race.csv.gz"),
            "HORSE_EDA_OUTCOMES_VIEW": str(output / "views/outcomes.csv.gz"),
            "HORSE_EDA_ROLLING_PREDICTIONS": str(args.rolling_predictions.resolve()),
            "HORSE_EDA_TARGET_OUTPUT": str(output / "workstreams/b_target/summary.json"),
            "HORSE_EDA_HISTORY_OUTPUT": str(output / "workstreams/c_history"),
            "HORSE_EDA_HISTORY_CACHE": str(output / "workstreams/c_history/runner_history_flat_private.pkl"),
            "HORSE_EDA_OPPONENT_OUTPUT": str(output / "workstreams/e_opponent/summary.json"),
            "HORSE_EDA_CONNECTIONS_OUTPUT": str(output / "workstreams/f_connections"),
            "HORSE_EDA_CONTEXT_OUTPUT": str(output / "workstreams/g_context"),
            "HORSE_EDA_ERRORS_OUTPUT": str(output / "workstreams/h_errors"),
        }
    )
    scripts = [
        ("b_target", "target_and_race_structure.py"),
        ("c_history", "horse_history.py"),
        ("e_opponent", "opponent_and_field.py"),
        ("f_connections", "connections.py"),
        ("g_context", "context_and_interactions.py"),
        ("h_errors", "model_errors.py"),
    ]
    script_root = Path(__file__).parent / "workstreams"
    for name, filename in scripts:
        _run_workstream(
            script_root / filename,
            env,
            output / "logs" / f"{name}.log",
        )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workstream_scripts"] = {
        name: {
            "path": str((script_root / filename).relative_to(repo)),
            "sha256": _sha256(script_root / filename),
        }
        for name, filename in scripts
    }
    manifest["files"] = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "workstreams": [x[0] for x in scripts]}))


if __name__ == "__main__":
    main()
