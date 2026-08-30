"""Run the preregistered field-size temperature calibration experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horse_pred.field_size_calibration import run_field_size_calibration_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run_field_size_calibration_experiment(
        repo_root=args.repo_root,
        baseline_dir=args.baseline,
        config_path=args.config,
        output_dir=args.output,
    )
    summary = {
        "output": str(args.output.resolve()),
        "experiment_id": result["experiment_id"],
        "calibration_races": result["data"]["calibration_races"],
        "development_races": result["data"]["development_races"],
        "retrospective_used": result["scope"]["retrospective_used"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
