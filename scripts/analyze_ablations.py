"""Aggregate paired uncertainty for completed cached ablation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from horse_pred.ablation_analysis import run_ablation_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20240830)
    args = parser.parse_args()
    candidates = {
        path.name: path / "predictions_2024.csv.gz"
        for path in sorted(args.artifacts_root.glob("abl_*"))
        if (path / "predictions_2024.csv.gz").is_file()
    }
    if not candidates:
        raise SystemExit("no completed abl_* prediction artifacts found")
    run_ablation_analysis(
        args.baseline,
        candidates,
        args.output,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
