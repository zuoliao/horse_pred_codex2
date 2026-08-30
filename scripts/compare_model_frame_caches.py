#!/usr/bin/env python3
"""Compare the corrected baseline cache with the surface-Elo candidate cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from horse_pred.cache_control import compare_surface_elo_cache_control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    result = compare_surface_elo_cache_control(
        args.baseline_cache,
        args.candidate_cache,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
