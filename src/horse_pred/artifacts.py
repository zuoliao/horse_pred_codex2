"""Small, reproducible experiment metadata artifacts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from horse_pred.config import canonical_json_hash


def git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def build_run_meta(
    *,
    repo_root: str | Path,
    experiment_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_config["experiment_id"],
        "hypothesis": experiment_config["hypothesis"],
        "model_family": experiment_config["model_family"],
        "seed": experiment_config["seed"],
        "feature_groups": experiment_config["feature_groups"],
        "experiment_config_hash": canonical_json_hash(experiment_config),
        "split_config_hash": canonical_json_hash(split_config),
        "data_fingerprint": data_manifest["sha256"],
        "git": git_state(repo_root),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")
