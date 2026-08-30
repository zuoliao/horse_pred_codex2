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
from horse_pred.data import sha256_file


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
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def write_artifact_manifest(directory: str | Path) -> None:
    """Hash every completed file below an artifact directory."""

    root = Path(directory)
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(root / "artifact_manifest.json", {"schema_version": 1, "files": files})


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value
