#!/usr/bin/env python3
"""Fingerprint or verify the code and environment used by a local study run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_TOOLS = (
    "tools/freeze_execution_source.py",
    "tools/run_frozen_36_local.py",
    "tools/rebuild_historical_fuels.py",
    "tools/materialize_operational_hrrr_forcing.py",
    "tools/run_frozen_historical_benchmark.py",
    "tools/run_frozen_physics_benchmark.py",
)
FIXED_INPUTS = (
    "pyproject.toml",
    "requirements/research.lock",
    "configs/historical_validation_expanded.yaml",
    "configs/historical_validation_frozen_36.yaml",
    "src/aeolus/resources/fire_behavior_lookup.npz",
    "native/CMakeLists.txt",
    "native/include/aeolus/reference_kernel.hpp",
    "native/src/reference_kernel.cpp",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_paths(root: Path) -> list[Path]:
    paths = [*root.glob("src/aeolus/**/*.py")]
    paths.extend(root / relative for relative in (*PIPELINE_TOOLS, *FIXED_INPUTS))
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"execution inputs are missing: {missing}")
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def _packages() -> dict[str, str]:
    return dict(
        sorted(
            (distribution.metadata["Name"].lower(), distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        )
    )


def _create(root: Path) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _source_paths(root)
    }
    tree_digest = hashlib.sha256(
        "".join(f"{name}\0{record['sha256']}\n" for name, record in files.items()).encode()
    ).hexdigest()
    status = _git(root, "status", "--porcelain=v1")
    return {
        "schema_version": 1,
        "purpose": "frozen local benchmark execution-source fingerprint",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(root),
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
            "tracked_and_untracked_status_sha256": (
                hashlib.sha256((status or "").encode()).hexdigest()
            ),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": _packages(),
        },
        "tree_sha256": tree_digest,
        "file_count": len(files),
        "files": files,
    }


def _verify(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    missing: list[str] = []
    for relative, record in expected["files"].items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif _sha256(path) != record["sha256"]:
            changed.append(relative)
    current_packages = _packages()
    expected_packages = expected["runtime"]["packages"]
    added_packages = sorted(set(current_packages) - set(expected_packages))
    removed_packages = sorted(set(expected_packages) - set(current_packages))
    changed_packages = sorted(
        name
        for name in set(current_packages) & set(expected_packages)
        if current_packages[name] != expected_packages[name]
    )
    return {
        "valid": not (changed or missing or added_packages or removed_packages or changed_packages),
        "expected_tree_sha256": expected["tree_sha256"],
        "changed_files": changed,
        "missing_files": missing,
        "environment": {
            "added_packages": added_packages,
            "removed_packages": removed_packages,
            "changed_packages": changed_packages,
            "python_executable_changed": (
                sys.executable != expected["runtime"]["python_executable"]
            ),
            "python_version_changed": sys.version != expected["runtime"]["python_version"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if args.verify:
        result = _verify(root, json.loads(output.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    result = _create(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: result[key] for key in ("tree_sha256", "file_count")}, indent=2))


if __name__ == "__main__":
    main()
