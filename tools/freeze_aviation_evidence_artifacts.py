#!/usr/bin/env python3
"""Freeze the aviation evidence inputs, implementation, and audit outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/aviation_evidence"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/aviation/evidence_registry_v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/aviation_evidence/artifact_manifest.json"),
    )
    parser.add_argument("--pytest-summary", required=True)
    parser.add_argument("--ruff-summary", required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    output = (repository / args.out).resolve()
    results = (repository / args.results).resolve()
    registry_path = (repository / args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    source_files = (
        "README.md",
        "configs/aviation/delivery_surfaces/calfire_s2t_mtdc_2006_gum_v1.json",
        "configs/aviation/evidence_registry_v1.json",
        "configs/aviation/us_wildfire_reference_fleet_v1.json",
        "configs/aviation/us_wildfire_reference_operations.yaml",
        "docs/aviation_evidence_acquisition.md",
        "docs/aviation_records_request.md",
        "docs/aviation_vehicle_closure.md",
        "docs/noncompute_p1_remaining_work.md",
        "docs/system_report_for_students.md",
        "src/aeolus/config.py",
        "src/aeolus/core/simulator.py",
        "src/aeolus/core/state.py",
        "src/aeolus/core/suppression.py",
        "src/aeolus/data/aerial_delivery.py",
        "src/aeolus/data/__init__.py",
        "src/aeolus/data/aviation_catalog.py",
        "src/aeolus/data/aviation_evidence.py",
        "src/aeolus/replay/recorder.py",
        "tests/test_aerial_delivery.py",
        "tests/test_aviation_catalog.py",
        "tests/test_aviation_evidence.py",
        "tests/test_reference_operations.py",
        "tests/test_replay.py",
        "tools/fetch_aviation_evidence.py",
        "tools/freeze_aviation_evidence_artifacts.py",
        "tools/run_aviation_evidence_closure.py",
    )
    missing = [relative for relative in source_files if not (repository / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source files: {missing}")

    cache_records: dict[str, Any] = {}
    for document in registry["documents"]:
        relative = document.get("local_cache_path")
        if relative is None:
            continue
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing public-document cache: {relative}")
        actual = _record(path)
        expected = document.get("sha256")
        if actual["sha256"] != expected:
            raise ValueError(f"public-document checksum mismatch: {relative}")
        cache_records[document["document_id"]] = {
            "path": relative,
            "url": document["url"],
            **actual,
        }

    result_files = sorted(path for path in results.rglob("*") if path.is_file() and path.resolve() != output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    manifest = {
        "schema_version": 1,
        "study": "exact-configuration wildfire aviation public-evidence closure",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_worktree_dirty": dirty,
        "verification": {
            "pytest": args.pytest_summary,
            "ruff": args.ruff_summary,
            "public_documents_checksum_verified": len(cache_records),
            "vehicle_profiles": len(registry["profiles"]),
            "field_closed_profiles": 0,
        },
        "source_files": {relative: _record(repository / relative) for relative in source_files},
        "public_document_cache": cache_records,
        "result_artifacts": {str(path.relative_to(results)): _record(path) for path in result_files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
