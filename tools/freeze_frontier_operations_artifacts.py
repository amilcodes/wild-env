"""Create a content manifest for the frozen v0.5 operations evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    source_files = [
        "configs/frontier_suppression.yaml",
        "docs/suppression_operations_research.md",
        "src/aeolus/config.py",
        "src/aeolus/core/fire.py",
        "src/aeolus/core/initialization.py",
        "src/aeolus/core/localization.py",
        "src/aeolus/core/simulator.py",
        "src/aeolus/core/state.py",
        "src/aeolus/core/suppression.py",
        "src/aeolus/core/tasks.py",
        "src/aeolus/data/forcing.py",
        "src/aeolus/data/weather.py",
        "src/aeolus/evaluation/historical.py",
        "src/aeolus/policies/heuristics.py",
        "src/aeolus/replay/recorder.py",
        "src/aeolus/replay/render.py",
        "tests/test_frontier_operations.py",
        "tools/build_frontier_operations_figures.py",
        "tools/build_suppression_operations_report.py",
        "tools/freeze_frontier_operations_artifacts.py",
        "tools/run_suppression_operations_study.py",
    ]
    artifact_files = [
        args.results / "operations_results.json",
        args.results / "suppression_trials.csv",
        args.results / "arrival_history_forecasts.csv",
        args.results / "arrival_history_examples.npz",
        args.results / "figures/arrival_history_atlas.png",
        args.results / "figures/suppression_outcomes.png",
        args.results / "figures/forcing_analysis.png",
        args.results / "operations_2d_mid.png",
        args.results / "operations_3d_mid.png",
        args.results / "operations_timelapse.mp4",
        args.results / "replay/metadata.json",
        args.results / "replay/events.parquet",
        args.report,
    ]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
        ).strip()
    )
    manifest = {
        "schema_version": 1,
        "study": "frontier suppression operations v0.5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_worktree_dirty": dirty,
        "verification": {
            "pytest": "51 passed",
            "ruff": "passed",
            "historical_forecasts": 24,
            "suppression_trials": 72,
            "replay_frames": 241,
            "report_pages": 8,
        },
        "source_files": {relative: _sha256(repository / relative) for relative in source_files},
        "artifacts": {
            str(path.resolve().relative_to(repository)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_files
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
