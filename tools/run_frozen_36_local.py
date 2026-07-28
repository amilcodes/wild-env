#!/usr/bin/env python3
"""Run the complete frozen 36-incident benchmark as resumable local stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aeolus.evaluation.frozen_benchmark import (
    audit_frozen_contract,
    load_frozen_contract,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prepared_count(root: Path) -> int:
    return len(list(root.glob("*/item.json")))


def _run_stage(
    name: str,
    command: list[str],
    *,
    status: dict[str, Any],
    status_path: Path,
    log_path: Path,
) -> None:
    record = {
        "name": name,
        "status": "running",
        "started_at": _utc_now(),
        "command": command,
    }
    status["current_stage"] = name
    status["stages"].append(record)
    _write_status(status_path, status)
    print(f"[frozen-36] starting {name}", flush=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] START {name}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )
        log.write(f"[{_utc_now()}] END {name} returncode={result.returncode}\n")
    record["finished_at"] = _utc_now()
    record["returncode"] = result.returncode
    record["status"] = "completed" if result.returncode == 0 else "failed"
    status["current_stage"] = None
    status["status"] = "running" if result.returncode == 0 else "failed"
    _write_status(status_path, status)
    if result.returncode != 0:
        raise RuntimeError(f"stage {name} failed with return code {result.returncode}")
    print(f"[frozen-36] completed {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-shapefile", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--forcing-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--existing-prepare-pid", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    contract = load_frozen_contract(args.contract)
    audit = audit_frozen_contract(contract)
    if not audit["valid"] or audit["counts"] != {
        "development": 7,
        "train": 22,
        "test": 7,
    }:
        raise ValueError(f"unexpected frozen 36-incident contract: {audit}")
    base_manifest = Path(contract["base_manifest_path"])
    results = args.results_root.resolve()
    results.mkdir(parents=True, exist_ok=True)
    status_path = results / "local_run_status.json"
    log_path = results / "local_run.log"
    status: dict[str, Any] = {
        "schema_version": 1,
        "study": "frozen 36-incident local deterministic benchmark",
        "status": "running",
        "started_at": _utc_now(),
        "current_stage": "waiting_for_existing_prepare",
        "contract_audit": audit,
        "workers": args.workers,
        "paths": {
            "prepared_root": str(args.prepared_root.resolve()),
            "historical_root": str(args.historical_root.resolve()),
            "forcing_root": str(args.forcing_root.resolve()),
            "results_root": str(results),
        },
        "stages": [],
    }
    _write_status(status_path, status)

    if args.existing_prepare_pid is not None:
        last_report = 0.0
        while _pid_running(args.existing_prepare_pid):
            now = time.monotonic()
            if now - last_report >= 300.0:
                count = _prepared_count(args.prepared_root)
                print(
                    f"[frozen-36] existing preparation active: {count}/36 incidents",
                    flush=True,
                )
                status["prepared_incidents"] = count
                _write_status(status_path, status)
                last_report = now
            time.sleep(30.0)

    repo = Path(__file__).resolve().parents[1]
    python = sys.executable
    _run_stage(
        "prepare_metric_corpus",
        [
            str(repo / ".venv/bin/aeolus-study"),
            "prepare",
            "--manifest",
            str(base_manifest),
            "--source-shapefile",
            str(args.source_shapefile.resolve()),
            "--out",
            str(args.prepared_root.resolve()),
        ],
        status=status,
        status_path=status_path,
        log_path=log_path,
    )
    _run_stage(
        "rebuild_time_admissible_fuels",
        [
            python,
            str(repo / "tools/rebuild_historical_fuels.py"),
            "--source-root",
            str(args.prepared_root.resolve()),
            "--out",
            str(args.historical_root.resolve()),
            "--workers",
            str(args.workers),
        ],
        status=status,
        status_path=status_path,
        log_path=log_path,
    )
    _run_stage(
        "materialize_operational_hrrr",
        [
            python,
            str(repo / "tools/materialize_operational_hrrr_forcing.py"),
            str(args.contract.resolve()),
            str(args.historical_root.resolve()),
            str(args.forcing_root.resolve()),
        ],
        status=status,
        status_path=status_path,
        log_path=log_path,
    )
    _run_stage(
        "geometric_baselines",
        [
            python,
            str(repo / "tools/run_frozen_historical_benchmark.py"),
            str(args.contract.resolve()),
            str(args.historical_root.resolve()),
            str(results / "baseline_results.json"),
        ],
        status=status,
        status_path=status_path,
        log_path=log_path,
    )
    _run_stage(
        "canonical_operational_physics",
        [
            python,
            str(repo / "tools/run_frozen_physics_benchmark.py"),
            str(args.contract.resolve()),
            str(args.historical_root.resolve()),
            str(results / "physics_operational_hrrr_results.json"),
            "--workers",
            str(args.workers),
            "--cache",
            str(results / "physics_cache"),
            "--operational-forcing-root",
            str(args.forcing_root.resolve()),
        ],
        status=status,
        status_path=status_path,
        log_path=log_path,
    )
    status["status"] = "completed"
    status["finished_at"] = _utc_now()
    status["prepared_incidents"] = _prepared_count(args.prepared_root)
    _write_status(status_path, status)
    print("[frozen-36] complete", flush=True)


if __name__ == "__main__":
    main()
