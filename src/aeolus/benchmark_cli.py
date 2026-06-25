"""Reproducible single-process simulator throughput benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle
from aeolus.workflows import resolve_policy, scenario_from_incident


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark headless simulator throughput")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--incident")
    parser.add_argument("--policy", default="no_aerial")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--horizon-min", type=int, default=180)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.envs < 1 or args.decisions < 1:
        raise ValueError("envs and decisions must be positive")

    if args.incident:
        config = scenario_from_incident(
            IncidentBundle.load(args.incident),
            horizon_min=args.horizon_min,
        )
    else:
        config = load_config(args.config).scenario
    policy = resolve_policy(args.policy)[0]
    simulators = [AeolusSimulator(config) for _ in range(args.envs)]
    for index, simulator in enumerate(simulators):
        simulator.reset(config.seed + index * 7919)

    minutes_advanced = 0
    resets = 0
    durations: list[float] = []
    start = time.perf_counter()
    for decision in range(args.decisions):
        simulator = simulators[decision % len(simulators)]
        before = simulator.state.minute
        tick = time.perf_counter()
        simulator.decision_step(policy(simulator))
        durations.append(time.perf_counter() - tick)
        minutes_advanced += simulator.state.minute - before
        if simulator.state.terminated or simulator.state.truncated:
            resets += 1
            simulator.reset(config.seed + (resets + decision) * 7919)
    elapsed = time.perf_counter() - start
    result = {
        "schema_version": 1,
        "policy": args.policy,
        "envs": args.envs,
        "decisions": args.decisions,
        "minutes_advanced": minutes_advanced,
        "elapsed_s": elapsed,
        "decisions_per_s": args.decisions / elapsed,
        "simulated_minutes_per_s": minutes_advanced / elapsed,
        "grid_cell_minutes_per_s": (minutes_advanced * config.width * config.height / elapsed),
        "decision_latency_ms": {
            "p50": float(np.percentile(durations, 50) * 1000.0),
            "p95": float(np.percentile(durations, 95) * 1000.0),
            "p99": float(np.percentile(durations, 99) * 1000.0),
        },
        "grid": [config.height, config.width],
        "resources": len(config.resources),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
