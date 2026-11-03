"""Verify reached-state values from public packets using an independent scheduling MIP.

Conditional states before forecast revelation remain the Bellman solver's domain.
This verifier never reads the episode's hidden world or calls its transition/planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from aeolus.agents.benchmark import write_json
from aeolus.agents.contract import digest


def public_forecast(observation):
    if observation["forecast"] != "unknown":
        return observation["forecast"]
    jobs = [t for t in observation["tasks"] if t["id"]]
    if all(t["value_by_forecast"]["alpha"] == t["value_by_forecast"]["beta"] for t in jobs):
        return "alpha"
    return None


def committed_value(observation, forecast):
    jobs = [t for t in observation["tasks"] if t["id"]]
    active = {r["active_task"] for r in observation["resources"] if r["active_task"]}
    assert all(t["started"] == (t["completed"] or t["id"] in active) for t in jobs)
    # Every active job is nonpreemptive and already scheduled to finish by its deadline.
    return sum(t["value_by_forecast"][forecast] - t["start_cost"] for t in jobs if t["started"])


def optimum(observation):
    forecast = public_forecast(observation)
    if forecast is None:
        raise ValueError("MIP cannot replace information-matched planning before revelation")
    jobs = {t["id"]: t for t in observation["tasks"] if t["id"]}
    resources = observation["resources"]
    horizon, now = observation["horizon_ticks"], observation["tick"]
    active_finish = {r["active_task"]: r["available_at_tick"] for r in resources if r["active_task"]}
    base = committed_value(observation, forecast)
    for task, finish in active_finish.items():
        assert now < finish <= min(horizon, jobs[task]["deadline_tick"])
    variables = [
        (task, r["id"], t)
        for task, job in jobs.items()
        if not job["started"] and job["kind"] != "SURVEY"
        for r in resources
        if job["duration_by_resource"][r["id"]] > 0
        for t in range(
            max(job["release_tick"], now, r["available_at_tick"]),
            min(job["deadline_tick"], horizon) - job["duration_by_resource"][r["id"]] + 1,
        )
    ]
    if not variables:
        return float(base)
    rows, limits = [], []
    for task in jobs:
        rows.append({k: 1 for k, (j, _, _) in enumerate(variables) if j == task})
        limits.append(1)
    for resource in resources:
        for tick in range(now, horizon):
            rows.append(
                {
                    k: 1
                    for k, (j, r, t) in enumerate(variables)
                    if r == resource["id"] and t <= tick < t + jobs[j]["duration_by_resource"][r]
                }
            )
            limits.append(1)
    for k, (task, _, t) in enumerate(variables):
        for predecessor in jobs[task]["prerequisites"]:
            if jobs[predecessor]["completed"] or active_finish.get(predecessor, horizon + 1) <= t:
                continue
            row = {
                kk: -1
                for kk, (j, r, tt) in enumerate(variables)
                if j == predecessor and tt + jobs[j]["duration_by_resource"][r] <= t
            }
            row[k] = 1
            rows.append(row)
            limits.append(0)
    rows.append({k: jobs[j]["water_required"] for k, (j, _, _) in enumerate(variables)})
    limits.append(observation["water_remaining"])
    matrix = lil_matrix((len(rows), len(variables)), dtype=float)
    for row_index, row in enumerate(rows):
        for k, value in row.items():
            matrix[row_index, k] = value
    objective = [jobs[j]["start_cost"] - jobs[j]["value_by_forecast"][forecast] for j, _, _ in variables]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(
            np.array(objective, dtype=float),
            integrality=np.ones(len(variables)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsc(), -np.inf, limits),
            options={"mip_rel_gap": 0, "time_limit": 60, "threads": 1},
        )
    if not result.success or result.mip_gap > 1e-9:
        raise RuntimeError("Independent reached-state solver did not prove optimality")
    return float(base - result.fun)


def validate(root):
    run = root / "benchmark"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    cache, values_checked, regrets_checked, skipped = {}, 0, 0, 0
    max_difference = 0.0

    def checked_value(step):
        nonlocal values_checked, max_difference
        obs = step["observation"]
        key = digest({k: v for k, v in obs.items() if k not in {"memory", "last_dispatch", "decision_id"}})
        if key not in cache:
            cache[key] = optimum(obs)
        expected = step["evaluation"]["optimal_value_before"]
        difference = abs(cache[key] - expected)
        assert difference < 1e-7, (key, cache[key], expected)
        max_difference = max(max_difference, difference)
        values_checked += 1
        return cache[key]

    for record in summary["records"]:
        path = run / record["trace"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["trace_sha256"]
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        steps = entries[1:-1]
        for index, step in enumerate(steps):
            obs = step["observation"]
            forecast = public_forecast(obs)
            if forecast is None:
                skipped += 1
                continue
            before = checked_value(step)
            if index + 1 < len(steps):
                after = checked_value(steps[index + 1])
                values_checked -= 1  # Count each pre-decision value once, despite cache lookahead.
            else:
                assert obs["tick"] + 1 == obs["horizon_ticks"]
                jobs = {t["id"]: t for t in obs["tasks"] if t["id"]}
                assigned = [task for task in step["receipt"]["actions"].values() if task]
                after = committed_value(obs, forecast) + sum(
                    jobs[t]["value_by_forecast"][forecast] - jobs[t]["start_cost"] for t in assigned
                )
                assert abs(after - record["net_reward"]) < 1e-7
            difference = abs(before - after - step["evaluation"]["regret"])
            assert difference < 1e-7, (path.name, index, before, after)
            max_difference = max(max_difference, difference)
            regrets_checked += 1
    result = {
        "all_checked_values_and_regrets_verified": True,
        "episodes": len(summary["records"]),
        "unique_public_states_solved": len(cache),
        "decision_values_checked": values_checked,
        "decision_regrets_checked": regrets_checked,
        "uncertain_pre_revelation_decisions": skipped,
        "maximum_absolute_difference": max_difference,
        "scope": "Independent MIP from public packets; conditional pre-revelation states use Bellman tests.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_values": cache,
    }
    write_json(root / "independent-state-verification.json", result)
    return {k: v for k, v in result.items() if k != "state_values"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_root), indent=2))
