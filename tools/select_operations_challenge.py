"""Freeze challenge cases using reference-only difficulty checks, never model outcomes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from aeolus.agents.benchmark import source_identity, write_json
from aeolus.agents.operations import (
    OperationsDesk,
    Planner,
    ShortPlanner,
    dispatch,
    generate,
    initial_state,
    with_forecast,
)

CRITERIA = {
    "two_step_regret_min_pct": 8.0,
    "deep_chain_value_min_pct": 5.0,
    "information_value_min_pct": 5.0,
    "max_reference_states": 350000,
}


def profile(seed):
    instance = generate("compound", seed, 3)
    planner, short = Planner(instance), ShortPlanner(instance)
    optimum = planner.value(initial_state(instance))
    loss = 0.0
    for world in (0, 1):
        desk = OperationsDesk(instance, world)
        while not desk.done:
            action = short.action(desk.state)
            loss += planner.regret(desk.state, action)
            desk.step(dispatch(instance, desk.state, desk.observation(), action))
    # Include informed roots in the resource-budget check used by independent verification.
    informed = sum(planner.value(with_forecast(initial_state(instance), w)) for w in (0, 1)) / 2
    deep_endpoint = [j for j in instance.jobs if len(j.prerequisites) == 2]
    assert len(deep_endpoint) == 1
    without_deep = replace(instance, jobs=tuple(j for j in instance.jobs if j != deep_endpoint[0]))
    without_info = replace(
        instance, jobs=tuple(j for j in instance.jobs if j.kind != "SURVEY"), bulletin_tick=instance.horizon
    )
    shallow_value = Planner(without_deep).value(initial_state(without_deep))
    blind_value = Planner(without_info).value(initial_state(without_info))
    result = {
        "family": "compound",
        "seed": seed,
        "level": 3,
        "instance_sha256": instance.identity,
        "optimal_expected_reward": optimum,
        "states": planner.states,
        "two_step_regret_pct": 50 * loss / optimum,
        "deep_chain_value_pct": 100 * (optimum - shallow_value) / optimum,
        "information_value_pct": 100 * (optimum - blind_value) / optimum,
        "perfect_information_value_pct": 100 * (informed - optimum) / optimum,
    }
    result["accepted"] = (
        result["two_step_regret_pct"] >= CRITERIA["two_step_regret_min_pct"]
        and result["deep_chain_value_pct"] >= CRITERIA["deep_chain_value_min_pct"]
        and result["information_value_pct"] >= CRITERIA["information_value_min_pct"]
        and result["states"] <= CRITERIA["max_reference_states"]
    )
    return result


def select(start_seed, count, split, repeats, suite_path, report_path, max_candidates):
    if suite_path.exists() or report_path.exists():
        raise ValueError("Selection outputs must be new; preserve previous candidate decisions")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source": source_identity(),
        "start_seed": start_seed,
        "requested_instances": count,
        "criteria": CRITERIA,
        "candidates": [],
        "complete": False,
        "selection_rule": "First qualifying seeds in ascending order; no model scores or requests are used.",
        "scope": (
            "Compound stress distribution; two-step is a selection control, not an independent baseline."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cases = []
    for seed in range(start_seed, start_seed + max_candidates):
        try:
            row = profile(seed)
        except RuntimeError as exc:
            if "state budget" not in str(exc):
                raise
            row = {"seed": seed, "accepted": False, "error": "exact_reference_state_budget_exceeded"}
        report["candidates"].append(row)
        if row["accepted"]:
            cases.append({"family": "compound", "seed": seed, "level": 3})
        report["selected_instances"] = len(cases)
        write_json(report_path, report)
        print(json.dumps({"seed": seed, "accepted": row["accepted"], "selected": len(cases)}), flush=True)
        gc.collect()
        if len(cases) == count:
            break
    if len(cases) != count:
        raise RuntimeError(
            "Not enough qualifying cases; do not issue model requests or weaken criteria silently"
        )
    suite = {
        "schema_version": 2,
        "suite_id": "operations-challenge-" + split,
        "split": split,
        "repeats": repeats,
        "cases": cases,
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    report["complete"] = True
    report["suite_sha256_bytes"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    write_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.count <= args.max_candidates or not 1 <= args.repeats <= 8:
        parser.error("Require a positive count within the candidate budget and 1..8 repeats")
    select(
        args.start_seed, args.count, args.split, args.repeats, args.suite, args.report, args.max_candidates
    )
