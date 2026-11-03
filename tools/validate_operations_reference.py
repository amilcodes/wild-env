"""Independently check the exact planner against a time-indexed mixed-integer model.

The MIP sees the forecast from the start and verifies the full-information subproblem.
Two-world policy replay separately checks the information-matched Bellman value.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from aeolus.agents.benchmark import write_json
from aeolus.agents.operations import OperationsDesk, Planner, dispatch, initial_state, payoff, with_forecast
from aeolus.agents.operations_benchmark import load_suite


def mip_value(instance, world):
    variables = [
        (j, r, t)
        for j, job in enumerate(instance.jobs)
        for r, duration in enumerate(job.durations)
        if duration > 0 and job.kind != "SURVEY"
        for t in range(job.release, min(job.deadline, instance.horizon) - duration + 1)
    ]
    rows, upper = [], []
    for j in range(len(instance.jobs)):
        rows.append({k: 1 for k, (job, _, _) in enumerate(variables) if j == job})
        upper.append(1)
    for r in range(len(instance.resources)):
        for tick in range(instance.horizon):
            rows.append(
                {
                    k: 1
                    for k, (j, rr, t) in enumerate(variables)
                    if rr == r and t <= tick < t + instance.jobs[j].durations[r]
                }
            )
            upper.append(1)
    positions = {job.id: j for j, job in enumerate(instance.jobs)}
    for k, (j, _, t) in enumerate(variables):
        for prerequisite in instance.jobs[j].prerequisites:
            row = {
                kk: -1
                for kk, (jj, rr, tt) in enumerate(variables)
                if jj == positions[prerequisite] and tt + instance.jobs[jj].durations[rr] <= t
            }
            row[k] = 1
            rows.append(row)
            upper.append(0)
    rows.append({k: instance.jobs[j].water for k, (j, _, _) in enumerate(variables)})
    upper.append(instance.water)
    matrix = lil_matrix((len(rows), len(variables)), dtype=float)
    for i, row in enumerate(rows):
        for k, value in row.items():
            matrix[i, k] = value
    objective = [-instance.jobs[j].values[world] + instance.jobs[j].cost for j, _, _ in variables]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(
            np.array(objective, dtype=float),
            integrality=np.ones(len(variables)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsc(), -np.inf, upper),
            options={"mip_rel_gap": 0, "time_limit": 60, "threads": 1},
        )
    if not result.success or result.mip_gap > 1e-9:
        raise RuntimeError("independent MIP did not prove optimality")
    return float(-result.fun)


def validate(suite):
    _, instances = load_suite(suite)
    checks = []
    for instance in instances:
        planner = Planner(instance)
        expected = planner.value(initial_state(instance))
        realized = []
        mip = []
        for world in (0, 1):
            optimal_known = planner.value(with_forecast(initial_state(instance), world))
            independent = mip_value(instance, world)
            if abs(optimal_known - independent) > 1e-7:
                raise AssertionError("dynamic program and independent MIP disagree")
            desk = OperationsDesk(instance, world)
            while not desk.done:
                action = planner.action(desk.state)
                desk.step(dispatch(instance, desk.state, desk.observation(), action))
            realized.append(payoff(instance, desk.state, world))
            mip.append(independent)
        if abs(sum(p * value for p, value in zip(instance.prior, realized, strict=True)) - expected) > 1e-7:
            raise AssertionError("information-matched policy replay disagrees with Bellman value")
        checks.append(
            {
                "instance_sha256": instance.identity,
                "family": instance.family,
                "level": instance.level,
                "optimal_expected_reward": expected,
                "full_information_mip": mip,
                "optimal_policy_realized_rewards": realized,
                "verified": True,
                "states": planner.states,
            }
        )
        print(f"Verified {instance.family} / level {instance.level} / seed {instance.seed}", flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    checks = validate(args.suite)
    write_json(args.out, checks)
    print(json.dumps({"instances": len(checks), "all_verified": True}))
