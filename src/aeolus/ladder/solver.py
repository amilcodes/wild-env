"""Exact information-matched optimum as one two-world mixed-integer program.

Variables ``x[w, j, r, t]`` start job ``j`` on resource ``r`` at tick ``t`` in
forecast world ``w``. Per world the constraints are the v2 rules: each job at
most once, one job per resource at a time, prerequisites complete at start,
finish by min(deadline, horizon), and total water within the budget (water is
only consumed, so the total bound is equivalent to the running bound).

Information is modelled exactly. Before the forecast is revealed the desk knows
nothing new and the dynamics are deterministic, so the optimal policy is an
open-loop schedule until the reveal and may branch on the world after it. With
``R[t]`` = 1 once a survey has completed by tick ``t``, non-anticipativity is

    |x[0, j, r, t] - x[1, j, r, t]| <= R[t]       for every t < bulletin tick.

Survey timing is therefore a decision of the MIP itself, so one solve gives the
optimum over all admissible policies, including the value of information.
``mode="clairvoyant"`` drops the coupling (perfect information upper bound);
``mode="open_loop"`` couples every tick (no adaptation, lower reference).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from aeolus.agents.operations import Instance
from aeolus.ladder.plan import evaluate_plan, plan_from_starts

MODES = ("information", "clairvoyant", "open_loop")


@dataclass
class Reference:
    mode: str
    value: float
    plan: dict[str, Any] | None
    reveal_tick: int | None
    variables: int
    constraints: int
    seconds: float

    def public(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "value": self.value,
            "reveal_tick": self.reveal_tick,
            "variables": self.variables,
            "constraints": self.constraints,
            "seconds": round(self.seconds, 3),
        }


def uncertain(instance: Instance) -> bool:
    return any(job.values[0] != job.values[1] for job in instance.jobs)


def _starts(instance: Instance) -> list[tuple[int, int, int]]:
    out = []
    for j, job in enumerate(instance.jobs):
        for r, duration in enumerate(job.durations):
            if duration <= 0:
                continue
            for t in range(job.release, min(job.deadline, instance.horizon) - duration + 1):
                out.append((j, r, t))
    return out


def solve(instance: Instance, mode: str = "information", time_limit: float = 600.0) -> Reference:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    began = time.monotonic()
    starts = _starts(instance)
    K = len(starts)
    worlds = (0, 1) if uncertain(instance) else (0,)
    W = len(worlds)
    positions = {job.id: j for j, job in enumerate(instance.jobs)}
    by_job: dict[int, list[int]] = {}
    for k, (j, _, _) in enumerate(starts):
        by_job.setdefault(j, []).append(k)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(row: dict[int, float], lo: float, hi: float) -> None:
        index = len(lower)
        for column, value in row.items():
            rows.append(index)
            cols.append(column)
            data.append(value)
        lower.append(lo)
        upper.append(hi)

    for w in range(W):
        offset = w * K
        for ks in by_job.values():
            add({offset + k: 1.0 for k in ks}, -np.inf, 1.0)
        for r in range(len(instance.resources)):
            for tick in range(instance.horizon):
                row = {
                    offset + k: 1.0
                    for k, (j, rr, t) in enumerate(starts)
                    if rr == r and t <= tick < t + instance.jobs[j].durations[r]
                }
                if len(row) > 1:
                    add(row, -np.inf, 1.0)
        for j, job in enumerate(instance.jobs):
            for prerequisite in job.prerequisites:
                p = positions[prerequisite]
                for tick in sorted({starts[k][2] for k in by_job.get(j, [])}):
                    row = {offset + k: 1.0 for k in by_job[j] if starts[k][2] == tick}
                    for k in by_job.get(p, []):
                        _, rr, t = starts[k]
                        if t + instance.jobs[p].durations[rr] <= tick:
                            row[offset + k] = -1.0
                    add(row, -np.inf, 0.0)
        if instance.water >= 0:
            add(
                {
                    offset + k: float(instance.jobs[j].water)
                    for k, (j, _, _) in enumerate(starts)
                    if instance.jobs[j].water
                },
                -np.inf,
                float(instance.water),
            )

    if W == 2 and mode != "clairvoyant":
        cutoff = instance.horizon if mode == "open_loop" else instance.bulletin_tick
        surveys = [k for k, (j, _, _) in enumerate(starts) if instance.jobs[j].kind == "SURVEY"]
        for k, (j, r, t) in enumerate(starts):
            if t >= cutoff:
                continue
            revealed = (
                {}
                if mode == "open_loop"
                else {
                    s: -1.0
                    for s in surveys
                    if starts[s][2] + instance.jobs[starts[s][0]].durations[starts[s][1]] <= t
                }
            )
            add({k: 1.0, K + k: -1.0, **revealed}, -np.inf, 0.0)
            add({K + k: 1.0, k: -1.0, **revealed}, -np.inf, 0.0)

    probabilities = instance.prior if W == 2 else (1.0,)
    objective = np.zeros(W * K)
    for w in range(W):
        for k, (j, _, _) in enumerate(starts):
            job = instance.jobs[j]
            objective[w * K + k] = -probabilities[w] * (job.values[worlds[w]] - job.cost)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(lower), W * K)).tocsc()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(
            objective,
            integrality=np.ones(W * K),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
            options={"mip_rel_gap": 0.0, "time_limit": time_limit, "threads": 1, "disp": False},
        )
    if result.status != 0 or result.x is None or (result.mip_gap or 0.0) > 1e-9:
        raise RuntimeError(f"MIP did not prove optimality (status={result.status}: {result.message})")
    value = float(-result.fun)
    chosen = np.round(result.x).astype(int)
    by_world = [
        [(starts[k][2], starts[k][1], starts[k][0]) for k in range(K) if chosen[w * K + k]] for w in range(W)
    ]
    if W == 1:
        by_world = [by_world[0], by_world[0]]

    plan, reveal_tick = None, None
    if mode == "information":
        reveal_tick = instance.bulletin_tick if W == 2 else instance.horizon
        for t, r, j in by_world[0]:
            if instance.jobs[j].kind == "SURVEY":
                reveal_tick = min(reveal_tick, t + instance.jobs[j].durations[r])
        reveal_tick = min(reveal_tick, instance.horizon)
        plan = plan_from_starts(instance, by_world, reveal_tick)
        replayed = evaluate_plan(instance, plan)
        if abs(replayed.expected_value - value) > 1e-6 or replayed.violations:
            raise AssertionError(
                f"optimal plan replays to {replayed.expected_value} (MIP {value}); "
                f"violations={replayed.violations}"
            )
    return Reference(mode, value, plan, reveal_tick, W * K, len(lower), time.monotonic() - began)
