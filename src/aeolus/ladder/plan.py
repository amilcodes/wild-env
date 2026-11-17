"""Contingent incident action plans: schema, parsing, execution and scoring.

A plan is a flat list of entries ``{resource, job, start, branch}``. ``branch`` is
``both`` (executed whatever the forecast) or ``alpha``/``beta`` (executed only in
that forecast world, and only once the forecast is known). Execution reuses the
v2 tick dynamics (:func:`aeolus.agents.operations.transition`) so the plan mode,
the RL environment and the v2 Bellman reference share one set of rules.

Execution rules (also stated verbatim to models in :mod:`aeolus.ladder.prompt`):

1. Each resource works through its entries for the current world in order of
   ``start`` (ties: listing order). An entry begins at the first tick that is
   >= ``start`` and at which the resource is free.
2. At that tick the job must be startable: released, not already started, all
   prerequisites complete, finishable by min(deadline, horizon), enough water,
   the resource capable, and (for SURVEY) the forecast still unknown. Otherwise
   the entry is skipped and recorded as a violation; the resource moves on.
3. Within a tick, resources are processed in the listed resource order.
4. Forecast-specific entries must have ``start`` >= the reveal tick (bulletin, or
   first completed survey). Earlier ones would use information the desk does not
   have yet; they are rejected. This makes every plan non-anticipative, so the
   information-matched optimum bounds every admissible plan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from aeolus.agents.operations import (
    Instance,
    initial_state,
    legal_jobs,
    payoff,
    reveals,
    transition,
    with_forecast,
)

WORLDS = ("alpha", "beta")
BRANCHES = ("both", *WORLDS)
MAX_ENTRIES = 400

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "job": {"type": "integer"},
                    "start": {"type": "integer"},
                    "branch": {"type": "string", "enum": list(BRANCHES)},
                },
                "required": ["resource", "job", "start", "branch"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "plan"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Entry:
    resource: int
    job: int  # position in instance.jobs
    start: int
    branch: int  # -1 both, 0 alpha, 1 beta
    index: int  # listing order in the submitted plan


@dataclass
class WorldTrace:
    world: int
    value: float
    starts: list[tuple[int, int, int]]  # (tick, resource, job position)
    reveal_tick: int


@dataclass
class PlanOutcome:
    expected_value: float
    world_values: tuple[float, float]
    reveal_tick: int
    entries: int
    violations: list[dict[str, Any]] = field(default_factory=list)
    unexecuted: int = 0
    traces: tuple[WorldTrace, ...] = ()
    parse_error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "expected_value": self.expected_value,
            "world_values": list(self.world_values),
            "reveal_tick": self.reveal_tick,
            "entries": self.entries,
            "violations": self.violations,
            "unexecuted": self.unexecuted,
            "parse_error": self.parse_error,
            "starts": {
                WORLDS[t.world]: [{"tick": k, "resource": r, "job": j} for k, r, j in t.starts]
                for t in self.traces
            },
        }


def parse_plan(instance: Instance, response: Any) -> tuple[list[Entry], list[dict[str, Any]], str | None]:
    """Structural parsing. Malformed entries are dropped and reported; nothing is repaired."""
    if not isinstance(response, dict) or not isinstance(response.get("plan"), list):
        return [], [], "response is not an object with a plan list"
    raw = response["plan"]
    if len(raw) > MAX_ENTRIES:
        return [], [], f"plan has more than {MAX_ENTRIES} entries"
    resources = {name: r for r, name in enumerate(instance.resources)}
    jobs = {job.id: j for j, job in enumerate(instance.jobs)}
    entries, violations = [], []
    for index, item in enumerate(raw):
        reason = None
        if not isinstance(item, dict) or set(item) != {"resource", "job", "start", "branch"}:
            reason = "malformed entry"
        elif item["resource"] not in resources:
            reason = "unknown resource"
        elif type(item["job"]) is not int or item["job"] not in jobs:
            reason = "unknown job"
        elif type(item["start"]) is not int or item["start"] < 0:
            reason = "start must be a non-negative integer"
        elif item["branch"] not in BRANCHES:
            reason = "unknown branch"
        if reason:
            violations.append({"index": index, "reason": reason})
            continue
        entries.append(
            Entry(
                resources[item["resource"]],
                jobs[item["job"]],
                item["start"],
                BRANCHES.index(item["branch"]) - 1,
                index,
            )
        )
    return entries, violations, None


def _skip_reason(instance: Instance, state, r: int, j: int, taken: set[int], water: int) -> str:
    job = instance.jobs[j]
    if job.durations[r] <= 0:
        return "resource cannot do this job"
    if state.completed & (1 << j) or any(a == j for a, _ in state.active) or j in taken:
        return "job already started"
    if job.kind == "SURVEY" and state.forecast >= 0:
        return "survey after forecast is known"
    if job.release > state.tick:
        return "job not released yet"
    if state.tick + job.durations[r] > min(job.deadline, instance.horizon):
        return "cannot finish by deadline"
    positions = {x.id: k for k, x in enumerate(instance.jobs)}
    if any(not state.completed & (1 << positions[p]) for p in job.prerequisites):
        return "prerequisite not complete"
    if job.water > water:
        return "not enough water"
    return "not startable"


def _simulate(instance: Instance, entries: list[Entry], world: int):
    queues = [deque() for _ in instance.resources]
    for entry in sorted(entries, key=lambda e: (e.start, e.index)):
        queues[entry.resource].append(entry)
    state = initial_state(instance)
    starts, skipped = [], []
    reveal_tick = instance.horizon
    if reveals(instance, state):
        state, reveal_tick = with_forecast(state, world), 0
    while state.tick < instance.horizon:
        action = [-1] * len(instance.resources)
        taken: set[int] = set()
        water = state.water
        for r in range(len(instance.resources)):
            if state.active[r][0] >= 0:
                continue
            while queues[r] and queues[r][0].start <= state.tick:
                entry = queues[r].popleft()
                j = entry.job
                if j in legal_jobs(instance, state, r) and j not in taken and instance.jobs[j].water <= water:
                    action[r] = j
                    taken.add(j)
                    water -= instance.jobs[j].water
                    starts.append((state.tick, r, j))
                    break
                skipped.append(
                    {
                        "index": entry.index,
                        "tick": state.tick,
                        "reason": _skip_reason(instance, state, r, j, taken, water),
                    }
                )
        state = transition(instance, state, tuple(action))
        if reveals(instance, state):
            state, reveal_tick = with_forecast(state, world), state.tick
    unexecuted = sum(len(q) for q in queues)
    return state, starts, skipped, reveal_tick, unexecuted


def execute_plan(instance: Instance, entries: list[Entry]) -> PlanOutcome:
    """Execute a parsed plan in both forecast worlds and score it exactly."""
    shared = [e for e in entries if e.branch < 0]
    _, _, _, reveal_tick, _ = _simulate(instance, shared, 0)
    violations = []
    admissible = []
    for entry in entries:
        if entry.branch >= 0 and entry.start < reveal_tick:
            violations.append(
                {"index": entry.index, "tick": entry.start, "reason": "forecast-specific entry before reveal"}
            )
        else:
            admissible.append(entry)
    traces, unexecuted = [], 0
    seen_skips: dict[tuple[int, str], dict[str, Any]] = {}
    for world in (0, 1):
        own = [e for e in admissible if e.branch in (-1, world)]
        state, starts, skipped, world_reveal, left = _simulate(instance, own, world)
        if world_reveal != reveal_tick:
            raise AssertionError("pre-reveal execution diverged between worlds")
        traces.append(WorldTrace(world, float(payoff(instance, state, world)), starts, reveal_tick))
        unexecuted += left
        for skip in skipped:
            seen_skips.setdefault((skip["index"], skip["reason"]), {**skip, "worlds": []})["worlds"].append(
                WORLDS[world]
            )
    violations.extend(seen_skips.values())
    values = (traces[0].value, traces[1].value)
    expected = sum(p * v for p, v in zip(instance.prior, values, strict=True))
    return PlanOutcome(
        expected,
        values,
        reveal_tick,
        len(entries),
        sorted(violations, key=lambda v: v["index"]),
        unexecuted,
        tuple(traces),
    )


def evaluate_plan(instance: Instance, response: Any) -> PlanOutcome:
    entries, violations, error = parse_plan(instance, response)
    if error:
        outcome = execute_plan(instance, [])
        outcome.parse_error = error
        return outcome
    outcome = execute_plan(instance, entries)
    outcome.violations = sorted(violations + outcome.violations, key=lambda v: v["index"])
    outcome.entries = len(response["plan"])
    return outcome


def plan_from_starts(instance: Instance, starts_by_world, reveal_tick: int) -> dict[str, Any]:
    """Build a contingent plan from per-world start lists that agree before ``reveal_tick``."""
    plan = []
    for k, r, j in sorted(starts_by_world[0]):
        if k < reveal_tick:
            plan.append((k, r, j, "both"))
    for world, starts in enumerate(starts_by_world):
        for k, r, j in sorted(starts):
            if k >= reveal_tick:
                plan.append((k, r, j, WORLDS[world]))
    pre = [s for s in starts_by_world[0] if s[0] < reveal_tick]
    if sorted(pre) != sorted(s for s in starts_by_world[1] if s[0] < reveal_tick):
        raise ValueError("schedules differ before the reveal tick; they are not non-anticipative")
    plan.sort(key=lambda x: (x[0], x[1], BRANCHES.index(x[3])))
    return {
        "rationale": "",
        "plan": [
            {"resource": instance.resources[r], "job": instance.jobs[j].id, "start": k, "branch": b}
            for k, r, j, b in plan
        ],
    }
