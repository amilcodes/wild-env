"""Finite wildfire response planning with an exact, information-matched reference.

This is an operational scheduling abstraction, not the continuous fire solver.
All uncertainty is an exogenous two-world forecast; actions never change it.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from aeolus.agents.contract import digest, validate_dispatch

PROTOCOL = "wildfire-operations-v2"
PROMPT = """You operate a synthetic wildfire response desk. Maximize final net protected
asset value: sum the realized value of completed protection jobs, minus the costs
of every job started. These are abstract planning units, not dollars or lives.

Each decision advances ONE planning tick (10 minutes). Submit exactly one JSON
dispatch with decision_id, one assignment per resource, rationale, and memory.
Use only that resource's legal_tasks. Task 0 means hold/continue. Jobs are unique:
do not assign the same nonzero job to multiple resources. Joint water use must
not exceed remaining water; each job consumes its stated water when STARTED.
Invalid dispatches hold ALL resources for one tick. There are no free retries.

Jobs are nonpreemptive. duration_by_resource includes travel, work and turnaround.
A job started at tick t with duration d finishes at t+d, earns value if it finishes
no later than its deadline, and frees the resource at that tick. A prerequisite
must already be completed at the START of a job. Jobs with zero value can unlock
valuable dependent jobs. Release ticks and deadlines are absolute, not relative.
Resources cannot work on two jobs at once. A large immediate value can block
several better jobs, consume scarce water, or miss a prerequisite chain.

The forecast has two possible wind corridors, alpha and beta. value_by_forecast
states each job's protection value in each corridor; all possible values and the
prior probabilities are public. The actual corridor is fixed before dispatch and
is hidden until the scheduled bulletin OR a completed SURVEY, whichever is first.
The bulletin/survey result is exact and persists. Before that, plan using the
stated probabilities; do not pretend to know the actual corridor. Surveying uses
time and has a cost. Some tasks require preserving flexibility for later news.

legal_tasks filters only immediate feasibility; it does NOT choose a good plan or
reserve supplies for later work. Task IDs and display order carry no priority.
Terminal reward, the reference planner, and hidden forecast are not available to
you. You have only this public packet, the last receipt and up to 2000 characters
of carried memory. Return a short operational rationale, not a reasoning transcript.
"""


@dataclass(frozen=True)
class Job:
    id: int
    label: str
    kind: str
    durations: tuple[int, ...]
    release: int
    deadline: int
    prerequisites: tuple[int, ...]
    water: int
    cost: int
    values: tuple[int, int]


@dataclass(frozen=True)
class Instance:
    family: str
    seed: int
    level: int
    horizon: int
    resources: tuple[str, ...]
    water: int
    bulletin_tick: int
    jobs: tuple[Job, ...]
    prior: tuple[float, float] = (0.5, 0.5)

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class State:
    tick: int
    completed: int
    active: tuple[tuple[int, int], ...]  # job position (-1 for idle), absolute finish tick
    water: int
    cost: int
    forecast: int  # -1 unknown, otherwise 0/1


def initial_state(instance: Instance) -> State:
    return State(0, 0, tuple((-1, 0) for _ in instance.resources), instance.water, 0, -1)


def legal_jobs(instance: Instance, state: State, resource: int) -> tuple[int, ...]:
    if state.active[resource][0] >= 0 or state.tick >= instance.horizon:
        return (-1,)
    occupied = {job for job, _ in state.active}
    positions = {job.id: j for j, job in enumerate(instance.jobs)}
    legal = [-1]
    for j, job in enumerate(instance.jobs):
        if state.completed & (1 << j) or j in occupied or job.durations[resource] <= 0:
            continue
        if job.kind == "SURVEY" and state.forecast >= 0:
            continue
        if job.release > state.tick or state.tick + job.durations[resource] > min(
            job.deadline, instance.horizon
        ):
            continue
        if job.water > state.water or any(
            not state.completed & (1 << positions[p]) for p in job.prerequisites
        ):
            continue
        legal.append(j)
    return tuple(legal)


def joint_actions(instance: Instance, state: State):
    choices = [legal_jobs(instance, state, r) for r in range(len(instance.resources))]
    for action in itertools.product(*choices):
        assigned = [j for j in action if j >= 0]
        if (
            len(set(assigned)) == len(assigned)
            and sum(instance.jobs[j].water for j in assigned) <= state.water
        ):
            yield action


def transition(instance: Instance, state: State, action: tuple[int, ...]) -> State:
    """Advance public mechanics; leave a newly revealed forecast unknown for branching."""
    active = list(state.active)
    water, cost, completed = state.water, state.cost, state.completed
    for r, j in enumerate(action):
        if j >= 0:
            job = instance.jobs[j]
            active[r] = (j, state.tick + job.durations[r])
            water -= job.water
            cost += job.cost
    tick = state.tick + 1
    for r, (j, finish) in enumerate(active):
        if j >= 0 and finish <= tick:
            completed |= 1 << j
            active[r] = (-1, 0)
    return State(tick, completed, tuple(active), water, cost, state.forecast)


def reveals(instance: Instance, state: State) -> bool:
    return state.forecast < 0 and (
        state.tick >= instance.bulletin_tick
        or any(job.kind == "SURVEY" and state.completed & (1 << j) for j, job in enumerate(instance.jobs))
    )


def with_forecast(state: State, forecast: int) -> State:
    return State(state.tick, state.completed, state.active, state.water, state.cost, forecast)


def payoff(instance: Instance, state: State, forecast: int) -> int:
    return (
        sum(job.values[forecast] for j, job in enumerate(instance.jobs) if state.completed & (1 << j))
        - state.cost
    )


class Planner:
    """Exact Bellman reference. Unknown forecast branches by prior, never by hidden truth."""

    def __init__(self, instance: Instance, max_states: int = 500_000):
        self.instance = instance
        self.max_states = max_states
        self.states = 0

        @cache
        def value(state):
            self.states += 1
            if self.states > max_states:
                raise RuntimeError(
                    "exact planner state budget exceeded; do not report an approximate optimum"
                )
            if reveals(instance, state):
                return sum(p * value(with_forecast(state, w)) for w, p in enumerate(instance.prior))
            if state.tick == instance.horizon:
                if state.forecast >= 0:
                    return float(payoff(instance, state, state.forecast))
                return sum(p * payoff(instance, state, w) for w, p in enumerate(instance.prior))
            return max(
                value(transition(instance, state, action)) for action in joint_actions(instance, state)
            )

        self.value = value

    def action(self, state):
        return max(
            joint_actions(self.instance, state), key=lambda a: self.value(transition(self.instance, state, a))
        )

    def regret(self, state, action):
        return max(0.0, self.value(state) - self.value(transition(self.instance, state, action)))


class OperationsDesk:
    def __init__(self, instance: Instance, forecast: int):
        if forecast not in (0, 1):
            raise ValueError("forecast must be 0 or 1")
        self.instance, self.hidden_forecast = instance, forecast
        self.state = initial_state(instance)
        self.memory, self.receipt = "", None
        self.invalid = 0

    @property
    def done(self):
        return self.state.tick == self.instance.horizon

    def observation(self):
        instance, state = self.instance, self.state
        jobs = []
        for j, job in enumerate(instance.jobs):
            jobs.append(
                {
                    "id": job.id,
                    "label": job.label,
                    "kind": job.kind,
                    "capacity": 1,
                    "duration_by_resource": dict(zip(instance.resources, job.durations, strict=True)),
                    "release_tick": job.release,
                    "deadline_tick": job.deadline,
                    "prerequisites": list(job.prerequisites),
                    "water_required": job.water,
                    "start_cost": job.cost,
                    "value_by_forecast": dict(zip(("alpha", "beta"), job.values, strict=True)),
                    "completed": bool(state.completed & (1 << j)),
                    "started": any(a == j for a, _ in state.active) or bool(state.completed & (1 << j)),
                }
            )
        resources = []
        for r, name in enumerate(instance.resources):
            active, finish = state.active[r]
            resources.append(
                {
                    "id": name,
                    "active_task": instance.jobs[active].id if active >= 0 else 0,
                    "available_at_tick": finish if active >= 0 else state.tick,
                    "legal_tasks": [
                        instance.jobs[j].id if j >= 0 else 0 for j in legal_jobs(instance, state, r)
                    ],
                }
            )
        return {
            "protocol": PROTOCOL,
            "decision_id": f"{instance.identity[:16]}:{state.tick}",
            "tick": state.tick,
            "tick_minutes": 10,
            "horizon_ticks": instance.horizon,
            "water_remaining": state.water,
            "forecast_prior": dict(zip(("alpha", "beta"), instance.prior, strict=True)),
            "forecast": ("alpha", "beta")[state.forecast] if state.forecast >= 0 else "unknown",
            "scheduled_bulletin_tick": instance.bulletin_tick,
            "resources": resources,
            "tasks": [{"id": 0, "kind": "HOLD", "capacity": 99}, *jobs],
            "memory": self.memory,
            "last_dispatch": self.receipt,
        }

    def step(self, response: Any, *, failure: str | None = None):
        if self.done:
            raise RuntimeError("episode complete")
        observation = self.observation()
        error = failure
        action = tuple(-1 for _ in self.instance.resources)
        if error is None:
            try:
                assignments = validate_dispatch(response, observation)
                positions = {job.id: j for j, job in enumerate(self.instance.jobs)}
                proposed = tuple(
                    positions[assignments[r]] if assignments[r] else -1 for r in self.instance.resources
                )
                if sum(self.instance.jobs[j].water for j in proposed if j >= 0) > self.state.water:
                    raise ValueError("joint dispatch exceeds available water")
                action = proposed
            except ValueError as exc:
                error = str(exc)
        if error is None:
            self.memory = response["memory"]
        else:
            self.invalid += 1
        self.state = transition(self.instance, self.state, action)
        if reveals(self.instance, self.state):
            self.state = with_forecast(self.state, self.hidden_forecast)
        self.receipt = {
            "decision_id": observation["decision_id"],
            "valid": error is None,
            "error": error,
            "actions": {
                r: self.instance.jobs[j].id if j >= 0 else 0
                for r, j in zip(self.instance.resources, action, strict=True)
            },
        }
        return action, self.receipt


def dispatch(instance: Instance, state: State, observation, action):
    return {
        "decision_id": observation["decision_id"],
        "assignments": [
            {"resource_id": r, "task_id": instance.jobs[j].id if j >= 0 else 0}
            for r, j in zip(instance.resources, action, strict=True)
        ],
        "rationale": "",
        "memory": "",
    }


def greedy_action(instance: Instance, state: State, *, deadline=False):
    """Joint immediate-value baseline; no hidden world, search, or task-order tie advantage."""

    def utility(action):
        total = 0.0
        for r, j in enumerate(action):
            if j < 0:
                continue
            job = instance.jobs[j]
            value = (
                job.values[state.forecast]
                if state.forecast >= 0
                else sum(p * v for p, v in zip(instance.prior, job.values, strict=True))
            )
            net = value - job.cost
            if deadline:
                total += max(0, net) / max(1, job.deadline - state.tick)
            else:
                total += max(0, net) / job.durations[r]
        return total

    return max(joint_actions(instance, state), key=utility)


FAMILIES = ("deadline", "prerequisite", "water", "information", "forecast", "compound")


class ShortPlanner:
    """Two-step search with a myopic continuation, using only the public belief."""

    def __init__(self, instance):
        self.instance = instance

        @cache
        def value(state, depth):
            if reveals(instance, state):
                return sum(p * value(with_forecast(state, w), depth) for w, p in enumerate(instance.prior))
            if state.tick == instance.horizon:
                return (
                    float(payoff(instance, state, state.forecast))
                    if state.forecast >= 0
                    else sum(p * payoff(instance, state, w) for w, p in enumerate(instance.prior))
                )
            if depth == 0:
                return value(transition(instance, state, greedy_action(instance, state)), 0)
            return max(
                value(transition(instance, state, a), depth - 1) for a in joint_actions(instance, state)
            )

        self.value = value

    def action(self, state):
        return max(
            joint_actions(self.instance, state),
            key=lambda a: self.value(transition(self.instance, state, a), 1),
        )


def instance_from_dict(raw):
    values = dict(raw)
    values["resources"] = tuple(values["resources"])
    values["prior"] = tuple(values["prior"])
    values["jobs"] = tuple(
        Job(
            **{
                **j,
                "durations": tuple(j["durations"]),
                "prerequisites": tuple(j["prerequisites"]),
                "values": tuple(j["values"]),
            }
        )
        for j in values["jobs"]
    )
    return Instance(**values)


def generate(family: str, seed: int, level: int = 1) -> Instance:
    """Versioned templates vary labels, values and schedules without inspecting model outcomes."""
    if family not in FAMILIES or level not in (1, 2, 3):
        raise ValueError("unknown family or level")
    if level == 3:
        from aeolus.agents.operations_challenge import generate_challenge

        return generate_challenge(family, seed)
    rng = random.Random(seed)
    horizon = 7 + level
    resource_names = tuple(rng.sample(["team-north", "team-west", "team-red", "team-blue"], 2))
    ids = rng.sample(range(11, 198), 16)
    labels = rng.sample(
        [
            "orchard",
            "substation",
            "ranch",
            "tower",
            "school",
            "village",
            "relay",
            "depot",
            "bridge",
            "clinic",
            "intake",
            "shelter",
            "mill",
            "station",
            "farm",
            "plant",
        ],
        16,
    )
    jobs = []

    def add(durations, deadline, values, *, release=0, prerequisites=(), water=0, cost=1, kind="PROTECT"):
        j = len(jobs)
        jobs.append(
            Job(
                ids[j],
                labels[j],
                kind,
                durations,
                release,
                deadline,
                tuple(prerequisites),
                water,
                cost,
                values,
            )
        )
        return ids[j]

    # Two non-interchangeable teams, plus cross-trained jobs. Public duration tables
    # define feasibility; resource names intentionally do not encode capabilities.
    a = rng.randint(12, 20)
    long = rng.randint(3, 4)
    value = rng.randint(22, 58)
    add((long, 0), long, (value, value), cost=rng.randint(1, 3))
    value = rng.randint(14, 30)
    add((1, rng.randint(2, 3)), 2, (value, value))
    value = rng.randint(18, 38)
    add((2, 0), rng.randint(4, 5), (value, value), release=1)
    value = rng.randint(16, 34)
    add((0, 2), rng.randint(3, 4), (value, value))
    if family in {"prerequisite", "compound"}:
        access = add((rng.randint(1, 2), 2), 4, (0, 0), kind="ACCESS")
        value = rng.randint(45, 95)
        add((0, 2), rng.randint(5, 7), (value, value), prerequisites=(access,))
        value = rng.randint(35, 75)
        add((2, 0), horizon, (value, value), prerequisites=(access,), release=rng.randint(2, 4))
    elif family in {"information", "forecast"}:
        add((1, 1), 2, (0, 0), cost=3, kind="SURVEY")
        deadline = 5 if family == "information" else rng.randint(6, 7)
        add((3, 0), deadline, (a * 5, 0), release=1, water=2)
        add((0, 3), deadline, (0, a * 5), release=1, water=2)
    else:
        value = rng.randint(25, 65)
        add((2, 3), 5, (value, value), release=2, water=2)
        value = rng.randint(40, 80)
        add((0, 2), horizon, (value, value), release=rng.randint(3, 5), water=2)
        value = rng.randint(40, 80)
        add((2, 0), horizon, (value, value), release=rng.randint(3, 5), water=2)
    if family in {"water", "compound"}:
        add((2, 2), 5, (a * 2, a * 2), water=2)
        add((2, 3), horizon, (a * 5, a * 5), release=4, water=3)
    else:
        add((1, 2), horizon, (a, a), release=4)
    if level == 2:
        value = rng.randint(20, 40)
        precursor = add((2, 1), horizon - 2, (value, value), release=rng.randint(1, 3), water=1)
        add((1, 2), horizon, (a * 3, a * 3), release=4, prerequisites=(precursor,))
        add((3, 2), horizon, (a * 2, a * 2), release=3)
    if family == "compound":
        add((1, 1), 3, (0, 0), cost=3, kind="SURVEY")
        add((2, 2), 6, (a * 4, 0), release=1, water=2)
        add((2, 2), 6, (0, a * 4), release=1, water=2)
    # Swap scenario labels independently of numerical IDs and listing order.
    if rng.choice([False, True]):
        jobs = [
            Job(
                j.id,
                j.label,
                j.kind,
                j.durations,
                j.release,
                j.deadline,
                j.prerequisites,
                j.water,
                j.cost,
                j.values[::-1],
            )
            for j in jobs
        ]
    if rng.choice([False, True]):
        jobs = [
            Job(
                j.id,
                j.label,
                j.kind,
                j.durations[::-1],
                j.release,
                j.deadline,
                j.prerequisites,
                j.water,
                j.cost,
                j.values,
            )
            for j in jobs
        ]
    rng.shuffle(jobs)
    return Instance(
        family,
        seed,
        level,
        horizon,
        resource_names,
        (
            rng.randint(3, 5)
            if family in {"water", "compound"}
            else 2
            if family in {"information", "forecast"}
            else 6
        ),
        4 if family in {"information", "compound"} else 3,
        tuple(jobs),
    )
