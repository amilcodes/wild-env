"""Public-information baselines that emit ordinary contingent plans.

Every baseline sees only what a model sees (the public instance and, once
revealed, the forecast) and is scored by exactly the same executor. Dispatch
rules run tick by tick in both worlds; because they act on the belief, their two
traces agree until the reveal and convert to an admissible plan.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from aeolus.agents.operations import Instance, initial_state, legal_jobs, reveals, transition, with_forecast
from aeolus.ladder.plan import evaluate_plan, plan_from_starts

BASELINES = ("hold", "greedy", "deadline", "search")


def _belief_value(instance: Instance, state, j: int) -> float:
    job = instance.jobs[j]
    if state.forecast >= 0:
        return float(job.values[state.forecast])
    return sum(p * v for p, v in zip(instance.prior, job.values, strict=True))


Score = Callable[[Instance, object, int, int], float]


def _rollout(instance: Instance, score: Score):
    """Run a per-resource dispatch rule in both worlds and return a plan."""
    by_world, reveal_tick = [], instance.horizon
    for world in (0, 1):
        state = initial_state(instance)
        starts = []
        world_reveal = instance.horizon
        if reveals(instance, state):
            state, world_reveal = with_forecast(state, world), 0
        while state.tick < instance.horizon:
            action = [-1] * len(instance.resources)
            taken, water = set(), state.water
            for r in range(len(instance.resources)):
                options = [
                    j
                    for j in legal_jobs(instance, state, r)
                    if j >= 0 and j not in taken and instance.jobs[j].water <= water
                ]
                scored = [(score(instance, state, r, j), -instance.jobs[j].id, j) for j in options]
                scored = [s for s in scored if s[0] > 0]
                if scored:
                    _, _, j = max(scored)
                    action[r] = j
                    taken.add(j)
                    water -= instance.jobs[j].water
                    starts.append((state.tick, r, j))
            state = transition(instance, state, tuple(action))
            if reveals(instance, state):
                state, world_reveal = with_forecast(state, world), state.tick
        by_world.append(starts)
        reveal_tick = world_reveal
    return plan_from_starts(instance, by_world, reveal_tick)


def greedy_score(instance, state, r, j):
    job = instance.jobs[j]
    return (_belief_value(instance, state, j) - job.cost) / job.durations[r]


def deadline_score(instance, state, r, j):
    job = instance.jobs[j]
    return (_belief_value(instance, state, j) - job.cost) / max(1, job.deadline - state.tick)


def _unlock_value(instance: Instance) -> list[float]:
    """Expected value a job (transitively) unlocks: lets search consider ACCESS jobs."""
    positions = {job.id: j for j, job in enumerate(instance.jobs)}
    children: dict[int, list[int]] = {}
    for j, job in enumerate(instance.jobs):
        for p in job.prerequisites:
            children.setdefault(positions[p], []).append(j)

    def own(j):
        return sum(p * v for p, v in zip(instance.prior, instance.jobs[j].values, strict=True))

    memo: dict[int, float] = {}

    def total(j):
        if j not in memo:
            memo[j] = own(j) + 0.5 * sum(total(c) for c in children.get(j, []))
        return memo[j]

    return [total(j) for j in range(len(instance.jobs))]


def search_plan(instance: Instance, samples: int = 256, seed: int = 0):
    """Randomized priority-rule search (GRASP-style) over public information.

    Each sample perturbs job priorities (including value unlocked through
    prerequisites) and runs the dispatch rule in both worlds. The plan with the
    best expected value is kept. Search uses only the public model.
    """
    rng = random.Random(seed)
    unlock = _unlock_value(instance)
    best, best_value = None, -math.inf
    for sample in range(samples):
        sigma = 0.0 if sample == 0 else 0.6
        noise = [math.exp(rng.gauss(0, sigma)) for _ in instance.jobs]
        horizon_weight = rng.choice([0.0, 0.5, 1.0])

        def score(inst, state, r, j, noise=noise, horizon_weight=horizon_weight):
            job = inst.jobs[j]
            base = _belief_value(inst, state, j) if state.forecast >= 0 else unlock[j]
            urgency = 1.0 + horizon_weight / max(1, job.deadline - state.tick)
            return (base - job.cost) * noise[j] * urgency / job.durations[r]

        plan = _rollout(instance, score)
        value = evaluate_plan(instance, plan).expected_value
        if value > best_value:
            best, best_value = plan, value
    return best


def baseline_plan(instance: Instance, name: str):
    if name == "hold":
        return {"rationale": "", "plan": []}
    if name == "greedy":
        return _rollout(instance, greedy_score)
    if name == "deadline":
        return _rollout(instance, deadline_score)
    if name == "search":
        return search_plan(instance)
    raise ValueError(f"unknown baseline {name}")
