"""Inspectable non-learning baselines using the operational belief/task set."""

from __future__ import annotations

from math import inf

from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import TaskKind, task_distance_min


def no_aerial_action(sim: AeolusSimulator) -> dict[str, int]:
    return {resource.resource_id: 0 for resource in sim.state.resources}


def nearest_feasible(sim: AeolusSimulator) -> dict[str, int]:
    actions: dict[str, int] = {}
    for resource in sim.state.resources:
        candidate = min(
            (task for task in sim.tasks if task.compatible(resource)),
            key=lambda task: task_distance_min(resource, task, sim.config.cell_size_m),
            default=sim.tasks[0],
        )
        actions[resource.resource_id] = candidate.index
    return actions


def anchor_flank(sim: AeolusSimulator) -> dict[str, int]:
    """Doctrine-inspired line building: favor connected, downwind retardant support."""

    actions: dict[str, int] = {}
    for resource in sim.state.resources:
        best_score, best = -inf, sim.tasks[0]
        for task in sim.tasks:
            if not task.compatible(resource):
                continue
            distance = task_distance_min(resource, task, sim.config.cell_size_m)
            line_bonus = 2.2 if task.kind in (TaskKind.RETARDANT, TaskKind.REINFORCE) else 0.0
            ground_bonus = (1.0 - task.ground_dependency) * 0.8
            score = task.expected_value + line_bonus + ground_bonus - 0.08 * distance
            if score > best_score:
                best_score, best = score, task
        actions[resource.resource_id] = best.index
    return actions


def greedy_value(sim: AeolusSimulator) -> dict[str, int]:
    """Greedy maximum marginal task score with one task reserved per resource."""

    remaining = {task.index for task in sim.tasks}
    actions: dict[str, int] = {}
    for resource in sim.state.resources:
        best_score, best = -inf, sim.tasks[0]
        for task in sim.tasks:
            if task.index not in remaining or not task.compatible(resource):
                continue
            travel = task_distance_min(resource, task, sim.config.cell_size_m)
            resource_bonus = 0.9 if task.kind == TaskKind.WATER and resource.spec.kind == "water" else 0.0
            score = task.expected_value * (1.0 - 0.35 * task.uncertainty) + resource_bonus - 0.07 * travel
            if score > best_score:
                best_score, best = score, task
        actions[resource.resource_id] = best.index
        remaining.discard(best.index)
    return actions
