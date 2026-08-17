"""Inspectable non-learning baselines using the operational belief/task set."""

from __future__ import annotations

import copy
from functools import cache
from math import inf
from typing import Any

from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import Task, TaskKind, action_mask, task_distance_min


def _feasible(sim: AeolusSimulator, resource, task: Task) -> bool:
    return bool(action_mask(resource, sim.tasks, sim.config.max_tasks, sim)[task.index])


def no_aerial_action(sim: AeolusSimulator) -> dict[str, int]:
    return {resource.resource_id: 0 for resource in sim.state.resources}


def nearest_feasible(sim: AeolusSimulator) -> dict[str, int]:
    actions: dict[str, int] = {}
    for resource in sim.state.resources:
        candidate = min(
            (task for task in sim.tasks if task.kind != TaskKind.HOLD and _feasible(sim, resource, task)),
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
            if not _feasible(sim, resource, task):
                continue
            distance = task_distance_min(resource, task, sim.config.cell_size_m)
            line_bonus = 2.2 if task.kind in (TaskKind.RETARDANT, TaskKind.REINFORCE, TaskKind.LINE) else 0.0
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
            if task.index not in remaining or not _feasible(
                sim,
                resource,
                task,
            ):
                continue
            travel = task_distance_min(resource, task, sim.config.cell_size_m)
            resource_bonus = (
                0.9
                if (task.kind == TaskKind.WATER and resource.spec.kind == "water")
                or (task.kind == TaskKind.LINE and resource.spec.kind == "crew")
                else 0.0
            )
            score = task.expected_value * (1.0 - 0.35 * task.uncertainty) + resource_bonus - 0.07 * travel
            if score > best_score:
                best_score, best = score, task
        actions[resource.resource_id] = best.index
        remaining.discard(best.index)
    return actions


def _assignment_utility(sim: AeolusSimulator, resource_index: int, task: Task) -> float:
    if task.kind == TaskKind.HOLD:
        return 0.0
    resource = sim.state.resources[resource_index]
    travel = task_distance_min(resource, task, sim.config.cell_size_m)
    specialization = (
        0.9
        if (task.kind == TaskKind.WATER and resource.spec.kind == "water")
        or (task.kind == TaskKind.LINE and resource.spec.kind == "crew")
        else 0.0
    )
    information = 0.7 * task.uncertainty if task.kind == TaskKind.OBSERVE else 0.0
    return (
        task.expected_value * (1.0 - 0.35 * task.uncertainty) + specialization + information - 0.07 * travel
    )


def joint_assignment(sim: AeolusSimulator) -> dict[str, int]:
    """Exact maximum-weight assignment over the current resource/task graph.

    This is an operationally legible comparator for learned decentralized
    policies. It sees the same belief-derived task set as the agents, honors
    resource compatibility and task capacity, and jointly resolves competition
    instead of depending on resource iteration order.
    """

    resources = sim.state.resources
    selected = [0] * len(resources)
    # Compatibility partitions tasks by resource kind. Solving these independent
    # components avoids a Cartesian product across unrelated task types while
    # preserving the global optimum.
    resource_groups: dict[str, list[int]] = {}
    for index, resource in enumerate(resources):
        resource_groups.setdefault(resource.spec.kind, []).append(index)

    for resource_indices in resource_groups.values():
        task_indices = sorted(
            {
                task.index
                for resource_index in resource_indices
                for task in sim.tasks
                if task.index and _feasible(sim, resources[resource_index], task)
            }
        )
        task_position = {task_index: position for position, task_index in enumerate(task_indices)}
        initial_capacity = tuple(sim.tasks[index].capacity for index in task_indices)
        candidates = tuple(
            (
                0,
                *(
                    task.index
                    for task in sim.tasks
                    if task.index and _feasible(sim, resources[resource_index], task)
                ),
            )
            for resource_index in resource_indices
        )

        @cache
        def solve(
            local_resource_index: int,
            remaining_capacity: tuple[int, ...],
        ) -> tuple[float, tuple[int, ...]]:
            if local_resource_index == len(resource_indices):
                return 0.0, ()
            best_value = -inf
            best_actions: tuple[int, ...] = ()
            global_resource_index = resource_indices[local_resource_index]
            for task_index in candidates[local_resource_index]:
                position = task_position.get(task_index)
                if position is not None and remaining_capacity[position] <= 0:
                    continue
                updated = list(remaining_capacity)
                if position is not None:
                    updated[position] -= 1
                future_value, future_actions = solve(
                    local_resource_index + 1,
                    tuple(updated),
                )
                value = (
                    _assignment_utility(
                        sim,
                        global_resource_index,
                        sim.tasks[task_index],
                    )
                    + future_value
                )
                actions = (task_index, *future_actions)
                if value > best_value or (value == best_value and actions < best_actions):
                    best_value, best_actions = value, actions
            return best_value, best_actions

        _, group_actions = solve(0, initial_capacity)
        for local_index, resource_index in enumerate(resource_indices):
            selected[resource_index] = group_actions[local_index]

    return {resource.resource_id: int(selected[index]) for index, resource in enumerate(resources)}


def rollout_lookahead_diagnostics(
    sim: AeolusSimulator,
    *,
    horizon_decisions: int = 3,
    discount: float = 0.99,
) -> dict[str, Any]:
    """Evaluate legible joint actions with a receding-horizon model rollout.

    The first-action set comes from the retained doctrine and assignment
    comparators.  Subsequent decisions use exact joint assignment.  Every
    candidate receives a cloned simulator and identical random state, making
    this a deterministic privileged-information planning comparator rather
    than a deployable decentralized policy.
    """

    if horizon_decisions < 1:
        raise ValueError("horizon_decisions must be positive")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be within (0, 1]")
    proposal_functions = (
        ("hold", no_aerial_action),
        ("nearest", nearest_feasible),
        ("anchor_flank", anchor_flank),
        ("greedy_value", greedy_value),
        ("joint_assignment", joint_assignment),
    )
    resource_ids = tuple(resource.resource_id for resource in sim.state.resources)
    proposals: dict[tuple[int, ...], tuple[str, dict[str, int]]] = {}
    for name, function in proposal_functions:
        actions = function(sim)
        key = tuple(int(actions.get(resource_id, 0)) for resource_id in resource_ids)
        proposals.setdefault(key, (name, actions))

    trials: list[dict[str, Any]] = []
    for key, (name, actions) in proposals.items():
        branch = copy.deepcopy(sim)
        branch._state_observers = []
        total = 0.0
        rewards: list[float] = []
        blocked_before = branch.state.blocked_actions
        first_actions = actions
        for decision in range(horizon_decisions):
            selected = first_actions if decision == 0 else joint_assignment(branch)
            _, reward, terminated, truncated, _ = branch.decision_step(selected)
            rewards.append(float(reward))
            total += discount**decision * float(reward)
            if terminated or truncated:
                break
        trials.append(
            {
                "proposal": name,
                "action_tuple": list(key),
                "discounted_return": float(total),
                "rewards": rewards,
                "blocked_actions": (branch.state.blocked_actions - blocked_before),
                "terminal": bool(branch.state.terminated or branch.state.truncated),
            }
        )
    best = max(
        trials,
        key=lambda trial: (
            trial["discounted_return"],
            -trial["blocked_actions"],
            tuple(-value for value in trial["action_tuple"]),
        ),
    )
    selected_actions = {
        resource_id: int(action)
        for resource_id, action in zip(
            resource_ids,
            best["action_tuple"],
            strict=True,
        )
    }
    return {
        "method": "receding-horizon comparator rollout",
        "horizon_decisions": horizon_decisions,
        "discount": float(discount),
        "selected_proposal": best["proposal"],
        "actions": selected_actions,
        "trials": sorted(
            trials,
            key=lambda trial: trial["proposal"],
        ),
        "information_note": (
            "Simulator rollouts use privileged model state and define an upper-level "
            "planning comparator, not an actor observation policy."
        ),
    }


def rollout_lookahead(sim: AeolusSimulator) -> dict[str, int]:
    """Three-decision receding-horizon comparator."""

    return rollout_lookahead_diagnostics(sim)["actions"]
