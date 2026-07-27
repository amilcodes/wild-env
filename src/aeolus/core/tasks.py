"""Candidate-task generation and fixed-shape features for masked assignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import hypot
from typing import TYPE_CHECKING

import numpy as np

from aeolus.core.state import ResourceRuntime, ResourceStatus

if TYPE_CHECKING:
    from aeolus.core.simulator import AeolusSimulator


class TaskKind(IntEnum):
    HOLD = 0
    OBSERVE = 1
    WATER = 2
    RETARDANT = 3
    REINFORCE = 4


TASK_FEATURE_DIM = 11
RESOURCE_FEATURE_DIM = 13
GLOBAL_FEATURE_DIM = 10


@dataclass(frozen=True)
class Task:
    index: int
    kind: TaskKind
    x: int
    y: int
    expected_value: float
    uncertainty: float
    ground_dependency: float
    capacity: int = 1

    def compatible(self, resource: ResourceRuntime) -> bool:
        if self.kind == TaskKind.HOLD:
            return True
        if resource.status != ResourceStatus.AVAILABLE:
            return False
        if self.kind == TaskKind.OBSERVE:
            return resource.spec.kind == "sensor"
        if self.kind == TaskKind.WATER:
            return resource.spec.kind == "water"
        return resource.spec.kind == "retardant"


def _front_cells(sim: AeolusSimulator) -> list[tuple[float, int, int, float, float]]:
    belief = sim.state.belief
    truth = sim.state.truth
    height, width = belief.intensity_mean.shape
    candidates: list[tuple[float, int, int, float, float]] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            intensity = float(belief.intensity_mean[y, x])
            if intensity < 35.0 or truth.barrier[y, x]:
                continue
            exposed = any(
                belief.intensity_mean[y + dy, x + dx] < 15.0 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if not exposed:
                continue
            # Normalized epistemic uncertainty; all policy-facing task features
            # and baseline scores stay dimensionless.
            uncertainty = float(np.clip(belief.intensity_std[y, x] / 140.0, 0.0, 1.5))
            asset_neighborhood = truth.asset_value[max(0, y - 7) : y + 8, max(0, x - 7) : x + 8]
            asset_threat = float(asset_neighborhood.sum()) / 30.0
            score = intensity / 100.0 + 1.8 * asset_threat + 0.45 * uncertainty
            candidates.append((score, x, y, intensity, uncertainty))
    candidates.sort(reverse=True)
    return candidates


def generate_tasks(sim: AeolusSimulator) -> list[Task]:
    """Build a stable task set from the belief, never the hidden truth fire."""

    tasks = [Task(0, TaskKind.HOLD, sim.state.base_xy[0], sim.state.base_xy[1], 0.0, 0.0, 0.0, capacity=99)]
    front_cells = _front_cells(sim)
    # At most five task variants per selected front. This reserves stable action
    # slots and avoids unbounded action spaces in the learner.
    per_front = 5
    max_fronts = max(1, (sim.config.max_tasks - 1) // per_front)
    for _, x, y, intensity, uncertainty in front_cells[:max_fronts]:
        asset_threat = (
            float(sim.state.truth.asset_value[max(0, y - 7) : y + 8, max(0, x - 7) : x + 8].sum()) / 30.0
        )
        value = intensity / 100.0 + asset_threat * 2.0
        ground_dependency = 1.0 if sim.state.minute < sim.config.ground_arrival_min else 0.35
        variants = (
            (TaskKind.OBSERVE, value * (0.65 + uncertainty), uncertainty, 0.0),
            (TaskKind.WATER, value * 1.15, uncertainty * 0.5, 0.35),
            (TaskKind.RETARDANT, value * 1.25, uncertainty * 0.4, ground_dependency),
            (TaskKind.REINFORCE, value * 0.85, uncertainty * 0.25, ground_dependency),
        )
        for kind, expected_value, task_uncertainty, dependency in variants:
            if len(tasks) >= sim.config.max_tasks:
                return tasks
            tasks.append(Task(len(tasks), kind, x, y, expected_value, task_uncertainty, dependency))
    return tasks


def task_tensor(tasks: list[Task], max_tasks: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((max_tasks, TASK_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(max_tasks, dtype=np.bool_)
    for task in tasks:
        if task.index >= max_tasks:
            break
        values[task.index] = np.array(
            [
                1.0,
                task.kind / float(TaskKind.REINFORCE),
                task.x / max(width - 1, 1),
                task.y / max(height - 1, 1),
                task.expected_value / 12.0,
                task.uncertainty / 12.0,
                task.ground_dependency,
                # Capacity is a routing constraint, not a feature whose raw
                # sentinel value can dominate the policy logits.
                min(float(task.capacity), 3.0) / 3.0,
                float(task.kind == TaskKind.OBSERVE),
                float(task.kind == TaskKind.WATER),
                float(task.kind in (TaskKind.RETARDANT, TaskKind.REINFORCE)),
            ],
            dtype=np.float32,
        )
        valid[task.index] = True
    return values, valid


def resource_features(resource: ResourceRuntime, sim: AeolusSimulator) -> np.ndarray:
    width, height = sim.config.width, sim.config.height
    resource_kind = {"retardant": 0, "water": 1, "sensor": 2}[resource.spec.kind]
    return np.array(
        [
            resource.x / max(width - 1, 1),
            resource.y / max(height - 1, 1),
            resource_kind / 2.0,
            resource.payload_fraction,
            resource.status / float(ResourceStatus.WITHDRAWN),
            resource.eta_min / max(sim.config.horizon_min, 1),
            resource.flight_min / max(resource.spec.endurance_min, 1),
            resource.spec.cruise_speed_m_s / 80.0,
            resource.spec.payload_l / 12000.0,
            resource.reload_cycles / 20.0,
            float(sim.state.ground_engaged),
            sim.state.minute / sim.config.horizon_min,
            len(sim.state.events[-8:]) / 8.0,
        ],
        dtype=np.float32,
    )


def action_mask(resource: ResourceRuntime, tasks: list[Task], max_tasks: int) -> np.ndarray:
    mask = np.zeros(max_tasks, dtype=np.bool_)
    for task in tasks:
        if task.index < max_tasks and task.compatible(resource):
            mask[task.index] = True
    mask[0] = True
    return mask


def global_features(sim: AeolusSimulator) -> np.ndarray:
    truth = sim.state.truth
    belief = sim.state.belief
    flaming = truth.phase == 1
    asset_loss = float((truth.observed_burned * truth.asset_value).sum())
    observed_age = np.maximum(0, sim.state.minute - belief.observed_at)
    resource_ready = sum(resource.status == ResourceStatus.AVAILABLE for resource in sim.state.resources)
    return np.array(
        [
            float(flaming.mean()),
            float(truth.observed_burned.mean()),
            float(truth.intensity_kw_m.mean() / 1000.0),
            float(belief.intensity_std.mean() / 100.0),
            float(observed_age[flaming].mean() / max(sim.config.horizon_min, 1)) if flaming.any() else 0.0,
            asset_loss / 50.0,
            resource_ready / max(len(sim.state.resources), 1),
            float(sim.state.ground_engaged),
            sim.state.minute / sim.config.horizon_min,
            sim.state.cumulative_exposure / 500.0,
        ],
        dtype=np.float32,
    )


def task_distance_min(resource: ResourceRuntime, task: Task, cell_size_m: float) -> float:
    return (
        hypot(resource.x - task.x, resource.y - task.y)
        * cell_size_m
        / max(resource.spec.cruise_speed_m_s * 60.0, 1.0)
    )
