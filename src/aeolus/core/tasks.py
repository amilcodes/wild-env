"""Candidate-task generation and fixed-shape features for masked assignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, hypot
from typing import TYPE_CHECKING

import numpy as np

from aeolus.core.aviation import evaluate_simulator_leg
from aeolus.core.state import ResourceRuntime, ResourceStatus

if TYPE_CHECKING:
    from aeolus.core.simulator import AeolusSimulator


class TaskKind(IntEnum):
    HOLD = 0
    OBSERVE = 1
    WATER = 2
    RETARDANT = 3
    REINFORCE = 4
    LINE = 5
    SERVICE = 6
    AERIAL_LINE = 7


TASK_FEATURE_DIM = 21
RESOURCE_FEATURE_DIM = 17
ACTOR_GLOBAL_FEATURE_DIM = 10
CRITIC_GLOBAL_FEATURE_DIM = 12
TASK_CAPACITY_SCALE = 16.0


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
    heading_deg: float = 0.0
    service_site_id: str | None = None
    service_water: bool = False
    service_retardant: bool = False
    service_refuels: bool = False
    service_rate_l_min: float = 0.0
    queue_pressure: float = 0.0

    def compatible(self, resource: ResourceRuntime) -> bool:
        if self.kind == TaskKind.HOLD:
            return True
        if resource.status != ResourceStatus.AVAILABLE:
            return False
        if self.kind == TaskKind.SERVICE:
            return resource.spec.kind != "crew"
        if self.kind == TaskKind.OBSERVE:
            return resource.spec.kind == "sensor"
        if self.kind == TaskKind.WATER:
            return resource.spec.kind == "water"
        if self.kind == TaskKind.AERIAL_LINE:
            return resource.spec.kind in {"water", "retardant"}
        if self.kind == TaskKind.LINE:
            return resource.spec.kind == "crew"
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

    tasks = [
        Task(
            0,
            TaskKind.HOLD,
            sim.state.base_xy[0],
            sim.state.base_xy[1],
            0.0,
            0.0,
            0.0,
            capacity=99,
        )
    ]
    for site in sim.state.service_sites:
        if len(tasks) >= sim.config.max_tasks:
            return tasks
        in_service = sum(
            resource.service_site_id == site.site_id
            and resource.status in (ResourceStatus.QUEUED, ResourceStatus.RELOADING)
            for resource in sim.state.resources
        )
        committed_to_approach = sum(
            resource.service_site_id == site.site_id
            and resource.status
            in (
                ResourceStatus.OUTBOUND,
                ResourceStatus.QUEUED,
                ResourceStatus.RELOADING,
            )
            for resource in sim.state.resources
        )
        queue_pressure = in_service / max(site.spec.bays, 1)
        remaining_fraction = (
            1.0
            if not np.isfinite(site.spec.available_volume_l)
            else np.clip(site.remaining_volume_l / max(site.spec.available_volume_l, 1.0), 0.0, 1.0)
        )
        tasks.append(
            Task(
                len(tasks),
                TaskKind.SERVICE,
                site.spec.x,
                site.spec.y,
                expected_value=float(1.0 + 0.25 * remaining_fraction - 0.20 * queue_pressure),
                uncertainty=0.0,
                ground_dependency=0.0,
                capacity=max(0, site.spec.approach_capacity - committed_to_approach),
                service_site_id=site.site_id,
                service_water="water" in site.spec.services,
                service_retardant="retardant" in site.spec.services,
                service_refuels=bool({"fuel", "charge"}.intersection(site.spec.services)),
                service_rate_l_min=site.spec.refill_rate_l_min,
                queue_pressure=float(queue_pressure),
            )
        )
    front_cells = _front_cells(sim)
    # Six task variants per selected front. This reserves stable action
    # slots and avoids unbounded action spaces in the learner.
    per_front = 6
    max_fronts = max(0, (sim.config.max_tasks - len(tasks)) // per_front)
    weather = sim.current_weather()
    explicit_ground_resources = any(resource.spec.kind == "crew" for resource in sim.state.resources)
    asset_y, asset_x = np.nonzero(sim.state.truth.asset_value > 0.0)
    asset_weights = sim.state.truth.asset_value[asset_y, asset_x]
    asset_centroid_x = (
        float(np.average(asset_x, weights=asset_weights)) if asset_x.size else float(sim.config.width / 2)
    )
    asset_centroid_y = (
        float(np.average(asset_y, weights=asset_weights)) if asset_y.size else float(sim.config.height / 2)
    )
    for _, x, y, intensity, uncertainty in front_cells[:max_fronts]:
        asset_threat = (
            float(sim.state.truth.asset_value[max(0, y - 7) : y + 8, max(0, x - 7) : x + 8].sum()) / 30.0
        )
        value = intensity / 100.0 + asset_threat * 2.0
        ground_dependency = 1.0 if sim.state.minute < sim.config.ground_arrival_min else 0.35
        wind_direction = np.deg2rad(
            sim._weather_at_cell(
                weather["wind_direction_deg"],
                x,
                y,
            )
        )
        lead_cells = 8.0
        asset_dx = asset_centroid_x - x
        asset_dy = asset_centroid_y - y
        asset_distance = max(float(np.hypot(asset_dx, asset_dy)), 1.0)
        if asset_threat > 0.01:
            line_head_x = asset_dx / asset_distance
            line_head_y = asset_dy / asset_distance
        else:
            line_head_x = -float(np.sin(wind_direction))
            line_head_y = float(np.cos(wind_direction))
        line_x = int(
            np.clip(
                round(x + lead_cells * line_head_x),
                1,
                sim.config.width - 2,
            )
        )
        line_y = int(
            np.clip(
                round(y + lead_cells * line_head_y),
                1,
                sim.config.height - 2,
            )
        )
        line_heading_deg = float((np.rad2deg(np.arctan2(line_head_y, line_head_x)) + 90.0) % 360.0)
        variants = (
            (
                TaskKind.OBSERVE,
                x,
                y,
                value * (0.65 + uncertainty),
                uncertainty,
                0.0,
                0.0,
            ),
            (
                TaskKind.WATER,
                x,
                y,
                value * 1.15,
                uncertainty * 0.5,
                0.35,
                0.0,
            ),
            (
                TaskKind.RETARDANT,
                x,
                y,
                value * 1.25,
                uncertainty * 0.4,
                ground_dependency,
                0.0,
            ),
            (
                TaskKind.REINFORCE,
                line_x if explicit_ground_resources else x,
                line_y if explicit_ground_resources else y,
                value * (1.40 if explicit_ground_resources else 0.85),
                uncertainty * 0.25,
                ground_dependency,
                line_heading_deg,
            ),
            (
                TaskKind.LINE,
                line_x,
                line_y,
                value * 1.35,
                uncertainty * 0.35,
                0.0,
                line_heading_deg,
            ),
            (
                TaskKind.AERIAL_LINE,
                line_x,
                line_y,
                value * 1.08,
                uncertainty * 0.30,
                ground_dependency,
                line_heading_deg,
            ),
        )
        for (
            kind,
            target_x,
            target_y,
            expected_value,
            task_uncertainty,
            dependency,
            heading_deg,
        ) in variants:
            if len(tasks) >= sim.config.max_tasks:
                return tasks
            tasks.append(
                Task(
                    len(tasks),
                    kind,
                    target_x,
                    target_y,
                    expected_value,
                    task_uncertainty,
                    dependency,
                    capacity=(
                        max(
                            2,
                            sum(
                                resource.spec.kind in {"water", "retardant"}
                                for resource in sim.state.resources
                            ),
                        )
                        if kind == TaskKind.AERIAL_LINE
                        else 1
                    ),
                    heading_deg=heading_deg,
                )
            )
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
                task.kind / float(TaskKind.AERIAL_LINE),
                task.x / max(width - 1, 1),
                task.y / max(height - 1, 1),
                task.expected_value / 12.0,
                task.uncertainty / 12.0,
                task.ground_dependency,
                # Capacity is a routing constraint, not a feature whose raw
                # sentinel value can dominate the policy logits.
                min(float(task.capacity), TASK_CAPACITY_SCALE) / TASK_CAPACITY_SCALE,
                float(task.kind == TaskKind.OBSERVE),
                float(task.kind == TaskKind.WATER),
                float(
                    task.kind
                    in (
                        TaskKind.RETARDANT,
                        TaskKind.REINFORCE,
                        TaskKind.LINE,
                        TaskKind.AERIAL_LINE,
                    )
                ),
                float(np.sin(np.deg2rad(task.heading_deg))),
                float(np.cos(np.deg2rad(task.heading_deg))),
                float(task.kind == TaskKind.SERVICE),
                float(task.kind == TaskKind.SERVICE and task.service_site_id is not None),
                float(task.kind == TaskKind.AERIAL_LINE),
                float(task.service_water),
                float(task.service_retardant),
                float(task.service_refuels),
                min(task.service_rate_l_min / 20_000.0, 1.0),
                min(task.queue_pressure / 4.0, 1.0),
            ],
            dtype=np.float32,
        )
        valid[task.index] = True
    return values, valid


def resource_features(resource: ResourceRuntime, sim: AeolusSimulator) -> np.ndarray:
    width, height = sim.config.width, sim.config.height
    resource_kind = {"retardant": 0, "water": 1, "sensor": 2, "crew": 3}[resource.spec.kind]
    return np.array(
        [
            resource.x / max(width - 1, 1),
            resource.y / max(height - 1, 1),
            resource_kind / 3.0,
            resource.payload_fraction,
            resource.status / float(ResourceStatus.QUEUED),
            resource.eta_min / max(sim.config.horizon_min, 1),
            resource.flight_min / max(resource.spec.endurance_min, 1),
            resource.spec.cruise_speed_m_s / 80.0,
            resource.spec.payload_l / 12000.0,
            resource.reload_cycles / 20.0,
            float(sim.state.ground_engaged),
            sim.state.minute / sim.config.horizon_min,
            resource.accepted_tasks / max(resource.attempted_tasks, 1),
            resource.endurance_remaining_min / max(resource.spec.endurance_min, 1),
            float(resource.current_site_id is not None),
            (
                (sim.state.minute - resource.queue_entered_min) / max(sim.config.horizon_min, 1)
                if resource.queue_entered_min is not None
                else 0.0
            ),
            float(resource.service_site_id is not None),
        ],
        dtype=np.float32,
    )


def _site_for_task(task: Task, sim: AeolusSimulator):
    return next(
        (site for site in sim.state.service_sites if site.site_id == task.service_site_id),
        None,
    )


def _site_supports(resource: ResourceRuntime, site, *, payload_required: bool) -> bool:
    if site.spec.service_mode not in resource.spec.service_modes:
        return False
    payload_service = resource.spec.kind in site.spec.services
    endurance_service = bool({"fuel", "charge"}.intersection(site.spec.services))
    return payload_service if payload_required else payload_service or endurance_service


def _reserve_feasible(resource: ResourceRuntime, task: Task, sim: AeolusSimulator) -> bool:
    reserve = max(
        resource.spec.reserve_endurance_min,
        sim.config.suppression.minimum_reserve_endurance_min,
    )
    if resource.spec.kind == "crew":
        outbound_min = (
            task_distance_min(resource, task, sim.config.cell_size_m) + resource.spec.dispatch_latency_min
        )
        return_min = (
            hypot(task.x - sim.state.base_xy[0], task.y - sim.state.base_xy[1])
            * sim.config.cell_size_m
            / max(resource.spec.cruise_speed_m_s * 60.0, 1.0)
        )
        return resource.flight_min + outbound_min + return_min <= resource.spec.endurance_min - reserve
    outbound = (
        evaluate_simulator_leg(
            resource,
            (float(task.x), float(task.y)),
            sim,
        )
        if resource.spec.kind in {"water", "retardant", "sensor"}
        else None
    )
    if outbound is not None and not outbound.feasible:
        return False
    travel = (
        outbound.travel_min
        if outbound is not None
        else task_distance_min(resource, task, sim.config.cell_size_m)
    ) + resource.spec.dispatch_latency_min
    projected = resource.flight_min + travel
    endurance_limit = (
        outbound.available_endurance_min if outbound is not None else float(resource.spec.endurance_min)
    )
    if task.kind == TaskKind.SERVICE:
        site = _site_for_task(task, sim)
        if site is None:
            return False
        if site.spec.service_mode != "land":
            missing_l = (
                min(
                    resource.spec.payload_l * (1.0 - resource.payload_fraction),
                    site.remaining_volume_l,
                )
                if resource.spec.kind in site.spec.services
                else 0.0
            )
            projected += ceil(site.spec.fixed_turnaround_min + missing_l / site.spec.refill_rate_l_min)
            # A hover-fill or scoop queue is airborne. Treat all work already
            # committed to the site as preceding this resource. This is a
            # conservative reserve check; approach capacity keeps the bound
            # finite.
            for other in sim.state.resources:
                if other is resource or other.service_site_id != site.site_id:
                    continue
                if other.status == ResourceStatus.RELOADING:
                    projected += other.eta_min
                elif other.status in (ResourceStatus.OUTBOUND, ResourceStatus.QUEUED):
                    other_missing_l = (
                        min(
                            other.spec.payload_l * (1.0 - other.payload_fraction),
                            site.remaining_volume_l,
                        )
                        if other.spec.kind in site.spec.services
                        else 0.0
                    )
                    projected += ceil(
                        site.spec.fixed_turnaround_min + other_missing_l / site.spec.refill_rate_l_min
                    )
        if {"fuel", "charge"}.intersection(site.spec.services):
            return projected <= endurance_limit - reserve
        fuel_sites = [
            candidate
            for candidate in sim.state.service_sites
            if {"fuel", "charge"}.intersection(candidate.spec.services)
            and _site_supports(resource, candidate, payload_required=False)
        ]
        if fuel_sites:
            legs = [
                evaluate_simulator_leg(
                    resource,
                    (float(candidate.spec.x), float(candidate.spec.y)),
                    sim,
                    start_xy=(float(task.x), float(task.y)),
                    payload_fraction=(
                        1.0 if resource.spec.kind in site.spec.services else resource.payload_fraction
                    ),
                )
                for candidate in fuel_sites
            ]
            feasible_legs = [leg for leg in legs if leg.feasible]
            if not feasible_legs:
                return False
            continuation_leg = min(
                feasible_legs,
                key=lambda leg: leg.travel_min,
            )
            continuation = continuation_leg.travel_min
            endurance_limit = min(
                endurance_limit,
                continuation_leg.available_endurance_min,
            )
            projected += continuation
        return projected <= endurance_limit - reserve
    if not sim.state.service_sites:
        continuation_leg = evaluate_simulator_leg(
            resource,
            (
                float(sim.state.base_xy[0]),
                float(sim.state.base_xy[1]),
            ),
            sim,
            start_xy=(float(task.x), float(task.y)),
            payload_fraction=0.0,
        )
        if not continuation_leg.feasible:
            return False
        continuation = continuation_leg.travel_min
        endurance_limit = min(
            endurance_limit,
            continuation_leg.available_endurance_min,
        )
    else:
        recovery_sites = [
            site
            for site in sim.state.service_sites
            if {"fuel", "charge"}.intersection(site.spec.services)
            and _site_supports(resource, site, payload_required=False)
        ]
        if not recovery_sites:
            return False
        legs = [
            evaluate_simulator_leg(
                resource,
                (float(site.spec.x), float(site.spec.y)),
                sim,
                start_xy=(float(task.x), float(task.y)),
                payload_fraction=0.0,
            )
            for site in recovery_sites
        ]
        feasible_legs = [leg for leg in legs if leg.feasible]
        if not feasible_legs:
            return False
        continuation_leg = min(
            feasible_legs,
            key=lambda leg: leg.travel_min,
        )
        continuation = continuation_leg.travel_min
        endurance_limit = min(
            endurance_limit,
            continuation_leg.available_endurance_min,
        )
    return projected + continuation <= endurance_limit - reserve


def action_mask(
    resource: ResourceRuntime,
    tasks: list[Task],
    max_tasks: int,
    sim: AeolusSimulator | None = None,
) -> np.ndarray:
    mask = np.zeros(max_tasks, dtype=np.bool_)
    for task in tasks:
        if task.index >= max_tasks or not task.compatible(resource):
            continue
        feasible = True
        if sim is not None and task.kind != TaskKind.HOLD:
            weather = sim.current_weather()
            wind = sim._weather_at_cell(
                weather["wind_speed_m_s"],
                task.x,
                task.y,
            )
            intensity = float(sim.state.belief.intensity_mean[task.y, task.x])
            outbound = (
                evaluate_simulator_leg(
                    resource,
                    (float(task.x), float(task.y)),
                    sim,
                )
                if resource.spec.kind in {"water", "retardant", "sensor"}
                else None
            )
            if outbound is not None:
                feasible &= outbound.feasible
            if task.kind == TaskKind.SERVICE:
                site = _site_for_task(task, sim)
                feasible &= site is not None
                if site is not None:
                    mode_compatible = site.spec.service_mode in resource.spec.service_modes
                    payload_useful = (
                        resource.payload_fraction < 1.0 - 1e-6 and resource.spec.kind in site.spec.services
                    )
                    endurance_useful = resource.flight_min > 1e-6 and bool(
                        {"fuel", "charge"}.intersection(site.spec.services)
                    )
                    feasible &= mode_compatible and task.capacity > 0 and (payload_useful or endurance_useful)
                    arrival_minute = (
                        sim.state.minute
                        + ceil(
                            outbound.travel_min
                            if outbound is not None
                            else task_distance_min(
                                resource,
                                task,
                                sim.config.cell_size_m,
                            )
                        )
                        + resource.spec.dispatch_latency_min
                    )
                    feasible &= site.spec.open_minute <= arrival_minute < site.spec.close_minute
                    feasible &= wind <= site.spec.max_operating_wind_m_s
                    feasible &= site.spec.minimum_depth_m >= resource.spec.minimum_service_depth_m
                    feasible &= site.spec.minimum_length_m >= resource.spec.minimum_service_length_m
                    if resource.spec.kind in site.spec.services:
                        feasible &= site.remaining_volume_l > 0.0
            elif resource.spec.kind in {"water", "retardant"}:
                feasible &= (
                    resource.payload_fraction >= sim.config.suppression.minimum_dispatch_payload_fraction
                )
            if resource.spec.kind in {"water", "retardant"}:
                feasible &= wind <= min(
                    resource.spec.max_operating_wind_m_s,
                    sim.config.suppression.aviation_max_wind_m_s,
                )
            if resource.spec.kind == "crew":
                feasible &= intensity <= min(
                    resource.spec.max_direct_intensity_kw_m,
                    sim.config.suppression.direct_attack_max_intensity_kw_m,
                )
            feasible &= _reserve_feasible(resource, task, sim)
        mask[task.index] = feasible
    mask[0] = True
    return mask


def actor_global_features(sim: AeolusSimulator) -> np.ndarray:
    """Shared information available to every executing resource.

    Every value is derived from the delivered belief, the public incident
    clock, or observable fleet state. Hidden fire truth is deliberately absent.
    """

    belief = sim.state.belief
    observed = belief.observed_at >= 0
    estimated_active = belief.intensity_mean >= 20.0
    observed_age = np.maximum(0, sim.state.minute - belief.observed_at[observed])
    resource_ready = sum(resource.status == ResourceStatus.AVAILABLE for resource in sim.state.resources)
    return np.array(
        [
            float(estimated_active.mean()),
            float(belief.burn_probability.mean()),
            float(belief.intensity_mean.mean() / 1000.0),
            float(belief.intensity_std.mean() / 100.0),
            float(observed_age.mean() / max(sim.config.horizon_min, 1)) if observed_age.size else 1.0,
            float(observed.mean()),
            resource_ready / max(len(sim.state.resources), 1),
            float(sim.state.ground_engaged),
            sim.state.minute / sim.config.horizon_min,
            sim.state.cumulative_exposure / 500.0,
        ],
        dtype=np.float32,
    )


def critic_global_features(sim: AeolusSimulator) -> np.ndarray:
    """Privileged centralized-training state, never placed in agent observations."""

    truth = sim.state.truth
    belief = sim.state.belief
    flaming = truth.phase == 1
    asset_loss = float((truth.observed_burned * truth.asset_value).sum())
    observed = belief.observed_at >= 0
    resource_ready = sum(resource.status == ResourceStatus.AVAILABLE for resource in sim.state.resources)
    return np.array(
        [
            float(flaming.mean()),
            float(truth.observed_burned.mean()),
            float(truth.intensity_kw_m.mean() / 1000.0),
            float(truth.fuel_remaining.mean()),
            float(truth.water.mean()),
            float(truth.retardant.mean()),
            float(truth.ground_hold.mean()),
            asset_loss / 50.0,
            float(observed.mean()),
            resource_ready / max(len(sim.state.resources), 1),
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


def task_travel_min(
    resource: ResourceRuntime,
    task: Task,
    sim: AeolusSimulator,
) -> float:
    """Travel time with tactical performance when available."""

    if resource.spec.kind not in {"water", "retardant", "sensor"}:
        return task_distance_min(resource, task, sim.config.cell_size_m)
    return evaluate_simulator_leg(
        resource,
        (float(task.x), float(task.y)),
        sim,
    ).travel_min
