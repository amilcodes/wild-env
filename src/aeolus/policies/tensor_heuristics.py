"""Device-resident operational comparators for tensor-environment studies."""

from __future__ import annotations

import torch
from torch import Tensor

from aeolus.core.tasks import TASK_CAPACITY_SCALE
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.envs.tensor_operations import TensorOperationsEnv


@torch.no_grad()
def cycle_time_greedy(env: TensorOperationsEnv) -> Tensor:
    """Assign loads by marginal protected value per travel minute.

    Empty resources route to the compatible service node with the smallest
    travel-plus-queue-plus-turnaround estimate.  Loaded resources are assigned
    sequentially to avoid crediting litres beyond an attack segment's remaining
    demand.  The comparator reads exactly the tensor observation and public
    operations state used by the learner.
    """

    state = env.state
    mask = env.action_mask()
    batch = env.batch_size
    actions = torch.zeros(
        (batch, env.num_resources),
        device=env.device,
        dtype=torch.long,
    )
    expected_remaining = state.segment_remaining_l.clone()
    batch_index = torch.arange(batch, device=env.device)
    for resource_index in range(env.num_resources):
        payload_l = (
            env.resource_payload_l[resource_index] * state.resource_payload_fraction[:, resource_index]
        )
        segment_mask = mask[
            :,
            resource_index,
            1 : 1 + env.max_segments,
        ]
        distance = torch.linalg.vector_norm(
            state.segment_xy - state.resource_xy[:, resource_index, None, :],
            dim=-1,
        )
        travel = distance / torch.clamp(
            env.resource_speed_cells_min[resource_index],
            min=1e-6,
        )
        useful_l = torch.minimum(payload_l[:, None], expected_remaining)
        segment_score = (
            state.segment_priority
            * useful_l
            / torch.clamp(
                travel + env.resource_dispatch_latency_min[resource_index],
                min=1.0,
            )
        )
        segment_score = segment_score.masked_fill(~segment_mask, -torch.inf)
        best_segment_score, best_segment = segment_score.max(dim=1)

        site_mask = mask[:, resource_index, 1 + env.max_segments :]
        site_distance = torch.linalg.vector_norm(
            env.site_xy[None, :, :] - state.resource_xy[:, resource_index, None, :],
            dim=-1,
        )
        site_travel = site_distance / torch.clamp(
            env.resource_speed_cells_min[resource_index],
            min=1e-6,
        )
        queue = torch.zeros(
            (batch, env.num_sites),
            device=env.device,
            dtype=env.dtype,
        )
        for site_index in range(env.num_sites):
            queue[:, site_index] = (
                (state.resource_site_index == site_index)
                & ((state.resource_status == 3) | (state.resource_status == 4))
            ).sum(dim=1) / env.site_bays[site_index]
        missing_l = env.resource_payload_l[resource_index] * (
            1.0 - state.resource_payload_fraction[:, resource_index]
        )
        service_time = (
            env.site_turnaround_min[None, :]
            + missing_l[:, None] / env.site_rate_l_min[None, :]
            + queue * env.site_turnaround_min[None, :]
        )
        site_score = -(site_travel + service_time)
        site_score = site_score.masked_fill(~site_mask, -torch.inf)
        _, best_site = site_score.max(dim=1)

        choose_segment = torch.isfinite(best_segment_score)
        empty = state.resource_payload_fraction[:, resource_index] < (
            env.config.suppression.minimum_dispatch_payload_fraction
        )
        choose_site = empty & site_mask.any(dim=1)
        action = torch.where(
            choose_segment & (~choose_site),
            best_segment + 1,
            torch.where(
                choose_site,
                best_site + 1 + env.max_segments,
                torch.zeros_like(best_segment),
            ),
        )
        actions[:, resource_index] = action
        assigned_segment = action - 1
        contributes = (action >= 1) & (action <= env.max_segments)
        selected_segment = torch.clamp(
            assigned_segment,
            0,
            env.max_segments - 1,
        )
        expected_remaining[
            batch_index[contributes],
            selected_segment[contributes],
        ] = torch.clamp(
            expected_remaining[
                batch_index[contributes],
                selected_segment[contributes],
            ]
            - payload_l[contributes],
            min=0.0,
        )
    return actions


@torch.no_grad()
def incident_risk_greedy(env: TensorIncidentEnv) -> Tensor:
    """Public-belief comparator for the fire-coupled tensor environment.

    Loaded aircraft maximize front priority and uncertainty reduction per
    estimated travel minute. Empty aircraft minimize service cycle time. The
    function consumes actor-visible task/resource tensors and action masks;
    hidden fire truth and privileged critic tensors are never read.
    """

    observation = env.observations()
    mask = observation.action_mask
    tasks = observation.tasks
    state = env.state
    batch_index = torch.arange(env.batch_size, device=env.device)
    actions = torch.zeros(
        (env.batch_size, env.num_resources),
        device=env.device,
        dtype=torch.long,
    )
    capacity = torch.ceil(tasks[..., 7] * TASK_CAPACITY_SCALE).long()
    used = torch.zeros_like(capacity)
    # Front scores describe where to work, while the actor-global belief gives
    # a coarse estimate of how much simultaneous aerial work is warranted.
    # This prevents the comparator from emptying the entire fleet onto a small
    # ignition simply because several task slots are technically available.
    burning_cell_equivalent = observation.global_state[:, 0] * env.grid_size * env.grid_size
    attack_budget = (
        torch.ceil(burning_cell_equivalent / 8.0)
        .long()
        .clamp(
            1,
            env.num_resources,
        )
    )
    attacks_selected = torch.zeros(
        env.batch_size,
        device=env.device,
        dtype=torch.long,
    )

    for resource_index in range(env.num_resources):
        segment_mask = mask[:, resource_index, 1 : 1 + env.max_segments] & (
            used[:, 1 : 1 + env.max_segments] < capacity[:, 1 : 1 + env.max_segments]
        )
        target_xy = tasks[:, 1 : 1 + env.max_segments, 2:4] * max(
            env.grid_size - 1,
            1,
        )
        origin = state.resource_xy[:, resource_index, None].expand(
            -1,
            env.max_segments,
            -1,
        )
        travel = env._distance_minutes(
            origin,
            target_xy,
            resource_index=resource_index,
        )
        priority = tasks[:, 1 : 1 + env.max_segments, 4] * 12.0
        uncertainty = tasks[:, 1 : 1 + env.max_segments, 5] * 1.5
        persistence = 1.25 if env.resource_kind_names[resource_index] == "retardant" else 1.0
        segment_score = (
            priority
            * (1.0 + persistence * uncertainty)
            / (travel + env.resource_dispatch_latency_min[resource_index]).clamp_min(1.0)
        ).masked_fill(~segment_mask, -torch.inf)
        best_segment_score, best_segment = segment_score.max(dim=1)

        site_mask = mask[:, resource_index, 1 + env.max_segments :] & (
            used[:, 1 + env.max_segments :] < capacity[:, 1 + env.max_segments :]
        )
        site_origin = state.resource_xy[:, resource_index, None].expand(
            -1,
            env.num_sites,
            -1,
        )
        site_destination = env.site_xy[None].expand(env.batch_size, -1, -1)
        site_travel = env._distance_minutes(
            site_origin,
            site_destination,
            resource_index=resource_index,
        )
        queue_pressure = tasks[:, 1 + env.max_segments :, 20] * 4.0
        service_time = (
            env.site_turnaround_min[None]
            + env.resource_payload_l[resource_index]
            * (1.0 - state.resource_payload_fraction[:, resource_index, None])
            / env.site_rate_l_min[None]
        )
        site_score = (site_travel + service_time * (1.0 + queue_pressure)).masked_fill(~site_mask, torch.inf)
        best_site_score, best_site = site_score.min(dim=1)

        needs_service = state.resource_payload_fraction[:, resource_index] < (
            env.config.suppression.minimum_dispatch_payload_fraction
        )
        choose_site = needs_service & torch.isfinite(best_site_score)
        choose_segment = (
            (~choose_site) & torch.isfinite(best_segment_score) & (attacks_selected < attack_budget)
        )
        action = torch.where(
            choose_segment,
            best_segment + 1,
            torch.where(
                choose_site,
                best_site + 1 + env.max_segments,
                torch.zeros_like(best_segment),
            ),
        )
        actions[:, resource_index] = action
        used[batch_index, action] += 1
        attacks_selected += choose_segment.long()
    return actions
