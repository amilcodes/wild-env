"""Accelerator-resident batched aerial logistics and line-attack environment.

This module is the high-throughput operations-training path.  It keeps fleet,
service-node, queue, payload, endurance, attack-segment, mask, and reward state
in PyTorch tensors.  The canonical incident simulator remains the fire-coupled
semantic oracle used for fine-tuning and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch
from torch import Tensor

from aeolus.config import ScenarioConfig
from aeolus.core.tasks import (
    ACTOR_GLOBAL_FEATURE_DIM,
    RESOURCE_FEATURE_DIM,
    TASK_CAPACITY_SCALE,
    TASK_FEATURE_DIM,
    TaskKind,
)


class TensorResourceStatus(IntEnum):
    AVAILABLE = 0
    TO_ATTACK = 1
    TO_SERVICE = 2
    QUEUED = 3
    SERVICING = 4
    WITHDRAWN = 5


@dataclass
class TensorOperationsState:
    minute: Tensor
    resource_xy: Tensor
    resource_status: Tensor
    resource_eta_min: Tensor
    resource_leg_total_min: Tensor
    resource_leg_start_xy: Tensor
    resource_leg_end_xy: Tensor
    resource_payload_fraction: Tensor
    resource_endurance_remaining_min: Tensor
    resource_target_index: Tensor
    resource_queue_age_min: Tensor
    resource_reserved_load_l: Tensor
    resource_site_index: Tensor
    resource_service_cycles: Tensor
    resource_attempted_tasks: Tensor
    resource_accepted_tasks: Tensor
    segment_xy: Tensor
    segment_heading_deg: Tensor
    segment_kind: Tensor
    segment_priority: Tensor
    segment_required_l: Tensor
    segment_remaining_l: Tensor
    site_remaining_l: Tensor
    done: Tensor


@dataclass(frozen=True)
class TensorOperationsObservation:
    resource: Tensor
    tasks: Tensor
    action_mask: Tensor
    task_valid: Tensor
    global_state: Tensor


@dataclass(frozen=True)
class TensorOperationsStep:
    observation: TensorOperationsObservation
    reward: Tensor
    done: Tensor
    delivered_l: Tensor
    wasted_l: Tensor
    blocked_actions: Tensor


class TensorOperationsEnv:
    """Fixed-shape GPU/ROCm/CPU environment for sortie-level MARL pretraining.

    Actions are stable within an episode:

    - ``0``: hold;
    - ``1..K``: contribute the current load to an oriented attack segment;
    - ``K+1..K+S``: route to a service site.

    The environment advances event state at one-minute resolution inside each
    tactical decision.  No host transfer is required by ``reset``,
    ``observations``, ``action_mask``, or ``step``.
    """

    _MODE_IDS = {"land": 0, "hover_fill": 1, "scoop": 2}
    _KIND_IDS = {"retardant": 0, "water": 1}

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        batch_size: int,
        max_segments: int = 32,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        terminate_on_completion: bool = True,
    ):
        if not config.service_sites:
            raise ValueError("tensor operations training requires explicit service_sites")
        aerial = [resource for resource in config.resources if resource.kind in self._KIND_IDS]
        if not aerial:
            raise ValueError("tensor operations training requires water or retardant resources")
        if batch_size < 1 or max_segments < 1:
            raise ValueError("batch_size and max_segments must be positive")
        self.config = config
        self.batch_size = int(batch_size)
        self.max_segments = int(max_segments)
        self.num_resources = len(aerial)
        self.num_sites = len(config.service_sites)
        self.num_tasks = 1 + self.max_segments + self.num_sites
        self.device = torch.device(device)
        self.dtype = dtype
        self.terminate_on_completion = bool(terminate_on_completion)
        self.resources = tuple(aerial)
        self.sites = config.service_sites
        self._generator = torch.Generator(device=self.device)

        def tensor(values, *, tensor_dtype: torch.dtype = dtype) -> Tensor:
            return torch.as_tensor(values, device=self.device, dtype=tensor_dtype)

        self.resource_kind = tensor(
            [self._KIND_IDS[resource.kind] for resource in aerial],
            tensor_dtype=torch.long,
        )
        self.resource_speed_cells_min = tensor(
            [resource.cruise_speed_m_s * 60.0 / config.cell_size_m for resource in aerial]
        )
        self.resource_payload_l = tensor([resource.payload_l for resource in aerial])
        self.resource_endurance_min = tensor([resource.endurance_min for resource in aerial])
        self.resource_reserve_min = tensor(
            [
                max(
                    resource.reserve_endurance_min,
                    config.suppression.minimum_reserve_endurance_min,
                )
                for resource in aerial
            ]
        )
        self.resource_dispatch_latency_min = tensor([resource.dispatch_latency_min for resource in aerial])
        self.resource_max_wind_m_s = tensor(
            [
                min(
                    resource.max_operating_wind_m_s,
                    config.suppression.aviation_max_wind_m_s,
                )
                for resource in aerial
            ]
        )
        self.resource_mode = torch.zeros(
            (self.num_resources, len(self._MODE_IDS)),
            device=self.device,
            dtype=torch.bool,
        )
        for resource_index, resource in enumerate(aerial):
            for mode in resource.service_modes:
                self.resource_mode[resource_index, self._MODE_IDS[mode]] = True

        self.site_xy = tensor([[site.x, site.y] for site in self.sites])
        self.site_mode = tensor(
            [self._MODE_IDS[site.service_mode] for site in self.sites],
            tensor_dtype=torch.long,
        )
        self.site_bays = tensor([site.bays for site in self.sites], tensor_dtype=torch.long)
        self.site_approach_capacity = tensor(
            [site.approach_capacity for site in self.sites],
            tensor_dtype=torch.long,
        )
        self.site_rate_l_min = tensor([site.refill_rate_l_min for site in self.sites])
        self.site_turnaround_min = tensor([site.fixed_turnaround_min for site in self.sites])
        self.site_open_min = tensor([site.open_minute for site in self.sites], tensor_dtype=torch.long)
        self.site_close_min = tensor([site.close_minute for site in self.sites], tensor_dtype=torch.long)
        self.site_max_wind_m_s = tensor([site.max_operating_wind_m_s for site in self.sites])
        self.site_refuels = tensor(
            [bool({"fuel", "charge"}.intersection(site.services)) for site in self.sites],
            tensor_dtype=torch.bool,
        )
        self.site_payload_service = torch.zeros(
            (self.num_sites, len(self._KIND_IDS)),
            device=self.device,
            dtype=torch.bool,
        )
        for site_index, site in enumerate(self.sites):
            for payload_kind, kind_index in self._KIND_IDS.items():
                self.site_payload_service[site_index, kind_index] = payload_kind in site.services
        self.resource_site_mode_compatible = self.resource_mode[:, self.site_mode]
        self.resource_site_payload_compatible = (
            self.site_payload_service[None, :, :]
            .expand(self.num_resources, -1, -1)
            .gather(
                2,
                self.resource_kind[:, None, None].expand(-1, self.num_sites, 1),
            )[..., 0]
        )
        self.resource_site_compatible = self.resource_site_mode_compatible & (
            self.resource_site_payload_compatible | self.site_refuels[None, :]
        )
        self.home_site_index = tensor(
            [self._home_site_index(resource) for resource in aerial],
            tensor_dtype=torch.long,
        )
        self.wind_speed_m_s = torch.full(
            (self.batch_size,),
            float(config.wind_speed_m_s),
            device=self.device,
            dtype=self.dtype,
        )
        self.state: TensorOperationsState
        self.reset(seed=config.seed)

    def _home_site_index(self, resource) -> int:
        if resource.home_site_id is not None:
            return next(
                index for index, site in enumerate(self.sites) if site.site_id == resource.home_site_id
            )
        resource_index = self.resources.index(resource)
        compatible = torch.nonzero(
            self.resource_site_compatible[resource_index],
            as_tuple=False,
        )
        if compatible.numel() == 0:
            raise ValueError(f"resource {resource.resource_id} has no compatible service site")
        return int(compatible[0, 0])

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: int | None = None,
        segment_xy: Tensor | None = None,
        segment_required_l: Tensor | None = None,
        segment_kind: Tensor | None = None,
        segment_priority: Tensor | None = None,
        segment_heading_deg: Tensor | None = None,
    ) -> TensorOperationsObservation:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        batch, resources, segments = (
            self.batch_size,
            self.num_resources,
            self.max_segments,
        )
        home_xy = self.site_xy[self.home_site_index]
        resource_xy = home_xy[None, :, :].expand(batch, -1, -1).clone()
        if segment_xy is None:
            centre = torch.tensor(
                [0.55 * (self.config.width - 1), 0.50 * (self.config.height - 1)],
                device=self.device,
                dtype=self.dtype,
            )
            scale = torch.tensor(
                [0.20 * self.config.width, 0.20 * self.config.height],
                device=self.device,
                dtype=self.dtype,
            )
            segment_xy = (
                centre
                + torch.randn(
                    (batch, segments, 2),
                    device=self.device,
                    dtype=self.dtype,
                    generator=self._generator,
                )
                * scale
            )
            segment_xy[..., 0].clamp_(1.0, self.config.width - 2.0)
            segment_xy[..., 1].clamp_(1.0, self.config.height - 2.0)
        else:
            segment_xy = self._checked_segment_tensor(segment_xy, (batch, segments, 2), self.dtype)
        if segment_required_l is None:
            segment_required_l = torch.empty(
                (batch, segments),
                device=self.device,
                dtype=self.dtype,
            ).uniform_(1800.0, 9000.0, generator=self._generator)
        else:
            segment_required_l = self._checked_segment_tensor(
                segment_required_l,
                (batch, segments),
                self.dtype,
            )
        if segment_kind is None:
            segment_kind = torch.randint(
                0,
                3,
                (batch, segments),
                device=self.device,
                generator=self._generator,
            )
        else:
            segment_kind = self._checked_segment_tensor(
                segment_kind,
                (batch, segments),
                torch.long,
            )
        if segment_priority is None:
            segment_priority = torch.empty(
                (batch, segments),
                device=self.device,
                dtype=self.dtype,
            ).uniform_(0.5, 2.0, generator=self._generator)
        else:
            segment_priority = self._checked_segment_tensor(
                segment_priority,
                (batch, segments),
                self.dtype,
            )
        if segment_heading_deg is None:
            segment_heading_deg = torch.empty(
                (batch, segments),
                device=self.device,
                dtype=self.dtype,
            ).uniform_(0.0, 360.0, generator=self._generator)
        else:
            segment_heading_deg = self._checked_segment_tensor(
                segment_heading_deg,
                (batch, segments),
                self.dtype,
            )
        site_stock = torch.tensor(
            [
                site.available_volume_l
                if site.available_volume_l != float("inf")
                else torch.finfo(self.dtype).max
                for site in self.sites
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.state = TensorOperationsState(
            minute=torch.zeros(batch, device=self.device, dtype=torch.long),
            resource_xy=resource_xy,
            resource_status=torch.full(
                (batch, resources),
                int(TensorResourceStatus.AVAILABLE),
                device=self.device,
                dtype=torch.uint8,
            ),
            resource_eta_min=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_leg_total_min=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_leg_start_xy=resource_xy.clone(),
            resource_leg_end_xy=resource_xy.clone(),
            resource_payload_fraction=torch.ones((batch, resources), device=self.device, dtype=self.dtype),
            resource_endurance_remaining_min=self.resource_endurance_min[None, :].expand(batch, -1).clone(),
            resource_target_index=torch.full((batch, resources), -1, device=self.device, dtype=torch.long),
            resource_queue_age_min=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_reserved_load_l=torch.zeros((batch, resources), device=self.device, dtype=self.dtype),
            resource_site_index=self.home_site_index[None, :].expand(batch, -1).clone(),
            resource_service_cycles=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_attempted_tasks=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_accepted_tasks=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            segment_xy=segment_xy,
            segment_heading_deg=segment_heading_deg,
            segment_kind=segment_kind,
            segment_priority=segment_priority,
            segment_required_l=segment_required_l,
            segment_remaining_l=segment_required_l.clone(),
            site_remaining_l=site_stock[None, :].expand(batch, -1).clone(),
            done=torch.zeros(batch, device=self.device, dtype=torch.bool),
        )
        return self.observations()

    def _checked_segment_tensor(
        self,
        value: Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"expected tensor shape {shape}, received {tuple(tensor.shape)}")
        return tensor.clone()

    @torch.no_grad()
    def reset_done(self, done: Tensor) -> TensorOperationsObservation:
        """Replace completed batch elements without transferring indices to the host."""

        done = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        if tuple(done.shape) != (self.batch_size,):
            raise ValueError(f"done must have shape {(self.batch_size,)}")
        old_state = self.state
        self.reset()
        new_state = self.state
        for state_field in fields(TensorOperationsState):
            old_value = getattr(old_state, state_field.name)
            new_value = getattr(new_state, state_field.name)
            selection = done.reshape(
                self.batch_size,
                *((1,) * (old_value.ndim - 1)),
            )
            setattr(
                new_state,
                state_field.name,
                torch.where(selection, new_value, old_value),
            )
        self.state = new_state
        return self.observations()

    def _travel_minutes(self, origin_xy: Tensor, destination_xy: Tensor) -> Tensor:
        distance_cells = torch.linalg.vector_norm(destination_xy - origin_xy, dim=-1)
        return distance_cells / torch.clamp(self.resource_speed_cells_min[None, :], min=1e-6)

    def _recovery_minutes_from(self, destination_xy: Tensor) -> Tensor:
        # destination_xy: [B,N,2].  Return nearest compatible fuel/charge site.
        delta = destination_xy[:, :, None, :] - self.site_xy[None, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        travel = distance / torch.clamp(
            self.resource_speed_cells_min[None, :, None],
            min=1e-6,
        )
        valid = self.resource_site_compatible & self.site_refuels[None, :]
        return torch.where(
            valid[None, :, :],
            travel,
            torch.full_like(travel, torch.inf),
        ).amin(dim=-1)

    def _site_commitment_counts(self) -> Tensor:
        """Count aircraft inbound, queued, or servicing at each approach."""

        state = self.state
        committed = (
            (state.resource_status == int(TensorResourceStatus.TO_SERVICE))
            | (state.resource_status == int(TensorResourceStatus.QUEUED))
            | (state.resource_status == int(TensorResourceStatus.SERVICING))
        )
        committed_site = torch.where(
            state.resource_status == int(TensorResourceStatus.TO_SERVICE),
            state.resource_target_index,
            state.resource_site_index,
        )
        counts = torch.zeros(
            (self.batch_size, self.num_sites),
            device=self.device,
            dtype=torch.long,
        )
        for site_index in range(self.num_sites):
            counts[:, site_index] = (committed & (committed_site == site_index)).sum(dim=1)
        return counts

    def _airborne_service_workload_min(self) -> Tensor:
        """Conservative work already ahead at hover-fill and scoop sites."""

        state = self.state
        workload = torch.zeros(
            (self.batch_size, self.num_sites),
            device=self.device,
            dtype=self.dtype,
        )
        for site_index in range(self.num_sites):
            if self.sites[site_index].service_mode == "land":
                continue
            at_site = state.resource_site_index == site_index
            servicing = (state.resource_status == int(TensorResourceStatus.SERVICING)) & at_site
            workload[:, site_index] += torch.where(
                servicing,
                state.resource_eta_min.to(self.dtype),
                torch.zeros_like(state.resource_eta_min, dtype=self.dtype),
            ).sum(dim=1)
            awaiting = (
                (state.resource_status == int(TensorResourceStatus.TO_SERVICE))
                & (state.resource_target_index == site_index)
            ) | ((state.resource_status == int(TensorResourceStatus.QUEUED)) & at_site)
            missing_l = self.resource_payload_l[None, :] * (1.0 - state.resource_payload_fraction)
            payload_supported = self.resource_site_payload_compatible[
                None,
                :,
                site_index,
            ]
            service_l = torch.where(
                payload_supported,
                missing_l,
                torch.zeros_like(missing_l),
            )
            duration = torch.ceil(
                self.site_turnaround_min[site_index] + service_l / self.site_rate_l_min[site_index]
            )
            workload[:, site_index] += torch.where(
                awaiting,
                duration,
                torch.zeros_like(duration),
            ).sum(dim=1)
        return workload

    @torch.no_grad()
    def action_mask(self) -> Tensor:
        state = self.state
        batch, resources = self.batch_size, self.num_resources
        available = state.resource_status == int(TensorResourceStatus.AVAILABLE)
        mask = torch.zeros(
            (batch, resources, self.num_tasks),
            device=self.device,
            dtype=torch.bool,
        )
        mask[..., 0] = True

        segment_destination = state.segment_xy[:, None, :, :].expand(-1, resources, -1, -1)
        segment_origin = state.resource_xy[:, :, None, :].expand(-1, -1, self.max_segments, -1)
        segment_travel = torch.linalg.vector_norm(
            segment_destination - segment_origin,
            dim=-1,
        ) / torch.clamp(self.resource_speed_cells_min[None, :, None], min=1e-6)
        # Recovery depends on each segment, so compute it without a host loop.
        segment_to_site = torch.linalg.vector_norm(
            segment_destination[:, :, :, None, :] - self.site_xy[None, None, None, :, :],
            dim=-1,
        ) / torch.clamp(
            self.resource_speed_cells_min[None, :, None, None],
            min=1e-6,
        )
        recovery_valid = (self.resource_site_compatible & self.site_refuels[None, :])[None, :, None, :]
        segment_recovery = torch.where(
            recovery_valid,
            segment_to_site,
            torch.full_like(segment_to_site, torch.inf),
        ).amin(dim=-1)
        kind = self.resource_kind[None, :, None]
        kind_compatible = (state.segment_kind[:, None, :] == 2) | (state.segment_kind[:, None, :] == kind)
        payload_ready = (
            state.resource_payload_fraction >= self.config.suppression.minimum_dispatch_payload_fraction
        )
        wind_safe = self.wind_speed_m_s[:, None] <= self.resource_max_wind_m_s[None, :]
        segment_energy = segment_travel + self.resource_dispatch_latency_min[None, :, None] + segment_recovery
        segment_safe = (
            segment_energy
            <= state.resource_endurance_remaining_min[:, :, None] - self.resource_reserve_min[None, :, None]
        )
        segment_open = state.segment_remaining_l > 1e-3
        mask[..., 1 : 1 + self.max_segments] = (
            available[:, :, None]
            & payload_ready[:, :, None]
            & wind_safe[:, :, None]
            & kind_compatible
            & segment_safe
            & segment_open[:, None, :]
            & (~state.done[:, None, None])
        )

        site_destination = self.site_xy[None, None, :, :].expand(batch, resources, -1, -1)
        site_origin = state.resource_xy[:, :, None, :].expand(-1, -1, self.num_sites, -1)
        site_travel = torch.linalg.vector_norm(
            site_destination - site_origin,
            dim=-1,
        ) / torch.clamp(self.resource_speed_cells_min[None, :, None], min=1e-6)
        projected_minute = (
            state.minute[:, None, None]
            + torch.ceil(site_travel).long()
            + self.resource_dispatch_latency_min[None, :, None].long()
        )
        site_open = (projected_minute >= self.site_open_min[None, None, :]) & (
            projected_minute < self.site_close_min[None, None, :]
        )
        site_wind_safe = self.wind_speed_m_s[:, None, None] <= self.site_max_wind_m_s[None, None, :]
        payload_needed = state.resource_payload_fraction < 1.0 - 1e-6
        payload_possible = self.resource_site_payload_compatible[None, :, :] & (
            state.site_remaining_l[:, None, :] > 1e-3
        )
        fuel_needed = state.resource_endurance_remaining_min < self.resource_endurance_min[None, :] - 1e-6
        useful = (payload_needed[:, :, None] & payload_possible) | (
            fuel_needed[:, :, None] & self.site_refuels[None, None, :]
        )
        # Compute continuation to a fuel/charge site for non-refuelling payload sites.
        site_to_site = torch.linalg.vector_norm(
            self.site_xy[None, :, None, :] - self.site_xy[None, None, :, :],
            dim=-1,
        )
        continuation = site_to_site[None, None, :, :] / torch.clamp(
            self.resource_speed_cells_min[None, :, None, None],
            min=1e-6,
        )
        continuation_valid = (self.resource_site_compatible & self.site_refuels[None, :])[None, :, None, :]
        nearest_fuel = torch.where(
            continuation_valid,
            continuation,
            torch.full_like(continuation, torch.inf),
        ).amin(dim=-1)
        site_recovery = torch.where(
            self.site_refuels[None, None, :],
            torch.zeros_like(nearest_fuel),
            nearest_fuel,
        )
        missing_l = self.resource_payload_l[None, :, None] * (
            1.0 - state.resource_payload_fraction[:, :, None]
        )
        desired_l = torch.where(
            self.resource_site_payload_compatible[None, :, :],
            missing_l,
            torch.zeros_like(missing_l),
        )
        service_l = torch.minimum(
            desired_l,
            state.site_remaining_l[:, None, :],
        )
        own_service_min = torch.ceil(
            self.site_turnaround_min[None, None, :] + service_l / self.site_rate_l_min[None, None, :]
        )
        airborne_site = (self.site_mode != self._MODE_IDS["land"])[None, None, :]
        airborne_service_min = torch.where(
            airborne_site,
            own_service_min + self._airborne_service_workload_min()[:, None, :],
            torch.zeros_like(own_service_min),
        )
        site_safe = (
            site_travel
            + self.resource_dispatch_latency_min[None, :, None]
            + airborne_service_min
            + site_recovery
            <= state.resource_endurance_remaining_min[:, :, None] - self.resource_reserve_min[None, :, None]
        )
        remaining_approach = torch.clamp(
            self.site_approach_capacity[None, :] - self._site_commitment_counts(),
            min=0,
        )
        mask[..., 1 + self.max_segments :] = (
            available[:, :, None]
            & self.resource_site_compatible[None, :, :]
            & useful
            & site_open
            & site_wind_safe
            & site_safe
            & (remaining_approach[:, None, :] > 0)
            & (~state.done[:, None, None])
        )
        return mask

    @torch.no_grad()
    def observations(self) -> TensorOperationsObservation:
        state = self.state
        task = torch.zeros(
            (self.batch_size, self.num_tasks, TASK_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        valid = torch.ones(
            (self.batch_size, self.num_tasks),
            device=self.device,
            dtype=torch.bool,
        )
        task[:, 0, 0] = 1.0
        task[:, 0, 2] = self.site_xy[self.home_site_index[0], 0] / max(self.config.width - 1, 1)
        task[:, 0, 3] = self.site_xy[self.home_site_index[0], 1] / max(self.config.height - 1, 1)
        segment_slice = slice(1, 1 + self.max_segments)
        task[:, segment_slice, 0] = 1.0
        task[:, segment_slice, 1] = float(TaskKind.AERIAL_LINE) / float(TaskKind.AERIAL_LINE)
        task[:, segment_slice, 2] = state.segment_xy[..., 0] / max(self.config.width - 1, 1)
        task[:, segment_slice, 3] = state.segment_xy[..., 1] / max(self.config.height - 1, 1)
        task[:, segment_slice, 4] = state.segment_priority / 12.0
        task[:, segment_slice, 7] = min(
            self.num_resources / TASK_CAPACITY_SCALE,
            1.0,
        )
        task[:, segment_slice, 10] = 1.0
        radians = torch.deg2rad(state.segment_heading_deg)
        task[:, segment_slice, 11] = torch.sin(radians)
        task[:, segment_slice, 12] = torch.cos(radians)
        task[:, segment_slice, 15] = 1.0
        site_slice = slice(1 + self.max_segments, self.num_tasks)
        task[:, site_slice, 0] = 1.0
        task[:, site_slice, 1] = float(TaskKind.SERVICE) / float(TaskKind.AERIAL_LINE)
        task[:, site_slice, 2] = self.site_xy[None, :, 0] / max(self.config.width - 1, 1)
        task[:, site_slice, 3] = self.site_xy[None, :, 1] / max(self.config.height - 1, 1)
        queue_counts = torch.zeros(
            (self.batch_size, self.num_sites),
            device=self.device,
            dtype=self.dtype,
        )
        for site_index in range(self.num_sites):
            queue_counts[:, site_index] = (
                (state.resource_status == int(TensorResourceStatus.QUEUED))
                & (state.resource_target_index == site_index)
            ).sum(dim=1)
        finite_stock = torch.tensor(
            [site.available_volume_l != float("inf") for site in self.sites],
            device=self.device,
            dtype=torch.bool,
        )
        initial_stock = torch.tensor(
            [
                site.available_volume_l if site.available_volume_l != float("inf") else 1.0
                for site in self.sites
            ],
            device=self.device,
            dtype=self.dtype,
        )
        stock_fraction = torch.where(
            finite_stock[None, :],
            state.site_remaining_l / initial_stock[None, :],
            torch.ones_like(state.site_remaining_l),
        )
        task[:, site_slice, 4] = (
            1.0 + 0.25 * stock_fraction - 0.20 * queue_counts / self.site_bays[None, :]
        ) / 12.0
        remaining_approach = torch.clamp(
            self.site_approach_capacity[None, :] - self._site_commitment_counts(),
            min=0,
        )
        task[:, site_slice, 7] = torch.clamp(
            remaining_approach / TASK_CAPACITY_SCALE,
            max=1.0,
        )
        task[:, site_slice, 13] = 1.0
        task[:, site_slice, 14] = 1.0
        task[:, site_slice, 16] = self.site_payload_service[None, :, 1].to(self.dtype)
        task[:, site_slice, 17] = self.site_payload_service[None, :, 0].to(self.dtype)
        task[:, site_slice, 18] = self.site_refuels[None, :].to(self.dtype)
        task[:, site_slice, 19] = torch.clamp(
            self.site_rate_l_min[None, :] / 20_000.0,
            max=1.0,
        )
        task[:, site_slice, 20] = torch.clamp(
            queue_counts / self.site_bays[None, :] / 4.0,
            max=1.0,
        )

        resource = torch.zeros(
            (self.batch_size, self.num_resources, RESOURCE_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        resource[..., 0] = state.resource_xy[..., 0] / max(self.config.width - 1, 1)
        resource[..., 1] = state.resource_xy[..., 1] / max(self.config.height - 1, 1)
        resource[..., 2] = self.resource_kind[None, :] / 3.0
        resource[..., 3] = state.resource_payload_fraction
        resource[..., 4] = state.resource_status / float(TensorResourceStatus.WITHDRAWN)
        resource[..., 5] = state.resource_eta_min / max(self.config.horizon_min, 1)
        resource[..., 6] = (
            self.resource_endurance_min[None, :] - state.resource_endurance_remaining_min
        ) / self.resource_endurance_min[None, :]
        resource[..., 7] = self.resource_speed_cells_min[None, :] * self.config.cell_size_m / 60.0 / 80.0
        resource[..., 8] = self.resource_payload_l[None, :] / 12000.0
        resource[..., 9] = state.resource_service_cycles / 20.0
        resource[..., 11] = state.minute[:, None] / max(self.config.horizon_min, 1)
        resource[..., 12] = state.resource_accepted_tasks / torch.clamp(
            state.resource_attempted_tasks,
            min=1,
        )
        resource[..., 13] = state.resource_endurance_remaining_min / self.resource_endurance_min[None, :]
        resource[..., 14] = (state.resource_site_index >= 0).to(self.dtype)
        resource[..., 15] = state.resource_queue_age_min / max(self.config.horizon_min, 1)
        resource[..., 16] = (state.resource_status == int(TensorResourceStatus.TO_SERVICE)).to(self.dtype)

        global_state = torch.zeros(
            (self.batch_size, ACTOR_GLOBAL_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        remaining_fraction = state.segment_remaining_l.sum(dim=1) / torch.clamp(
            state.segment_required_l.sum(dim=1),
            min=1.0,
        )
        global_state[:, 0] = (state.segment_remaining_l > 1e-3).to(self.dtype).mean(dim=1)
        global_state[:, 1] = 1.0 - remaining_fraction
        global_state[:, 2] = (state.segment_priority * (state.segment_remaining_l > 1e-3)).mean(dim=1)
        global_state[:, 6] = (
            (state.resource_status == int(TensorResourceStatus.AVAILABLE)).to(self.dtype).mean(dim=1)
        )
        global_state[:, 8] = state.minute / max(self.config.horizon_min, 1)
        global_state[:, 9] = (
            self.resource_endurance_min[None, :] - state.resource_endurance_remaining_min
        ).sum(dim=1) / 500.0
        return TensorOperationsObservation(
            resource=resource,
            tasks=task,
            action_mask=self.action_mask(),
            task_valid=valid,
            global_state=global_state,
        )

    @torch.no_grad()
    def critic_state(self) -> Tensor:
        """Privileged fixed-shape team state for centralized training."""

        state = self.state
        value = torch.zeros(
            (self.batch_size, 12),
            device=self.device,
            dtype=self.dtype,
        )
        required = torch.clamp(state.segment_required_l.sum(dim=1), min=1.0)
        remaining = state.segment_remaining_l.sum(dim=1)
        value[:, 0] = (state.segment_remaining_l > 1e-3).to(self.dtype).mean(dim=1)
        value[:, 1] = 1.0 - remaining / required
        value[:, 2] = (state.segment_priority * state.segment_remaining_l / required[:, None]).sum(dim=1)
        value[:, 3] = state.resource_payload_fraction.mean(dim=1)
        value[:, 4] = (state.resource_endurance_remaining_min / self.resource_endurance_min[None, :]).mean(
            dim=1
        )
        value[:, 5] = (
            (state.resource_status == int(TensorResourceStatus.AVAILABLE)).to(self.dtype).mean(dim=1)
        )
        value[:, 6] = (state.resource_status == int(TensorResourceStatus.QUEUED)).to(self.dtype).mean(dim=1)
        value[:, 7] = (
            (state.resource_status == int(TensorResourceStatus.SERVICING)).to(self.dtype).mean(dim=1)
        )
        value[:, 8] = state.resource_queue_age_min.to(self.dtype).mean(dim=1) / max(
            self.config.horizon_min, 1
        )
        finite_stock = torch.tensor(
            [site.available_volume_l != float("inf") for site in self.sites],
            device=self.device,
            dtype=torch.bool,
        )
        initial_stock = torch.tensor(
            [
                site.available_volume_l if site.available_volume_l != float("inf") else 1.0
                for site in self.sites
            ],
            device=self.device,
            dtype=self.dtype,
        )
        stock_fraction = torch.where(
            finite_stock[None, :],
            state.site_remaining_l / initial_stock[None, :],
            torch.ones_like(state.site_remaining_l),
        )
        value[:, 9] = stock_fraction.mean(dim=1)
        value[:, 10] = state.minute / max(self.config.horizon_min, 1)
        value[:, 11] = state.done.to(self.dtype)
        return value

    @torch.no_grad()
    def step(self, actions: Tensor) -> TensorOperationsStep:
        state = self.state
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        if tuple(actions.shape) != (self.batch_size, self.num_resources):
            raise ValueError(
                "actions must have shape "
                f"{(self.batch_size, self.num_resources)}, received {tuple(actions.shape)}"
            )
        mask = self.action_mask()
        in_range = (actions >= 0) & (actions < self.num_tasks)
        safe_actions = torch.where(in_range, actions, torch.zeros_like(actions))
        selected_valid = mask.gather(2, safe_actions[..., None])[..., 0]
        blocked = (~selected_valid) & (safe_actions != 0)
        safe_actions = torch.where(selected_valid, safe_actions, torch.zeros_like(safe_actions))
        remaining_approach = torch.clamp(
            self.site_approach_capacity[None, :] - self._site_commitment_counts(),
            min=0,
        )
        for site_index in range(self.num_sites):
            site_action = 1 + self.max_segments + site_index
            selected_site = safe_actions == site_action
            within_approach_capacity = (
                selected_site.long().cumsum(dim=1) <= remaining_approach[:, site_index, None]
            )
            approach_blocked = selected_site & (~within_approach_capacity)
            blocked |= approach_blocked
            safe_actions = torch.where(
                approach_blocked,
                torch.zeros_like(safe_actions),
                safe_actions,
            )
        active_assignment = safe_actions != 0
        state.resource_attempted_tasks += (actions != 0).long()
        state.resource_accepted_tasks += active_assignment.long()

        to_segment = (safe_actions >= 1) & (safe_actions <= self.max_segments)
        to_site = safe_actions > self.max_segments
        segment_index = torch.clamp(safe_actions - 1, 0, self.max_segments - 1)
        site_index = torch.clamp(
            safe_actions - 1 - self.max_segments,
            0,
            self.num_sites - 1,
        )
        target_xy = torch.where(
            to_segment[..., None],
            state.segment_xy.gather(
                1,
                segment_index[..., None].expand(-1, -1, 2),
            ),
            self.site_xy[site_index],
        )
        travel = torch.linalg.vector_norm(target_xy - state.resource_xy, dim=-1) / torch.clamp(
            self.resource_speed_cells_min[None, :],
            min=1e-6,
        )
        eta = torch.ceil(travel + self.resource_dispatch_latency_min[None, :]).long()
        eta.clamp_(min=1)
        state.resource_leg_start_xy = torch.where(
            active_assignment[..., None],
            state.resource_xy,
            state.resource_leg_start_xy,
        )
        state.resource_leg_end_xy = torch.where(
            active_assignment[..., None],
            target_xy,
            state.resource_leg_end_xy,
        )
        state.resource_eta_min = torch.where(
            active_assignment,
            eta,
            state.resource_eta_min,
        )
        state.resource_leg_total_min = torch.where(
            active_assignment,
            eta,
            state.resource_leg_total_min,
        )
        state.resource_target_index = torch.where(
            to_segment,
            segment_index,
            torch.where(to_site, site_index, state.resource_target_index),
        )
        state.resource_status = torch.where(
            to_segment,
            torch.full_like(
                state.resource_status,
                int(TensorResourceStatus.TO_ATTACK),
            ),
            torch.where(
                to_site,
                torch.full_like(
                    state.resource_status,
                    int(TensorResourceStatus.TO_SERVICE),
                ),
                state.resource_status,
            ),
        )
        state.resource_site_index = torch.where(
            active_assignment,
            torch.full_like(state.resource_site_index, -1),
            state.resource_site_index,
        )

        total_delivered = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        total_wasted = torch.zeros_like(total_delivered)
        flight_minutes = torch.zeros_like(total_delivered)
        queue_minutes = torch.zeros_like(total_delivered)
        for _ in range(self.config.decision_interval_min):
            delivered, wasted, flying, queued = self._advance_minute()
            total_delivered += delivered
            total_wasted += wasted
            flight_minutes += flying
            queue_minutes += queued
        completion = state.segment_remaining_l <= 1e-3
        if self.terminate_on_completion:
            state.done |= completion.all(dim=1)
        state.done |= state.minute >= self.config.horizon_min
        reward = (
            total_delivered / 5000.0
            - total_wasted / 10000.0
            - 0.002 * flight_minutes
            - 0.003 * queue_minutes
            - 0.02 * blocked.sum(dim=1)
        )
        return TensorOperationsStep(
            observation=self.observations(),
            reward=reward,
            done=state.done.clone(),
            delivered_l=total_delivered,
            wasted_l=total_wasted,
            blocked_actions=blocked.sum(dim=1),
        )

    def _advance_minute(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = self.state
        state.minute += (~state.done).long()
        flying = (state.resource_status == int(TensorResourceStatus.TO_ATTACK)) | (
            state.resource_status == int(TensorResourceStatus.TO_SERVICE)
        )
        service_site_index = torch.clamp(
            state.resource_site_index,
            0,
            self.num_sites - 1,
        )
        airborne_service = (
            (state.resource_status == int(TensorResourceStatus.QUEUED))
            | (state.resource_status == int(TensorResourceStatus.SERVICING))
        ) & (self.site_mode[service_site_index] != self._MODE_IDS["land"])
        endurance_active = flying | airborne_service
        state.resource_eta_min = torch.where(
            flying,
            torch.clamp(state.resource_eta_min - 1, min=0),
            state.resource_eta_min,
        )
        state.resource_endurance_remaining_min -= endurance_active.to(self.dtype)
        fraction = torch.where(
            flying,
            1.0 - state.resource_eta_min / torch.clamp(state.resource_leg_total_min, min=1),
            torch.zeros_like(state.resource_eta_min, dtype=self.dtype),
        )
        interpolated = (
            state.resource_leg_start_xy
            + (state.resource_leg_end_xy - state.resource_leg_start_xy) * fraction[..., None]
        )
        state.resource_xy = torch.where(
            flying[..., None],
            interpolated,
            state.resource_xy,
        )
        arrived_attack = (state.resource_status == int(TensorResourceStatus.TO_ATTACK)) & (
            state.resource_eta_min == 0
        )
        arrived_site = (state.resource_status == int(TensorResourceStatus.TO_SERVICE)) & (
            state.resource_eta_min == 0
        )

        contribution_l = (
            self.resource_payload_l[None, :] * state.resource_payload_fraction * arrived_attack.to(self.dtype)
        )
        contribution_by_segment = torch.zeros(
            (self.batch_size, self.max_segments),
            device=self.device,
            dtype=self.dtype,
        )
        contribution_by_segment.scatter_add_(
            1,
            torch.clamp(state.resource_target_index, 0, self.max_segments - 1),
            contribution_l,
        )
        accepted_by_segment = torch.minimum(
            contribution_by_segment,
            state.segment_remaining_l,
        )
        excess_by_segment = contribution_by_segment - accepted_by_segment
        delivered = (accepted_by_segment * state.segment_priority).sum(dim=1)
        wasted = excess_by_segment.sum(dim=1)
        state.segment_remaining_l = torch.clamp(
            state.segment_remaining_l - contribution_by_segment,
            min=0.0,
        )
        state.resource_payload_fraction = torch.where(
            arrived_attack,
            torch.zeros_like(state.resource_payload_fraction),
            state.resource_payload_fraction,
        )
        state.resource_status = torch.where(
            arrived_attack,
            torch.full_like(
                state.resource_status,
                int(TensorResourceStatus.AVAILABLE),
            ),
            state.resource_status,
        )
        state.resource_status = torch.where(
            arrived_site,
            torch.full_like(
                state.resource_status,
                int(TensorResourceStatus.QUEUED),
            ),
            state.resource_status,
        )
        state.resource_site_index = torch.where(
            arrived_site,
            state.resource_target_index,
            state.resource_site_index,
        )

        servicing = state.resource_status == int(TensorResourceStatus.SERVICING)
        state.resource_eta_min = torch.where(
            servicing,
            torch.clamp(state.resource_eta_min - 1, min=0),
            state.resource_eta_min,
        )
        service_complete = servicing & (state.resource_eta_min == 0)
        state.resource_payload_fraction = torch.where(
            service_complete,
            torch.clamp(
                state.resource_payload_fraction
                + state.resource_reserved_load_l / torch.clamp(self.resource_payload_l[None, :], min=1.0),
                max=1.0,
            ),
            state.resource_payload_fraction,
        )
        completed_site = torch.clamp(
            state.resource_site_index,
            0,
            self.num_sites - 1,
        )
        resets_endurance = self.site_refuels[completed_site]
        state.resource_endurance_remaining_min = torch.where(
            service_complete & resets_endurance,
            self.resource_endurance_min[None, :],
            state.resource_endurance_remaining_min,
        )
        state.resource_service_cycles += service_complete.long()
        state.resource_reserved_load_l = torch.where(
            service_complete,
            torch.zeros_like(state.resource_reserved_load_l),
            state.resource_reserved_load_l,
        )
        state.resource_status = torch.where(
            service_complete,
            torch.full_like(
                state.resource_status,
                int(TensorResourceStatus.AVAILABLE),
            ),
            state.resource_status,
        )

        queued_before = state.resource_status == int(TensorResourceStatus.QUEUED)
        state.resource_queue_age_min += queued_before.long()
        self._admit_queues()
        state.resource_queue_age_min = torch.where(
            state.resource_status == int(TensorResourceStatus.QUEUED),
            state.resource_queue_age_min,
            torch.zeros_like(state.resource_queue_age_min),
        )
        exhausted = state.resource_endurance_remaining_min <= 0.0
        state.resource_status = torch.where(
            exhausted,
            torch.full_like(
                state.resource_status,
                int(TensorResourceStatus.WITHDRAWN),
            ),
            state.resource_status,
        )
        return (
            delivered,
            wasted,
            endurance_active.to(self.dtype).sum(dim=1),
            queued_before.to(self.dtype).sum(dim=1),
        )

    def _admit_queues(self) -> None:
        state = self.state
        resource_order = torch.arange(
            self.num_resources,
            device=self.device,
            dtype=self.dtype,
        )[None, :]
        for site_index in range(self.num_sites):
            servicing = (state.resource_status == int(TensorResourceStatus.SERVICING)) & (
                state.resource_site_index == site_index
            )
            slots = torch.clamp(
                self.site_bays[site_index] - servicing.sum(dim=1),
                min=0,
            )
            waiting = (state.resource_status == int(TensorResourceStatus.QUEUED)) & (
                state.resource_site_index == site_index
            )
            score = torch.where(
                waiting,
                state.resource_queue_age_min.to(self.dtype) * (self.num_resources + 1.0) - resource_order,
                torch.full_like(resource_order, -torch.inf),
            )
            ranking = torch.argsort(score, dim=1, descending=True)
            for rank in range(self.num_resources):
                candidate = ranking[:, rank]
                candidate_waiting = waiting.gather(1, candidate[:, None])[:, 0]
                admit = candidate_waiting & (slots > rank)
                batch_index = torch.arange(self.batch_size, device=self.device)
                selected_batch = batch_index[admit]
                selected_resource = candidate[admit]
                payload_kind = self.resource_kind[selected_resource]
                payload_supported = self.site_payload_service[
                    site_index,
                    payload_kind,
                ]
                missing_l = self.resource_payload_l[selected_resource] * (
                    1.0
                    - state.resource_payload_fraction[
                        selected_batch,
                        selected_resource,
                    ]
                )
                desired_l = torch.where(
                    payload_supported,
                    missing_l,
                    torch.zeros_like(missing_l),
                )
                load_l = torch.minimum(
                    desired_l,
                    state.site_remaining_l[selected_batch, site_index],
                )
                state.site_remaining_l[selected_batch, site_index] -= load_l
                state.resource_reserved_load_l[
                    selected_batch,
                    selected_resource,
                ] = load_l
                duration = torch.ceil(
                    self.site_turnaround_min[site_index] + load_l / self.site_rate_l_min[site_index]
                ).long()
                state.resource_eta_min[
                    selected_batch,
                    selected_resource,
                ] = torch.clamp(duration, min=1)
                state.resource_status[
                    selected_batch,
                    selected_resource,
                ] = int(TensorResourceStatus.SERVICING)
