"""Massively batched wildfire decision environment for MARL training.

The environment is a calibrated decision surrogate.  It retains the coupled
state needed for assignment learning -- hidden fire truth, delayed belief,
subcell suppression, payload/endurance, service contention, and dynamic front
tasks -- while replacing the canonical level-set and aircraft models with a
fixed-shape probabilistic transition suitable for accelerator graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from aeolus.config import ScenarioConfig
from aeolus.core.tasks import (
    ACTOR_GLOBAL_FEATURE_DIM,
    CRITIC_GLOBAL_FEATURE_DIM,
    RESOURCE_FEATURE_DIM,
    TASK_CAPACITY_SCALE,
    TASK_FEATURE_DIM,
)
from aeolus.envs.tensor_operations import TensorOperationsObservation

if TYPE_CHECKING:
    from aeolus.core.simulator import AeolusSimulator

GPC_L_M2 = 3.785411784 / 9.290304
STATUS_AVAILABLE = 0
STATUS_TO_ATTACK = 1
STATUS_TO_SERVICE = 2
STATUS_QUEUED = 3
STATUS_SERVICING = 4
STATUS_WITHDRAWN = 5


@dataclass(frozen=True)
class TensorIncidentParameterRanges:
    """Per-world latent parameter ranges for robust surrogate training.

    These intervals are deliberately provisional. They are exposed as a
    calibration surface and sampled once per episode; they are not claims of
    posterior uncertainty for any particular incident.
    """

    spread_rate_scale: tuple[float, float] = (0.80, 1.60)
    wind_coefficient: tuple[float, float] = (0.052, 0.088)
    slope_coefficient: tuple[float, float] = (1.00, 1.70)
    burn_duration_min: tuple[float, float] = (15.0, 26.0)
    water_line_reduction: tuple[float, float] = (0.42, 0.68)
    retardant_line_reduction: tuple[float, float] = (0.66, 0.88)
    observation_weight: tuple[float, float] = (0.68, 0.94)

    def __post_init__(self) -> None:
        for name, bounds in self.__dict__.items():
            if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
                raise ValueError(f"invalid tensor incident parameter range: {name}")
        if self.water_line_reduction[1] > 1.0:
            raise ValueError("water line reduction cannot exceed one")
        if self.retardant_line_reduction[1] > 1.0:
            raise ValueError("retardant line reduction cannot exceed one")
        if self.observation_weight[1] > 1.0:
            raise ValueError("observation weight cannot exceed one")


class TensorIncidentState(NamedTuple):
    """Fixed-shape mutable world state represented only by tensors."""

    minute: Tensor
    unburned: Tensor
    burning: Tensor
    burned: Tensor
    belief_unburned: Tensor
    belief_burning: Tensor
    belief_burned: Tensor
    belief_uncertainty: Tensor
    fuel_factor: Tensor
    asset_value: Tensor
    slope_x: Tensor
    slope_y: Tensor
    barrier: Tensor
    water_coverage_gpc: Tensor
    retardant_coverage_gpc: Tensor
    water_line_strength: Tensor
    retardant_line_strength: Tensor
    wind_speed_m_s: Tensor
    wind_direction_rad: Tensor
    spread_rate_scale: Tensor
    wind_coefficient: Tensor
    slope_coefficient: Tensor
    burn_duration_min: Tensor
    water_line_reduction: Tensor
    retardant_line_reduction: Tensor
    observation_weight: Tensor
    resource_xy: Tensor
    resource_status: Tensor
    resource_arrival_min: Tensor
    resource_event_min: Tensor
    resource_service_start_min: Tensor
    resource_leg_start_min: Tensor
    resource_leg_start_xy: Tensor
    resource_leg_end_xy: Tensor
    resource_payload_fraction: Tensor
    resource_endurance_remaining_min: Tensor
    resource_target_site: Tensor
    resource_target_xy: Tensor
    resource_target_heading_rad: Tensor
    resource_reserved_load_l: Tensor
    resource_attempted_tasks: Tensor
    resource_accepted_tasks: Tensor
    resource_service_cycles: Tensor
    site_remaining_l: Tensor
    site_slot_available_min: Tensor
    contained: Tensor
    escaped: Tensor
    done: Tensor


class FrontSegments(NamedTuple):
    xy: Tensor
    heading_rad: Tensor
    priority: Tensor
    uncertainty: Tensor
    required_l: Tensor
    capacity: Tensor
    valid: Tensor


class TensorIncidentTransition(NamedTuple):
    state: TensorIncidentState
    resource: Tensor
    tasks: Tensor
    action_mask: Tensor
    task_valid: Tensor
    actor_global: Tensor
    critic_global: Tensor
    reward: Tensor
    delivered_l: Tensor
    wasted_l: Tensor
    blocked_actions: Tensor
    expected_loss: Tensor
    burned_fraction: Tensor
    constraint_costs: Tensor


@dataclass(frozen=True)
class TensorIncidentStep:
    observation: TensorOperationsObservation
    reward: Tensor
    done: Tensor
    delivered_l: Tensor
    wasted_l: Tensor
    blocked_actions: Tensor
    expected_loss: Tensor
    burned_fraction: Tensor
    constraint_costs: Tensor


class TensorIncidentEnv:
    """Fixed-shape fire/belief/operations environment for GPU-scale MARL.

    Actions share the canonical task-pointer layout:

    - ``0`` holds;
    - ``1..K`` assigns a resource to a belief-derived front segment;
    - ``K+1..K+S`` assigns it to a service site.

    Aircraft motion is event-level.  Fire and treatment fields advance at the
    tactical decision interval with a fixed number of probabilistic substeps.
    The actor observation is constructed from incident belief; hidden truth is
    exposed only through :meth:`critic_state`.
    """

    _KIND_IDS = {"retardant": 0, "water": 1}
    _MODE_IDS = {"land": 0, "hover_fill": 1, "scoop": 2}

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        batch_size: int,
        max_segments: int = 32,
        grid_size: int = 64,
        fire_substeps: int = 2,
        observation_period_min: int = 12,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        terminate_on_completion: bool = False,
        terminate_on_escape: bool | None = None,
        parameter_ranges: TensorIncidentParameterRanges | None = None,
    ):
        if batch_size < 1 or max_segments < 1:
            raise ValueError("batch size and segment count must be positive")
        if grid_size < 16 or fire_substeps < 1 or observation_period_min < 1:
            raise ValueError("invalid tensor incident discretization")
        if max_segments > grid_size * grid_size:
            raise ValueError("max_segments cannot exceed the surrogate grid cell count")
        if not config.service_sites:
            raise ValueError("tensor incident training requires service sites")
        resources = tuple(resource for resource in config.resources if resource.kind in self._KIND_IDS)
        if not resources:
            raise ValueError("tensor incident training requires water or retardant resources")

        self.config = config
        self.batch_size = int(batch_size)
        self.max_segments = int(max_segments)
        self.grid_size = int(grid_size)
        self.fire_substeps = int(fire_substeps)
        self.observation_period_min = int(observation_period_min)
        self.device = torch.device(device)
        self.dtype = dtype
        self.terminate_on_completion = bool(terminate_on_completion)
        self.terminate_on_escape = (
            config.terminate_on_escape if terminate_on_escape is None else bool(terminate_on_escape)
        )
        self.parameter_ranges = parameter_ranges or TensorIncidentParameterRanges()
        self.resources = resources
        self.sites = config.service_sites
        self.num_resources = len(resources)
        self.num_sites = len(self.sites)
        self.num_tasks = 1 + self.max_segments + self.num_sites
        self.max_bays = max(site.bays for site in self.sites)
        self._generator = torch.Generator(device=self.device)
        self._compiled_transition = None

        def tensor(values, *, tensor_dtype: torch.dtype = dtype) -> Tensor:
            return torch.as_tensor(values, device=self.device, dtype=tensor_dtype)

        self.resource_kind_names = tuple(resource.kind for resource in resources)
        self.resource_kind = tensor(
            [self._KIND_IDS[resource.kind] for resource in resources],
            tensor_dtype=torch.long,
        )
        self.resource_speed_m_min = tensor([resource.cruise_speed_m_s * 60.0 for resource in resources])
        self.resource_payload_l = tensor([resource.payload_l for resource in resources])
        self.resource_endurance_min = tensor([resource.endurance_min for resource in resources])
        self.resource_reserve_min = tensor(
            [
                max(
                    resource.reserve_endurance_min,
                    config.suppression.minimum_reserve_endurance_min,
                )
                for resource in resources
            ]
        )
        self.resource_dispatch_latency_min = tensor([resource.dispatch_latency_min for resource in resources])
        self.resource_max_wind_m_s = tensor(
            [
                min(resource.max_operating_wind_m_s, config.suppression.aviation_max_wind_m_s)
                for resource in resources
            ]
        )
        self.resource_drop_length_cells = tensor(
            [
                max(
                    resource.minimum_drop_length_m,
                    min(resource.maximum_drop_length_m, resource.retardant_length_m),
                )
                / self.surrogate_cell_m
                for resource in resources
            ]
        ).clamp(1.0, self.grid_size / 2.0)
        self.resource_drop_width_cells = tensor(
            [
                (
                    resource.retardant_width_m
                    if resource.kind == "retardant"
                    else max(2.0 * resource.water_radius_m, 30.0)
                )
                / self.surrogate_cell_m
                for resource in resources
            ]
        ).clamp(0.35, 4.0)
        self.resource_target_coverage_gpc = tensor(
            [
                resource.target_coverage_level_gpc if resource.kind == "retardant" else 2.0
                for resource in resources
            ]
        )

        x_scale = (self.grid_size - 1) / max(config.width - 1, 1)
        y_scale = (self.grid_size - 1) / max(config.height - 1, 1)
        self.site_xy = tensor([[site.x * x_scale, site.y * y_scale] for site in self.sites])
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
        self.site_open_min = tensor([site.open_minute for site in self.sites])
        self.site_close_min = tensor([site.close_minute for site in self.sites])
        self.site_max_wind_m_s = tensor([site.max_operating_wind_m_s for site in self.sites])
        self.site_refuels = tensor(
            [bool({"fuel", "charge"}.intersection(site.services)) for site in self.sites],
            tensor_dtype=torch.bool,
        )
        self.site_initial_l = tensor(
            [
                site.available_volume_l if site.available_volume_l != float("inf") else 1.0e12
                for site in self.sites
            ]
        )
        self.site_payload_service = torch.zeros(
            (self.num_sites, len(self._KIND_IDS)),
            device=self.device,
            dtype=torch.bool,
        )
        for site_index, site in enumerate(self.sites):
            for kind_name, kind_index in self._KIND_IDS.items():
                self.site_payload_service[site_index, kind_index] = kind_name in site.services
        resource_mode = torch.zeros(
            (self.num_resources, len(self._MODE_IDS)),
            device=self.device,
            dtype=torch.bool,
        )
        for resource_index, resource in enumerate(resources):
            for mode in resource.service_modes:
                resource_mode[resource_index, self._MODE_IDS[mode]] = True
        self.resource_site_compatible = resource_mode[:, self.site_mode] & (
            self.site_payload_service[:, self.resource_kind].T | self.site_refuels[None, :]
        )
        self.resource_site_payload_compatible = self.site_payload_service[:, self.resource_kind].T
        max_payload_by_site = torch.where(
            self.resource_site_payload_compatible,
            self.resource_payload_l[:, None],
            torch.zeros(
                (self.num_resources, self.num_sites),
                device=self.device,
                dtype=self.dtype,
            ),
        ).amax(dim=0)
        self.site_max_service_duration_min = torch.ceil(
            self.site_turnaround_min + max_payload_by_site / self.site_rate_l_min
        ).clamp_min(1.0)
        self.home_site_index = tensor(
            [self._home_site_index(resource_index) for resource_index in range(self.num_resources)],
            tensor_dtype=torch.long,
        )
        slot_index = torch.arange(self.max_bays, device=self.device)
        self.site_slot_valid = slot_index[None, :] < self.site_bays[:, None]

        coordinate = torch.arange(self.grid_size, device=self.device, dtype=self.dtype)
        self.grid_y, self.grid_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.cell_area_m2 = self.surrogate_cell_m**2
        self._neighbor_offsets = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )
        # Calibrated initial value from the local canonical-teacher check. The
        # inverse-cell-size factor preserves approximate physical front speed
        # when the surrogate grid resolution changes.
        self.base_ignition_rate_min = 0.006 * 180.0 / self.surrogate_cell_m

        self.state: TensorIncidentState
        self._observation: TensorOperationsObservation
        self._critic: Tensor
        self.reset(seed=config.seed)

    @property
    def surrogate_cell_m(self) -> float:
        domain_area = (
            self.config.width * self.config.height * self.config.cell_size_m * self.config.cell_size_m
        )
        return float((domain_area / (self.grid_size * self.grid_size)) ** 0.5)

    def _home_site_index(self, resource_index: int) -> int:
        resource = self.resources[resource_index]
        if resource.home_site_id is not None:
            return next(
                index for index, site in enumerate(self.sites) if site.site_id == resource.home_site_id
            )
        compatible = torch.nonzero(
            self.resource_site_compatible[resource_index],
            as_tuple=False,
        )
        if compatible.numel() == 0:
            raise ValueError(f"resource {resource.resource_id} has no compatible site")
        return int(compatible[0, 0])

    def _random_smooth(self, scale: float, kernel: int) -> Tensor:
        noise = torch.rand(
            (self.batch_size, 1, self.grid_size, self.grid_size),
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )
        return (
            F.avg_pool2d(
                noise * 2.0 - 1.0,
                kernel,
                stride=1,
                padding=kernel // 2,
            )[:, 0]
            * scale
        )

    def _random_parameter(self, bounds: tuple[float, float]) -> Tensor:
        return torch.empty(
            (self.batch_size,),
            device=self.device,
            dtype=self.dtype,
        ).uniform_(bounds[0], bounds[1], generator=self._generator)

    def _initial_state(self) -> TensorIncidentState:
        batch, size, resources = self.batch_size, self.grid_size, self.num_resources
        fuel = (0.95 + self._random_smooth(2.4, 9)).clamp(0.35, 1.55)
        slope_x = self._random_smooth(0.55, 11).clamp(-0.35, 0.35)
        slope_y = self._random_smooth(0.55, 11).clamp(-0.35, 0.35)
        barrier = torch.zeros((batch, size, size), device=self.device, dtype=torch.bool)
        barrier[:, 0] = True
        barrier[:, -1] = True
        barrier[:, :, 0] = True
        barrier[:, :, -1] = True
        barrier |= fuel < 0.43

        centre_x = torch.empty((batch, 3), device=self.device, dtype=self.dtype).uniform_(
            0.12 * size,
            0.88 * size,
            generator=self._generator,
        )
        centre_y = torch.empty_like(centre_x).uniform_(
            0.12 * size,
            0.88 * size,
            generator=self._generator,
        )
        asset_sigma = torch.empty_like(centre_x).uniform_(
            2.5,
            7.0,
            generator=self._generator,
        )
        asset_weight = torch.empty_like(centre_x).uniform_(
            0.5,
            1.5,
            generator=self._generator,
        )
        dx = self.grid_x[None, None] - centre_x[:, :, None, None]
        dy = self.grid_y[None, None] - centre_y[:, :, None, None]
        asset = (
            asset_weight[:, :, None, None]
            * torch.exp(-(dx.square() + dy.square()) / (2.0 * asset_sigma[:, :, None, None].square()))
        ).sum(dim=1)
        asset = asset / asset.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)

        ignition_x = torch.empty((batch,), device=self.device, dtype=self.dtype).uniform_(
            0.28 * size,
            0.72 * size,
            generator=self._generator,
        )
        ignition_y = torch.empty_like(ignition_x).uniform_(
            0.28 * size,
            0.72 * size,
            generator=self._generator,
        )
        ignition_radius = torch.empty_like(ignition_x).uniform_(
            1.0,
            2.4,
            generator=self._generator,
        )
        ignition_distance = torch.sqrt(
            (self.grid_x[None] - ignition_x[:, None, None]).square()
            + (self.grid_y[None] - ignition_y[:, None, None]).square()
        )
        burning = (ignition_distance <= ignition_radius[:, None, None]).to(self.dtype)
        burning = burning.masked_fill(barrier, 0.0)
        burned = torch.zeros_like(burning)
        unburned = (~barrier).to(self.dtype) - burning
        blurred_burning = F.avg_pool2d(
            burning[:, None],
            3,
            stride=1,
            padding=1,
        )[:, 0]
        belief_burning = blurred_burning.clamp(0.0, 1.0)
        belief_burned = burned.clone()
        belief_unburned = ((~barrier).to(self.dtype) - belief_burning).clamp(0.0, 1.0)

        wind_speed = torch.empty((batch,), device=self.device, dtype=self.dtype).uniform_(
            max(1.0, 0.55 * self.config.wind_speed_m_s),
            max(2.0, 1.45 * self.config.wind_speed_m_s),
            generator=self._generator,
        )
        wind_direction = torch.deg2rad(
            torch.full(
                (batch,),
                float(self.config.wind_direction_deg),
                device=self.device,
                dtype=self.dtype,
            )
        ) + torch.empty((batch,), device=self.device, dtype=self.dtype).uniform_(
            -pi / 3.0,
            pi / 3.0,
            generator=self._generator,
        )

        home_xy = self.site_xy[self.home_site_index]
        inf_min = torch.full((batch, resources), 1.0e9, device=self.device, dtype=self.dtype)
        slot_available = torch.where(
            self.site_slot_valid[None],
            torch.zeros((batch, self.num_sites, self.max_bays), device=self.device, dtype=self.dtype),
            torch.full(
                (batch, self.num_sites, self.max_bays),
                1.0e9,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        return TensorIncidentState(
            minute=torch.zeros(batch, device=self.device, dtype=self.dtype),
            unburned=unburned,
            burning=burning,
            burned=burned,
            belief_unburned=belief_unburned,
            belief_burning=belief_burning,
            belief_burned=belief_burned,
            belief_uncertainty=torch.full_like(burning, 0.18),
            fuel_factor=fuel,
            asset_value=asset,
            slope_x=slope_x,
            slope_y=slope_y,
            barrier=barrier,
            water_coverage_gpc=torch.zeros_like(burning),
            retardant_coverage_gpc=torch.zeros_like(burning),
            water_line_strength=torch.zeros_like(burning),
            retardant_line_strength=torch.zeros_like(burning),
            wind_speed_m_s=wind_speed,
            wind_direction_rad=wind_direction,
            spread_rate_scale=self._random_parameter(self.parameter_ranges.spread_rate_scale),
            wind_coefficient=self._random_parameter(self.parameter_ranges.wind_coefficient),
            slope_coefficient=self._random_parameter(self.parameter_ranges.slope_coefficient),
            burn_duration_min=self._random_parameter(self.parameter_ranges.burn_duration_min),
            water_line_reduction=self._random_parameter(self.parameter_ranges.water_line_reduction),
            retardant_line_reduction=self._random_parameter(self.parameter_ranges.retardant_line_reduction),
            observation_weight=self._random_parameter(self.parameter_ranges.observation_weight),
            resource_xy=home_xy[None].expand(batch, -1, -1).clone(),
            resource_status=torch.full(
                (batch, resources),
                STATUS_AVAILABLE,
                device=self.device,
                dtype=torch.uint8,
            ),
            resource_arrival_min=inf_min.clone(),
            resource_event_min=inf_min.clone(),
            resource_service_start_min=inf_min.clone(),
            resource_leg_start_min=torch.zeros((batch, resources), device=self.device, dtype=self.dtype),
            resource_leg_start_xy=home_xy[None].expand(batch, -1, -1).clone(),
            resource_leg_end_xy=home_xy[None].expand(batch, -1, -1).clone(),
            resource_payload_fraction=torch.ones((batch, resources), device=self.device, dtype=self.dtype),
            resource_endurance_remaining_min=self.resource_endurance_min[None].expand(batch, -1).clone(),
            resource_target_site=torch.full((batch, resources), -1, device=self.device, dtype=torch.long),
            resource_target_xy=home_xy[None].expand(batch, -1, -1).clone(),
            resource_target_heading_rad=torch.zeros((batch, resources), device=self.device, dtype=self.dtype),
            resource_reserved_load_l=torch.zeros((batch, resources), device=self.device, dtype=self.dtype),
            resource_attempted_tasks=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_accepted_tasks=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            resource_service_cycles=torch.zeros((batch, resources), device=self.device, dtype=torch.long),
            site_remaining_l=self.site_initial_l[None].expand(batch, -1).clone(),
            site_slot_available_min=slot_available,
            contained=torch.zeros(batch, device=self.device, dtype=torch.bool),
            escaped=torch.zeros(batch, device=self.device, dtype=torch.bool),
            done=torch.zeros(batch, device=self.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def reset(self, *, seed: int | None = None) -> TensorOperationsObservation:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        self.state = self._initial_state()
        self._refresh_observation()
        return self._observation

    @torch.no_grad()
    def reset_done(self, done: Tensor) -> TensorOperationsObservation:
        done = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        if tuple(done.shape) != (self.batch_size,):
            raise ValueError(f"done must have shape {(self.batch_size,)}")
        replacement = self._initial_state()
        merged: list[Tensor] = []
        for old_value, new_value in zip(self.state, replacement, strict=True):
            selection = done.reshape(self.batch_size, *((1,) * (old_value.ndim - 1)))
            merged.append(torch.where(selection, new_value, old_value))
        self.state = TensorIncidentState(*merged)
        self._refresh_observation()
        return self._observation

    @torch.no_grad()
    def initialize_fire_from_canonical(
        self,
        simulator: AeolusSimulator,
    ) -> TensorOperationsObservation:
        """Project one canonical snapshot into every surrogate batch world.

        This is a calibration bridge, not part of rollout collection. Static
        resource/site state and sampled latent parameters are retained from the
        latest reset while truth, belief, terrain, fuel, treatments, wind, and
        incident time are replaced by a common canonical snapshot.
        """

        canonical = simulator.config
        if (
            canonical.width != self.config.width
            or canonical.height != self.config.height
            or canonical.cell_size_m != self.config.cell_size_m
        ):
            raise ValueError("canonical and surrogate physical domains must match")

        truth = simulator.state.truth
        belief = simulator.state.belief

        def project(value, *, mode: str = "bilinear") -> Tensor:
            source = torch.as_tensor(
                value,
                device=self.device,
                dtype=self.dtype,
            )[None, None]
            kwargs = {"size": (self.grid_size, self.grid_size), "mode": mode}
            if mode in {"bilinear", "bicubic"}:
                kwargs["align_corners"] = False
            projected = F.interpolate(source, **kwargs)[0, 0]
            return projected[None].expand(self.batch_size, -1, -1).clone()

        phase = torch.as_tensor(truth.phase, device=self.device)
        source_unburned = (phase == 0).to(self.dtype)
        source_burning = (phase == 1).to(self.dtype)
        source_burned = (phase == 2).to(self.dtype)
        unburned = project(source_unburned, mode="area")
        burning = project(source_burning, mode="area")
        burned = project(source_burned, mode="area")
        barrier = project(truth.barrier, mode="nearest") >= 0.5
        total = (unburned + burning + burned).clamp_min(1.0e-6)
        unburned = (unburned / total).masked_fill(barrier, 0.0)
        burning = (burning / total).masked_fill(barrier, 0.0)
        burned = (burned / total).masked_fill(barrier, 0.0)

        source_barrier = torch.as_tensor(
            truth.barrier,
            device=self.device,
            dtype=torch.bool,
        )
        source_fuel_load = torch.as_tensor(
            truth.fuel_load,
            device=self.device,
            dtype=self.dtype,
        )
        reference_load = source_fuel_load.masked_select(~source_barrier).median().clamp_min(1.0e-6)
        fuel_factor = project(
            truth.fuel_remaining * truth.fuel_load / float(reference_load),
            mode="area",
        ).clamp(0.05, 2.0)
        asset_value = project(truth.asset_value, mode="area")
        elevation = project(truth.elevation_m, mode="bilinear")
        elevation_pad = F.pad(elevation[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
        slope_x = 0.5 * (elevation_pad[:, 1:-1, 2:] - elevation_pad[:, 1:-1, :-2]) / self.surrogate_cell_m
        slope_y = 0.5 * (elevation_pad[:, 2:, 1:-1] - elevation_pad[:, :-2, 1:-1]) / self.surrogate_cell_m

        belief_burned = project(belief.known_burned, mode="area").clamp(0.0, 1.0)
        source_active = torch.as_tensor(
            belief.intensity_mean,
            device=self.device,
            dtype=self.dtype,
        )
        source_active = 1.0 - torch.exp(-source_active / 700.0)
        belief_burning = project(source_active, mode="area") * (1.0 - belief_burned)
        belief_unburned = ((~barrier).to(self.dtype) - belief_burned - belief_burning).clamp(
            0.0,
            1.0,
        )
        belief_total = (belief_unburned + belief_burning + belief_burned).clamp_min(1.0e-6)
        belief_unburned /= belief_total
        belief_burning /= belief_total
        belief_burned /= belief_total
        source_uncertainty = torch.as_tensor(
            belief.intensity_std,
            device=self.device,
            dtype=self.dtype,
        ) / (
            torch.as_tensor(
                belief.intensity_mean,
                device=self.device,
                dtype=self.dtype,
            )
            + torch.as_tensor(
                belief.intensity_std,
                device=self.device,
                dtype=self.dtype,
            )
            + 1.0
        )
        belief_uncertainty = project(source_uncertainty, mode="area").clamp(0.0, 1.0)

        water_coverage = project(truth.water_coverage_gpc, mode="area")
        retardant_coverage = project(
            truth.retardant_effective_coverage_gpc,
            mode="area",
        )
        water_line = (1.0 - torch.exp(-water_coverage / 2.0)).clamp(0.0, 1.0)
        retardant_line = torch.maximum(
            (1.0 - torch.exp(-retardant_coverage / 2.0)).clamp(0.0, 1.0),
            project(truth.line_strength, mode="area").clamp(0.0, 1.0),
        )

        weather = simulator.current_weather()
        wind_speed = torch.as_tensor(
            weather["wind_speed_m_s"],
            device=self.device,
            dtype=self.dtype,
        ).mean()
        wind_direction = torch.deg2rad(
            torch.as_tensor(
                weather["wind_direction_deg"],
                device=self.device,
                dtype=self.dtype,
            )
        )
        mean_direction = torch.atan2(
            torch.sin(wind_direction).mean(),
            torch.cos(wind_direction).mean(),
        )
        self.state = self.state._replace(
            minute=torch.full_like(self.state.minute, float(simulator.state.minute)),
            unburned=unburned,
            burning=burning,
            burned=burned,
            belief_unburned=belief_unburned,
            belief_burning=belief_burning,
            belief_burned=belief_burned,
            belief_uncertainty=belief_uncertainty,
            fuel_factor=fuel_factor,
            asset_value=asset_value,
            slope_x=slope_x,
            slope_y=slope_y,
            barrier=barrier,
            water_coverage_gpc=water_coverage,
            retardant_coverage_gpc=retardant_coverage,
            water_line_strength=water_line,
            retardant_line_strength=retardant_line,
            wind_speed_m_s=wind_speed.expand(self.batch_size).clone(),
            wind_direction_rad=mean_direction.expand(self.batch_size).clone(),
            contained=torch.zeros_like(self.state.contained),
            escaped=torch.zeros_like(self.state.escaped),
            done=torch.zeros_like(self.state.done),
        )
        self._refresh_observation()
        return self._observation

    def _distance_minutes(
        self,
        origin: Tensor,
        destination: Tensor,
        *,
        resource_index: int | None = None,
    ) -> Tensor:
        delta = destination - origin
        physical = torch.sqrt(
            (delta[..., 0] * self.config.width * self.config.cell_size_m / self.grid_size).square()
            + (delta[..., 1] * self.config.height * self.config.cell_size_m / self.grid_size).square()
        )
        if resource_index is None:
            # The unindexed form is used for tensors with shape [batch,
            # resource, ...].  Keeping the resource axis explicit prevents a
            # silent broadcast against the batch axis when B happens to equal N.
            speed = self.resource_speed_m_min.reshape(
                1,
                self.num_resources,
                *((1,) * (physical.ndim - 2)),
            )
        else:
            speed = self.resource_speed_m_min[resource_index]
        return physical / speed.clamp_min(1.0)

    def _fire_transition_fields(
        self,
        unburned: Tensor,
        burning: Tensor,
        burned: Tensor,
        *,
        fuel_factor: Tensor,
        slope_x: Tensor,
        slope_y: Tensor,
        barrier: Tensor,
        wind_speed_m_s: Tensor,
        wind_direction_rad: Tensor,
        water_coverage_gpc: Tensor,
        retardant_coverage_gpc: Tensor,
        water_line_strength: Tensor,
        retardant_line_strength: Tensor,
        spread_rate_scale: Tensor,
        wind_coefficient: Tensor,
        slope_coefficient: Tensor,
        burn_duration_min: Tensor,
        water_line_reduction: Tensor,
        retardant_line_reduction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dt = float(self.config.decision_interval_min) / float(self.fire_substeps)
        height = width = self.grid_size
        for _ in range(self.fire_substeps):
            padded = F.pad(burning, (1, 1, 1, 1))
            hazard = torch.zeros_like(burning)
            for dx, dy in self._neighbor_offsets:
                source = padded[
                    :,
                    1 - dy : 1 - dy + height,
                    1 - dx : 1 - dx + width,
                ]
                direction = torch.atan2(
                    burning.new_tensor(float(dy)),
                    burning.new_tensor(float(dx)),
                )
                alignment = torch.cos(direction - wind_direction_rad[:, None, None])
                slope_projection = (slope_x * dx + slope_y * dy) / max(
                    (dx * dx + dy * dy) ** 0.5,
                    1.0,
                )
                directional = torch.exp(
                    wind_coefficient[:, None, None] * wind_speed_m_s[:, None, None] * alignment
                    + slope_coefficient[:, None, None] * slope_projection
                )
                diagonal = 0.70710678 if dx and dy else 1.0
                hazard = (
                    hazard
                    + self.base_ignition_rate_min
                    * spread_rate_scale[:, None, None]
                    * diagonal
                    * source
                    * directional
                )
            treatment = (
                (1.0 - retardant_line_reduction[:, None, None] * retardant_line_strength)
                * (1.0 - water_line_reduction[:, None, None] * water_line_strength)
                * torch.exp(-0.16 * retardant_coverage_gpc - 0.08 * water_coverage_gpc)
            ).clamp(0.02, 1.0)
            ignition_probability = 1.0 - torch.exp(-hazard * fuel_factor * treatment * dt)
            newly_burning = unburned * ignition_probability
            continuation = burning * torch.exp(-dt / burn_duration_min[:, None, None])
            continuation = continuation * (
                1.0 - water_line_reduction[:, None, None] * water_line_strength
            ).clamp(0.05, 1.0)
            burned_increment = (burning - continuation).clamp_min(0.0)
            unburned = (unburned - newly_burning).clamp(0.0, 1.0)
            burning = (newly_burning + continuation).clamp(0.0, 1.0)
            burned = (burned + burned_increment).clamp(0.0, 1.0)
            total = (unburned + burning + burned).clamp_min(1.0e-6)
            unburned = (unburned / total).masked_fill(barrier, 0.0)
            burning = (burning / total).masked_fill(barrier, 0.0)
            burned = (burned / total).masked_fill(barrier, 0.0)
        return unburned, burning, burned

    def _front_segments(self, state: TensorIncidentState) -> FrontSegments:
        frontier = (
            state.belief_burning
            * F.max_pool2d(
                state.belief_unburned[:, None],
                3,
                stride=1,
                padding=1,
            )[:, 0]
        )
        local_value = F.avg_pool2d(
            state.asset_value[:, None],
            9,
            stride=1,
            padding=4,
        )[:, 0]
        treatment_gap = 1.0 - torch.maximum(
            state.water_line_strength,
            state.retardant_line_strength,
        )
        score = (
            frontier
            * (0.35 + 0.45 * state.fuel_factor + 2.4 * local_value + 0.55 * state.belief_uncertainty)
            * (0.45 + 0.55 * treatment_gap)
        )
        local_maximum = (
            score
            >= F.max_pool2d(
                score[:, None],
                5,
                stride=1,
                padding=2,
            )[:, 0]
        )
        candidate = torch.where(local_maximum, score, torch.zeros_like(score))
        values, indices = torch.topk(
            candidate.reshape(self.batch_size, -1),
            self.max_segments,
            dim=1,
        )
        y = torch.div(indices, self.grid_size, rounding_mode="floor")
        x = indices % self.grid_size
        xy = torch.stack((x, y), dim=-1).to(self.dtype)
        valid = values > 1.0e-5

        padded = F.pad(state.belief_burning, (1, 1, 1, 1), mode="replicate")
        gradient_x = 0.5 * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
        gradient_y = 0.5 * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
        flat_index = indices
        gx = gradient_x.reshape(self.batch_size, -1).gather(1, flat_index)
        gy = gradient_y.reshape(self.batch_size, -1).gather(1, flat_index)
        gradient_heading = torch.atan2(gy, gx) + pi / 2.0
        wind_heading = state.wind_direction_rad[:, None] + pi / 2.0
        heading = torch.where(
            gx.square() + gy.square() > 1.0e-5,
            gradient_heading,
            wind_heading,
        )
        uncertainty = state.belief_uncertainty.reshape(self.batch_size, -1).gather(1, indices)
        priority = values / values.mean(dim=1, keepdim=True).clamp_min(1.0e-5)
        required_l = (
            1800.0
            + 6200.0 * state.belief_burning.reshape(self.batch_size, -1).gather(1, indices)
            + 2200.0 * local_value.reshape(self.batch_size, -1).gather(1, indices)
        )
        median_payload = self.resource_payload_l.median().clamp_min(1.0)
        capacity = torch.ceil(required_l / median_payload).clamp(1, self.num_resources)
        return FrontSegments(
            xy=xy,
            heading_rad=heading,
            priority=priority,
            uncertainty=uncertainty,
            required_l=required_l,
            capacity=capacity,
            valid=valid,
        )

    def _service_commitments(self, state: TensorIncidentState) -> Tensor:
        active = (
            (state.resource_status == STATUS_TO_SERVICE)
            | (state.resource_status == STATUS_QUEUED)
            | (state.resource_status == STATUS_SERVICING)
        ) & (state.resource_target_site >= 0)
        one_hot = F.one_hot(
            state.resource_target_site.clamp(0, self.num_sites - 1),
            num_classes=self.num_sites,
        )
        return (one_hot * active[..., None]).sum(dim=1)

    def _action_mask(
        self,
        state: TensorIncidentState,
        segments: FrontSegments,
    ) -> Tensor:
        batch, resources = self.batch_size, self.num_resources
        available = state.resource_status == STATUS_AVAILABLE
        mask = torch.zeros(
            (batch, resources, self.num_tasks),
            device=self.device,
            dtype=torch.bool,
        )
        mask[..., 0] = True

        segment_destination = segments.xy[:, None].expand(-1, resources, -1, -1)
        segment_origin = state.resource_xy[:, :, None].expand(-1, -1, self.max_segments, -1)
        segment_travel = self._distance_minutes(segment_origin, segment_destination)
        delta = segment_destination[:, :, :, None] - self.site_xy[None, None, None]
        physical = torch.sqrt(
            (delta[..., 0] * self.config.width * self.config.cell_size_m / self.grid_size).square()
            + (delta[..., 1] * self.config.height * self.config.cell_size_m / self.grid_size).square()
        )
        recovery = physical / self.resource_speed_m_min[None, :, None, None].clamp_min(1.0)
        recovery_valid = (self.resource_site_compatible & self.site_refuels[None])[None, :, None]
        recovery = torch.where(recovery_valid, recovery, torch.full_like(recovery, torch.inf))
        recovery = recovery.amin(dim=-1)
        segment_safe = (
            segment_travel + self.resource_dispatch_latency_min[None, :, None] + recovery
            <= state.resource_endurance_remaining_min[:, :, None] - self.resource_reserve_min[None, :, None]
        )
        payload_ready = (
            state.resource_payload_fraction >= self.config.suppression.minimum_dispatch_payload_fraction
        )
        wind_safe = state.wind_speed_m_s[:, None] <= self.resource_max_wind_m_s[None]
        mask[..., 1 : 1 + self.max_segments] = (
            available[:, :, None]
            & payload_ready[:, :, None]
            & wind_safe[:, :, None]
            & segment_safe
            & segments.valid[:, None]
            & (~state.done[:, None, None])
        )

        site_destination = self.site_xy[None, None].expand(batch, resources, -1, -1)
        site_origin = state.resource_xy[:, :, None].expand(-1, -1, self.num_sites, -1)
        site_travel = self._distance_minutes(site_origin, site_destination)
        arrival = state.minute[:, None, None] + torch.ceil(
            site_travel + self.resource_dispatch_latency_min[None, :, None]
        )
        site_open = (arrival >= self.site_open_min[None, None]) & (arrival < self.site_close_min[None, None])
        payload_needed = state.resource_payload_fraction < 1.0 - 1.0e-6
        payload_possible = self.resource_site_payload_compatible[None] & (
            state.site_remaining_l[:, None] > 1.0e-3
        )
        fuel_needed = state.resource_endurance_remaining_min < self.resource_endurance_min[None] - 1.0e-6
        useful = payload_needed[:, :, None] & payload_possible
        useful |= fuel_needed[:, :, None] & self.site_refuels[None, None]

        selected_slots = state.site_slot_available_min[:, None].expand(-1, resources, -1, -1)
        first_slot = selected_slots.amin(dim=-1)
        valid_slots = self.site_slot_valid[None, None]
        last_slot = selected_slots.masked_fill(~valid_slots, -torch.inf).amax(dim=-1)
        extra_service_waves = torch.ceil(self.site_approach_capacity / self.site_bays).clamp_min(1) - 1
        conservative_airborne_start = last_slot + (
            extra_service_waves[None, None] * self.site_max_service_duration_min[None, None]
        )
        airborne = self.site_mode != self._MODE_IDS["land"]
        service_start = torch.maximum(
            arrival,
            torch.where(
                airborne[None, None],
                conservative_airborne_start,
                first_slot,
            ),
        )
        missing_l = self.resource_payload_l[None, :, None] * (
            1.0 - state.resource_payload_fraction[:, :, None]
        )
        service_l = torch.minimum(missing_l, state.site_remaining_l[:, None])
        duration = torch.ceil(
            self.site_turnaround_min[None, None] + service_l / self.site_rate_l_min[None, None]
        ).clamp_min(1.0)
        airborne_service = torch.where(
            airborne[None, None],
            service_start - arrival + duration,
            torch.zeros_like(duration),
        )

        site_to_site = self.site_xy[:, None] - self.site_xy[None]
        site_physical = torch.sqrt(
            (site_to_site[..., 0] * self.config.width * self.config.cell_size_m / self.grid_size).square()
            + (site_to_site[..., 1] * self.config.height * self.config.cell_size_m / self.grid_size).square()
        )
        continuation = site_physical[None, None] / self.resource_speed_m_min[None, :, None, None].clamp_min(
            1.0
        )
        continuation_valid = (self.resource_site_compatible & self.site_refuels[None])[None, :, None]
        nearest_refuel = torch.where(
            continuation_valid,
            continuation,
            torch.full_like(continuation, torch.inf),
        ).amin(dim=-1)
        recovery = torch.where(
            self.site_refuels[None, None],
            torch.zeros_like(nearest_refuel),
            nearest_refuel,
        )
        site_safe = (
            site_travel + self.resource_dispatch_latency_min[None, :, None] + airborne_service + recovery
            <= state.resource_endurance_remaining_min[:, :, None] - self.resource_reserve_min[None, :, None]
        )
        remaining_approach = torch.clamp(
            self.site_approach_capacity[None] - self._service_commitments(state),
            min=0,
        )
        mask[..., 1 + self.max_segments :] = (
            available[:, :, None]
            & self.resource_site_compatible[None]
            & useful
            & site_open
            & (state.wind_speed_m_s[:, None, None] <= self.site_max_wind_m_s[None, None])
            & site_safe
            & (remaining_approach[:, None] > 0)
            & (~state.done[:, None, None])
        )
        return mask

    def _observation_tensors(
        self,
        state: TensorIncidentState,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        segments = self._front_segments(state)
        mask = self._action_mask(state, segments)
        batch = self.batch_size
        tasks = torch.zeros(
            (batch, self.num_tasks, TASK_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        valid = torch.zeros((batch, self.num_tasks), device=self.device, dtype=torch.bool)
        tasks[:, 0, 0] = 1.0
        tasks[:, 0, 7] = 1.0
        valid[:, 0] = True

        segment_slice = slice(1, 1 + self.max_segments)
        tasks[:, segment_slice, 0] = segments.valid.to(self.dtype)
        tasks[:, segment_slice, 1] = 1.0
        tasks[:, segment_slice, 2] = segments.xy[..., 0] / max(self.grid_size - 1, 1)
        tasks[:, segment_slice, 3] = segments.xy[..., 1] / max(self.grid_size - 1, 1)
        tasks[:, segment_slice, 4] = torch.clamp(segments.priority / 12.0, max=1.0)
        tasks[:, segment_slice, 5] = torch.clamp(segments.uncertainty / 1.5, max=1.0)
        tasks[:, segment_slice, 7] = torch.clamp(
            segments.capacity / TASK_CAPACITY_SCALE,
            max=1.0,
        )
        tasks[:, segment_slice, 10] = 1.0
        tasks[:, segment_slice, 11] = torch.sin(segments.heading_rad)
        tasks[:, segment_slice, 12] = torch.cos(segments.heading_rad)
        tasks[:, segment_slice, 15] = 1.0
        valid[:, segment_slice] = segments.valid

        site_slice = slice(1 + self.max_segments, self.num_tasks)
        tasks[:, site_slice, 0] = 1.0
        tasks[:, site_slice, 1] = 6.0 / 7.0
        tasks[:, site_slice, 2] = self.site_xy[None, :, 0] / max(self.grid_size - 1, 1)
        tasks[:, site_slice, 3] = self.site_xy[None, :, 1] / max(self.grid_size - 1, 1)
        stock_fraction = state.site_remaining_l / self.site_initial_l[None].clamp_min(1.0)
        wait_min = (state.site_slot_available_min.amin(dim=-1) - state.minute[:, None]).clamp_min(0.0)
        queue_pressure = wait_min / self.site_turnaround_min[None].clamp_min(1.0)
        tasks[:, site_slice, 4] = (1.0 + 0.25 * stock_fraction - 0.20 * queue_pressure) / 12.0
        remaining_approach = torch.clamp(
            self.site_approach_capacity[None] - self._service_commitments(state),
            min=0,
        )
        tasks[:, site_slice, 7] = torch.clamp(
            remaining_approach / TASK_CAPACITY_SCALE,
            max=1.0,
        )
        tasks[:, site_slice, 13] = 1.0
        tasks[:, site_slice, 14] = 1.0
        tasks[:, site_slice, 16] = self.site_payload_service[None, :, 1].to(self.dtype)
        tasks[:, site_slice, 17] = self.site_payload_service[None, :, 0].to(self.dtype)
        tasks[:, site_slice, 18] = self.site_refuels[None].to(self.dtype)
        tasks[:, site_slice, 19] = torch.clamp(
            self.site_rate_l_min[None] / 20_000.0,
            max=1.0,
        )
        tasks[:, site_slice, 20] = torch.clamp(queue_pressure / 4.0, max=1.0)
        valid[:, site_slice] = True

        resource = torch.zeros(
            (batch, self.num_resources, RESOURCE_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        resource[..., 0] = state.resource_xy[..., 0] / max(self.grid_size - 1, 1)
        resource[..., 1] = state.resource_xy[..., 1] / max(self.grid_size - 1, 1)
        resource[..., 2] = self.resource_kind[None] / 3.0
        resource[..., 3] = state.resource_payload_fraction
        resource[..., 4] = state.resource_status / float(STATUS_WITHDRAWN)
        resource[..., 5] = (state.resource_event_min - state.minute[:, None]).clamp(
            0.0, self.config.horizon_min
        ) / max(self.config.horizon_min, 1)
        resource[..., 6] = (
            self.resource_endurance_min[None] - state.resource_endurance_remaining_min
        ) / self.resource_endurance_min[None].clamp_min(1.0)
        resource[..., 7] = self.resource_speed_m_min[None] / 4800.0
        resource[..., 8] = self.resource_payload_l[None] / 12_000.0
        resource[..., 9] = state.resource_service_cycles / 20.0
        resource[..., 11] = state.minute[:, None] / max(self.config.horizon_min, 1)
        resource[..., 12] = state.resource_accepted_tasks / torch.clamp(
            state.resource_attempted_tasks,
            min=1,
        )
        resource[..., 13] = state.resource_endurance_remaining_min / self.resource_endurance_min[None]
        resource[..., 14] = (state.resource_target_site >= 0).to(self.dtype)
        resource[..., 15] = (state.resource_service_start_min - state.resource_arrival_min).clamp(
            0.0, self.config.horizon_min
        ) / max(self.config.horizon_min, 1)
        resource[..., 16] = (
            (state.resource_status == STATUS_TO_SERVICE)
            | (state.resource_status == STATUS_QUEUED)
            | (state.resource_status == STATUS_SERVICING)
        ).to(self.dtype)

        actor = torch.zeros(
            (batch, ACTOR_GLOBAL_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        burnable = (~state.barrier).to(self.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        actor[:, 0] = state.belief_burning.sum(dim=(1, 2)) / burnable
        actor[:, 1] = state.belief_burned.sum(dim=(1, 2)) / burnable
        actor[:, 2] = (state.belief_burning * state.asset_value).sum(dim=(1, 2)) / state.asset_value.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        actor[:, 3] = state.belief_uncertainty.mean(dim=(1, 2))
        actor[:, 4] = state.water_line_strength.mean(dim=(1, 2))
        actor[:, 5] = state.retardant_line_strength.mean(dim=(1, 2))
        actor[:, 6] = available_fraction = (
            (state.resource_status == STATUS_AVAILABLE).to(self.dtype).mean(dim=1)
        )
        actor[:, 7] = state.wind_speed_m_s / max(
            self.config.suppression.aviation_max_wind_m_s,
            1.0,
        )
        actor[:, 8] = state.minute / max(self.config.horizon_min, 1)
        actor[:, 9] = 1.0 - (state.resource_endurance_remaining_min / self.resource_endurance_min[None]).mean(
            dim=1
        )

        critic = torch.zeros(
            (batch, CRITIC_GLOBAL_FEATURE_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        value_sum = state.asset_value.sum(dim=(1, 2)).clamp_min(1.0)
        critic[:, 0] = state.burning.sum(dim=(1, 2)) / burnable
        critic[:, 1] = state.burned.sum(dim=(1, 2)) / burnable
        critic[:, 2] = ((state.burned + 0.5 * state.burning) * state.asset_value).sum(dim=(1, 2)) / value_sum
        critic[:, 3] = state.resource_payload_fraction.mean(dim=1)
        critic[:, 4] = (state.resource_endurance_remaining_min / self.resource_endurance_min[None]).mean(
            dim=1
        )
        critic[:, 5] = available_fraction
        critic[:, 6] = (state.resource_status == STATUS_QUEUED).to(self.dtype).mean(dim=1)
        critic[:, 7] = (state.resource_status == STATUS_SERVICING).to(self.dtype).mean(dim=1)
        critic[:, 8] = state.belief_uncertainty.mean(dim=(1, 2))
        critic[:, 9] = stock_fraction.mean(dim=1)
        critic[:, 10] = state.minute / max(self.config.horizon_min, 1)
        critic[:, 11] = (state.spread_rate_scale * state.wind_coefficient / 0.14).clamp(max=1.5)
        return resource, tasks, mask, valid, actor, critic

    def _outcome_metrics(self, state: TensorIncidentState) -> tuple[Tensor, Tensor, Tensor]:
        value_sum = state.asset_value.sum(dim=(1, 2)).clamp_min(1.0)
        expected_loss = ((state.burned + 0.5 * state.burning) * state.asset_value).sum(dim=(1, 2)) / value_sum
        burnable = (~state.barrier).to(self.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        burned_fraction = state.burned.sum(dim=(1, 2)) / burnable
        treatment = (
            1.0
            - state.water_line_reduction[:, None, None] * state.water_line_strength
            - state.retardant_line_reduction[:, None, None] * state.retardant_line_strength
        ).clamp(0.02, 1.0)
        active_risk = (state.burning * state.asset_value * state.fuel_factor * treatment).sum(
            dim=(1, 2)
        ) / value_sum
        return expected_loss, burned_fraction, active_risk

    def _apply_drops(
        self,
        state: TensorIncidentState,
        completed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        water_coverage = state.water_coverage_gpc
        retardant_coverage = state.retardant_coverage_gpc
        water_line = state.water_line_strength
        retardant_line = state.retardant_line_strength
        delivered = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        wasted = torch.zeros_like(delivered)
        front = state.burning * F.max_pool2d(state.unburned[:, None], 3, stride=1, padding=1)[:, 0]
        risk_scale = front * (0.25 + state.asset_value + 0.5 * state.fuel_factor)
        risk_scale = risk_scale / risk_scale.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        for resource_index in range(self.num_resources):
            active = completed[:, resource_index]
            volume = (
                self.resource_payload_l[resource_index]
                * state.resource_payload_fraction[:, resource_index]
                * active.to(self.dtype)
            )
            target = state.resource_target_xy[:, resource_index]
            heading = state.resource_target_heading_rad[:, resource_index]
            dx = self.grid_x[None] - target[:, None, None, 0]
            dy = self.grid_y[None] - target[:, None, None, 1]
            along = dx * torch.cos(heading)[:, None, None] + dy * torch.sin(heading)[:, None, None]
            cross = -dx * torch.sin(heading)[:, None, None] + dy * torch.cos(heading)[:, None, None]
            half_length = 0.5 * self.resource_drop_length_cells[resource_index]
            half_width = 0.5 * self.resource_drop_width_cells[resource_index]
            kernel = torch.exp(
                -0.5 * (along / half_length.clamp_min(0.25)).square()
                - 0.5 * (cross / half_width.clamp_min(0.20)).square()
            )
            kernel = kernel * (~state.barrier).to(self.dtype)
            normalized = kernel / kernel.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-9)
            coverage_increment = volume[:, None, None] * normalized / (self.cell_area_m2 * GPC_L_M2)
            local_line = (kernel / kernel.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-9)) * active[
                :, None, None
            ].to(self.dtype)
            local_line = 1.0 - torch.exp(
                -local_line * self.resource_target_coverage_gpc[resource_index] / 2.0
            )
            if self.resource_kind_names[resource_index] == "water":
                water_coverage = water_coverage + coverage_increment
                water_line = 1.0 - (1.0 - water_line) * (1.0 - local_line)
            else:
                retardant_coverage = retardant_coverage + coverage_increment
                retardant_line = 1.0 - (1.0 - retardant_line) * (1.0 - local_line)
            useful_fraction = (normalized * risk_scale).sum(dim=(1, 2)).clamp(0.0, 1.0)
            delivered = delivered + volume
            wasted = wasted + volume * (1.0 - useful_fraction)
        return (
            water_coverage,
            retardant_coverage,
            water_line,
            retardant_line,
            delivered,
            wasted,
        )

    def _task_capacities(
        self,
        state: TensorIncidentState,
        segments: FrontSegments,
    ) -> Tensor:
        capacity = torch.ones(
            (self.batch_size, self.num_tasks),
            device=self.device,
            dtype=torch.long,
        )
        capacity[:, 0] = self.num_resources
        capacity[:, 1 : 1 + self.max_segments] = segments.capacity.long()
        remaining = torch.clamp(
            self.site_approach_capacity[None] - self._service_commitments(state),
            min=0,
        )
        capacity[:, 1 + self.max_segments :] = remaining
        return capacity

    def _schedule_service(
        self,
        state: TensorIncidentState,
        service_assignments: Tensor,
        selected_site: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        status = state.resource_status
        arrival_min = state.resource_arrival_min
        event_min = state.resource_event_min
        service_start_min = state.resource_service_start_min
        leg_start_min = state.resource_leg_start_min
        leg_start_xy = state.resource_leg_start_xy
        leg_end_xy = state.resource_leg_end_xy
        target_site = state.resource_target_site
        target_xy = state.resource_target_xy
        reserved_load = state.resource_reserved_load_l
        endurance = state.resource_endurance_remaining_min
        site_stock = state.site_remaining_l
        slot_available = state.site_slot_available_min
        flight_cost = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        queue_cost = torch.zeros_like(flight_cost)

        for resource_index in range(self.num_resources):
            assign = service_assignments[:, resource_index]
            site_index = selected_site[:, resource_index]
            destination = self.site_xy[site_index]
            travel = self._distance_minutes(
                state.resource_xy[:, resource_index],
                destination,
                resource_index=resource_index,
            )
            arrival = state.minute + torch.ceil(
                travel + self.resource_dispatch_latency_min[resource_index]
            ).clamp_min(1.0)
            site_slots = slot_available.gather(
                1,
                site_index[:, None, None].expand(-1, 1, self.max_bays),
            )[:, 0]
            slot_index = site_slots.argmin(dim=1)
            earliest = site_slots.gather(1, slot_index[:, None])[:, 0]
            service_start = torch.maximum(arrival, earliest)
            missing_l = self.resource_payload_l[resource_index] * (
                1.0 - state.resource_payload_fraction[:, resource_index]
            )
            stock = site_stock.gather(1, site_index[:, None])[:, 0]
            payload_supported = self.resource_site_payload_compatible[resource_index, site_index]
            desired_l = torch.where(payload_supported, missing_l, torch.zeros_like(missing_l))
            load_l = torch.minimum(desired_l, stock)
            duration = torch.ceil(
                self.site_turnaround_min[site_index] + load_l / self.site_rate_l_min[site_index]
            ).clamp_min(1.0)
            completion = service_start + duration
            flat_index = site_index * self.max_bays + slot_index
            flat_slots = slot_available.reshape(self.batch_size, -1)
            previous_slot = flat_slots.gather(1, flat_index[:, None])[:, 0]
            flat_slots = flat_slots.scatter(
                1,
                flat_index[:, None],
                torch.where(assign, completion, previous_slot)[:, None],
            )
            slot_available = flat_slots.reshape(self.batch_size, self.num_sites, self.max_bays)
            stock_delta = torch.where(assign, -load_l, torch.zeros_like(load_l))
            site_stock = site_stock.scatter_add(1, site_index[:, None], stock_delta[:, None])

            status[:, resource_index] = torch.where(
                assign,
                torch.full_like(status[:, resource_index], STATUS_TO_SERVICE),
                status[:, resource_index],
            )
            arrival_min[:, resource_index] = torch.where(assign, arrival, arrival_min[:, resource_index])
            event_min[:, resource_index] = torch.where(assign, completion, event_min[:, resource_index])
            service_start_min[:, resource_index] = torch.where(
                assign, service_start, service_start_min[:, resource_index]
            )
            leg_start_min[:, resource_index] = torch.where(
                assign, state.minute, leg_start_min[:, resource_index]
            )
            leg_start_xy[:, resource_index] = torch.where(
                assign[:, None],
                state.resource_xy[:, resource_index],
                leg_start_xy[:, resource_index],
            )
            leg_end_xy[:, resource_index] = torch.where(
                assign[:, None], destination, leg_end_xy[:, resource_index]
            )
            target_xy[:, resource_index] = torch.where(
                assign[:, None], destination, target_xy[:, resource_index]
            )
            target_site[:, resource_index] = torch.where(assign, site_index, target_site[:, resource_index])
            reserved_load[:, resource_index] = torch.where(assign, load_l, reserved_load[:, resource_index])
            airborne = self.site_mode[site_index] != self._MODE_IDS["land"]
            airborne_min = (
                travel
                + self.resource_dispatch_latency_min[resource_index]
                + torch.where(
                    airborne,
                    service_start - arrival + duration,
                    torch.zeros_like(duration),
                )
            )
            endurance[:, resource_index] = torch.where(
                assign,
                endurance[:, resource_index] - airborne_min,
                endurance[:, resource_index],
            )
            flight_cost = flight_cost + torch.where(assign, airborne_min, torch.zeros_like(airborne_min))
            queue_cost = queue_cost + torch.where(
                assign,
                (service_start - arrival).clamp_min(0.0),
                torch.zeros_like(service_start),
            )
        return (
            status,
            arrival_min,
            event_min,
            service_start_min,
            leg_start_min,
            leg_start_xy,
            leg_end_xy,
            target_site,
            target_xy,
            reserved_load,
            endurance,
            site_stock,
            slot_available,
            flight_cost,
            queue_cost,
        )

    def _transition(self, state: TensorIncidentState, actions: Tensor) -> TensorIncidentTransition:
        segments = self._front_segments(state)
        mask = self._action_mask(state, segments)
        in_range = (actions >= 0) & (actions < self.num_tasks)
        safe = torch.where(in_range, actions, torch.zeros_like(actions))
        valid = mask.gather(2, safe[..., None])[..., 0]
        one_hot = F.one_hot(safe, num_classes=self.num_tasks)
        rank = one_hot.cumsum(dim=1).gather(2, safe[..., None])[..., 0]
        capacity = self._task_capacities(state, segments).gather(1, safe)
        valid &= rank <= capacity
        blocked = (~valid) & (safe != 0)
        safe = torch.where(valid, safe, torch.zeros_like(safe))
        assigned = safe != 0
        attack = (safe >= 1) & (safe <= self.max_segments)
        service = safe > self.max_segments
        segment_index = (safe - 1).clamp(0, self.max_segments - 1)
        site_index = (safe - 1 - self.max_segments).clamp(0, self.num_sites - 1)

        status = state.resource_status.clone()
        arrival_min = state.resource_arrival_min.clone()
        event_min = state.resource_event_min.clone()
        service_start_min = state.resource_service_start_min.clone()
        leg_start_min = state.resource_leg_start_min.clone()
        leg_start_xy = state.resource_leg_start_xy.clone()
        leg_end_xy = state.resource_leg_end_xy.clone()
        target_site = state.resource_target_site.clone()
        target_xy = state.resource_target_xy.clone()
        target_heading = state.resource_target_heading_rad.clone()
        reserved_load = state.resource_reserved_load_l.clone()
        endurance = state.resource_endurance_remaining_min.clone()

        selected_segment_xy = segments.xy.gather(1, segment_index[..., None].expand(-1, -1, 2))
        selected_heading = segments.heading_rad.gather(1, segment_index)
        attack_travel = self._distance_minutes(state.resource_xy, selected_segment_xy)
        attack_arrival = state.minute[:, None] + torch.ceil(
            attack_travel + self.resource_dispatch_latency_min[None]
        ).clamp_min(1.0)
        status = torch.where(
            attack,
            torch.full_like(status, STATUS_TO_ATTACK),
            status,
        )
        arrival_min = torch.where(attack, attack_arrival, arrival_min)
        event_min = torch.where(attack, attack_arrival, event_min)
        service_start_min = torch.where(attack, torch.full_like(service_start_min, 1.0e9), service_start_min)
        leg_start_min = torch.where(attack, state.minute[:, None], leg_start_min)
        leg_start_xy = torch.where(attack[..., None], state.resource_xy, leg_start_xy)
        leg_end_xy = torch.where(attack[..., None], selected_segment_xy, leg_end_xy)
        target_xy = torch.where(attack[..., None], selected_segment_xy, target_xy)
        target_heading = torch.where(attack, selected_heading, target_heading)
        target_site = torch.where(attack, torch.full_like(target_site, -1), target_site)
        attack_airborne = attack_travel + self.resource_dispatch_latency_min[None]
        endurance = torch.where(attack, endurance - attack_airborne, endurance)
        flight_cost = torch.where(
            attack,
            attack_airborne,
            torch.zeros_like(attack_airborne),
        ).sum(dim=1)

        service_state = state._replace(
            resource_status=status,
            resource_arrival_min=arrival_min,
            resource_event_min=event_min,
            resource_service_start_min=service_start_min,
            resource_leg_start_min=leg_start_min,
            resource_leg_start_xy=leg_start_xy,
            resource_leg_end_xy=leg_end_xy,
            resource_endurance_remaining_min=endurance,
            resource_target_site=target_site,
            resource_target_xy=target_xy,
            resource_target_heading_rad=target_heading,
            resource_reserved_load_l=reserved_load,
        )
        (
            status,
            arrival_min,
            event_min,
            service_start_min,
            leg_start_min,
            leg_start_xy,
            leg_end_xy,
            target_site,
            target_xy,
            reserved_load,
            endurance,
            site_stock,
            slot_available,
            service_flight_cost,
            queue_cost,
        ) = self._schedule_service(service_state, service, site_index)
        flight_cost = flight_cost + service_flight_cost

        water_decay = 0.5 ** (
            self.config.decision_interval_min / max(self.config.suppression.water_half_life_min, 1.0e-6)
        )
        retardant_decay = 0.5 ** (
            self.config.decision_interval_min / max(self.config.suppression.retardant_half_life_min, 1.0e-6)
        )
        water_coverage = state.water_coverage_gpc * water_decay
        retardant_coverage = state.retardant_coverage_gpc * retardant_decay
        water_line = state.water_line_strength * water_decay
        retardant_line = state.retardant_line_strength * retardant_decay

        before_loss, before_burned, before_risk = self._outcome_metrics(state)
        unburned, burning, burned = self._fire_transition_fields(
            state.unburned,
            state.burning,
            state.burned,
            fuel_factor=state.fuel_factor,
            slope_x=state.slope_x,
            slope_y=state.slope_y,
            barrier=state.barrier,
            wind_speed_m_s=state.wind_speed_m_s,
            wind_direction_rad=state.wind_direction_rad,
            water_coverage_gpc=water_coverage,
            retardant_coverage_gpc=retardant_coverage,
            water_line_strength=water_line,
            retardant_line_strength=retardant_line,
            spread_rate_scale=state.spread_rate_scale,
            wind_coefficient=state.wind_coefficient,
            slope_coefficient=state.slope_coefficient,
            burn_duration_min=state.burn_duration_min,
            water_line_reduction=state.water_line_reduction,
            retardant_line_reduction=state.retardant_line_reduction,
        )
        new_minute = state.minute + (~state.done).to(self.dtype) * self.config.decision_interval_min

        flight_status = (status == STATUS_TO_ATTACK) | (status == STATUS_TO_SERVICE)
        progress = (
            (new_minute[:, None] - leg_start_min) / (arrival_min - leg_start_min).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        resource_xy = torch.where(
            flight_status[..., None],
            leg_start_xy + (leg_end_xy - leg_start_xy) * progress[..., None],
            state.resource_xy,
        )
        arrived_service = (status == STATUS_TO_SERVICE) & (arrival_min <= new_minute[:, None])
        queued = arrived_service & (service_start_min > new_minute[:, None])
        servicing = arrived_service & (service_start_min <= new_minute[:, None])
        status = torch.where(
            queued,
            torch.full_like(status, STATUS_QUEUED),
            status,
        )
        status = torch.where(
            servicing,
            torch.full_like(status, STATUS_SERVICING),
            status,
        )
        completed_attack = (status == STATUS_TO_ATTACK) & (event_min <= new_minute[:, None])
        drop_state = state._replace(
            minute=new_minute,
            unburned=unburned,
            burning=burning,
            burned=burned,
            water_coverage_gpc=water_coverage,
            retardant_coverage_gpc=retardant_coverage,
            water_line_strength=water_line,
            retardant_line_strength=retardant_line,
            resource_xy=resource_xy,
            resource_status=status,
            resource_arrival_min=arrival_min,
            resource_event_min=event_min,
            resource_service_start_min=service_start_min,
            resource_leg_start_min=leg_start_min,
            resource_leg_start_xy=leg_start_xy,
            resource_leg_end_xy=leg_end_xy,
            resource_endurance_remaining_min=endurance,
            resource_target_site=target_site,
            resource_target_xy=target_xy,
            resource_target_heading_rad=target_heading,
            resource_reserved_load_l=reserved_load,
            site_remaining_l=site_stock,
            site_slot_available_min=slot_available,
        )
        (
            water_coverage,
            retardant_coverage,
            water_line,
            retardant_line,
            delivered,
            wasted,
        ) = self._apply_drops(drop_state, completed_attack)
        payload = torch.where(
            completed_attack,
            torch.zeros_like(state.resource_payload_fraction),
            state.resource_payload_fraction,
        )
        status = torch.where(
            completed_attack,
            torch.full_like(status, STATUS_AVAILABLE),
            status,
        )
        event_min = torch.where(completed_attack, torch.full_like(event_min, 1.0e9), event_min)

        service_active = (
            (status == STATUS_TO_SERVICE) | (status == STATUS_QUEUED) | (status == STATUS_SERVICING)
        )
        service_complete = service_active & (event_min <= new_minute[:, None])
        payload = torch.where(
            service_complete,
            (payload + reserved_load / self.resource_payload_l[None].clamp_min(1.0)).clamp(max=1.0),
            payload,
        )
        completed_site = target_site.clamp(0, self.num_sites - 1)
        resets_endurance = self.site_refuels[completed_site]
        endurance = torch.where(
            service_complete & resets_endurance,
            self.resource_endurance_min[None],
            endurance,
        )
        status = torch.where(
            service_complete,
            torch.full_like(status, STATUS_AVAILABLE),
            status,
        )
        event_min = torch.where(service_complete, torch.full_like(event_min, 1.0e9), event_min)
        reserved_load = torch.where(service_complete, torch.zeros_like(reserved_load), reserved_load)
        service_cycles = state.resource_service_cycles + service_complete.long()
        target_site = torch.where(service_complete, completed_site, target_site)
        exhausted = endurance <= 0.0
        status = torch.where(
            exhausted,
            torch.full_like(status, STATUS_WITHDRAWN),
            status,
        )

        belief_unburned, belief_burning, belief_burned = self._fire_transition_fields(
            state.belief_unburned,
            state.belief_burning,
            state.belief_burned,
            fuel_factor=state.fuel_factor,
            slope_x=state.slope_x,
            slope_y=state.slope_y,
            barrier=state.barrier,
            wind_speed_m_s=state.wind_speed_m_s,
            wind_direction_rad=state.wind_direction_rad,
            water_coverage_gpc=water_coverage,
            retardant_coverage_gpc=retardant_coverage,
            water_line_strength=water_line,
            retardant_line_strength=retardant_line,
            spread_rate_scale=state.spread_rate_scale,
            wind_coefficient=state.wind_coefficient,
            slope_coefficient=state.slope_coefficient,
            burn_duration_min=state.burn_duration_min,
            water_line_reduction=state.water_line_reduction,
            retardant_line_reduction=state.retardant_line_reduction,
        )
        observe_due = torch.remainder(
            new_minute,
            float(self.observation_period_min),
        ) < float(self.config.decision_interval_min)
        observed_burning = F.avg_pool2d(burning[:, None], 3, stride=1, padding=1)[:, 0]
        observed_burned = F.avg_pool2d(burned[:, None], 3, stride=1, padding=1)[:, 0]
        observed_unburned = ((~state.barrier).to(self.dtype) - observed_burning - observed_burned).clamp(
            0.0, 1.0
        )
        observation_weight = state.observation_weight[:, None, None]
        belief_burning = torch.where(
            observe_due[:, None, None],
            observation_weight * observed_burning + (1.0 - observation_weight) * belief_burning,
            belief_burning,
        )
        belief_burned = torch.where(
            observe_due[:, None, None],
            observation_weight * observed_burned + (1.0 - observation_weight) * belief_burned,
            belief_burned,
        )
        belief_unburned = torch.where(
            observe_due[:, None, None],
            observation_weight * observed_unburned + (1.0 - observation_weight) * belief_unburned,
            belief_unburned,
        )
        belief_total = (belief_unburned + belief_burning + belief_burned).clamp_min(1.0e-6)
        belief_unburned = belief_unburned / belief_total
        belief_burning = belief_burning / belief_total
        belief_burned = belief_burned / belief_total
        uncertainty = torch.where(
            observe_due[:, None, None],
            torch.full_like(state.belief_uncertainty, 0.10),
            (
                state.belief_uncertainty
                + 0.08 * self.config.decision_interval_min / self.observation_period_min
            ).clamp(max=1.0),
        )

        active_fire = burning.sum(dim=(1, 2))
        contained = active_fire < 0.20
        border_fire = torch.maximum(
            burning[:, 1].amax(dim=1),
            burning[:, -2].amax(dim=1),
        )
        border_fire = torch.maximum(
            border_fire,
            torch.maximum(burning[:, :, 1].amax(dim=1), burning[:, :, -2].amax(dim=1)),
        )
        escaped = state.escaped | (border_fire > 0.20)
        done = state.done | (new_minute >= self.config.horizon_min)
        if self.terminate_on_escape:
            done |= escaped
        if self.terminate_on_completion:
            done |= contained

        attempted = state.resource_attempted_tasks + (actions != 0).long()
        accepted = state.resource_accepted_tasks + assigned.long()
        new_state = TensorIncidentState(
            minute=new_minute,
            unburned=unburned,
            burning=burning,
            burned=burned,
            belief_unburned=belief_unburned,
            belief_burning=belief_burning,
            belief_burned=belief_burned,
            belief_uncertainty=uncertainty,
            fuel_factor=state.fuel_factor,
            asset_value=state.asset_value,
            slope_x=state.slope_x,
            slope_y=state.slope_y,
            barrier=state.barrier,
            water_coverage_gpc=water_coverage,
            retardant_coverage_gpc=retardant_coverage,
            water_line_strength=water_line,
            retardant_line_strength=retardant_line,
            wind_speed_m_s=state.wind_speed_m_s,
            wind_direction_rad=state.wind_direction_rad,
            spread_rate_scale=state.spread_rate_scale,
            wind_coefficient=state.wind_coefficient,
            slope_coefficient=state.slope_coefficient,
            burn_duration_min=state.burn_duration_min,
            water_line_reduction=state.water_line_reduction,
            retardant_line_reduction=state.retardant_line_reduction,
            observation_weight=state.observation_weight,
            resource_xy=resource_xy,
            resource_status=status,
            resource_arrival_min=arrival_min,
            resource_event_min=event_min,
            resource_service_start_min=service_start_min,
            resource_leg_start_min=leg_start_min,
            resource_leg_start_xy=leg_start_xy,
            resource_leg_end_xy=leg_end_xy,
            resource_payload_fraction=payload,
            resource_endurance_remaining_min=endurance,
            resource_target_site=target_site,
            resource_target_xy=target_xy,
            resource_target_heading_rad=target_heading,
            resource_reserved_load_l=reserved_load,
            resource_attempted_tasks=attempted,
            resource_accepted_tasks=accepted,
            resource_service_cycles=service_cycles,
            site_remaining_l=site_stock,
            site_slot_available_min=slot_available,
            contained=contained,
            escaped=escaped,
            done=done,
        )
        # Terminal worlds are absorbing. This matters when fixed-horizon
        # collectors retain completed worlds until the next batched reset.
        frozen: list[Tensor] = []
        for old_value, new_value in zip(state, new_state, strict=True):
            selection = state.done.reshape(
                self.batch_size,
                *((1,) * (old_value.ndim - 1)),
            )
            frozen.append(torch.where(selection, old_value, new_value))
        new_state = TensorIncidentState(*frozen)
        after_loss, after_burned, after_risk = self._outcome_metrics(new_state)
        active = (~state.done).to(self.dtype)
        containment_bonus = (new_state.contained & (~state.contained)).to(
            self.dtype
        ) * self.config.containment_bonus
        escape_penalty = (new_state.escaped & (~state.escaped)).to(self.dtype) * self.config.escape_penalty
        reward = (
            -80.0 * (after_loss - before_loss)
            - 8.0 * (after_burned - before_burned)
            + 6.0 * (before_risk - after_risk)
            + containment_bonus
            - escape_penalty
            - 0.0004 * flight_cost
            - 0.0005 * queue_cost
            - 0.04 * blocked.sum(dim=1)
            - 0.10 * wasted / self.resource_payload_l.sum().clamp_min(1.0)
        ) * active
        delivered = delivered * active
        wasted = wasted * active
        blocked_count = blocked.sum(dim=1) * active.long()
        constraint_costs = (
            torch.stack(
                (
                    blocked_count.to(self.dtype),
                    exhausted.to(self.dtype).sum(dim=1) * active,
                    queue_cost / max(self.config.decision_interval_min, 1),
                    wasted / self.resource_payload_l.sum().clamp_min(1.0),
                ),
                dim=1,
            )
            * active[:, None]
        )
        resource, tasks, action_mask, task_valid, actor, critic = self._observation_tensors(new_state)
        return TensorIncidentTransition(
            state=new_state,
            resource=resource,
            tasks=tasks,
            action_mask=action_mask,
            task_valid=task_valid,
            actor_global=actor,
            critic_global=critic,
            reward=reward,
            delivered_l=delivered,
            wasted_l=wasted,
            blocked_actions=blocked_count,
            expected_loss=after_loss,
            burned_fraction=after_burned,
            constraint_costs=constraint_costs,
        )

    def compile(self, *, backend: str | None = None) -> None:
        """Compile the complete fixed-shape transition and reject graph breaks."""

        options = {"fullgraph": True, "dynamic": False}
        if backend is None:
            options["mode"] = "reduce-overhead"
        else:
            options["backend"] = backend
        self._compiled_transition = torch.compile(self._transition, **options)

    def _refresh_observation(self) -> None:
        resource, tasks, mask, valid, actor, critic = self._observation_tensors(self.state)
        self._observation = TensorOperationsObservation(
            resource=resource,
            tasks=tasks,
            action_mask=mask,
            task_valid=valid,
            global_state=actor,
        )
        self._critic = critic

    @torch.no_grad()
    def observations(self) -> TensorOperationsObservation:
        return self._observation

    @torch.no_grad()
    def action_mask(self) -> Tensor:
        return self._observation.action_mask

    @torch.no_grad()
    def critic_state(self) -> Tensor:
        return self._critic

    @torch.no_grad()
    def step(self, actions: Tensor) -> TensorIncidentStep:
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        if tuple(actions.shape) != (self.batch_size, self.num_resources):
            raise ValueError(
                f"actions must have shape {(self.batch_size, self.num_resources)}, "
                f"received {tuple(actions.shape)}"
            )
        transition = (
            self._compiled_transition(self.state, actions)
            if self._compiled_transition is not None
            else self._transition(self.state, actions)
        )
        self.state = transition.state
        self._observation = TensorOperationsObservation(
            resource=transition.resource,
            tasks=transition.tasks,
            action_mask=transition.action_mask,
            task_valid=transition.task_valid,
            global_state=transition.actor_global,
        )
        self._critic = transition.critic_global
        return TensorIncidentStep(
            observation=self._observation,
            reward=transition.reward,
            done=self.state.done.clone(),
            delivered_l=transition.delivered_l,
            wasted_l=transition.wasted_l,
            blocked_actions=transition.blocked_actions,
            expected_loss=transition.expected_loss,
            burned_fraction=transition.burned_fraction,
            constraint_costs=transition.constraint_costs,
        )
