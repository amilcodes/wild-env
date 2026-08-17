"""Truth, belief, resource, and mission state for one Aeolus episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aeolus.config import ResourceSpec, ServiceSiteSpec


class FirePhase(IntEnum):
    UNBURNED = 0
    FLAMING = 1
    BURNED = 2


class FireType(IntEnum):
    UNBURNED = 0
    SURFACE = 1
    PASSIVE_CROWN = 2
    ACTIVE_CROWN = 3


class ResourceStatus(IntEnum):
    AVAILABLE = 0
    OUTBOUND = 1
    RETURNING = 2
    RELOADING = 3
    WITHDRAWN = 4
    WORKING = 5
    QUEUED = 6


@dataclass
class ResourceRuntime:
    spec: ResourceSpec
    x: float
    y: float
    status: ResourceStatus = ResourceStatus.AVAILABLE
    eta_min: int = 0
    target_xy: tuple[int, int] | None = None
    leg_start_xy: tuple[float, float] | None = None
    leg_end_xy: tuple[float, float] | None = None
    leg_total_min: int = 0
    task_index: int = 0
    task_kind: int = 0
    task_heading_deg: float = 0.0
    payload_fraction: float = 1.0
    flight_min: float = 0.0
    reload_cycles: int = 0
    attempted_tasks: int = 0
    accepted_tasks: int = 0
    work_remaining_m: float = 0.0
    line_start_xy: tuple[float, float] | None = None
    line_end_xy: tuple[float, float] | None = None
    line_progress_m: float = 0.0
    production_multiplier: float = 1.0
    queue_entered_min: int | None = None
    current_site_id: str | None = None
    service_site_id: str | None = None
    service_volume_l: float = 0.0

    @property
    def resource_id(self) -> str:
        return self.spec.resource_id

    @property
    def payload_l(self) -> float:
        return self.spec.payload_l * self.payload_fraction

    @property
    def endurance_remaining_min(self) -> float:
        return max(0.0, self.spec.endurance_min - self.flight_min)


@dataclass
class ServiceSiteRuntime:
    spec: ServiceSiteSpec
    remaining_volume_l: float

    @property
    def site_id(self) -> str:
        return self.spec.site_id


@dataclass
class PendingObservation:
    deliver_minute: int
    x: int
    y: int
    radius_cells: int
    source: str
    intensity_measurement: np.ndarray | None = None
    burned_measurement: np.ndarray | None = None


@dataclass
class TruthState:
    phase: np.ndarray
    intensity_kw_m: np.ndarray
    fuel_remaining: np.ndarray
    fuel_load: np.ndarray
    elevation_m: np.ndarray
    barrier: np.ndarray
    asset_value: np.ndarray
    water: np.ndarray
    retardant: np.ndarray
    ground_hold: np.ndarray
    water_coverage_gpc: np.ndarray
    retardant_coverage_gpc: np.ndarray
    retardant_effective_coverage_gpc: np.ndarray
    constructed_line: np.ndarray
    line_strength: np.ndarray
    line_status: np.ndarray
    residual_field: np.ndarray
    observed_burned: np.ndarray
    fuel_model_number: np.ndarray
    moisture_dead_1h: np.ndarray
    moisture_dead_10h: np.ndarray
    moisture_dead_100h: np.ndarray
    moisture_live_herbaceous: np.ndarray
    moisture_live_woody: np.ndarray
    foliar_moisture: np.ndarray
    canopy_cover: np.ndarray
    canopy_height_m: np.ndarray
    canopy_base_height_m: np.ndarray
    canopy_bulk_density_kg_m3: np.ndarray
    fire_type: np.ndarray
    spread_rate_m_min: np.ndarray
    flame_length_m: np.ndarray
    ignition_progress: np.ndarray
    level_set_m: np.ndarray
    arrival_time_min: np.ndarray
    burn_age_min: np.ndarray
    history_speed_m_min: np.ndarray
    history_head_x: np.ndarray
    history_head_y: np.ndarray
    history_confidence: np.ndarray
    history_heat_flux_kw_m2: np.ndarray


@dataclass
class BeliefState:
    intensity_mean: np.ndarray
    intensity_std: np.ndarray
    observed_at: np.ndarray
    known_burned: np.ndarray
    burn_probability: np.ndarray
    arrival_time_mean: np.ndarray
    arrival_time_std: np.ndarray
    perimeter_source: str | None = None
    pending: list[PendingObservation] = field(default_factory=list)


@dataclass
class EpisodeState:
    minute: int
    truth: TruthState
    belief: BeliefState
    resources: list[ResourceRuntime]
    service_sites: list[ServiceSiteRuntime]
    base_xy: tuple[int, int]
    rng: np.random.Generator
    ground_engaged: bool = False
    terminated: bool = False
    truncated: bool = False
    escaped: bool = False
    contained: bool = False
    cumulative_cost: float = 0.0
    cumulative_exposure: float = 0.0
    blocked_actions: int = 0
    events: list[dict[str, object]] = field(default_factory=list)

    def event(self, kind: str, **payload: object) -> None:
        self.events.append({"minute": self.minute, "kind": kind, **payload})

    def copy_rng_state(self) -> dict[str, object]:
        return dict(self.rng.bit_generator.state)
