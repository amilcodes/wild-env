"""Versioned experiment configuration with no hidden global defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FuelModel:
    """Surface-fuel fields in SI units used by the fast kernel.

    Values are deliberately carried in the scenario manifest so future
    fuel-model imports can replace them without changing simulator semantics.
    """

    name: str = "shrub-grass-mix"
    standard_number: int = 122
    fuel_load_kg_m2: float = 1.15
    fuel_depth_m: float = 0.65
    surface_area_to_volume_m_inv: float = 5200.0
    fuel_particle_density_kg_m3: float = 510.0
    heat_content_kj_kg: float = 18600.0
    mineral_damping: float = 0.9
    moisture_of_extinction: float = 0.30
    dead_moisture: float = 0.075


@dataclass(frozen=True)
class FireBehaviorConfig:
    """Numerical and physical controls for the operational-equation kernel."""

    backend: str = "operational"
    front_solver: str = "weno5_level_set"
    moisture_model: str = "equilibrium_time_lag"
    enable_crown_fire: bool = True
    enable_spotting: bool = True
    max_spread_rate_m_min: float = 180.0
    propagation_cfl: float = 0.35
    max_substeps: int = 16
    level_set_band_width_cells: float = 12.0
    level_set_reinitialization_interval_min: int = 15
    min_front_residence_min: float = 20.0
    max_front_residence_min: float = 360.0
    surface_spread_adjustment: float = 1.0
    crown_spread_adjustment: float = 1.0
    wind_speed_adjustment: float = 1.0
    wind_direction_bias_deg: float = 0.0
    dead_fuel_moisture_bias: float = 0.0
    dead_moisture_10h: float = 0.095
    dead_moisture_100h: float = 0.115
    live_herbaceous_moisture: float = 0.75
    live_woody_moisture: float = 0.60
    foliar_moisture: float = 1.00
    spotting_embers_per_source_min: float = 0.025
    spotting_median_distance_m: float = 220.0
    spotting_log_sigma: float = 0.85
    spotting_crosswind_fraction: float = 0.12
    spotting_wind_exponent: float = 1.35
    spotting_intensity_exponent: float = 0.30
    spotting_survival_distance_m: float = 4200.0
    spotting_ignition_probability: float = 0.28
    spotting_max_distance_m: float = 18_000.0
    max_spot_embers_per_minute: int = 4096
    history_correction_half_life_min: float = 180.0
    history_direction_weight: float = 0.45
    history_rate_ratio_min: float = 0.55
    history_rate_ratio_max: float = 3.00

    def __post_init__(self) -> None:
        if self.backend not in {"operational"}:
            raise ValueError("fire backend must be 'operational'")
        if self.front_solver not in {
            "weno5_level_set",
            "godunov_level_set",
            "adaptive_huygens",
        }:
            raise ValueError("unknown fire-front solver")
        if self.moisture_model not in {"equilibrium_time_lag", "fixed"}:
            raise ValueError("unknown fuel-moisture model")
        if not 0 < self.propagation_cfl <= 1:
            raise ValueError("propagation_cfl must be within (0, 1]")
        if self.max_substeps < 1:
            raise ValueError("max_substeps must be positive")
        if self.max_spread_rate_m_min <= 0:
            raise ValueError("max_spread_rate_m_min must be positive")
        if self.level_set_band_width_cells <= 0:
            raise ValueError("level-set band width must be positive")
        if self.level_set_reinitialization_interval_min < 0:
            raise ValueError("level-set reinitialization interval cannot be negative")
        if self.wind_speed_adjustment <= 0:
            raise ValueError("wind-speed adjustment must be positive")
        if not -0.25 <= self.dead_fuel_moisture_bias <= 0.25:
            raise ValueError("dead-fuel moisture bias must be within [-0.25, 0.25]")
        if self.history_correction_half_life_min <= 0.0:
            raise ValueError("history correction half-life must be positive")
        if not 0.0 <= self.history_direction_weight <= 1.0:
            raise ValueError("history direction weight must be within [0, 1]")
        if not 0.0 < self.history_rate_ratio_min <= self.history_rate_ratio_max:
            raise ValueError("history rate-ratio limits are invalid")


@dataclass(frozen=True)
class SuppressionConfig:
    """Physical and operational controls for suppression resources.

    Liquid quantities are conserved as volume per ground area.  One coverage
    level (GPC) is one US gallon per 100 square feet.
    """

    gpc_l_m2: float = 3.785411784 / 9.290304
    water_half_life_min: float = 8.0
    retardant_half_life_min: float = 720.0
    retardant_rain_wash_fraction_per_mm: float = 0.08
    water_intensity_reduction: float = 0.78
    retardant_spread_reduction: float = 0.82
    line_hold_spread_reduction: float = 1.0
    line_capacity_base_kw_m: float = 420.0
    line_capacity_per_m_width_kw_m: float = 850.0
    line_capacity_retardant_kw_m: float = 650.0
    line_breach_logistic_scale_kw_m: float = 230.0
    line_production_cv: float = 0.38
    direct_attack_max_intensity_kw_m: float = 2400.0
    aviation_max_wind_m_s: float = 18.0
    base_reload_bays: int = 2
    drop_drift_m_per_m_s: float = 7.0
    drop_dispersion_growth_per_m_s: float = 0.025
    minimum_dispatch_payload_fraction: float = 0.80
    minimum_reserve_endurance_min: float = 15.0

    def __post_init__(self) -> None:
        if self.gpc_l_m2 <= 0.0:
            raise ValueError("GPC conversion must be positive")
        if self.water_half_life_min <= 0.0 or self.retardant_half_life_min <= 0.0:
            raise ValueError("treatment half-lives must be positive")
        if self.line_breach_logistic_scale_kw_m <= 0.0:
            raise ValueError("line breach scale must be positive")
        if self.line_production_cv < 0.0:
            raise ValueError("line production CV cannot be negative")
        if self.base_reload_bays < 1:
            raise ValueError("at least one reload bay is required")
        if not 0.0 < self.minimum_dispatch_payload_fraction <= 1.0:
            raise ValueError("minimum dispatch payload fraction must be within (0, 1]")
        if self.minimum_reserve_endurance_min < 0.0:
            raise ValueError("minimum reserve endurance cannot be negative")


@dataclass(frozen=True)
class ServiceSiteSpec:
    """A manually or geospatially defined aircraft service node.

    ``services`` uses payload names (``water`` and ``retardant``) plus
    ``fuel`` or ``charge``.  Water-only dip sites refill payload without
    resetting sortie endurance.
    """

    site_id: str
    kind: str
    x: int
    y: int
    services: tuple[str, ...]
    service_mode: str
    bays: int = 1
    refill_rate_l_min: float = 1000.0
    fixed_turnaround_min: float = 1.0
    available_volume_l: float = 1.0e12
    open_minute: int = 0
    close_minute: int = 24 * 60
    max_operating_wind_m_s: float = 20.0
    approach_capacity: int = 1
    minimum_depth_m: float = 0.0
    minimum_length_m: float = 0.0
    manually_verified: bool = False

    def __post_init__(self) -> None:
        allowed_kinds = {
            "airport",
            "retardant_base",
            "helibase",
            "dip_site",
            "scoopable_water",
            "temporary_tank",
        }
        if self.kind not in allowed_kinds:
            raise ValueError(f"unknown service-site kind: {self.kind}")
        if not self.services or not set(self.services) <= {"water", "retardant", "fuel", "charge"}:
            raise ValueError("service site has an invalid services set")
        if self.service_mode not in {"land", "hover_fill", "scoop"}:
            raise ValueError("service_mode must be land, hover_fill, or scoop")
        if self.bays < 1 or self.approach_capacity < 1:
            raise ValueError("service-site capacities must be positive")
        if self.refill_rate_l_min <= 0.0 or self.fixed_turnaround_min < 0.0:
            raise ValueError("service-site timing values are invalid")
        if not isfinite(self.available_volume_l) or self.available_volume_l <= 0.0:
            raise ValueError("service-site available volume must be finite and positive")
        if self.open_minute < 0 or self.close_minute <= self.open_minute:
            raise ValueError("service-site operating window is invalid")
        if self.minimum_depth_m < 0.0 or self.minimum_length_m < 0.0:
            raise ValueError("service-site geometry limits cannot be negative")
        if not isfinite(self.max_operating_wind_m_s) or self.max_operating_wind_m_s <= 0.0:
            raise ValueError("service-site wind limit must be finite and positive")
        if self.kind == "dip_site" and (self.service_mode != "hover_fill" or "water" not in self.services):
            raise ValueError("dip sites must provide water by hover_fill")
        if self.kind == "scoopable_water" and (self.service_mode != "scoop" or "water" not in self.services):
            raise ValueError("scoopable-water sites must provide water by scoop")
        if self.kind == "retardant_base" and "retardant" not in self.services:
            raise ValueError("retardant bases must provide retardant")


@dataclass(frozen=True)
class AirspaceVolumeSpec:
    """A time-active horizontal volume in scenario-grid coordinates."""

    volume_id: str
    polygon_xy: tuple[tuple[float, float], ...]
    lower_altitude_m_msl: float
    upper_altitude_m_msl: float
    start_minute: int = 0
    end_minute: int = 24 * 60
    kind: str = "prohibited"
    allowed_resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.volume_id.strip():
            raise ValueError("airspace-volume identifier cannot be empty")
        if len(self.polygon_xy) < 3:
            raise ValueError("airspace-volume polygon requires at least three vertices")
        if any(
            len(point) != 2 or not all(isfinite(float(value)) for value in point) for point in self.polygon_xy
        ):
            raise ValueError("airspace-volume polygon contains an invalid vertex")
        if self.lower_altitude_m_msl < 0.0 or self.upper_altitude_m_msl <= self.lower_altitude_m_msl:
            raise ValueError("airspace-volume altitude band is invalid")
        if self.start_minute < 0 or self.end_minute <= self.start_minute:
            raise ValueError("airspace-volume time interval is invalid")
        if self.kind not in {"prohibited", "reserved"}:
            raise ValueError("airspace-volume kind must be prohibited or reserved")


@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    kind: str
    cruise_speed_m_s: float
    payload_l: float
    reload_min: int
    dispatch_latency_min: int
    endurance_min: int
    water_radius_m: float = 150.0
    retardant_length_m: float = 750.0
    retardant_width_m: float = 110.0
    line_length_m: float = 600.0
    line_width_m: float = 1.2
    line_production_m_min: float = 0.0
    max_operating_wind_m_s: float = 18.0
    max_direct_intensity_kw_m: float = 2400.0
    target_coverage_level_gpc: float = 2.0
    home_site_id: str | None = None
    service_modes: tuple[str, ...] = ("land",)
    reserve_endurance_min: float = 15.0
    drop_speed_m_s: float = 40.0
    minimum_drop_length_m: float = 120.0
    maximum_drop_length_m: float = 1200.0
    performance_surface_path: str | None = None
    delivery_surface_path: str | None = None
    delivery_evidence_grade: str = "scenario_assumption"
    cruise_altitude_agl_m: float = 150.0
    minimum_terrain_clearance_m: float = 60.0
    maximum_operating_altitude_m_msl: float = 10_000.0
    maximum_crosswind_m_s: float = 18.0
    minimum_service_depth_m: float = 0.0
    minimum_service_length_m: float = 0.0
    vehicle_profile_id: str | None = None
    performance_evidence_grade: str = "scenario_assumption"
    autonomy_level: str = "crewed"
    operational_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"retardant", "water", "sensor", "crew"}:
            raise ValueError(f"unknown resource kind: {self.kind}")
        if self.cruise_speed_m_s <= 0.0 or self.endurance_min <= 0:
            raise ValueError("resource speed and endurance must be positive")
        if self.payload_l < 0.0 or self.reload_min < 0 or self.dispatch_latency_min < 0:
            raise ValueError("resource payload and timing values cannot be negative")
        if not set(self.service_modes) <= {"land", "hover_fill", "scoop"}:
            raise ValueError("resource has an invalid service mode")
        if self.reserve_endurance_min < 0.0 or self.reserve_endurance_min >= self.endurance_min:
            raise ValueError("resource reserve endurance is invalid")
        if not 0.0 < self.minimum_drop_length_m <= self.maximum_drop_length_m:
            raise ValueError("resource drop-length limits are invalid")
        if (
            self.cruise_altitude_agl_m < self.minimum_terrain_clearance_m
            or self.minimum_terrain_clearance_m < 0.0
            or self.maximum_operating_altitude_m_msl <= 0.0
        ):
            raise ValueError("resource altitude limits are invalid")
        if self.maximum_crosswind_m_s <= 0.0:
            raise ValueError("resource crosswind limit must be positive")
        if self.minimum_service_depth_m < 0.0 or self.minimum_service_length_m < 0.0:
            raise ValueError("resource service-site geometry requirements are invalid")
        if self.performance_evidence_grade not in {
            "scenario_assumption",
            "public_specification",
            "flight_manual",
            "engineering_validated",
        }:
            raise ValueError("unknown resource performance evidence grade")
        if self.delivery_evidence_grade not in {
            "scenario_assumption",
            "public_specification",
            "flight_manual",
            "engineering_validated",
        }:
            raise ValueError("unknown resource delivery evidence grade")
        if self.autonomy_level not in {
            "crewed",
            "remotely_piloted",
            "supervised_autonomy",
            "research_autonomy",
        }:
            raise ValueError("unknown resource autonomy level")
        if any(not str(role).strip() for role in self.operational_roles):
            raise ValueError("resource operational roles cannot contain an empty value")


DEFAULT_RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec("tanker_12", "retardant", 70.0, 11000.0, 14, 3, 210),
    ResourceSpec("heli_07", "water", 48.0, 2800.0, 8, 2, 150),
    ResourceSpec("ir_scout", "sensor", 38.0, 0.0, 0, 1, 120),
)


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str = "synthetic-domain"
    title: str = "Synthetic research domain"
    location_name: str | None = None
    time_origin: str | None = None
    seed: int = 20260726
    width: int = 96
    height: int = 96
    cell_size_m: float = 60.0
    horizon_min: int = 240
    decision_interval_min: int = 3
    max_tasks: int = 64
    observation_delay_min: int = 3
    wind_speed_m_s: float = 6.0
    wind_direction_deg: float = 25.0
    wind_variability: float = 0.25
    air_temperature_c: float = 30.0
    relative_humidity_pct: float = 24.0
    precipitation_rate_mm_h: float = 0.0
    spotting_rate: float = 0.01
    ground_arrival_min: int = 25
    reward_loss_scale: float = 0.05
    escape_penalty: float = 12.0
    containment_bonus: float = 12.0
    terminate_on_escape: bool = True
    initial_perimeter_radius_cells: float = 2.5
    residual_spread_std: float = 0.18
    landscape_bundle: str | None = None
    weather_forcing: str | None = None
    fuel: FuelModel = field(default_factory=FuelModel)
    fire: FireBehaviorConfig = field(default_factory=FireBehaviorConfig)
    suppression: SuppressionConfig = field(default_factory=SuppressionConfig)
    resources: tuple[ResourceSpec, ...] = DEFAULT_RESOURCES
    service_sites: tuple[ServiceSiteSpec, ...] = ()
    airspace_volumes: tuple[AirspaceVolumeSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.title.strip():
            raise ValueError("scenario_id and title cannot be empty")
        if self.time_origin is not None:
            try:
                origin = datetime.fromisoformat(self.time_origin.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("time_origin must be an ISO 8601 timestamp") from exc
            if origin.tzinfo is None:
                raise ValueError("time_origin must include an explicit UTC offset")
        if self.width < 16 or self.height < 16:
            raise ValueError("domain must be at least 16 cells in each dimension")
        if self.decision_interval_min < 1:
            raise ValueError("decision_interval_min must be positive")
        if self.max_tasks < 8:
            raise ValueError("max_tasks must reserve room for hold and front tasks")
        if self.reward_loss_scale <= 0:
            raise ValueError("reward_loss_scale must be positive")
        site_ids = [site.site_id for site in self.service_sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("service-site identifiers must be unique")
        volume_ids = [volume.volume_id for volume in self.airspace_volumes]
        if len(volume_ids) != len(set(volume_ids)):
            raise ValueError("airspace-volume identifiers must be unique")
        for site in self.service_sites:
            if not (0 <= site.x < self.width and 0 <= site.y < self.height):
                raise ValueError(f"service site {site.site_id} is outside the scenario grid")
        known_sites = set(site_ids)
        for resource in self.resources:
            if resource.home_site_id is not None and resource.home_site_id not in known_sites:
                raise ValueError(
                    f"resource {resource.resource_id} references unknown home site {resource.home_site_id}"
                )
            if resource.home_site_id is not None and resource.kind != "crew":
                home = next(site for site in self.service_sites if site.site_id == resource.home_site_id)
                if home.service_mode not in resource.service_modes or not (
                    resource.kind in home.services or {"fuel", "charge"}.intersection(home.services)
                ):
                    raise ValueError(
                        f"home site {home.site_id} cannot service resource {resource.resource_id}"
                    )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260726
    device: str = "auto"
    num_envs: int = 32
    rollout_steps: int = 128
    recurrent_sequence_length: int = 16
    updates: int = 1000
    epochs_per_update: int = 4
    minibatch_size: int = 2048
    learning_rate: float = 2.0e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip_ratio: float = 0.20
    entropy_coef: float = 0.012
    value_coef: float = 0.5
    max_grad_norm: float = 0.8
    hidden_dim: int = 192
    checkpoint_dir: str = "runs/default"
    checkpoint_every: int = 25
    use_amp: bool = True
    expert_warmstart_steps: int = 0
    expert_policy: str = "joint_assignment"
    environment_backend: str = "canonical"
    policy_architecture: str = "task_pointer"
    tensor_max_segments: int = 32
    tensor_grid_size: int = 64
    tensor_fire_substeps: int = 2
    tensor_observation_period_min: int = 12
    attention_heads: int = 4
    attention_layers: int = 2
    compile_model: bool = False
    compile_environment: bool = False

    def __post_init__(self) -> None:
        if self.environment_backend not in {
            "canonical",
            "tensor_operations",
            "tensor_incident",
        }:
            raise ValueError("unknown training environment backend")
        if self.policy_architecture not in {"task_pointer", "entity_attention"}:
            raise ValueError("unknown policy architecture")
        if self.tensor_max_segments < 1:
            raise ValueError("tensor_max_segments must be positive")
        if self.tensor_grid_size < 16:
            raise ValueError("tensor_grid_size must be at least 16")
        if self.tensor_fire_substeps < 1:
            raise ValueError("tensor_fire_substeps must be positive")
        if self.tensor_observation_period_min < 1:
            raise ValueError("tensor observation period must be positive")
        if self.recurrent_sequence_length < 1:
            raise ValueError("recurrent sequence length must be positive")
        if self.rollout_steps % self.recurrent_sequence_length:
            raise ValueError("rollout_steps must be divisible by recurrent_sequence_length")
        if self.minibatch_size < self.recurrent_sequence_length:
            raise ValueError("minibatch_size must contain at least one sequence")
        if self.attention_heads < 1 or self.attention_layers < 1:
            raise ValueError("attention configuration must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resource(value: dict[str, Any]) -> ResourceSpec:
    payload = dict(value)
    if "service_modes" in payload:
        payload["service_modes"] = tuple(payload["service_modes"])
    if "operational_roles" in payload:
        payload["operational_roles"] = tuple(payload["operational_roles"])
    return ResourceSpec(**payload)


def _service_site(value: dict[str, Any]) -> ServiceSiteSpec:
    payload = dict(value)
    if "services" in payload:
        payload["services"] = tuple(payload["services"])
    return ServiceSiteSpec(**payload)


def _airspace_volume(value: dict[str, Any]) -> AirspaceVolumeSpec:
    payload = dict(value)
    if "polygon_xy" in payload:
        payload["polygon_xy"] = tuple(
            tuple(float(coordinate) for coordinate in point) for point in payload["polygon_xy"]
        )
    if "allowed_resource_ids" in payload:
        payload["allowed_resource_ids"] = tuple(payload["allowed_resource_ids"])
    return AirspaceVolumeSpec(**payload)


def _scenario(value: dict[str, Any]) -> ScenarioConfig:
    payload = dict(value)
    if "fuel" in payload:
        payload["fuel"] = FuelModel(**payload["fuel"])
    if "fire" in payload:
        payload["fire"] = FireBehaviorConfig(**payload["fire"])
    if "suppression" in payload:
        payload["suppression"] = SuppressionConfig(**payload["suppression"])
    if "resources" in payload:
        payload["resources"] = tuple(_resource(item) for item in payload["resources"])
    if "service_sites" in payload:
        payload["service_sites"] = tuple(_service_site(item) for item in payload["service_sites"])
    if "airspace_volumes" in payload:
        payload["airspace_volumes"] = tuple(_airspace_volume(item) for item in payload["airspace_volumes"])
    return ScenarioConfig(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment manifest and reject unknown dataclass fields."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("experiment config must be a mapping")
    return ExperimentConfig(
        scenario=_scenario(raw.get("scenario", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
