"""Versioned experiment configuration with no hidden global defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    moisture_model: str = "equilibrium_time_lag"
    enable_crown_fire: bool = True
    enable_spotting: bool = True
    max_spread_rate_m_min: float = 180.0
    propagation_cfl: float = 0.35
    max_substeps: int = 16
    min_front_residence_min: float = 20.0
    max_front_residence_min: float = 360.0
    surface_spread_adjustment: float = 1.0
    crown_spread_adjustment: float = 1.0
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

    def __post_init__(self) -> None:
        if self.backend not in {"operational"}:
            raise ValueError("fire backend must be 'operational'")
        if self.moisture_model not in {"equilibrium_time_lag", "fixed"}:
            raise ValueError("unknown fuel-moisture model")
        if not 0 < self.propagation_cfl <= 1:
            raise ValueError("propagation_cfl must be within (0, 1]")
        if self.max_substeps < 1:
            raise ValueError("max_substeps must be positive")
        if self.max_spread_rate_m_min <= 0:
            raise ValueError("max_spread_rate_m_min must be positive")


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


DEFAULT_RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec("tanker_12", "retardant", 70.0, 11000.0, 14, 3, 210),
    ResourceSpec("heli_07", "water", 48.0, 2800.0, 8, 2, 150),
    ResourceSpec("ir_scout", "sensor", 38.0, 0.0, 0, 1, 120),
)


@dataclass(frozen=True)
class ScenarioConfig:
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
    resources: tuple[ResourceSpec, ...] = DEFAULT_RESOURCES

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("domain must be at least 16 cells in each dimension")
        if self.decision_interval_min < 1:
            raise ValueError("decision_interval_min must be positive")
        if self.max_tasks < 8:
            raise ValueError("max_tasks must reserve room for hold and front tasks")
        if self.reward_loss_scale <= 0:
            raise ValueError("reward_loss_scale must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260726
    device: str = "auto"
    num_envs: int = 32
    rollout_steps: int = 128
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


@dataclass(frozen=True)
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resource(value: dict[str, Any]) -> ResourceSpec:
    return ResourceSpec(**value)


def _scenario(value: dict[str, Any]) -> ScenarioConfig:
    payload = dict(value)
    if "fuel" in payload:
        payload["fuel"] = FuelModel(**payload["fuel"])
    if "fire" in payload:
        payload["fire"] = FireBehaviorConfig(**payload["fire"])
    if "resources" in payload:
        payload["resources"] = tuple(_resource(item) for item in payload["resources"])
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
