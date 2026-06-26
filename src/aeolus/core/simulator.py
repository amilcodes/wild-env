"""Event-driven truth simulator behind all Aeolus environment adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil, hypot
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.config import ResourceSpec, ScenarioConfig
from aeolus.core.aviation import evaluate_simulator_leg
from aeolus.core.fire import step_fire
from aeolus.core.front import signed_distance
from aeolus.core.initialization import reconstruct_arrival_history
from aeolus.core.localization import localize_front_correction
from aeolus.core.state import (
    BeliefState,
    EpisodeState,
    FirePhase,
    FireType,
    PendingObservation,
    ResourceRuntime,
    ResourceStatus,
    ServiceSiteRuntime,
    TruthState,
)
from aeolus.core.suppression import (
    apply_aerial_drop,
    construct_line_segment,
    required_coverage_level_gpc,
    update_suppression_state,
)
from aeolus.core.tasks import (
    Task,
    TaskKind,
    action_mask,
    actor_global_features,
    generate_tasks,
    resource_features,
    task_tensor,
    task_travel_min,
)
from aeolus.data import IncidentBundle, WeatherForcing, load_bundle
from aeolus.data.aerial_delivery import (
    delivery_geometry,
    load_aerial_delivery_surface,
)


class AeolusSimulator:
    """One reproducible incident with truth/belief separation.

    `decision_step` executes a resource-task assignment, advances all internal
    minute dynamics to the next decision point, and returns a shared reward.
    This keeps mission/turnaround time in the environment rather than asking an
    actor to emit useless per-minute flight controls.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.weather = self._load_weather_forcing()
        self._forcing_minute_offset = 0.0
        self.state: EpisodeState
        self.tasks: list[Task] = []
        self._state_observers: list[Callable[[AeolusSimulator], None]] = []
        self.reset(config.seed)

    def _load_weather_forcing(self) -> WeatherForcing | None:
        weather_path = Path(self.config.weather_forcing) if self.config.weather_forcing else None
        if weather_path is None and self.config.landscape_bundle:
            landscape_path = Path(self.config.landscape_bundle)
            if landscape_path.is_dir() and (landscape_path / "item.json").exists():
                weather_path = IncidentBundle.load(landscape_path).asset_path("weather", required=False)
        return WeatherForcing.load(weather_path) if weather_path is not None else None

    def current_weather(self) -> dict[str, float | np.ndarray]:
        """Return forcing values at the current simulation minute."""

        if self.weather is not None:
            return self.weather.at_minute(self.forcing_minute)
        return {
            "wind_speed_m_s": float(self.config.wind_speed_m_s),
            "wind_direction_deg": float(self.config.wind_direction_deg),
            "air_temperature_c": float(self.config.air_temperature_c),
            "relative_humidity_pct": float(self.config.relative_humidity_pct),
            "precipitation_rate_mm_h": float(self.config.precipitation_rate_mm_h),
        }

    @property
    def forcing_minute(self) -> float:
        """Current minute in the weather forcing's absolute clock."""

        return float(self._forcing_minute_offset + self.state.minute)

    def set_simulation_start(self, timestamp: datetime | str) -> float:
        """Align episode minute zero with an absolute incident timestamp.

        Historical forecasts often begin well after the first incident
        perimeter. This offset prevents every held-out interval from replaying
        the first hours of its weather file.
        """

        start = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if isinstance(timestamp, str)
            else timestamp
        )
        if start.tzinfo is None:
            raise ValueError("simulation start must include an explicit UTC offset")
        start = start.astimezone(timezone.utc)
        weather_origin = self.weather.time_origin if self.weather is not None else None
        if weather_origin is not None:
            origin = weather_origin
        elif self.config.time_origin is not None:
            origin = datetime.fromisoformat(self.config.time_origin.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        elif self.weather is None:
            self._forcing_minute_offset = 0.0
            return 0.0
        else:
            raise ValueError("absolute simulation start requires weather time units or config.time_origin")
        self._forcing_minute_offset = (start - origin).total_seconds() / 60.0
        self.state.event(
            "forcing_clock_aligned",
            simulation_start=start.isoformat(),
            forcing_origin=origin.isoformat(),
            forcing_minute_offset=self._forcing_minute_offset,
        )
        self._synchronize_forcing_state()
        return float(self._forcing_minute_offset)

    def _synchronize_forcing_state(self) -> tuple[str, ...]:
        """Copy explicitly forced coupled-state variables into truth state."""

        if self.weather is None:
            return ()
        weather = self.current_weather()
        moisture_targets = (
            ("moisture_dead_1h", self.state.truth.moisture_dead_1h),
            ("moisture_dead_10h", self.state.truth.moisture_dead_10h),
            ("moisture_dead_100h", self.state.truth.moisture_dead_100h),
            (
                "moisture_live_herbaceous",
                self.state.truth.moisture_live_herbaceous,
            ),
            ("moisture_live_woody", self.state.truth.moisture_live_woody),
        )
        applied: list[str] = []
        for name, target in moisture_targets:
            if name not in weather:
                continue
            values = np.asarray(weather[name], dtype=np.float32)
            if values.ndim == 0:
                target[:] = float(values)
            elif values.shape == target.shape:
                target[:] = values
            else:
                raise ValueError(
                    f"forcing field {name} shape {values.shape} does not match fire grid {target.shape}"
                )
            applied.append(name)
        return tuple(applied)

    @staticmethod
    def _weather_at_cell(
        value: float | np.ndarray,
        x: int,
        y: int,
    ) -> float:
        field = np.asarray(value)
        return float(field if field.ndim == 0 else field[y, x])

    @property
    def agent_ids(self) -> list[str]:
        return [resource.resource_id for resource in self.state.resources]

    def reset(self, seed: int | None = None) -> dict[str, dict[str, np.ndarray]]:
        episode_seed = self.config.seed if seed is None else int(seed)
        self._forcing_minute_offset = 0.0
        rng = np.random.default_rng(episode_seed)
        truth, base_xy = self._build_truth(rng)
        belief = BeliefState(
            intensity_mean=np.zeros_like(truth.intensity_kw_m, dtype=np.float32),
            intensity_std=np.full_like(truth.intensity_kw_m, 1.0, dtype=np.float32),
            observed_at=np.full(truth.intensity_kw_m.shape, -9999, dtype=np.int32),
            known_burned=np.zeros_like(truth.intensity_kw_m, dtype=np.float32),
            burn_probability=np.zeros_like(truth.intensity_kw_m, dtype=np.float32),
            arrival_time_mean=np.full(truth.intensity_kw_m.shape, np.inf, dtype=np.float32),
            arrival_time_std=np.full(truth.intensity_kw_m.shape, np.inf, dtype=np.float32),
        )
        resources = [self._initial_resource_runtime(spec, base_xy) for spec in self.config.resources]
        service_sites = [
            ServiceSiteRuntime(
                spec=spec,
                remaining_volume_l=float(spec.available_volume_l),
            )
            for spec in self.config.service_sites
        ]
        self.state = EpisodeState(
            minute=0,
            truth=truth,
            belief=belief,
            resources=resources,
            service_sites=service_sites,
            base_xy=base_xy,
            rng=rng,
        )
        # Discovery observation is an explicit pre-episode information event.
        ignition_x, ignition_y = self.config.width // 2 - 7, self.config.height // 2 + 5
        self._capture_observation(ignition_x, ignition_y, 9, "initial_attack", delay=0)
        self._deliver_observations()
        self.tasks = generate_tasks(self)
        self.state.event("reset", scenario_seed=episode_seed)
        self._notify_state_observers()
        return self.observations()

    def _initial_resource_runtime(
        self,
        spec: ResourceSpec,
        base_xy: tuple[int, int],
    ) -> ResourceRuntime:
        site = None
        if spec.home_site_id is not None:
            site = next(item for item in self.config.service_sites if item.site_id == spec.home_site_id)
        elif self.config.service_sites and spec.kind != "crew":
            site = next(
                (
                    item
                    for item in self.config.service_sites
                    if item.service_mode in spec.service_modes
                    and (spec.kind in item.services or bool({"fuel", "charge"}.intersection(item.services)))
                ),
                None,
            )
        position = base_xy if site is None else (site.x, site.y)
        return ResourceRuntime(
            spec=spec,
            x=float(position[0]),
            y=float(position[1]),
            current_site_id=None if site is None else site.site_id,
        )

    def add_state_observer(
        self, observer: Callable[[AeolusSimulator], None], *, capture_initial: bool = True
    ) -> None:
        """Attach a read-only observer used by replay recorders."""

        self._state_observers.append(observer)
        if capture_initial:
            observer(self)

    def remove_state_observer(self, observer: Callable[[AeolusSimulator], None]) -> None:
        self._state_observers.remove(observer)

    def _notify_state_observers(self) -> None:
        for observer in tuple(self._state_observers):
            observer(self)

    def _build_truth(self, rng: np.random.Generator) -> tuple[TruthState, tuple[int, int]]:
        height, width = self.config.height, self.config.width
        y, x = np.mgrid[0:height, 0:width]
        if self.config.landscape_bundle:
            landscape_path = Path(self.config.landscape_bundle)
            bundle = (
                IncidentBundle.load(landscape_path).scenario_bundle()
                if landscape_path.is_dir()
                else load_bundle(landscape_path)
            )
            if bundle.elevation_m.shape != (height, width):
                raise ValueError(
                    "scenario dimensions do not match the landscape bundle: "
                    f"expected {(height, width)}, received {bundle.elevation_m.shape}"
                )
            if not np.isclose(float(bundle.metadata["cell_size_m"]), self.config.cell_size_m):
                raise ValueError("scenario cell_size_m does not match the landscape bundle")
            elevation = bundle.elevation_m.copy()
            fuel_load = bundle.fuel_load_kg_m2.copy()
            barrier = bundle.barrier.copy()
            asset_value = bundle.asset_value.copy()
            fuel_model_number = (
                bundle.fuel_model_number.copy()
                if bundle.fuel_model_number is not None
                else np.full(
                    (height, width),
                    self.config.fuel.standard_number,
                    dtype=np.int16,
                )
            )
            canopy_cover = (
                bundle.canopy_cover.copy()
                if bundle.canopy_cover is not None
                else np.zeros((height, width), dtype=np.float32)
            )
            canopy_height = (
                bundle.canopy_height_m.copy()
                if bundle.canopy_height_m is not None
                else np.zeros((height, width), dtype=np.float32)
            )
            canopy_base_height = (
                bundle.canopy_base_height_m.copy()
                if bundle.canopy_base_height_m is not None
                else np.zeros((height, width), dtype=np.float32)
            )
            canopy_bulk_density = (
                bundle.canopy_bulk_density_kg_m3.copy()
                if bundle.canopy_bulk_density_kg_m3 is not None
                else np.zeros((height, width), dtype=np.float32)
            )
        else:
            elevation = (
                310.0
                + 95.0 * np.sin(x / 12.0)
                + 66.0 * np.cos(y / 16.0)
                + 35.0 * np.sin((x + y) / 17.0)
                + rng.normal(0.0, 4.0, size=(height, width))
            ).astype(np.float32)
            fuel_multiplier = np.clip(
                0.75
                + 0.18 * np.sin(x / 7.0)
                + 0.17 * np.cos(y / 8.0)
                + rng.normal(0.0, 0.08, size=(height, width)),
                0.12,
                1.35,
            ).astype(np.float32)
            fuel_load = fuel_multiplier * self.config.fuel.fuel_load_kg_m2
            fuel_model_number = np.full((height, width), self.config.fuel.standard_number, dtype=np.int16)
            fuel_model_number[fuel_multiplier < 0.66] = 101
            fuel_model_number[fuel_multiplier > 1.02] = 145
            barrier = np.zeros((height, width), dtype=np.bool_)
            barrier[:, 3:5] = True
            road_y = int(height * 0.72)
            barrier[road_y : road_y + 2, int(width * 0.55) :] = True
            asset_value = np.zeros((height, width), dtype=np.float32)
            asset_x, asset_y = int(width * 0.77), int(height * 0.25)
            asset_dist = np.hypot(x - asset_x, y - asset_y)
            asset_value[asset_dist <= 5.2] = np.clip(1.0 - asset_dist[asset_dist <= 5.2] / 7.0, 0.3, 1.0)
            woodland = np.clip(
                0.20 + 0.38 * np.sin((x + 2 * y) / 29.0) + 0.22 * np.cos((2 * x - y) / 21.0),
                0.0,
                0.82,
            ).astype(np.float32)
            canopy_cover = np.where(fuel_multiplier > 0.88, woodland, 0.0).astype(np.float32)
            canopy_height = np.where(canopy_cover >= 0.20, 8.0 + 10.0 * canopy_cover, 0.0).astype(np.float32)
            canopy_base_height = np.where(canopy_cover >= 0.20, 2.1 + 2.5 * (1.0 - canopy_cover), 0.0).astype(
                np.float32
            )
            canopy_bulk_density = np.where(canopy_cover >= 0.20, 0.06 + 0.17 * canopy_cover, 0.0).astype(
                np.float32
            )
        phase = np.full((height, width), FirePhase.UNBURNED, dtype=np.uint8)
        intensity = np.zeros((height, width), dtype=np.float32)
        ignition_x, ignition_y = width // 2 - 7, height // 2 + 5
        ignition_distance = np.hypot(x - ignition_x, y - ignition_y)
        ignition = ignition_distance <= self.config.initial_perimeter_radius_cells
        phase[ignition] = FirePhase.FLAMING
        intensity[ignition] = (760.0 * np.clip(1.0 - ignition_distance[ignition] / 5.0, 0.35, 1.0)).astype(
            np.float32
        )
        barrier |= (fuel_model_number >= 91) & (fuel_model_number <= 99)
        residual_base = rng.normal(
            0.0,
            self.config.residual_spread_std,
            size=(max(2, ceil(height / 6)), max(2, ceil(width / 6))),
        )
        residual = np.kron(residual_base, np.ones((6, 6), dtype=np.float32))[:height, :width]
        residual = np.clip(np.exp(residual), 0.55, 1.65).astype(np.float32)
        truth = TruthState(
            phase=phase,
            intensity_kw_m=intensity,
            fuel_remaining=np.ones((height, width), dtype=np.float32),
            fuel_load=fuel_load,
            elevation_m=elevation,
            barrier=barrier,
            asset_value=asset_value,
            water=np.zeros((height, width), dtype=np.float32),
            retardant=np.zeros((height, width), dtype=np.float32),
            ground_hold=np.zeros((height, width), dtype=np.float32),
            water_coverage_gpc=np.zeros((height, width), dtype=np.float32),
            retardant_coverage_gpc=np.zeros((height, width), dtype=np.float32),
            retardant_effective_coverage_gpc=np.zeros((height, width), dtype=np.float32),
            constructed_line=np.zeros((height, width), dtype=np.float32),
            line_strength=np.zeros((height, width), dtype=np.float32),
            line_status=np.zeros((height, width), dtype=np.uint8),
            residual_field=residual,
            observed_burned=np.zeros((height, width), dtype=np.float32),
            fuel_model_number=fuel_model_number,
            moisture_dead_1h=np.full(
                (height, width),
                self.config.fuel.dead_moisture,
                dtype=np.float32,
            ),
            moisture_dead_10h=np.full(
                (height, width),
                self.config.fire.dead_moisture_10h,
                dtype=np.float32,
            ),
            moisture_dead_100h=np.full(
                (height, width),
                self.config.fire.dead_moisture_100h,
                dtype=np.float32,
            ),
            moisture_live_herbaceous=np.full(
                (height, width),
                self.config.fire.live_herbaceous_moisture,
                dtype=np.float32,
            ),
            moisture_live_woody=np.full(
                (height, width),
                self.config.fire.live_woody_moisture,
                dtype=np.float32,
            ),
            foliar_moisture=np.full(
                (height, width),
                self.config.fire.foliar_moisture,
                dtype=np.float32,
            ),
            canopy_cover=canopy_cover,
            canopy_height_m=canopy_height,
            canopy_base_height_m=canopy_base_height,
            canopy_bulk_density_kg_m3=canopy_bulk_density,
            fire_type=np.where(ignition, FireType.SURFACE, FireType.UNBURNED).astype(np.uint8),
            spread_rate_m_min=np.zeros((height, width), dtype=np.float32),
            flame_length_m=np.zeros((height, width), dtype=np.float32),
            ignition_progress=np.zeros((height, width), dtype=np.float32),
            level_set_m=signed_distance(ignition, self.config.cell_size_m),
            arrival_time_min=np.where(ignition, 0.0, np.inf).astype(np.float32),
            burn_age_min=np.zeros((height, width), dtype=np.float32),
            history_speed_m_min=np.zeros((height, width), dtype=np.float32),
            history_head_x=np.zeros((height, width), dtype=np.float32),
            history_head_y=np.zeros((height, width), dtype=np.float32),
            history_confidence=np.zeros((height, width), dtype=np.float32),
            history_heat_flux_kw_m2=np.zeros((height, width), dtype=np.float32),
        )
        return truth, (6, height - 7)

    def _capture_observation(
        self, x: int, y: int, radius: int, source: str, delay: int | None = None
    ) -> None:
        truth = self.state.truth
        height, width = truth.phase.shape
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        measure = truth.intensity_kw_m[y0:y1, x0:x1].copy()
        noise_scale = np.maximum(18.0, measure * 0.12)
        measure = np.clip(measure + self.state.rng.normal(0.0, noise_scale), 0.0, None).astype(np.float32)
        burned = truth.observed_burned[y0:y1, x0:x1].copy()
        self.state.belief.pending.append(
            PendingObservation(
                deliver_minute=self.state.minute
                + (self.config.observation_delay_min if delay is None else delay),
                x=x,
                y=y,
                radius_cells=radius,
                source=source,
                intensity_measurement=measure,
                burned_measurement=burned,
            )
        )

    def _deliver_observations(self) -> None:
        belief = self.state.belief
        delivered: list[PendingObservation] = []
        for item in belief.pending:
            if item.deliver_minute > self.state.minute:
                continue
            radius = item.radius_cells
            y0, y1 = max(0, item.y - radius), min(self.config.height, item.y + radius + 1)
            x0, x1 = max(0, item.x - radius), min(self.config.width, item.x + radius + 1)
            if item.intensity_measurement is not None:
                local_y, local_x = np.ogrid[y0:y1, x0:x1]
                circle = (local_x - item.x) ** 2 + (local_y - item.y) ** 2 <= radius**2
                current = belief.intensity_mean[y0:y1, x0:x1]
                current[circle] = item.intensity_measurement[circle]
                belief.intensity_mean[y0:y1, x0:x1] = current
                belief.intensity_std[y0:y1, x0:x1][circle] = np.maximum(
                    12.0, item.intensity_measurement[circle] * 0.14
                )
                belief.observed_at[y0:y1, x0:x1][circle] = self.state.minute
            if item.burned_measurement is not None:
                positive = item.burned_measurement >= 0.5
                if item.intensity_measurement is not None:
                    positive |= item.intensity_measurement >= 20.0
                local_probability = belief.burn_probability[y0:y1, x0:x1]
                local_probability[circle] = np.where(positive[circle], 0.98, 0.02)
                belief.burn_probability[y0:y1, x0:x1] = local_probability
                burned = belief.known_burned[y0:y1, x0:x1]
                burned[circle] = (local_probability[circle] >= 0.90).astype(np.float32)
                belief.known_burned[y0:y1, x0:x1] = burned
                local_arrival = belief.arrival_time_mean[y0:y1, x0:x1]
                local_arrival[circle & positive] = np.minimum(
                    local_arrival[circle & positive],
                    float(self.state.minute),
                )
                belief.arrival_time_mean[y0:y1, x0:x1] = local_arrival
                local_arrival_std = belief.arrival_time_std[y0:y1, x0:x1]
                local_arrival_std[circle & positive] = 1.0
                belief.arrival_time_std[y0:y1, x0:x1] = local_arrival_std
            delivered.append(item)
            self.state.event("observation_delivered", source=item.source, x=item.x, y=item.y)
        if delivered:
            belief.pending = [item for item in belief.pending if item not in delivered]

    def _advance_resource(self, resource: ResourceRuntime) -> None:
        if resource.status == ResourceStatus.AVAILABLE or resource.status == ResourceStatus.WITHDRAWN:
            return
        if resource.status == ResourceStatus.WORKING:
            self._advance_line_work(resource)
            return
        if resource.status == ResourceStatus.QUEUED:
            if self._service_slot_available(resource):
                self._start_service(resource)
            return
        resource.eta_min = max(0, resource.eta_min - 1)
        if (
            resource.status in (ResourceStatus.OUTBOUND, ResourceStatus.RETURNING)
            and resource.leg_start_xy is not None
            and resource.leg_end_xy is not None
            and resource.leg_total_min > 0
        ):
            fraction = 1.0 - resource.eta_min / resource.leg_total_min
            resource.x = (
                resource.leg_start_xy[0] + (resource.leg_end_xy[0] - resource.leg_start_xy[0]) * fraction
            )
            resource.y = (
                resource.leg_start_xy[1] + (resource.leg_end_xy[1] - resource.leg_start_xy[1]) * fraction
            )
        if resource.eta_min > 0:
            return
        if resource.status == ResourceStatus.OUTBOUND:
            assert resource.target_xy is not None
            self._execute_mission(resource)
            return
        if resource.status == ResourceStatus.RETURNING:
            resource.x, resource.y = self.state.base_xy
            if resource.spec.reload_min:
                if self._reload_slots_used() < self.config.suppression.base_reload_bays:
                    self._start_service(resource)
                else:
                    resource.status = ResourceStatus.QUEUED
                    resource.eta_min = 0
                    resource.queue_entered_min = self.state.minute
                    self.state.event(
                        "reload_queued",
                        resource=resource.resource_id,
                    )
            else:
                self._make_resource_available(resource)
            return
        if resource.status == ResourceStatus.RELOADING:
            if resource.spec.payload_l > 0.0:
                resource.payload_fraction = min(
                    1.0,
                    resource.payload_fraction + resource.service_volume_l / resource.spec.payload_l,
                )
            site = self._service_site(resource.service_site_id)
            if site is None or {"fuel", "charge"}.intersection(site.spec.services):
                resource.flight_min = 0.0
            resource.reload_cycles += 1
            resource.service_volume_l = 0.0
            self._make_resource_available(resource)

    def _reload_slots_used(self) -> int:
        return sum(resource.status == ResourceStatus.RELOADING for resource in self.state.resources)

    def _service_site(self, site_id: str | None) -> ServiceSiteRuntime | None:
        if site_id is None:
            return None
        return next(
            (site for site in self.state.service_sites if site.site_id == site_id),
            None,
        )

    def _service_slots_used(self, site_id: str) -> int:
        return sum(
            resource.status == ResourceStatus.RELOADING and resource.service_site_id == site_id
            for resource in self.state.resources
        )

    def _service_slot_available(self, resource: ResourceRuntime) -> bool:
        site = self._service_site(resource.service_site_id)
        if site is None:
            return self._reload_slots_used() < self.config.suppression.base_reload_bays
        return (
            site.spec.open_minute <= self.state.minute < site.spec.close_minute
            and self._service_slots_used(site.site_id) < site.spec.bays
        )

    def _start_service(self, resource: ResourceRuntime) -> None:
        site = self._service_site(resource.service_site_id)
        queue_wait = (
            self.state.minute - resource.queue_entered_min if resource.queue_entered_min is not None else 0
        )
        if site is None:
            resource.status = ResourceStatus.RELOADING
            resource.eta_min = max(1, resource.spec.reload_min)
            resource.service_volume_l = resource.spec.payload_l * (1.0 - resource.payload_fraction)
            self.state.event(
                "reload_started",
                resource=resource.resource_id,
                queue_wait_min=queue_wait,
            )
            return
        missing_l = (
            resource.spec.payload_l * (1.0 - resource.payload_fraction)
            if resource.spec.kind in site.spec.services
            else 0.0
        )
        load_l = min(missing_l, site.remaining_volume_l)
        site.remaining_volume_l -= load_l
        resource.service_volume_l = load_l
        service_min = site.spec.fixed_turnaround_min + load_l / site.spec.refill_rate_l_min
        resource.status = ResourceStatus.RELOADING
        resource.eta_min = max(1, ceil(service_min))
        resource.queue_entered_min = None
        self.state.event(
            "service_started",
            resource=resource.resource_id,
            site=site.site_id,
            queue_wait_min=queue_wait,
            reserved_volume_l=float(load_l),
            eta_min=resource.eta_min,
        )

    def _arrive_at_service_site(self, resource: ResourceRuntime) -> None:
        site = self._service_site(resource.service_site_id)
        if site is None:
            raise RuntimeError("resource arrived at an unknown service site")
        resource.x = float(site.spec.x)
        resource.y = float(site.spec.y)
        resource.current_site_id = site.site_id
        resource.queue_entered_min = self.state.minute
        if self._service_slot_available(resource):
            self._start_service(resource)
        else:
            resource.status = ResourceStatus.QUEUED
            resource.eta_min = 0
            self.state.event(
                "service_queued",
                resource=resource.resource_id,
                site=site.site_id,
            )

    def _make_resource_available(self, resource: ResourceRuntime) -> None:
        resource.status = ResourceStatus.AVAILABLE
        resource.target_xy = None
        resource.task_kind = int(TaskKind.HOLD)
        resource.leg_start_xy = None
        resource.leg_end_xy = None
        resource.leg_total_min = 0
        resource.eta_min = 0
        resource.queue_entered_min = None
        resource.work_remaining_m = 0.0
        resource.line_start_xy = None
        resource.line_end_xy = None
        resource.line_progress_m = 0.0
        resource.service_site_id = None
        self.state.event("resource_available", resource=resource.resource_id)

    def _begin_return(self, resource: ResourceRuntime) -> None:
        distance_back = (
            hypot(
                resource.x - self.state.base_xy[0],
                resource.y - self.state.base_xy[1],
            )
            * self.config.cell_size_m
        )
        if resource.spec.kind in {"water", "retardant", "sensor"}:
            leg = evaluate_simulator_leg(
                resource,
                (
                    float(self.state.base_xy[0]),
                    float(self.state.base_xy[1]),
                ),
                self,
                payload_fraction=0.0,
            )
            if not leg.feasible:
                resource.status = ResourceStatus.WITHDRAWN
                self.state.event(
                    "return_route_infeasible",
                    resource=resource.resource_id,
                    violations=list(leg.violations),
                )
                return
            return_min = leg.travel_min
        else:
            return_min = distance_back / max(
                resource.spec.cruise_speed_m_s * 60.0,
                1.0,
            )
        resource.status = ResourceStatus.RETURNING
        resource.eta_min = max(
            1,
            ceil(return_min),
        )
        resource.leg_start_xy = (float(resource.x), float(resource.y))
        resource.leg_end_xy = (
            float(self.state.base_xy[0]),
            float(self.state.base_xy[1]),
        )
        resource.leg_total_min = resource.eta_min

    def _advance_line_work(self, resource: ResourceRuntime) -> None:
        if resource.line_start_xy is None or resource.line_end_xy is None:
            raise RuntimeError("working crew has no line geometry")
        production = resource.spec.line_production_m_min * resource.production_multiplier
        completed_m = min(production, resource.work_remaining_m)
        previous_m = resource.line_progress_m
        resource.line_progress_m += completed_m
        resource.work_remaining_m -= completed_m
        length_m = max(resource.spec.line_length_m, 1.0)
        start = np.asarray(resource.line_start_xy, dtype=np.float64)
        end = np.asarray(resource.line_end_xy, dtype=np.float64)
        before = start + (end - start) * np.clip(previous_m / length_m, 0.0, 1.0)
        after = start + (end - start) * np.clip(resource.line_progress_m / length_m, 0.0, 1.0)
        constructed_cells = construct_line_segment(
            self.state.truth,
            self.config,
            (float(before[0]), float(before[1])),
            (float(after[0]), float(after[1])),
            resource.spec.line_width_m,
        )
        resource.x, resource.y = float(after[0]), float(after[1])
        self.state.cumulative_cost += 0.025
        self.state.event(
            "line_progress",
            resource=resource.resource_id,
            completed_m=float(completed_m),
            total_completed_m=float(resource.line_progress_m),
            constructed_cells=constructed_cells,
        )
        if resource.work_remaining_m <= 1.0e-6:
            self.state.event(
                "line_complete",
                resource=resource.resource_id,
                length_m=float(resource.line_progress_m),
            )
            self._begin_return(resource)

    def _execute_mission(self, resource: ResourceRuntime) -> None:
        assert resource.target_xy is not None
        task_kind = TaskKind(resource.task_kind)
        x, y = resource.target_xy
        resource.x, resource.y = x, y
        if task_kind == TaskKind.SERVICE:
            self._arrive_at_service_site(resource)
            return
        if task_kind == TaskKind.OBSERVE:
            self._capture_observation(x, y, 9, resource.resource_id)
            self.state.event("observe", resource=resource.resource_id, x=x, y=y)
        elif task_kind == TaskKind.WATER:
            weather = self.current_weather()
            wind_speed = self._weather_at_cell(
                weather["wind_speed_m_s"],
                x,
                y,
            )
            wind_direction = self._weather_at_cell(
                weather["wind_direction_deg"],
                x,
                y,
            )
            diagnostics = apply_aerial_drop(
                self.state.truth,
                self.config,
                kind="water",
                payload_l=resource.spec.payload_l * resource.payload_fraction,
                x=x,
                y=y,
                length_m=2.0 * resource.spec.water_radius_m,
                width_m=2.0 * resource.spec.water_radius_m,
                heading_deg=(wind_direction + 90.0) % 360.0,
                wind_speed_m_s=wind_speed,
                wind_from_direction_deg=wind_direction,
            )
            if self.state.ground_engaged and not any(
                item.spec.kind == "crew" for item in self.state.resources
            ):
                self.state.truth.ground_hold[:] = np.maximum(
                    self.state.truth.ground_hold,
                    0.72 * self.state.truth.retardant,
                )
            resource.payload_fraction = 0.0
            self.state.cumulative_cost += 0.8
            self.state.event(
                "water_drop",
                resource=resource.resource_id,
                x=x,
                y=y,
                **diagnostics,
            )
        elif task_kind in (
            TaskKind.RETARDANT,
            TaskKind.REINFORCE,
            TaskKind.AERIAL_LINE,
        ):
            weather = self.current_weather()
            wind_speed = self._weather_at_cell(
                weather["wind_speed_m_s"],
                x,
                y,
            )
            wind_direction = self._weather_at_cell(
                weather["wind_direction_deg"],
                x,
                y,
            )
            payload_l = resource.spec.payload_l * resource.payload_fraction
            delivery_diagnostics: dict[str, Any] = {}
            local_effective_coverage_gpc: float | None = None
            if resource.spec.delivery_surface_path is not None:
                surface = load_aerial_delivery_surface(resource.spec.delivery_surface_path)
                target_x = int(np.clip(round(x), 0, self.state.truth.phase.shape[1] - 1))
                target_y = int(np.clip(round(y), 0, self.state.truth.phase.shape[0] - 1))
                requested_coverage = float(
                    required_coverage_level_gpc(
                        self.state.truth.fuel_model_number[target_y : target_y + 1, target_x : target_x + 1],
                        self.state.truth.intensity_kw_m[target_y : target_y + 1, target_x : target_x + 1],
                    )[0, 0]
                )
                geometry = delivery_geometry(
                    surface,
                    requested_coverage_gpc=requested_coverage,
                    payload_l=payload_l,
                )
                drop_length_m = geometry.line_length_m
                drop_width_m = geometry.effective_width_m
                local_effective_coverage_gpc = geometry.requested_coverage_gpc
                delivery_diagnostics = {
                    "delivery_surface_id": geometry.surface_id,
                    "requested_coverage_gpc": geometry.requested_coverage_gpc,
                    "delivery_flow_rate_l_s": geometry.flow_rate_l_s,
                    "delivery_controller_setting": geometry.controller_setting,
                    "delivery_surface_extrapolated": geometry.extrapolated,
                }
            else:
                drop_length_m = (
                    float(
                        np.clip(
                            (
                                resource.spec.retardant_length_m
                                if resource.spec.kind == "retardant"
                                else 4.0 * resource.spec.water_radius_m
                            ),
                            resource.spec.minimum_drop_length_m,
                            resource.spec.maximum_drop_length_m,
                        )
                    )
                    if task_kind == TaskKind.AERIAL_LINE
                    else resource.spec.retardant_length_m
                )
                drop_width_m = (
                    resource.spec.retardant_width_m
                    if resource.spec.kind == "retardant"
                    else 2.0 * resource.spec.water_radius_m
                )
            diagnostics = apply_aerial_drop(
                self.state.truth,
                self.config,
                kind=resource.spec.kind,
                payload_l=payload_l,
                x=x,
                y=y,
                length_m=drop_length_m,
                width_m=drop_width_m,
                heading_deg=(
                    resource.task_heading_deg
                    if task_kind in (TaskKind.REINFORCE, TaskKind.AERIAL_LINE)
                    else (wind_direction + 90.0) % 360.0
                ),
                wind_speed_m_s=wind_speed,
                wind_from_direction_deg=wind_direction,
                local_effective_coverage_gpc=local_effective_coverage_gpc,
            )
            diagnostics.update(delivery_diagnostics)
            resource.payload_fraction = 0.0
            self.state.cumulative_cost += 0.8 if resource.spec.kind == "water" else 2.4
            self.state.event(
                ("aerial_line_drop" if task_kind == TaskKind.AERIAL_LINE else "retardant_drop"),
                resource=resource.resource_id,
                x=x,
                y=y,
                suppressant=resource.spec.kind,
                task_index=resource.task_index,
                heading_deg=resource.task_heading_deg,
                **diagnostics,
            )
        elif task_kind == TaskKind.LINE:
            theta = np.deg2rad(resource.task_heading_deg)
            half_cells = 0.5 * resource.spec.line_length_m / self.config.cell_size_m
            dx = float(np.cos(theta) * half_cells)
            dy = float(np.sin(theta) * half_cells)
            resource.line_start_xy = (
                float(np.clip(x - dx, 0, self.config.width - 1)),
                float(np.clip(y - dy, 0, self.config.height - 1)),
            )
            resource.line_end_xy = (
                float(np.clip(x + dx, 0, self.config.width - 1)),
                float(np.clip(y + dy, 0, self.config.height - 1)),
            )
            actual_length = (
                np.hypot(
                    resource.line_end_xy[0] - resource.line_start_xy[0],
                    resource.line_end_xy[1] - resource.line_start_xy[1],
                )
                * self.config.cell_size_m
            )
            cv = self.config.suppression.line_production_cv
            sigma = np.sqrt(np.log1p(cv * cv))
            resource.production_multiplier = float(self.state.rng.lognormal(-0.5 * sigma * sigma, sigma))
            resource.line_progress_m = 0.0
            resource.work_remaining_m = float(actual_length)
            resource.status = ResourceStatus.WORKING
            resource.x, resource.y = resource.line_start_xy
            self.state.event(
                "line_started",
                resource=resource.resource_id,
                x=x,
                y=y,
                planned_length_m=float(actual_length),
                production_m_min=float(resource.spec.line_production_m_min * resource.production_multiplier),
            )
            return
        if self.state.service_sites and task_kind in {
            TaskKind.OBSERVE,
            TaskKind.WATER,
            TaskKind.RETARDANT,
            TaskKind.REINFORCE,
            TaskKind.AERIAL_LINE,
        }:
            resource.current_site_id = None
            self._make_resource_available(resource)
        else:
            self._begin_return(resource)

    def _advance_internal_minute(self) -> None:
        self.state.minute += 1
        for resource in self.state.resources:
            site = self._service_site(resource.service_site_id)
            airborne_service = (
                resource.status in (ResourceStatus.QUEUED, ResourceStatus.RELOADING)
                and site is not None
                and site.spec.service_mode != "land"
            )
            was_flying = airborne_service or resource.status in (
                ResourceStatus.OUTBOUND,
                ResourceStatus.RETURNING,
            )
            if was_flying:
                resource.flight_min += 1.0
                if resource.flight_min >= resource.spec.endurance_min:
                    resource.status = ResourceStatus.WITHDRAWN
                    resource.eta_min = 0
                    self.state.event("resource_withdrawn", resource=resource.resource_id)
                    continue
            self._advance_resource(resource)
            if resource.status not in (ResourceStatus.WITHDRAWN, ResourceStatus.AVAILABLE):
                self.state.cumulative_exposure += 0.04
        explicit_ground = any(
            resource.spec.kind == "crew" and resource.status == ResourceStatus.WORKING
            for resource in self.state.resources
        )
        has_crew_resource = any(resource.spec.kind == "crew" for resource in self.state.resources)
        self.state.ground_engaged = explicit_ground or (
            not has_crew_resource and self.state.minute >= self.config.ground_arrival_min
        )
        weather = self.current_weather()
        explicit_moisture = self._synchronize_forcing_state()
        suppression_outcome = update_suppression_state(
            self.state.truth,
            self.config,
            self.state.rng,
            precipitation_rate_mm_h=weather["precipitation_rate_mm_h"],
        )
        if suppression_outcome["breached_cells"]:
            self.state.event(
                "line_breached",
                cells=suppression_outcome["breached_cells"],
            )
        new_ignitions = step_fire(
            self.state.truth,
            self.config,
            self.state.rng,
            self.state.minute,
            wind_speed_m_s=weather["wind_speed_m_s"] if self.weather is not None else None,
            wind_direction_deg=(weather["wind_direction_deg"] if self.weather is not None else None),
            air_temperature_c=weather["air_temperature_c"],
            relative_humidity_pct=weather["relative_humidity_pct"],
            precipitation_rate_mm_h=weather["precipitation_rate_mm_h"],
            update_dead_fuel_moisture=not {
                "moisture_dead_1h",
                "moisture_dead_10h",
                "moisture_dead_100h",
            }.issubset(explicit_moisture),
        )
        if new_ignitions:
            self.state.event("fire_growth", cells=new_ignitions)
        self._deliver_observations()
        self._update_terminal_state()
        self._notify_state_observers()

    def _update_terminal_state(self) -> None:
        truth = self.state.truth
        flaming = truth.phase == FirePhase.FLAMING
        boundary = np.concatenate((flaming[0], flaming[-1], flaming[:, 0], flaming[:, -1]))
        if boundary.any():
            first_escape = not self.state.escaped
            self.state.escaped = True
            if self.config.terminate_on_escape:
                self.state.terminated = True
            if first_escape:
                self.state.event("escape")
        elif not flaming.any() and self.state.minute > 5:
            self.state.contained = True
            self.state.terminated = True
            self.state.event("contained")
        elif self.state.minute >= self.config.horizon_min:
            self.state.truncated = True

    def _weighted_loss(self) -> float:
        truth = self.state.truth
        consumed_fraction = np.clip(1.0 - truth.fuel_remaining, 0.0, 1.0)
        burned = truth.observed_burned * consumed_fraction
        # Active cells have already incurred their consumed-fuel loss plus a
        # small current-intensity term; this avoids a discontinuity when a
        # successfully held cell changes from flaming to burned.
        active_proxy = (truth.phase == FirePhase.FLAMING) * np.clip(
            consumed_fraction + truth.intensity_kw_m / 5000.0, 0.0, 1.0
        )
        return float(
            (burned * (1.0 + 9.0 * truth.asset_value)).sum()
            + (active_proxy * (1.0 + 9.0 * truth.asset_value)).sum()
        )

    def _assign(self, actions: dict[str, int]) -> dict[str, dict[str, Any]]:
        taken: dict[int, int] = {}
        accepted: dict[str, bool] = {}
        details: dict[str, dict[str, Any]] = {}
        # Random-but-reproducible auction order removes agent-ID priority as an
        # accidental policy advantage while preserving deterministic replay.
        order = self.state.rng.permutation(len(self.state.resources))
        for position in order:
            resource = self.state.resources[int(position)]
            action = int(actions.get(resource.resource_id, 0))
            mask = action_mask(resource, self.tasks, self.config.max_tasks, self)
            resource.attempted_tasks += int(action != 0)
            if action < 0 or action >= len(mask) or not mask[action]:
                action = 0
                self.state.blocked_actions += 1
                details[resource.resource_id] = {"accepted": False, "reason": "masked"}
            task = self.tasks[action] if action < len(self.tasks) else self.tasks[0]
            if action and taken.get(action, 0) >= task.capacity:
                self.state.blocked_actions += 1
                accepted[resource.resource_id] = False
                details[resource.resource_id] = {"accepted": False, "reason": "task_capacity"}
                continue
            if action == 0:
                accepted[resource.resource_id] = False
                details.setdefault(resource.resource_id, {"accepted": False, "reason": "hold"})
                continue
            route = (
                evaluate_simulator_leg(
                    resource,
                    (float(task.x), float(task.y)),
                    self,
                )
                if resource.spec.kind in {"water", "retardant", "sensor"}
                else None
            )
            travel_min = route.travel_min if route is not None else task_travel_min(resource, task, self)
            resource.target_xy = (task.x, task.y)
            resource.task_index = action
            resource.task_kind = int(task.kind)
            resource.task_heading_deg = float(task.heading_deg)
            resource.service_site_id = task.service_site_id
            resource.current_site_id = None
            resource.status = ResourceStatus.OUTBOUND
            resource.eta_min = max(1, ceil(travel_min) + resource.spec.dispatch_latency_min)
            resource.leg_start_xy = (float(resource.x), float(resource.y))
            resource.leg_end_xy = (float(task.x), float(task.y))
            resource.leg_total_min = resource.eta_min
            resource.accepted_tasks += 1
            resource.flight_min += 0.0
            taken[action] = taken.get(action, 0) + 1
            accepted[resource.resource_id] = True
            details[resource.resource_id] = {
                "accepted": True,
                "task": task.kind.name,
                "eta_min": resource.eta_min,
                **(
                    {
                        "route_density_altitude_m": route.density_altitude_m,
                        "route_groundspeed_m_s": route.groundspeed_m_s,
                        "route_crosswind_m_s": route.crosswind_m_s,
                    }
                    if route is not None
                    else {}
                ),
            }
            self.state.event(
                "assignment",
                resource=resource.resource_id,
                task=task.kind.name,
                x=task.x,
                y=task.y,
                **(
                    {
                        "density_altitude_m": route.density_altitude_m,
                        "groundspeed_m_s": route.groundspeed_m_s,
                        "crosswind_m_s": route.crosswind_m_s,
                        "planned_altitude_m_msl": route.planned_altitude_m_msl,
                    }
                    if route is not None
                    else {}
                ),
            )
        return details

    def decision_step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, dict[str, np.ndarray]], float, bool, bool, dict[str, dict[str, Any]]]:
        """Apply one joint task assignment and advance to next tactical event."""

        before_loss = self._weighted_loss()
        before_cost = self.state.cumulative_cost
        before_blocked = self.state.blocked_actions
        self.state.event(
            "joint_action",
            actions={resource_id: int(action) for resource_id, action in actions.items()},
        )
        assignment_info = self._assign(actions)
        for _ in range(self.config.decision_interval_min):
            if self.state.terminated or self.state.truncated:
                break
            self._advance_internal_minute()
        after_loss = self._weighted_loss()
        reward = (
            -self.config.reward_loss_scale * (after_loss - before_loss)
            - 0.02 * (self.state.cumulative_cost - before_cost)
            - 0.01 * (self.state.blocked_actions - before_blocked)
        )
        if self.state.escaped:
            reward -= self.config.escape_penalty
        if self.state.contained:
            reward += self.config.containment_bonus
        self.tasks = generate_tasks(self)
        infos = {
            resource.resource_id: {
                **assignment_info.get(resource.resource_id, {}),
                "minute": self.state.minute,
                "weighted_loss": after_loss,
                "escaped": self.state.escaped,
                "contained": self.state.contained,
                "blocked_actions": self.state.blocked_actions,
            }
            for resource in self.state.resources
        }
        return self.observations(), float(reward), self.state.terminated, self.state.truncated, infos

    @staticmethod
    def _perimeter_boundary(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.bool_), 1, constant_values=False)
        neighbors_inside = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        return mask.astype(np.bool_) & ~neighbors_inside

    def assimilate_observed_perimeter(
        self,
        mask: np.ndarray,
        *,
        source: str = "historical-perimeter",
        localization_sigma_m: float | None = None,
        confidence: float = 0.95,
    ) -> None:
        """Update the policy belief from a cumulative observed perimeter."""

        if mask.shape != self.state.truth.phase.shape:
            raise ValueError(f"perimeter shape {mask.shape} does not match simulator grid")
        observed = mask.astype(np.bool_)
        boundary = self._perimeter_boundary(observed)
        interior = observed & ~boundary
        belief = self.state.belief
        sigma = self.config.cell_size_m if localization_sigma_m is None else float(localization_sigma_m)
        if sigma <= 0.0:
            raise ValueError("perimeter localization sigma must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("perimeter confidence must be within (0, 1)")
        distance = signed_distance(observed, self.config.cell_size_m)
        scale = max(0.55 * sigma, 1e-6)
        observation_probability = 1.0 / (1.0 + np.exp(np.clip(distance / scale, -60.0, 60.0)))
        prior = np.clip(belief.burn_probability, 1e-5, 1.0 - 1e-5)
        observed_probability = np.clip(observation_probability, 1e-5, 1.0 - 1e-5)
        prior_log_odds = np.log(prior / (1.0 - prior))
        observation_log_odds = np.log(observed_probability / (1.0 - observed_probability))
        posterior = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    (1.0 - confidence) * prior_log_odds + confidence * observation_log_odds,
                    -60.0,
                    60.0,
                )
            )
        )
        belief.burn_probability[:] = posterior.astype(np.float32)
        belief.known_burned[:] = (posterior >= 0.90).astype(np.float32)
        belief.intensity_mean[interior] = 0.0
        belief.intensity_mean[boundary] = np.maximum(belief.intensity_mean[boundary], 520.0)
        boundary_uncertainty = np.clip(35.0 + 90.0 * (1.0 - posterior), 35.0, 125.0)
        belief.intensity_std[observed] = boundary_uncertainty[observed]
        observed_support = np.abs(distance) <= 3.0 * sigma
        belief.observed_at[observed_support] = self.state.minute
        belief.arrival_time_mean[boundary] = float(self.state.minute)
        belief.arrival_time_std[boundary] = max(1.0, sigma / max(self.config.cell_size_m, 1e-6))
        belief.arrival_time_mean[interior] = np.minimum(
            belief.arrival_time_mean[interior], float(self.state.minute)
        )
        belief.perimeter_source = source
        self.state.event(
            "perimeter_assimilated",
            source=source,
            observed_cells=int(observed.sum()),
            boundary_cells=int(boundary.sum()),
            localization_sigma_m=sigma,
            confidence=confidence,
        )
        self.tasks = generate_tasks(self)

    def initialize_from_observed_perimeter(
        self, mask: np.ndarray, *, source: str = "historical-perimeter"
    ) -> None:
        """Set an episode's initial fire state from an observed perimeter."""

        if mask.shape != self.state.truth.phase.shape:
            raise ValueError(f"perimeter shape {mask.shape} does not match simulator grid")
        observed = mask.astype(np.bool_)
        boundary = self._perimeter_boundary(observed)
        interior = observed & ~boundary
        truth = self.state.truth
        truth.phase[:] = FirePhase.UNBURNED
        truth.phase[interior] = FirePhase.BURNED
        truth.phase[boundary] = FirePhase.FLAMING
        truth.intensity_kw_m[:] = 0.0
        truth.intensity_kw_m[boundary] = 720.0
        truth.fuel_remaining[:] = 1.0
        truth.fuel_remaining[interior] = 0.0
        truth.fuel_remaining[boundary] = 0.65
        truth.observed_burned[:] = 0.0
        truth.observed_burned[interior] = 1.0
        truth.water[:] = 0.0
        truth.retardant[:] = 0.0
        truth.ground_hold[:] = 0.0
        truth.water_coverage_gpc[:] = 0.0
        truth.retardant_coverage_gpc[:] = 0.0
        truth.retardant_effective_coverage_gpc[:] = 0.0
        truth.constructed_line[:] = 0.0
        truth.line_strength[:] = 0.0
        truth.line_status[:] = 0
        truth.fire_type[:] = FireType.UNBURNED
        truth.fire_type[boundary] = FireType.SURFACE
        truth.spread_rate_m_min[:] = 0.0
        truth.flame_length_m[:] = 0.0
        truth.ignition_progress[:] = 0.0
        truth.level_set_m[:] = signed_distance(observed, self.config.cell_size_m)
        truth.arrival_time_min[:] = np.inf
        truth.arrival_time_min[observed] = 0.0
        truth.burn_age_min[:] = 0.0
        truth.history_speed_m_min[:] = 0.0
        truth.history_head_x[:] = 0.0
        truth.history_head_y[:] = 0.0
        truth.history_confidence[:] = 0.0
        truth.history_heat_flux_kw_m2[:] = 0.0
        self.state.belief.intensity_mean[:] = 0.0
        self.state.belief.intensity_std[:] = 1.0
        self.state.belief.observed_at[:] = -9999
        self.state.belief.known_burned[:] = 0.0
        self.state.belief.burn_probability[:] = 0.0
        self.state.belief.arrival_time_mean[:] = np.inf
        self.state.belief.arrival_time_std[:] = np.inf
        self.state.belief.perimeter_source = None
        self.state.belief.pending.clear()
        self.state.terminated = False
        self.state.truncated = False
        self.state.escaped = False
        self.state.contained = False
        self.assimilate_observed_perimeter(observed, source=source)
        self.state.event("historical_initialization", source=source)
        self._notify_state_observers()

    def initialize_from_arrival_history(
        self,
        earlier_mask: np.ndarray,
        later_mask: np.ndarray,
        elapsed_min: float,
        *,
        source: str = "two-perimeter-arrival-history",
    ) -> dict[str, float | int]:
        """Initialize coupled fire state from two perimeter observations."""

        history = reconstruct_arrival_history(
            earlier_mask,
            later_mask,
            elapsed_min,
            self.config.cell_size_m,
        )
        observed = later_mask.astype(np.bool_)
        truth = self.state.truth
        residence = np.clip(
            18.0 + 0.18 * self.config.cell_size_m,
            18.0,
            90.0,
        )
        active = observed & (history.burn_age_min <= residence)
        active |= self._perimeter_boundary(observed)
        interior = observed & ~active

        truth.phase[:] = FirePhase.UNBURNED
        truth.phase[interior] = FirePhase.BURNED
        truth.phase[active] = FirePhase.FLAMING
        truth.burn_age_min[:] = history.burn_age_min
        truth.arrival_time_min[:] = history.arrival_time_min
        truth.fuel_remaining[:] = 1.0
        truth.fuel_remaining[observed] = np.exp(-history.burn_age_min[observed] / residence)
        truth.observed_burned[:] = 0.0
        truth.observed_burned[interior] = 1.0
        truth.intensity_kw_m[:] = 0.0
        truth.intensity_kw_m[active] = 720.0 * np.clip(
            history.heat_flux_fraction[active],
            0.15,
            1.0,
        )
        truth.fire_type[:] = FireType.UNBURNED
        truth.fire_type[active] = FireType.SURFACE
        truth.spread_rate_m_min[:] = 0.0
        truth.flame_length_m[:] = 0.0
        truth.ignition_progress[:] = 0.0
        truth.level_set_m[:] = signed_distance(
            observed,
            self.config.cell_size_m,
        )
        truth.water[:] = 0.0
        truth.retardant[:] = 0.0
        truth.ground_hold[:] = 0.0
        truth.water_coverage_gpc[:] = 0.0
        truth.retardant_coverage_gpc[:] = 0.0
        truth.retardant_effective_coverage_gpc[:] = 0.0
        truth.constructed_line[:] = 0.0
        truth.line_strength[:] = 0.0
        truth.line_status[:] = 0
        truth.history_speed_m_min[:] = history.speed_m_min
        truth.history_head_x[:] = history.head_x
        truth.history_head_y[:] = history.head_y
        truth.history_confidence[:] = history.confidence
        truth.history_heat_flux_kw_m2[:] = 65.0 * history.heat_flux_fraction

        belief = self.state.belief
        belief.intensity_mean[:] = 0.0
        belief.intensity_std[:] = 1.0
        belief.observed_at[:] = -9999
        belief.known_burned[:] = 0.0
        belief.burn_probability[:] = 0.0
        belief.arrival_time_mean[:] = np.inf
        belief.arrival_time_std[:] = np.inf
        belief.perimeter_source = None
        belief.pending.clear()
        self.state.terminated = False
        self.state.truncated = False
        self.state.escaped = False
        self.state.contained = False
        self.assimilate_observed_perimeter(observed, source=source)
        belief.arrival_time_mean[observed] = history.arrival_time_min[observed]
        belief.arrival_time_std[observed] = np.maximum(
            1.0,
            (1.0 - history.confidence[observed]) * elapsed_min * 0.25,
        )
        self.state.event(
            "arrival_history_initialized",
            source=source,
            **history.diagnostics,
        )
        self._notify_state_observers()
        return history.diagnostics

    def correct_advancing_front(
        self,
        observed_mask: np.ndarray,
        *,
        localization_radius_m: float | None = None,
        gain: float = 0.90,
        source: str = "sequential-perimeter-correction",
    ) -> dict[str, float | int]:
        """Apply a perimeter innovation in a localized level-set band."""

        radius = (
            4.0 * self.config.cell_size_m if localization_radius_m is None else float(localization_radius_m)
        )
        truth = self.state.truth
        correction = localize_front_correction(
            truth.level_set_m,
            observed_mask,
            self.config.cell_size_m,
            localization_radius_m=radius,
            gain=gain,
        )
        was_inside = truth.level_set_m <= 0.0
        now_inside = correction.corrected_level_set_m <= 0.0
        newly_inside = now_inside & ~was_inside
        newly_outside = was_inside & ~now_inside & (correction.localization_weight > 0.05)
        truth.level_set_m[:] = correction.corrected_level_set_m
        truth.phase[newly_inside] = FirePhase.FLAMING
        truth.fire_type[newly_inside] = FireType.SURFACE
        truth.fuel_remaining[newly_inside] = np.maximum(
            truth.fuel_remaining[newly_inside],
            0.65,
        )
        truth.arrival_time_min[newly_inside] = float(self.state.minute)
        truth.burn_age_min[newly_inside] = 0.0
        truth.intensity_kw_m[newly_inside] = np.maximum(
            truth.intensity_kw_m[newly_inside],
            320.0,
        )
        reversible_outside = newly_outside & (truth.phase == FirePhase.FLAMING)
        truth.phase[reversible_outside] = FirePhase.UNBURNED
        truth.fire_type[reversible_outside] = FireType.UNBURNED
        truth.intensity_kw_m[reversible_outside] = 0.0
        truth.arrival_time_min[reversible_outside] = np.inf
        truth.burn_age_min[reversible_outside] = 0.0
        self.assimilate_observed_perimeter(
            observed_mask,
            source=source,
            localization_sigma_m=radius,
            confidence=gain,
        )
        diagnostics = {
            **correction.diagnostics,
            "newly_inside_cells": int(newly_inside.sum()),
            "newly_outside_cells": int(reversible_outside.sum()),
        }
        self.state.event(
            "advancing_front_corrected",
            source=source,
            **diagnostics,
        )
        self._notify_state_observers()
        return diagnostics

    def observations(self) -> dict[str, dict[str, np.ndarray]]:
        task_values, valid = task_tensor(
            self.tasks, self.config.max_tasks, self.config.width, self.config.height
        )
        global_value = actor_global_features(self)
        result: dict[str, dict[str, np.ndarray]] = {}
        for resource in self.state.resources:
            result[resource.resource_id] = {
                "resource": resource_features(resource, self),
                "tasks": task_values.copy(),
                "action_mask": action_mask(resource, self.tasks, self.config.max_tasks, self),
                "task_valid": valid.copy(),
                "global": global_value.copy(),
            }
        return result

    def episode_record(self) -> dict[str, Any]:
        truth = self.state.truth
        return {
            "schema_version": 1,
            "scenario": asdict(self.config),
            "minute": self.state.minute,
            "escaped": self.state.escaped,
            "contained": self.state.contained,
            "truncated": self.state.truncated,
            "weighted_loss": self._weighted_loss(),
            "burned_fraction": float(truth.observed_burned.mean()),
            "active_fraction": float((truth.phase == FirePhase.FLAMING).mean()),
            "blocked_actions": self.state.blocked_actions,
            "resource": [
                {
                    "id": item.resource_id,
                    "accepted_tasks": item.accepted_tasks,
                    "attempted_tasks": item.attempted_tasks,
                    "reload_cycles": item.reload_cycles,
                    "flight_min": item.flight_min,
                    "endurance_remaining_min": item.endurance_remaining_min,
                    "payload_l": item.payload_l,
                    "current_site_id": item.current_site_id,
                    "status": item.status.name,
                }
                for item in self.state.resources
            ],
            "service_sites": [
                {
                    "id": site.site_id,
                    "remaining_volume_l": site.remaining_volume_l,
                    "active_bays": self._service_slots_used(site.site_id),
                }
                for site in self.state.service_sites
            ],
            "events": self.state.events,
            "rng_state": self.state.copy_rng_state(),
        }
