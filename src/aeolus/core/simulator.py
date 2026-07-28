"""Event-driven truth simulator behind all Aeolus environment adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from math import ceil, hypot
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.config import ScenarioConfig
from aeolus.core.fire import apply_retardant, apply_water, step_fire
from aeolus.core.state import (
    BeliefState,
    EpisodeState,
    FirePhase,
    FireType,
    PendingObservation,
    ResourceRuntime,
    ResourceStatus,
    TruthState,
)
from aeolus.core.tasks import (
    Task,
    TaskKind,
    action_mask,
    actor_global_features,
    generate_tasks,
    resource_features,
    task_distance_min,
    task_tensor,
)
from aeolus.data import IncidentBundle, WeatherForcing, load_bundle


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
        self.state: EpisodeState
        self.tasks: list[Task] = []
        self._state_observers: list[Callable[[AeolusSimulator], None]] = []
        self.reset(config.seed)

    def _load_weather_forcing(self) -> WeatherForcing | None:
        weather_path = Path(self.config.weather_forcing) if self.config.weather_forcing else None
        if weather_path is None and self.config.landscape_bundle:
            landscape_path = Path(self.config.landscape_bundle)
            if landscape_path.is_dir() and (landscape_path / "item.json").exists():
                weather_path = IncidentBundle.load(landscape_path).asset_path(
                    "weather", required=False
                )
        return WeatherForcing.load(weather_path) if weather_path is not None else None

    def current_weather(self) -> dict[str, float]:
        """Return forcing values at the current simulation minute."""

        if self.weather is not None:
            return self.weather.at_minute(self.state.minute)
        return {
            "wind_speed_m_s": float(self.config.wind_speed_m_s),
            "wind_direction_deg": float(self.config.wind_direction_deg),
            "air_temperature_c": float(self.config.air_temperature_c),
            "relative_humidity_pct": float(self.config.relative_humidity_pct),
            "precipitation_rate_mm_h": float(
                self.config.precipitation_rate_mm_h
            ),
        }

    @property
    def agent_ids(self) -> list[str]:
        return [resource.resource_id for resource in self.state.resources]

    def reset(self, seed: int | None = None) -> dict[str, dict[str, np.ndarray]]:
        episode_seed = self.config.seed if seed is None else int(seed)
        rng = np.random.default_rng(episode_seed)
        truth, base_xy = self._build_truth(rng)
        belief = BeliefState(
            intensity_mean=np.zeros_like(truth.intensity_kw_m, dtype=np.float32),
            intensity_std=np.full_like(truth.intensity_kw_m, 1.0, dtype=np.float32),
            observed_at=np.full(truth.intensity_kw_m.shape, -9999, dtype=np.int32),
            known_burned=np.zeros_like(truth.intensity_kw_m, dtype=np.float32),
        )
        resources = [
            ResourceRuntime(spec=spec, x=float(base_xy[0]), y=float(base_xy[1]))
            for spec in self.config.resources
        ]
        self.state = EpisodeState(
            minute=0,
            truth=truth,
            belief=belief,
            resources=resources,
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
            fuel_model_number = np.full(
                (height, width), self.config.fuel.standard_number, dtype=np.int16
            )
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
                0.20
                + 0.38 * np.sin((x + 2 * y) / 29.0)
                + 0.22 * np.cos((2 * x - y) / 21.0),
                0.0,
                0.82,
            ).astype(np.float32)
            canopy_cover = np.where(fuel_multiplier > 0.88, woodland, 0.0).astype(
                np.float32
            )
            canopy_height = np.where(
                canopy_cover >= 0.20, 8.0 + 10.0 * canopy_cover, 0.0
            ).astype(np.float32)
            canopy_base_height = np.where(
                canopy_cover >= 0.20, 2.1 + 2.5 * (1.0 - canopy_cover), 0.0
            ).astype(np.float32)
            canopy_bulk_density = np.where(
                canopy_cover >= 0.20, 0.06 + 0.17 * canopy_cover, 0.0
            ).astype(np.float32)
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
            fire_type=np.where(
                ignition, FireType.SURFACE, FireType.UNBURNED
            ).astype(np.uint8),
            spread_rate_m_min=np.zeros((height, width), dtype=np.float32),
            flame_length_m=np.zeros((height, width), dtype=np.float32),
            ignition_progress=np.zeros((height, width), dtype=np.float32),
            arrival_time_min=np.where(ignition, 0.0, np.inf).astype(np.float32),
            burn_age_min=np.zeros((height, width), dtype=np.float32),
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
                burned = belief.known_burned[y0:y1, x0:x1]
                burned[circle] = np.maximum(burned[circle], item.burned_measurement[circle])
                belief.known_burned[y0:y1, x0:x1] = burned
            delivered.append(item)
            self.state.event("observation_delivered", source=item.source, x=item.x, y=item.y)
        if delivered:
            belief.pending = [item for item in belief.pending if item not in delivered]

    def _advance_resource(self, resource: ResourceRuntime) -> None:
        if resource.status == ResourceStatus.AVAILABLE or resource.status == ResourceStatus.WITHDRAWN:
            return
        resource.eta_min = max(0, resource.eta_min - 1)
        if (
            resource.status in (ResourceStatus.OUTBOUND, ResourceStatus.RETURNING)
            and resource.leg_start_xy is not None
            and resource.leg_end_xy is not None
            and resource.leg_total_min > 0
        ):
            fraction = 1.0 - resource.eta_min / resource.leg_total_min
            resource.x = resource.leg_start_xy[0] + (
                resource.leg_end_xy[0] - resource.leg_start_xy[0]
            ) * fraction
            resource.y = resource.leg_start_xy[1] + (
                resource.leg_end_xy[1] - resource.leg_start_xy[1]
            ) * fraction
        if resource.eta_min > 0:
            return
        if resource.status == ResourceStatus.OUTBOUND:
            assert resource.target_xy is not None
            self._execute_mission(resource)
            return
        if resource.status == ResourceStatus.RETURNING:
            resource.x, resource.y = self.state.base_xy
            if resource.spec.reload_min:
                resource.status = ResourceStatus.RELOADING
                resource.eta_min = resource.spec.reload_min
            else:
                resource.status = ResourceStatus.AVAILABLE
                resource.target_xy = None
                resource.task_kind = int(TaskKind.HOLD)
                resource.leg_start_xy = None
                resource.leg_end_xy = None
                resource.leg_total_min = 0
                self.state.event("resource_available", resource=resource.resource_id)
            return
        if resource.status == ResourceStatus.RELOADING:
            resource.status = ResourceStatus.AVAILABLE
            resource.target_xy = None
            resource.task_kind = int(TaskKind.HOLD)
            resource.leg_start_xy = None
            resource.leg_end_xy = None
            resource.leg_total_min = 0
            resource.payload_fraction = 1.0
            resource.reload_cycles += 1
            self.state.event("resource_available", resource=resource.resource_id)

    def _execute_mission(self, resource: ResourceRuntime) -> None:
        assert resource.target_xy is not None
        task_kind = TaskKind(resource.task_kind)
        x, y = resource.target_xy
        resource.x, resource.y = x, y
        if task_kind == TaskKind.OBSERVE:
            self._capture_observation(x, y, 9, resource.resource_id)
            self.state.event("observe", resource=resource.resource_id, x=x, y=y)
        elif task_kind == TaskKind.WATER:
            radius = max(1.5, resource.spec.water_radius_m / self.config.cell_size_m)
            apply_water(self.state.truth, x, y, radius)
            resource.payload_fraction = 0.0
            self.state.cumulative_cost += 0.8
            self.state.event("water_drop", resource=resource.resource_id, x=x, y=y)
        elif task_kind in (TaskKind.RETARDANT, TaskKind.REINFORCE):
            weather = self.current_weather()
            apply_retardant(
                self.state.truth,
                x,
                y,
                resource.spec.retardant_length_m / self.config.cell_size_m,
                resource.spec.retardant_width_m / self.config.cell_size_m,
                weather["wind_direction_deg"],
                self.state.ground_engaged,
            )
            resource.payload_fraction = 0.0
            self.state.cumulative_cost += 2.4
            self.state.event("retardant_drop", resource=resource.resource_id, x=x, y=y)
        distance_back = hypot(x - self.state.base_xy[0], y - self.state.base_xy[1]) * self.config.cell_size_m
        resource.status = ResourceStatus.RETURNING
        resource.eta_min = max(1, ceil(distance_back / max(resource.spec.cruise_speed_m_s * 60.0, 1.0)))
        resource.leg_start_xy = (float(x), float(y))
        resource.leg_end_xy = (
            float(self.state.base_xy[0]),
            float(self.state.base_xy[1]),
        )
        resource.leg_total_min = resource.eta_min

    def _advance_internal_minute(self) -> None:
        self.state.minute += 1
        self.state.ground_engaged = self.state.minute >= self.config.ground_arrival_min
        for resource in self.state.resources:
            self._advance_resource(resource)
            if resource.status not in (ResourceStatus.WITHDRAWN, ResourceStatus.AVAILABLE):
                self.state.cumulative_exposure += 0.04
                resource.flight_min += 1.0
                if resource.flight_min >= resource.spec.endurance_min:
                    resource.status = ResourceStatus.WITHDRAWN
                    resource.eta_min = 0
                    self.state.event("resource_withdrawn", resource=resource.resource_id)
        weather = self.current_weather()
        new_ignitions = step_fire(
            self.state.truth,
            self.config,
            self.state.rng,
            self.state.minute,
            wind_speed_m_s=weather["wind_speed_m_s"] if self.weather is not None else None,
            wind_direction_deg=(
                weather["wind_direction_deg"] if self.weather is not None else None
            ),
            air_temperature_c=weather["air_temperature_c"],
            relative_humidity_pct=weather["relative_humidity_pct"],
            precipitation_rate_mm_h=weather["precipitation_rate_mm_h"],
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
            mask = action_mask(resource, self.tasks, self.config.max_tasks)
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
            travel_min = task_distance_min(resource, task, self.config.cell_size_m)
            resource.target_xy = (task.x, task.y)
            resource.task_index = action
            resource.task_kind = int(task.kind)
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
            }
            self.state.event(
                "assignment", resource=resource.resource_id, task=task.kind.name, x=task.x, y=task.y
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
        neighbors_inside = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        return mask.astype(np.bool_) & ~neighbors_inside

    def assimilate_observed_perimeter(
        self, mask: np.ndarray, *, source: str = "historical-perimeter"
    ) -> None:
        """Update the policy belief from a cumulative observed perimeter."""

        if mask.shape != self.state.truth.phase.shape:
            raise ValueError(f"perimeter shape {mask.shape} does not match simulator grid")
        observed = mask.astype(np.bool_)
        boundary = self._perimeter_boundary(observed)
        interior = observed & ~boundary
        belief = self.state.belief
        belief.known_burned[interior] = 1.0
        belief.intensity_mean[interior] = 0.0
        belief.intensity_mean[boundary] = np.maximum(belief.intensity_mean[boundary], 520.0)
        belief.intensity_std[observed] = 55.0
        belief.observed_at[observed] = self.state.minute
        self.state.event(
            "perimeter_assimilated",
            source=source,
            observed_cells=int(observed.sum()),
            boundary_cells=int(boundary.sum()),
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
        truth.fire_type[:] = FireType.UNBURNED
        truth.fire_type[boundary] = FireType.SURFACE
        truth.spread_rate_m_min[:] = 0.0
        truth.flame_length_m[:] = 0.0
        truth.ignition_progress[:] = 0.0
        truth.arrival_time_min[:] = np.inf
        truth.arrival_time_min[observed] = 0.0
        truth.burn_age_min[:] = 0.0
        self.state.belief.intensity_mean[:] = 0.0
        self.state.belief.intensity_std[:] = 1.0
        self.state.belief.observed_at[:] = -9999
        self.state.belief.known_burned[:] = 0.0
        self.state.belief.pending.clear()
        self.state.terminated = False
        self.state.truncated = False
        self.state.escaped = False
        self.state.contained = False
        self.assimilate_observed_perimeter(observed, source=source)
        self.state.event("historical_initialization", source=source)
        self._notify_state_observers()

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
                "action_mask": action_mask(resource, self.tasks, self.config.max_tasks),
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
                    "status": item.status.name,
                }
                for item in self.state.resources
            ],
            "events": self.state.events,
            "rng_state": self.state.copy_rng_state(),
        }
