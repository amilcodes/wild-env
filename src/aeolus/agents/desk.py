"""A finite, partially observed incident desk around the canonical simulator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from aeolus.agents.contract import PROTOCOL_VERSION, digest, validate_dispatch
from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import action_mask, task_distance_min


class IncidentDesk:
    """Only observation() crosses the agent boundary; metrics() is evaluator-only.

    In-process adapters are trusted research code, not a security sandbox. External
    agents should receive serialized observations in a separately isolated process.
    """

    def __init__(self, config: ScenarioConfig, seed: int):
        self.sim = AeolusSimulator(config)
        self.sim.reset(seed)
        self.episode_id = digest({"scenario": asdict(config), "seed": seed})[:16]
        self.decision = 0
        self.memory = ""
        self.receipt: dict[str, Any] | None = None
        self.invalid_dispatches = 0
        self.total_reward = 0.0
        self.initial_loss = self.sim.episode_record()["weighted_loss"]

    @property
    def done(self) -> bool:
        return self.sim.state.terminated or self.sim.state.truncated

    def observation(self) -> dict[str, Any]:
        sim, state, config = self.sim, self.sim.state, self.sim.config
        belief = state.belief
        resources = []
        for resource in state.resources:
            mask = action_mask(resource, sim.tasks, config.max_tasks, sim)
            legal = [task.index for task in sim.tasks if mask[task.index]]
            resources.append(
                {
                    "id": resource.resource_id,
                    "kind": resource.spec.kind,
                    "status": resource.status.name,
                    "xy": [round(resource.x, 3), round(resource.y, 3)],
                    "payload_l": round(resource.payload_l, 3),
                    "payload_capacity_l": resource.spec.payload_l,
                    "endurance_remaining_min": round(resource.endurance_remaining_min, 3),
                    "reserve_endurance_min": resource.spec.reserve_endurance_min,
                    "eta_min": resource.eta_min,
                    "destination_xy": resource.target_xy,
                    "service_site_id": resource.service_site_id,
                    "legal_tasks": legal,
                    "travel_min": {
                        str(task.index): round(task_distance_min(resource, task, config.cell_size_m), 3)
                        for task in sim.tasks
                        if task.index in legal
                    },
                }
            )
        tasks = []
        for task in sim.tasks:
            observed_at = int(belief.observed_at[task.y, task.x])
            tasks.append(
                {
                    "id": task.index,
                    "kind": task.kind.name,
                    "xy": [task.x, task.y],
                    "capacity": task.capacity,
                    "heuristic_value": round(task.expected_value, 4),
                    "uncertainty": round(task.uncertainty, 4),
                    "ground_dependency": task.ground_dependency,
                    "heading_deg": task.heading_deg,
                    "service_site_id": task.service_site_id,
                    "belief_intensity_kw_m": round(float(belief.intensity_mean[task.y, task.x]), 3),
                    "observation_age_min": state.minute - observed_at if observed_at >= 0 else None,
                }
            )
        # Static asset geography is public; dynamic fire truth is never used here.
        sectors = []
        for y_indices in np.array_split(np.arange(config.height), 4):
            for x_indices in np.array_split(np.arange(config.width), 4):
                region = np.ix_(y_indices, x_indices)
                ages = belief.observed_at[region]
                known = ages >= 0
                sectors.append(
                    {
                        "bounds_xy": [
                            int(x_indices[0]),
                            int(y_indices[0]),
                            int(x_indices[-1]),
                            int(y_indices[-1]),
                        ],
                        "mapped_asset_value": round(float(state.truth.asset_value[region].sum()), 3),
                        "observed_fraction": round(float(known.mean()), 3),
                        "belief_burned_fraction": round(float(belief.known_burned[region].mean()), 3),
                        "belief_peak_intensity_kw_m": round(float(belief.intensity_mean[region].max()), 3),
                        "mean_observation_age_min": (
                            round(float((state.minute - ages[known]).mean()), 2) if known.any() else None
                        ),
                    }
                )
        return {
            "protocol": PROTOCOL_VERSION,
            "decision_id": f"{self.episode_id}:{self.decision}",
            "minute": state.minute,
            "horizon_min": config.horizon_min,
            "decision_interval_min": config.decision_interval_min,
            "observation_delay_min": config.observation_delay_min,
            "map": {"width": config.width, "height": config.height, "cell_size_m": config.cell_size_m},
            "weather_current_domain_mean": {
                name: round(float(np.asarray(value).mean()), 4)
                for name, value in sim.current_weather().items()
                if name
                in {
                    "wind_speed_m_s",
                    "wind_direction_deg",
                    "air_temperature_c",
                    "relative_humidity_pct",
                    "precipitation_rate_mm_h",
                }
            },
            "ground_engaged": state.ground_engaged,
            "resources": resources,
            "tasks": tasks,
            "sectors": sectors,
            "service_sites": [
                {
                    "id": site.site_id,
                    "kind": site.spec.kind,
                    "xy": [site.spec.x, site.spec.y],
                    "services": list(site.spec.services),
                    "mode": site.spec.service_mode,
                    "remaining_volume_l": round(site.remaining_volume_l, 3),
                    "bays": site.spec.bays,
                    "open_minute": site.spec.open_minute,
                    "close_minute": site.spec.close_minute,
                }
                for site in state.service_sites
            ],
            "memory": self.memory,
            "last_dispatch": self.receipt,
        }

    def step(self, response: Any, *, failure: str | None = None) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("episode is already complete")
        observation = self.observation()
        error = failure
        actions = {item.resource_id: 0 for item in self.sim.state.resources}
        if error is None:
            try:
                actions = validate_dispatch(response, observation)
            except ValueError as exc:
                error = str(exc)
        if error:
            self.invalid_dispatches += 1
        else:
            self.memory = response["memory"]
        _, reward, _, _, infos = self.sim.decision_step(actions)
        self.total_reward += reward
        self.receipt = {
            "decision_id": observation["decision_id"],
            "valid": error is None,
            "error": error,
            "actions": actions,
            # Explicit allowlist: simulator infos also contain hidden evaluator metrics.
            "assignments": {
                resource_id: {
                    key: value
                    for key, value in info.items()
                    if key in {"accepted", "reason", "task", "eta_min"}
                }
                for resource_id, info in infos.items()
            },
        }
        self.decision += 1
        return self.receipt

    def metrics(self) -> dict[str, Any]:
        record = self.sim.episode_record()
        config = self.sim.config
        return {
            "minute": record["minute"],
            "weighted_loss": record["weighted_loss"],
            "initial_weighted_loss": self.initial_loss,
            "loss_growth": record["weighted_loss"] - self.initial_loss,
            "burned_area_ha": record["burned_fraction"]
            * config.width
            * config.height
            * config.cell_size_m**2
            / 10000.0,
            "contained": record["contained"],
            "escaped": record["escaped"],
            "simulator_blocked_actions": record["blocked_actions"],
            "invalid_dispatches": self.invalid_dispatches,
            "decisions": self.decision,
            "resource_cost_units": self.sim.state.cumulative_cost,
            "exposure_units": self.sim.state.cumulative_exposure,
            "reward_return": self.total_reward,
        }
