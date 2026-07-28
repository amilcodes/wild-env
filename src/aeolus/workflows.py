"""Shared construction helpers for command-line research workflows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aeolus.config import ResourceSpec, ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle
from aeolus.policies import (
    anchor_flank,
    greedy_value,
    joint_assignment,
    nearest_feasible,
    no_aerial_action,
)

Policy = Callable[[AeolusSimulator], dict[str, int]]

INITIAL_ATTACK_FLEET: tuple[ResourceSpec, ...] = (
    ResourceSpec("tanker_12", "retardant", 70.0, 11000.0, 14, 3, 210),
    ResourceSpec("tanker_21", "retardant", 66.0, 9000.0, 15, 4, 220),
    ResourceSpec("seat_03", "retardant", 58.0, 3000.0, 10, 2, 180),
    ResourceSpec("heli_07", "water", 48.0, 2800.0, 8, 2, 150),
    ResourceSpec("heli_14", "water", 44.0, 2200.0, 7, 2, 145),
    ResourceSpec("ir_scout", "sensor", 38.0, 0.0, 0, 1, 120),
    ResourceSpec("uas_mapper", "sensor", 27.0, 0.0, 0, 1, 105),
)


def scenario_from_incident(
    incident: IncidentBundle,
    *,
    seed: int = 20260726,
    horizon_min: int = 240,
    decision_interval_min: int = 3,
    max_tasks: int = 64,
    wind_speed_m_s: float = 4.0,
    wind_direction_deg: float = 25.0,
    spotting_rate: float = 0.003,
) -> ScenarioConfig:
    landscape = incident.scenario_bundle()
    height, width = landscape.elevation_m.shape
    return ScenarioConfig(
        seed=seed,
        width=width,
        height=height,
        cell_size_m=float(landscape.metadata["cell_size_m"]),
        horizon_min=horizon_min,
        decision_interval_min=decision_interval_min,
        max_tasks=max_tasks,
        wind_speed_m_s=wind_speed_m_s,
        wind_direction_deg=wind_direction_deg,
        spotting_rate=spotting_rate,
        landscape_bundle=str(incident.root.resolve()),
        resources=INITIAL_ATTACK_FLEET,
    )


def resolve_policy(
    name: str,
    *,
    checkpoint: str | Path | None = None,
) -> tuple[Policy, str | None]:
    policies: dict[str, Policy] = {
        "no_aerial": no_aerial_action,
        "nearest": nearest_feasible,
        "anchor_flank": anchor_flank,
        "greedy_value": greedy_value,
        "joint_assignment": joint_assignment,
    }
    if name in policies:
        return policies[name], None
    if name != "mappo":
        raise KeyError(f"unknown policy {name!r}; choices are {sorted([*policies, 'mappo'])}")
    if checkpoint is None:
        raise ValueError("the mappo policy requires --checkpoint")

    import torch

    from aeolus.evaluation.evaluate import learned_policy
    from aeolus.training.networks import TaskPointerActorCritic

    checkpoint_path = Path(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = int(payload.get("config", {}).get("training", {}).get("hidden_dim", 192))
    model = TaskPointerActorCritic(hidden_dim).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return learned_policy(model, device), str(checkpoint_path.resolve())
