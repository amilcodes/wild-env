from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from aeolus.config import ResourceSpec, ScenarioConfig, ServiceSiteSpec
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import ResourceStatus
from aeolus.core.tasks import TaskKind
from aeolus.data import ScenarioBundle, load_service_sites_geojson
from aeolus.envs.tensor_operations import (
    TensorOperationsEnv,
    TensorResourceStatus,
)
from aeolus.policies import cycle_time_greedy
from aeolus.training.networks import (
    EntityAttentionActorCritic,
    TaskPointerActorCritic,
)
from aeolus.training.rollout import TensorOperationsCollector


def operations_config(
    *,
    bays: int = 1,
    site_stock_l: float = 40_000.0,
    approach_capacity: int = 3,
) -> ScenarioConfig:
    sites = (
        ServiceSiteSpec(
            "regional_base",
            "airport",
            4,
            42,
            ("water", "retardant", "fuel"),
            "land",
            bays=2,
            approach_capacity=3,
            refill_rate_l_min=1800.0,
            fixed_turnaround_min=3.0,
            manually_verified=True,
        ),
        ServiceSiteSpec(
            "reservoir_dip",
            "dip_site",
            31,
            27,
            ("water",),
            "hover_fill",
            bays=bays,
            approach_capacity=approach_capacity,
            refill_rate_l_min=5000.0,
            fixed_turnaround_min=1.0,
            available_volume_l=site_stock_l,
            manually_verified=True,
        ),
    )
    resources = (
        ResourceSpec(
            "water_uav_0",
            "water",
            52.0,
            2500.0,
            5,
            0,
            100,
            home_site_id="regional_base",
            service_modes=("land", "hover_fill"),
        ),
        ResourceSpec(
            "water_uav_1",
            "water",
            48.0,
            2500.0,
            5,
            0,
            100,
            home_site_id="regional_base",
            service_modes=("land", "hover_fill"),
        ),
        ResourceSpec(
            "retardant_uav",
            "retardant",
            62.0,
            6000.0,
            9,
            0,
            150,
            home_site_id="regional_base",
        ),
    )
    return ScenarioConfig(
        seed=71,
        width=48,
        height=48,
        horizon_min=120,
        decision_interval_min=1,
        max_tasks=40,
        wind_speed_m_s=2.0,
        spotting_rate=0.0,
        resources=resources,
        service_sites=sites,
    )


def test_canonical_service_routing_preserves_dip_site_endurance() -> None:
    simulator = AeolusSimulator(operations_config())
    water = simulator.state.resources[0]
    line_task = next(task for task in simulator.tasks if task.kind == TaskKind.AERIAL_LINE)
    simulator.decision_step(
        {
            "water_uav_0": line_task.index,
            "water_uav_1": 0,
            "retardant_uav": 0,
        }
    )
    while water.status != ResourceStatus.AVAILABLE or water.payload_fraction > 0.0:
        simulator.decision_step({"water_uav_0": 0, "water_uav_1": 0, "retardant_uav": 0})
    endurance_used_before_dip = water.flight_min
    dip_task = next(
        task
        for task in simulator.tasks
        if task.kind == TaskKind.SERVICE and task.service_site_id == "reservoir_dip"
    )
    assert simulator.observations()["water_uav_0"]["action_mask"][dip_task.index]
    simulator.decision_step(
        {
            "water_uav_0": dip_task.index,
            "water_uav_1": 0,
            "retardant_uav": 0,
        }
    )
    while water.status != ResourceStatus.AVAILABLE:
        simulator.decision_step({"water_uav_0": 0, "water_uav_1": 0, "retardant_uav": 0})
    assert water.payload_fraction == 1.0
    assert water.current_site_id == "reservoir_dip"
    assert water.flight_min > endurance_used_before_dip
    assert any(event["kind"] == "service_started" for event in simulator.state.events)


def test_canonical_site_queue_and_finite_stock_are_conserved() -> None:
    simulator = AeolusSimulator(operations_config(bays=1, site_stock_l=5000.0))
    dip_task = next(
        task
        for task in simulator.tasks
        if task.kind == TaskKind.SERVICE and task.service_site_id == "reservoir_dip"
    )
    for resource in simulator.state.resources[:2]:
        resource.payload_fraction = 0.0
    simulator.decision_step(
        {
            "water_uav_0": dip_task.index,
            "water_uav_1": dip_task.index,
            "retardant_uav": 0,
        }
    )
    for _ in range(30):
        simulator.decision_step({"water_uav_0": 0, "water_uav_1": 0, "retardant_uav": 0})
        if all(resource.payload_fraction == 1.0 for resource in simulator.state.resources[:2]):
            break
    site = next(site for site in simulator.state.service_sites if site.site_id == "reservoir_dip")
    assert site.remaining_volume_l == 0.0
    assert any(event["kind"] == "service_queued" for event in simulator.state.events)
    assert all(resource.payload_fraction == 1.0 for resource in simulator.state.resources[:2])


def test_tensor_environment_stays_fixed_shape_and_cycle_policy_services_loads() -> None:
    environment = TensorOperationsEnv(
        operations_config(),
        batch_size=8,
        max_segments=6,
        device="cpu",
    )
    observation = environment.reset(seed=9)
    assert observation.resource.shape == (8, 3, 17)
    assert observation.tasks.shape == (8, 9, 21)
    assert observation.action_mask.shape == (8, 3, 9)
    initial_remaining = environment.state.segment_remaining_l.sum().item()
    for _ in range(45):
        transition = environment.step(cycle_time_greedy(environment))
        if transition.done.all():
            break
    assert environment.state.segment_remaining_l.sum().item() < initial_remaining
    assert torch.isfinite(transition.reward).all()
    assert environment.state.site_remaining_l.min() >= 0.0


def test_tensor_collector_and_entity_policy_do_not_cross_host_contract() -> None:
    config = operations_config()
    model = EntityAttentionActorCritic(
        48,
        attention_heads=4,
        attention_layers=1,
    )
    collector = TensorOperationsCollector(
        config,
        num_envs=4,
        seed=13,
        device=torch.device("cpu"),
        hidden_dim=48,
        max_segments=5,
    )
    rollout = collector.collect(model, steps=4)
    assert rollout.resource.shape == (4, 4, 3, 17)
    assert rollout.tasks.shape == (4, 4, 8, 21)
    assert rollout.rewards.device.type == "cpu"
    assert torch.isfinite(rollout.logp).all()
    assert torch.isfinite(rollout.values).all()


def test_capacity_aware_sampler_prevents_duplicate_unit_capacity_action() -> None:
    logits = torch.tensor([[[0.0, 9.0], [0.0, 9.0]]])
    tasks = torch.zeros((1, 2, 21))
    tasks[:, 0, 7] = 1.0
    tasks[:, 1, 7] = 1.0 / 16.0
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    actions, _, _ = TaskPointerActorCritic._capacity_aware_actions(
        logits,
        tasks,
        mask,
        deterministic=True,
    )
    assert (actions == 1).sum() == 1
    assert (actions == 0).sum() == 1


def test_tensor_queue_capacity_is_enforced() -> None:
    environment = TensorOperationsEnv(
        operations_config(bays=1),
        batch_size=2,
        max_segments=2,
    )
    environment.reset(seed=21)
    state = environment.state
    state.resource_payload_fraction[:, :2] = 0.0
    state.resource_status[:, :2] = int(TensorResourceStatus.QUEUED)
    state.resource_site_index[:, :2] = 1
    state.resource_target_index[:, :2] = 1
    environment._admit_queues()
    active = (state.resource_status[:, :2] == int(TensorResourceStatus.SERVICING)).sum(dim=1)
    queued = (state.resource_status[:, :2] == int(TensorResourceStatus.QUEUED)).sum(dim=1)
    assert torch.equal(active, torch.ones_like(active))
    assert torch.equal(queued, torch.ones_like(queued))


def test_hover_service_burns_endurance_and_land_service_does_not() -> None:
    environment = TensorOperationsEnv(
        operations_config(bays=1),
        batch_size=1,
        max_segments=2,
    )
    state = environment.state
    state.resource_status[0, 0] = int(TensorResourceStatus.QUEUED)
    state.resource_site_index[0, 0] = 1
    state.resource_target_index[0, 0] = 1
    state.resource_payload_fraction[0, 0] = 0.0
    state.resource_status[0, 2] = int(TensorResourceStatus.QUEUED)
    state.resource_site_index[0, 2] = 0
    state.resource_target_index[0, 2] = 0
    state.resource_payload_fraction[0, 2] = 0.0
    before = state.resource_endurance_remaining_min.clone()
    environment.step(torch.zeros((1, 3), dtype=torch.long))
    assert state.resource_endurance_remaining_min[0, 0] == before[0, 0] - 1.0
    assert state.resource_endurance_remaining_min[0, 2] == before[0, 2]


def test_total_approach_capacity_and_dispatch_reserve_are_hard_constraints() -> None:
    constrained = operations_config(approach_capacity=1)
    environment = TensorOperationsEnv(
        constrained,
        batch_size=1,
        max_segments=2,
    )
    environment.state.resource_payload_fraction[0, :2] = 0.0
    dip_action = 1 + environment.max_segments + 1
    transition = environment.step(torch.tensor([[dip_action, dip_action, 0]]))
    assert transition.blocked_actions.item() == 1
    available_water = environment.state.resource_status[0, :2] == int(TensorResourceStatus.AVAILABLE)
    assert available_water.sum() == 1
    assert not environment.action_mask()[0, :2, dip_action].any()

    slow_launch = replace(
        constrained.resources[0],
        dispatch_latency_min=90,
    )
    reserve_config = replace(
        constrained,
        resources=(slow_launch, *constrained.resources[1:]),
    )
    simulator = AeolusSimulator(reserve_config)
    attack_indices = [
        task.index for task in simulator.tasks if task.kind in (TaskKind.WATER, TaskKind.AERIAL_LINE)
    ]
    mask = simulator.observations()[slow_launch.resource_id]["action_mask"]
    assert attack_indices
    assert not mask[attack_indices].any()
    tensor_environment = TensorOperationsEnv(
        reserve_config,
        batch_size=1,
        max_segments=2,
    )
    assert not tensor_environment.action_mask()[0, 0, 1:3].any()


def test_verified_service_site_geojson_maps_through_bundle_affine(tmp_path) -> None:
    shape = (32, 32)
    scenario = ScenarioBundle(
        elevation_m=np.zeros(shape, dtype=np.float32),
        fuel_load_kg_m2=np.ones(shape, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": "EPSG:32610",
            "cell_size_m": 60.0,
            "sources": [],
            "transformations": [],
            "split": "test",
            "transform": [60.0, 0.0, 500000.0, 0.0, -60.0, 4200000.0],
        },
    )
    source = tmp_path / "sites.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [500600.0, 4199400.0],
                        },
                        "properties": {
                            "site_id": "verified_dip",
                            "kind": "dip_site",
                            "services": ["water"],
                            "service_mode": "hover_fill",
                            "bays": 1,
                            "refill_rate_l_min": 3000.0,
                            "manually_verified": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sites = load_service_sites_geojson(source, scenario)
    assert (sites[0].x, sites[0].y) == (10, 10)
