from __future__ import annotations

import json

import numpy as np

from aeolus.config import (
    AirspaceVolumeSpec,
    ResourceSpec,
    ScenarioConfig,
    ServiceSiteSpec,
)
from aeolus.core.aviation import (
    density_altitude_m,
    evaluate_leg_performance,
    load_tactical_performance_surface,
    segment_intersects_polygon,
)
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import ResourceRuntime
from aeolus.core.tasks import TaskKind


def _performance_surface(tmp_path):
    path = tmp_path / "performance.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "density_altitude_m": [0.0, 2500.0, 5000.0],
                "payload_fraction": [0.0, 0.5, 1.0],
                "true_airspeed_m_s": [
                    [60.0, 58.0, 55.0],
                    [56.0, 53.0, 49.0],
                    [50.0, 46.0, 41.0],
                ],
                "endurance_multiplier": [
                    [1.08, 1.00, 0.92],
                    [1.00, 0.92, 0.84],
                    [0.90, 0.82, 0.72],
                ],
                "maximum_payload_fraction": [1.0, 0.85, 0.55],
                "metadata": {
                    "source": "unit-test flight-manual surrogate",
                    "vehicle": "test-only",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _resource(**updates) -> ResourceRuntime:
    values = {
        "resource_id": "aircraft",
        "kind": "water",
        "cruise_speed_m_s": 55.0,
        "payload_l": 2500.0,
        "reload_min": 5,
        "dispatch_latency_min": 0,
        "endurance_min": 120,
        "maximum_crosswind_m_s": 12.0,
    }
    values.update(updates)
    return ResourceRuntime(ResourceSpec(**values), 2.0, 10.0)


def test_density_altitude_and_performance_surface_payload_gate(tmp_path) -> None:
    path = _performance_surface(tmp_path)
    surface = load_tactical_performance_surface(path)
    assert surface.metadata["vehicle"] == "test-only"
    assert density_altitude_m(2500.0, 35.0) > 3500.0

    resource = _resource(performance_surface_path=str(path))
    elevation = np.full((20, 20), 2500.0)
    leg = evaluate_leg_performance(
        resource,
        start_xy=(2.0, 10.0),
        end_xy=(18.0, 10.0),
        cell_size_m=100.0,
        elevation_m=elevation,
        air_temperature_c=35.0,
        wind_speed_m_s=0.0,
        wind_from_direction_deg=0.0,
        minute=0.0,
    )
    assert leg.density_altitude_m > 3500.0
    assert leg.maximum_payload_fraction < 0.85
    assert "density_altitude_payload" in leg.violations


def test_wind_components_change_leg_time_and_crosswind_is_checked() -> None:
    resource = _resource()
    elevation = np.zeros((20, 20))
    calm = evaluate_leg_performance(
        resource,
        start_xy=(2.0, 10.0),
        end_xy=(18.0, 10.0),
        cell_size_m=100.0,
        elevation_m=elevation,
        air_temperature_c=20.0,
        wind_speed_m_s=0.0,
        wind_from_direction_deg=270.0,
        minute=0.0,
    )
    tailwind = evaluate_leg_performance(
        resource,
        start_xy=(2.0, 10.0),
        end_xy=(18.0, 10.0),
        cell_size_m=100.0,
        elevation_m=elevation,
        air_temperature_c=20.0,
        wind_speed_m_s=10.0,
        wind_from_direction_deg=270.0,
        minute=0.0,
    )
    crosswind = evaluate_leg_performance(
        resource,
        start_xy=(2.0, 10.0),
        end_xy=(18.0, 10.0),
        cell_size_m=100.0,
        elevation_m=elevation,
        air_temperature_c=20.0,
        wind_speed_m_s=15.0,
        wind_from_direction_deg=0.0,
        minute=0.0,
    )
    assert tailwind.travel_min < calm.travel_min
    assert "crosswind" in crosswind.violations


def test_airspace_volume_masks_crossing_attack_routes() -> None:
    volume = AirspaceVolumeSpec(
        "temporary-flight-restriction",
        ((10.0, 0.0), (20.0, 0.0), (20.0, 31.0), (10.0, 31.0)),
        0.0,
        5000.0,
        start_minute=0,
        end_minute=120,
    )
    resource = ResourceSpec(
        "blocked_tanker",
        "retardant",
        70.0,
        10_000.0,
        10,
        0,
        180,
    )
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=32,
            height=32,
            max_tasks=24,
            spotting_rate=0.0,
            resources=(resource,),
            airspace_volumes=(volume,),
        )
    )
    attack_indices = [
        task.index
        for task in simulator.tasks
        if task.kind
        in {
            TaskKind.RETARDANT,
            TaskKind.REINFORCE,
            TaskKind.AERIAL_LINE,
        }
    ]
    mask = simulator.observations()["blocked_tanker"]["action_mask"]
    assert attack_indices
    crossing_indices = [
        task.index
        for task in simulator.tasks
        if task.index in attack_indices
        and segment_intersects_polygon(
            (
                simulator.state.resources[0].x,
                simulator.state.resources[0].y,
            ),
            (float(task.x), float(task.y)),
            volume.polygon_xy,
        )
    ]
    assert crossing_indices
    assert not mask[crossing_indices].any()
    assert segment_intersects_polygon(
        (6.0, 25.0),
        (24.0, 16.0),
        volume.polygon_xy,
    )


def test_service_site_depth_and_length_requirements_are_hard_constraints() -> None:
    site = ServiceSiteSpec(
        "shallow_reservoir",
        "scoopable_water",
        10,
        10,
        ("water",),
        "scoop",
        minimum_depth_m=1.0,
        minimum_length_m=500.0,
        manually_verified=True,
    )
    resource = ResourceSpec(
        "scooper",
        "water",
        60.0,
        5000.0,
        8,
        0,
        120,
        home_site_id="shallow_reservoir",
        service_modes=("scoop",),
        minimum_service_depth_m=1.5,
        minimum_service_length_m=800.0,
    )
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=32,
            height=32,
            max_tasks=24,
            spotting_rate=0.0,
            resources=(resource,),
            service_sites=(site,),
        )
    )
    runtime = simulator.state.resources[0]
    runtime.payload_fraction = 0.0
    service_task = next(task for task in simulator.tasks if task.kind == TaskKind.SERVICE)
    mask = simulator.observations()["scooper"]["action_mask"]
    assert not mask[service_task.index]
