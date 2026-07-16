from __future__ import annotations

from dataclasses import replace

import numpy as np

from aeolus.config import ResourceSpec, ScenarioConfig
from aeolus.core.initialization import reconstruct_arrival_history
from aeolus.core.localization import localize_front_correction
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import ResourceStatus
from aeolus.core.suppression import (
    apply_aerial_drop,
    construct_line_segment,
    update_suppression_state,
)
from aeolus.core.tasks import TaskKind
from aeolus.data import (
    StationObservation,
    WeatherForcing,
    advance_dead_fuel_moisture,
    analyze_incident_forcing,
    dead_fuel_moisture_equilibria,
    derive_dead_fuel_moisture,
    write_weather_forcing,
)


def _circle(size: int, radius: float) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    return np.hypot(x - center, y - center) <= radius


def test_two_perimeter_history_recovers_radial_velocity() -> None:
    cell_size_m = 30.0
    history = reconstruct_arrival_history(
        _circle(81, 8.0),
        _circle(81, 14.0),
        elapsed_min=60.0,
        cell_size_m=cell_size_m,
    )
    later_front = _circle(81, 14.0) & ~_circle(81, 13.0)
    radial_y, radial_x = np.mgrid[:81, :81]
    radial_x = radial_x - 40.0
    radial_y = radial_y - 40.0
    norm = np.maximum(np.hypot(radial_x, radial_y), 1.0)
    alignment = history.head_x * radial_x / norm + history.head_y * radial_y / norm
    expected_speed = 6.0 * cell_size_m / 60.0
    assert np.isclose(
        np.median(history.speed_m_min[later_front]),
        expected_speed,
        rtol=0.45,
    )
    assert np.median(alignment[later_front]) > 0.85
    assert np.all(history.arrival_time_min[_circle(81, 8.0)] <= -60.0)
    assert np.all(np.isinf(history.arrival_time_min[~_circle(81, 14.0)]))


def test_wrf_sfire_moisture_hysteresis_and_exact_time_lag() -> None:
    drying, wetting = dead_fuel_moisture_equilibria(25.0, 35.0)
    assert float(drying) > float(wetting)
    inside = np.asarray(0.5 * (drying + wetting), dtype=np.float32)
    unchanged = advance_dead_fuel_moisture(
        inside,
        temperature_c=25.0,
        relative_humidity_pct=35.0,
        precipitation_rate_mm_h=0.0,
        lag_hours=1.0,
        dt_min=60.0,
    )
    assert np.isclose(unchanged, inside)
    dry_start = np.asarray(0.01, dtype=np.float32)
    advanced = advance_dead_fuel_moisture(
        dry_start,
        temperature_c=25.0,
        relative_humidity_pct=35.0,
        precipitation_rate_mm_h=0.0,
        lag_hours=1.0,
        dt_min=60.0,
    )
    expected = dry_start + (wetting - dry_start) * (1.0 - np.exp(-1.0))
    assert np.isclose(advanced, expected)


def test_weather_preparation_derives_prognostic_dead_fuel_moisture() -> None:
    forcing = WeatherForcing(
        minute=np.asarray([0.0, 60.0, 120.0]),
        wind_speed_m_s=np.full(3, 4.0, dtype=np.float32),
        wind_direction_deg=np.full(3, 270.0, dtype=np.float32),
        air_temperature_c=np.full(3, 25.0, dtype=np.float32),
        relative_humidity_pct=np.asarray([20.0, 80.0, 80.0], dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(3, dtype=np.float32),
        metadata={"source": "unit-test"},
    )
    derived = derive_dead_fuel_moisture(forcing)
    assert derived.moisture_dead_1h is not None
    assert derived.moisture_dead_10h is not None
    assert derived.moisture_dead_100h is not None
    one_hour_change = float(derived.moisture_dead_1h[-1] - derived.moisture_dead_1h[0])
    hundred_hour_change = float(derived.moisture_dead_100h[-1] - derived.moisture_dead_100h[0])
    assert one_hour_change > hundred_hour_change > 0.0
    assert derived.metadata["fuel_moisture_model"] == ("wrf-sfire-equilibrium-time-lag")


def test_history_initialization_carries_age_heat_and_localized_correction() -> None:
    simulator = AeolusSimulator(ScenarioConfig(width=48, height=48, spotting_rate=0.0))
    earlier = _circle(48, 5.0)
    later = _circle(48, 9.0)
    diagnostics = simulator.initialize_from_arrival_history(
        earlier,
        later,
        80.0,
    )
    truth = simulator.state.truth
    assert diagnostics["growth_cells"] > 0
    assert np.max(truth.burn_age_min[earlier]) >= 80.0
    assert np.max(truth.history_heat_flux_kw_m2[later]) > 0.0
    assert truth.history_confidence[24, 24] < truth.history_confidence[24, 33]
    assert np.isfinite(truth.arrival_time_min[later]).all()


def test_aerial_drop_conserves_payload_volume_and_treatment_half_life() -> None:
    config = ScenarioConfig(width=32, height=32, spotting_rate=0.0)
    simulator = AeolusSimulator(config)
    truth = simulator.state.truth
    diagnostics = apply_aerial_drop(
        truth,
        config,
        kind="retardant",
        payload_l=10_000.0,
        x=16.0,
        y=16.0,
        length_m=600.0,
        width_m=120.0,
        heading_deg=90.0,
        wind_speed_m_s=0.0,
        wind_from_direction_deg=0.0,
    )
    reconstructed_liters = (
        truth.retardant_coverage_gpc.sum() * config.suppression.gpc_l_m2 * config.cell_size_m**2
    )
    assert np.isclose(reconstructed_liters, 10_000.0, rtol=1.0e-5)
    assert np.isclose(diagnostics["applied_l"], 10_000.0, rtol=1.0e-5)
    before = truth.retardant_coverage_gpc.copy()
    long_half_life = replace(
        config,
        suppression=replace(
            config.suppression,
            retardant_half_life_min=8.0,
        ),
    )
    for _ in range(8):
        update_suppression_state(
            truth,
            long_half_life,
            np.random.default_rng(2),
            precipitation_rate_mm_h=0.0,
        )
    assert np.allclose(
        truth.retardant_coverage_gpc,
        0.5 * before,
        rtol=1.0e-5,
    )


def test_constructed_line_has_explicit_engagement_state() -> None:
    config = ScenarioConfig(width=32, height=32, spotting_rate=0.0)
    simulator = AeolusSimulator(config)
    truth = simulator.state.truth
    cells = construct_line_segment(
        truth,
        config,
        (8.0, 15.0),
        (24.0, 15.0),
        2.0,
    )
    truth.phase[14, 16] = 1
    truth.intensity_kw_m[14, 16] = 50.0
    safe = replace(
        config,
        suppression=replace(
            config.suppression,
            line_capacity_base_kw_m=1.0e7,
        ),
    )
    outcome = update_suppression_state(
        truth,
        safe,
        np.random.default_rng(9),
        precipitation_rate_mm_h=0.0,
    )
    assert cells > 0
    assert outcome["held_cells"] > 0
    assert np.any(truth.line_status == 2)


def test_incident_forcing_analysis_localizes_station_innovation() -> None:
    background = WeatherForcing(
        minute=np.array([0.0, 60.0]),
        wind_speed_m_s=np.array([4.0, 4.0], dtype=np.float32),
        wind_direction_deg=np.array([270.0, 270.0], dtype=np.float32),
        air_temperature_c=np.array([25.0, 25.0], dtype=np.float32),
        relative_humidity_pct=np.array([30.0, 30.0], dtype=np.float32),
        metadata={"source": "test background"},
    )
    coordinate = np.linspace(0.0, 20_000.0, 9)
    grid_x, grid_y = np.meshgrid(coordinate, coordinate)
    observations = [
        StationObservation(
            minute=0.0,
            x_m=10_000.0,
            y_m=10_000.0,
            wind_speed_m_s=9.0,
            wind_from_direction_deg=270.0,
            air_temperature_c=31.0,
            relative_humidity_pct=18.0,
            moisture_dead_1h=0.045,
            station_id="RAWS-test",
        )
    ]
    analysis = analyze_incident_forcing(
        background,
        observations,
        grid_x,
        grid_y,
        length_scale_m=4_000.0,
        time_window_min=10.0,
    )
    analyzed = analysis.forcing.at_minute(0.0)
    assert analyzed["wind_speed_m_s"][4, 4] > analyzed["wind_speed_m_s"][0, 0]
    assert analyzed["air_temperature_c"][4, 4] > analyzed["air_temperature_c"][0, 0]
    assert analyzed["moisture_dead_1h"][4, 4] < analyzed["moisture_dead_1h"][0, 0]
    assert analysis.wind_correction_std_m_s[0, 4, 4] < 0.5


def test_advancing_front_correction_is_band_limited() -> None:
    forecast = _circle(65, 8.0)
    observed = _circle(65, 11.0)
    from aeolus.core.front import signed_distance

    correction = localize_front_correction(
        signed_distance(forecast, 30.0),
        observed,
        30.0,
        localization_radius_m=60.0,
    )
    assert correction.localization_weight[32, 32] < 0.02
    assert correction.corrected_level_set_m[32, 42] < 0.0
    assert correction.diagnostics["support_cells"] < forecast.size


def test_explicit_crew_constructs_line_over_multiple_minutes() -> None:
    crew = ResourceSpec(
        "crew_test",
        "crew",
        40.0,
        0.0,
        0,
        0,
        300,
        line_length_m=240.0,
        line_width_m=2.0,
        line_production_m_min=60.0,
        max_operating_wind_m_s=30.0,
        max_direct_intensity_kw_m=10_000.0,
    )
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=32,
            height=32,
            horizon_min=30,
            decision_interval_min=1,
            max_tasks=24,
            spotting_rate=0.0,
            resources=(crew,),
        )
    )
    line_task = next(task for task in simulator.tasks if task.kind == TaskKind.LINE)
    simulator.decision_step({"crew_test": line_task.index})
    for _ in range(12):
        if simulator.state.resources[0].status not in {
            ResourceStatus.OUTBOUND,
            ResourceStatus.WORKING,
        }:
            break
        simulator.decision_step({"crew_test": 0})
    assert simulator.state.truth.constructed_line.sum() > 0
    assert any(event["kind"] == "line_complete" for event in simulator.state.events)


def test_incident_forcing_correction_fields_round_trip(tmp_path) -> None:
    shape = (2, 4, 5)
    forcing = WeatherForcing(
        minute=np.array([0.0, 60.0]),
        wind_speed_m_s=np.full(shape, 5.0, dtype=np.float32),
        wind_direction_deg=np.full(shape, 270.0, dtype=np.float32),
        air_temperature_c=np.full(shape, 28.0, dtype=np.float32),
        relative_humidity_pct=np.full(shape, 25.0, dtype=np.float32),
        moisture_dead_1h=np.full(shape, 0.07, dtype=np.float32),
        moisture_dead_10h=np.full(shape, 0.09, dtype=np.float32),
        moisture_dead_100h=np.full(shape, 0.11, dtype=np.float32),
        moisture_live_herbaceous=np.full(shape, 0.65, dtype=np.float32),
        moisture_live_woody=np.full(shape, 0.55, dtype=np.float32),
        wind_u_correction_m_s=np.full(shape, 1.5, dtype=np.float32),
        wind_v_correction_m_s=np.full(shape, -0.5, dtype=np.float32),
        metadata={"source": "round-trip-test"},
    )
    path = write_weather_forcing(
        tmp_path / "incident_forcing.nc",
        forcing,
        start_datetime="2026-07-28T00:00:00Z",
    )
    loaded = WeatherForcing.load(path)
    assert np.allclose(loaded.moisture_dead_1h, forcing.moisture_dead_1h)
    assert np.allclose(
        loaded.wind_u_correction_m_s,
        forcing.wind_u_correction_m_s,
    )
    sample = loaded.at_minute(30.0)
    assert np.asarray(sample["wind_speed_m_s"]).shape == shape[1:]


def test_aviation_wind_gate_is_in_actor_action_mask() -> None:
    tanker = ResourceSpec(
        "wind_limited_tanker",
        "retardant",
        70.0,
        10_000.0,
        10,
        0,
        180,
        max_operating_wind_m_s=12.0,
    )
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=32,
            height=32,
            max_tasks=24,
            wind_speed_m_s=16.0,
            spotting_rate=0.0,
            resources=(tanker,),
        )
    )
    observation = simulator.observations()["wind_limited_tanker"]
    retardant_indices = [
        task.index for task in simulator.tasks if task.kind in {TaskKind.RETARDANT, TaskKind.REINFORCE}
    ]
    assert retardant_indices
    assert not observation["action_mask"][retardant_indices].any()


def test_shared_reload_capacity_creates_queue_event() -> None:
    resources = tuple(
        ResourceSpec(
            f"heli_{index}",
            "water",
            100.0,
            2500.0,
            6,
            0,
            120,
        )
        for index in range(2)
    )
    config = ScenarioConfig(
        width=32,
        height=32,
        horizon_min=30,
        decision_interval_min=1,
        max_tasks=24,
        spotting_rate=0.0,
        resources=resources,
        suppression=replace(
            ScenarioConfig().suppression,
            base_reload_bays=1,
        ),
    )
    simulator = AeolusSimulator(config)
    water_tasks = [task for task in simulator.tasks if task.kind == TaskKind.WATER]
    assert len(water_tasks) >= 2
    simulator.decision_step(
        {
            "heli_0": water_tasks[0].index,
            "heli_1": water_tasks[1].index,
        }
    )
    for _ in range(20):
        simulator.decision_step({"heli_0": 0, "heli_1": 0})
        if any(event["kind"] == "reload_queued" for event in simulator.state.events):
            break
    assert any(event["kind"] == "reload_queued" for event in simulator.state.events)
