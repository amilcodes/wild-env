from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from rasterio.transform import from_origin

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import (
    IncidentBundle,
    ScenarioBundle,
    WeatherForcing,
    write_incident_bundle,
    write_weather_forcing,
)
from aeolus.data.importers import (
    fetch_feds_perimeters,
    fetch_nasa_power_hourly,
    geojson_bbox,
    load_nirops_perimeters,
)
from aeolus.evaluation.ensemble import (
    calibrate_particle_ensemble,
    incremental_growth_log_likelihood,
    normalize_log_weights,
    probability_metrics,
    run_ensemble_hindcast,
    systematic_resample,
    tempered_log_weights,
)
from aeolus.evaluation.historical import (
    PerimeterFrame,
    PerimeterSeries,
    boundary_distance_metrics,
    compare_counterfactual_policies,
    perimeter_metrics,
    run_hindcast,
    run_shadow_replay,
)
from aeolus.policies import no_aerial_action
from aeolus.workflows import scenario_from_incident


def _polygon(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _incident(tmp_path) -> IncidentBundle:
    shape = (24, 24)
    transform = from_origin(0.0, 24.0, 1.0, 1.0)
    landscape = ScenarioBundle(
        elevation_m=np.add.outer(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
        ),
        fuel_load_kg_m2=np.full(shape, 0.8, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        metadata={
            "schema_version": 1,
            "crs": "EPSG:4326",
            "cell_size_m": 1.0,
            "sources": [{"name": "synthetic-test"}],
            "transformations": ["unit-test"],
            "split": "test",
            "transform": list(tuple(transform)[:6]),
            "bounds": [0.0, 0.0, 24.0, 24.0],
        },
    )
    perimeters = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _polygon(9.0, 9.0, 13.0, 13.0),
                "properties": {"observed_at": "2026-01-01T00:00:00Z", "source": "test"},
            },
            {
                "type": "Feature",
                "geometry": _polygon(8.0, 8.0, 15.0, 15.0),
                "properties": {"observed_at": "2026-01-01T00:06:00Z", "source": "test"},
            },
        ],
    }
    return write_incident_bundle(
        tmp_path / "incident",
        incident_id="test-fire",
        bbox=(0.0, 0.0, 24.0, 24.0),
        start_datetime="2026-01-01T00:00:00Z",
        end_datetime="2026-01-01T00:06:00Z",
        scenario_bundle=landscape,
        perimeter_collection=perimeters,
    )


def test_incident_bundle_round_trip_and_rasterization(tmp_path) -> None:
    incident = _incident(tmp_path)
    restored = IncidentBundle.load(incident.root)
    assert restored.incident_id == "test-fire"
    series = PerimeterSeries.from_incident(restored)
    assert len(series.frames) == 2
    assert series.frames[0].mask.sum() < series.frames[1].mask.sum()
    config = scenario_from_incident(restored)
    assert config.time_origin == "2026-01-01T00:00:00Z"
    assert config.scenario_id == "test-fire"


def test_perimeter_series_coalesces_duplicate_timestamps(tmp_path) -> None:
    incident = _incident(tmp_path)
    perimeter_path = incident.asset_path("observed-perimeters")
    collection = json.loads(perimeter_path.read_text(encoding="utf-8"))
    collection["features"].append(
        {
            "type": "Feature",
            "geometry": _polygon(2.0, 2.0, 4.0, 4.0),
            "properties": {
                "observed_at": "2026-01-01T00:00:00Z",
                "source": "second-scene-fragment",
            },
        }
    )
    perimeter_path.write_text(json.dumps(collection), encoding="utf-8")
    series = PerimeterSeries.from_incident(IncidentBundle.load(incident.root))
    assert len(series.frames) == 2
    assert series.frames[0].properties["source_feature_count"] == 2
    assert series.frames[0].properties["coalesced_duplicate_features"] == 1
    assert series.frames[0].mask[21, 2]


def test_hindcast_produces_spatial_scores(tmp_path) -> None:
    incident = _incident(tmp_path)
    series = PerimeterSeries.from_incident(incident)
    config = ScenarioConfig(
        width=24,
        height=24,
        cell_size_m=1.0,
        horizon_min=12,
        decision_interval_min=3,
        max_tasks=16,
        wind_speed_m_s=0.6,
        spotting_rate=0.0,
        landscape_bundle=str(incident.root),
    )
    result = run_hindcast(AeolusSimulator(config), series, no_aerial_action)
    assert result["simulated_minutes"] == 6
    assert 0.0 <= result["metrics"]["iou"] <= 1.0
    assert 0.0 <= result["growth_metrics"]["iou"] <= 1.0
    assert 0.0 <= result["growth_tolerance_1_cell"]["f1"] <= 1.0
    assert result["metrics"]["observed_area_km2"] > 0.0


def test_shadow_replay_and_paired_counterfactual_modes(tmp_path) -> None:
    incident = _incident(tmp_path)
    series = PerimeterSeries.from_incident(incident)
    config = ScenarioConfig(
        width=24,
        height=24,
        cell_size_m=1.0,
        horizon_min=12,
        decision_interval_min=3,
        max_tasks=16,
        wind_speed_m_s=0.6,
        spotting_rate=0.0,
        landscape_bundle=str(incident.root),
    )
    shadow = run_shadow_replay(
        AeolusSimulator(config),
        series,
        no_aerial_action,
        start_index=0,
        end_index=1,
    )
    assert shadow["assimilated_perimeters"] == 1
    comparison = compare_counterfactual_policies(
        config,
        series,
        {"none": no_aerial_action},
        [3, 5],
    )
    assert comparison["mode"] == "paired-counterfactual"
    assert comparison["summary"]["none"]["episodes"] == 2


def test_perimeter_assimilation_updates_continuous_belief() -> None:
    simulator = AeolusSimulator(ScenarioConfig(width=24, height=24, max_tasks=16))
    observed = np.zeros((24, 24), dtype=np.bool_)
    observed[6:18, 7:17] = True
    simulator.initialize_from_observed_perimeter(
        observed,
        source="unit-test-perimeter",
    )
    belief = simulator.state.belief
    assert belief.burn_probability[12, 12] > 0.95
    assert belief.burn_probability[0, 0] < 0.05
    assert np.isfinite(belief.arrival_time_mean[12, 12])
    assert belief.perimeter_source == "unit-test-perimeter"


def test_perimeter_metrics_have_expected_identity() -> None:
    mask = np.eye(8, dtype=np.bool_)
    scores = perimeter_metrics(mask, mask, 30.0)
    assert scores["iou"] == 1.0
    assert scores["area_bias_km2"] == 0.0
    boundary = boundary_distance_metrics(mask, mask, 30.0)
    assert boundary["mean_symmetric_distance_m"] == 0.0
    assert boundary["hausdorff_95_m"] == 0.0


def test_feds_import_normalizes_and_sorts(monkeypatch) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _polygon(-121.0, 39.0, -120.9, 39.1),
                "properties": {"t": 2000, "fireid": 7, "region": "CONUS"},
            },
            {
                "type": "Feature",
                "geometry": _polygon(-121.0, 39.0, -120.95, 39.05),
                "properties": {"t": 1000, "fireid": 7, "region": "CONUS"},
            },
        ],
    }

    def fake_download(*_args, **_kwargs) -> bytes:
        return json.dumps(collection).encode()

    monkeypatch.setattr("aeolus.data.importers._download", fake_download)
    restored = fetch_feds_perimeters("CONUS", 7)
    assert [item["properties"]["t"] for item in restored["features"]] == [1000, 2000]
    assert restored["features"][0]["properties"]["source"] == "NASA-FEDS-VIIRS"
    assert geojson_bbox(restored) == (-121.0, 39.0, -120.9, 39.1)


def test_nasa_power_import_builds_relative_hourly_forcing(monkeypatch) -> None:
    payload = {
        "header": {"fill_value": -999.0, "api": {"version": "test"}},
        "properties": {
            "parameter": {
                "WS10M": {"2022070502": 3.0, "2022070503": 4.0},
                "WD10M": {"2022070502": 250.0, "2022070503": 260.0},
                "T2M": {"2022070502": 20.0, "2022070503": 18.0},
                "RH2M": {"2022070502": 30.0, "2022070503": 40.0},
                "PRECTOTCORR": {"2022070502": 0.0, "2022070503": 0.1},
            }
        },
    }

    def fake_download(*_args, **_kwargs) -> bytes:
        return json.dumps(payload).encode()

    monkeypatch.setattr("aeolus.data.importers._download", fake_download)
    forcing = fetch_nasa_power_hourly(
        38.0,
        -120.0,
        "2022-07-05T02:30:00Z",
        "2022-07-05T04:00:00Z",
    )
    assert forcing.minute.tolist() == [-30.0, 30.0]
    assert forcing.at_minute(0.0)["wind_speed_m_s"] == 3.5
    assert forcing.at_minute(0.0)["precipitation_rate_mm_h"] == pytest.approx(0.05)


def test_spatial_weather_round_trip_and_circular_direction(tmp_path) -> None:
    shape = (2, 3, 4)
    forcing = WeatherForcing(
        minute=np.asarray([0.0, 60.0]),
        wind_speed_m_s=np.stack(
            (
                np.full(shape[1:], 2.0, dtype=np.float32),
                np.full(shape[1:], 4.0, dtype=np.float32),
            )
        ),
        wind_direction_deg=np.stack(
            (
                np.full(shape[1:], 359.0, dtype=np.float32),
                np.full(shape[1:], 1.0, dtype=np.float32),
            )
        ),
        air_temperature_c=np.full(shape, 30.0, dtype=np.float32),
        relative_humidity_pct=np.full(shape, 20.0, dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(shape, dtype=np.float32),
        metadata={"source": "unit-test", "nested": {"coverage": 0.99}},
    )
    path = write_weather_forcing(
        tmp_path / "spatial.nc",
        forcing,
        start_datetime="2026-01-01T00:00:00Z",
    )
    restored = WeatherForcing.load(path)
    assert restored.metadata["nested"]["coverage"] == 0.99
    midpoint = restored.at_minute(30.0)
    assert np.asarray(midpoint["wind_speed_m_s"]).shape == shape[1:]
    assert np.allclose(midpoint["wind_speed_m_s"], 3.0)
    direction = np.asarray(midpoint["wind_direction_deg"])
    assert np.all((direction < 0.1) | (direction > 359.9))


def test_hindcast_uses_absolute_weather_clock_for_later_interval(tmp_path) -> None:
    forcing = WeatherForcing(
        minute=np.asarray([0.0, 60.0, 120.0]),
        wind_speed_m_s=np.asarray([1.0, 5.0, 9.0], dtype=np.float32),
        wind_direction_deg=np.zeros(3, dtype=np.float32),
        air_temperature_c=np.full(3, 25.0, dtype=np.float32),
        relative_humidity_pct=np.full(3, 30.0, dtype=np.float32),
        metadata={"source": "unit-test"},
    )
    weather_path = write_weather_forcing(
        tmp_path / "clock.nc",
        forcing,
        start_datetime="2026-01-01T00:00:00Z",
    )
    masks = []
    for radius in (2, 3, 4):
        y, x = np.mgrid[:24, :24]
        masks.append(np.hypot(x - 12, y - 12) <= radius)
    series = PerimeterSeries(
        frames=tuple(
            PerimeterFrame(
                datetime(2026, 1, 1, hour, tzinfo=timezone.utc),
                mask,
                {},
            )
            for hour, mask in enumerate(masks)
        ),
        cell_size_m=30.0,
    )
    config = ScenarioConfig(
        width=24,
        height=24,
        cell_size_m=30.0,
        horizon_min=65,
        decision_interval_min=3,
        max_tasks=16,
        spotting_rate=0.0,
        terminate_on_escape=False,
        time_origin="2026-01-01T00:00:00Z",
        weather_forcing=str(weather_path),
    )
    simulator = AeolusSimulator(config)
    result = run_hindcast(
        simulator,
        series,
        no_aerial_action,
        start_index=1,
        target_index=2,
    )
    assert result["forcing_clock"]["forecast_start_offset_min"] == 60.0
    assert result["forcing_clock"]["forecast_end_offset_min"] == 120.0
    simulator.reset()
    simulator.set_simulation_start(series.frames[1].timestamp)
    assert simulator.current_weather()["wind_speed_m_s"] == 5.0


def test_probability_scores_reward_calibrated_identity() -> None:
    observed = np.zeros((20, 20), dtype=np.bool_)
    observed[6:14, 7:13] = True
    perfect = np.where(observed, 0.99, 0.01)
    uncertain = np.full(observed.shape, 0.5)
    perfect_score = probability_metrics(perfect, observed)
    uncertain_score = probability_metrics(uncertain, observed)
    assert perfect_score["brier_score"] < uncertain_score["brier_score"]
    assert perfect_score["log_score"] < uncertain_score["log_score"]
    assert np.isclose(normalize_log_weights(np.array([-1.0, -1.0])).sum(), 1.0)
    indices = systematic_resample([0.0, 1.0], seed=7)
    assert np.array_equal(indices, np.ones(2, dtype=np.int64))
    tempered, beta, raw_ess = tempered_log_weights(np.asarray([0.0, -100.0, -200.0, -300.0]))
    assert np.isclose(tempered.sum(), 1.0)
    assert beta < 1.0
    assert raw_ess < 2.0
    assert 1.0 / np.sum(tempered**2) >= 1.4 - 1e-8
    empty = np.zeros((8, 8), dtype=np.bool_)
    growth = empty.copy()
    growth[3:5, 3:5] = True
    identity = incremental_growth_log_likelihood(growth, growth)
    missing = incremental_growth_log_likelihood(empty, growth)
    assert identity["log_likelihood"] == 0.0
    assert np.isfinite(missing["log_likelihood"])
    assert missing["log_likelihood"] < identity["log_likelihood"]


def test_parameter_ensemble_calibrates_and_forecasts(tmp_path) -> None:
    incident = _incident(tmp_path)
    series = PerimeterSeries.from_incident(incident)
    config = ScenarioConfig(
        width=24,
        height=24,
        cell_size_m=1.0,
        horizon_min=12,
        decision_interval_min=3,
        max_tasks=16,
        wind_speed_m_s=0.6,
        spotting_rate=0.0,
        landscape_bundle=str(incident.root),
    )
    calibration = calibrate_particle_ensemble(
        config,
        series,
        no_aerial_action,
        start_index=0,
        target_index=1,
        spread_candidates=[0.03, 0.10, 0.30, 1.0],
        particle_count=4,
        seed=19,
        localization_sigma_m=2.0,
    )
    assert np.isclose(sum(calibration["posterior_weights"]), 1.0)
    result = run_ensemble_hindcast(
        config,
        series,
        no_aerial_action,
        start_index=0,
        target_index=1,
        particles=calibration["particles"],
        weights=calibration["posterior_weights"],
        return_probability=True,
    )
    assert result["probability"].shape == series.frames[0].mask.shape
    assert 0.0 <= result["probabilistic_metrics"]["brier_score"] <= 1.0


def test_nirops_import_filters_and_sorts_acquisition_times(tmp_path) -> None:
    import shapefile

    source = tmp_path / "nirops"
    with shapefile.Writer(str(source), shapeType=shapefile.POLYGON) as writer:
        writer.field("Incident_C", "C")
        writer.field("Inc Name", "C")
        writer.field("Inc Number", "C")
        writer.field("UTC", "C")
        writer.field("IRWINID", "C")
        writer.field("Acres", "N", decimal=1)
        for code, timestamp, acres, offset in (
            ("CA-TEST-001_Fire", "2022-07-06T04:00:00", 20.0, 0.0),
            ("OTHER", "2022-07-05T03:00:00", 8.0, 2.0),
            ("CA-TEST-001_Fire", "2022-07-05T04:00:00", 10.0, 1.0),
        ):
            writer.poly(
                [
                    [
                        [offset, offset],
                        [offset, offset + 0.5],
                        [offset + 0.5, offset + 0.5],
                        [offset + 0.5, offset],
                        [offset, offset],
                    ]
                ]
            )
            writer.record(
                code,
                "Test Fire",
                "TEST-001",
                timestamp,
                "test-irwin",
                acres,
            )

    collection = load_nirops_perimeters(
        source.with_suffix(".shp"),
        "CA-TEST-001_Fire",
    )
    assert len(collection["features"]) == 2
    assert [feature["properties"]["observed_at"] for feature in collection["features"]] == [
        "2022-07-05T04:00:00Z",
        "2022-07-06T04:00:00Z",
    ]
    assert collection["features"][1]["properties"]["reported_acres"] == 20.0
    assert collection["aeolus:source"]["doi"].endswith("95rj5d379g.1")


def test_weather_forcing_round_trip_and_direction_interpolation(tmp_path) -> None:
    forcing = WeatherForcing(
        minute=np.asarray([0.0, 30.0, 60.0]),
        wind_speed_m_s=np.asarray([2.0, 4.0, 8.0], dtype=np.float32),
        wind_direction_deg=np.asarray([350.0, 0.0, 10.0], dtype=np.float32),
        air_temperature_c=np.asarray([18.0, 20.0, 23.0], dtype=np.float32),
        relative_humidity_pct=np.asarray([60.0, 45.0, 30.0], dtype=np.float32),
        metadata={"source": "unit-test"},
    )
    path = write_weather_forcing(
        tmp_path / "weather.nc",
        forcing,
        start_datetime="2026-07-26T00:00:00Z",
    )
    loaded = WeatherForcing.load(path)
    midpoint = loaded.at_minute(15)
    assert midpoint["wind_speed_m_s"] == 3.0
    assert midpoint["wind_direction_deg"] == pytest.approx(355.0)
    assert midpoint["air_temperature_c"] == pytest.approx(19.0)
