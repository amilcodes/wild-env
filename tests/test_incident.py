from __future__ import annotations

import json

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
from aeolus.data.importers import fetch_feds_perimeters, geojson_bbox
from aeolus.evaluation.historical import (
    PerimeterSeries,
    compare_counterfactual_policies,
    perimeter_metrics,
    run_hindcast,
    run_shadow_replay,
)
from aeolus.policies import no_aerial_action


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


def test_perimeter_metrics_have_expected_identity() -> None:
    mask = np.eye(8, dtype=np.bool_)
    scores = perimeter_metrics(mask, mask, 30.0)
    assert scores["iou"] == 1.0
    assert scores["area_bias_km2"] == 0.0


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
