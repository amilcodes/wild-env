from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from aeolus.data import (
    ScenarioBundle,
    WeatherForcing,
    local_utm_crs,
    reproject_scenario_to_metric,
    reproject_weather_to_scenario,
)
from aeolus.data.importers import (
    buffered_web_mercator_bbox,
    build_landscape_from_services,
)
from aeolus.evaluation.validity import assess_metric_crs


def _web_mercator_scenario() -> ScenarioBundle:
    shape = (24, 24)
    bounds = (-13_300_000.0, 5_600_000.0, -13_276_000.0, 5_624_000.0)
    transform = from_bounds(*bounds, shape[1], shape[0])
    return ScenarioBundle(
        elevation_m=np.add.outer(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
        ),
        fuel_load_kg_m2=np.ones(shape, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        fuel_model_number=np.full(shape, 101, dtype=np.int16),
        canopy_cover=np.full(shape, 0.4, dtype=np.float32),
        canopy_height_m=np.full(shape, 5.0, dtype=np.float32),
        canopy_base_height_m=np.full(shape, 2.0, dtype=np.float32),
        canopy_bulk_density_kg_m3=np.full(shape, 0.1, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": "EPSG:3857",
            "cell_size_m": 1000.0,
            "sources": [{"name": "synthetic"}],
            "transformations": [],
            "split": "test",
            "transform": tuple(transform)[:6],
            "bounds": bounds,
            "source_bbox_wgs84": (-119.5, 44.8, -119.2, 45.0),
        },
    )


def test_metric_crs_gate_rejects_web_mercator_and_accepts_local_utm() -> None:
    source = _web_mercator_scenario()
    before = assess_metric_crs(source)
    assert before["web_mercator"]
    assert not before["supports_physical_distance_claims"]
    assert before["approximate_ground_area_per_map_area"] == pytest.approx(
        0.5,
        abs=0.01,
    )

    metric = reproject_scenario_to_metric(source)
    after = assess_metric_crs(metric)
    assert after["supports_physical_distance_claims"]
    assert after["epsg"] == 32611
    assert 650.0 < metric.metadata["cell_size_m"] < 750.0
    assert metric.fuel_model_number is not None
    assert np.all(metric.fuel_model_number[~metric.barrier] == 101)
    assert metric.barrier.any()


def test_spatial_weather_reprojection_preserves_constant_vector_wind() -> None:
    source = _web_mercator_scenario()
    metric = reproject_scenario_to_metric(source)
    shape = (2, *source.elevation_m.shape)
    weather = WeatherForcing(
        minute=np.asarray([0.0, 60.0]),
        wind_speed_m_s=np.full(shape, 8.0, dtype=np.float32),
        wind_direction_deg=np.full(shape, 270.0, dtype=np.float32),
        air_temperature_c=np.full(shape, 25.0, dtype=np.float32),
        relative_humidity_pct=np.full(shape, 20.0, dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(shape, dtype=np.float32),
        metadata={"source": "synthetic"},
    )
    result = reproject_weather_to_scenario(weather, source, metric)
    assert result.wind_speed_m_s.shape[1:] == metric.elevation_m.shape
    valid = ~metric.barrier
    assert np.allclose(result.wind_speed_m_s[:, valid], 8.0, atol=1e-4)
    assert np.allclose(result.wind_direction_deg[:, valid], 270.0, atol=1e-4)
    assert np.allclose(result.air_temperature_c[:, valid], 25.0)


def test_local_utm_zone_selection() -> None:
    assert local_utm_crs(-120.0, 38.0) == "EPSG:32611"
    assert local_utm_crs(-70.0, -35.0) == "EPSG:32719"


def test_web_mercator_service_buffer_preserves_requested_ground_distance() -> None:
    bbox = (-121.0, 44.9, -120.8, 45.1)
    unbuffered = buffered_web_mercator_bbox(bbox, 0.0)
    buffered = buffered_web_mercator_bbox(bbox, 4_500.0)
    map_buffer = unbuffered[0] - buffered[0]
    ground_buffer = map_buffer * np.cos(np.deg2rad(45.0))
    assert ground_buffer == pytest.approx(4_500.0)


def test_service_landscape_is_metric_by_construction(monkeypatch, tmp_path) -> None:
    def write_raster(path, bbox, size, value):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=size[0],
            height=size[1],
            count=1,
            dtype="float32",
            crs="EPSG:3857",
            transform=from_bounds(*bbox, size[0], size[1]),
        ) as dataset:
            dataset.write(
                np.full((size[1], size[0]), value, dtype=np.float32),
                1,
            )
        return destination

    def elevation(bbox, size, destination):
        return write_raster(destination, bbox, size, 1_500.0)

    values = {
        "fuel_model": 101.0,
        "canopy_cover": 40.0,
        "canopy_height": 50.0,
        "canopy_base_height": 20.0,
        "canopy_bulk_density": 10.0,
    }

    def landfire(coverage, bbox, size, destination):
        return write_raster(destination, bbox, size, values[coverage])

    monkeypatch.setattr("aeolus.data.importers.download_usgs_3dep", elevation)
    monkeypatch.setattr(
        "aeolus.data.importers.download_landfire_coverage",
        landfire,
    )
    scenario, path = build_landscape_from_services(
        (-121.0, 44.9, -120.8, 45.1),
        tmp_path / "prepared",
        size=(24, 24),
        buffer_m=4_500.0,
    )
    assert assess_metric_crs(scenario)["supports_physical_distance_claims"]
    assert scenario.metadata["crs"] == "EPSG:32610"
    with rasterio.open(path) as dataset:
        assert dataset.crs.to_string() == scenario.metadata["crs"]
        assert (dataset.height, dataset.width) == scenario.elevation_m.shape
        assert dataset.descriptions == (
            "elevation_m",
            "fbfm40_code",
            "canopy_cover",
            "canopy_height",
            "canopy_base_height",
            "canopy_bulk_density",
        )
