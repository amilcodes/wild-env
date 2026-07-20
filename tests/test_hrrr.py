from datetime import datetime, timezone

import numpy as np

from aeolus.data import WeatherForcing, initialize_causal_forecast_moisture
from aeolus.data.hrrr import (
    fetch_hrrr_analysis,
    fetch_hrrr_forecast,
    nearest_hrrr_indices,
    overlay_hrrr_analysis,
    select_hrrr_forecast_cycle,
)


def test_nearest_hrrr_indices_maps_target_cells() -> None:
    latitude = np.repeat(np.linspace(30.0, 40.0, 101)[:, None], 121, axis=1)
    longitude = np.repeat(np.linspace(-125.0, -113.0, 121)[None, :], 101, axis=0)
    target_latitude = np.asarray([[34.99, 35.01], [35.49, 35.51]])
    target_longitude = np.asarray([[-120.01, -119.99], [-119.51, -119.49]])
    y, x = nearest_hrrr_indices(
        latitude,
        longitude,
        target_latitude,
        target_longitude,
        coarse_stride=5,
        search_radius_cells=20,
    )
    assert np.allclose(latitude[y, x], target_latitude, atol=0.051)
    assert np.allclose(longitude[y, x], target_longitude, atol=0.051)


def test_overlay_hrrr_analysis_replaces_only_incident_window() -> None:
    shape = (5, 2, 2)
    background = WeatherForcing(
        minute=np.arange(5, dtype=np.float64) * 60.0,
        wind_speed_m_s=np.ones(shape, dtype=np.float32),
        wind_direction_deg=np.zeros(shape, dtype=np.float32),
        air_temperature_c=np.full(shape, 20.0, dtype=np.float32),
        relative_humidity_pct=np.full(shape, 40.0, dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(shape, dtype=np.float32),
        metadata={"source": "background"},
    )
    analysis_shape = (2, 2, 2)
    analysis = WeatherForcing(
        minute=np.asarray([0.0, 60.0]),
        wind_speed_m_s=np.full(analysis_shape, 8.0, dtype=np.float32),
        wind_direction_deg=np.full(analysis_shape, 270.0, dtype=np.float32),
        air_temperature_c=np.full(analysis_shape, 30.0, dtype=np.float32),
        relative_humidity_pct=np.full(analysis_shape, 15.0, dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(analysis_shape, dtype=np.float32),
        metadata={"time_units": "minutes since 2022-07-05T02:00:00Z"},
    )
    combined = overlay_hrrr_analysis(
        background,
        analysis,
        background_start="2022-07-05T00:00:00Z",
    )
    assert np.all(combined.wind_speed_m_s[:2] == 1.0)
    assert np.all(combined.wind_speed_m_s[2:4] == 8.0)
    assert np.all(combined.wind_speed_m_s[4] == 1.0)
    assert combined.metadata["hrrr_overlay_sample_count"] == 2


def test_forecast_cycle_selection_respects_availability_and_long_range() -> None:
    issue = datetime(2022, 7, 5, 10, 35, tzinfo=timezone.utc)
    assert (
        select_hrrr_forecast_cycle(
            issue,
            required_horizon_hours=12.0,
        ).isoformat()
        == "2022-07-05T08:00:00+00:00"
    )
    assert (
        select_hrrr_forecast_cycle(
            issue,
            required_horizon_hours=30.0,
        ).isoformat()
        == "2022-07-05T06:00:00+00:00"
    )


def test_fetch_hrrr_forecast_uses_one_preissue_cycle(monkeypatch, tmp_path) -> None:
    grid_latitude = np.repeat(np.linspace(38.0, 38.2, 5)[:, None], 5, axis=1)
    grid_longitude = np.repeat(np.linspace(-121.2, -121.0, 5)[None, :], 5, axis=0)
    monkeypatch.setattr(
        "aeolus.data.hrrr._load_grid",
        lambda *_args: (grid_latitude, grid_longitude),
    )

    def fake_read(_filesystem, _cycle, specification, bounds, lead_indices):
        _, variable, _ = specification
        values = {
            "UGRD": 3.0,
            "VGRD": 4.0,
            "TMP": 300.0,
            "RH": 20.0,
            "PRATE": 0.0001,
        }
        y0, y1, x0, x1 = bounds
        return np.full(
            (len(lead_indices), y1 - y0, x1 - x0),
            values[variable],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "aeolus.data.hrrr._read_forecast_variable_tile",
        fake_read,
    )
    target_latitude = np.asarray([[38.05, 38.10], [38.10, 38.15]])
    target_longitude = np.asarray([[-121.15, -121.10], [-121.10, -121.05]])
    forcing = fetch_hrrr_forecast(
        target_latitude,
        target_longitude,
        "2022-07-05T02:54:00Z",
        "2022-07-06T02:54:00Z",
        cache_directory=tmp_path,
    )
    assert forcing.metadata["forecast_reference_time"] == "2022-07-05T00:00:00Z"
    assert forcing.metadata["forecast_lead_hours"][0] == 2
    assert forcing.metadata["forecast_lead_hours"][-1] == 27
    assert np.allclose(forcing.wind_speed_m_s, 5.0)


def test_fetch_hrrr_analysis_repairs_only_absent_field_from_forecast(
    monkeypatch,
    tmp_path,
) -> None:
    from zarr.errors import PathNotFoundError

    grid_latitude = np.repeat(np.linspace(38.0, 38.2, 5)[:, None], 5, axis=1)
    grid_longitude = np.repeat(np.linspace(-121.2, -121.0, 5)[None, :], 5, axis=0)
    monkeypatch.setattr(
        "aeolus.data.hrrr._load_grid",
        lambda *_args: (grid_latitude, grid_longitude),
    )

    def fake_analysis(_filesystem, timestamp, specification, bounds):
        _, variable, _ = specification
        if timestamp.hour == 1 and variable == "PRATE":
            raise PathNotFoundError("PRATE")
        values = {
            "UGRD": 3.0 + timestamp.hour,
            "VGRD": 4.0,
            "TMP": 300.0,
            "RH": 20.0,
            "PRATE": 0.0001,
        }
        y0, y1, x0, x1 = bounds
        return np.full((y1 - y0, x1 - x0), values[variable], dtype=np.float32)

    forecast_calls = []

    def fake_forecast(_filesystem, cycle, specification, bounds, lead_indices):
        _, variable, _ = specification
        forecast_calls.append((cycle, variable, lead_indices.copy()))
        y0, y1, x0, x1 = bounds
        return np.full(
            (len(lead_indices), y1 - y0, x1 - x0),
            0.0002,
            dtype=np.float32,
        )

    monkeypatch.setattr("aeolus.data.hrrr._read_variable_tile", fake_analysis)
    monkeypatch.setattr("aeolus.data.hrrr._read_forecast_variable_tile", fake_forecast)
    target_latitude = np.asarray([[38.05, 38.10], [38.10, 38.15]])
    target_longitude = np.asarray([[-121.15, -121.10], [-121.10, -121.05]])
    forcing = fetch_hrrr_analysis(
        target_latitude,
        target_longitude,
        "2022-07-05T00:00:00Z",
        "2022-07-05T03:00:00Z",
        cache_directory=tmp_path,
        workers=1,
    )

    assert len(forecast_calls) == 1
    cycle, variable, lead_indices = forecast_calls[0]
    assert cycle.isoformat() == "2022-07-05T00:00:00+00:00"
    assert variable == "PRATE"
    assert lead_indices.tolist() == [0]
    assert np.allclose(forcing.wind_speed_m_s[1], np.hypot(4.0, 4.0))
    assert np.allclose(forcing.precipitation_rate_mm_h[1], 0.72)
    assert forcing.metadata["analysis_coverage_fraction"] == 0.75
    assert forcing.metadata["weather_coverage_fraction"] == 1.0
    assert forcing.metadata["forecast_repaired_hour_count"] == 1
    assert forcing.metadata["forecast_repaired_field_count"] == 1
    assert forcing.metadata["forecast_gap_repair"][0]["variable"] == "precipitation_rate"
    assert forcing.metadata["unresolved_weather_hours"] == []


def test_causal_forecast_moisture_uses_preissue_state_and_forecast_weather() -> None:
    background = WeatherForcing(
        minute=np.arange(5, dtype=np.float64) * 60.0,
        wind_speed_m_s=np.ones(5, dtype=np.float32),
        wind_direction_deg=np.zeros(5, dtype=np.float32),
        air_temperature_c=np.full(5, 20.0, dtype=np.float32),
        relative_humidity_pct=np.full(5, 40.0, dtype=np.float32),
        moisture_dead_1h=np.asarray([0.05, 0.06, 0.07, 0.95, 0.95], dtype=np.float32),
        moisture_dead_10h=np.asarray([0.08, 0.09, 0.10, 0.95, 0.95], dtype=np.float32),
        moisture_dead_100h=np.asarray([0.11, 0.12, 0.13, 0.95, 0.95], dtype=np.float32),
        moisture_live_herbaceous=np.asarray([0.7, 0.8, 0.9, 2.5, 2.5], dtype=np.float32),
        moisture_live_woody=np.asarray([0.6, 0.7, 0.8, 2.0, 2.0], dtype=np.float32),
        metadata={
            "source": "causal background",
            "time_units": "minutes since 2022-07-05T00:00:00Z",
        },
    )
    shape = (3, 2, 2)
    forecast = WeatherForcing(
        minute=np.asarray([60.0, 120.0, 180.0]),
        wind_speed_m_s=np.ones(shape, dtype=np.float32),
        wind_direction_deg=np.zeros(shape, dtype=np.float32),
        air_temperature_c=np.full(shape, 30.0, dtype=np.float32),
        relative_humidity_pct=np.full(shape, 10.0, dtype=np.float32),
        precipitation_rate_mm_h=np.zeros(shape, dtype=np.float32),
        metadata={
            "source": "forecast",
            "analysis_or_forecast": "forecast",
            "time_units": "minutes since 2022-07-05T00:00:00Z",
        },
    )
    result = initialize_causal_forecast_moisture(
        forecast,
        background,
        issue_time="2022-07-05T02:30:00Z",
    )
    assert np.allclose(result.moisture_dead_1h[0], 0.06)
    assert np.all(result.moisture_dead_1h[1:] < result.moisture_dead_1h[0])
    assert np.allclose(result.moisture_live_herbaceous, 0.9)
    assert np.allclose(result.moisture_live_woody, 0.8)
    assert result.metadata["fuel_moisture_initialization_causal"] is True
    assert result.metadata["fuel_moisture_initialization_time"] == "2022-07-05T01:00:00Z"
