from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from aeolus.data import (
    WeatherForcing,
    daylength_seconds,
    derive_dead_fuel_moisture,
    derive_live_fuel_moisture,
    downscale_weather_to_topography,
    herbaceous_curing_fraction,
)


def _forcing(days: int, *, wet: bool) -> WeatherForcing:
    count = days * 24
    hour = np.arange(count, dtype=np.float64)
    local = hour % 24.0
    temperature = 15.0 + 9.0 * np.sin(2.0 * np.pi * (local - 8.0) / 24.0)
    humidity = 45.0 - 15.0 * np.sin(2.0 * np.pi * (local - 8.0) / 24.0)
    if not wet:
        temperature += 16.0
        humidity[:] = 8.0
    return WeatherForcing(
        minute=hour * 60.0,
        wind_speed_m_s=np.full(count, 4.0, dtype=np.float32),
        wind_direction_deg=np.full(count, 270.0, dtype=np.float32),
        air_temperature_c=temperature.astype(np.float32),
        relative_humidity_pct=humidity.astype(np.float32),
        precipitation_rate_mm_h=(
            np.full(count, 0.08, dtype=np.float32) if wet else np.zeros(count, dtype=np.float32)
        ),
        metadata={"source": "synthetic"},
    )


def test_daylength_and_curing_endpoints() -> None:
    summer = daylength_seconds(45.0, 172.0)
    winter = daylength_seconds(45.0, 355.0)
    assert summer > winter
    curing = herbaceous_curing_fraction(np.asarray([0.30, 0.75, 1.20, 2.00]))
    assert np.allclose(curing, [1.0, 0.5, 0.0, 0.0])


def test_gsi_separates_growing_and_hot_dry_conditions() -> None:
    start = datetime(2022, 5, 1, tzinfo=timezone.utc)
    growing = derive_live_fuel_moisture(
        _forcing(65, wet=True),
        latitude_deg=38.5,
        longitude_deg=-120.5,
        start_datetime=start,
    )
    dormant = derive_live_fuel_moisture(
        _forcing(65, wet=False),
        latitude_deg=38.5,
        longitude_deg=-120.5,
        start_datetime=start,
    )
    assert growing.moisture_live_herbaceous is not None
    assert dormant.moisture_live_herbaceous is not None
    assert growing.moisture_live_herbaceous[-1] > dormant.moisture_live_herbaceous[-1]
    assert growing.moisture_live_woody[-1] > dormant.moisture_live_woody[-1]
    assert 0.30 <= dormant.moisture_live_herbaceous.min()
    assert growing.moisture_live_herbaceous.max() <= 2.50


def test_topographic_downscaling_creates_spatial_moisture() -> None:
    forcing = _forcing(2, wet=False)
    elevation = np.asarray([[500.0, 1_500.0], [1_000.0, 2_000.0]], dtype=np.float32)
    downscaled = downscale_weather_to_topography(forcing, elevation)
    assert downscaled.air_temperature_c.shape == (48, 2, 2)
    assert downscaled.air_temperature_c[0, 0, 0] > downscaled.air_temperature_c[0, 1, 1]
    assert downscaled.relative_humidity_pct[0, 0, 0] < downscaled.relative_humidity_pct[0, 1, 1]
    moist = derive_dead_fuel_moisture(downscaled)
    assert moist.moisture_dead_1h is not None
    assert np.ptp(moist.moisture_dead_1h[-1]) > 0.0
