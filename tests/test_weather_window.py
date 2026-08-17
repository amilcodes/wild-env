from __future__ import annotations

import numpy as np

from aeolus.data import WeatherForcing, trim_weather_forcing


def test_trim_weather_forcing_retains_bracketing_sample_and_rebases() -> None:
    forcing = WeatherForcing(
        minute=np.arange(8, dtype=np.float64) * 60.0,
        wind_speed_m_s=np.arange(8, dtype=np.float32),
        wind_direction_deg=np.zeros(8, dtype=np.float32),
        air_temperature_c=np.full(8, 20.0, dtype=np.float32),
        relative_humidity_pct=np.full(8, 40.0, dtype=np.float32),
        moisture_dead_1h=np.arange(8, dtype=np.float32) / 100.0,
        metadata={"time_units": "minutes since 2022-07-01T00:00:00Z"},
    )
    trimmed = trim_weather_forcing(
        forcing,
        start_minute=210.0,
        end_minute=330.0,
    )
    assert trimmed.minute.tolist() == [-30.0, 30.0, 90.0, 150.0]
    assert trimmed.wind_speed_m_s.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert trimmed.moisture_dead_1h.tolist() == [
        np.float32(0.03),
        np.float32(0.04),
        np.float32(0.05),
        np.float32(0.06),
    ]
    assert trimmed.time_origin is not None
    assert trimmed.time_origin.isoformat() == "2022-07-01T03:30:00+00:00"
    assert trimmed.metadata["spinup_sample_count"] == 3
