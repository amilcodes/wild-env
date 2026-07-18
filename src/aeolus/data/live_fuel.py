"""NFDRS-v4-style growing-season and live-fuel moisture diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from aeolus.data.weather import WeatherForcing


def ramp(value: np.ndarray | float, lower: float, upper: float) -> np.ndarray:
    """Linear ramp constrained to [0, 1]."""

    if upper <= lower:
        raise ValueError("ramp upper limit must exceed lower limit")
    return np.clip((np.asarray(value, dtype=np.float64) - lower) / (upper - lower), 0.0, 1.0)


def saturation_vapor_pressure_pa(temperature_c: np.ndarray | float) -> np.ndarray:
    """Saturation vapor pressure using the standard exponential approximation."""

    temperature = np.asarray(temperature_c, dtype=np.float64)
    return 610.94 * np.exp(17.625 * temperature / (temperature + 243.04))


def daylength_seconds(latitude_deg: float, day_of_year: np.ndarray | float) -> np.ndarray:
    """Astronomical day length using solar declination and sunset hour angle."""

    latitude = np.deg2rad(np.clip(float(latitude_deg), -89.8, 89.8))
    day = np.asarray(day_of_year, dtype=np.float64)
    declination = 0.409 * np.sin(2.0 * np.pi * day / 365.0 - 1.39)
    cosine = np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
    sunset_hour_angle = np.arccos(cosine)
    return 86_400.0 * sunset_hour_angle / np.pi


def herbaceous_curing_fraction(moisture_kg_kg: np.ndarray | float) -> np.ndarray:
    """Scott-Burgan dynamic herbaceous load transfer fraction.

    Transfer begins below 120 percent live moisture and is complete at
    30 percent.
    """

    moisture = np.asarray(moisture_kg_kg, dtype=np.float64)
    return np.clip((1.20 - moisture) / 0.90, 0.0, 1.0).astype(np.float32)


def _origin(forcing: WeatherForcing, start_datetime: datetime | str | None) -> datetime:
    if start_datetime is None:
        origin = forcing.time_origin
        if origin is None:
            raise ValueError("start_datetime is required when forcing has no CF time origin")
        return origin
    if isinstance(start_datetime, str):
        origin = datetime.fromisoformat(start_datetime.replace("Z", "+00:00"))
    else:
        origin = start_datetime
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=timezone.utc)
    return origin.astimezone(timezone.utc)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(
        (np.zeros((1, *values.shape[1:]), dtype=np.float64), np.cumsum(values, axis=0)),
        axis=0,
    )
    output = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        lower = max(0, index + 1 - window)
        output[index] = cumulative[index + 1] - cumulative[lower]
    return output


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    summed = _rolling_sum(values, window)
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return summed / counts.reshape((-1,) + (1,) * (values.ndim - 1))


def derive_live_fuel_moisture(
    forcing: WeatherForcing,
    *,
    latitude_deg: float,
    longitude_deg: float,
    start_datetime: datetime | str | None = None,
    precipitation_window_days: int = 28,
    gsi_window_days: int = 28,
    greenup_threshold: float = 0.20,
) -> WeatherForcing:
    """Derive hourly live herbaceous and woody moisture from daily GSI.

    The daily limiting factors and ranges follow NFDRS version 4: minimum
    temperature, maximum vapor-pressure deficit, photoperiod, and trailing
    precipitation.  The unsmoothed product is averaged over 28 days before
    conversion to the documented live-moisture ranges.
    """

    forcing.validate()
    if precipitation_window_days < 1 or gsi_window_days < 1:
        raise ValueError("rolling windows must contain at least one day")
    if not 0.0 <= greenup_threshold < 1.0:
        raise ValueError("greenup_threshold must be within [0, 1)")

    origin = _origin(forcing, start_datetime)
    local_offset = timedelta(hours=float(longitude_deg) / 15.0)
    timestamps = [origin + timedelta(minutes=float(minute)) + local_offset for minute in forcing.minute]
    dates = np.asarray([item.date().toordinal() for item in timestamps], dtype=np.int64)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    field_shape = forcing.air_temperature_c.shape[1:]
    daily_shape = (len(unique_dates), *field_shape)
    daily_tmin = np.empty(daily_shape, dtype=np.float64)
    daily_vpd_max = np.empty(daily_shape, dtype=np.float64)
    daily_precipitation = np.zeros(daily_shape, dtype=np.float64)
    precipitation = (
        np.zeros_like(forcing.air_temperature_c, dtype=np.float64)
        if forcing.precipitation_rate_mm_h is None
        else np.asarray(forcing.precipitation_rate_mm_h, dtype=np.float64)
    )
    temperature = np.asarray(forcing.air_temperature_c, dtype=np.float64)
    humidity = np.asarray(forcing.relative_humidity_pct, dtype=np.float64)
    vpd = saturation_vapor_pressure_pa(temperature) * (1.0 - humidity / 100.0)

    sample_duration_h = np.empty(len(forcing.minute), dtype=np.float64)
    if len(sample_duration_h) == 1:
        sample_duration_h[0] = 1.0
    else:
        sample_duration_h[1:] = np.diff(forcing.minute) / 60.0
        sample_duration_h[0] = sample_duration_h[1]
    for day_index in range(len(unique_dates)):
        samples = np.flatnonzero(inverse == day_index)
        daily_tmin[day_index] = np.min(temperature[samples], axis=0)
        daily_vpd_max[day_index] = np.max(vpd[samples], axis=0)
        weights = sample_duration_h[samples].reshape((-1,) + (1,) * len(field_shape))
        daily_precipitation[day_index] = np.sum(precipitation[samples] * weights, axis=0)

    day_of_year = np.asarray(
        [datetime.fromordinal(int(value)).timetuple().tm_yday for value in unique_dates],
        dtype=np.float64,
    )
    photoperiod = daylength_seconds(latitude_deg, day_of_year).reshape((-1,) + (1,) * len(field_shape))
    trailing_precipitation = _rolling_sum(daily_precipitation, precipitation_window_days)
    temperature_index = ramp(daily_tmin, -2.0, 5.0)
    vpd_index = 1.0 - ramp(daily_vpd_max, 900.0, 4100.0)
    daylength_index = ramp(photoperiod, 36_000.0, 39_600.0)
    precipitation_index = ramp(trailing_precipitation, 0.0, 10.0)
    daily_gsi = temperature_index * vpd_index * daylength_index * precipitation_index
    smoothed_gsi = _rolling_mean(daily_gsi, gsi_window_days)
    normalized_gsi = np.clip(
        (smoothed_gsi - greenup_threshold) / (1.0 - greenup_threshold),
        0.0,
        1.0,
    )
    herbaceous_daily = 0.30 + (2.50 - 0.30) * normalized_gsi
    woody_daily = 0.60 + (2.00 - 0.60) * normalized_gsi
    herbaceous = herbaceous_daily[inverse].astype(np.float32)
    woody = woody_daily[inverse].astype(np.float32)

    history = str(forcing.metadata.get("history", "")).strip()
    derivation = "live-fuel moisture derived from NFDRS-v4-style 28-day GSI"
    return replace(
        forcing,
        metadata={
            **forcing.metadata,
            "history": f"{history}; {derivation}".strip("; "),
            "live_fuel_moisture_model": "nfdrs-v4-gsi",
            "live_fuel_gsi_precipitation_window_days": precipitation_window_days,
            "live_fuel_gsi_smoothing_days": gsi_window_days,
            "live_fuel_greenup_threshold": greenup_threshold,
            "live_fuel_local_solar_longitude_deg": float(longitude_deg),
            "live_fuel_spinup_days_available": max(0, len(unique_dates) - 1),
            "herbaceous_moisture_range_kg_kg": [0.30, 2.50],
            "woody_moisture_range_kg_kg": [0.60, 2.00],
        },
        moisture_live_herbaceous=herbaceous,
        moisture_live_woody=woody,
    )
