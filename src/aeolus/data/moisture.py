"""Dead-fuel moisture state integration used by preparation and simulation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aeolus.data.weather import WeatherForcing


def dead_fuel_moisture_equilibria(
    temperature_c: float | np.ndarray,
    relative_humidity_pct: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Van Wagner-Pickett drying and wetting equilibria in kg/kg.

    These are the equations used by the WRF-SFIRE equilibrium time-lag
    implementation. The gap between the equilibria represents sorption
    hysteresis: moisture remains unchanged while it lies inside the gap.
    """

    humidity = np.clip(
        np.asarray(relative_humidity_pct, dtype=np.float64),
        0.0,
        100.0,
    )
    temperature = np.asarray(temperature_c, dtype=np.float64)
    temperature_term = 0.18 * (21.1 - temperature) * (1.0 - np.exp(-0.115 * humidity))
    drying = 0.942 * humidity**0.679 + 0.4994e-4 * np.exp(0.1 * humidity) + temperature_term
    wetting = 0.618 * humidity**0.753 + 0.4540e-4 * np.exp(0.1 * humidity) + temperature_term
    upper = np.maximum(drying, wetting) * 0.01
    lower = np.minimum(drying, wetting) * 0.01
    return (
        np.clip(upper, 0.0, 3.0).astype(np.float32),
        np.clip(lower, 0.0, 3.0).astype(np.float32),
    )


def advance_dead_fuel_moisture(
    moisture: np.ndarray,
    *,
    temperature_c: float | np.ndarray,
    relative_humidity_pct: float | np.ndarray,
    precipitation_rate_mm_h: float | np.ndarray,
    lag_hours: float,
    dt_min: float,
    saturation_moisture: float = 2.5,
    rain_soaking_lag_multiplier: float = 14.0,
    rain_threshold_mm_h: float = 0.05,
    rain_saturation_mm_h: float = 8.0,
) -> np.ndarray:
    """Advance one moisture class with an exact exponential time-lag step."""

    if lag_hours <= 0.0 or dt_min < 0.0:
        raise ValueError("fuel-moisture lag must be positive and dt non-negative")
    current = np.asarray(moisture, dtype=np.float64)
    rain = np.maximum(
        np.asarray(precipitation_rate_mm_h, dtype=np.float64),
        0.0,
    )
    drying, wetting = dead_fuel_moisture_equilibria(
        temperature_c,
        relative_humidity_pct,
    )
    dry_target = np.maximum(
        np.minimum(current, np.asarray(drying, dtype=np.float64)),
        np.asarray(wetting, dtype=np.float64),
    )
    rain_excess = np.maximum(rain - rain_threshold_mm_h, 0.0)
    rain_response = 1.0 - np.exp(-rain_excess / rain_saturation_mm_h)
    dry_rate_per_min = 1.0 / (lag_hours * 60.0)
    rain_rate_per_min = rain_response / (rain_soaking_lag_multiplier * lag_hours * 60.0)
    raining = rain_excess > 0.0
    target = np.where(raining, saturation_moisture, dry_target)
    rate = np.where(raining, rain_rate_per_min, dry_rate_per_min)
    fraction = -np.expm1(-dt_min * rate)
    advanced = current + (target - current) * fraction
    return np.clip(advanced, 0.0, 3.0).astype(np.float32)


def derive_dead_fuel_moisture(
    forcing: WeatherForcing,
    *,
    initial_1h: float | np.ndarray | None = None,
    initial_10h: float | np.ndarray | None = None,
    initial_100h: float | np.ndarray | None = None,
) -> WeatherForcing:
    """Integrate 1/10/100-hour moisture along an existing weather series.

    Atmospheric inputs are evaluated at interval midpoints. Each step uses
    the exact solution for constant midpoint coefficients, matching the
    numerical structure documented for WRF-SFIRE.
    """

    forcing.validate()
    shape = forcing.wind_speed_m_s.shape
    output_shape = shape
    precipitation = (
        np.zeros(shape, dtype=np.float32)
        if forcing.precipitation_rate_mm_h is None
        else forcing.precipitation_rate_mm_h
    )
    initial_drying, initial_wetting = dead_fuel_moisture_equilibria(
        forcing.air_temperature_c[0],
        forcing.relative_humidity_pct[0],
    )
    initial_equilibrium = 0.5 * (initial_drying + initial_wetting)

    def integrate(
        lag_hours: float,
        supplied: float | np.ndarray | None,
        existing: np.ndarray | None,
    ) -> np.ndarray:
        if existing is not None:
            return np.asarray(existing, dtype=np.float32).copy()
        field = np.empty(output_shape, dtype=np.float32)
        field[0] = (
            initial_equilibrium
            if supplied is None
            else np.broadcast_to(np.asarray(supplied, dtype=np.float32), shape[1:])
        )
        for index in range(1, len(forcing.minute)):
            dt_min = float(forcing.minute[index] - forcing.minute[index - 1])
            temperature = 0.5 * (forcing.air_temperature_c[index - 1] + forcing.air_temperature_c[index])
            humidity = 0.5 * (forcing.relative_humidity_pct[index - 1] + forcing.relative_humidity_pct[index])
            rain = 0.5 * (precipitation[index - 1] + precipitation[index])
            field[index] = advance_dead_fuel_moisture(
                field[index - 1],
                temperature_c=temperature,
                relative_humidity_pct=humidity,
                precipitation_rate_mm_h=rain,
                lag_hours=lag_hours,
                dt_min=dt_min,
            )
        return field

    history = str(forcing.metadata.get("history", "")).strip()
    derivation = (
        "dead-fuel moisture integrated with WRF-SFIRE-compatible "
        "Van Wagner-Pickett hysteretic time-lag equations"
    )
    return replace(
        forcing,
        metadata={
            **forcing.metadata,
            "history": f"{history}; {derivation}".strip("; "),
            "fuel_moisture_model": "wrf-sfire-equilibrium-time-lag",
            "fuel_moisture_initialization": (
                "provided"
                if any(value is not None for value in (initial_1h, initial_10h, initial_100h))
                else "midpoint-of-initial-wetting-and-drying-equilibria"
            ),
        },
        moisture_dead_1h=integrate(
            1.0,
            initial_1h,
            forcing.moisture_dead_1h,
        ),
        moisture_dead_10h=integrate(
            10.0,
            initial_10h,
            forcing.moisture_dead_10h,
        ),
        moisture_dead_100h=integrate(
            100.0,
            initial_100h,
            forcing.moisture_dead_100h,
        ),
    )
