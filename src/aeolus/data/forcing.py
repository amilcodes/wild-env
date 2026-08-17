"""Incident forcing analysis and causal forecast-state construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from aeolus.data.weather import WeatherForcing


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _forecast_field(
    value: float | np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"initial moisture field shape {array.shape} does not match forecast grid {shape}")
    return array.copy()


def _causal_moisture_sample(
    forcing: WeatherForcing,
    minute: float,
) -> dict[str, float | np.ndarray]:
    """Sample only values whose timestamps do not follow the query time."""

    index = int(np.searchsorted(forcing.minute, minute, side="right")) - 1
    index = min(max(index, 0), len(forcing.minute) - 1)
    result: dict[str, float | np.ndarray] = {}
    for name in (
        "moisture_dead_1h",
        "moisture_dead_10h",
        "moisture_dead_100h",
        "moisture_live_herbaceous",
        "moisture_live_woody",
    ):
        values = getattr(forcing, name)
        if values is not None:
            sample = values[index]
            result[name] = float(sample) if np.ndim(sample) == 0 else sample.astype(np.float32)
    return result


def initialize_causal_forecast_moisture(
    forecast: WeatherForcing,
    background: WeatherForcing,
    *,
    issue_time: datetime | str,
    default_live_herbaceous: float = 0.75,
    default_live_woody: float = 0.60,
) -> WeatherForcing:
    """Attach a forecast-issue-consistent fuel-moisture trajectory.

    Dead-fuel moisture is initialized at the first forecast valid time from a
    background trajectory and then integrated only with fields from the
    archived forecast cycle. Live-fuel moisture changes on much longer time
    scales than a 48-hour HRRR window, so its issue-time state is held fixed.
    The background may be a retrospective analysis because only samples at or
    before the issue time are read from it.
    """

    from aeolus.data.moisture import derive_dead_fuel_moisture

    forecast.validate()
    background.validate()
    if forecast.wind_speed_m_s.ndim != 3:
        raise ValueError("operational forecast forcing must be gridded")
    forecast_origin = forecast.time_origin
    background_origin = background.time_origin
    if forecast_origin is None or background_origin is None:
        raise ValueError("forecast and background require absolute CF time origins")
    issue = _utc(issue_time)
    first_valid = forecast_origin + timedelta(minutes=float(forecast.minute[0]))
    if first_valid > issue:
        raise ValueError("first forecast valid time follows the declared issue time")
    background_start = background_origin + timedelta(minutes=float(background.minute[0]))
    background_end = background_origin + timedelta(minutes=float(background.minute[-1]))
    if not background_start <= first_valid <= background_end:
        raise ValueError("background forcing does not cover moisture initialization time")
    if not background_start <= issue <= background_end:
        raise ValueError("background forcing does not cover forecast issue time")

    initial_minute = (first_valid - background_origin).total_seconds() / 60.0
    issue_minute = (issue - background_origin).total_seconds() / 60.0
    initial_sample = _causal_moisture_sample(background, initial_minute)
    issue_sample = _causal_moisture_sample(background, issue_minute)
    shape = forecast.wind_speed_m_s.shape[1:]
    initial_values: dict[str, np.ndarray | None] = {}
    defaults = {
        "moisture_dead_1h": 0.08,
        "moisture_dead_10h": 0.10,
        "moisture_dead_100h": 0.12,
    }
    for name, default in defaults.items():
        initial_values[name] = _forecast_field(
            initial_sample.get(name, default),
            shape,
        )
    conditioned = derive_dead_fuel_moisture(
        forecast,
        initial_1h=initial_values["moisture_dead_1h"],
        initial_10h=initial_values["moisture_dead_10h"],
        initial_100h=initial_values["moisture_dead_100h"],
    )
    live_herbaceous = _forecast_field(
        issue_sample.get("moisture_live_herbaceous", default_live_herbaceous),
        shape,
    )
    live_woody = _forecast_field(
        issue_sample.get("moisture_live_woody", default_live_woody),
        shape,
    )
    count = len(conditioned.minute)
    result = WeatherForcing(
        minute=conditioned.minute,
        wind_speed_m_s=conditioned.wind_speed_m_s,
        wind_direction_deg=conditioned.wind_direction_deg,
        air_temperature_c=conditioned.air_temperature_c,
        relative_humidity_pct=conditioned.relative_humidity_pct,
        precipitation_rate_mm_h=conditioned.precipitation_rate_mm_h,
        moisture_dead_1h=conditioned.moisture_dead_1h,
        moisture_dead_10h=conditioned.moisture_dead_10h,
        moisture_dead_100h=conditioned.moisture_dead_100h,
        moisture_live_herbaceous=np.broadcast_to(
            live_herbaceous,
            (count, *shape),
        ).copy(),
        moisture_live_woody=np.broadcast_to(
            live_woody,
            (count, *shape),
        ).copy(),
        metadata={
            **conditioned.metadata,
            "forecast_issue_time": issue.isoformat().replace("+00:00", "Z"),
            "fuel_moisture_background_source": str(background.metadata.get("source", "unknown")),
            "fuel_moisture_initialization_time": first_valid.isoformat().replace("+00:00", "Z"),
            "fuel_moisture_initialization_causal": True,
            "fuel_moisture_background_sampling": (
                "last sample at or before query time; no future interpolation"
            ),
            "dead_fuel_future_driver": "single archived forecast cycle",
            "live_fuel_future_evolution": "issue-time state held fixed",
            "live_fuel_default_used": {
                "herbaceous": "moisture_live_herbaceous" not in issue_sample,
                "woody": "moisture_live_woody" not in issue_sample,
            },
        },
    )
    result.validate()
    return result


@dataclass(frozen=True)
class StationObservation:
    minute: float
    x_m: float
    y_m: float
    wind_speed_m_s: float | None = None
    wind_from_direction_deg: float | None = None
    air_temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    moisture_dead_1h: float | None = None
    moisture_dead_10h: float | None = None
    moisture_dead_100h: float | None = None
    moisture_live_herbaceous: float | None = None
    moisture_live_woody: float | None = None
    station_id: str = "unknown"


@dataclass(frozen=True)
class IncidentForcingAnalysis:
    forcing: WeatherForcing
    wind_correction_std_m_s: np.ndarray
    temperature_analysis_std_c: np.ndarray
    humidity_analysis_std_pct: np.ndarray
    station_count: np.ndarray
    diagnostics: dict[str, float | int]


def _grid(value: float | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"background field shape {array.shape} does not match analysis grid {shape}")
    return array


def _optimal_interpolation(
    grid_x_m: np.ndarray,
    grid_y_m: np.ndarray,
    station_x_m: np.ndarray,
    station_y_m: np.ndarray,
    innovations: np.ndarray,
    *,
    length_scale_m: float,
    observation_error_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if innovations.size == 0:
        shape = grid_x_m.shape
        return np.zeros(shape, dtype=np.float64), np.ones(shape, dtype=np.float64)
    obs_dx = station_x_m[:, None] - station_x_m[None, :]
    obs_dy = station_y_m[:, None] - station_y_m[None, :]
    correlation = np.exp(-0.5 * (obs_dx * obs_dx + obs_dy * obs_dy) / length_scale_m**2)
    system = correlation + observation_error_ratio**2 * np.eye(len(innovations))
    coefficients = np.linalg.solve(system, innovations)
    grid_dx = grid_x_m[..., None] - station_x_m
    grid_dy = grid_y_m[..., None] - station_y_m
    grid_correlation = np.exp(-0.5 * (grid_dx * grid_dx + grid_dy * grid_dy) / length_scale_m**2)
    correction = np.einsum("...n,n->...", grid_correlation, coefficients)
    gain = np.linalg.solve(system, grid_correlation.reshape(-1, len(innovations)).T)
    posterior_variance = 1.0 - np.einsum(
        "in,ni->i",
        grid_correlation.reshape(-1, len(innovations)),
        gain,
    )
    return correction, np.sqrt(np.clip(posterior_variance, 0.0, 1.0)).reshape(grid_x_m.shape)


def analyze_incident_forcing(
    background: WeatherForcing,
    observations: list[StationObservation],
    grid_x_m: np.ndarray,
    grid_y_m: np.ndarray,
    *,
    length_scale_m: float = 12_000.0,
    observation_error_ratio: float = 0.35,
    time_window_min: float = 40.0,
) -> IncidentForcingAnalysis:
    """Fuse RAWS-like observations with a gridded weather background.

    The analysis is an explicit optimum-interpolation correction.  Wind is
    analyzed in Cartesian components; direction is never averaged as an angle.
    Posterior standard-deviation ratios are retained for downstream ensembles.
    """

    background.validate()
    if grid_x_m.shape != grid_y_m.shape or grid_x_m.ndim != 2:
        raise ValueError("analysis coordinates must be two-dimensional common grids")
    if length_scale_m <= 0.0 or observation_error_ratio <= 0.0:
        raise ValueError("analysis covariance controls must be positive")
    if time_window_min < 0.0:
        raise ValueError("time window cannot be negative")

    shape = grid_x_m.shape
    count = len(background.minute)
    output_shape = (count, *shape)
    speed = np.empty(output_shape, dtype=np.float32)
    direction = np.empty(output_shape, dtype=np.float32)
    temperature = np.empty(output_shape, dtype=np.float32)
    humidity = np.empty(output_shape, dtype=np.float32)
    precipitation = np.empty(output_shape, dtype=np.float32)
    u_correction = np.zeros(output_shape, dtype=np.float32)
    v_correction = np.zeros(output_shape, dtype=np.float32)
    wind_std = np.ones(output_shape, dtype=np.float32)
    temperature_std = np.ones(output_shape, dtype=np.float32)
    humidity_std = np.ones(output_shape, dtype=np.float32)
    station_count = np.zeros(count, dtype=np.int32)

    moisture_defaults = {
        "moisture_dead_1h": 0.08,
        "moisture_dead_10h": 0.10,
        "moisture_dead_100h": 0.12,
        "moisture_live_herbaceous": 0.75,
        "moisture_live_woody": 0.60,
    }
    moisture_fields = {name: np.empty(output_shape, dtype=np.float32) for name in moisture_defaults}

    flat_x = np.asarray(grid_x_m, dtype=np.float64)
    flat_y = np.asarray(grid_y_m, dtype=np.float64)
    for time_index, minute in enumerate(background.minute):
        sample = background.at_minute(float(minute))
        base_speed = _grid(sample["wind_speed_m_s"], shape)
        base_direction = np.deg2rad(_grid(sample["wind_direction_deg"], shape))
        base_u = -base_speed * np.sin(base_direction)
        base_v = -base_speed * np.cos(base_direction)
        base_temperature = _grid(sample["air_temperature_c"], shape)
        base_humidity = _grid(sample["relative_humidity_pct"], shape)
        base_precipitation = _grid(sample["precipitation_rate_mm_h"], shape)
        nearby = [item for item in observations if abs(item.minute - float(minute)) <= time_window_min]
        station_count[time_index] = len(nearby)

        def analyze(
            attribute: str,
            base: np.ndarray,
            *,
            transform=None,
        ) -> tuple[np.ndarray, np.ndarray]:
            selected = [item for item in nearby if getattr(item, attribute) is not None]
            if not selected:
                return np.zeros(shape), np.ones(shape)
            sx = np.asarray([item.x_m for item in selected])
            sy = np.asarray([item.y_m for item in selected])
            ix = np.abs(grid_x_m[0, :] - sx[:, None]).argmin(axis=1)
            iy = np.abs(grid_y_m[:, 0] - sy[:, None]).argmin(axis=1)
            values = np.asarray(
                [
                    (transform(item) if transform is not None else float(getattr(item, attribute)))
                    for item in selected
                ],
                dtype=np.float64,
            )
            innovations = values - base[iy, ix]
            return _optimal_interpolation(
                flat_x,
                flat_y,
                sx,
                sy,
                innovations,
                length_scale_m=length_scale_m,
                observation_error_ratio=observation_error_ratio,
            )

        wind_selected = [
            item
            for item in nearby
            if item.wind_speed_m_s is not None and item.wind_from_direction_deg is not None
        ]
        if wind_selected:
            sx = np.asarray([item.x_m for item in wind_selected])
            sy = np.asarray([item.y_m for item in wind_selected])
            ix = np.abs(grid_x_m[0, :] - sx[:, None]).argmin(axis=1)
            iy = np.abs(grid_y_m[:, 0] - sy[:, None]).argmin(axis=1)
            obs_speed = np.asarray([item.wind_speed_m_s for item in wind_selected])
            obs_direction = np.deg2rad([item.wind_from_direction_deg for item in wind_selected])
            obs_u = -obs_speed * np.sin(obs_direction)
            obs_v = -obs_speed * np.cos(obs_direction)
            u_delta, u_std = _optimal_interpolation(
                flat_x,
                flat_y,
                sx,
                sy,
                obs_u - base_u[iy, ix],
                length_scale_m=length_scale_m,
                observation_error_ratio=observation_error_ratio,
            )
            v_delta, v_std = _optimal_interpolation(
                flat_x,
                flat_y,
                sx,
                sy,
                obs_v - base_v[iy, ix],
                length_scale_m=length_scale_m,
                observation_error_ratio=observation_error_ratio,
            )
            u_correction[time_index] = u_delta
            v_correction[time_index] = v_delta
            wind_std[time_index] = np.sqrt(0.5 * (u_std * u_std + v_std * v_std))

        temperature_delta, temperature_sigma = analyze(
            "air_temperature_c",
            base_temperature,
        )
        humidity_delta, humidity_sigma = analyze(
            "relative_humidity_pct",
            base_humidity,
        )
        speed[time_index] = base_speed
        direction[time_index] = np.rad2deg(base_direction) % 360.0
        temperature[time_index] = base_temperature + temperature_delta
        humidity[time_index] = np.clip(base_humidity + humidity_delta, 0.0, 100.0)
        precipitation[time_index] = base_precipitation
        temperature_std[time_index] = temperature_sigma
        humidity_std[time_index] = humidity_sigma

        for name, default in moisture_defaults.items():
            base_value = _grid(sample.get(name, default), shape)
            delta, _ = analyze(name, base_value)
            moisture_fields[name][time_index] = np.clip(
                base_value + delta,
                0.01,
                3.0,
            )

    forcing = WeatherForcing(
        minute=background.minute.copy(),
        wind_speed_m_s=speed,
        wind_direction_deg=direction,
        air_temperature_c=temperature,
        relative_humidity_pct=humidity,
        precipitation_rate_mm_h=precipitation,
        moisture_dead_1h=moisture_fields["moisture_dead_1h"],
        moisture_dead_10h=moisture_fields["moisture_dead_10h"],
        moisture_dead_100h=moisture_fields["moisture_dead_100h"],
        moisture_live_herbaceous=moisture_fields["moisture_live_herbaceous"],
        moisture_live_woody=moisture_fields["moisture_live_woody"],
        wind_u_correction_m_s=u_correction,
        wind_v_correction_m_s=v_correction,
        metadata={
            **background.metadata,
            "analysis": "Gaussian optimum interpolation",
            "station_observations": len(observations),
            "length_scale_m": length_scale_m,
            "observation_error_ratio": observation_error_ratio,
        },
    )
    forcing.validate()
    return IncidentForcingAnalysis(
        forcing=forcing,
        wind_correction_std_m_s=wind_std,
        temperature_analysis_std_c=temperature_std,
        humidity_analysis_std_pct=humidity_std,
        station_count=station_count,
        diagnostics={
            "station_observations": len(observations),
            "analysis_times": count,
            "times_with_stations": int((station_count > 0).sum()),
            "maximum_stations_in_window": int(station_count.max(initial=0)),
            "length_scale_m": float(length_scale_m),
            "observation_error_ratio": float(observation_error_ratio),
        },
    )
