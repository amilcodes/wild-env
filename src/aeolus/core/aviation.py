"""Tactical aircraft performance and route-feasibility calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import atan2, ceil, degrees, hypot
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aeolus.config import AirspaceVolumeSpec
from aeolus.core.state import ResourceRuntime

if TYPE_CHECKING:
    from aeolus.core.simulator import AeolusSimulator


@dataclass(frozen=True)
class TacticalPerformanceSurface:
    """Flight-manual-derived tactical performance on two declared axes."""

    density_altitude_m: np.ndarray
    payload_fraction: np.ndarray
    true_airspeed_m_s: np.ndarray
    endurance_multiplier: np.ndarray
    maximum_payload_fraction: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        altitude = np.asarray(self.density_altitude_m, dtype=np.float64)
        payload = np.asarray(self.payload_fraction, dtype=np.float64)
        expected = (altitude.size, payload.size)
        if altitude.ndim != 1 or payload.ndim != 1 or min(expected) < 2:
            raise ValueError("performance axes must be one-dimensional with at least two values")
        if np.any(np.diff(altitude) <= 0.0) or np.any(np.diff(payload) <= 0.0):
            raise ValueError("performance axes must be strictly increasing")
        if payload[0] < 0.0 or payload[-1] > 1.0:
            raise ValueError("performance payload fractions must remain within [0, 1]")
        for name, values in (
            ("true_airspeed_m_s", self.true_airspeed_m_s),
            ("endurance_multiplier", self.endurance_multiplier),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != expected or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} must be a positive {expected} table")
        maximum_payload = np.asarray(self.maximum_payload_fraction, dtype=np.float64)
        if (
            maximum_payload.shape != altitude.shape
            or np.any(~np.isfinite(maximum_payload))
            or np.any((maximum_payload <= 0.0) | (maximum_payload > 1.0))
        ):
            raise ValueError("maximum payload must contain one fraction per altitude")
        if not str(self.metadata.get("source", "")).strip():
            raise ValueError("performance surface requires source provenance")


@dataclass(frozen=True)
class LegPerformance:
    distance_m: float
    track_heading_deg: float
    maximum_terrain_m_msl: float
    planned_altitude_m_msl: float
    density_altitude_m: float
    true_airspeed_m_s: float
    tailwind_m_s: float
    crosswind_m_s: float
    groundspeed_m_s: float
    travel_min: float
    available_endurance_min: float
    maximum_payload_fraction: float
    intersected_airspace_volumes: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations


def _surface_from_payload(payload: dict[str, Any]) -> TacticalPerformanceSurface:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported tactical-performance schema")
    surface = TacticalPerformanceSurface(
        density_altitude_m=np.asarray(payload["density_altitude_m"], dtype=np.float64),
        payload_fraction=np.asarray(payload["payload_fraction"], dtype=np.float64),
        true_airspeed_m_s=np.asarray(payload["true_airspeed_m_s"], dtype=np.float64),
        endurance_multiplier=np.asarray(payload["endurance_multiplier"], dtype=np.float64),
        maximum_payload_fraction=np.asarray(
            payload["maximum_payload_fraction"],
            dtype=np.float64,
        ),
        metadata=dict(payload.get("metadata", {})),
    )
    surface.validate()
    return surface


@lru_cache(maxsize=32)
def load_tactical_performance_surface(
    path: str | Path,
) -> TacticalPerformanceSurface:
    """Load and validate a small JSON performance surface."""

    source = Path(path)
    return _surface_from_payload(json.loads(source.read_text(encoding="utf-8")))


def density_altitude_m(
    elevation_m_msl: float,
    air_temperature_c: float,
) -> float:
    """Approximate density altitude from pressure altitude and ISA deviation.

    Scenario elevation is used as pressure altitude when pressure/altimeter
    setting is unavailable.  The standard pilot approximation is sufficient
    for feasibility screening; a flight-manual workflow should supply actual
    pressure altitude.
    """

    isa_temperature_c = 15.0 - 0.0065 * elevation_m_msl
    return float(elevation_m_msl + 36.576 * (air_temperature_c - isa_temperature_c))


def _axis_bracket(value: float, axis: np.ndarray) -> tuple[int, int, float]:
    clipped = float(np.clip(value, axis[0], axis[-1]))
    upper = int(np.clip(np.searchsorted(axis, clipped, side="right"), 1, len(axis) - 1))
    lower = upper - 1
    fraction = (clipped - float(axis[lower])) / max(
        float(axis[upper] - axis[lower]),
        1e-12,
    )
    return lower, upper, fraction


def _bilinear(
    table: np.ndarray,
    x_value: float,
    x_axis: np.ndarray,
    y_value: float,
    y_axis: np.ndarray,
) -> float:
    x0, x1, xf = _axis_bracket(x_value, x_axis)
    y0, y1, yf = _axis_bracket(y_value, y_axis)
    lower = (1.0 - yf) * table[x0, y0] + yf * table[x0, y1]
    upper = (1.0 - yf) * table[x1, y0] + yf * table[x1, y1]
    return float((1.0 - xf) * lower + xf * upper)


def _linear(values: np.ndarray, value: float, axis: np.ndarray) -> float:
    lower, upper, fraction = _axis_bracket(value, axis)
    return float((1.0 - fraction) * values[lower] + fraction * values[upper])


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)

    def on_segment(
        start: tuple[float, float],
        point: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        return (
            min(start[0], end[0]) - 1e-12 <= point[0] <= max(start[0], end[0]) + 1e-12
            and min(start[1], end[1]) - 1e-12 <= point[1] <= max(start[1], end[1]) + 1e-12
        )

    if first * second < 0.0 and third * fourth < 0.0:
        return True
    return (
        (abs(first) <= 1e-12 and on_segment(a, c, b))
        or (abs(second) <= 1e-12 and on_segment(a, d, b))
        or (abs(third) <= 1e-12 and on_segment(c, a, d))
        or (abs(fourth) <= 1e-12 and on_segment(c, b, d))
    )


def segment_intersects_polygon(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    polygon_xy: tuple[tuple[float, float], ...],
) -> bool:
    if _point_in_polygon(start_xy, polygon_xy) or _point_in_polygon(
        end_xy,
        polygon_xy,
    ):
        return True
    previous = polygon_xy[-1]
    for current in polygon_xy:
        if _segments_intersect(start_xy, end_xy, previous, current):
            return True
        previous = current
    return False


def _sample_raster_along_leg(
    values: np.ndarray | float,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim != 2:
        raise ValueError("leg fields must be scalar or two-dimensional")
    samples = max(
        2,
        int(
            ceil(
                hypot(
                    end_xy[0] - start_xy[0],
                    end_xy[1] - start_xy[1],
                )
            )
        )
        + 1,
    )
    x = np.clip(
        np.rint(np.linspace(start_xy[0], end_xy[0], samples)).astype(np.int64),
        0,
        array.shape[1] - 1,
    )
    y = np.clip(
        np.rint(np.linspace(start_xy[1], end_xy[1], samples)).astype(np.int64),
        0,
        array.shape[0] - 1,
    )
    return array[y, x]


def evaluate_leg_performance(
    resource: ResourceRuntime,
    *,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    cell_size_m: float,
    elevation_m: np.ndarray,
    air_temperature_c: np.ndarray | float,
    wind_speed_m_s: np.ndarray | float,
    wind_from_direction_deg: np.ndarray | float,
    minute: float,
    airspace_volumes: tuple[AirspaceVolumeSpec, ...] = (),
    payload_fraction: float | None = None,
) -> LegPerformance:
    """Evaluate one straight tactical leg against declared constraints."""

    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    distance_m = hypot(dx, dy) * cell_size_m
    track_norm = max(hypot(dx, dy), 1e-12)
    track_x, track_y = dx / track_norm, dy / track_norm
    heading = float((degrees(atan2(track_x, -track_y)) + 360.0) % 360.0)
    terrain = _sample_raster_along_leg(elevation_m, start_xy, end_xy)
    temperature = _sample_raster_along_leg(
        air_temperature_c,
        start_xy,
        end_xy,
    )
    maximum_terrain = float(np.max(terrain))
    planned_altitude = maximum_terrain + resource.spec.cruise_altitude_agl_m
    density_altitude = density_altitude_m(
        maximum_terrain,
        float(np.mean(temperature)),
    )
    load = resource.payload_fraction if payload_fraction is None else float(payload_fraction)
    if resource.spec.performance_surface_path is None:
        true_airspeed = resource.spec.cruise_speed_m_s
        endurance_multiplier = 1.0
        maximum_payload = 1.0
        altitude_domain: tuple[float, float] | None = None
    else:
        surface = load_tactical_performance_surface(resource.spec.performance_surface_path)
        true_airspeed = _bilinear(
            surface.true_airspeed_m_s,
            density_altitude,
            surface.density_altitude_m,
            load,
            surface.payload_fraction,
        )
        endurance_multiplier = _bilinear(
            surface.endurance_multiplier,
            density_altitude,
            surface.density_altitude_m,
            load,
            surface.payload_fraction,
        )
        maximum_payload = _linear(
            surface.maximum_payload_fraction,
            density_altitude,
            surface.density_altitude_m,
        )
        altitude_domain = (
            float(surface.density_altitude_m[0]),
            float(surface.density_altitude_m[-1]),
        )

    wind_speed_samples = _sample_raster_along_leg(
        wind_speed_m_s,
        start_xy,
        end_xy,
    )
    wind_direction_samples = _sample_raster_along_leg(
        wind_from_direction_deg,
        start_xy,
        end_xy,
    )
    if wind_speed_samples.size == 1 and wind_direction_samples.size > 1:
        wind_speed_samples = np.full(
            wind_direction_samples.shape,
            float(wind_speed_samples[0]),
        )
    if wind_direction_samples.size == 1 and wind_speed_samples.size > 1:
        wind_direction_samples = np.full(
            wind_speed_samples.shape,
            float(wind_direction_samples[0]),
        )
    wind_radians = np.deg2rad(wind_direction_samples)
    wind_x = float(np.mean(-wind_speed_samples * np.sin(wind_radians)))
    wind_y = float(np.mean(wind_speed_samples * np.cos(wind_radians)))
    tailwind = wind_x * track_x + wind_y * track_y
    crosswind = abs(-wind_y * track_x + wind_x * track_y)
    groundspeed = max(5.0, true_airspeed + tailwind)
    travel_min = distance_m / groundspeed / 60.0
    available_endurance = resource.spec.endurance_min * endurance_multiplier

    intersected: list[str] = []
    estimated_end = minute + travel_min
    for volume in airspace_volumes:
        if resource.resource_id in volume.allowed_resource_ids:
            continue
        time_overlap = minute < volume.end_minute and estimated_end >= volume.start_minute
        altitude_overlap = volume.lower_altitude_m_msl <= planned_altitude <= volume.upper_altitude_m_msl
        if (
            time_overlap
            and altitude_overlap
            and segment_intersects_polygon(
                start_xy,
                end_xy,
                volume.polygon_xy,
            )
        ):
            intersected.append(volume.volume_id)

    violations: list[str] = []
    if planned_altitude > resource.spec.maximum_operating_altitude_m_msl:
        violations.append("terrain_ceiling")
    if crosswind > resource.spec.maximum_crosswind_m_s:
        violations.append("crosswind")
    if load > maximum_payload + 1e-9:
        violations.append("density_altitude_payload")
    if altitude_domain is not None and not (altitude_domain[0] <= density_altitude <= altitude_domain[1]):
        violations.append("performance_surface_extrapolation")
    if intersected:
        violations.append("airspace_volume")
    return LegPerformance(
        distance_m=float(distance_m),
        track_heading_deg=heading,
        maximum_terrain_m_msl=maximum_terrain,
        planned_altitude_m_msl=float(planned_altitude),
        density_altitude_m=density_altitude,
        true_airspeed_m_s=float(true_airspeed),
        tailwind_m_s=float(tailwind),
        crosswind_m_s=float(crosswind),
        groundspeed_m_s=float(groundspeed),
        travel_min=float(travel_min),
        available_endurance_min=float(available_endurance),
        maximum_payload_fraction=float(maximum_payload),
        intersected_airspace_volumes=tuple(intersected),
        violations=tuple(violations),
    )


def evaluate_simulator_leg(
    resource: ResourceRuntime,
    end_xy: tuple[float, float],
    sim: AeolusSimulator,
    *,
    start_xy: tuple[float, float] | None = None,
    payload_fraction: float | None = None,
) -> LegPerformance:
    """Evaluate a leg from canonical simulator state."""

    weather = sim.current_weather()
    return evaluate_leg_performance(
        resource,
        start_xy=((float(resource.x), float(resource.y)) if start_xy is None else start_xy),
        end_xy=end_xy,
        cell_size_m=sim.config.cell_size_m,
        elevation_m=sim.state.truth.elevation_m,
        air_temperature_c=weather["air_temperature_c"],
        wind_speed_m_s=weather["wind_speed_m_s"],
        wind_from_direction_deg=weather["wind_direction_deg"],
        minute=float(sim.state.minute),
        airspace_volumes=sim.config.airspace_volumes,
        payload_fraction=payload_fraction,
    )
