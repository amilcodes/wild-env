"""Fast, inspectable surface spread and intervention operators.

The rate of spread follows the dimensional structure of Rothermel's surface
spread equation. This module is a training-kernel implementation rather than a
certified fire-behavior implementation; its purpose is to expose parameters and
support validation against independent tools.
"""

from __future__ import annotations

from math import ceil, cos, exp, pi, sin, sqrt

import numpy as np

from aeolus.config import FuelModel, ScenarioConfig
from aeolus.core.state import FirePhase, TruthState

_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rothermel_ros_m_min(fuel: FuelModel, wind_m_s: float, slope_tan: float) -> float:
    """Return an approximate no-direction surface ROS in metres/minute.

    The conversion uses the conventional Rothermel dimensional terms, then
    bounds output to avoid numerical pathologies outside the kernel's stated
    validity envelope. Fuel values are scenario parameters, not a LANDFIRE fuel
    model lookup.
    """

    w0 = fuel.fuel_load_kg_m2 * 0.204816  # lb / ft^2
    delta = fuel.fuel_depth_m * 3.28084  # ft
    sigma = fuel.surface_area_to_volume_m_inv / 3.28084  # ft^-1
    rho_p = fuel.fuel_particle_density_kg_m3 * 0.062428  # lb / ft^3
    heat = fuel.heat_content_kj_kg * 0.429923  # Btu / lb
    moisture = fuel.dead_moisture
    beta = _clip((w0 / max(delta, 1e-4)) / max(rho_p, 1e-4), 1e-5, 1.0)
    beta_op = 3.348 * sigma**-0.8189
    a_value = 1.0 / max(4.77 * sigma**0.1 - 7.27, 0.2)
    gamma_max = sigma**1.5 / (495.0 + 0.0594 * sigma**1.5)
    gamma = gamma_max * (beta / beta_op) ** a_value * exp(a_value * (1.0 - beta / beta_op))
    eta_m = max(
        0.0,
        1.0
        - 2.59 * moisture / fuel.moisture_of_extinction
        + 5.11 * (moisture / fuel.moisture_of_extinction) ** 2
        - 3.52 * (moisture / fuel.moisture_of_extinction) ** 3,
    )
    reaction_intensity = gamma * w0 * 0.95 * heat * eta_m * fuel.mineral_damping
    propagating_flux = exp((0.792 + 0.681 * sqrt(sigma)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    heat_sink = max((w0 / max(delta, 1e-4)) * exp(-138.0 / sigma) * (250.0 + 1116.0 * moisture), 1e-4)
    c_value = 7.47 * exp(-0.133 * sigma**0.55)
    b_value = 0.02526 * sigma**0.54
    e_value = 0.715 * exp(-0.000359 * sigma)
    # The equation's wind response is extremely sensitive outside its fuel-model
    # calibration range. The fast kernel uses a documented midflame-equivalent
    # cap and a scenario-level coarse-grid calibration factor; full validation
    # against a fire-behavior reference remains required.
    wind_ft_min = max(0.0, min(wind_m_s, 2.5) * 196.8504)
    phi_w = c_value * wind_ft_min**b_value * (beta / beta_op) ** (-e_value)
    phi_s = 5.275 * beta**-0.3 * slope_tan**2
    ros_ft_min = reaction_intensity * propagating_flux * (1.0 + phi_w + phi_s) / heat_sink
    return _clip(ros_ft_min * 0.3048 * 0.10, 0.005, 16.0)


def terrain_gradient(elevation: np.ndarray, x: int, y: int) -> tuple[float, float]:
    height, width = elevation.shape
    left = elevation[y, max(0, x - 1)]
    right = elevation[y, min(width - 1, x + 1)]
    down = elevation[max(0, y - 1), x]
    up = elevation[min(height - 1, y + 1), x]
    return float((right - left) * 0.5), float((up - down) * 0.5)


def _coverage_factor(truth: TruthState, x: int, y: int, intensity: float) -> float:
    retardant = float(truth.retardant[y, x])
    water = float(truth.water[y, x])
    ground = float(truth.ground_hold[y, x])
    # Water loses effectiveness as intensity rises; retardant/ground influence
    # ignition hazard and are not treated as deterministic barriers.
    water_effect = water * (0.74 / (1.0 + intensity / 1600.0))
    return _clip(1.0 - 0.82 * retardant - water_effect - 0.42 * ground, 0.05, 1.0)


def _ignite(truth: TruthState, x: int, y: int, intensity: float) -> None:
    if truth.barrier[y, x] or truth.phase[y, x] == FirePhase.BURNED:
        return
    truth.phase[y, x] = FirePhase.FLAMING
    truth.intensity_kw_m[y, x] = max(float(truth.intensity_kw_m[y, x]), intensity)


def step_fire(
    truth: TruthState,
    config: ScenarioConfig,
    rng: np.random.Generator,
    minute: int,
    *,
    wind_speed_m_s: float | None = None,
    wind_direction_deg: float | None = None,
) -> int:
    """Advance one minute and return the number of new flaming cells."""

    height, width = truth.phase.shape
    new_ignitions: list[tuple[int, int, float]] = []
    forced_speed = config.wind_speed_m_s if wind_speed_m_s is None else wind_speed_m_s
    forced_direction = (
        config.wind_direction_deg if wind_direction_deg is None else wind_direction_deg
    )
    # Synthetic scenarios retain a smooth sub-hourly perturbation. A supplied
    # weather forcing is already time varying and therefore passes zero
    # variability through the caller.
    variability = 0.0 if wind_speed_m_s is not None else config.wind_variability
    direction_variation = 0.0 if wind_direction_deg is not None else 7.0 * sin(minute / 17.0)
    wind_angle = np.deg2rad(forced_direction + direction_variation)
    wind_speed = max(0.2, forced_speed * (1.0 + variability * sin(minute / 13.0)))
    wx, wy = cos(wind_angle), -sin(wind_angle)
    flaming = np.argwhere(truth.phase == FirePhase.FLAMING)

    for y_raw, x_raw in flaming:
        y, x = int(y_raw), int(x_raw)
        intensity = float(truth.intensity_kw_m[y, x])
        dx_elev, dy_elev = terrain_gradient(truth.elevation_m, x, y)
        slope_tan = sqrt(dx_elev**2 + dy_elev**2) / config.cell_size_m
        base_ros = rothermel_ros_m_min(config.fuel, wind_speed, slope_tan)
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height or truth.barrier[ny, nx]:
                continue
            if truth.phase[ny, nx] != FirePhase.UNBURNED:
                continue
            distance_cells = sqrt(dx * dx + dy * dy)
            alignment = (dx / distance_cells) * wx + (dy / distance_cells) * wy
            directional_ros = base_ros * _clip(0.38 + 1.28 * max(alignment, -0.2), 0.12, 1.75)
            local_slope = (truth.elevation_m[ny, nx] - truth.elevation_m[y, x]) / config.cell_size_m
            directional_ros *= _clip(1.0 + 0.45 * local_slope, 0.3, 1.7)
            fuel_factor = _clip(
                float(truth.fuel_load[ny, nx]) / max(config.fuel.fuel_load_kg_m2, 1e-6), 0.05, 1.8
            )
            residual = float(truth.residual_field[ny, nx])
            treatment = _coverage_factor(truth, nx, ny, intensity)
            hazard = (
                directional_ros * fuel_factor * residual * treatment * distance_cells / config.cell_size_m
            )
            probability = 1.0 - exp(-max(0.0, hazard))
            if rng.random() < probability:
                new_ignitions.append((nx, ny, max(60.0, intensity * 0.72)))

        # Short-range spotting is explicitly stochastic and independently logged.
        spot_probability = config.spotting_rate * _clip(intensity / 1200.0, 0.0, 2.0) * wind_speed / 8.0
        if rng.random() < spot_probability:
            range_cells = int(rng.integers(3, 10))
            sx = int(round(x + wx * range_cells + rng.normal(0.0, 1.2)))
            sy = int(round(y + wy * range_cells + rng.normal(0.0, 1.2)))
            if 0 <= sx < width and 0 <= sy < height and not truth.barrier[sy, sx]:
                new_ignitions.append((sx, sy, max(45.0, intensity * 0.42)))

        local_water = float(truth.water[y, x])
        local_retardant = float(truth.retardant[y, x])
        truth.intensity_kw_m[y, x] *= _clip(0.88 - 0.28 * local_water - 0.08 * local_retardant, 0.18, 0.94)
        truth.fuel_remaining[y, x] = max(0.0, truth.fuel_remaining[y, x] - 0.045 * (1.0 + intensity / 2400.0))
        if truth.fuel_remaining[y, x] <= 0.04 or truth.intensity_kw_m[y, x] < 20.0:
            truth.phase[y, x] = FirePhase.BURNED
            truth.intensity_kw_m[y, x] = 0.0
            truth.observed_burned[y, x] = 1.0

    for x, y, intensity in new_ignitions:
        _ignite(truth, x, y, intensity)
    truth.water *= 0.74
    truth.retardant *= 0.996
    truth.ground_hold *= 0.999
    return len(new_ignitions)


def apply_water(truth: TruthState, x: int, y: int, radius_cells: float) -> None:
    height, width = truth.phase.shape
    for ny in range(max(0, int(y - ceil(radius_cells))), min(height, int(y + ceil(radius_cells) + 1))):
        for nx in range(max(0, int(x - ceil(radius_cells))), min(width, int(x + ceil(radius_cells) + 1))):
            distance = sqrt((nx - x) ** 2 + (ny - y) ** 2)
            if distance > radius_cells:
                continue
            coverage = _clip(1.0 - distance / (radius_cells + 0.5), 0.0, 1.0)
            truth.water[ny, nx] = max(float(truth.water[ny, nx]), coverage)
            if truth.phase[ny, nx] == FirePhase.FLAMING:
                truth.intensity_kw_m[ny, nx] *= 1.0 - 0.48 * coverage


def apply_retardant(
    truth: TruthState,
    x: int,
    y: int,
    length_cells: float,
    width_cells: float,
    wind_direction_deg: float,
    ground_engaged: bool,
) -> None:
    """Rasterize an oriented coverage footprint perpendicular to wind."""

    height, width = truth.phase.shape
    radians = wind_direction_deg * pi / 180.0
    along_x, along_y = -sin(radians), -cos(radians)
    cross_x, cross_y = cos(radians), -sin(radians)
    for along in np.arange(-length_cells / 2.0, length_cells / 2.0 + 0.5, 0.5):
        for cross in np.arange(-width_cells / 2.0, width_cells / 2.0 + 0.5, 0.5):
            nx = int(round(x + along_x * along + cross_x * cross))
            ny = int(round(y + along_y * along + cross_y * cross))
            if nx < 0 or ny < 0 or nx >= width or ny >= height or truth.barrier[ny, nx]:
                continue
            lateral = abs(cross) / max(width_cells / 2.0, 0.1)
            coverage = _clip(0.96 - 0.44 * lateral, 0.25, 0.96)
            truth.retardant[ny, nx] = max(float(truth.retardant[ny, nx]), coverage)
            if ground_engaged:
                truth.ground_hold[ny, nx] = max(float(truth.ground_hold[ny, nx]), 0.72 * coverage)
