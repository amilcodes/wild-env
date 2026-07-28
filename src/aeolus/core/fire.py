"""Vectorized operational-equation fire spread and intervention operators."""

from __future__ import annotations

from math import ceil, cos, exp, pi, sin, sqrt

import numpy as np

from aeolus.config import FuelModel, ScenarioConfig
from aeolus.core.fire_behavior import fire_behavior_lookup
from aeolus.core.state import FirePhase, FireType, TruthState

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


def _shift_to_target(source: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
    """Shift source-cell values to neighboring targets without wraparound."""

    output = np.full_like(source, fill)
    height, width = source.shape[-2:]
    source_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    output[..., target_y, target_x] = source[..., source_y, source_x]
    return output


def _neighbor_any(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.bool_)
    for dx, dy in _NEIGHBORS:
        output |= _shift_to_target(mask, dx, dy, fill=False)
    return output


def _fuel_moisture_equilibrium(temperature_c: float, relative_humidity_pct: float) -> float:
    """Simard equilibrium moisture content as kg/kg dry fuel."""

    rh = float(np.clip(relative_humidity_pct, 0.0, 100.0))
    temperature_f = temperature_c * 9.0 / 5.0 + 32.0
    if rh < 10.0:
        percent = 0.03229 + 0.281073 * rh - 0.000578 * rh * temperature_f
    elif rh <= 50.0:
        percent = 2.22749 + 0.160107 * rh - 0.01478 * temperature_f
    else:
        percent = (
            21.0606
            + 0.005565 * rh * rh
            - 0.00035 * rh * temperature_f
            - 0.483199 * rh
        )
    return float(np.clip(percent / 100.0, 0.01, 0.60))


def update_fuel_moisture(
    truth: TruthState,
    config: ScenarioConfig,
    *,
    air_temperature_c: float,
    relative_humidity_pct: float,
    precipitation_rate_mm_h: float = 0.0,
    dt_min: float = 1.0,
) -> None:
    if config.fire.moisture_model == "fixed":
        return
    equilibrium = _fuel_moisture_equilibrium(
        air_temperature_c, relative_humidity_pct
    )
    for field, lag_min in (
        (truth.moisture_dead_1h, 60.0),
        (truth.moisture_dead_10h, 600.0),
        (truth.moisture_dead_100h, 6000.0),
    ):
        field += (equilibrium - field) * (1.0 - exp(-dt_min / lag_min))
        if precipitation_rate_mm_h > 0.0:
            wetting_rate = (
                precipitation_rate_mm_h
                * dt_min
                / 60.0
                / max(lag_min / 60.0, 1.0)
            )
            field += (0.60 - field) * (1.0 - exp(-wetting_rate))
        np.clip(field, 0.01, 0.60, out=field)


def rothermel_ros_m_min(fuel: FuelModel, wind_m_s: float, slope_tan: float) -> float:
    """Compatibility point query backed by the packaged reference table."""

    shape = (1, 1)
    behavior = fire_behavior_lookup().resolve_numpy(
        fuel_model_number=np.full(shape, fuel.standard_number, dtype=np.int16),
        moisture_dead_1h=np.full(shape, fuel.dead_moisture, dtype=np.float32),
        wind_speed_10m_m_s=wind_m_s,
        wind_from_direction_deg=270.0,
        terrain_slope_x=np.full(shape, slope_tan, dtype=np.float32),
        terrain_slope_y=np.zeros(shape, dtype=np.float32),
        canopy_cover=np.zeros(shape, dtype=np.float32),
        canopy_height_m=np.zeros(shape, dtype=np.float32),
        canopy_base_height_m=np.zeros(shape, dtype=np.float32),
        canopy_bulk_density_kg_m3=np.zeros(shape, dtype=np.float32),
        foliar_moisture=np.ones(shape, dtype=np.float32),
        config=ScenarioConfig(fuel=fuel).fire,
    )
    return float(behavior.spread_rate_m_min[0, 0])


def _terrain_slopes(elevation_m: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    slope_y, slope_x = np.gradient(elevation_m, cell_size_m)
    return slope_x.astype(np.float32), slope_y.astype(np.float32)


def _coverage_factor(truth: TruthState) -> np.ndarray:
    # Water changes both immediate intensity and dead-fuel moisture.  Retardant
    # and constructed line reduce the local spread multiplier but do not become
    # unconditional barriers.
    intensity = truth.intensity_kw_m
    water_effect = truth.water * (0.76 / (1.0 + intensity / 1600.0))
    return np.clip(
        1.0 - 0.84 * truth.retardant - water_effect - 0.48 * truth.ground_hold,
        0.025,
        1.0,
    ).astype(np.float32)


def _resolve_behavior(
    truth: TruthState,
    config: ScenarioConfig,
    wind_speed_m_s: float,
    wind_direction_deg: float,
):
    slope_x, slope_y = _terrain_slopes(truth.elevation_m, config.cell_size_m)
    nominal_load = max(config.fuel.fuel_load_kg_m2, 1e-5)
    fuel_adjustment = np.clip(truth.fuel_load / nominal_load, 0.05, 2.5)
    spread_adjustment = (
        truth.residual_field * fuel_adjustment * _coverage_factor(truth)
    )
    return fire_behavior_lookup().resolve_numpy(
        fuel_model_number=truth.fuel_model_number,
        moisture_dead_1h=truth.moisture_dead_1h,
        wind_speed_10m_m_s=wind_speed_m_s,
        wind_from_direction_deg=wind_direction_deg,
        terrain_slope_x=slope_x,
        terrain_slope_y=slope_y,
        canopy_cover=truth.canopy_cover,
        canopy_height_m=truth.canopy_height_m,
        canopy_base_height_m=truth.canopy_base_height_m,
        canopy_bulk_density_kg_m3=truth.canopy_bulk_density_kg_m3,
        foliar_moisture=truth.foliar_moisture,
        spread_adjustment=spread_adjustment,
        config=config.fire,
    )


def _propagate_substep(
    truth: TruthState,
    config: ScenarioConfig,
    behavior,
    dt_min: float,
    current_time_min: float,
) -> np.ndarray:
    flaming = truth.phase == FirePhase.FLAMING
    burnable = (~truth.barrier) & (truth.fuel_load > 0.0)
    unburned = (truth.phase == FirePhase.UNBURNED) & burnable
    incoming = np.zeros_like(truth.ignition_progress)
    for dx, dy in _NEIGHBORS:
        distance = sqrt(dx * dx + dy * dy) * config.cell_size_m
        direction_x = dx / sqrt(dx * dx + dy * dy)
        direction_y = dy / sqrt(dx * dx + dy * dy)
        cosine = behavior.head_x * direction_x + behavior.head_y * direction_y
        directional_factor = (1.0 - behavior.eccentricity) / np.maximum(
            1.0 - behavior.eccentricity * cosine, 1e-4
        )
        travel_fraction = (
            behavior.spread_rate_m_min
            * directional_factor
            * dt_min
            / max(distance, 1e-5)
        )
        candidate = _shift_to_target(
            np.where(flaming, travel_fraction, 0.0), dx, dy
        )
        incoming = np.maximum(incoming, candidate)
    truth.ignition_progress[unburned] += incoming[unburned]
    ignited = unburned & (truth.ignition_progress >= 1.0)
    if np.any(ignited):
        truth.phase[ignited] = FirePhase.FLAMING
        truth.arrival_time_min[ignited] = current_time_min
        truth.burn_age_min[ignited] = 0.0
        truth.ignition_progress[ignited] = 0.0
        truth.intensity_kw_m[ignited] = behavior.fireline_intensity_kw_m[ignited]
        truth.fire_type[ignited] = behavior.fire_type[ignited]
        truth.spread_rate_m_min[ignited] = behavior.spread_rate_m_min[ignited]
        truth.flame_length_m[ignited] = behavior.flame_length_m[ignited]
    return ignited


def _spot_fires(
    truth: TruthState,
    config: ScenarioConfig,
    rng: np.random.Generator,
    behavior,
    wind_speed_m_s: float,
    wind_from_direction_deg: float,
    minute: int,
) -> int:
    settings = config.fire
    if (
        not settings.enable_spotting
        or config.spotting_rate <= 0.0
        or wind_speed_m_s < 0.5
    ):
        return 0
    source_mask = (
        (truth.phase == FirePhase.FLAMING)
        & (behavior.fireline_intensity_kw_m >= 350.0)
    )
    sources = np.argwhere(source_mask)
    if not len(sources):
        return 0
    expected = (
        settings.spotting_embers_per_source_min
        * config.spotting_rate
        / 0.01
        * len(sources)
    )
    count = min(
        int(rng.poisson(expected)),
        settings.max_spot_embers_per_minute,
    )
    if count <= 0:
        return 0
    selected = sources[rng.integers(0, len(sources), size=count)]
    intensity = behavior.fireline_intensity_kw_m[selected[:, 0], selected[:, 1]]
    median = settings.spotting_median_distance_m * np.power(
        np.maximum(wind_speed_m_s / 6.0, 0.1), settings.spotting_wind_exponent
    ) * np.power(
        np.maximum(intensity / 2000.0, 0.1), settings.spotting_intensity_exponent
    )
    distance = np.minimum(
        rng.lognormal(np.log(np.maximum(median, 1.0)), settings.spotting_log_sigma),
        settings.spotting_max_distance_m,
    )
    cross = rng.normal(
        0.0, settings.spotting_crosswind_fraction * np.maximum(distance, 1.0)
    )
    radians = np.deg2rad(wind_from_direction_deg)
    down_x, down_y = -np.sin(radians), np.cos(radians)
    cross_x, cross_y = -down_y, down_x
    target_x = np.rint(
        selected[:, 1]
        + (distance * down_x + cross * cross_x) / config.cell_size_m
    ).astype(np.int64)
    target_y = np.rint(
        selected[:, 0]
        + (distance * down_y + cross * cross_y) / config.cell_size_m
    ).astype(np.int64)
    height, width = truth.phase.shape
    valid = (
        (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )
    if not np.any(valid):
        return 0
    target_x, target_y, distance = target_x[valid], target_y[valid], distance[valid]
    survival = np.exp(-distance / settings.spotting_survival_distance_m)
    moisture_factor = np.clip(
        1.0 - truth.moisture_dead_1h[target_y, target_x] / 0.35, 0.0, 1.0
    )
    treatment = _coverage_factor(truth)[target_y, target_x]
    probability = (
        settings.spotting_ignition_probability
        * survival
        * moisture_factor
        * treatment
    )
    accepted = rng.random(len(probability)) < probability
    ignitions = 0
    for x, y in zip(target_x[accepted], target_y[accepted], strict=True):
        if (
            truth.phase[y, x] == FirePhase.UNBURNED
            and not truth.barrier[y, x]
            and truth.fuel_load[y, x] > 0.0
        ):
            truth.phase[y, x] = FirePhase.FLAMING
            truth.arrival_time_min[y, x] = float(minute)
            truth.burn_age_min[y, x] = 0.0
            truth.fire_type[y, x] = FireType.SURFACE
            truth.intensity_kw_m[y, x] = max(
                60.0, float(behavior.fireline_intensity_kw_m[y, x])
            )
            ignitions += 1
    return ignitions


def step_fire(
    truth: TruthState,
    config: ScenarioConfig,
    rng: np.random.Generator,
    minute: int,
    *,
    wind_speed_m_s: float | None = None,
    wind_direction_deg: float | None = None,
    air_temperature_c: float | None = None,
    relative_humidity_pct: float | None = None,
    precipitation_rate_mm_h: float | None = None,
) -> int:
    """Advance one minute using adaptive Huygens-style raster front tracking."""

    forced_speed = config.wind_speed_m_s if wind_speed_m_s is None else wind_speed_m_s
    forced_direction = (
        config.wind_direction_deg if wind_direction_deg is None else wind_direction_deg
    )
    variability = 0.0 if wind_speed_m_s is not None else config.wind_variability
    direction_variation = (
        0.0 if wind_direction_deg is not None else 7.0 * sin(minute / 17.0)
    )
    wind_speed = max(
        0.0, forced_speed * (1.0 + variability * sin(minute / 13.0))
    )
    wind_direction = forced_direction + direction_variation
    temperature = (
        config.air_temperature_c
        if air_temperature_c is None or np.isnan(air_temperature_c)
        else air_temperature_c
    )
    humidity = (
        config.relative_humidity_pct
        if relative_humidity_pct is None or np.isnan(relative_humidity_pct)
        else relative_humidity_pct
    )
    precipitation = (
        config.precipitation_rate_mm_h
        if precipitation_rate_mm_h is None or np.isnan(precipitation_rate_mm_h)
        else precipitation_rate_mm_h
    )
    update_fuel_moisture(
        truth,
        config,
        air_temperature_c=temperature,
        relative_humidity_pct=humidity,
        precipitation_rate_mm_h=precipitation,
    )
    # Wetting from drops participates in subsequent behavior, not just the
    # momentary intensity reduction.
    truth.moisture_dead_1h[:] = np.maximum(
        truth.moisture_dead_1h, config.fuel.dead_moisture + 0.24 * truth.water
    )
    behavior = _resolve_behavior(
        truth, config, float(wind_speed), float(wind_direction)
    )
    max_fraction = float(
        np.max(behavior.spread_rate_m_min) / max(config.cell_size_m, 1e-5)
    )
    substeps = int(
        np.clip(
            ceil(max_fraction / config.fire.propagation_cfl),
            1,
            config.fire.max_substeps,
        )
    )
    new_ignitions = 0
    for substep in range(substeps):
        ignited = _propagate_substep(
            truth,
            config,
            behavior,
            1.0 / substeps,
            minute - 1.0 + (substep + 1.0) / substeps,
        )
        new_ignitions += int(ignited.sum())

    flaming = truth.phase == FirePhase.FLAMING
    truth.burn_age_min[flaming] += 1.0
    truth.spread_rate_m_min[flaming] = behavior.spread_rate_m_min[flaming]
    truth.flame_length_m[flaming] = behavior.flame_length_m[flaming]
    truth.fire_type[flaming] = behavior.fire_type[flaming]
    truth.intensity_kw_m[flaming] = behavior.fireline_intensity_kw_m[flaming]
    truth.intensity_kw_m[flaming] *= np.clip(
        1.0 - 0.68 * truth.water[flaming] - 0.12 * truth.retardant[flaming],
        0.05,
        1.0,
    )
    # Exponential consumption preserves a spreading front at coarse resolution;
    # cells become burned after the front has passed, rather than on a fixed
    # residence clock that can halt slow fires between cell centers.
    truth.fuel_remaining[flaming] *= np.exp(
        -1.0 / np.clip(18.0 + 0.18 * config.cell_size_m, 18.0, 90.0)
    )
    adjacent_unburned = _neighbor_any(
        (truth.phase == FirePhase.UNBURNED)
        & (~truth.barrier)
        & (truth.fuel_load > 0.0)
    )
    passed_front = flaming & (~adjacent_unburned)
    expired_front = flaming & (
        truth.burn_age_min >= config.fire.max_front_residence_min
    )
    consumed = flaming & (
        (truth.fuel_remaining <= 0.05)
        & (truth.burn_age_min >= config.fire.min_front_residence_min)
        & (truth.intensity_kw_m < 20.0)
    )
    suppression_hold = flaming & (
        (truth.burn_age_min >= config.fire.min_front_residence_min)
        & (truth.intensity_kw_m < 20.0)
        & (_coverage_factor(truth) < 0.35)
    )
    burned = passed_front | expired_front | consumed | suppression_hold
    truth.phase[burned] = FirePhase.BURNED
    truth.fire_type[burned] = FireType.UNBURNED
    truth.intensity_kw_m[burned] = 0.0
    truth.spread_rate_m_min[burned] = 0.0
    truth.flame_length_m[burned] = 0.0
    truth.observed_burned[burned] = 1.0
    new_ignitions += _spot_fires(
        truth,
        config,
        rng,
        behavior,
        float(wind_speed),
        float(wind_direction),
        minute,
    )
    truth.water *= 0.74
    truth.retardant *= 0.996
    truth.ground_hold *= 0.999
    return new_ignitions


def apply_water(truth: TruthState, x: int, y: int, radius_cells: float) -> None:
    height, width = truth.phase.shape
    for ny in range(
        max(0, int(y - ceil(radius_cells))),
        min(height, int(y + ceil(radius_cells) + 1)),
    ):
        for nx in range(
            max(0, int(x - ceil(radius_cells))),
            min(width, int(x + ceil(radius_cells) + 1)),
        ):
            distance = sqrt((nx - x) ** 2 + (ny - y) ** 2)
            if distance > radius_cells:
                continue
            coverage = _clip(
                1.0 - distance / (radius_cells + 0.5), 0.0, 1.0
            )
            truth.water[ny, nx] = max(float(truth.water[ny, nx]), coverage)
            truth.moisture_dead_1h[ny, nx] = max(
                float(truth.moisture_dead_1h[ny, nx]), 0.28 * coverage
            )
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
    for along in np.arange(
        -length_cells / 2.0, length_cells / 2.0 + 0.5, 0.5
    ):
        for cross in np.arange(
            -width_cells / 2.0, width_cells / 2.0 + 0.5, 0.5
        ):
            nx = int(round(x + along_x * along + cross_x * cross))
            ny = int(round(y + along_y * along + cross_y * cross))
            if (
                nx < 0
                or ny < 0
                or nx >= width
                or ny >= height
                or truth.barrier[ny, nx]
            ):
                continue
            lateral = abs(cross) / max(width_cells / 2.0, 0.1)
            coverage = _clip(0.96 - 0.44 * lateral, 0.25, 0.96)
            truth.retardant[ny, nx] = max(
                float(truth.retardant[ny, nx]), coverage
            )
            if ground_engaged:
                truth.ground_hold[ny, nx] = max(
                    float(truth.ground_hold[ny, nx]), 0.72 * coverage
                )
