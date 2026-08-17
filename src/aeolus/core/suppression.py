"""Volume-conserving drops, constructed line, and engagement outcomes."""

from __future__ import annotations

from math import cos, sin

import numpy as np

from aeolus.config import ScenarioConfig
from aeolus.core.state import FirePhase, TruthState


def required_coverage_level_gpc(
    fuel_model_number: np.ndarray,
    intensity_kw_m: np.ndarray,
) -> np.ndarray:
    """Operational coverage-level guide with an intensity adjustment.

    The base classes follow the USFS coverage-level guide.  The intensity term
    is explicit because the guide requires adjustment for observed behavior.
    """

    fuel = np.asarray(fuel_model_number)
    base = np.full(fuel.shape, 2.0, dtype=np.float32)
    base[np.isin(fuel, (101, 102, 103, 104, 105, 106, 107, 121, 122))] = 1.0
    base[np.isin(fuel, (123, 124, 141, 142, 143, 144, 145, 146, 147))] = 3.0
    base[np.isin(fuel, (161, 162, 163, 164, 165, 181, 182, 183, 184, 185))] = 4.0
    base[np.isin(fuel, (186, 187, 188, 189, 201, 202, 203, 204))] = 6.0
    return np.clip(
        base + np.maximum(np.asarray(intensity_kw_m) - 1000.0, 0.0) / 1800.0,
        1.0,
        8.0,
    ).astype(np.float32)


def _oriented_gaussian_pattern(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    length_cells: float,
    width_cells: float,
    heading_deg: float,
) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    theta = np.deg2rad(heading_deg)
    dx = x - center_x
    dy = y - center_y
    along = dx * cos(theta) + dy * sin(theta)
    across = -dx * sin(theta) + dy * cos(theta)
    sigma_along = max(length_cells / 4.0, 0.6)
    sigma_across = max(width_cells / 4.0, 0.45)
    pattern = np.exp(-0.5 * ((along / sigma_along) ** 2 + (across / sigma_across) ** 2))
    pattern[
        (np.abs(along) > max(length_cells / 2.0, 1.0)) | (np.abs(across) > max(width_cells / 2.0, 0.75))
    ] = 0.0
    total = float(pattern.sum())
    if total <= 0.0:
        iy = int(np.clip(round(center_y), 0, shape[0] - 1))
        ix = int(np.clip(round(center_x), 0, shape[1] - 1))
        pattern[iy, ix] = 1.0
        total = 1.0
    return (pattern / total).astype(np.float32)


def apply_aerial_drop(
    truth: TruthState,
    config: ScenarioConfig,
    *,
    kind: str,
    payload_l: float,
    x: float,
    y: float,
    length_m: float,
    width_m: float,
    heading_deg: float,
    wind_speed_m_s: float,
    wind_from_direction_deg: float,
    local_effective_coverage_gpc: float | None = None,
) -> dict[str, float]:
    """Apply a wind-displaced, volume-conserving ground pattern.

    ``local_effective_coverage_gpc`` is an optional subcell closure for a
    measured line narrower than the fire grid. Physical coverage remains the
    cell-area average in ``retardant_coverage_gpc``; the separate effective
    field preserves the delivery table's target coverage for spread response.
    """

    if kind not in {"water", "retardant"}:
        raise ValueError("aerial drop kind must be water or retardant")
    if payload_l <= 0.0:
        return {"applied_l": 0.0, "peak_coverage_gpc": 0.0, "mean_coverage_gpc": 0.0}
    settings = config.suppression
    wind_to_rad = np.deg2rad((wind_from_direction_deg + 180.0) % 360.0)
    drift_m = settings.drop_drift_m_per_m_s * max(wind_speed_m_s, 0.0)
    center_x = x + drift_m * cos(wind_to_rad) / config.cell_size_m
    center_y = y + drift_m * sin(wind_to_rad) / config.cell_size_m
    dispersion = 1.0 + settings.drop_dispersion_growth_per_m_s * max(wind_speed_m_s, 0.0)
    pattern = _oriented_gaussian_pattern(
        truth.phase.shape,
        center_x,
        center_y,
        max(length_m / config.cell_size_m, 1.0) * dispersion,
        max(width_m / config.cell_size_m, 1.0) * dispersion,
        heading_deg,
    )
    liters_per_m2 = payload_l * pattern / (config.cell_size_m**2)
    added_gpc = liters_per_m2 / settings.gpc_l_m2
    required = required_coverage_level_gpc(
        truth.fuel_model_number,
        truth.intensity_kw_m,
    )
    support = added_gpc > 1.0e-6
    if kind == "water":
        truth.water_coverage_gpc += added_gpc
        effectiveness = 1.0 - np.exp(-truth.water_coverage_gpc / np.maximum(required, 1.0))
        truth.water[:] = np.clip(effectiveness, 0.0, 1.0)
        truth.intensity_kw_m *= 1.0 - settings.water_intensity_reduction * np.clip(
            added_gpc / required, 0.0, 1.0
        )
    else:
        truth.retardant_coverage_gpc += added_gpc
        if local_effective_coverage_gpc is None:
            truth.retardant_effective_coverage_gpc += added_gpc
        else:
            truth.retardant_effective_coverage_gpc += (
                support.astype(np.float32) * local_effective_coverage_gpc
            )
        effectiveness = 1.0 - np.exp(-truth.retardant_effective_coverage_gpc / np.maximum(required, 1.0))
        truth.retardant[:] = np.clip(effectiveness, 0.0, 1.0)
    return {
        "applied_l": float((added_gpc * settings.gpc_l_m2 * config.cell_size_m**2).sum()),
        "peak_coverage_gpc": float(added_gpc.max()),
        "mean_coverage_gpc": float(added_gpc[support].mean()) if support.any() else 0.0,
        "peak_effective_coverage_gpc": float(
            truth.retardant_effective_coverage_gpc.max()
            if kind == "retardant"
            else truth.water_coverage_gpc.max()
        ),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "drop_length_m": float(length_m),
        "drop_width_m": float(width_m),
    }


def construct_line_segment(
    truth: TruthState,
    config: ScenarioConfig,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width_m: float,
) -> int:
    """Rasterize newly completed fireline while retaining its physical width."""

    x0, y0 = start_xy
    x1, y1 = end_xy
    segment_length_cells = float(np.hypot(x1 - x0, y1 - y0))
    samples = max(2, int(np.ceil(segment_length_cells * 3.0)) + 1)
    xs = np.linspace(x0, x1, samples)
    ys = np.linspace(y0, y1, samples)
    radius_cells = max(0.5, 0.5 * width_m / config.cell_size_m)
    grid_y, grid_x = np.mgrid[: truth.phase.shape[0], : truth.phase.shape[1]]
    new_line = np.zeros(truth.phase.shape, dtype=np.bool_)
    for x, y in zip(xs, ys, strict=True):
        new_line |= (grid_x - x) ** 2 + (grid_y - y) ** 2 <= radius_cells**2
    new_line &= ~truth.barrier
    truth.constructed_line[new_line] = 1.0
    truth.line_strength[new_line] = np.maximum(
        truth.line_strength[new_line],
        max(width_m, 0.5),
    )
    truth.ground_hold[new_line] = 1.0
    truth.line_status[new_line & (truth.line_status == 0)] = 1
    return int(new_line.sum())


def update_suppression_state(
    truth: TruthState,
    config: ScenarioConfig,
    rng: np.random.Generator,
    *,
    precipitation_rate_mm_h: float | np.ndarray,
    dt_min: float = 1.0,
) -> dict[str, int]:
    """Age treatments and resolve held, overrun, and unengaged line."""

    settings = config.suppression
    water_decay = 0.5 ** (dt_min / settings.water_half_life_min)
    retardant_decay = 0.5 ** (dt_min / settings.retardant_half_life_min)
    rain_mm = np.maximum(np.asarray(precipitation_rate_mm_h), 0.0) * dt_min / 60.0
    rain_factor = np.exp(-settings.retardant_rain_wash_fraction_per_mm * rain_mm)
    truth.water_coverage_gpc *= water_decay
    truth.retardant_coverage_gpc *= retardant_decay * rain_factor
    truth.retardant_effective_coverage_gpc *= retardant_decay * rain_factor
    required = required_coverage_level_gpc(
        truth.fuel_model_number,
        truth.intensity_kw_m,
    )
    truth.water[:] = np.clip(
        1.0 - np.exp(-truth.water_coverage_gpc / np.maximum(required, 1.0)),
        0.0,
        1.0,
    )
    truth.retardant[:] = np.clip(
        1.0 - np.exp(-truth.retardant_effective_coverage_gpc / np.maximum(required, 1.0)),
        0.0,
        1.0,
    )

    line = truth.constructed_line > 0.0
    adjacent_fire = np.zeros_like(line)
    flaming = truth.phase == FirePhase.FLAMING
    padded = np.pad(flaming, 1)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        adjacent_fire |= padded[1 + dy : 1 + dy + line.shape[0], 1 + dx : 1 + dx + line.shape[1]]
    engaged = line & adjacent_fire & (truth.line_status != 3)
    capacity = (
        settings.line_capacity_base_kw_m
        + settings.line_capacity_per_m_width_kw_m * truth.line_strength
        + settings.line_capacity_retardant_kw_m * truth.retardant
    )
    demand = truth.intensity_kw_m.copy()
    neighbor_max = demand.copy()
    padded_intensity = np.pad(demand, 1)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_max = np.maximum(
            neighbor_max,
            padded_intensity[
                1 + dy : 1 + dy + line.shape[0],
                1 + dx : 1 + dx + line.shape[1],
            ],
        )
    breach_probability = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (capacity - neighbor_max) / settings.line_breach_logistic_scale_kw_m,
                -60.0,
                60.0,
            )
        )
    )
    breached = engaged & (rng.random(line.shape) < breach_probability)
    held = engaged & ~breached
    truth.line_status[held] = 2
    truth.line_status[breached] = 3
    truth.ground_hold[line] = 1.0
    truth.ground_hold[breached] = 0.0
    return {
        "engaged_cells": int(engaged.sum()),
        "held_cells": int(held.sum()),
        "breached_cells": int(breached.sum()),
    }
