"""Coupled fire-state initialization from two observed perimeters.

The reconstructed arrival-time field is a causal state variable, not merely a
burned-area mask.  It supplies burn age, residual heat release, and a
front-localized estimate of recent spread velocity for the free forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrivalHistory:
    arrival_time_min: np.ndarray
    burn_age_min: np.ndarray
    speed_m_min: np.ndarray
    head_x: np.ndarray
    head_y: np.ndarray
    confidence: np.ndarray
    heat_flux_fraction: np.ndarray
    diagnostics: dict[str, float | int]


def _boundary(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for arrival-history reconstruction") from exc
    return mask & ~binary_erosion(mask)


def reconstruct_arrival_history(
    earlier_mask: np.ndarray,
    later_mask: np.ndarray,
    elapsed_min: float,
    cell_size_m: float,
    *,
    max_iterations: int = 600,
    tolerance_min: float = 1.0e-3,
    localization_distance_cells: float = 8.0,
) -> ArrivalHistory:
    """Reconstruct a smooth, causal arrival history between two perimeters.

    Dirichlet values are fixed at the earlier and later fronts and a harmonic
    field is solved through the intervening growth band.  Non-nested pixels in
    the earlier observation are excluded and reported rather than allowed to
    create a negative-time reversal.  Local velocity follows the arrival-time
    gradient and is extended a finite distance ahead of the later front.
    """

    try:
        from scipy.ndimage import (
            binary_dilation,
            distance_transform_edt,
            gaussian_filter,
            label,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for arrival-history reconstruction") from exc

    if earlier_mask.shape != later_mask.shape or earlier_mask.ndim != 2:
        raise ValueError("perimeter masks must be two-dimensional on one common grid")
    if elapsed_min <= 0.0 or cell_size_m <= 0.0:
        raise ValueError("elapsed time and cell size must be positive")
    if max_iterations < 1 or tolerance_min <= 0.0:
        raise ValueError("arrival solver controls must be positive")

    earlier_raw = earlier_mask.astype(np.bool_)
    later = later_mask.astype(np.bool_)
    earlier = earlier_raw & later
    if not earlier.any() or not later.any():
        raise ValueError("both perimeter observations must contain burned cells")
    growth = later & ~earlier
    removed_cells = int((earlier_raw & ~later).sum())

    earlier_front = _boundary(earlier)
    later_front = _boundary(later)
    distance_from_earlier = distance_transform_edt(~earlier) * cell_size_m
    distance_to_later_front = distance_transform_edt(later) * cell_size_m
    denominator = np.maximum(
        distance_from_earlier + distance_to_later_front,
        0.25 * cell_size_m,
    )
    progress = np.clip(distance_from_earlier / denominator, 0.0, 1.0)

    inner_condition = growth & binary_dilation(earlier)
    outer_condition = growth & later_front
    fixed = inner_condition | outer_condition
    solve = growth & ~fixed
    normalized_time = progress.astype(np.float64)
    normalized_time[inner_condition] = 0.0
    normalized_time[outer_condition] = 1.0

    residual_min = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        padded = np.pad(normalized_time, 1, mode="edge")
        average = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        if solve.any():
            residual_min = float(np.max(np.abs(average[solve] - normalized_time[solve])) * elapsed_min)
            normalized_time[solve] = average[solve]
        else:
            residual_min = 0.0
        if residual_min <= tolerance_min:
            break

    arrival = np.full(later.shape, np.inf, dtype=np.float64)
    arrival[growth] = -elapsed_min + elapsed_min * normalized_time[growth]
    arrival[later_front] = 0.0

    # The earlier interior must carry age older than the first observation.
    # A robust speed inferred from the annular width prevents a flat, freshly
    # ignited interior while avoiding unbounded ages for long incidents.
    growth_width = distance_from_earlier[growth]
    robust_width_m = float(np.quantile(growth_width, 0.65)) if growth_width.size else cell_size_m
    reference_speed = max(0.05, robust_width_m / elapsed_min)
    depth_inside_earlier = np.maximum(
        distance_transform_edt(earlier) * cell_size_m - cell_size_m,
        0.0,
    )
    arrival[earlier] = -elapsed_min - depth_inside_earlier[earlier] / reference_speed
    arrival[earlier_front] = -elapsed_min

    # Extend the finite arrival field through the exterior solely for a stable
    # derivative at the observed front.  Forecast arrival times remain infinite.
    _, nearest_later = distance_transform_edt(~later, return_indices=True)
    derivative_field = arrival.copy()
    derivative_field[~later] = arrival[nearest_later[0, ~later], nearest_later[1, ~later]]
    finite_fill = -elapsed_min
    derivative_field[~np.isfinite(derivative_field)] = finite_fill
    derivative_field = gaussian_filter(derivative_field, sigma=0.8, mode="nearest")
    gradient_y, gradient_x = np.gradient(
        derivative_field,
        cell_size_m,
        cell_size_m,
    )
    gradient_norm = np.hypot(gradient_x, gradient_y)
    local_speed = np.divide(
        1.0,
        gradient_norm,
        out=np.full_like(gradient_norm, reference_speed),
        where=gradient_norm > 1.0e-7,
    )
    local_speed = np.clip(
        gaussian_filter(local_speed, sigma=0.7, mode="nearest"),
        0.05,
        300.0,
    )
    head_x = np.divide(
        gradient_x,
        gradient_norm,
        out=np.zeros_like(gradient_x),
        where=gradient_norm > 1.0e-7,
    )
    head_y = np.divide(
        gradient_y,
        gradient_norm,
        out=np.zeros_like(gradient_y),
        where=gradient_norm > 1.0e-7,
    )

    # Project the recent front velocity ahead of the observed perimeter and
    # taper its influence.  This is the advancing-front localization operator.
    # The normal displacement between observed fronts is more stable at the
    # rasterized outer boundary than a one-sided derivative next to the
    # constant exterior extension.
    observed_front_speed = np.clip(
        distance_from_earlier / elapsed_min,
        0.05,
        300.0,
    )
    _, nearest_front = distance_transform_edt(~later_front, return_indices=True)
    front_speed = observed_front_speed[
        nearest_front[0],
        nearest_front[1],
    ]
    front_head_x = head_x[nearest_front[0], nearest_front[1]]
    front_head_y = head_y[nearest_front[0], nearest_front[1]]
    distance_from_front = distance_transform_edt(~later_front) * cell_size_m
    localization_m = max(localization_distance_cells * cell_size_m, cell_size_m)
    confidence = np.exp(-0.5 * (distance_from_front / localization_m) ** 2)
    confidence[earlier & ~earlier_front] *= 0.15
    confidence = np.clip(confidence, 0.0, 1.0)

    burn_age = np.zeros(later.shape, dtype=np.float64)
    burn_age[later] = np.maximum(-arrival[later], 0.0)
    heat_flux_fraction = np.zeros(later.shape, dtype=np.float64)
    heat_flux_fraction[later] = np.exp(-burn_age[later] / max(30.0, 0.35 * elapsed_min))

    _, growth_components = label(growth)
    frontier_speeds = front_speed[later_front]
    diagnostics: dict[str, float | int] = {
        "elapsed_min": float(elapsed_min),
        "earlier_cells": int(earlier.sum()),
        "later_cells": int(later.sum()),
        "growth_cells": int(growth.sum()),
        "non_nested_earlier_cells_removed": removed_cells,
        "growth_components": int(growth_components),
        "harmonic_iterations": int(iterations),
        "harmonic_residual_min": residual_min,
        "reference_speed_m_min": float(reference_speed),
        "front_speed_median_m_min": (
            float(np.median(frontier_speeds)) if frontier_speeds.size else float(reference_speed)
        ),
        "front_speed_p95_m_min": (
            float(np.quantile(frontier_speeds, 0.95)) if frontier_speeds.size else float(reference_speed)
        ),
    }
    return ArrivalHistory(
        arrival_time_min=arrival.astype(np.float32),
        burn_age_min=burn_age.astype(np.float32),
        speed_m_min=front_speed.astype(np.float32),
        head_x=front_head_x.astype(np.float32),
        head_y=front_head_y.astype(np.float32),
        confidence=confidence.astype(np.float32),
        heat_flux_fraction=heat_flux_fraction.astype(np.float32),
        diagnostics=diagnostics,
    )
