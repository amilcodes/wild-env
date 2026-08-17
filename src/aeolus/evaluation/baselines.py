"""Leakage-resistant geometric baselines for historical fire progression."""

from __future__ import annotations

from math import ceil, pi, sqrt

import numpy as np


def equivalent_radius_m(mask: np.ndarray, cell_size_m: float) -> float:
    """Radius of a circle with the same rasterized area."""

    area_m2 = float(np.asarray(mask, dtype=np.bool_).sum()) * cell_size_m**2
    return sqrt(area_m2 / pi)


def isotropic_spread_prediction(
    initial_mask: np.ndarray,
    *,
    rate_m_min: float,
    duration_min: float,
    cell_size_m: float,
    burnable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Constant normal-rate dilation, including persistence at zero rate."""

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for geometric baselines") from exc
    if rate_m_min < 0.0 or duration_min < 0.0 or cell_size_m <= 0.0:
        raise ValueError("spread rate and duration must be non-negative; cell size positive")
    initial = np.asarray(initial_mask, dtype=np.bool_)
    distance_m = distance_transform_edt(~initial) * cell_size_m
    predicted = initial | (distance_m <= rate_m_min * duration_min)
    if burnable_mask is not None:
        predicted &= np.asarray(burnable_mask, dtype=np.bool_) | initial
    return predicted


def recent_equivalent_radius_rate_m_min(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    *,
    elapsed_min: float,
    cell_size_m: float,
) -> float:
    """Non-negative observed equivalent-radius trend available at issue time."""

    if elapsed_min <= 0.0:
        raise ValueError("elapsed observation time must be positive")
    previous = equivalent_radius_m(previous_mask, cell_size_m)
    current = equivalent_radius_m(current_mask, cell_size_m)
    return max(0.0, current - previous) / elapsed_min


def _wind_ellipse_footprint(
    *,
    heading_deg: float,
    head_distance_cells: float,
    flank_ratio: float,
    backing_ratio: float,
    maximum_radius_cells: int,
) -> np.ndarray:
    if not 0.0 < flank_ratio <= 1.0 or not 0.0 < backing_ratio <= 1.0:
        raise ValueError("ellipse flank and backing ratios must be within (0, 1]")
    head = min(float(maximum_radius_cells), max(0.0, head_distance_cells))
    back = max(1.0, backing_ratio * head)
    flank = max(1.0, flank_ratio * head)
    radius = max(1, min(maximum_radius_cells, int(ceil(max(head, back, flank)))))
    rows, columns = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    bearing = np.deg2rad(heading_deg)
    along = columns * np.sin(bearing) - rows * np.cos(bearing)
    cross = columns * np.cos(bearing) + rows * np.sin(bearing)
    longitudinal_radius = np.where(along >= 0.0, max(head, 1.0), back)
    return (along / longitudinal_radius) ** 2 + (cross / flank) ** 2 <= 1.0


def wind_ellipse_prediction(
    initial_mask: np.ndarray,
    *,
    head_rate_m_min: float,
    duration_min: float,
    cell_size_m: float,
    wind_from_direction_deg: float,
    flank_ratio: float = 0.35,
    backing_ratio: float = 0.15,
    burnable_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Single-step asymmetric elliptical dilation aligned with issue-time wind."""

    try:
        from scipy.ndimage import binary_dilation
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for geometric baselines") from exc
    if head_rate_m_min < 0.0 or duration_min < 0.0 or cell_size_m <= 0.0:
        raise ValueError("spread rate and duration must be non-negative; cell size positive")
    initial = np.asarray(initial_mask, dtype=np.bool_)
    if head_rate_m_min == 0.0 or duration_min == 0.0:
        return initial.copy()
    footprint = _wind_ellipse_footprint(
        heading_deg=(float(wind_from_direction_deg) + 180.0) % 360.0,
        head_distance_cells=head_rate_m_min * duration_min / cell_size_m,
        flank_ratio=flank_ratio,
        backing_ratio=backing_ratio,
        maximum_radius_cells=max(initial.shape),
    )
    predicted = binary_dilation(initial, structure=footprint)
    if burnable_mask is not None:
        predicted &= np.asarray(burnable_mask, dtype=np.bool_) | initial
    return predicted
