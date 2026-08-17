"""Signed level-set front tracking for the canonical NumPy fire model.

The zero contour of ``phi`` is the fire perimeter.  Negative values are
burned/active and positive values are unburned.  Propagation solves the
anisotropic Hamilton--Jacobi equation

    phi_t + R(n) |grad(phi)| = 0

with third-order strong-stability-preserving Runge--Kutta time integration and
fifth-order Jiang--Shu WENO one-sided derivatives.  A first-order Godunov
derivative remains available as a verification comparator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrontStepDiagnostics:
    solver: str
    dt_min: float
    maximum_courant: float
    active_band_cells: int


def signed_distance(mask: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Return an approximately cell-centred signed Euclidean distance field."""

    inside = np.asarray(mask, dtype=np.bool_)
    if inside.ndim != 2:
        raise ValueError("signed-distance input must be a 2-D mask")
    if cell_size_m <= 0.0:
        raise ValueError("cell size must be positive")
    # A domain without an interface still needs a finite level-set state:
    # infinities contaminate WENO stencils if an extinguished or fully reached
    # domain is advanced again.
    domain_diagonal_m = float(np.hypot(*inside.shape) * cell_size_m)
    if not inside.any():
        return np.full(inside.shape, domain_diagonal_m, dtype=np.float32)
    if inside.all():
        return np.full(inside.shape, -domain_diagonal_m, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        outside_distance = distance_transform_edt(~inside)
        inside_distance = distance_transform_edt(inside)
    except ImportError:  # pragma: no cover - SciPy is part of the geo extra
        outside_distance = _chamfer_distance(inside)
        inside_distance = _chamfer_distance(~inside)
    return ((outside_distance - inside_distance) * cell_size_m).astype(np.float32)


def _chamfer_distance(features: np.ndarray) -> np.ndarray:
    """Dependency-free 8-neighbour distance fallback."""

    distance = np.where(features, 0.0, np.inf)
    diagonal = np.sqrt(2.0)
    height, width = distance.shape
    for y in range(height):
        for x in range(width):
            if y:
                distance[y, x] = min(distance[y, x], distance[y - 1, x] + 1.0)
                if x:
                    distance[y, x] = min(distance[y, x], distance[y - 1, x - 1] + diagonal)
                if x + 1 < width:
                    distance[y, x] = min(distance[y, x], distance[y - 1, x + 1] + diagonal)
            if x:
                distance[y, x] = min(distance[y, x], distance[y, x - 1] + 1.0)
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if y + 1 < height:
                distance[y, x] = min(distance[y, x], distance[y + 1, x] + 1.0)
                if x:
                    distance[y, x] = min(distance[y, x], distance[y + 1, x - 1] + diagonal)
                if x + 1 < width:
                    distance[y, x] = min(distance[y, x], distance[y + 1, x + 1] + diagonal)
            if x + 1 < width:
                distance[y, x] = min(distance[y, x], distance[y, x + 1] + 1.0)
    return distance


def _axis_last(values: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)


def _weno5_backward(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Fifth-order left-biased Hamilton--Jacobi derivative."""

    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    field = _axis_last(values, axis)
    if field.shape[-1] < 7:
        return _first_order_backward(values, spacing, axis)
    padded = np.pad(
        field,
        [(0, 0)] * (field.ndim - 1) + [(3, 3)],
        mode="edge",
    )
    size = field.shape[-1]
    v_im2 = (padded[..., 1 : size + 1] - padded[..., 0:size]) / spacing
    v_im1 = (padded[..., 2 : size + 2] - padded[..., 1 : size + 1]) / spacing
    v_i = (padded[..., 3 : size + 3] - padded[..., 2 : size + 2]) / spacing
    v_ip1 = (padded[..., 4 : size + 4] - padded[..., 3 : size + 3]) / spacing
    v_ip2 = (padded[..., 5 : size + 5] - padded[..., 4 : size + 4]) / spacing

    candidate_0 = v_im2 / 3.0 - 7.0 * v_im1 / 6.0 + 11.0 * v_i / 6.0
    candidate_1 = -v_im1 / 6.0 + 5.0 * v_i / 6.0 + v_ip1 / 3.0
    candidate_2 = v_i / 3.0 + 5.0 * v_ip1 / 6.0 - v_ip2 / 6.0

    beta_0 = 13.0 * (v_im2 - 2.0 * v_im1 + v_i) ** 2 / 12.0 + (v_im2 - 4.0 * v_im1 + 3.0 * v_i) ** 2 / 4.0
    beta_1 = 13.0 * (v_im1 - 2.0 * v_i + v_ip1) ** 2 / 12.0 + (v_im1 - v_ip1) ** 2 / 4.0
    beta_2 = 13.0 * (v_i - 2.0 * v_ip1 + v_ip2) ** 2 / 12.0 + (3.0 * v_i - 4.0 * v_ip1 + v_ip2) ** 2 / 4.0
    scale = np.maximum.reduce(
        (
            np.abs(v_im2),
            np.abs(v_im1),
            np.abs(v_i),
            np.abs(v_ip1),
            np.abs(v_ip2),
            np.ones_like(v_i),
        )
    )
    epsilon = 1e-12 * scale**2
    alpha_0 = 0.1 / (epsilon + beta_0) ** 2
    alpha_1 = 0.6 / (epsilon + beta_1) ** 2
    alpha_2 = 0.3 / (epsilon + beta_2) ** 2
    total = alpha_0 + alpha_1 + alpha_2
    derivative = (alpha_0 * candidate_0 + alpha_1 * candidate_1 + alpha_2 * candidate_2) / total
    return np.moveaxis(derivative, -1, axis).astype(np.float32)


def _first_order_backward(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    field = _axis_last(values, axis)
    padded = np.pad(
        field,
        [(0, 0)] * (field.ndim - 1) + [(1, 0)],
        mode="edge",
    )
    derivative = (padded[..., 1:] - padded[..., :-1]) / spacing
    return np.moveaxis(derivative, -1, axis).astype(np.float32)


def one_sided_derivatives(
    values: np.ndarray,
    spacing: float,
    axis: int,
    *,
    solver: str = "weno5",
) -> tuple[np.ndarray, np.ndarray]:
    """Return backward and forward derivatives along one grid axis."""

    if solver == "weno5":
        backward = _weno5_backward(values, spacing, axis)
        reversed_values = np.flip(values, axis=axis)
        forward = -np.flip(_weno5_backward(reversed_values, spacing, axis), axis=axis)
    elif solver == "godunov":
        backward = _first_order_backward(values, spacing, axis)
        reversed_values = np.flip(values, axis=axis)
        forward = -np.flip(_first_order_backward(reversed_values, spacing, axis), axis=axis)
    else:
        raise ValueError(f"unknown level-set derivative solver: {solver}")
    return backward, forward


def _hamiltonian(
    phi: np.ndarray,
    *,
    head_rate_m_min: np.ndarray,
    head_x: np.ndarray,
    head_y: np.ndarray,
    eccentricity: np.ndarray,
    burnable: np.ndarray,
    cell_size_m: float,
    solver: str,
    band_width_m: float | None,
) -> tuple[np.ndarray, int]:
    dx_minus, dx_plus = one_sided_derivatives(phi, cell_size_m, 1, solver=solver)
    dy_minus, dy_plus = one_sided_derivatives(phi, cell_size_m, 0, solver=solver)
    grad_norm = np.sqrt(
        np.maximum(dx_minus, 0.0) ** 2
        + np.minimum(dx_plus, 0.0) ** 2
        + np.maximum(dy_minus, 0.0) ** 2
        + np.minimum(dy_plus, 0.0) ** 2
    )
    centred_x = 0.5 * (dx_minus + dx_plus)
    centred_y = 0.5 * (dy_minus + dy_plus)
    centred_norm = np.hypot(centred_x, centred_y)
    normal_x = np.divide(
        centred_x,
        centred_norm,
        out=np.asarray(head_x, dtype=np.float32).copy(),
        where=centred_norm > 1e-7,
    )
    normal_y = np.divide(
        centred_y,
        centred_norm,
        out=np.asarray(head_y, dtype=np.float32).copy(),
        where=centred_norm > 1e-7,
    )
    cosine = np.clip(head_x * normal_x + head_y * normal_y, -1.0, 1.0)
    directional = (1.0 - eccentricity) / np.maximum(1.0 - eccentricity * cosine, 1e-4)
    speed = np.maximum(head_rate_m_min * directional, 0.0)
    active = np.asarray(burnable, dtype=np.bool_)
    if band_width_m is not None:
        active &= np.abs(phi) <= band_width_m
    hamiltonian = np.where(active, speed * grad_norm, 0.0)
    return hamiltonian.astype(np.float32), int(active.sum())


def advance_level_set(
    phi: np.ndarray,
    *,
    head_rate_m_min: np.ndarray,
    head_x: np.ndarray,
    head_y: np.ndarray,
    eccentricity: np.ndarray,
    burnable: np.ndarray,
    cell_size_m: float,
    dt_min: float,
    solver: str = "weno5",
    band_width_cells: float | None = 12.0,
) -> tuple[np.ndarray, FrontStepDiagnostics]:
    """Advance an anisotropic fire perimeter by one SSP-RK3 step."""

    if dt_min <= 0.0:
        raise ValueError("level-set timestep must be positive")
    original = np.asarray(phi, dtype=np.float32)
    band_width_m = None if band_width_cells is None else float(band_width_cells) * cell_size_m

    def rhs(value: np.ndarray) -> tuple[np.ndarray, int]:
        hamiltonian, band_count = _hamiltonian(
            value,
            head_rate_m_min=head_rate_m_min,
            head_x=head_x,
            head_y=head_y,
            eccentricity=eccentricity,
            burnable=burnable,
            cell_size_m=cell_size_m,
            solver=solver,
            band_width_m=band_width_m,
        )
        return -hamiltonian, band_count

    derivative_0, band_count = rhs(original)
    stage_1 = original + dt_min * derivative_0
    derivative_1, _ = rhs(stage_1)
    stage_2 = 0.75 * original + 0.25 * (stage_1 + dt_min * derivative_1)
    derivative_2, _ = rhs(stage_2)
    result = (original + 2.0 * (stage_2 + dt_min * derivative_2)) / 3.0
    result = np.where(burnable, result, np.maximum(result, 0.5 * cell_size_m))
    maximum_courant = float(np.max(head_rate_m_min) * dt_min / max(cell_size_m, 1e-9))
    return result.astype(np.float32), FrontStepDiagnostics(
        solver=solver,
        dt_min=float(dt_min),
        maximum_courant=maximum_courant,
        active_band_cells=band_count,
    )


def reinitialize_level_set(
    phi: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Restore a signed-distance field while preserving the represented mask."""

    return signed_distance(np.asarray(phi) <= 0.0, cell_size_m)
