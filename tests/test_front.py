from __future__ import annotations

import numpy as np
import torch

from aeolus.config import FireBehaviorConfig, ScenarioConfig
from aeolus.core.front import (
    advance_level_set,
    one_sided_derivatives,
    signed_distance,
)
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import FirePhase
from aeolus.core.tensor_fire import TensorFireKernel, make_synthetic_batch
from aeolus.policies import no_aerial_action


def test_weno5_derivative_is_exact_for_linear_field() -> None:
    y, x = np.mgrid[:32, :35]
    field = (2.5 * x - 1.75 * y + 4.0).astype(np.float32)
    x_minus, x_plus = one_sided_derivatives(field, 1.0, 1, solver="weno5")
    y_minus, y_plus = one_sided_derivatives(field, 1.0, 0, solver="weno5")
    interior = np.s_[4:-4, 4:-4]
    assert np.allclose(x_minus[interior], 2.5, atol=2e-5)
    assert np.allclose(x_plus[interior], 2.5, atol=2e-5)
    assert np.allclose(y_minus[interior], -1.75, atol=2e-5)
    assert np.allclose(y_plus[interior], -1.75, atol=2e-5)


def test_level_set_expands_circle_at_prescribed_rate() -> None:
    size = 81
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    initial_radius = 10.0
    phi = (np.hypot(x - center, y - center) - initial_radius).astype(np.float32)
    ones = np.ones_like(phi)
    zeros = np.zeros_like(phi)
    for _ in range(10):
        phi, diagnostics = advance_level_set(
            phi,
            head_rate_m_min=ones,
            head_x=ones,
            head_y=zeros,
            eccentricity=zeros,
            burnable=np.ones_like(phi, dtype=np.bool_),
            cell_size_m=1.0,
            dt_min=0.2,
            solver="weno5",
            band_width_cells=16.0,
        )
    represented = phi <= 0.0
    radial_distance = np.hypot(x - center, y - center)
    measured_radius = float(radial_distance[represented].max())
    assert abs(measured_radius - 12.0) <= 0.75
    assert np.isclose(diagnostics.maximum_courant, 0.2)


def test_coarse_subgrid_front_survives_nominal_cell_residence() -> None:
    config = ScenarioConfig(
        width=32,
        height=32,
        cell_size_m=150.0,
        horizon_min=45,
        decision_interval_min=3,
        wind_speed_m_s=0.4,
        wind_variability=0.0,
        spotting_rate=0.0,
        fire=FireBehaviorConfig(
            surface_spread_adjustment=0.02,
            crown_spread_adjustment=0.02,
            min_front_residence_min=5.0,
            max_front_residence_min=20.0,
        ),
    )
    simulator = AeolusSimulator(config)
    while not simulator.state.terminated and not simulator.state.truncated:
        simulator.decision_step(no_aerial_action(simulator))
    assert simulator.state.minute == config.horizon_min
    assert simulator.state.truncated
    assert not simulator.state.contained
    assert np.any(simulator.state.truth.phase == FirePhase.FLAMING)


def test_tensor_level_set_respects_spanning_barrier() -> None:
    state = make_synthetic_batch(
        batch_size=1,
        height=41,
        width=41,
        cell_size_m=10.0,
    )
    state.barrier[:, :, 24] = True
    settings = FireBehaviorConfig(
        front_solver="weno5_level_set",
        enable_spotting=False,
        level_set_reinitialization_interval_min=5,
    )
    kernel = TensorFireKernel(cell_size_m=10.0, config=settings)
    for minute in range(1, 31):
        kernel.step(
            state,
            minute=minute,
            wind_speed_m_s=4.0,
            wind_from_direction_deg=270.0,
        )
    assert not torch.any(state.phase[:, :, 25:] != int(FirePhase.UNBURNED))


def test_numpy_and_tensor_front_steps_agree() -> None:
    state = make_synthetic_batch(
        batch_size=1,
        height=33,
        width=35,
        cell_size_m=15.0,
    )
    settings = FireBehaviorConfig(
        front_solver="weno5_level_set",
        enable_spotting=False,
        level_set_reinitialization_interval_min=0,
    )
    kernel = TensorFireKernel(cell_size_m=15.0, config=settings)
    behavior = kernel.behavior(
        state,
        wind_speed_m_s=5.0,
        wind_from_direction_deg=245.0,
    )
    burnable = torch.ones_like(state.barrier)
    tensor_result = kernel._advance_level_set(
        state.level_set_m,
        behavior,
        burnable,
        0.2,
    )
    numpy_result, _ = advance_level_set(
        state.level_set_m[0].numpy(),
        head_rate_m_min=behavior.spread_rate_m_min[0].numpy(),
        head_x=behavior.head_x[0].numpy(),
        head_y=behavior.head_y[0].numpy(),
        eccentricity=behavior.eccentricity[0].numpy(),
        burnable=burnable[0].numpy(),
        cell_size_m=15.0,
        dt_min=0.2,
        solver="weno5",
        band_width_cells=settings.level_set_band_width_cells,
    )
    assert np.allclose(
        tensor_result[0].numpy(),
        numpy_result,
        rtol=2e-5,
        atol=2e-4,
    )


def test_signed_distance_has_consistent_sign() -> None:
    mask = np.zeros((25, 25), dtype=np.bool_)
    mask[8:17, 9:16] = True
    phi = signed_distance(mask, 30.0)
    assert np.all(phi[mask] < 0.0)
    assert np.all(phi[~mask] > 0.0)


def test_signed_distance_degenerate_domains_remain_finite() -> None:
    empty = signed_distance(np.zeros((12, 15), dtype=np.bool_), 30.0)
    full = signed_distance(np.ones((12, 15), dtype=np.bool_), 30.0)
    assert np.isfinite(empty).all()
    assert np.isfinite(full).all()
    assert np.all(empty > 0.0)
    assert np.all(full < 0.0)
