from __future__ import annotations

import numpy as np
import torch

from aeolus.config import FireBehaviorConfig, ScenarioConfig
from aeolus.core.fire import _resolve_behavior, update_fuel_moisture
from aeolus.core.fire_behavior import fire_behavior_lookup
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import FireType
from aeolus.core.tensor_fire import TensorFireKernel, make_synthetic_batch
from aeolus.data import fuel_load_from_fbfm40


def _numpy_behavior(
    *,
    fuel: int = 122,
    moisture: float = 0.07,
    live_herbaceous: float = 0.75,
    live_woody: float = 0.60,
    wind: float = 6.0,
    slope_x: float = 0.0,
    canopy_cover: float = 0.0,
    canopy_height: float = 0.0,
    canopy_base_height: float = 0.0,
    canopy_bulk_density: float = 0.0,
):
    shape = (1, 1)

    def full(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    return fire_behavior_lookup().resolve_numpy(
        fuel_model_number=np.full(shape, fuel, dtype=np.int16),
        moisture_dead_1h=full(moisture),
        moisture_live_herbaceous=full(live_herbaceous),
        moisture_live_woody=full(live_woody),
        wind_speed_10m_m_s=wind,
        wind_from_direction_deg=270.0,
        terrain_slope_x=full(slope_x),
        terrain_slope_y=full(0.0),
        canopy_cover=full(canopy_cover),
        canopy_height_m=full(canopy_height),
        canopy_base_height_m=full(canopy_base_height),
        canopy_bulk_density_kg_m3=full(canopy_bulk_density),
        foliar_moisture=full(1.0),
        config=FireBehaviorConfig(),
    )


def test_reference_nodes_match_pyretechnics_fixtures() -> None:
    expected = {
        1: (21.3326473, 357.094971),
        101: (2.43117833, 34.8823547),
        122: (7.64058399, 652.840576),
        145: (33.6632195, 10242.7627),
        183: (0.351458907, 12.6108122),
        202: (3.80055141, 690.542480),
    }
    for model, (ros, intensity) in expected.items():
        behavior = _numpy_behavior(fuel=model)
        assert np.isclose(behavior.spread_rate_m_min.item(), ros, rtol=2e-6)
        assert np.isclose(behavior.fireline_intensity_kw_m.item(), intensity, rtol=2e-6)


def test_crown_transition_resolves_passive_and_active_types() -> None:
    surface = _numpy_behavior()
    active = _numpy_behavior(
        canopy_cover=0.75,
        canopy_height=18.0,
        canopy_base_height=2.0,
        canopy_bulk_density=0.20,
    )
    assert surface.fire_type.item() == FireType.SURFACE
    assert active.fire_type.item() == FireType.ACTIVE_CROWN
    assert active.fireline_intensity_kw_m.item() > surface.fireline_intensity_kw_m.item()


def test_dynamic_grass_model_responds_to_live_herbaceous_moisture() -> None:
    cured = _numpy_behavior(fuel=102, live_herbaceous=0.30, wind=4.0)
    green = _numpy_behavior(fuel=102, live_herbaceous=2.50, wind=4.0)
    static_cured = _numpy_behavior(fuel=1, live_herbaceous=0.30, wind=4.0)
    static_green = _numpy_behavior(fuel=1, live_herbaceous=2.50, wind=4.0)
    assert cured.spread_rate_m_min.item() > 100.0 * green.spread_rate_m_min.item()
    assert np.isclose(
        static_cured.spread_rate_m_min.item(),
        static_green.spread_rate_m_min.item(),
    )


def test_numpy_and_torch_behavior_paths_agree() -> None:
    rng = np.random.default_rng(4)
    shape = (2, 8, 7)
    fuel = rng.choice([1, 101, 122, 145, 183, 202], size=shape).astype(np.int16)
    moisture = rng.uniform(0.04, 0.20, size=shape).astype(np.float32)
    live_herbaceous = rng.uniform(0.30, 2.50, size=shape).astype(np.float32)
    live_woody = rng.uniform(0.60, 2.00, size=shape).astype(np.float32)
    slope_x = rng.uniform(-0.35, 0.35, size=shape).astype(np.float32)
    slope_y = rng.uniform(-0.35, 0.35, size=shape).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    numpy_result = fire_behavior_lookup().resolve_numpy(
        fuel_model_number=fuel,
        moisture_dead_1h=moisture,
        moisture_live_herbaceous=live_herbaceous,
        moisture_live_woody=live_woody,
        wind_speed_10m_m_s=5.4,
        wind_from_direction_deg=238.0,
        terrain_slope_x=slope_x,
        terrain_slope_y=slope_y,
        canopy_cover=zeros,
        canopy_height_m=zeros,
        canopy_base_height_m=zeros,
        canopy_bulk_density_kg_m3=zeros,
        foliar_moisture=np.ones(shape, dtype=np.float32),
        config=FireBehaviorConfig(),
    )
    tensor_result = fire_behavior_lookup().resolve_torch(
        fuel_model_number=torch.from_numpy(fuel),
        moisture_dead_1h=torch.from_numpy(moisture),
        moisture_live_herbaceous=torch.from_numpy(live_herbaceous),
        moisture_live_woody=torch.from_numpy(live_woody),
        wind_speed_10m_m_s=torch.full((2, 1, 1), 5.4),
        wind_from_direction_deg=torch.full((2, 1, 1), 238.0),
        terrain_slope_x=torch.from_numpy(slope_x),
        terrain_slope_y=torch.from_numpy(slope_y),
        canopy_cover=torch.from_numpy(zeros),
        canopy_height_m=torch.from_numpy(zeros),
        canopy_base_height_m=torch.from_numpy(zeros),
        canopy_bulk_density_kg_m3=torch.from_numpy(zeros),
        foliar_moisture=torch.ones(shape),
        spread_adjustment=torch.ones(shape),
        config=FireBehaviorConfig(),
    )
    for numpy_value, torch_value in (
        (numpy_result.spread_rate_m_min, tensor_result.spread_rate_m_min),
        (
            numpy_result.fireline_intensity_kw_m,
            tensor_result.fireline_intensity_kw_m,
        ),
        (numpy_result.eccentricity, tensor_result.eccentricity),
        (numpy_result.head_x, tensor_result.head_x),
        (numpy_result.head_y, tensor_result.head_y),
    ):
        assert np.allclose(numpy_value, torch_value.numpy(), rtol=2e-5, atol=2e-6)


def test_equilibrium_moisture_drives_drying_and_rain_wetting() -> None:
    config = ScenarioConfig(width=24, height=24)
    dry = AeolusSimulator(config).state.truth
    wet = AeolusSimulator(config).state.truth
    dry.moisture_dead_1h[:] = 0.18
    wet.moisture_dead_1h[:] = 0.18
    update_fuel_moisture(
        dry,
        config,
        air_temperature_c=38.0,
        relative_humidity_pct=9.0,
        precipitation_rate_mm_h=0.0,
        dt_min=60.0,
    )
    update_fuel_moisture(
        wet,
        config,
        air_temperature_c=18.0,
        relative_humidity_pct=95.0,
        precipitation_rate_mm_h=12.0,
        dt_min=60.0,
    )
    assert dry.moisture_dead_1h.mean() < 0.18
    assert wet.moisture_dead_1h.mean() > 0.18


def test_tensor_batch_preserves_independent_weather_response() -> None:
    state = make_synthetic_batch(
        batch_size=3,
        height=48,
        width=48,
        cell_size_m=20.0,
    )
    kernel = TensorFireKernel(cell_size_m=20.0)
    for minute in range(1, 16):
        kernel.step(
            state,
            minute=minute,
            wind_speed_m_s=torch.tensor([1.0, 5.0, 10.0]),
            wind_from_direction_deg=270.0,
        )
    burned_or_active = (state.phase != 0).flatten(1).sum(dim=1)
    assert burned_or_active[0] < burned_or_active[1] < burned_or_active[2]


def test_lookup_provenance_is_packaged() -> None:
    provenance = fire_behavior_lookup().provenance
    assert provenance["reference"] == "pyretechnics"
    assert provenance["reference_version"] == "2025.5.15"
    assert provenance["surface_lw_ratio_model"] == "behave"
    assert provenance["units"]["fuel_load"] == "kg m-2 oven-dry surface fuel"


def test_standard_fuel_load_comes_from_model_loading_classes() -> None:
    codes = np.asarray([91, 101, 122, 204], dtype=np.int16)
    loads = fuel_load_from_fbfm40(codes)
    assert loads[0] == 0.0
    assert np.isclose(loads[1], 0.0184 * 4.88242763638305, rtol=1e-5)
    assert np.isclose(loads[2], 0.1194 * 4.88242763638305, rtol=1e-5)
    assert np.isclose(loads[3], 0.6427 * 4.88242763638305, rtol=1e-5)


def test_physical_fuel_mass_is_not_applied_twice_to_standard_ros() -> None:
    config = ScenarioConfig(
        width=24,
        height=24,
        wind_variability=0.0,
        residual_spread_std=0.0,
    )
    truth = AeolusSimulator(config).state.truth
    truth.elevation_m[:] = 0.0
    truth.residual_field[:] = 1.0
    truth.fuel_model_number[:] = 122
    truth.moisture_dead_1h[:] = 0.07
    truth.fuel_load[:, :12] = 0.1
    truth.fuel_load[:, 12:] = 2.0
    behavior = _resolve_behavior(
        truth,
        config,
        wind_speed_m_s=4.0,
        wind_direction_deg=270.0,
    )
    assert np.allclose(
        behavior.spread_rate_m_min[:, :12],
        behavior.spread_rate_m_min[:, 12:],
    )
