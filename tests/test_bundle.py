from __future__ import annotations

import numpy as np

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import ScenarioBundle, load_bundle, write_bundle


def test_bundle_round_trip_and_simulator_load(tmp_path) -> None:
    shape = (24, 24)
    bundle = ScenarioBundle(
        elevation_m=np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape),
        fuel_load_kg_m2=np.full(shape, 0.8, dtype=np.float32),
        barrier=np.zeros(shape, dtype=bool),
        asset_value=np.eye(shape[0], dtype=np.float32),
        metadata={
            "schema_version": 1,
            "crs": "EPSG:32610",
            "cell_size_m": 60.0,
            "sources": [{"name": "synthetic-test", "version": "1"}],
            "transformations": ["unit-test"],
            "split": "test",
        },
    )
    path = tmp_path / "scenario.npz"
    write_bundle(path, bundle)
    restored = load_bundle(path)
    assert np.array_equal(restored.elevation_m, bundle.elevation_m)
    sim = AeolusSimulator(
        ScenarioConfig(width=24, height=24, cell_size_m=60.0, landscape_bundle=str(path), max_tasks=16)
    )
    assert np.array_equal(sim.state.truth.asset_value, bundle.asset_value)


def test_v2_bundle_preserves_operational_fuel_and_canopy_fields(tmp_path) -> None:
    shape = (24, 24)
    fuel_model = np.full(shape, 145, dtype=np.int16)
    canopy_cover = np.full(shape, 0.62, dtype=np.float32)
    bundle = ScenarioBundle(
        elevation_m=np.zeros(shape, dtype=np.float32),
        fuel_load_kg_m2=np.full(shape, 1.05, dtype=np.float32),
        barrier=np.zeros(shape, dtype=bool),
        asset_value=np.zeros(shape, dtype=np.float32),
        fuel_model_number=fuel_model,
        canopy_cover=canopy_cover,
        canopy_height_m=np.full(shape, 16.0, dtype=np.float32),
        canopy_base_height_m=np.full(shape, 2.3, dtype=np.float32),
        canopy_bulk_density_kg_m3=np.full(shape, 0.17, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": "EPSG:32610",
            "cell_size_m": 30.0,
            "sources": [{"name": "synthetic-test"}],
            "transformations": ["unit-test"],
            "split": "test",
        },
    )
    path = tmp_path / "scenario-v2.npz"
    write_bundle(path, bundle)
    restored = load_bundle(path)
    assert np.array_equal(restored.fuel_model_number, fuel_model)
    assert np.array_equal(restored.canopy_cover, canopy_cover)
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=24,
            height=24,
            cell_size_m=30.0,
            landscape_bundle=str(path),
            max_tasks=16,
        )
    )
    assert np.array_equal(simulator.state.truth.fuel_model_number, fuel_model)
    assert np.array_equal(simulator.state.truth.canopy_cover, canopy_cover)
