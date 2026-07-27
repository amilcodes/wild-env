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
