"""Portable, versioned scenario bundles independent of a particular GIS stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScenarioBundle:
    """Rasterized inputs plus provenance for a single reproducible landscape.

    Arrays use `(y, x)` order and SI units: elevation in metres, fuel load in
    kg/m², and asset value as a non-negative dimensionless incident objective
    layer. Raw LANDFIRE/3DEP/HRRR processing is deliberately outside this class
    so every transformation is captured in the metadata manifest.
    """

    elevation_m: np.ndarray
    fuel_load_kg_m2: np.ndarray
    barrier: np.ndarray
    asset_value: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        shape = self.elevation_m.shape
        if len(shape) != 2 or min(shape) < 16:
            raise ValueError("bundle rasters must be two-dimensional and at least 16 cells per axis")
        for name, value in {
            "fuel_load_kg_m2": self.fuel_load_kg_m2,
            "barrier": self.barrier,
            "asset_value": self.asset_value,
        }.items():
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match elevation shape {shape}")
        required = {"schema_version", "crs", "cell_size_m", "sources", "transformations", "split"}
        missing = required.difference(self.metadata)
        if missing:
            raise ValueError(f"bundle metadata missing required fields: {sorted(missing)}")
        if self.metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema {self.metadata['schema_version']}")
        if float(self.metadata["cell_size_m"]) <= 0:
            raise ValueError("cell_size_m must be positive")
        if np.any(self.fuel_load_kg_m2 < 0) or np.any(self.asset_value < 0):
            raise ValueError("fuel load and asset value must be non-negative")


def write_bundle(path: str | Path, bundle: ScenarioBundle) -> None:
    bundle.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        elevation_m=bundle.elevation_m.astype(np.float32),
        fuel_load_kg_m2=bundle.fuel_load_kg_m2.astype(np.float32),
        barrier=bundle.barrier.astype(np.bool_),
        asset_value=bundle.asset_value.astype(np.float32),
        metadata_json=np.array(json.dumps(bundle.metadata, sort_keys=True)),
    )


def load_bundle(path: str | Path) -> ScenarioBundle:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        bundle = ScenarioBundle(
            elevation_m=payload["elevation_m"].astype(np.float32),
            fuel_load_kg_m2=payload["fuel_load_kg_m2"].astype(np.float32),
            barrier=payload["barrier"].astype(np.bool_),
            asset_value=payload["asset_value"].astype(np.float32),
            metadata=json.loads(str(payload["metadata_json"].item())),
        )
    bundle.validate()
    return bundle
