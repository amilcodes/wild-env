"""Portable, versioned scenario bundles independent of a particular GIS stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


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
    fuel_model_number: np.ndarray | None = None
    canopy_cover: np.ndarray | None = None
    canopy_height_m: np.ndarray | None = None
    canopy_base_height_m: np.ndarray | None = None
    canopy_bulk_density_kg_m3: np.ndarray | None = None

    def validate(self) -> None:
        shape = self.elevation_m.shape
        if len(shape) != 2 or min(shape) < 16:
            raise ValueError("bundle rasters must be two-dimensional and at least 16 cells per axis")
        required_arrays = {
            "fuel_load_kg_m2": self.fuel_load_kg_m2,
            "barrier": self.barrier,
            "asset_value": self.asset_value,
        }
        optional_arrays = {
            "fuel_model_number": self.fuel_model_number,
            "canopy_cover": self.canopy_cover,
            "canopy_height_m": self.canopy_height_m,
            "canopy_base_height_m": self.canopy_base_height_m,
            "canopy_bulk_density_kg_m3": self.canopy_bulk_density_kg_m3,
        }
        for name, value in required_arrays.items():
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match elevation shape {shape}")
        for name, value in optional_arrays.items():
            if value is not None and value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match elevation shape {shape}")
        required = {"schema_version", "crs", "cell_size_m", "sources", "transformations", "split"}
        missing = required.difference(self.metadata)
        if missing:
            raise ValueError(f"bundle metadata missing required fields: {sorted(missing)}")
        if self.metadata["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported bundle schema {self.metadata['schema_version']}")
        if float(self.metadata["cell_size_m"]) <= 0:
            raise ValueError("cell_size_m must be positive")
        if np.any(self.fuel_load_kg_m2 < 0) or np.any(self.asset_value < 0):
            raise ValueError("fuel load and asset value must be non-negative")
        if self.canopy_cover is not None and np.any(
            (self.canopy_cover < 0) | (self.canopy_cover > 1)
        ):
            raise ValueError("canopy_cover must be a fraction within [0, 1]")
        for name, value in optional_arrays.items():
            if name != "fuel_model_number" and value is not None and np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")


def write_bundle(path: str | Path, bundle: ScenarioBundle) -> None:
    bundle.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "elevation_m": bundle.elevation_m.astype(np.float32),
        "fuel_load_kg_m2": bundle.fuel_load_kg_m2.astype(np.float32),
        "barrier": bundle.barrier.astype(np.bool_),
        "asset_value": bundle.asset_value.astype(np.float32),
        "metadata_json": np.array(json.dumps(bundle.metadata, sort_keys=True)),
    }
    for name, value in {
        "fuel_model_number": bundle.fuel_model_number,
        "canopy_cover": bundle.canopy_cover,
        "canopy_height_m": bundle.canopy_height_m,
        "canopy_base_height_m": bundle.canopy_base_height_m,
        "canopy_bulk_density_kg_m3": bundle.canopy_bulk_density_kg_m3,
    }.items():
        if value is not None:
            payload[name] = value.astype(np.int16 if name == "fuel_model_number" else np.float32)
    np.savez_compressed(
        destination,
        **payload,
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
            fuel_model_number=(
                payload["fuel_model_number"].astype(np.int16)
                if "fuel_model_number" in payload
                else None
            ),
            canopy_cover=(
                payload["canopy_cover"].astype(np.float32)
                if "canopy_cover" in payload
                else None
            ),
            canopy_height_m=(
                payload["canopy_height_m"].astype(np.float32)
                if "canopy_height_m" in payload
                else None
            ),
            canopy_base_height_m=(
                payload["canopy_base_height_m"].astype(np.float32)
                if "canopy_base_height_m" in payload
                else None
            ),
            canopy_bulk_density_kg_m3=(
                payload["canopy_bulk_density_kg_m3"].astype(np.float32)
                if "canopy_bulk_density_kg_m3" in payload
                else None
            ),
        )
    bundle.validate()
    return bundle
