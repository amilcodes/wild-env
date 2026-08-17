"""Standard surface-fuel properties packaged with the behavior lookup."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files

import numpy as np


@lru_cache(maxsize=1)
def _fuel_property_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resource = files("aeolus").joinpath("resources/fire_behavior_lookup.npz")
    with as_file(resource) as path, np.load(path, allow_pickle=False) as payload:
        models = payload["fuel_model_numbers"].astype(np.int64)
        fuel_load = payload["fuel_load_kg_m2"].astype(np.float32)
        fuel_depth = payload["fuel_bed_depth_m"].astype(np.float32)
    load_by_code = np.zeros(256, dtype=np.float32)
    depth_by_code = np.zeros(256, dtype=np.float32)
    load_by_code[models] = fuel_load
    depth_by_code[models] = fuel_depth
    return models, load_by_code, depth_by_code


def fuel_load_from_fbfm40(codes: np.ndarray) -> np.ndarray:
    """Map FBFM40 codes to oven-dry surface load in kg/m²."""

    _, load_by_code, _ = _fuel_property_tables()
    indices = np.clip(np.asarray(codes, dtype=np.int64), 0, 255)
    return load_by_code[indices].astype(np.float32)


def fuel_bed_depth_from_fbfm40(codes: np.ndarray) -> np.ndarray:
    """Map FBFM40 codes to standard fuel-bed depth in metres."""

    _, _, depth_by_code = _fuel_property_tables()
    indices = np.clip(np.asarray(codes, dtype=np.int64), 0, 255)
    return depth_by_code[indices].astype(np.float32)
