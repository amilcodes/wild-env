#!/usr/bin/env python3
"""Generate the accelerator fire-behavior table from Pyretechnics.

The generated table is a derived package artifact.  It lets the training
kernel use the same Rothermel/Scott-Burgan implementation as Pyretechnics
without invoking a scalar Cython reference model for every raster cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyretechnics import fuel_models
from pyretechnics.burn_cells import burn_cell_as_head_fire, burn_cell_toward_azimuth
from pyretechnics.space_time_cube import SpaceTimeCube

MOISTURE_1H = np.asarray([0.03, 0.05, 0.07, 0.09, 0.12, 0.16, 0.22, 0.30, 0.40], dtype=np.float32)
MOISTURE_LIVE_HERBACEOUS = np.asarray(
    [0.30, 0.60, 0.75, 0.90, 1.20, 1.60, 2.00, 2.50],
    dtype=np.float32,
)
MOISTURE_LIVE_WOODY = np.asarray([0.60, 1.00, 1.40, 2.00], dtype=np.float32)
WIND_10M_M_S = np.asarray(
    [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0],
    dtype=np.float32,
)
SLOPE_TAN = np.asarray([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00], dtype=np.float32)
LB_FT2_TO_KG_M2 = 4.88242763638305
FT_TO_M = 0.3048


def _cubes(
    fuel_model: int,
    moisture: float,
    live_herbaceous: float,
    live_woody: float,
    wind_m_s: float,
    slope: float,
) -> dict:
    values = {
        "slope": slope,
        "aspect": 270.0,  # downslope west, so the upslope heading is east
        "fuel_model": fuel_model,
        "canopy_cover": 0.0,
        "canopy_height": 0.0,
        "canopy_base_height": 0.0,
        "canopy_bulk_density": 0.0,
        "wind_speed_10m": wind_m_s * 3.6,
        "upwind_direction": 270.0,  # wind is from west and heads east
        "fuel_moisture_dead_1hr": moisture,
        "fuel_moisture_dead_10hr": min(0.40, moisture + 0.02),
        "fuel_moisture_dead_100hr": min(0.40, moisture + 0.04),
        "fuel_moisture_live_herbaceous": live_herbaceous,
        "fuel_moisture_live_woody": live_woody,
        "foliar_moisture": 1.00,
    }
    return {key: SpaceTimeCube((1, 1, 1), value) for key, value in values.items()}


def _behavior(
    fuel_model: int,
    moisture: float,
    live_herbaceous: float,
    live_woody: float,
    wind: float,
    slope: float,
) -> tuple:
    cubes = _cubes(
        fuel_model,
        moisture,
        live_herbaceous,
        live_woody,
        wind,
        slope,
    )
    head = burn_cell_as_head_fire(cubes, (0, 0, 0), surface_lw_ratio_model="behave")
    back = burn_cell_toward_azimuth(cubes, (0, 0, 0), 270.0, surface_lw_ratio_model="behave")
    return (
        float(head["spread_rate"]),
        float(back["spread_rate"]),
        float(head["fireline_intensity"]),
        float(head["flame_length"]),
    )


def build(destination: Path) -> None:
    models = np.asarray(
        sorted(model for model in fuel_models.list_fuel_model_numbers() if not 91 <= model <= 99),
        dtype=np.int16,
    )
    fuel_load = np.asarray(
        [sum(fuel_models.get_fuel_model(int(model))["w_o"]) * LB_FT2_TO_KG_M2 for model in models],
        dtype=np.float32,
    )
    fuel_bed_depth = np.asarray(
        [fuel_models.get_fuel_model(int(model))["delta"] * FT_TO_M for model in models],
        dtype=np.float32,
    )
    moisture_shape = (
        len(models),
        len(MOISTURE_1H),
        len(MOISTURE_LIVE_HERBACEOUS),
        len(MOISTURE_LIVE_WOODY),
    )
    wind_shape = (*moisture_shape, len(WIND_10M_M_S))
    slope_shape = (*moisture_shape, len(SLOPE_TAN))
    wind_head = np.zeros(wind_shape, dtype=np.float32)
    wind_back = np.zeros(wind_shape, dtype=np.float32)
    wind_intensity = np.zeros(wind_shape, dtype=np.float32)
    wind_flame = np.zeros(wind_shape, dtype=np.float32)
    slope_head = np.zeros(slope_shape, dtype=np.float32)
    slope_back = np.zeros(slope_shape, dtype=np.float32)
    slope_intensity = np.zeros(slope_shape, dtype=np.float32)
    slope_flame = np.zeros(slope_shape, dtype=np.float32)

    for model_i, model in enumerate(models):
        for moisture_i, moisture in enumerate(MOISTURE_1H):
            for herb_i, live_herbaceous in enumerate(MOISTURE_LIVE_HERBACEOUS):
                for woody_i, live_woody in enumerate(MOISTURE_LIVE_WOODY):
                    index = (model_i, moisture_i, herb_i, woody_i)
                    for wind_i, wind in enumerate(WIND_10M_M_S):
                        values = _behavior(
                            int(model),
                            float(moisture),
                            float(live_herbaceous),
                            float(live_woody),
                            float(wind),
                            0.0,
                        )
                        wind_head[index + (wind_i,)] = values[0]
                        wind_back[index + (wind_i,)] = values[1]
                        wind_intensity[index + (wind_i,)] = values[2]
                        wind_flame[index + (wind_i,)] = values[3]
                    for slope_i, slope in enumerate(SLOPE_TAN):
                        values = _behavior(
                            int(model),
                            float(moisture),
                            float(live_herbaceous),
                            float(live_woody),
                            0.0,
                            float(slope),
                        )
                        slope_head[index + (slope_i,)] = values[0]
                        slope_back[index + (slope_i,)] = values[1]
                        slope_intensity[index + (slope_i,)] = values[2]
                        slope_flame[index + (slope_i,)] = values[3]

    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generator": "tools/build_fire_lookup.py",
        "reference": "pyretechnics",
        "reference_version": "2025.5.15",
        "surface_lw_ratio_model": "behave",
        "wind_limit": True,
        "moisture_assumptions": {
            "dead_10h": "dead_1h + 0.02, capped at 0.40",
            "dead_100h": "dead_1h + 0.04, capped at 0.40",
            "live_herbaceous": ("explicit interpolation axis spanning NFDRS v4 0.30-2.50 kg/kg"),
            "live_woody": ("explicit interpolation axis spanning NFDRS v4 0.60-2.00 kg/kg"),
        },
        "canopy": "disabled; crown behavior is resolved at runtime",
        "fuel_load_source": ("sum of six Pyretechnics/Scott-Burgan oven-dry w_o classes"),
        "units": {
            "spread_rate": "m min-1",
            "fireline_intensity": "kW m-1",
            "flame_length": "m",
            "wind": "m s-1 at 10 m",
            "slope": "rise/run",
            "moisture": "kg kg-1 dry fuel",
            "fuel_load": "kg m-2 oven-dry surface fuel",
            "fuel_bed_depth": "m",
        },
    }
    np.savez_compressed(
        destination,
        fuel_model_numbers=models,
        fuel_load_kg_m2=fuel_load,
        fuel_bed_depth_m=fuel_bed_depth,
        moisture_1h=MOISTURE_1H,
        moisture_live_herbaceous=MOISTURE_LIVE_HERBACEOUS,
        moisture_live_woody=MOISTURE_LIVE_WOODY,
        wind_10m_m_s=WIND_10M_M_S,
        slope_tan=SLOPE_TAN,
        wind_head_ros=wind_head,
        wind_back_ros=wind_back,
        wind_head_intensity=wind_intensity,
        wind_head_flame_length=wind_flame,
        slope_head_ros=slope_head,
        slope_back_ros=slope_back,
        slope_head_intensity=slope_intensity,
        slope_head_flame_length=slope_flame,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.destination)


if __name__ == "__main__":
    main()
