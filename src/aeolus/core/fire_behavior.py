"""Operational-equation fire behavior for NumPy and accelerator tensors.

Surface behavior is interpolated from a reproducible Pyretechnics table.  Wind
and slope factors are combined as vectors, following the approach used by
ELMFIRE, and crown transition uses Van Wagner plus Cruz et al.  This module
resolves local behavior; propagation is handled separately by the raster
front-tracking kernels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from typing import Any

import numpy as np

from aeolus.config import FireBehaviorConfig
from aeolus.core.state import FireType


@dataclass(frozen=True)
class FireBehaviorFields:
    spread_rate_m_min: Any
    fireline_intensity_kw_m: Any
    flame_length_m: Any
    head_x: Any
    head_y: Any
    eccentricity: Any
    fire_type: Any


def _axis_indices(values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(values, float(grid[0]), float(grid[-1]))
    high = np.searchsorted(grid, clipped, side="right")
    high = np.clip(high, 1, len(grid) - 1)
    low = high - 1
    fraction = (clipped - grid[low]) / np.maximum(grid[high] - grid[low], 1e-12)
    return low, high, fraction.astype(np.float32)


class FireBehaviorLookup:
    """The same local behavior model exposed through NumPy and PyTorch."""

    def __init__(self) -> None:
        resource = files("aeolus").joinpath("resources/fire_behavior_lookup.npz")
        with as_file(resource) as path, np.load(path, allow_pickle=False) as payload:
            self.fuel_models = payload["fuel_model_numbers"].astype(np.int64)
            self.moisture_grid = payload["moisture_1h"].astype(np.float32)
            self.wind_grid = payload["wind_10m_m_s"].astype(np.float32)
            self.slope_grid = payload["slope_tan"].astype(np.float32)
            self.wind_head = payload["wind_head_ros"].astype(np.float32)
            self.wind_back = payload["wind_back_ros"].astype(np.float32)
            self.wind_intensity = payload["wind_head_intensity"].astype(np.float32)
            self.slope_head = payload["slope_head_ros"].astype(np.float32)
            self.slope_back = payload["slope_back_ros"].astype(np.float32)
            self.slope_intensity = payload["slope_head_intensity"].astype(np.float32)
            self.provenance = json.loads(str(payload["provenance_json"].item()))
        self.code_to_index = np.full(256, -1, dtype=np.int64)
        self.code_to_index[self.fuel_models] = np.arange(len(self.fuel_models))
        self.default_index = int(np.flatnonzero(self.fuel_models == 122)[0])
        self._torch_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _fuel_indices(self, fuel_model_number: np.ndarray) -> np.ndarray:
        codes = np.clip(fuel_model_number.astype(np.int64), 0, 255)
        indices = self.code_to_index[codes]
        return np.where(indices >= 0, indices, self.default_index)

    @staticmethod
    def _interp_numpy(
        table: np.ndarray,
        fuel_index: np.ndarray,
        axis_1: tuple[np.ndarray, np.ndarray, np.ndarray],
        axis_2: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        a0, a1, af = axis_1
        b0, b1, bf = axis_2
        v00 = table[fuel_index, a0, b0]
        v01 = table[fuel_index, a0, b1]
        v10 = table[fuel_index, a1, b0]
        v11 = table[fuel_index, a1, b1]
        low = v00 + bf * (v01 - v00)
        high = v10 + bf * (v11 - v10)
        return (low + af * (high - low)).astype(np.float32)

    def resolve_numpy(
        self,
        *,
        fuel_model_number: np.ndarray,
        moisture_dead_1h: np.ndarray,
        wind_speed_10m_m_s: float | np.ndarray,
        wind_from_direction_deg: float | np.ndarray,
        terrain_slope_x: np.ndarray,
        terrain_slope_y: np.ndarray,
        canopy_cover: np.ndarray,
        canopy_height_m: np.ndarray,
        canopy_base_height_m: np.ndarray,
        canopy_bulk_density_kg_m3: np.ndarray,
        foliar_moisture: np.ndarray,
        spread_adjustment: np.ndarray | float = 1.0,
        config: FireBehaviorConfig,
    ) -> FireBehaviorFields:
        shape = fuel_model_number.shape
        moisture = np.broadcast_to(moisture_dead_1h, shape).astype(np.float32)
        wind = np.broadcast_to(wind_speed_10m_m_s, shape).astype(np.float32)
        slope = np.hypot(terrain_slope_x, terrain_slope_y).astype(np.float32)
        fuel_index = self._fuel_indices(fuel_model_number)
        moisture_axis = _axis_indices(moisture, self.moisture_grid)
        wind_axis = _axis_indices(wind, self.wind_grid)
        slope_axis = _axis_indices(slope, self.slope_grid)
        zero = np.zeros(shape, dtype=np.float32)
        zero_axis = _axis_indices(zero, self.wind_grid)

        base_ros = self._interp_numpy(
            self.wind_head, fuel_index, moisture_axis, zero_axis
        )
        base_intensity = self._interp_numpy(
            self.wind_intensity, fuel_index, moisture_axis, zero_axis
        )
        wind_head = self._interp_numpy(
            self.wind_head, fuel_index, moisture_axis, wind_axis
        )
        wind_back = self._interp_numpy(
            self.wind_back, fuel_index, moisture_axis, wind_axis
        )
        slope_head = self._interp_numpy(
            self.slope_head, fuel_index, moisture_axis, slope_axis
        )
        slope_back = self._interp_numpy(
            self.slope_back, fuel_index, moisture_axis, slope_axis
        )

        safe_base = np.maximum(base_ros, 1e-4)
        phi_w = np.maximum(wind_head / safe_base - 1.0, 0.0)
        phi_s = np.maximum(slope_head / safe_base - 1.0, 0.0)
        radians = np.deg2rad(np.broadcast_to(wind_from_direction_deg, shape))
        # Meteorological direction is where wind comes from; +y is south.
        wind_x = -np.sin(radians)
        wind_y = np.cos(radians)
        slope_safe = np.maximum(slope, 1e-8)
        upslope_x = terrain_slope_x / slope_safe
        upslope_y = terrain_slope_y / slope_safe
        upslope_x = np.where(slope > 1e-8, upslope_x, 0.0)
        upslope_y = np.where(slope > 1e-8, upslope_y, 0.0)
        vector_x = phi_w * wind_x + phi_s * upslope_x
        vector_y = phi_w * wind_y + phi_s * upslope_y
        phi = np.hypot(vector_x, vector_y)
        head_x = np.where(phi > 1e-8, vector_x / np.maximum(phi, 1e-8), wind_x)
        head_y = np.where(phi > 1e-8, vector_y / np.maximum(phi, 1e-8), wind_y)

        eccentricity_w = np.clip(
            (wind_head - wind_back) / np.maximum(wind_head + wind_back, 1e-5), 0.0, 0.985
        )
        eccentricity_s = np.clip(
            (slope_head - slope_back) / np.maximum(slope_head + slope_back, 1e-5),
            0.0,
            0.985,
        )
        eccentricity = np.where(
            phi > 1e-8,
            (phi_w * eccentricity_w + phi_s * eccentricity_s) / np.maximum(phi, 1e-8),
            0.0,
        )
        adjustment = np.broadcast_to(spread_adjustment, shape).astype(np.float32)
        surface_ros = np.clip(
            base_ros * (1.0 + phi) * adjustment * config.surface_spread_adjustment,
            0.0,
            config.max_spread_rate_m_min,
        )
        intensity = np.maximum(
            0.0, base_intensity * surface_ros / np.maximum(base_ros, 1e-4)
        )
        fire_type = np.full(shape, FireType.SURFACE, dtype=np.uint8)

        if config.enable_crown_fire:
            canopy_present = (
                (canopy_cover >= 0.20)
                & (canopy_height_m > canopy_base_height_m)
                & (canopy_base_height_m > 0.0)
                & (canopy_bulk_density_kg_m3 > 0.0)
            )
            critical_intensity = np.power(
                np.maximum(
                    0.01
                    * canopy_base_height_m
                    * (460.0 + 2600.0 * np.clip(foliar_moisture, 0.3, 2.0)),
                    0.0,
                ),
                1.5,
            )
            initiated = canopy_present & (intensity >= critical_intensity)
            active_ros = (
                11.02
                * np.power(np.maximum(wind * 3.6, 0.0), 0.90)
                * np.power(np.maximum(canopy_bulk_density_kg_m3, 1e-5), 0.19)
                * np.exp(-17.0 * np.clip(moisture, 0.0, 0.6))
                * config.crown_spread_adjustment
            )
            critical_ros = 3.0 / np.maximum(canopy_bulk_density_kg_m3, 1e-5)
            active = initiated & (active_ros >= critical_ros)
            passive_ros = active_ros * np.exp(
                -active_ros / np.maximum(critical_ros, 1e-5)
            )
            crown_ros = np.where(active, active_ros, passive_ros)
            surface_ros = np.where(initiated, np.maximum(surface_ros, crown_ros), surface_ros)
            canopy_fuel = (
                canopy_cover
                * canopy_bulk_density_kg_m3
                * np.maximum(canopy_height_m - canopy_base_height_m, 0.0)
            )
            crown_intensity = intensity + (
                18_000.0 * canopy_fuel * surface_ros / 60.0
            )
            intensity = np.where(initiated, crown_intensity, intensity)
            fire_type = np.where(
                active,
                FireType.ACTIVE_CROWN,
                np.where(initiated, FireType.PASSIVE_CROWN, FireType.SURFACE),
            ).astype(np.uint8)
            eccentricity = np.where(
                active,
                np.maximum(
                    eccentricity,
                    np.sqrt(
                        np.maximum(
                            0.0,
                            1.0
                            - 1.0
                            / np.square(np.clip(1.0 + 0.125 * wind * 3.6, 1.0, 100.0)),
                        )
                    ),
                ),
                eccentricity,
            )

        flame_length = 0.0775 * np.power(np.maximum(intensity, 0.0), 0.46)
        return FireBehaviorFields(
            spread_rate_m_min=surface_ros.astype(np.float32),
            fireline_intensity_kw_m=intensity.astype(np.float32),
            flame_length_m=flame_length.astype(np.float32),
            head_x=head_x.astype(np.float32),
            head_y=head_y.astype(np.float32),
            eccentricity=np.clip(eccentricity, 0.0, 0.985).astype(np.float32),
            fire_type=fire_type,
        )

    def _torch_tables(self, device: Any, dtype: Any) -> dict[str, Any]:
        import torch

        key = (str(device), str(dtype))
        if key not in self._torch_cache:
            arrays = {
                "moisture": self.moisture_grid,
                "wind": self.wind_grid,
                "slope": self.slope_grid,
                "wind_head": self.wind_head,
                "wind_back": self.wind_back,
                "wind_intensity": self.wind_intensity,
                "slope_head": self.slope_head,
                "slope_back": self.slope_back,
                "code_map": self.code_to_index,
            }
            self._torch_cache[key] = {
                name: torch.as_tensor(
                    value,
                    device=device,
                    dtype=torch.long if name == "code_map" else dtype,
                )
                for name, value in arrays.items()
            }
        return self._torch_cache[key]

    @staticmethod
    def _torch_axis(values: Any, grid: Any) -> tuple[Any, Any, Any]:
        import torch

        clipped = torch.clamp(values, float(grid[0]), float(grid[-1]))
        high = torch.bucketize(clipped.contiguous(), grid)
        high = torch.clamp(high, 1, grid.numel() - 1)
        low = high - 1
        fraction = (clipped - grid[low]) / torch.clamp(grid[high] - grid[low], min=1e-12)
        return low, high, fraction

    @staticmethod
    def _interp_torch(table: Any, fuel: Any, axis_1: tuple, axis_2: tuple) -> Any:
        a0, a1, af = axis_1
        b0, b1, bf = axis_2
        low = table[fuel, a0, b0] + bf * (
            table[fuel, a0, b1] - table[fuel, a0, b0]
        )
        high = table[fuel, a1, b0] + bf * (
            table[fuel, a1, b1] - table[fuel, a1, b0]
        )
        return low + af * (high - low)

    def resolve_torch(
        self,
        *,
        fuel_model_number: Any,
        moisture_dead_1h: Any,
        wind_speed_10m_m_s: Any,
        wind_from_direction_deg: Any,
        terrain_slope_x: Any,
        terrain_slope_y: Any,
        canopy_cover: Any,
        canopy_height_m: Any,
        canopy_base_height_m: Any,
        canopy_bulk_density_kg_m3: Any,
        foliar_moisture: Any,
        spread_adjustment: Any,
        config: FireBehaviorConfig,
    ) -> FireBehaviorFields:
        import torch

        dtype, device = moisture_dead_1h.dtype, moisture_dead_1h.device
        table = self._torch_tables(device, dtype)
        fuel_codes = torch.clamp(fuel_model_number.long(), 0, 255)
        fuel = table["code_map"][fuel_codes]
        fuel = torch.where(fuel >= 0, fuel, self.default_index)
        slope = torch.hypot(terrain_slope_x, terrain_slope_y)
        ma = self._torch_axis(moisture_dead_1h, table["moisture"])
        wa = self._torch_axis(wind_speed_10m_m_s.expand_as(moisture_dead_1h), table["wind"])
        sa = self._torch_axis(slope, table["slope"])
        za = self._torch_axis(torch.zeros_like(moisture_dead_1h), table["wind"])
        base_ros = self._interp_torch(table["wind_head"], fuel, ma, za)
        base_i = self._interp_torch(table["wind_intensity"], fuel, ma, za)
        wind_head = self._interp_torch(table["wind_head"], fuel, ma, wa)
        wind_back = self._interp_torch(table["wind_back"], fuel, ma, wa)
        slope_head = self._interp_torch(table["slope_head"], fuel, ma, sa)
        slope_back = self._interp_torch(table["slope_back"], fuel, ma, sa)
        safe_base = torch.clamp(base_ros, min=1e-4)
        phi_w = torch.clamp(wind_head / safe_base - 1.0, min=0.0)
        phi_s = torch.clamp(slope_head / safe_base - 1.0, min=0.0)
        radians = torch.deg2rad(wind_from_direction_deg.expand_as(moisture_dead_1h))
        wind_x, wind_y = -torch.sin(radians), torch.cos(radians)
        slope_safe = torch.clamp(slope, min=1e-8)
        ux = torch.where(slope > 1e-8, terrain_slope_x / slope_safe, 0.0)
        uy = torch.where(slope > 1e-8, terrain_slope_y / slope_safe, 0.0)
        vx, vy = phi_w * wind_x + phi_s * ux, phi_w * wind_y + phi_s * uy
        phi = torch.hypot(vx, vy)
        hx = torch.where(phi > 1e-8, vx / torch.clamp(phi, min=1e-8), wind_x)
        hy = torch.where(phi > 1e-8, vy / torch.clamp(phi, min=1e-8), wind_y)
        ew = torch.clamp(
            (wind_head - wind_back) / torch.clamp(wind_head + wind_back, min=1e-5),
            0.0,
            0.985,
        )
        es = torch.clamp(
            (slope_head - slope_back) / torch.clamp(slope_head + slope_back, min=1e-5),
            0.0,
            0.985,
        )
        eccentricity = torch.where(
            phi > 1e-8,
            (phi_w * ew + phi_s * es) / torch.clamp(phi, min=1e-8),
            0.0,
        )
        ros = torch.clamp(
            base_ros
            * (1.0 + phi)
            * spread_adjustment
            * config.surface_spread_adjustment,
            0.0,
            config.max_spread_rate_m_min,
        )
        intensity = torch.clamp(base_i * ros / safe_base, min=0.0)
        fire_type = torch.full_like(fuel_codes, int(FireType.SURFACE), dtype=torch.uint8)
        if config.enable_crown_fire:
            canopy = (
                (canopy_cover >= 0.20)
                & (canopy_height_m > canopy_base_height_m)
                & (canopy_base_height_m > 0.0)
                & (canopy_bulk_density_kg_m3 > 0.0)
            )
            critical_i = torch.clamp(
                0.01
                * canopy_base_height_m
                * (460.0 + 2600.0 * torch.clamp(foliar_moisture, 0.3, 2.0)),
                min=0.0,
            ).pow(1.5)
            initiated = canopy & (intensity >= critical_i)
            active_ros = (
                11.02
                * torch.clamp(wind_speed_10m_m_s.expand_as(ros) * 3.6, min=0.0).pow(0.90)
                * torch.clamp(canopy_bulk_density_kg_m3, min=1e-5).pow(0.19)
                * torch.exp(-17.0 * torch.clamp(moisture_dead_1h, 0.0, 0.6))
                * config.crown_spread_adjustment
            )
            critical_ros = 3.0 / torch.clamp(canopy_bulk_density_kg_m3, min=1e-5)
            active = initiated & (active_ros >= critical_ros)
            passive_ros = active_ros * torch.exp(
                -active_ros / torch.clamp(critical_ros, min=1e-5)
            )
            crown_ros = torch.where(active, active_ros, passive_ros)
            ros = torch.where(initiated, torch.maximum(ros, crown_ros), ros)
            canopy_fuel = (
                canopy_cover
                * canopy_bulk_density_kg_m3
                * torch.clamp(canopy_height_m - canopy_base_height_m, min=0.0)
            )
            crown_i = intensity + 18_000.0 * canopy_fuel * ros / 60.0
            intensity = torch.where(initiated, crown_i, intensity)
            fire_type = torch.where(
                active,
                int(FireType.ACTIVE_CROWN),
                torch.where(initiated, int(FireType.PASSIVE_CROWN), int(FireType.SURFACE)),
            ).to(torch.uint8)
            crown_lw = torch.clamp(
                1.0 + 0.125 * wind_speed_10m_m_s.expand_as(ros) * 3.6,
                1.0,
                100.0,
            )
            crown_eccentricity = torch.sqrt(
                torch.clamp(1.0 - 1.0 / crown_lw.square(), min=0.0)
            )
            eccentricity = torch.where(
                active,
                torch.maximum(eccentricity, crown_eccentricity),
                eccentricity,
            )
        flame = 0.0775 * torch.clamp(intensity, min=0.0).pow(0.46)
        return FireBehaviorFields(
            ros,
            intensity,
            flame,
            hx,
            hy,
            torch.clamp(eccentricity, 0.0, 0.985),
            fire_type,
        )


@lru_cache(maxsize=1)
def fire_behavior_lookup() -> FireBehaviorLookup:
    return FireBehaviorLookup()
