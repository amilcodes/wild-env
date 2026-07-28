"""Batched accelerator-resident fire propagation.

The environment simulator keeps a NumPy truth state for interoperability with
GIS and PettingZoo.  This module is the throughput path: landscapes remain in
PyTorch tensors across steps and batches, including behavior interpolation,
front propagation, crown transition, moisture response, and stochastic ember
transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Any

import torch

from aeolus.config import FireBehaviorConfig
from aeolus.core.fire_behavior import fire_behavior_lookup
from aeolus.core.state import FirePhase, FireType

_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


@dataclass
class TensorFireState:
    phase: torch.Tensor
    fuel_model_number: torch.Tensor
    elevation_m: torch.Tensor
    barrier: torch.Tensor
    moisture_dead_1h: torch.Tensor
    moisture_dead_10h: torch.Tensor
    moisture_dead_100h: torch.Tensor
    foliar_moisture: torch.Tensor
    canopy_cover: torch.Tensor
    canopy_height_m: torch.Tensor
    canopy_base_height_m: torch.Tensor
    canopy_bulk_density_kg_m3: torch.Tensor
    spread_adjustment: torch.Tensor
    ignition_progress: torch.Tensor
    fuel_remaining: torch.Tensor
    burn_age_min: torch.Tensor
    fire_type: torch.Tensor
    intensity_kw_m: torch.Tensor
    spread_rate_m_min: torch.Tensor
    flame_length_m: torch.Tensor
    arrival_time_min: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.phase.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.phase.shape)  # type: ignore[return-value]


def _shift(source: torch.Tensor, dx: int, dy: int, fill: float | bool = 0.0) -> torch.Tensor:
    output = torch.full_like(source, fill)
    height, width = source.shape[-2:]
    sy = slice(max(0, -dy), min(height, height - dy))
    ty = slice(max(0, dy), min(height, height + dy))
    sx = slice(max(0, -dx), min(width, width - dx))
    tx = slice(max(0, dx), min(width, width + dx))
    output[..., ty, tx] = source[..., sy, sx]
    return output


def _neighbor_any(mask: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(mask)
    for dx, dy in _NEIGHBORS:
        result |= _shift(mask, dx, dy, False)
    return result


def _terrain_slopes(elevation: torch.Tensor, cell_size_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    slope_x = torch.zeros_like(elevation)
    slope_y = torch.zeros_like(elevation)
    slope_x[..., 1:-1] = (
        elevation[..., 2:] - elevation[..., :-2]
    ) / (2.0 * cell_size_m)
    slope_x[..., 0] = (elevation[..., 1] - elevation[..., 0]) / cell_size_m
    slope_x[..., -1] = (elevation[..., -1] - elevation[..., -2]) / cell_size_m
    slope_y[..., 1:-1, :] = (
        elevation[..., 2:, :] - elevation[..., :-2, :]
    ) / (2.0 * cell_size_m)
    slope_y[..., 0, :] = (
        elevation[..., 1, :] - elevation[..., 0, :]
    ) / cell_size_m
    slope_y[..., -1, :] = (
        elevation[..., -1, :] - elevation[..., -2, :]
    ) / cell_size_m
    return slope_x, slope_y


def _equilibrium_moisture(
    temperature_c: torch.Tensor, relative_humidity_pct: torch.Tensor
) -> torch.Tensor:
    rh = torch.clamp(relative_humidity_pct, 0.0, 100.0)
    temperature_f = temperature_c * 1.8 + 32.0
    low = 0.03229 + 0.281073 * rh - 0.000578 * rh * temperature_f
    middle = 2.22749 + 0.160107 * rh - 0.01478 * temperature_f
    high = (
        21.0606
        + 0.005565 * rh.square()
        - 0.00035 * rh * temperature_f
        - 0.483199 * rh
    )
    return torch.clamp(
        torch.where(rh < 10.0, low, torch.where(rh <= 50.0, middle, high))
        / 100.0,
        0.01,
        0.60,
    )


class TensorFireKernel:
    """Batched raster-front kernel for CUDA, ROCm, MPS, or CPU tensors."""

    def __init__(
        self,
        *,
        cell_size_m: float,
        config: FireBehaviorConfig | None = None,
    ):
        self.cell_size_m = float(cell_size_m)
        self.config = config or FireBehaviorConfig()
        self.lookup = fire_behavior_lookup()

    @staticmethod
    def _weather_field(value: float | torch.Tensor, state: TensorFireState) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=state.moisture_dead_1h.dtype,
            device=state.moisture_dead_1h.device,
        )
        if tensor.ndim == 0:
            tensor = tensor.expand(state.batch_size, 1, 1)
        elif tensor.ndim == 1:
            tensor = tensor[:, None, None]
        return tensor

    def _update_moisture(
        self,
        state: TensorFireState,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
        precipitation: torch.Tensor,
    ) -> None:
        if self.config.moisture_model == "fixed":
            return
        equilibrium = _equilibrium_moisture(temperature, humidity)
        for field, lag in (
            (state.moisture_dead_1h, 60.0),
            (state.moisture_dead_10h, 600.0),
            (state.moisture_dead_100h, 6000.0),
        ):
            field.add_((equilibrium - field) * (1.0 - torch.exp(torch.tensor(
                -1.0 / lag, device=field.device, dtype=field.dtype
            ))))
            wetting_rate = precipitation / 60.0 / max(lag / 60.0, 1.0)
            field.add_((0.60 - field) * (1.0 - torch.exp(-wetting_rate)))
            field.clamp_(0.01, 0.60)

    def behavior(
        self,
        state: TensorFireState,
        *,
        wind_speed_m_s: float | torch.Tensor,
        wind_from_direction_deg: float | torch.Tensor,
    ):
        wind = self._weather_field(wind_speed_m_s, state)
        direction = self._weather_field(wind_from_direction_deg, state)
        slope_x, slope_y = _terrain_slopes(state.elevation_m, self.cell_size_m)
        return self.lookup.resolve_torch(
            fuel_model_number=state.fuel_model_number,
            moisture_dead_1h=state.moisture_dead_1h,
            wind_speed_10m_m_s=wind,
            wind_from_direction_deg=direction,
            terrain_slope_x=slope_x,
            terrain_slope_y=slope_y,
            canopy_cover=state.canopy_cover,
            canopy_height_m=state.canopy_height_m,
            canopy_base_height_m=state.canopy_base_height_m,
            canopy_bulk_density_kg_m3=state.canopy_bulk_density_kg_m3,
            foliar_moisture=state.foliar_moisture,
            spread_adjustment=state.spread_adjustment,
            config=self.config,
        )

    def _propagate(
        self,
        state: TensorFireState,
        behavior: Any,
        *,
        minute: float,
        dt_min: float,
    ) -> torch.Tensor:
        flaming = state.phase == int(FirePhase.FLAMING)
        unburned = (
            (state.phase == int(FirePhase.UNBURNED))
            & (~state.barrier)
        )
        incoming = torch.zeros_like(state.ignition_progress)
        for dx, dy in _NEIGHBORS:
            distance_cells = sqrt(dx * dx + dy * dy)
            cosine = (
                behavior.head_x * (dx / distance_cells)
                + behavior.head_y * (dy / distance_cells)
            )
            directional = (1.0 - behavior.eccentricity) / torch.clamp(
                1.0 - behavior.eccentricity * cosine, min=1e-4
            )
            fraction = (
                behavior.spread_rate_m_min
                * directional
                * dt_min
                / (distance_cells * self.cell_size_m)
            )
            candidate = _shift(torch.where(flaming, fraction, 0.0), dx, dy)
            incoming = torch.maximum(incoming, candidate)
        state.ignition_progress[unburned] += incoming[unburned]
        ignited = unburned & (state.ignition_progress >= 1.0)
        state.phase[ignited] = int(FirePhase.FLAMING)
        state.arrival_time_min[ignited] = minute
        state.burn_age_min[ignited] = 0.0
        state.ignition_progress[ignited] = 0.0
        state.intensity_kw_m[ignited] = behavior.fireline_intensity_kw_m[ignited]
        state.spread_rate_m_min[ignited] = behavior.spread_rate_m_min[ignited]
        state.flame_length_m[ignited] = behavior.flame_length_m[ignited]
        state.fire_type[ignited] = behavior.fire_type[ignited]
        return ignited

    def _spot(
        self,
        state: TensorFireState,
        behavior: Any,
        wind_speed: torch.Tensor,
        wind_direction: torch.Tensor,
        minute: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        settings = self.config
        source = (
            (state.phase == int(FirePhase.FLAMING))
            & (behavior.fireline_intensity_kw_m >= 350.0)
        )
        source_indices = torch.nonzero(source, as_tuple=False)
        if not settings.enable_spotting or source_indices.numel() == 0:
            return torch.zeros(state.batch_size, device=state.phase.device, dtype=torch.int64)
        expected = settings.spotting_embers_per_source_min * source_indices.shape[0]
        count = min(
            int(torch.poisson(torch.tensor(
                expected, device=state.phase.device
            ), generator=generator).item()),
            settings.max_spot_embers_per_minute,
        )
        if count == 0:
            return torch.zeros(state.batch_size, device=state.phase.device, dtype=torch.int64)
        choices = torch.randint(
            source_indices.shape[0],
            (count,),
            device=state.phase.device,
            generator=generator,
        )
        selected = source_indices[choices]
        batch, sy, sx = selected[:, 0], selected[:, 1], selected[:, 2]
        source_intensity = behavior.fireline_intensity_kw_m[batch, sy, sx]
        local_wind = wind_speed[batch, 0, 0]
        median = settings.spotting_median_distance_m * torch.clamp(
            local_wind / 6.0, min=0.1
        ).pow(settings.spotting_wind_exponent) * torch.clamp(
            source_intensity / 2000.0, min=0.1
        ).pow(settings.spotting_intensity_exponent)
        normal = torch.randn(
            count, device=state.phase.device, generator=generator
        )
        distance = torch.clamp(
            torch.exp(torch.log(torch.clamp(median, min=1.0)) + settings.spotting_log_sigma * normal),
            max=settings.spotting_max_distance_m,
        )
        cross = (
            torch.randn(count, device=state.phase.device, generator=generator)
            * settings.spotting_crosswind_fraction
            * distance
        )
        radians = torch.deg2rad(wind_direction[batch, 0, 0])
        down_x, down_y = -torch.sin(radians), torch.cos(radians)
        tx = torch.round(
            sx + (distance * down_x - cross * down_y) / self.cell_size_m
        ).long()
        ty = torch.round(
            sy + (distance * down_y + cross * down_x) / self.cell_size_m
        ).long()
        valid = (
            (tx >= 0)
            & (tx < state.phase.shape[-1])
            & (ty >= 0)
            & (ty < state.phase.shape[-2])
        )
        batch, tx, ty, distance = batch[valid], tx[valid], ty[valid], distance[valid]
        if batch.numel() == 0:
            return torch.zeros(state.batch_size, device=state.phase.device, dtype=torch.int64)
        probability = (
            settings.spotting_ignition_probability
            * torch.exp(-distance / settings.spotting_survival_distance_m)
            * torch.clamp(1.0 - state.moisture_dead_1h[batch, ty, tx] / 0.35, 0.0, 1.0)
        )
        accepted = torch.rand(
            probability.shape,
            device=state.phase.device,
            generator=generator,
        ) < probability
        batch, tx, ty = batch[accepted], tx[accepted], ty[accepted]
        fresh = state.phase[batch, ty, tx] == int(FirePhase.UNBURNED)
        batch, tx, ty = batch[fresh], tx[fresh], ty[fresh]
        state.phase[batch, ty, tx] = int(FirePhase.FLAMING)
        state.fire_type[batch, ty, tx] = int(FireType.SURFACE)
        state.burn_age_min[batch, ty, tx] = 0.0
        state.arrival_time_min[batch, ty, tx] = float(minute)
        state.intensity_kw_m[batch, ty, tx] = torch.clamp(
            behavior.fireline_intensity_kw_m[batch, ty, tx], min=60.0
        )
        state.spread_rate_m_min[batch, ty, tx] = behavior.spread_rate_m_min[
            batch, ty, tx
        ]
        state.flame_length_m[batch, ty, tx] = behavior.flame_length_m[
            batch, ty, tx
        ]
        counts = torch.zeros(state.batch_size, device=state.phase.device, dtype=torch.int64)
        counts.scatter_add_(0, batch, torch.ones_like(batch))
        return counts

    @torch.no_grad()
    def step(
        self,
        state: TensorFireState,
        *,
        minute: int,
        wind_speed_m_s: float | torch.Tensor,
        wind_from_direction_deg: float | torch.Tensor,
        air_temperature_c: float | torch.Tensor = 30.0,
        relative_humidity_pct: float | torch.Tensor = 25.0,
        precipitation_rate_mm_h: float | torch.Tensor = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Advance one minute and return new-ignition counts per batch item."""

        wind = self._weather_field(wind_speed_m_s, state)
        direction = self._weather_field(wind_from_direction_deg, state)
        self._update_moisture(
            state,
            self._weather_field(air_temperature_c, state),
            self._weather_field(relative_humidity_pct, state),
            self._weather_field(precipitation_rate_mm_h, state),
        )
        behavior = self.behavior(
            state,
            wind_speed_m_s=wind,
            wind_from_direction_deg=direction,
        )
        maximum_fraction = float(
            behavior.spread_rate_m_min.max().item() / self.cell_size_m
        )
        substeps = max(
            1,
            min(
                self.config.max_substeps,
                ceil(maximum_fraction / self.config.propagation_cfl),
            ),
        )
        counts = torch.zeros(
            state.batch_size, device=state.phase.device, dtype=torch.int64
        )
        for substep in range(substeps):
            ignited = self._propagate(
                state,
                behavior,
                minute=minute - 1.0 + (substep + 1.0) / substeps,
                dt_min=1.0 / substeps,
            )
            counts += ignited.flatten(1).sum(dim=1)
        flaming = state.phase == int(FirePhase.FLAMING)
        state.burn_age_min[flaming] += 1.0
        state.intensity_kw_m[flaming] = behavior.fireline_intensity_kw_m[flaming]
        state.spread_rate_m_min[flaming] = behavior.spread_rate_m_min[flaming]
        state.flame_length_m[flaming] = behavior.flame_length_m[flaming]
        state.fire_type[flaming] = behavior.fire_type[flaming]
        state.fuel_remaining[flaming] *= torch.exp(torch.tensor(
            -1.0 / min(max(18.0 + 0.18 * self.cell_size_m, 18.0), 90.0),
            device=state.phase.device,
            dtype=state.fuel_remaining.dtype,
        ))
        adjacent = _neighbor_any(state.phase == int(FirePhase.UNBURNED))
        burned = flaming & (
            (~adjacent)
            | (state.burn_age_min >= self.config.max_front_residence_min)
        )
        state.phase[burned] = int(FirePhase.BURNED)
        state.fire_type[burned] = int(FireType.UNBURNED)
        state.intensity_kw_m[burned] = 0.0
        state.spread_rate_m_min[burned] = 0.0
        state.flame_length_m[burned] = 0.0
        counts += self._spot(
            state, behavior, wind, direction, minute, generator
        )
        return counts


def make_synthetic_batch(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    fuel_model_number: int = 122,
    moisture_dead_1h: float = 0.07,
) -> TensorFireState:
    """Create a deterministic homogeneous batch for benchmarks and tests."""

    shape = (batch_size, height, width)

    def zeros() -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=dtype)

    phase = torch.zeros(shape, device=device, dtype=torch.uint8)
    cy, cx = height // 2, width // 2
    phase[:, cy - 1 : cy + 2, cx - 1 : cx + 2] = int(FirePhase.FLAMING)
    arrival = torch.full(shape, torch.inf, device=device, dtype=dtype)
    arrival[phase == int(FirePhase.FLAMING)] = 0.0
    return TensorFireState(
        phase=phase,
        fuel_model_number=torch.full(
            shape, fuel_model_number, device=device, dtype=torch.int16
        ),
        elevation_m=zeros(),
        barrier=torch.zeros(shape, device=device, dtype=torch.bool),
        moisture_dead_1h=torch.full(shape, moisture_dead_1h, device=device, dtype=dtype),
        moisture_dead_10h=torch.full(shape, moisture_dead_1h + 0.02, device=device, dtype=dtype),
        moisture_dead_100h=torch.full(shape, moisture_dead_1h + 0.04, device=device, dtype=dtype),
        foliar_moisture=torch.ones(shape, device=device, dtype=dtype),
        canopy_cover=zeros(),
        canopy_height_m=zeros(),
        canopy_base_height_m=zeros(),
        canopy_bulk_density_kg_m3=zeros(),
        spread_adjustment=torch.ones(shape, device=device, dtype=dtype),
        ignition_progress=zeros(),
        fuel_remaining=torch.ones(shape, device=device, dtype=dtype),
        burn_age_min=zeros(),
        fire_type=torch.where(
            phase == int(FirePhase.FLAMING),
            int(FireType.SURFACE),
            int(FireType.UNBURNED),
        ).to(torch.uint8),
        intensity_kw_m=zeros(),
        spread_rate_m_min=zeros(),
        flame_length_m=zeros(),
        arrival_time_min=arrival,
    )
