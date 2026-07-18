"""Conservative terrain downscaling for point or coarse atmospheric forcing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aeolus.data.live_fuel import saturation_vapor_pressure_pa
from aeolus.data.weather import WeatherForcing


def _broadcast_field(values: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if values is None:
        return None
    if values.ndim == 3:
        if values.shape[1:] != shape:
            raise ValueError("gridded forcing and elevation must share a spatial shape")
        return np.asarray(values, dtype=np.float32).copy()
    return np.broadcast_to(values[:, None, None], (len(values), *shape)).astype(np.float32).copy()


def downscale_weather_to_topography(
    forcing: WeatherForcing,
    elevation_m: np.ndarray,
    *,
    lapse_rate_c_m: float = -0.0065,
    reference_elevation_m: float | None = None,
) -> WeatherForcing:
    """Project forcing to terrain with a lapse rate and conserved vapor pressure.

    This is an explicit thermodynamic microclimate correction.  It does not
    infer terrain wind, cold-air pooling, cloud, or precipitation orographic
    effects.  Incident-scale wind must come from a mesoscale analysis or
    observations.
    """

    forcing.validate()
    elevation = np.asarray(elevation_m, dtype=np.float32)
    if elevation.ndim != 2 or not np.all(np.isfinite(elevation)):
        raise ValueError("elevation_m must be a finite two-dimensional field")
    reference = float(np.median(elevation)) if reference_elevation_m is None else float(reference_elevation_m)
    temperature_background = _broadcast_field(forcing.air_temperature_c, elevation.shape)
    humidity_background = _broadcast_field(forcing.relative_humidity_pct, elevation.shape)
    assert temperature_background is not None
    assert humidity_background is not None
    temperature = temperature_background + lapse_rate_c_m * (elevation[None] - reference)
    vapor_pressure = humidity_background / 100.0 * saturation_vapor_pressure_pa(temperature_background)
    humidity = np.clip(
        100.0 * vapor_pressure / saturation_vapor_pressure_pa(temperature),
        0.0,
        100.0,
    ).astype(np.float32)

    history = str(forcing.metadata.get("history", "")).strip()
    derivation = (
        "forcing projected to scenario topography with temperature lapse rate "
        "and conserved water-vapor pressure"
    )
    result = replace(
        forcing,
        wind_speed_m_s=_broadcast_field(forcing.wind_speed_m_s, elevation.shape),
        wind_direction_deg=_broadcast_field(forcing.wind_direction_deg, elevation.shape),
        air_temperature_c=temperature.astype(np.float32),
        relative_humidity_pct=humidity,
        precipitation_rate_mm_h=_broadcast_field(
            forcing.precipitation_rate_mm_h,
            elevation.shape,
        ),
        moisture_dead_1h=_broadcast_field(forcing.moisture_dead_1h, elevation.shape),
        moisture_dead_10h=_broadcast_field(forcing.moisture_dead_10h, elevation.shape),
        moisture_dead_100h=_broadcast_field(forcing.moisture_dead_100h, elevation.shape),
        moisture_live_herbaceous=_broadcast_field(
            forcing.moisture_live_herbaceous,
            elevation.shape,
        ),
        moisture_live_woody=_broadcast_field(forcing.moisture_live_woody, elevation.shape),
        wind_u_correction_m_s=_broadcast_field(
            forcing.wind_u_correction_m_s,
            elevation.shape,
        ),
        wind_v_correction_m_s=_broadcast_field(
            forcing.wind_v_correction_m_s,
            elevation.shape,
        ),
        metadata={
            **forcing.metadata,
            "history": f"{history}; {derivation}".strip("; "),
            "microclimate_model": "elevation-lapse-rate-conserved-vapor-pressure",
            "microclimate_lapse_rate_c_m": float(lapse_rate_c_m),
            "microclimate_reference_elevation_m": reference,
            "microclimate_wind_treatment": "unchanged from atmospheric forcing",
        },
    )
    result.validate()
    return result
