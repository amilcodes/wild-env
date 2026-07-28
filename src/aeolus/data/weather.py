"""CF-NetCDF weather forcing used by incident hindcasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeatherForcing:
    minute: np.ndarray
    wind_speed_m_s: np.ndarray
    wind_direction_deg: np.ndarray
    air_temperature_c: np.ndarray
    relative_humidity_pct: np.ndarray
    metadata: dict[str, Any]
    precipitation_rate_mm_h: np.ndarray | None = None

    def validate(self) -> None:
        count = len(self.minute)
        if count < 1 or any(
            len(value) != count
            for value in (
                self.wind_speed_m_s,
                self.wind_direction_deg,
                self.air_temperature_c,
                self.relative_humidity_pct,
            )
        ):
            raise ValueError("weather arrays must have one common non-empty time dimension")
        if self.precipitation_rate_mm_h is not None and len(
            self.precipitation_rate_mm_h
        ) != count:
            raise ValueError("precipitation must share the weather time dimension")
        if np.any(np.diff(self.minute) <= 0):
            raise ValueError("weather time coordinate must be strictly increasing")
        if np.any(self.wind_speed_m_s < 0):
            raise ValueError("wind speed must be non-negative")
        if np.any((self.relative_humidity_pct < 0) | (self.relative_humidity_pct > 100)):
            raise ValueError("relative humidity must be within [0, 100]")
        if self.precipitation_rate_mm_h is not None and np.any(
            self.precipitation_rate_mm_h < 0
        ):
            raise ValueError("precipitation rate must be non-negative")

    def at_minute(self, minute: float) -> dict[str, float]:
        self.validate()
        result = {
            "wind_speed_m_s": float(np.interp(minute, self.minute, self.wind_speed_m_s)),
            "wind_direction_deg": float(
                np.interp(minute, self.minute, np.unwrap(np.deg2rad(self.wind_direction_deg)))
                * 180.0
                / np.pi
                % 360.0
            ),
            "air_temperature_c": float(np.interp(minute, self.minute, self.air_temperature_c)),
            "relative_humidity_pct": float(
                np.interp(minute, self.minute, self.relative_humidity_pct)
            ),
        }
        result["precipitation_rate_mm_h"] = (
            float(np.interp(minute, self.minute, self.precipitation_rate_mm_h))
            if self.precipitation_rate_mm_h is not None
            else 0.0
        )
        return result

    @classmethod
    def load(cls, path: str | Path) -> WeatherForcing:
        try:
            from scipy.io import netcdf_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] to load weather forcing") from exc
        with netcdf_file(path, "r", mmap=False) as dataset:
            required = {
                "time",
                "wind_speed",
                "wind_from_direction",
                "air_temperature",
                "relative_humidity",
            }
            missing = required.difference(dataset.variables)
            if missing:
                raise ValueError(f"weather NetCDF is missing variables: {sorted(missing)}")
            time = dataset.variables["time"]
            raw_unit = time.units
            unit = raw_unit.decode() if isinstance(raw_unit, bytes) else str(raw_unit)
            factor = 1.0
            if unit.startswith("seconds since"):
                factor = 1.0 / 60.0
            elif unit.startswith("hours since"):
                factor = 60.0
            elif not unit.startswith("minutes since"):
                raise ValueError(f"unsupported weather time unit: {unit}")
            forcing = cls(
                minute=np.asarray(time[:], dtype=np.float64) * factor,
                wind_speed_m_s=np.asarray(dataset.variables["wind_speed"][:], dtype=np.float32),
                wind_direction_deg=np.asarray(
                    dataset.variables["wind_from_direction"][:], dtype=np.float32
                ),
                air_temperature_c=np.asarray(
                    dataset.variables["air_temperature"][:], dtype=np.float32
                )
                - 273.15,
                relative_humidity_pct=np.asarray(
                    dataset.variables["relative_humidity"][:], dtype=np.float32
                ),
                metadata={
                    "source": _decode_attribute(getattr(dataset, "source", "unknown")),
                    "history": _decode_attribute(getattr(dataset, "history", "")),
                    "time_units": unit,
                },
                precipitation_rate_mm_h=(
                    np.asarray(
                        dataset.variables["precipitation_rate"][:],
                        dtype=np.float32,
                    )
                    if "precipitation_rate" in dataset.variables
                    else None
                ),
            )
        forcing.validate()
        return forcing


def write_weather_forcing(
    path: str | Path,
    forcing: WeatherForcing,
    *,
    start_datetime: datetime | str,
) -> Path:
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to write weather forcing") from exc
    forcing.validate()
    if isinstance(start_datetime, str):
        start = datetime.fromisoformat(start_datetime.replace("Z", "+00:00"))
    else:
        start = start_datetime
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(destination, "w", version=2) as dataset:
        dataset.createDimension("time", len(forcing.minute))
        dataset.Conventions = "CF-1.11"
        dataset.title = "Aeolus wildfire incident weather forcing"
        dataset.source = str(forcing.metadata.get("source", "scenario-assumption"))
        dataset.history = str(forcing.metadata.get("history", "created by aeolus-ia"))

        time = dataset.createVariable("time", "d", ("time",))
        time.standard_name = "time"
        time.axis = "T"
        time.calendar = "proleptic_gregorian"
        time.units = f"minutes since {start.isoformat().replace('+00:00', 'Z')}"
        time[:] = forcing.minute

        wind_speed = dataset.createVariable("wind_speed", "f", ("time",))
        wind_speed.standard_name = "wind_speed"
        wind_speed.units = "m s-1"
        wind_speed[:] = forcing.wind_speed_m_s

        wind_direction = dataset.createVariable("wind_from_direction", "f", ("time",))
        wind_direction.standard_name = "wind_from_direction"
        wind_direction.units = "degree"
        wind_direction[:] = forcing.wind_direction_deg

        temperature = dataset.createVariable("air_temperature", "f", ("time",))
        temperature.standard_name = "air_temperature"
        temperature.units = "K"
        temperature[:] = forcing.air_temperature_c + 273.15

        humidity = dataset.createVariable("relative_humidity", "f", ("time",))
        humidity.standard_name = "relative_humidity"
        humidity.units = "%"
        humidity[:] = forcing.relative_humidity_pct

        if forcing.precipitation_rate_mm_h is not None:
            precipitation = dataset.createVariable(
                "precipitation_rate", "f", ("time",)
            )
            precipitation.standard_name = "precipitation_flux"
            precipitation.units = "mm h-1"
            precipitation[:] = forcing.precipitation_rate_mm_h
    return destination


def _decode_attribute(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
