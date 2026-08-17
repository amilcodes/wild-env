"""CF-NetCDF weather forcing used by incident hindcasts."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
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
    moisture_dead_1h: np.ndarray | None = None
    moisture_dead_10h: np.ndarray | None = None
    moisture_dead_100h: np.ndarray | None = None
    moisture_live_herbaceous: np.ndarray | None = None
    moisture_live_woody: np.ndarray | None = None
    wind_u_correction_m_s: np.ndarray | None = None
    wind_v_correction_m_s: np.ndarray | None = None

    @property
    def time_origin(self) -> datetime | None:
        """Return the absolute origin encoded by the CF time coordinate."""

        units = str(self.metadata.get("time_units", ""))
        _, separator, raw_origin = units.partition(" since ")
        if not separator:
            return None
        try:
            origin = datetime.fromisoformat(raw_origin.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if origin.tzinfo is None:
            origin = origin.replace(tzinfo=timezone.utc)
        return origin.astimezone(timezone.utc)

    def validate(self) -> None:
        count = len(self.minute)
        fields = (
            self.wind_speed_m_s,
            self.wind_direction_deg,
            self.air_temperature_c,
            self.relative_humidity_pct,
        )
        if count < 1 or any(value.ndim not in (1, 3) or value.shape[0] != count for value in fields):
            raise ValueError("weather fields must use shape (time,) or (time, y, x)")
        shapes = {value.shape for value in fields}
        if len(shapes) != 1:
            raise ValueError("weather fields must share one common shape")
        optional_fields = {
            "precipitation": self.precipitation_rate_mm_h,
            "dead 1h moisture": self.moisture_dead_1h,
            "dead 10h moisture": self.moisture_dead_10h,
            "dead 100h moisture": self.moisture_dead_100h,
            "live herbaceous moisture": self.moisture_live_herbaceous,
            "live woody moisture": self.moisture_live_woody,
            "wind u correction": self.wind_u_correction_m_s,
            "wind v correction": self.wind_v_correction_m_s,
        }
        for name, value in optional_fields.items():
            if value is not None and value.shape != fields[0].shape:
                raise ValueError(f"{name} must share the weather field shape")
            if value is not None and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
        if any(
            not np.all(np.isfinite(value))
            for value in (
                self.wind_speed_m_s,
                self.wind_direction_deg,
                self.air_temperature_c,
                self.relative_humidity_pct,
            )
        ):
            raise ValueError("weather fields must contain finite values")
        if np.any(np.diff(self.minute) <= 0):
            raise ValueError("weather time coordinate must be strictly increasing")
        if np.any(self.wind_speed_m_s < 0):
            raise ValueError("wind speed must be non-negative")
        if np.any((self.relative_humidity_pct < 0) | (self.relative_humidity_pct > 100)):
            raise ValueError("relative humidity must be within [0, 100]")
        if self.precipitation_rate_mm_h is not None and np.any(self.precipitation_rate_mm_h < 0):
            raise ValueError("precipitation rate must be non-negative")
        for field in (
            self.moisture_dead_1h,
            self.moisture_dead_10h,
            self.moisture_dead_100h,
            self.moisture_live_herbaceous,
            self.moisture_live_woody,
        ):
            if field is not None and np.any((field < 0.0) | (field > 3.0)):
                raise ValueError("fuel moisture must use kg/kg within [0, 3]")

    @staticmethod
    def _linear_sample(
        values: np.ndarray,
        lower: int,
        upper: int,
        fraction: float,
    ) -> float | np.ndarray:
        sample = values[lower] + fraction * (values[upper] - values[lower])
        return float(sample) if np.ndim(sample) == 0 else sample.astype(np.float32)

    def at_minute(self, minute: float) -> dict[str, float | np.ndarray]:
        """Interpolate a forcing already validated at its load/write boundary."""

        upper = int(np.searchsorted(self.minute, minute, side="right"))
        upper = min(max(upper, 1), len(self.minute) - 1)
        lower = upper - 1
        if minute <= self.minute[0]:
            lower = upper = 0
        elif minute >= self.minute[-1]:
            lower = upper = len(self.minute) - 1
        denominator = float(self.minute[upper] - self.minute[lower])
        fraction = 0.0 if denominator <= 0.0 else float((minute - self.minute[lower]) / denominator)
        lower_direction = np.deg2rad(self.wind_direction_deg[lower])
        upper_direction = np.deg2rad(self.wind_direction_deg[upper])
        direction = (
            np.rad2deg(
                np.arctan2(
                    (1.0 - fraction) * np.sin(lower_direction) + fraction * np.sin(upper_direction),
                    (1.0 - fraction) * np.cos(lower_direction) + fraction * np.cos(upper_direction),
                )
            )
            % 360.0
        )
        result = {
            "wind_speed_m_s": self._linear_sample(self.wind_speed_m_s, lower, upper, fraction),
            "wind_direction_deg": (
                float(direction) if np.ndim(direction) == 0 else direction.astype(np.float32)
            ),
            "air_temperature_c": self._linear_sample(self.air_temperature_c, lower, upper, fraction),
            "relative_humidity_pct": self._linear_sample(self.relative_humidity_pct, lower, upper, fraction),
        }
        result["precipitation_rate_mm_h"] = (
            self._linear_sample(self.precipitation_rate_mm_h, lower, upper, fraction)
            if self.precipitation_rate_mm_h is not None
            else 0.0
        )
        optional = {
            "moisture_dead_1h": self.moisture_dead_1h,
            "moisture_dead_10h": self.moisture_dead_10h,
            "moisture_dead_100h": self.moisture_dead_100h,
            "moisture_live_herbaceous": self.moisture_live_herbaceous,
            "moisture_live_woody": self.moisture_live_woody,
            "wind_u_correction_m_s": self.wind_u_correction_m_s,
            "wind_v_correction_m_s": self.wind_v_correction_m_s,
        }
        for name, values in optional.items():
            if values is not None:
                result[name] = self._linear_sample(
                    values,
                    lower,
                    upper,
                    fraction,
                )
        if "wind_u_correction_m_s" in result or "wind_v_correction_m_s" in result:
            base_speed = np.asarray(result["wind_speed_m_s"])
            base_direction = np.deg2rad(np.asarray(result["wind_direction_deg"]))
            u = -base_speed * np.sin(base_direction) + np.asarray(result.get("wind_u_correction_m_s", 0.0))
            v = -base_speed * np.cos(base_direction) + np.asarray(result.get("wind_v_correction_m_s", 0.0))
            corrected_speed = np.hypot(u, v)
            corrected_direction = np.rad2deg(np.arctan2(-u, -v)) % 360.0
            result["wind_speed_m_s"] = (
                float(corrected_speed) if corrected_speed.ndim == 0 else corrected_speed.astype(np.float32)
            )
            result["wind_direction_deg"] = (
                float(corrected_direction)
                if corrected_direction.ndim == 0
                else corrected_direction.astype(np.float32)
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
            serialized_metadata = _decode_attribute(getattr(dataset, "aeolus_metadata_json", "{}"))
            try:
                metadata = json.loads(serialized_metadata)
            except json.JSONDecodeError:
                metadata = {}
            forcing = cls(
                minute=np.asarray(time[:], dtype=np.float64) * factor,
                wind_speed_m_s=np.asarray(dataset.variables["wind_speed"][:], dtype=np.float32),
                wind_direction_deg=np.asarray(dataset.variables["wind_from_direction"][:], dtype=np.float32),
                air_temperature_c=np.asarray(dataset.variables["air_temperature"][:], dtype=np.float32)
                - 273.15,
                relative_humidity_pct=np.asarray(dataset.variables["relative_humidity"][:], dtype=np.float32),
                metadata={
                    **metadata,
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
                moisture_dead_1h=_optional_variable(dataset, "fuel_moisture_dead_1h"),
                moisture_dead_10h=_optional_variable(dataset, "fuel_moisture_dead_10h"),
                moisture_dead_100h=_optional_variable(dataset, "fuel_moisture_dead_100h"),
                moisture_live_herbaceous=_optional_variable(dataset, "fuel_moisture_live_herbaceous"),
                moisture_live_woody=_optional_variable(dataset, "fuel_moisture_live_woody"),
                wind_u_correction_m_s=_optional_variable(dataset, "wind_u_correction"),
                wind_v_correction_m_s=_optional_variable(dataset, "wind_v_correction"),
            )
        forcing.validate()
        return forcing


def trim_weather_forcing(
    forcing: WeatherForcing,
    *,
    start_minute: float,
    end_minute: float | None = None,
    rebase: bool = True,
) -> WeatherForcing:
    """Retain a forcing window while preserving one sample before its start.

    This is used after moisture spin-up so the initialized coupled state is
    retained without storing the full gridded atmospheric spin-up in every
    rollout worker.
    """

    forcing.validate()
    lower = max(0, int(np.searchsorted(forcing.minute, start_minute, side="right")) - 1)
    upper = (
        len(forcing.minute)
        if end_minute is None
        else min(
            len(forcing.minute),
            int(np.searchsorted(forcing.minute, end_minute, side="right")) + 1,
        )
    )
    if upper - lower < 2:
        raise ValueError("trimmed weather forcing requires at least two samples")
    offset = float(start_minute) if rebase else 0.0
    replacements: dict[str, Any] = {
        "minute": np.asarray(forcing.minute[lower:upper] - offset, dtype=np.float64),
    }
    for field in fields(forcing):
        name = field.name
        if name in {"minute", "metadata"}:
            continue
        value = getattr(forcing, name)
        if isinstance(value, np.ndarray):
            replacements[name] = np.asarray(value[lower:upper]).copy()
    origin = forcing.time_origin
    time_units = forcing.metadata.get("time_units")
    if rebase and origin is not None:
        rebased_origin = origin + timedelta(minutes=float(start_minute))
        time_units = f"minutes since {rebased_origin.isoformat().replace('+00:00', 'Z')}"
    replacements["metadata"] = {
        **forcing.metadata,
        "time_units": time_units,
        "spinup_sample_count": lower,
        "retained_forcing_sample_count": upper - lower,
        "forcing_trim_start_minute": float(start_minute),
        "forcing_trim_rebased": rebase,
    }
    result = replace(forcing, **replacements)
    result.validate()
    return result


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
        dimensions = ("time",)
        if forcing.wind_speed_m_s.ndim == 3:
            dataset.createDimension("y", forcing.wind_speed_m_s.shape[1])
            dataset.createDimension("x", forcing.wind_speed_m_s.shape[2])
            dimensions = ("time", "y", "x")
        dataset.Conventions = "CF-1.11"
        dataset.title = "Aeolus wildfire incident weather forcing"
        dataset.source = str(forcing.metadata.get("source", "scenario-assumption"))
        dataset.history = str(forcing.metadata.get("history", "created by aeolus-ia"))
        dataset.aeolus_metadata_json = json.dumps(
            forcing.metadata,
            sort_keys=True,
            default=str,
        )

        time = dataset.createVariable("time", "d", ("time",))
        time.standard_name = "time"
        time.axis = "T"
        time.calendar = "proleptic_gregorian"
        time.units = f"minutes since {start.isoformat().replace('+00:00', 'Z')}"
        time[:] = forcing.minute

        wind_speed = dataset.createVariable("wind_speed", "f", dimensions)
        wind_speed.standard_name = "wind_speed"
        wind_speed.units = "m s-1"
        wind_speed[:] = forcing.wind_speed_m_s

        wind_direction = dataset.createVariable("wind_from_direction", "f", dimensions)
        wind_direction.standard_name = "wind_from_direction"
        wind_direction.units = "degree"
        wind_direction[:] = forcing.wind_direction_deg

        temperature = dataset.createVariable("air_temperature", "f", dimensions)
        temperature.standard_name = "air_temperature"
        temperature.units = "K"
        temperature[:] = forcing.air_temperature_c + 273.15

        humidity = dataset.createVariable("relative_humidity", "f", dimensions)
        humidity.standard_name = "relative_humidity"
        humidity.units = "%"
        humidity[:] = forcing.relative_humidity_pct

        if forcing.precipitation_rate_mm_h is not None:
            precipitation = dataset.createVariable("precipitation_rate", "f", dimensions)
            precipitation.standard_name = "precipitation_flux"
            precipitation.units = "mm h-1"
            precipitation[:] = forcing.precipitation_rate_mm_h
        optional_variables = (
            (
                "fuel_moisture_dead_1h",
                forcing.moisture_dead_1h,
                "mass_fraction",
            ),
            (
                "fuel_moisture_dead_10h",
                forcing.moisture_dead_10h,
                "mass_fraction",
            ),
            (
                "fuel_moisture_dead_100h",
                forcing.moisture_dead_100h,
                "mass_fraction",
            ),
            (
                "fuel_moisture_live_herbaceous",
                forcing.moisture_live_herbaceous,
                "mass_fraction",
            ),
            (
                "fuel_moisture_live_woody",
                forcing.moisture_live_woody,
                "mass_fraction",
            ),
            ("wind_u_correction", forcing.wind_u_correction_m_s, "m s-1"),
            ("wind_v_correction", forcing.wind_v_correction_m_s, "m s-1"),
        )
        for name, values, units in optional_variables:
            if values is None:
                continue
            variable = dataset.createVariable(name, "f", dimensions)
            variable.units = units
            variable[:] = values
    return destination


def _decode_attribute(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _optional_variable(dataset: Any, name: str) -> np.ndarray | None:
    if name not in dataset.variables:
        return None
    return np.asarray(dataset.variables[name][:], dtype=np.float32)
