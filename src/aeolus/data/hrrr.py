"""Local-grid extraction from the public NOAA HRRR analysis archive."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data.weather import WeatherForcing

_GRID_STORE = "hrrrzarr/grid/HRRR_chunk_index.zarr"
_VARIABLES = {
    "u": ("10m_above_ground", "UGRD", "10m_above_ground"),
    "v": ("10m_above_ground", "VGRD", "10m_above_ground"),
    "temperature": ("2m_above_ground", "TMP", "2m_above_ground"),
    "relative_humidity": ("2m_above_ground", "RH", "2m_above_ground"),
    "precipitation_rate": ("surface", "PRATE", "surface"),
}


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def scenario_lonlat_grid(metadata: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center latitude/longitude fields for a scenario bundle."""

    try:
        from affine import Affine
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to project a scenario grid") from exc
    transform_values = metadata.get("transform")
    if not isinstance(transform_values, (list, tuple)) or len(transform_values) != 6:
        raise ValueError("scenario metadata requires a six-value affine transform")
    transform = Affine(*[float(value) for value in transform_values])
    rows, columns = np.indices(shape)
    x, y = transform * (columns + 0.5, rows + 0.5)
    transformer = Transformer.from_crs(str(metadata["crs"]), "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x, y)
    return (
        np.asarray(latitude, dtype=np.float64),
        np.asarray(longitude, dtype=np.float64),
    )


def nearest_hrrr_indices(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    *,
    coarse_stride: int = 12,
    search_radius_cells: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a compact target grid to HRRR cells without a CONUS-wide KD-tree."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for HRRR grid mapping") from exc
    if grid_latitude.shape != grid_longitude.shape:
        raise ValueError("HRRR latitude and longitude grids must share a shape")
    if target_latitude.shape != target_longitude.shape:
        raise ValueError("target latitude and longitude grids must share a shape")
    center_lat = float(np.mean(target_latitude))
    center_lon = float(np.mean(target_longitude))
    coarse_lat = grid_latitude[::coarse_stride, ::coarse_stride]
    coarse_lon = grid_longitude[::coarse_stride, ::coarse_stride]
    coarse_distance = (coarse_lat - center_lat) ** 2 + (
        (coarse_lon - center_lon) * np.cos(np.deg2rad(center_lat))
    ) ** 2
    coarse_y, coarse_x = np.unravel_index(np.argmin(coarse_distance), coarse_distance.shape)
    center_y = int(coarse_y * coarse_stride)
    center_x = int(coarse_x * coarse_stride)
    y0 = max(0, center_y - search_radius_cells)
    y1 = min(grid_latitude.shape[0], center_y + search_radius_cells + 1)
    x0 = max(0, center_x - search_radius_cells)
    x1 = min(grid_latitude.shape[1], center_x + search_radius_cells + 1)
    local_lat = grid_latitude[y0:y1, x0:x1]
    local_lon = grid_longitude[y0:y1, x0:x1]
    longitude_scale = np.cos(np.deg2rad(center_lat))
    tree = cKDTree(
        np.column_stack(
            (
                local_lat.ravel(),
                local_lon.ravel() * longitude_scale,
            )
        )
    )
    _, local_flat = tree.query(
        np.column_stack(
            (
                target_latitude.ravel(),
                target_longitude.ravel() * longitude_scale,
            )
        ),
        k=1,
    )
    local_y, local_x = np.unravel_index(local_flat, local_lat.shape)
    return (
        (local_y.reshape(target_latitude.shape) + y0).astype(np.int32),
        (local_x.reshape(target_latitude.shape) + x0).astype(np.int32),
    )


def _load_grid(cache_directory: Path, filesystem: Any) -> tuple[np.ndarray, np.ndarray]:
    cache = cache_directory / "hrrr_conus_grid_latlon.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as values:
            return values["latitude"], values["longitude"]
    try:
        import s3fs
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[met] to read the HRRR Zarr archive") from exc
    if not isinstance(filesystem, s3fs.S3FileSystem):
        raise TypeError("filesystem must be an s3fs.S3FileSystem")
    group = zarr.open(
        s3fs.S3Map(root=_GRID_STORE, s3=filesystem, check=False),
        mode="r",
    )
    latitude = np.asarray(group["latitude"][:], dtype=np.float64)
    longitude = np.asarray(group["longitude"][:], dtype=np.float64)
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, latitude=latitude, longitude=longitude)
    return latitude, longitude


def _read_variable_tile(
    filesystem: Any,
    timestamp: datetime,
    specification: tuple[str, str, str],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    try:
        import s3fs
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[met] to read the HRRR Zarr archive") from exc
    level, variable, subgroup = specification
    stamp = timestamp.strftime("%Y%m%d_%Hz")
    path = f"hrrrzarr/sfc/{timestamp:%Y%m%d}/{stamp}_anl.zarr/{level}/{variable}/{subgroup}"
    group = zarr.open(
        s3fs.S3Map(root=path, s3=filesystem, check=False),
        mode="r",
    )
    y0, y1, x0, x1 = bounds
    return np.asarray(group[variable][y0:y1, x0:x1], dtype=np.float32)


def select_hrrr_forecast_cycle(
    issue_time: datetime | str,
    *,
    required_horizon_hours: float,
    assumed_availability_lag_hours: float = 2.0,
) -> datetime:
    """Select a forecast cycle that could have been available at issue time.

    HRRR Zarr forecast arrays contain F01--F18 for hourly cycles and F01--F48
    for 00/06/12/18 UTC cycles. Long historical perimeter intervals therefore
    use the latest six-hour cycle satisfying the declared availability lag.
    """

    if not 0.0 < required_horizon_hours <= 48.0:
        raise ValueError("HRRR forecast horizon must be within (0, 48] hours")
    if assumed_availability_lag_hours < 0.0:
        raise ValueError("forecast availability lag cannot be negative")
    available_by = _utc(issue_time) - timedelta(hours=float(assumed_availability_lag_hours))
    cycle = available_by.replace(minute=0, second=0, microsecond=0)
    if required_horizon_hours > 18.0:
        cycle -= timedelta(hours=cycle.hour % 6)
    return cycle


def _read_forecast_variable_tile(
    filesystem: Any,
    cycle: datetime,
    specification: tuple[str, str, str],
    bounds: tuple[int, int, int, int],
    lead_indices: np.ndarray,
) -> np.ndarray:
    try:
        import s3fs
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[met] to read the HRRR archive") from exc
    level, variable, subgroup = specification
    stamp = cycle.strftime("%Y%m%d_%Hz")
    path = f"hrrrzarr/sfc/{cycle:%Y%m%d}/{stamp}_fcst.zarr/{level}/{variable}/{subgroup}"
    group = zarr.open(
        s3fs.S3Map(root=path, s3=filesystem, check=False),
        mode="r",
    )
    array = group[variable]
    if int(lead_indices.max(initial=-1)) >= array.shape[0]:
        raise RuntimeError(
            f"HRRR cycle {cycle.isoformat()} has {array.shape[0]} lead hours, "
            f"but lead F{int(lead_indices.max()) + 1:02d} was requested"
        )
    y0, y1, x0, x1 = bounds
    return np.asarray(array[lead_indices, y0:y1, x0:x1], dtype=np.float32)


def _contiguous_hour_blocks(timestamps: list[datetime]) -> list[list[datetime]]:
    """Partition sorted timestamps into contiguous one-hour blocks."""

    if not timestamps:
        return []
    ordered = sorted(timestamps)
    blocks = [[ordered[0]]]
    for timestamp in ordered[1:]:
        if timestamp - blocks[-1][-1] == timedelta(hours=1):
            blocks[-1].append(timestamp)
        else:
            blocks.append([timestamp])
    return blocks


def _analysis_repair_cycle_candidates(
    valid_start: datetime,
    valid_end: datetime,
    *,
    maximum_lead_hours: int,
) -> list[datetime]:
    """Return deterministic pre-valid-time extended-cycle candidates.

    Six-hour HRRR cycles are used because their archived forecast horizon is
    longer than that of hourly cycles.  The newest eligible cycle is tried
    first; every candidate precedes the missing valid-time block.
    """

    if maximum_lead_hours < 1:
        raise ValueError("maximum HRRR repair lead must be positive")
    if valid_end < valid_start:
        raise ValueError("repair valid end must not precede valid start")
    candidate = (valid_start - timedelta(hours=1)).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    candidate -= timedelta(hours=candidate.hour % 6)
    candidates: list[datetime] = []
    while candidate < valid_start:
        final_lead = int((valid_end - candidate).total_seconds() // 3600)
        if final_lead > maximum_lead_hours:
            break
        candidates.append(candidate)
        candidate -= timedelta(hours=6)
    return candidates


def fetch_hrrr_forecast(
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    issue_time: datetime | str,
    valid_end: datetime | str,
    *,
    cache_directory: str | Path,
    assumed_availability_lag_hours: float = 2.0,
) -> WeatherForcing:
    """Fetch one archived HRRR forecast available at a historical issue time.

    This differs materially from concatenating verifying analyses: every
    future atmospheric field comes from one model cycle whose assumed
    availability precedes the fire-perimeter forecast origin.
    """

    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[met] to read the HRRR archive") from exc
    start = _utc(issue_time)
    end = _utc(valid_end)
    if end <= start:
        raise ValueError("HRRR forecast valid end must follow issue time")
    horizon_hours = (end - start).total_seconds() / 3600.0
    cycle = select_hrrr_forecast_cycle(
        start,
        required_horizon_hours=horizon_hours,
        assumed_availability_lag_hours=assumed_availability_lag_hours,
    )
    first_lead = max(1, int(np.floor((start - cycle).total_seconds() / 3600.0)))
    last_lead = int(np.ceil((end - cycle).total_seconds() / 3600.0))
    if last_lead > 48:
        raise RuntimeError(f"required valid time is F{last_lead:02d}, beyond the HRRR archive limit")
    leads = np.arange(first_lead, last_lead + 1, dtype=np.int32)
    lead_indices = leads - 1

    target_lat = np.asarray(target_latitude, dtype=np.float64)
    target_lon = np.asarray(target_longitude, dtype=np.float64)
    if target_lat.ndim != 2 or target_lat.shape != target_lon.shape:
        raise ValueError("target latitude and longitude must be matching 2-D grids")
    cache_root = Path(cache_directory)
    cache_root.mkdir(parents=True, exist_ok=True)
    filesystem = s3fs.S3FileSystem(anon=True)
    grid_lat, grid_lon = _load_grid(cache_root, filesystem)
    source_y, source_x = nearest_hrrr_indices(
        grid_lat,
        grid_lon,
        target_lat,
        target_lon,
    )
    y0, y1 = int(source_y.min()), int(source_y.max()) + 1
    x0, x1 = int(source_x.min()), int(source_x.max()) + 1
    local_y, local_x = source_y - y0, source_x - x0
    values: dict[str, np.ndarray] = {}
    for name, specification in _VARIABLES.items():
        cache = cache_root / (
            f"hrrr_{cycle:%Y%m%d_%H}_forecast_"
            f"f{first_lead:02d}_f{last_lead:02d}_"
            f"y{y0}-{y1}_x{x0}-{x1}_{name}.npz"
        )
        if cache.exists():
            with np.load(cache, allow_pickle=False) as payload:
                tile = payload["tile"]
        else:
            tile = _read_forecast_variable_tile(
                filesystem,
                cycle,
                specification,
                (y0, y1, x0, x1),
                lead_indices,
            )
            np.savez_compressed(cache, tile=tile)
        values[name] = tile[:, local_y, local_x]

    u = values["u"]
    v = values["v"]
    forcing = WeatherForcing(
        minute=leads.astype(np.float64) * 60.0,
        wind_speed_m_s=np.hypot(u, v).astype(np.float32),
        wind_direction_deg=(np.rad2deg(np.arctan2(-u, -v)) % 360.0).astype(np.float32),
        air_temperature_c=(values["temperature"] - 273.15).astype(np.float32),
        relative_humidity_pct=np.clip(values["relative_humidity"], 0.0, 100.0).astype(np.float32),
        precipitation_rate_mm_h=np.maximum(
            values["precipitation_rate"] * 3600.0,
            0.0,
        ).astype(np.float32),
        metadata={
            "source": "NOAA HRRR public forecast Zarr archive",
            "history": "one archived forecast cycle sampled to scenario cell centres",
            "time_units": f"minutes since {cycle.isoformat().replace('+00:00', 'Z')}",
            "analysis_or_forecast": "forecast",
            "forecast_reference_time": cycle.isoformat().replace("+00:00", "Z"),
            "forecast_issue_time": start.isoformat().replace("+00:00", "Z"),
            "assumed_availability_lag_hours": float(assumed_availability_lag_hours),
            "forecast_lead_hours": leads.tolist(),
            "model_grid_spacing_m": 3_000.0,
            "wind_height_m": 10.0,
            "temperature_humidity_height_m": 2.0,
            "spatial_interpolation": "nearest native HRRR cell",
            "source_index_bounds": [y0, y1, x0, x1],
        },
    )
    forcing.validate()
    return forcing


def fetch_hrrr_analysis(
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    start: datetime | str,
    end: datetime | str,
    *,
    cache_directory: str | Path,
    workers: int = 8,
    minimum_coverage_fraction: float = 0.95,
    repair_incomplete_hours_with_forecast: bool = True,
    maximum_forecast_repair_lead_hours: int = 48,
) -> WeatherForcing:
    """Fetch hourly HRRR analyses and sample them to a scenario grid.

    The result contains 10-m vector wind, 2-m temperature and relative
    humidity, and surface precipitation rate.  Values are nearest-neighbor
    samples from the native approximately 3-km HRRR grid; no sub-grid wind
    structure is invented. If incomplete analysis groups would otherwise
    fail the declared coverage threshold, available analysis fields are
    retained and only absent fields are filled from a single archived HRRR
    forecast cycle that precedes each contiguous gap. Every substitution is
    recorded in the returned metadata.
    """

    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[met] to read the HRRR Zarr archive") from exc
    target_lat = np.asarray(target_latitude, dtype=np.float64)
    target_lon = np.asarray(target_longitude, dtype=np.float64)
    if target_lat.ndim != 2 or target_lat.shape != target_lon.shape:
        raise ValueError("target latitude and longitude must be matching two-dimensional grids")
    start_utc = _utc(start).replace(minute=0, second=0, microsecond=0)
    end_value = _utc(end)
    end_utc = end_value.replace(minute=0, second=0, microsecond=0)
    if end_utc < start_utc:
        raise ValueError("HRRR end time must not precede start time")
    count = int((end_utc - start_utc).total_seconds() // 3600) + 1
    timestamps = [start_utc + timedelta(hours=index) for index in range(count)]
    cache_root = Path(cache_directory)
    cache_root.mkdir(parents=True, exist_ok=True)
    filesystem = s3fs.S3FileSystem(anon=True)
    grid_lat, grid_lon = _load_grid(cache_root, filesystem)
    source_y, source_x = nearest_hrrr_indices(
        grid_lat,
        grid_lon,
        target_lat,
        target_lon,
    )
    y0, y1 = int(source_y.min()), int(source_y.max()) + 1
    x0, x1 = int(source_x.min()), int(source_x.max()) + 1
    local_y, local_x = source_y - y0, source_x - x0
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 < minimum_coverage_fraction <= 1.0:
        raise ValueError("minimum_coverage_fraction must be within (0, 1]")
    if maximum_forecast_repair_lead_hours < 1:
        raise ValueError("maximum forecast repair lead must be positive")

    def read_hour(
        timestamp: datetime,
    ) -> tuple[datetime, dict[str, np.ndarray] | None]:
        try:
            from zarr.errors import PathNotFoundError
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[met] to read the HRRR archive") from exc
        hour_cache = cache_root / f"hrrr_{timestamp:%Y%m%d_%H}_analysis_tile.npz"
        missing_marker = cache_root / f"hrrr_{timestamp:%Y%m%d_%H}_analysis_missing.txt"
        if hour_cache.exists():
            with np.load(hour_cache, allow_pickle=False) as values:
                return timestamp, {name: values[name] for name in _VARIABLES}
        if missing_marker.exists():
            return timestamp, None
        try:
            tiles = {
                name: _read_variable_tile(
                    filesystem,
                    timestamp,
                    specification,
                    (y0, y1, x0, x1),
                )
                for name, specification in _VARIABLES.items()
            }
        except PathNotFoundError:
            missing_marker.write_text(
                "No complete HRRR analysis group was available for this hour.\n",
                encoding="utf-8",
            )
            return timestamp, None
        np.savez_compressed(hour_cache, **tiles)
        return timestamp, tiles

    with ThreadPoolExecutor(max_workers=min(workers, count)) as executor:
        hourly_results = list(executor.map(read_hour, timestamps))
    raw_available = [(timestamp, tiles) for timestamp, tiles in hourly_results if tiles is not None]
    raw_coverage = len(raw_available) / count
    raw_missing = [timestamp for timestamp, tiles in hourly_results if tiles is None]
    repaired_fields: list[dict[str, Any]] = []
    resolved_by_timestamp = dict(hourly_results)

    # Preserve the established protocol for acceptable sparse archive loss.
    # Repair is activated only when incomplete analysis groups would reject
    # the incident under the frozen minimum-coverage rule.
    if repair_incomplete_hours_with_forecast and raw_available and raw_coverage < minimum_coverage_fraction:
        try:
            from zarr.errors import PathNotFoundError
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[met] to read the HRRR archive") from exc
        partial_by_timestamp: dict[datetime, dict[str, np.ndarray]] = {}
        for timestamp in raw_missing:
            partial_cache = cache_root / f"hrrr_{timestamp:%Y%m%d_%H}_analysis_partial_tile.npz"
            if partial_cache.exists():
                with np.load(partial_cache, allow_pickle=False) as payload:
                    partial = {name: payload[name] for name in payload.files}
            else:
                partial = {}
                for name, specification in _VARIABLES.items():
                    try:
                        partial[name] = _read_variable_tile(
                            filesystem,
                            timestamp,
                            specification,
                            (y0, y1, x0, x1),
                        )
                    except PathNotFoundError:
                        continue
                if partial:
                    np.savez_compressed(partial_cache, **partial)
            partial_by_timestamp[timestamp] = partial

        for name, specification in _VARIABLES.items():
            missing_for_variable = [
                timestamp for timestamp in raw_missing if name not in partial_by_timestamp[timestamp]
            ]
            for block in _contiguous_hour_blocks(missing_for_variable):
                cycles = _analysis_repair_cycle_candidates(
                    block[0],
                    block[-1],
                    maximum_lead_hours=maximum_forecast_repair_lead_hours,
                )
                repaired_tile: np.ndarray | None = None
                selected_cycle: datetime | None = None
                selected_leads: np.ndarray | None = None
                for cycle in cycles:
                    leads = np.asarray(
                        [int((timestamp - cycle).total_seconds() // 3600) for timestamp in block],
                        dtype=np.int32,
                    )
                    repair_cache = cache_root / (
                        f"hrrr_{block[0]:%Y%m%d_%H}_{block[-1]:%Y%m%d_%H}_"
                        f"repair_from_{cycle:%Y%m%d_%H}_forecast_{name}.npz"
                    )
                    try:
                        if repair_cache.exists():
                            with np.load(repair_cache, allow_pickle=False) as payload:
                                repaired_tile = payload["tile"]
                        else:
                            repaired_tile = _read_forecast_variable_tile(
                                filesystem,
                                cycle,
                                specification,
                                (y0, y1, x0, x1),
                                leads - 1,
                            )
                            np.savez_compressed(repair_cache, tile=repaired_tile)
                    except (PathNotFoundError, RuntimeError):
                        repaired_tile = None
                        continue
                    selected_cycle = cycle
                    selected_leads = leads
                    break
                if repaired_tile is None or selected_cycle is None or selected_leads is None:
                    continue
                for index, timestamp in enumerate(block):
                    partial_by_timestamp[timestamp][name] = repaired_tile[index]
                repaired_fields.append(
                    {
                        "variable": name,
                        "valid_start": block[0].isoformat().replace("+00:00", "Z"),
                        "valid_end": block[-1].isoformat().replace("+00:00", "Z"),
                        "hour_count": len(block),
                        "forecast_reference_time": selected_cycle.isoformat().replace("+00:00", "Z"),
                        "forecast_lead_hours": selected_leads.tolist(),
                    }
                )

        for timestamp, partial in partial_by_timestamp.items():
            if set(partial) == set(_VARIABLES):
                resolved_by_timestamp[timestamp] = partial

    available = [
        (timestamp, resolved_by_timestamp[timestamp])
        for timestamp in timestamps
        if resolved_by_timestamp[timestamp] is not None
    ]
    coverage = len(available) / count
    if not available or coverage < minimum_coverage_fraction:
        raise RuntimeError(
            f"HRRR weather coverage {coverage:.3f} after declared archive repair is below "
            f"required {minimum_coverage_fraction:.3f} (raw analysis coverage {raw_coverage:.3f})"
        )
    output = {name: np.empty((len(available), *target_lat.shape), dtype=np.float32) for name in _VARIABLES}
    for time_index, (_, tiles) in enumerate(available):
        assert tiles is not None
        for name, tile in tiles.items():
            output[name][time_index] = tile[local_y, local_x]

    u = output["u"]
    v = output["v"]
    forcing = WeatherForcing(
        minute=np.asarray(
            [(timestamp - start_utc).total_seconds() / 60.0 for timestamp, _ in available],
            dtype=np.float64,
        ),
        wind_speed_m_s=np.hypot(u, v).astype(np.float32),
        wind_direction_deg=(np.rad2deg(np.arctan2(-u, -v)) % 360.0).astype(np.float32),
        air_temperature_c=(output["temperature"] - 273.15).astype(np.float32),
        relative_humidity_pct=np.clip(output["relative_humidity"], 0.0, 100.0),
        precipitation_rate_mm_h=np.maximum(
            output["precipitation_rate"] * 3600.0,
            0.0,
        ).astype(np.float32),
        metadata={
            "source": (
                "NOAA HRRR public analysis Zarr archive"
                if not repaired_fields
                else "NOAA HRRR public Zarr archive: analysis with declared forecast field repair"
            ),
            "history": (
                "native HRRR cells sampled to scenario cell centers by nearest neighbor"
                if not repaired_fields
                else "available native HRRR analysis fields retained; absent fields repaired from "
                "one pre-valid-time HRRR forecast cycle per contiguous gap; native cells sampled "
                "to scenario cell centers by nearest neighbor"
            ),
            "time_units": f"minutes since {start_utc.isoformat().replace('+00:00', 'Z')}",
            "model_grid_spacing_m": 3_000.0,
            "wind_height_m": 10.0,
            "temperature_humidity_height_m": 2.0,
            "analysis_or_forecast": (
                "analysis" if not repaired_fields else "analysis_with_forecast_gap_repair"
            ),
            "spatial_interpolation": "nearest native HRRR cell",
            "source_index_bounds": [y0, y1, x0, x1],
            "requested_analysis_hour_count": count,
            "available_analysis_hour_count": len(raw_available),
            "analysis_coverage_fraction": raw_coverage,
            "available_weather_hour_count": len(available),
            "weather_coverage_fraction": coverage,
            "forecast_repaired_hour_count": sum(
                resolved_by_timestamp[timestamp] is not None for timestamp in raw_missing
            ),
            "forecast_repaired_field_count": sum(int(record["hour_count"]) for record in repaired_fields),
            "forecast_gap_repair": repaired_fields,
            "missing_analysis_hours": [
                timestamp.isoformat().replace("+00:00", "Z") for timestamp in raw_missing
            ],
            "unresolved_weather_hours": [
                timestamp.isoformat().replace("+00:00", "Z")
                for timestamp in timestamps
                if resolved_by_timestamp[timestamp] is None
            ],
        },
    )
    forcing.validate()
    return forcing


def overlay_hrrr_analysis(
    background: WeatherForcing,
    analysis: WeatherForcing,
    *,
    background_start: datetime | str,
) -> WeatherForcing:
    """Replace coincident background atmospheric fields with HRRR analyses."""

    background.validate()
    analysis.validate()
    analysis_origin = analysis.time_origin
    if analysis_origin is None:
        raise ValueError("HRRR analysis forcing requires an absolute time origin")
    background_origin = _utc(background_start)
    if background.wind_speed_m_s.shape[1:] != analysis.wind_speed_m_s.shape[1:]:
        raise ValueError("background and HRRR forcing must share a spatial grid")

    fields = {
        "wind_speed_m_s": np.asarray(background.wind_speed_m_s, dtype=np.float32).copy(),
        "wind_direction_deg": np.asarray(background.wind_direction_deg, dtype=np.float32).copy(),
        "air_temperature_c": np.asarray(background.air_temperature_c, dtype=np.float32).copy(),
        "relative_humidity_pct": np.asarray(
            background.relative_humidity_pct,
            dtype=np.float32,
        ).copy(),
        "precipitation_rate_mm_h": (
            np.zeros_like(background.wind_speed_m_s, dtype=np.float32)
            if background.precipitation_rate_mm_h is None
            else np.asarray(background.precipitation_rate_mm_h, dtype=np.float32).copy()
        ),
    }
    replaced = 0
    analysis_end = float(analysis.minute[-1])
    for index, minute in enumerate(background.minute):
        timestamp = background_origin + timedelta(minutes=float(minute))
        analysis_minute = (timestamp - analysis_origin).total_seconds() / 60.0
        if analysis_minute < 0.0 or analysis_minute > analysis_end:
            continue
        sample = analysis.at_minute(analysis_minute)
        for name in fields:
            fields[name][index] = np.asarray(sample[name], dtype=np.float32)
        replaced += 1
    history = str(background.metadata.get("history", "")).strip()
    repairs = analysis.metadata.get("forecast_gap_repair", [])
    repaired_variables = {
        str(record.get("variable"))
        for record in repairs
        if isinstance(record, dict) and record.get("variable") is not None
    }
    return replace(
        background,
        **fields,
        metadata={
            **background.metadata,
            "source": (f"{background.metadata.get('source', 'background')}; NOAA HRRR analysis overlay"),
            "history": (f"{history}; HRRR analysis replaced {replaced} coincident background samples").strip(
                "; "
            ),
            "incident_wind_source": "NOAA HRRR 10-m analysis",
            "incident_thermodynamic_source": "NOAA HRRR 2-m analysis",
            "incident_precipitation_source": (
                "NOAA HRRR surface analysis with declared archived-forecast gap repair"
                if "precipitation_rate" in repaired_variables
                else "NOAA HRRR surface analysis"
            ),
            "hrrr_overlay_sample_count": replaced,
            "hrrr_direct_analysis_hour_count": analysis.metadata.get("available_analysis_hour_count"),
            "hrrr_requested_analysis_hour_count": analysis.metadata.get("requested_analysis_hour_count"),
            "hrrr_analysis_coverage_fraction": analysis.metadata.get("analysis_coverage_fraction"),
            "hrrr_weather_coverage_fraction": analysis.metadata.get("weather_coverage_fraction"),
            "hrrr_missing_analysis_hours": analysis.metadata.get(
                "missing_analysis_hours",
                [],
            ),
            "hrrr_unresolved_weather_hours": analysis.metadata.get(
                "unresolved_weather_hours",
                [],
            ),
            "hrrr_forecast_gap_repair": repairs,
            "hrrr_overlay_temporal_interpolation": True,
        },
    )
