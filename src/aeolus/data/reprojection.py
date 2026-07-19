"""Reproject incident landscapes and gridded forcing to local metric grids."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np

from aeolus.data.bundle import ScenarioBundle
from aeolus.data.fuels import fuel_load_from_fbfm40
from aeolus.data.weather import WeatherForcing


def local_utm_crs(longitude_deg: float, latitude_deg: float) -> str:
    """Return the EPSG identifier for the local WGS84 UTM zone."""

    if not -80.0 <= latitude_deg <= 84.0:
        raise ValueError("UTM is only defined between 80 S and 84 N")
    zone = min(60, max(1, int((longitude_deg + 180.0) // 6.0) + 1))
    code = (32600 if latitude_deg >= 0.0 else 32700) + zone
    return f"EPSG:{code}"


def metric_grid_for_scenario(
    scenario: ScenarioBundle,
) -> tuple[str, Any, int, int]:
    """Resolve a square-pixel local UTM grid covering the source raster."""

    try:
        from affine import Affine
        from rasterio.warp import calculate_default_transform
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reproject scenarios") from exc

    metadata = scenario.metadata
    transform_values = metadata.get("transform")
    bounds = metadata.get("bounds")
    bbox = metadata.get("source_bbox_wgs84")
    if not isinstance(transform_values, (list, tuple)) or len(transform_values) != 6:
        raise ValueError("scenario metadata requires a six-value affine transform")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("scenario metadata requires projected raster bounds")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("scenario metadata requires source_bbox_wgs84")
    longitude = 0.5 * (float(bbox[0]) + float(bbox[2]))
    latitude = 0.5 * (float(bbox[1]) + float(bbox[3]))
    destination_crs = local_utm_crs(longitude, latitude)
    height, width = scenario.elevation_m.shape
    destination_transform, destination_width, destination_height = calculate_default_transform(
        str(metadata["crs"]),
        destination_crs,
        width,
        height,
        *[float(value) for value in bounds],
    )
    if not np.isclose(
        abs(float(destination_transform.a)),
        abs(float(destination_transform.e)),
        rtol=1e-6,
    ):
        raise ValueError("metric destination grid must use square pixels")
    return (
        destination_crs,
        Affine(*destination_transform[:6]),
        int(destination_width),
        int(destination_height),
    )


def _reproject_array(
    value: np.ndarray,
    *,
    source_transform: Any,
    source_crs: str,
    destination_transform: Any,
    destination_crs: str,
    destination_shape: tuple[int, int],
    categorical: bool,
) -> np.ndarray:
    try:
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reproject arrays") from exc

    source = np.asarray(value)
    if source.dtype == np.bool_:
        source = source.astype(np.uint8)
    destination = np.empty(destination_shape, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    )
    return destination


def reproject_scenario_to_metric(scenario: ScenarioBundle) -> ScenarioBundle:
    """Reproject a scenario into its local UTM zone."""

    try:
        from affine import Affine
        from rasterio.transform import array_bounds
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reproject scenarios") from exc

    scenario.validate()
    source_crs = str(scenario.metadata["crs"])
    source_transform = Affine(*[float(v) for v in scenario.metadata["transform"]])
    destination_crs, destination_transform, width, height = metric_grid_for_scenario(scenario)
    shape = (height, width)

    def project(value: np.ndarray | None, *, categorical: bool = False):
        if value is None:
            return None
        return _reproject_array(
            value,
            source_transform=source_transform,
            source_crs=source_crs,
            destination_transform=destination_transform,
            destination_crs=destination_crs,
            destination_shape=shape,
            categorical=categorical,
        )

    fuel_model_float = project(scenario.fuel_model_number, categorical=True)
    fuel_model = np.rint(fuel_model_float).astype(np.int16) if fuel_model_float is not None else None
    fuel_load = (
        fuel_load_from_fbfm40(fuel_model)
        if fuel_model is not None
        else np.maximum(project(scenario.fuel_load_kg_m2), 0.0).astype(np.float32)
    )
    valid_source_coverage = project(
        np.ones(scenario.elevation_m.shape, dtype=np.uint8),
        categorical=True,
    )
    outside_source = valid_source_coverage < 0.5
    bounds = array_bounds(height, width, destination_transform)
    transformations = list(scenario.metadata.get("transformations", []))
    transformations.append(
        "reprojected to local WGS84 UTM; categorical fields nearest-neighbour, continuous fields bilinear"
    )
    metadata = {
        **scenario.metadata,
        "crs": destination_crs,
        "cell_size_m": abs(float(destination_transform.a)),
        "transform": tuple(float(v) for v in destination_transform[:6]),
        "bounds": tuple(float(v) for v in bounds),
        "transformations": transformations,
        "distance_semantics": "local projected ground metres",
        "source_projection_before_metric_reprojection": source_crs,
    }
    result = ScenarioBundle(
        elevation_m=project(scenario.elevation_m).astype(np.float32),
        fuel_load_kg_m2=fuel_load.astype(np.float32),
        barrier=(project(scenario.barrier, categorical=True) >= 0.5) | outside_source,
        asset_value=np.maximum(project(scenario.asset_value), 0.0).astype(np.float32),
        metadata=metadata,
        fuel_model_number=fuel_model,
        canopy_cover=(
            np.clip(project(scenario.canopy_cover), 0.0, 1.0).astype(np.float32)
            if scenario.canopy_cover is not None
            else None
        ),
        canopy_height_m=(
            np.maximum(project(scenario.canopy_height_m), 0.0).astype(np.float32)
            if scenario.canopy_height_m is not None
            else None
        ),
        canopy_base_height_m=(
            np.maximum(project(scenario.canopy_base_height_m), 0.0).astype(np.float32)
            if scenario.canopy_base_height_m is not None
            else None
        ),
        canopy_bulk_density_kg_m3=(
            np.maximum(project(scenario.canopy_bulk_density_kg_m3), 0.0).astype(np.float32)
            if scenario.canopy_bulk_density_kg_m3 is not None
            else None
        ),
    )
    result.validate()
    return result


def reproject_weather_to_scenario(
    weather: WeatherForcing,
    source_scenario: ScenarioBundle,
    destination_scenario: ScenarioBundle,
) -> WeatherForcing:
    """Reproject spatial forcing; scalar/point forcing is retained unchanged."""

    try:
        from affine import Affine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reproject forcing") from exc

    weather.validate()
    if weather.wind_speed_m_s.ndim == 1:
        return replace(
            weather,
            metadata={
                **weather.metadata,
                "landscape_projection_updated": destination_scenario.metadata["crs"],
                "spatial_reprojection": "not applicable to scalar point forcing",
            },
        )
    if weather.wind_speed_m_s.shape[1:] != source_scenario.elevation_m.shape:
        raise ValueError("spatial weather and source scenario grids do not match")

    source_transform = Affine(*source_scenario.metadata["transform"])
    destination_transform = Affine(*destination_scenario.metadata["transform"])
    destination_shape = destination_scenario.elevation_m.shape

    def project_stack(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _reproject_array(
                    layer,
                    source_transform=source_transform,
                    source_crs=str(source_scenario.metadata["crs"]),
                    destination_transform=destination_transform,
                    destination_crs=str(destination_scenario.metadata["crs"]),
                    destination_shape=destination_shape,
                    categorical=False,
                )
                for layer in values
            ],
            axis=0,
        ).astype(np.float32)

    direction_rad = np.deg2rad(weather.wind_direction_deg)
    source_u = -weather.wind_speed_m_s * np.sin(direction_rad)
    source_v = -weather.wind_speed_m_s * np.cos(direction_rad)
    destination_u = project_stack(source_u)
    destination_v = project_stack(source_v)
    replacements: dict[str, Any] = {
        "wind_speed_m_s": np.hypot(destination_u, destination_v).astype(np.float32),
        "wind_direction_deg": (np.rad2deg(np.arctan2(-destination_u, -destination_v)) % 360.0).astype(
            np.float32
        ),
        "air_temperature_c": project_stack(weather.air_temperature_c),
        "relative_humidity_pct": np.clip(project_stack(weather.relative_humidity_pct), 0.0, 100.0),
    }
    excluded = {
        "minute",
        "metadata",
        "wind_speed_m_s",
        "wind_direction_deg",
        "air_temperature_c",
        "relative_humidity_pct",
    }
    for field in fields(weather):
        if field.name in excluded:
            continue
        value = getattr(weather, field.name)
        if isinstance(value, np.ndarray):
            replacements[field.name] = project_stack(value)
    replacements["metadata"] = {
        **weather.metadata,
        "spatial_reprojection": (
            f"{source_scenario.metadata['crs']} to "
            f"{destination_scenario.metadata['crs']}; wind reprojected as vectors"
        ),
    }
    result = replace(weather, **replacements)
    result.validate()
    return result
