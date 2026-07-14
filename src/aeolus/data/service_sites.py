"""Import incident service-node points into the simulator grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aeolus.config import ServiceSiteSpec
from aeolus.data.bundle import ScenarioBundle


def _world_to_grid(
    x: float,
    y: float,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[int, int]:
    a, b, c, d, e, f = transform
    determinant = a * e - b * d
    if abs(determinant) < 1e-12:
        raise ValueError("scenario affine transform is singular")
    translated_x, translated_y = x - c, y - f
    column = (e * translated_x - b * translated_y) / determinant
    row = (-d * translated_x + a * translated_y) / determinant
    return int(round(column)), int(round(row))


def load_service_sites_geojson(
    path: str | Path,
    scenario: ScenarioBundle,
    *,
    require_verified: bool = True,
) -> tuple[ServiceSiteSpec, ...]:
    """Load evaluated point sites whose CRS matches the scenario bundle.

    Polygon-to-dip-site suitability is deliberately outside this function:
    depth, obstacle clearance, approach/egress, ownership, temporary flight
    restrictions, and current access must be evaluated before a candidate is
    marked ``manually_verified``.
    """

    source = Path(path)
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("service-site GeoJSON must be a FeatureCollection")
    geojson_crs = payload.get("crs")
    if geojson_crs is not None:
        crs_name = geojson_crs.get("properties", {}).get("name")
        if crs_name is not None and str(crs_name) != str(scenario.metadata["crs"]):
            raise ValueError("service-site GeoJSON CRS does not match the scenario bundle")
    transform_value = scenario.metadata.get("transform")
    if transform_value is None or len(transform_value) != 6:
        raise ValueError("scenario bundle has no six-value affine transform")
    transform = tuple(float(value) for value in transform_value)
    height, width = scenario.elevation_m.shape
    sites: list[ServiceSiteSpec] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            raise ValueError("service-site features must be evaluated Point geometries")
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 2:
            raise ValueError("service-site point has invalid coordinates")
        properties = dict(feature.get("properties") or {})
        verified = bool(properties.get("manually_verified", False))
        if require_verified and not verified:
            raise ValueError(
                f"service site {properties.get('site_id', '<unknown>')} is not manually verified"
            )
        grid_x, grid_y = _world_to_grid(
            float(coordinates[0]),
            float(coordinates[1]),
            transform,
        )
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            raise ValueError(
                f"service site {properties.get('site_id', '<unknown>')} is outside the scenario grid"
            )
        properties["x"] = grid_x
        properties["y"] = grid_y
        properties["services"] = tuple(properties["services"])
        sites.append(ServiceSiteSpec(**properties))
    if not sites:
        raise ValueError("service-site GeoJSON contains no sites")
    return tuple(sites)
