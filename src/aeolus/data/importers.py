"""Import public wildfire observations and landscape services."""

from __future__ import annotations

import json
import math
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
import numpy as np

from aeolus.data.bundle import ScenarioBundle, load_bundle, write_bundle
from aeolus.data.weather import WeatherForcing

FEDS_PERIMETER_URL = (
    "https://gis.earthdata.nasa.gov/image/rest/services/FireTracking/"
    "Fire_Events_Data_Suite_Fire_Perimeters_nrt/MapServer/0/query"
)
NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NIROPS_PROGRESSION_DOI = "https://doi.org/10.17632/95rj5d379g.1"
LANDFIRE_WCS_URL = "https://edcintl.cr.usgs.gov/geoserver/landfire/conus_2025/wcs"
USGS_3DEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)

LANDFIRE_COVERAGES = {
    "fuel_model": "landfire__LF2025_FBFM40_CONUS",
    "canopy_cover": "landfire__LF2025_CC_CONUS",
    "canopy_height": "landfire__LF2025_CH_CONUS",
    "canopy_base_height": "landfire__LF2025_CBH_CONUS",
    "canopy_bulk_density": "landfire__LF2025_CBD_CONUS",
}


def _download(url: str, params: list[tuple[str, str]], *, timeout: float = 90.0) -> bytes:
    target = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(target, headers={"User-Agent": "aeolus-ia/0.2 research importer"})
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=tls_context) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    if "xml" in content_type and body.startswith(b"<?xml"):
        raise RuntimeError(body.decode("utf-8", errors="replace")[:1000])
    return body


def _feature_timestamp(feature: dict[str, Any]) -> int:
    value = feature.get("properties", {}).get("t")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    raise ValueError("FEDS feature has no usable timestamp")


def fetch_feds_perimeters(
    region: str,
    fire_id: int,
    *,
    service_url: str = FEDS_PERIMETER_URL,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Fetch and normalize one NASA FEDS fire-perimeter time series."""

    safe_region = region.replace("'", "''")
    body = _download(
        service_url,
        [
            ("where", f"region='{safe_region}' AND fireid={int(fire_id)}"),
            ("outFields", "*"),
            ("returnGeometry", "true"),
            ("outSR", "4326"),
            ("f", "geojson"),
        ],
        timeout=timeout,
    )
    collection = json.loads(body)
    if collection.get("type") != "FeatureCollection":
        raise RuntimeError(f"FEDS response is not GeoJSON: {collection}")
    features = collection.get("features", [])
    if not features:
        raise LookupError(f"no FEDS perimeter records for {region}:{fire_id}")
    features.sort(key=_feature_timestamp)
    for feature in features:
        millis = _feature_timestamp(feature)
        properties = feature.setdefault("properties", {})
        properties["observed_at"] = (
            datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        properties["source"] = "NASA-FEDS-VIIRS"
        properties["source_crs"] = "EPSG:4326"
        properties["nominal_resolution_m"] = 375
        properties["time_semantics"] = "FEDS 12-hour local-solar-time bin"
    collection["name"] = f"NASA FEDS {region}:{fire_id}"
    collection["aeolus:source"] = {
        "url": service_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "region": region,
        "fire_id": int(fire_id),
    }
    return collection


def load_nirops_perimeters(
    shapefile_path: str | Path,
    incident_code: str,
) -> dict[str, Any]:
    """Select one time series from the Magstadt et al. NIROPS archive.

    The source is the 2026 curated release of analyst-interpreted airborne
    infrared perimeters for western U.S. incidents from 2020 through 2024.
    """

    try:
        import shapefile
        from shapely.geometry import mapping
        from shapely.geometry import shape as shapely_shape
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to read NIROPS perimeters") from exc

    source = Path(shapefile_path)
    reader = shapefile.Reader(str(source))
    features: list[dict[str, Any]] = []
    for index, record in enumerate(reader.iterRecords()):
        properties = record.as_dict()
        if str(properties.get("Incident_C")) != incident_code:
            continue

        def field(*names: str) -> Any:
            for name in names:
                if name in properties:
                    return properties[name]
            raise KeyError(f"NIROPS record is missing all field aliases {names!r}")

        observed = datetime.fromisoformat(str(properties["UTC"])).replace(
            tzinfo=timezone.utc
        )
        geometry = shapely_shape(reader.shape(index).__geo_interface__)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "observed_at": observed.isoformat().replace("+00:00", "Z"),
                    "source": "USFS-NIROPS-curated-progression",
                    "source_crs": "EPSG:4326",
                    "incident_code": str(properties["Incident_C"]),
                    "incident_name": str(field("Inc Name", "Inc_Name")),
                    "incident_number": str(field("Inc Number", "Inc_Number")),
                    "irwin_id": str(properties["IRWINID"]),
                    "reported_acres": float(properties["Acres"]),
                    "time_semantics": "airborne infrared acquisition time in UTC",
                },
            }
        )
    if len(features) < 2:
        raise LookupError(
            f"NIROPS archive has fewer than two records for {incident_code!r}"
        )
    features.sort(key=lambda feature: feature["properties"]["observed_at"])
    return {
        "type": "FeatureCollection",
        "name": f"NIROPS progression {incident_code}",
        "features": features,
        "aeolus:source": {
            "title": (
                "A high spatial resolution daily fire perimeter progression "
                "dataset for wildfires in the Western United States: 2020-2024"
            ),
            "doi": NIROPS_PROGRESSION_DOI,
            "license": "CC BY 4.0",
            "source_path": str(source.resolve()),
            "incident_code": incident_code,
            "retrieved_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    }


def fetch_nasa_power_hourly(
    latitude: float,
    longitude: float,
    start_datetime: datetime | str,
    end_datetime: datetime | str,
    *,
    timeout: float = 120.0,
) -> WeatherForcing:
    """Fetch hourly MERRA-2 meteorology through the NASA POWER API."""

    def normalize(value: datetime | str) -> datetime:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    start = normalize(start_datetime)
    end = normalize(end_datetime)
    if end <= start:
        raise ValueError("weather end time must follow start time")
    parameters = ("WS10M", "WD10M", "T2M", "RH2M", "PRECTOTCORR")
    body = _download(
        NASA_POWER_HOURLY_URL,
        [
            ("parameters", ",".join(parameters)),
            ("community", "AG"),
            ("longitude", str(float(longitude))),
            ("latitude", str(float(latitude))),
            ("start", start.strftime("%Y%m%d")),
            ("end", end.strftime("%Y%m%d")),
            ("format", "JSON"),
            ("time-standard", "UTC"),
        ],
        timeout=timeout,
    )
    payload = json.loads(body)
    values = payload.get("properties", {}).get("parameter", {})
    if not all(name in values for name in parameters):
        raise RuntimeError(f"NASA POWER response is missing parameters: {payload}")
    fill = float(payload.get("header", {}).get("fill_value", -999.0))
    keys = sorted(set.intersection(*(set(values[name]) for name in parameters)))
    timestamps: list[datetime] = []
    rows: dict[str, list[float]] = {name: [] for name in parameters}
    for key in keys:
        sample = [float(values[name][key]) for name in parameters]
        if any(value <= fill + 1e-6 for value in sample):
            continue
        timestamp = datetime.strptime(key, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        timestamps.append(timestamp)
        for name, value in zip(parameters, sample, strict=True):
            rows[name].append(value)
    if len(timestamps) < 2:
        raise RuntimeError("NASA POWER returned fewer than two complete hourly samples")
    forcing = WeatherForcing(
        minute=np.asarray(
            [(timestamp - start).total_seconds() / 60.0 for timestamp in timestamps],
            dtype=np.float64,
        ),
        wind_speed_m_s=np.asarray(rows["WS10M"], dtype=np.float32),
        wind_direction_deg=np.asarray(rows["WD10M"], dtype=np.float32),
        air_temperature_c=np.asarray(rows["T2M"], dtype=np.float32),
        relative_humidity_pct=np.asarray(rows["RH2M"], dtype=np.float32),
        precipitation_rate_mm_h=np.asarray(rows["PRECTOTCORR"], dtype=np.float32),
        metadata={
            "source": "NASA POWER hourly API; MERRA-2 and POWER",
            "history": (
                f"point extraction at latitude={latitude:.6f}, "
                f"longitude={longitude:.6f}; API "
                f"{payload.get('header', {}).get('api', {}).get('version', 'unknown')}"
            ),
            "time_standard": "UTC",
            "source_url": NASA_POWER_HOURLY_URL,
        },
    )
    forcing.validate()
    return forcing


def geojson_bbox(collection: dict[str, Any]) -> tuple[float, float, float, float]:
    coordinates: list[tuple[float, float]] = []

    def visit(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            coordinates.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for feature in collection.get("features", []):
        visit(feature.get("geometry", {}).get("coordinates", []))
    if not coordinates:
        raise ValueError("GeoJSON contains no coordinates")
    xs, ys = zip(*coordinates, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def _web_mercator(lon: float, lat: float) -> tuple[float, float]:
    radius = 6378137.0
    latitude = max(-85.05112878, min(85.05112878, lat))
    return radius * math.radians(lon), radius * math.log(
        math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0)
    )


def buffered_web_mercator_bbox(
    bbox_wgs84: tuple[float, float, float, float], buffer_m: float
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wgs84
    min_x, min_y = _web_mercator(west, south)
    max_x, max_y = _web_mercator(east, north)
    return min_x - buffer_m, min_y - buffer_m, max_x + buffer_m, max_y + buffer_m


def download_usgs_3dep(
    bbox_3857: tuple[float, float, float, float],
    size: tuple[int, int],
    destination: str | Path,
) -> Path:
    body = _download(
        USGS_3DEP_EXPORT_URL,
        [
            ("bbox", ",".join(str(value) for value in bbox_3857)),
            ("bboxSR", "3857"),
            ("imageSR", "3857"),
            ("size", f"{size[0]},{size[1]}"),
            ("format", "tiff"),
            ("pixelType", "F32"),
            ("interpolation", "RSP_BilinearInterpolation"),
            ("f", "image"),
        ],
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def download_landfire_coverage(
    coverage: str,
    bbox_3857: tuple[float, float, float, float],
    size: tuple[int, int],
    destination: str | Path,
) -> Path:
    if coverage not in LANDFIRE_COVERAGES:
        raise KeyError(f"unknown LANDFIRE coverage: {coverage}")
    min_x, min_y, max_x, max_y = bbox_3857
    body = _download(
        LANDFIRE_WCS_URL,
        [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", LANDFIRE_COVERAGES[coverage]),
            ("format", "image/tiff"),
            ("subset", f"X({min_x},{max_x})"),
            ("subset", f"Y({min_y},{max_y})"),
            ("scaleSize", f"i({size[0]}),j({size[1]})"),
        ],
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _fuel_load_from_fbfm40(codes: np.ndarray) -> np.ndarray:
    """Coarse surface-fuel load proxy by Scott/Burgan model family."""

    output = np.full(codes.shape, 0.85, dtype=np.float32)
    output[(codes >= 101) & (codes <= 109)] = 0.35
    output[(codes >= 121) & (codes <= 124)] = 0.70
    output[(codes >= 141) & (codes <= 149)] = 1.05
    output[(codes >= 161) & (codes <= 165)] = 0.90
    output[(codes >= 181) & (codes <= 189)] = 1.20
    output[(codes >= 201) & (codes <= 204)] = 1.55
    output[(codes <= 0) | ((codes >= 90) & (codes <= 99))] = 0.0
    return output


def build_landscape_from_services(
    bbox_wgs84: tuple[float, float, float, float],
    destination: str | Path,
    *,
    size: tuple[int, int] = (192, 192),
    buffer_m: float = 4500.0,
    split: str = "evaluation",
) -> tuple[ScenarioBundle, Path]:
    """Download USGS 3DEP and LANDFIRE layers and build simulator arrays."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to import landscape services") from exc

    root = Path(destination)
    source_dir = root / "source"
    bbox_3857 = buffered_web_mercator_bbox(bbox_wgs84, buffer_m)
    elevation_path = download_usgs_3dep(bbox_3857, size, source_dir / "elevation.tif")
    layer_paths = {
        name: download_landfire_coverage(name, bbox_3857, size, source_dir / f"{name}.tif")
        for name in LANDFIRE_COVERAGES
    }

    with rasterio.open(elevation_path) as source:
        elevation = source.read(1).astype(np.float32)
        profile = source.profile.copy()
        transform = tuple(source.transform)[:6]
        crs = source.crs.to_string()
        bounds = tuple(float(value) for value in source.bounds)
    with rasterio.open(layer_paths["fuel_model"]) as source:
        fuel_model = source.read(1)
    canopy_layers: dict[str, np.ndarray] = {}
    for name in (
        "canopy_cover",
        "canopy_height",
        "canopy_base_height",
        "canopy_bulk_density",
    ):
        with rasterio.open(layer_paths[name]) as source:
            canopy_layers[name] = source.read(1).astype(np.float32)
    fuel_load = _fuel_load_from_fbfm40(fuel_model)
    barrier = (fuel_model <= 0) | ((fuel_model >= 90) & (fuel_model <= 99))
    asset_value = np.zeros(elevation.shape, dtype=np.float32)

    landscape_path = root / "landscape.tif"
    profile.update(count=6, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(landscape_path, "w", **profile) as destination_raster:
        destination_raster.write(elevation, 1)
        destination_raster.set_band_description(1, "elevation_m")
        destination_raster.write(fuel_model.astype(np.float32), 2)
        destination_raster.set_band_description(2, "fbfm40_code")
        for index, name in enumerate(
            ("canopy_cover", "canopy_height", "canopy_base_height", "canopy_bulk_density"),
            start=3,
        ):
            destination_raster.write(canopy_layers[name], index)
            destination_raster.set_band_description(index, name)

    cell_size = abs(float(profile["transform"].a))
    scenario = ScenarioBundle(
        elevation_m=elevation,
        fuel_load_kg_m2=fuel_load,
        barrier=barrier.astype(np.bool_),
        asset_value=asset_value,
        fuel_model_number=fuel_model.astype(np.int16),
        # LANDFIRE native scaling: cover percent, heights decimetres, and
        # canopy bulk density hundredths of kg/m³.
        canopy_cover=np.clip(canopy_layers["canopy_cover"] / 100.0, 0.0, 1.0),
        canopy_height_m=np.maximum(canopy_layers["canopy_height"] / 10.0, 0.0),
        canopy_base_height_m=np.maximum(
            canopy_layers["canopy_base_height"] / 10.0, 0.0
        ),
        canopy_bulk_density_kg_m3=np.maximum(
            canopy_layers["canopy_bulk_density"] / 100.0, 0.0
        ),
        metadata={
            "schema_version": 2,
            "crs": crs,
            "cell_size_m": cell_size,
            "sources": [
                {"name": "USGS 3DEP", "service": USGS_3DEP_EXPORT_URL},
                {
                    "name": "LANDFIRE 2025",
                    "service": LANDFIRE_WCS_URL,
                    "layers": list(LANDFIRE_COVERAGES.values()),
                },
            ],
            "transformations": [
                "Web Mercator incident crop",
                f"service resample to {size[0]}x{size[1]}",
                "FBFM40 retained as operational fuel model and mapped to a load proxy",
                "LANDFIRE canopy layers converted from native integer scaling to SI",
            ],
            "split": split,
            "transform": transform,
            "bounds": bounds,
            "source_bbox_wgs84": bbox_wgs84,
            "fuel_model_semantics": "Scott/Burgan FBFM40 code retained in landscape.tif band 2",
        },
    )
    scenario.validate()
    return scenario, landscape_path


def enrich_scenario_from_landscape(
    scenario_path: str | Path,
    landscape_path: str | Path,
    destination: str | Path,
) -> ScenarioBundle:
    """Add retained FBFM40 and canopy bands to a legacy scalar-fuel bundle."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to enrich landscape bundles") from exc
    original = load_bundle(scenario_path)
    with rasterio.open(landscape_path) as source:
        if source.count < 6:
            raise ValueError("landscape GeoTIFF requires six Aeolus source bands")
        arrays = [source.read(index).astype(np.float32) for index in range(1, 7)]
        descriptions = tuple(source.descriptions)
    expected = (
        "elevation_m",
        "fbfm40_code",
        "canopy_cover",
        "canopy_height",
        "canopy_base_height",
        "canopy_bulk_density",
    )
    if descriptions != expected:
        raise ValueError(
            f"unexpected landscape bands: expected {expected}, received {descriptions}"
        )
    if arrays[0].shape != original.elevation_m.shape:
        raise ValueError("source GeoTIFF and simulator bundle grids do not match")
    metadata = dict(original.metadata)
    metadata["schema_version"] = 2
    transformations = list(metadata.get("transformations", []))
    transformations.append(
        "legacy scenario enriched from retained FBFM40 and LANDFIRE canopy bands"
    )
    metadata["transformations"] = transformations
    enriched = ScenarioBundle(
        elevation_m=original.elevation_m,
        fuel_load_kg_m2=original.fuel_load_kg_m2,
        barrier=original.barrier,
        asset_value=original.asset_value,
        metadata=metadata,
        fuel_model_number=arrays[1].astype(np.int16),
        canopy_cover=np.clip(arrays[2] / 100.0, 0.0, 1.0),
        canopy_height_m=np.maximum(arrays[3] / 10.0, 0.0),
        canopy_base_height_m=np.maximum(arrays[4] / 10.0, 0.0),
        canopy_bulk_density_kg_m3=np.maximum(arrays[5] / 100.0, 0.0),
    )
    write_bundle(destination, enriched)
    return enriched
