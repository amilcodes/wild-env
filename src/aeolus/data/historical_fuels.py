"""Time-admissible LANDFIRE selection and landscape reconstruction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data.bundle import ScenarioBundle
from aeolus.data.fuels import fuel_load_from_fbfm40
from aeolus.data.importers import _download

LANDFIRE_COMPARISON_URL = "https://landfire.gov/data/comparison-table"
LANDFIRE_ALERTS_URL = "https://www.landfire.gov/data/alerts"
LANDFIRE_SERVICE_ROOT = "https://lfps.usgs.gov/arcgis/rest/services"
HISTORICAL_LAYER_NAMES = (
    "fuel_model",
    "canopy_cover",
    "canopy_height",
    "canopy_base_height",
    "canopy_bulk_density",
)


@dataclass(frozen=True)
class LandfireHistoricalVersion:
    """A LANDFIRE vintage and the time state represented by its inputs."""

    version_id: str
    display_name: str
    version_year: int
    disturbance_through_year: int
    effective_condition_year: int
    completion_year: int
    access_status: str
    streamable: bool
    layer_services: dict[str, str]
    evidence_urls: tuple[str, ...]

    def validate(self) -> None:
        if not self.version_id or not self.display_name:
            raise ValueError("LANDFIRE version requires identity fields")
        if self.disturbance_through_year > self.effective_condition_year:
            raise ValueError("disturbance cutoff follows effective condition")
        if self.streamable and set(self.layer_services) != set(HISTORICAL_LAYER_NAMES):
            raise ValueError("streamable LANDFIRE version lacks required layers")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LANDFIRE_HISTORICAL_VERSIONS = (
    LandfireHistoricalVersion(
        version_id="lf2016-remap",
        display_name="LANDFIRE 2016 Remap",
        version_year=2016,
        disturbance_through_year=2016,
        effective_condition_year=2019,
        completion_year=2021,
        access_status="active_image_service",
        streamable=True,
        layer_services={
            name: (f"{LANDFIRE_SERVICE_ROOT}/Landfire_LF2016/LF2016_{code}_CONUS/ImageServer/exportImage")
            for name, code in {
                "fuel_model": "FBFM40",
                "canopy_cover": "CC",
                "canopy_height": "CH",
                "canopy_base_height": "CBH",
                "canopy_bulk_density": "CBD",
            }.items()
        },
        evidence_urls=(LANDFIRE_COMPARISON_URL,),
    ),
    LandfireHistoricalVersion(
        version_id="lf2019l",
        display_name="LANDFIRE 2019 Limited",
        version_year=2019,
        disturbance_through_year=2019,
        effective_condition_year=2021,
        completion_year=2021,
        access_status="not_in_current_image_service",
        streamable=False,
        layer_services={},
        evidence_urls=(LANDFIRE_COMPARISON_URL,),
    ),
    LandfireHistoricalVersion(
        version_id="lf2020",
        display_name="LANDFIRE 2020",
        version_year=2020,
        disturbance_through_year=2020,
        effective_condition_year=2022,
        completion_year=2023,
        access_status="retired_request_or_archive",
        streamable=False,
        layer_services={},
        evidence_urls=(LANDFIRE_COMPARISON_URL, LANDFIRE_ALERTS_URL),
    ),
)
for _version in LANDFIRE_HISTORICAL_VERSIONS:
    _version.validate()


def _incident_year(value: datetime | str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return parsed.year


def select_historical_landfire_version(
    incident_start: datetime | str,
    *,
    require_streamable: bool,
) -> LandfireHistoricalVersion:
    """Select the most recent fuel state without incident-or-later disturbance."""

    year = _incident_year(incident_start)
    candidates = [
        version
        for version in LANDFIRE_HISTORICAL_VERSIONS
        if version.disturbance_through_year < year
        and version.effective_condition_year <= year
        and (version.streamable or not require_streamable)
    ]
    if not candidates:
        raise LookupError(f"no time-admissible LANDFIRE version for incident year {year}")
    return max(
        candidates,
        key=lambda version: (
            version.disturbance_through_year,
            version.effective_condition_year,
            version.version_year,
        ),
    )


def download_historical_landfire_layer(
    service_url: str,
    *,
    bounds: tuple[float, float, float, float],
    crs: str,
    size: tuple[int, int],
    destination: str | Path,
    categorical: bool,
    timeout: float = 120.0,
) -> Path:
    """Export one aligned layer from a LANDFIRE ArcGIS image service."""

    try:
        from rasterio.crs import CRS
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to resolve landscape CRS") from exc
    epsg = CRS.from_user_input(crs).to_epsg()
    if epsg is None:
        raise ValueError(f"historical landscape CRS requires an EPSG code: {crs}")

    body = _download(
        service_url,
        [
            ("bbox", ",".join(str(value) for value in bounds)),
            ("bboxSR", str(epsg)),
            ("imageSR", str(epsg)),
            ("size", f"{size[0]},{size[1]}"),
            ("format", "tiff"),
            ("pixelType", "S16"),
            (
                "interpolation",
                ("RSP_NearestNeighbor" if categorical else "RSP_BilinearInterpolation"),
            ),
            ("f", "image"),
        ],
        timeout=timeout,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_aligned_layer(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_crs: str,
    expected_transform: tuple[float, ...],
) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reconstruct historical fuels") from exc

    with rasterio.open(path) as source:
        if source.shape != expected_shape:
            raise ValueError(
                f"historical layer {path.name} has shape {source.shape}; expected {expected_shape}"
            )
        if source.crs is None or source.crs.to_string() != expected_crs:
            raise ValueError(f"historical layer {path.name} CRS is not aligned")
        if not np.allclose(
            tuple(source.transform)[:6],
            expected_transform,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"historical layer {path.name} transform is not aligned")
        values = source.read(1, masked=True)
    return np.asarray(values.filled(0), dtype=np.float32)


def reconstruct_historical_landscape(
    original: ScenarioBundle,
    source_landscape_path: str | Path,
    *,
    version: LandfireHistoricalVersion,
    provenance_directory: str | Path,
    output_landscape_path: str | Path,
    downloader: Callable[..., Path] = download_historical_landfire_layer,
) -> tuple[ScenarioBundle, dict[str, Any]]:
    """Replace future-vintage fuels while retaining the original terrain grid."""

    if not version.streamable:
        raise ValueError(f"LANDFIRE version {version.version_id} is not directly streamable")
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to reconstruct historical fuels") from exc

    provenance = Path(provenance_directory)
    provenance.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_landscape_path) as source:
        expected_descriptions = (
            "elevation_m",
            "fbfm40_code",
            "canopy_cover",
            "canopy_height",
            "canopy_base_height",
            "canopy_bulk_density",
        )
        if source.count < 6 or tuple(source.descriptions) != expected_descriptions:
            raise ValueError("source landscape lacks the six canonical bands")
        if source.crs is None:
            raise ValueError("source landscape has no CRS")
        profile = source.profile.copy()
        bounds = tuple(float(value) for value in source.bounds)
        shape = source.shape
        crs = source.crs.to_string()
        transform = tuple(source.transform)[:6]
    if shape != original.elevation_m.shape:
        raise ValueError("source landscape and scenario grids do not match")

    raw_paths: dict[str, Path] = {}
    for name in HISTORICAL_LAYER_NAMES:
        raw_paths[name] = downloader(
            version.layer_services[name],
            bounds=bounds,
            crs=crs,
            size=(shape[1], shape[0]),
            destination=provenance / f"{version.version_id}_{name}.tif",
            categorical=name == "fuel_model",
        )
    raw = {
        name: _read_aligned_layer(
            path,
            expected_shape=shape,
            expected_crs=crs,
            expected_transform=transform,
        )
        for name, path in raw_paths.items()
    }

    fuel_model = np.rint(raw["fuel_model"]).astype(np.int16)
    fuel_load = fuel_load_from_fbfm40(fuel_model)
    barrier = fuel_load <= 0.0
    canopy_cover = np.clip(raw["canopy_cover"] / 100.0, 0.0, 1.0)
    canopy_height = np.maximum(raw["canopy_height"] / 10.0, 0.0)
    canopy_base_height = np.maximum(raw["canopy_base_height"] / 10.0, 0.0)
    canopy_bulk_density = np.maximum(raw["canopy_bulk_density"] / 100.0, 0.0)

    source_record = {
        "name": version.display_name,
        "product_year": version.version_year,
        "version_year": version.version_year,
        "disturbance_through_year": version.disturbance_through_year,
        "effective_condition_year": version.effective_condition_year,
        "completion_year": version.completion_year,
        "data_cutoff": (f"{version.disturbance_through_year}-12-31T23:59:59Z"),
        "access_status": version.access_status,
        "services": version.layer_services,
        "evidence_urls": list(version.evidence_urls),
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sha256": {name: _sha256(path) for name, path in raw_paths.items()},
        "historical_use_note": (
            "The represented disturbance state predates the incident. "
            "Completion year records publication timing and is not treated "
            "as a vegetation-state cutoff."
        ),
    }
    metadata = dict(original.metadata)
    metadata["schema_version"] = 2
    metadata["sources"] = [
        dict(source)
        for source in original.metadata.get("sources", [])
        if not any(
            token in str(source.get("name", "")).lower() for token in ("landfire", "fuel", "vegetation")
        )
    ] + [source_record]
    metadata["transformations"] = [
        *list(original.metadata.get("transformations", [])),
        (
            f"future-vintage fuel and canopy bands replaced with "
            f"{version.display_name} on the unchanged terrain grid"
        ),
    ]

    rebuilt = ScenarioBundle(
        elevation_m=original.elevation_m.copy(),
        fuel_load_kg_m2=fuel_load,
        barrier=barrier.astype(np.bool_),
        asset_value=original.asset_value.copy(),
        metadata=metadata,
        fuel_model_number=fuel_model,
        canopy_cover=canopy_cover.astype(np.float32),
        canopy_height_m=canopy_height.astype(np.float32),
        canopy_base_height_m=canopy_base_height.astype(np.float32),
        canopy_bulk_density_kg_m3=canopy_bulk_density.astype(np.float32),
    )
    rebuilt.validate()

    destination = Path(output_landscape_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=6, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(destination, "w", **profile) as output:
        arrays = (
            rebuilt.elevation_m,
            rebuilt.fuel_model_number,
            raw["canopy_cover"],
            raw["canopy_height"],
            raw["canopy_base_height"],
            raw["canopy_bulk_density"],
        )
        descriptions = (
            "elevation_m",
            "fbfm40_code",
            "canopy_cover",
            "canopy_height",
            "canopy_base_height",
            "canopy_bulk_density",
        )
        for index, (array, description) in enumerate(
            zip(arrays, descriptions, strict=True),
            start=1,
        ):
            output.write(np.asarray(array, dtype=np.float32), index)
            output.set_band_description(index, description)

    old_fuel_model = original.fuel_model_number
    comparison = (
        np.asarray(old_fuel_model, dtype=np.int16)
        if old_fuel_model is not None
        else np.zeros(shape, dtype=np.int16)
    )
    stats = {
        "version": version.as_dict(),
        "cell_count": int(fuel_model.size),
        "fuel_model_changed_cells": int((fuel_model != comparison).sum()),
        "fuel_model_changed_fraction": float(np.mean(fuel_model != comparison)),
        "burnability_changed_cells": int((rebuilt.barrier != original.barrier).sum()),
        "burnability_changed_fraction": float(np.mean(rebuilt.barrier != original.barrier)),
        "old_fuel_model_codes": {
            str(code): int(count)
            for code, count in zip(
                *np.unique(comparison, return_counts=True),
                strict=True,
            )
        },
        "historical_fuel_model_codes": {
            str(code): int(count)
            for code, count in zip(
                *np.unique(fuel_model, return_counts=True),
                strict=True,
            )
        },
        "output_landscape_sha256": _sha256(destination),
        "source_layer_sha256": source_record["sha256"],
    }
    return rebuilt, stats
