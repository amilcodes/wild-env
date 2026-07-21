"""Rasterize time-indexed progression observations onto an incident grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data.incident import IncidentBundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError(f"progression timestamp has no UTC offset: {value}")
    return parsed


def _read_collection(path: Path) -> dict[str, Any]:
    collection = json.loads(path.read_text(encoding="utf-8"))
    if collection.get("type") != "FeatureCollection":
        raise ValueError(f"progression asset is not a FeatureCollection: {path}")
    return collection


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _rasterize_geometry(
    geometry: dict[str, Any],
    *,
    transformer: Any,
    affine: Any,
    shape: tuple[int, int],
    all_touched: bool,
) -> np.ndarray:
    import rasterio.features
    from shapely.geometry import shape as shape_geometry
    from shapely.ops import transform as transform_geometry

    projected = transform_geometry(transformer.transform, shape_geometry(geometry))
    return rasterio.features.rasterize(
        [(projected, 1)],
        out_shape=shape,
        transform=affine,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    ).astype(np.bool_)


def rasterize_progression_observation_bundle(
    incident_root: str | Path,
    observation_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create a compressed observation cube aligned to an IncidentBundle.

    Polygon occupancy uses cell-centre rasterization.  Concurrent active lines
    use every intersected cell, and dormant source lines are retained in a raw
    array while being zeroed in the effective active-line array.
    """

    try:
        from affine import Affine
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to rasterize progression observations") from exc

    incident = IncidentBundle.load(incident_root)
    observation = Path(observation_root)
    paths = {
        "observation_manifest": observation / "observation_manifest.json",
        "perimeters": observation / "perimeters.geojson",
        "concurrent_active_lines": observation / "concurrent_active_lines.geojson",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"progression observation assets are missing: {missing}")
    source_manifest = json.loads(paths["observation_manifest"].read_text(encoding="utf-8"))
    source_crs = source_manifest["observation_semantics"]["source_crs"]
    perimeter_collection = _read_collection(paths["perimeters"])
    line_collection = _read_collection(paths["concurrent_active_lines"])
    perimeters = sorted(
        perimeter_collection["features"],
        key=lambda feature: feature["properties"]["acquisition_end"],
    )
    if len(perimeters) < 2:
        raise ValueError("progression rasterization requires at least two perimeter frames")
    acquisition_end = [feature["properties"]["acquisition_end"] for feature in perimeters]
    if len(acquisition_end) != len(set(acquisition_end)):
        raise ValueError("progression perimeter acquisition ends must be unique")
    parsed_times = [_parse_utc(value) for value in acquisition_end]
    if any(current <= previous for previous, current in zip(parsed_times, parsed_times[1:], strict=False)):
        raise ValueError("progression perimeter times must increase strictly")

    landscape = incident.scenario_bundle()
    grid_shape = landscape.elevation_m.shape
    transform_values = landscape.metadata.get("transform")
    if not isinstance(transform_values, (list, tuple)) or len(transform_values) != 6:
        raise ValueError("incident landscape requires a six-value affine transform")
    affine = Affine(*[float(value) for value in transform_values])
    target_crs = str(landscape.metadata["crs"])
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    perimeter_mask = np.stack(
        [
            _rasterize_geometry(
                feature["geometry"],
                transformer=transformer,
                affine=affine,
                shape=grid_shape,
                all_touched=False,
            )
            for feature in perimeters
        ]
    )
    lines_by_time = {
        feature["properties"]["acquisition_end"]: feature
        for feature in line_collection["features"]
    }
    if len(lines_by_time) != len(line_collection["features"]):
        raise ValueError("concurrent active-line acquisition ends must be unique")
    active_line_raw = np.zeros_like(perimeter_mask)
    source_active_state = np.full(len(perimeters), np.nan, dtype=np.float32)
    missing_line_times: list[str] = []
    for index, acquisition_time in enumerate(acquisition_end):
        feature = lines_by_time.get(acquisition_time)
        if feature is None:
            missing_line_times.append(acquisition_time)
            continue
        active_line_raw[index] = _rasterize_geometry(
            feature["geometry"],
            transformer=transformer,
            affine=affine,
            shape=grid_shape,
            all_touched=True,
        )
        source_active_state[index] = float(feature["properties"]["source_active_state"])
    active_line_mask = active_line_raw & (source_active_state[:, None, None] > 0.0)

    first_observed_frame = np.full(grid_shape, -1, dtype=np.int32)
    for index, mask in enumerate(perimeter_mask):
        first_observed_frame[(first_observed_frame < 0) & mask] = index
    lost_cells = np.sum(perimeter_mask[:-1] & ~perimeter_mask[1:], axis=(1, 2))
    cell_area_km2 = abs(affine.a * affine.e - affine.b * affine.d) / 1_000_000.0
    rasterized_area_km2 = perimeter_mask.sum(axis=(1, 2)).astype(np.float64) * cell_area_km2
    reported_area_km2 = np.asarray(
        [feature["properties"]["reported_cumulative_area_km2"] for feature in perimeters],
        dtype=np.float64,
    )
    spatial_coverage_fraction = np.divide(
        rasterized_area_km2,
        reported_area_km2,
        out=np.full_like(rasterized_area_km2, np.nan),
        where=reported_area_km2 > 0.0,
    )
    elapsed_minute = np.asarray(
        [(value - parsed_times[0]).total_seconds() / 60.0 for value in parsed_times],
        dtype=np.float64,
    )

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("wb") as target:
        np.savez_compressed(
            target,
            perimeter_mask=perimeter_mask,
            active_line_mask=active_line_mask,
            active_line_mask_raw=active_line_raw,
            source_active_state=source_active_state,
            first_observed_frame=first_observed_frame,
            acquisition_start_utc=np.asarray(
                [feature["properties"]["acquisition_start"] for feature in perimeters]
            ),
            acquisition_end_utc=np.asarray(acquisition_end),
            elapsed_minute=elapsed_minute,
            reported_area_km2=reported_area_km2,
            rasterized_area_km2=rasterized_area_km2,
            spatial_coverage_fraction=spatial_coverage_fraction,
            cell_size_m=np.asarray(float(landscape.metadata["cell_size_m"])),
        )
    temporary.replace(output_path)

    finite_coverage = spatial_coverage_fraction[np.isfinite(spatial_coverage_fraction)]
    manifest = {
        "schema_version": 1,
        "dataset": "incident-grid progression observation cube",
        "incident_id": incident.incident_id,
        "target_grid": {
            "crs": target_crs,
            "shape": list(grid_shape),
            "transform": [float(value) for value in transform_values],
            "cell_size_m": float(landscape.metadata["cell_size_m"]),
            "polygon_rasterization": "cell_center_in_polygon",
            "line_rasterization": "all_intersected_cells",
        },
        "source": {
            "observation_manifest_sha256": _sha256(paths["observation_manifest"]),
            "perimeters_sha256": _sha256(paths["perimeters"]),
            "concurrent_active_lines_sha256": _sha256(paths["concurrent_active_lines"]),
            "source_crs": source_crs,
        },
        "audit": {
            "frame_count": len(perimeters),
            "nonempty_perimeter_frames": int(np.count_nonzero(perimeter_mask.any(axis=(1, 2)))),
            "cumulative_nesting_violation_frames": int(np.count_nonzero(lost_cells)),
            "maximum_lost_cells_between_frames": int(lost_cells.max(initial=0)),
            "missing_concurrent_active_line_count": len(missing_line_times),
            "missing_concurrent_active_line_times": missing_line_times,
            "dormant_line_frames": int(np.count_nonzero(source_active_state == 0.0)),
            "effective_active_line_frames": int(np.count_nonzero(active_line_mask.any(axis=(1, 2)))),
            "spatial_coverage_fraction": {
                "minimum": float(np.min(finite_coverage)),
                "median": float(np.median(finite_coverage)),
                "maximum": float(np.max(finite_coverage)),
                "final": float(spatial_coverage_fraction[-1]),
            },
        },
        "asset": {
            "href": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest

