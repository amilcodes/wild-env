#!/usr/bin/env python3
"""Clone a prepared historical corpus onto physically meaningful metric grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data import (
    IncidentBundle,
    WeatherForcing,
    reproject_scenario_to_metric,
    reproject_weather_to_scenario,
    write_bundle,
    write_weather_forcing,
)
from aeolus.evaluation.historical import PerimeterSeries
from aeolus.evaluation.validity import assess_metric_crs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_landscape_geotiff(path: Path, scenario: Any) -> None:
    try:
        import rasterio
        from affine import Affine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to write metric landscapes") from exc

    bands = (
        ("elevation_m", scenario.elevation_m),
        ("fbfm40_code", scenario.fuel_model_number),
        ("canopy_cover", scenario.canopy_cover),
        ("canopy_height", scenario.canopy_height_m),
        ("canopy_base_height", scenario.canopy_base_height_m),
        ("canopy_bulk_density", scenario.canopy_bulk_density_kg_m3),
    )
    if any(value is None for _, value in bands):
        raise ValueError("metric historical landscape requires all six source bands")
    profile = {
        "driver": "GTiff",
        "height": scenario.elevation_m.shape[0],
        "width": scenario.elevation_m.shape[1],
        "count": len(bands),
        "dtype": "float32",
        "crs": str(scenario.metadata["crs"]),
        "transform": Affine(*scenario.metadata["transform"]),
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as destination:
        for index, (name, value) in enumerate(bands, start=1):
            destination.write(np.asarray(value, dtype=np.float32), index)
            destination.set_band_description(index, name)


def _area_audit(bundle: IncidentBundle, series: PerimeterSeries) -> dict[str, Any]:
    try:
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely.ops import transform as transform_geometry
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for perimeter area audit") from exc

    scenario = bundle.scenario_bundle()
    transformer = Transformer.from_crs(
        "EPSG:4326",
        str(scenario.metadata["crs"]),
        always_xy=True,
    )
    by_time: dict[str, list[Any]] = defaultdict(list)
    for feature in bundle.perimeter_collection()["features"]:
        by_time[str(feature["properties"]["observed_at"])].append(
            transform_geometry(transformer.transform, shape(feature["geometry"]))
        )
    frame_by_time = {frame.timestamp.isoformat(): frame for frame in series.frames}
    cell_area_m2 = float(series.cell_size_m**2)
    records = []
    for timestamp, geometries in sorted(by_time.items()):
        normalized = timestamp.replace("Z", "+00:00")
        frame = frame_by_time.get(normalized)
        if frame is None:
            continue
        exact_area_m2 = float(unary_union(geometries).area)
        raster_area_m2 = float(frame.mask.sum()) * cell_area_m2
        records.append(
            {
                "timestamp": timestamp,
                "exact_area_km2": exact_area_m2 / 1_000_000.0,
                "raster_area_km2": raster_area_m2 / 1_000_000.0,
                "raster_to_exact_area_ratio": raster_area_m2 / exact_area_m2,
            }
        )
    ratios = np.asarray(
        [record["raster_to_exact_area_ratio"] for record in records],
        dtype=np.float64,
    )
    return {
        "records": records,
        "median_raster_to_exact_area_ratio": float(np.median(ratios)),
        "maximum_absolute_fractional_area_error": float(np.max(np.abs(ratios - 1.0))),
    }


def reproject_corpus(source_root: Path, destination_root: Path) -> dict[str, Any]:
    if destination_root.exists():
        raise FileExistsError(f"refusing to overwrite destination: {destination_root}")
    temporary_root = destination_root.with_name(f".{destination_root.name}.partial")
    if temporary_root.exists():
        raise FileExistsError(f"remove or inspect incomplete destination: {temporary_root}")
    shutil.copytree(source_root, temporary_root)
    records: list[dict[str, Any]] = []
    try:
        incident_roots = sorted(path.parent for path in temporary_root.glob("*/item.json"))
        if not incident_roots:
            raise ValueError("source corpus contains no incident bundles")
        for incident_root in incident_roots:
            bundle = IncidentBundle.load(incident_root)
            source_scenario = bundle.scenario_bundle()
            before = assess_metric_crs(source_scenario)
            metric_scenario = reproject_scenario_to_metric(source_scenario)
            after = assess_metric_crs(metric_scenario)
            if not after["supports_physical_distance_claims"]:
                raise RuntimeError(f"metric projection audit failed for {bundle.incident_id}")

            simulator_path = bundle.asset_path("simulator-landscape")
            assert simulator_path is not None
            write_bundle(simulator_path, metric_scenario)
            landscape_path = bundle.asset_path("landscape", required=False)
            if landscape_path is not None:
                _write_landscape_geotiff(landscape_path, metric_scenario)

            weather_path = bundle.asset_path("weather", required=False)
            if weather_path is not None:
                weather = WeatherForcing.load(weather_path)
                metric_weather = reproject_weather_to_scenario(
                    weather,
                    source_scenario,
                    metric_scenario,
                )
                origin = metric_weather.time_origin
                if origin is None:
                    origin = str(bundle.item["properties"]["start_datetime"])
                write_weather_forcing(
                    weather_path,
                    metric_weather,
                    start_datetime=origin,
                )

            item = dict(bundle.item)
            properties = dict(item["properties"])
            sources = list(properties.get("aeolus:sources", []))
            sources.append(
                {
                    "name": "Aeolus metric-grid correction",
                    "source_crs": before["crs"],
                    "destination_crs": after["crs"],
                    "reason": (
                        "physical simulation requires ground metres; Web Mercator "
                        "map units have latitude-dependent scale"
                    ),
                }
            )
            properties["aeolus:sources"] = sources
            properties["proj:epsg"] = after["epsg"]
            item["properties"] = properties
            (incident_root / "item.json").write_text(
                json.dumps(item, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            updated = IncidentBundle.load(incident_root)
            series = PerimeterSeries.from_incident(updated)
            record = {
                "incident_id": updated.incident_id,
                "before": before,
                "after": after,
                "source_shape": list(source_scenario.elevation_m.shape),
                "destination_shape": list(metric_scenario.elevation_m.shape),
                "source_cell_size_map_m": float(source_scenario.metadata["cell_size_m"]),
                "destination_cell_size_ground_m": float(metric_scenario.metadata["cell_size_m"]),
                "perimeter_area_audit": _area_audit(updated, series),
                "assets": {
                    "simulator_sha256": _sha256(simulator_path),
                    "perimeters_sha256": _sha256(updated.asset_path("observed-perimeters")),
                    "weather_sha256": (_sha256(weather_path) if weather_path is not None else None),
                },
            }
            records.append(record)

        manifest = {
            "schema_version": 1,
            "purpose": "metric-grid correction for historical physical validation",
            "source_root": str(source_root.resolve()),
            "incidents": records,
            "gate": {
                "all_metric_crs": all(item["after"]["supports_physical_distance_claims"] for item in records),
                "maximum_absolute_fractional_raster_area_error": max(
                    item["perimeter_area_audit"]["maximum_absolute_fractional_area_error"] for item in records
                ),
            },
        }
        (temporary_root / "metric_reprojection_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        temporary_root.rename(destination_root)
        return manifest
    except Exception:
        # Keep the partial corpus for forensic inspection; never publish it at
        # the requested destination path.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()
    result = reproject_corpus(args.source_root, args.destination_root)
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
