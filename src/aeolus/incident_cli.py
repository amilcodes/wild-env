"""Build and inspect portable historical incident bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.data import IncidentBundle, load_bundle, write_incident_bundle
from aeolus.data.importers import (
    build_landscape_from_services,
    enrich_scenario_from_landscape,
    fetch_feds_perimeters,
    geojson_bbox,
)


def _import_feds(args: argparse.Namespace) -> None:
    destination = Path(args.out)
    perimeters = fetch_feds_perimeters(args.region, args.fire_id)
    bbox = geojson_bbox(perimeters)
    landscape, landscape_path = build_landscape_from_services(
        bbox,
        destination / "provenance",
        size=(args.size, args.size),
        buffer_m=args.buffer_m,
        split=args.split,
    )
    features = perimeters["features"]
    start = features[0]["properties"]["observed_at"]
    end = features[-1]["properties"]["observed_at"]
    source = perimeters["aeolus:source"]
    bundle = write_incident_bundle(
        destination,
        incident_id=args.incident_id or f"feds-{args.region.lower()}-{args.fire_id}",
        bbox=bbox,
        start_datetime=start,
        end_datetime=end,
        scenario_bundle=landscape,
        perimeter_collection=perimeters,
        source_landscape=landscape_path,
        weather_path=args.weather,
        title=args.title,
        sources=[
            {"name": "NASA FEDS", **source},
            {"name": "USGS 3DEP"},
            {"name": "LANDFIRE 2025"},
        ],
    )
    print(json.dumps(_summary(bundle), indent=2))


def _summary(bundle: IncidentBundle) -> dict[str, object]:
    landscape = bundle.scenario_bundle()
    perimeters = bundle.perimeter_collection()["features"]
    return {
        "incident_id": bundle.incident_id,
        "root": str(bundle.root.resolve()),
        "bbox_wgs84": bundle.bbox,
        "grid_shape": list(landscape.elevation_m.shape),
        "cell_size_m": float(landscape.metadata["cell_size_m"]),
        "perimeter_frames": len(perimeters),
        "start_datetime": bundle.item["properties"]["start_datetime"],
        "end_datetime": bundle.item["properties"]["end_datetime"],
        "assets": sorted(bundle.item["assets"]),
    }


def _assemble(args: argparse.Namespace) -> None:
    perimeter_path = Path(args.perimeters)
    perimeters = json.loads(perimeter_path.read_text(encoding="utf-8"))
    if perimeters.get("type") != "FeatureCollection":
        raise ValueError("perimeters must be a GeoJSON FeatureCollection")
    observed_at = sorted(
        feature.get("properties", {}).get("observed_at")
        for feature in perimeters.get("features", [])
        if isinstance(feature.get("properties", {}).get("observed_at"), str)
    )
    if len(observed_at) < 2:
        raise ValueError("at least two perimeter features require normalized observed_at")
    bundle = write_incident_bundle(
        args.out,
        incident_id=args.incident_id,
        bbox=geojson_bbox(perimeters),
        start_datetime=observed_at[0],
        end_datetime=observed_at[-1],
        scenario_bundle=load_bundle(args.scenario),
        perimeter_collection=perimeters,
        source_landscape=args.landscape,
        weather_path=args.weather,
        title=args.title,
        sources=[{"name": args.source_name, "path": str(perimeter_path.resolve())}],
    )
    print(json.dumps(_summary(bundle), indent=2))


def _enrich(args: argparse.Namespace) -> None:
    bundle = enrich_scenario_from_landscape(
        args.scenario,
        args.landscape,
        args.out,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.out).resolve()),
                "schema_version": bundle.metadata["schema_version"],
                "grid_shape": list(bundle.elevation_m.shape),
                "fuel_models": sorted(
                    int(value) for value in set(bundle.fuel_model_number.flat)
                ),
                "canopy_cover_range": [
                    float(bundle.canopy_cover.min()),
                    float(bundle.canopy_cover.max()),
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical incident bundle operations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import-feds", help="import NASA FEDS and public terrain/fuels")
    importer.add_argument("--region", default="CONUS")
    importer.add_argument("--fire-id", type=int, required=True)
    importer.add_argument("--out", required=True)
    importer.add_argument("--incident-id")
    importer.add_argument("--title")
    importer.add_argument("--weather", help="optional CF-NetCDF weather forcing")
    importer.add_argument("--size", type=int, default=192)
    importer.add_argument("--buffer-m", type=float, default=4500.0)
    importer.add_argument("--split", choices=("train", "development", "evaluation"), default="evaluation")
    importer.set_defaults(handler=_import_feds)

    assembler = subparsers.add_parser(
        "assemble",
        help="assemble normalized local perimeters and aligned landscape assets",
    )
    assembler.add_argument("--incident-id", required=True)
    assembler.add_argument("--perimeters", required=True)
    assembler.add_argument("--scenario", required=True, help="aligned simulator NPZ")
    assembler.add_argument("--out", required=True)
    assembler.add_argument("--landscape", help="optional source GeoTIFF")
    assembler.add_argument("--weather", help="optional CF-NetCDF weather forcing")
    assembler.add_argument("--source-name", default="local-normalized-source")
    assembler.add_argument("--title")
    assembler.set_defaults(handler=_assemble)

    enrich = subparsers.add_parser(
        "enrich-scenario",
        help="add retained FBFM40 and canopy bands to a legacy scenario NPZ",
    )
    enrich.add_argument("--scenario", required=True)
    enrich.add_argument("--landscape", required=True)
    enrich.add_argument("--out", required=True)
    enrich.set_defaults(handler=_enrich)

    for command in ("validate", "inspect"):
        operation = subparsers.add_parser(command)
        operation.add_argument("incident")
        operation.set_defaults(
            handler=lambda args, validate=command == "validate": print(
                json.dumps(
                    {
                        **_summary(IncidentBundle.load(args.incident)),
                        "valid": True,
                    }
                    if validate
                    else _summary(IncidentBundle.load(args.incident)),
                    indent=2,
                )
            )
        )
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
