"""Import GOFER hourly fire progression as uncertainty-aware observations.

GOFER is a retrospective satellite reconstruction.  Its hourly timestamps are
kept as acquisition windows and are never represented as operational issue
times.  Spatial validation statistics are retained as population-level
metadata rather than mislabelled as a per-frame error distribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

GOFER_VERSION = "0.2"
GOFER_DOI = "10.5281/zenodo.14642378"
GOFER_PAPER_DOI = "10.5194/essd-16-1395-2024"
GOFER_ARCHIVE_MD5 = "8d495af1e4a0ed77df35b5a15d5ebb04"

_VARIANTS: dict[str, dict[str, Any]] = {
    "combined": {
        "directory": "GOFER_Combined",
        "prefix": "GOFERC",
        "nominal_resolution_m": [1600.0, 1700.0],
        "final_edge_error_mean_m": 750.0,
        "final_edge_error_between_fire_sd_m": 210.0,
        "final_edge_error_maximum_mean_m": 2860.0,
        "final_edge_error_maximum_between_fire_sd_m": 1140.0,
    },
    "east": {
        "directory": "GOFER_East",
        "prefix": "GOFERE",
        "nominal_resolution_m": [3100.0, 3600.0],
        "final_edge_error_mean_m": 1440.0,
        "final_edge_error_between_fire_sd_m": 440.0,
        "final_edge_error_maximum_mean_m": 5080.0,
        "final_edge_error_maximum_between_fire_sd_m": 1800.0,
    },
    "west": {
        "directory": "GOFER_West",
        "prefix": "GOFERW",
        "nominal_resolution_m": [2500.0, 2700.0],
        "final_edge_error_mean_m": 870.0,
        "final_edge_error_between_fire_sd_m": 310.0,
        "final_edge_error_maximum_mean_m": 2940.0,
        "final_edge_error_maximum_between_fire_sd_m": 1040.0,
    },
}


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _utc(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _product_root(path: Path) -> Path:
    candidate = path / "GOFER"
    return candidate if candidate.is_dir() else path


def _variant_paths(root: Path, variant: str) -> dict[str, Path]:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown GOFER variant {variant!r}; choose from {sorted(_VARIANTS)}")
    product_root = _product_root(root)
    specification = _VARIANTS[variant]
    directory = product_root / specification["directory"]
    prefix = specification["prefix"]
    paths = {
        "catalog": product_root / "fireData.csv",
        "perimeters": directory / f"{prefix}_fireProg.shp",
        "concurrent_lines": directory / f"{prefix}_cfireLine.shp",
        "retrospective_lines": directory / f"{prefix}_rfireLine.shp",
        "summary": directory / f"{prefix}_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GOFER product files are missing: {missing}")
    return paths


def gofer_fire_catalog(root: str | Path) -> list[dict[str, Any]]:
    """Read the source fire catalog without altering its ignition semantics."""

    catalog = _product_root(Path(root)) / "fireData.csv"
    with catalog.open(newline="", encoding="utf-8-sig") as source:
        records = list(csv.DictReader(source))
    return [
        {
            "fire_name": record["fname"],
            "fire_year": int(record["fyear"]),
            "official_acres": float(record["acres_official"]),
            "goes_ignition_utc": record["GOESIg_UTC"],
            "local_timezone": record["local_tz"],
            "fixed_offset_timezone": record["local_tzGMT"],
        }
        for record in records
    ]


def _select_catalog_record(
    records: Iterable[dict[str, Any]], fire_name: str, fire_year: int
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record["fire_name"].casefold() == fire_name.casefold()
        and record["fire_year"] == fire_year
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one GOFER catalog match for {fire_name!r} ({fire_year}), found {len(matches)}"
        )
    return matches[0]


def _shape_features(
    path: Path,
    *,
    fire_name: str,
    fire_year: int,
    record_type: str,
    variant: str,
    concurrent_confidence: float | None = None,
) -> list[dict[str, Any]]:
    try:
        import shapefile
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to import GOFER") from exc

    features: list[dict[str, Any]] = []
    reader = shapefile.Reader(str(path))
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        if str(record["fname"]).casefold() != fire_name.casefold():
            continue
        if int(record["fyear"]) != fire_year:
            continue
        if concurrent_confidence is not None and abs(
            float(record["fconf"]) - concurrent_confidence
        ) > 1e-8:
            continue
        observed_at = _utc(str(record["tUTC"]))
        acquisition_start = observed_at - timedelta(hours=1)
        properties: dict[str, Any] = {
            "source": "GOFER",
            "source_version": GOFER_VERSION,
            "source_variant": variant,
            "source_record_type": record_type,
            "observed_at": _iso(observed_at),
            "acquisition_start": _iso(acquisition_start),
            "acquisition_end": _iso(observed_at),
            "available_at": None,
            "operationally_available": False,
            "availability_reason": "retrospective satellite reconstruction",
            "timestep_hours": float(record["timestep"]),
        }
        field_map = {
            "farea": "reported_cumulative_area_km2",
            "fareaPer": "reported_final_area_percent",
            "fperim": "reported_perimeter_length_km",
            "cflinelen": "reported_active_line_length_km",
            "rflinelen": "reported_active_line_length_km",
            "fconf": "fire_detection_confidence_threshold",
            "fstate": "source_active_state",
        }
        for source_name, normalized_name in field_map.items():
            value = record.get(source_name)
            if value is not None:
                properties[normalized_name] = float(value)
        features.append(
            {
                "type": "Feature",
                "id": (
                    f"gofer-{GOFER_VERSION}-{variant}-{_slug(fire_name)}-"
                    f"{record_type}-{int(round(float(record['timestep']))):06d}"
                ),
                "geometry": shape_record.shape.__geo_interface__,
                "properties": properties,
            }
        )
    features.sort(key=lambda feature: feature["properties"]["observed_at"])
    if not features:
        raise ValueError(f"no {record_type} records found for {fire_name!r} ({fire_year})")
    return features


def _write_filtered_summary(
    source_path: Path,
    destination: Path,
    *,
    fire_name: str,
    fire_year: int,
) -> int:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    count = 0
    with source_path.open(newline="", encoding="utf-8-sig") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("GOFER summary has no header")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for record in reader:
            if record["fname"].casefold() == fire_name.casefold() and int(record["fyear"]) == fire_year:
                writer.writerow(record)
                count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"no GOFER summary records found for {fire_name!r} ({fire_year})")
    temporary.replace(destination)
    return count


def _temporal_audit(features: list[dict[str, Any]]) -> dict[str, Any]:
    times = [
        _utc(feature["properties"]["observed_at"].replace("T", " ").replace("Z", ""))
        for feature in features
    ]
    gaps = [
        {
            "after": _iso(previous),
            "before": _iso(current),
            "duration_hours": (current - previous).total_seconds() / 3600.0,
        }
        for previous, current in zip(times, times[1:], strict=False)
        if (current - previous) != timedelta(hours=1)
    ]
    areas = [feature["properties"]["reported_cumulative_area_km2"] for feature in features]
    area_violations = sum(
        current + 1e-9 < previous
        for previous, current in zip(areas, areas[1:], strict=False)
    )
    return {
        "first_acquisition_end": _iso(times[0]),
        "last_acquisition_end": _iso(times[-1]),
        "record_count": len(features),
        "non_hourly_gap_count": len(gaps),
        "non_hourly_gaps": gaps,
        "reported_cumulative_area_decrease_count": area_violations,
    }


def _geometry_audit(features: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from shapely.geometry import shape
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to audit GOFER geometry") from exc
    geometries = [shape(feature["geometry"]) for feature in features]
    pairs = zip(features, geometries, strict=True)
    invalid = [feature["id"] for feature, geometry in pairs if not geometry.is_valid]
    pairs = zip(features, geometries, strict=True)
    empty = [feature["id"] for feature, geometry in pairs if geometry.is_empty]
    return {
        "record_count": len(features),
        "valid_geometry_count": len(features) - len(invalid),
        "invalid_geometry_count": len(invalid),
        "invalid_feature_ids": invalid,
        "empty_geometry_count": len(empty),
        "empty_feature_ids": empty,
        "geometry_types": sorted({geometry.geom_type for geometry in geometries}),
    }


def write_gofer_observation_bundle(
    root: str | Path,
    destination: str | Path,
    *,
    fire_name: str,
    fire_year: int,
    variant: str = "combined",
    concurrent_confidence: float = 0.05,
    source_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Write one fire's normalized hourly perimeter and active-line bundle."""

    if concurrent_confidence not in {0.05, 0.1, 0.25, 0.5, 0.75, 0.9}:
        raise ValueError("GOFER concurrent confidence must be a published threshold")
    source_root = _product_root(Path(root))
    paths = _variant_paths(source_root, variant)
    catalog_record = _select_catalog_record(gofer_fire_catalog(source_root), fire_name, fire_year)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    populated = [path.name for path in output.iterdir()]
    if populated:
        raise FileExistsError(f"GOFER output directory is not empty: {output}")

    perimeters = _shape_features(
        paths["perimeters"],
        fire_name=fire_name,
        fire_year=fire_year,
        record_type="hourly_cumulative_perimeter",
        variant=variant,
    )
    concurrent = _shape_features(
        paths["concurrent_lines"],
        fire_name=fire_name,
        fire_year=fire_year,
        record_type="concurrent_active_fire_line",
        variant=variant,
        concurrent_confidence=concurrent_confidence,
    )
    retrospective = _shape_features(
        paths["retrospective_lines"],
        fire_name=fire_name,
        fire_year=fire_year,
        record_type="retrospective_active_fire_line",
        variant=variant,
    )
    output_files = {
        "perimeters": output / "perimeters.geojson",
        "concurrent_active_lines": output / "concurrent_active_lines.geojson",
        "retrospective_active_lines": output / "retrospective_active_lines.geojson",
        "progression_summary": output / "progression_summary.csv",
    }
    collections = {
        "perimeters": perimeters,
        "concurrent_active_lines": concurrent,
        "retrospective_active_lines": retrospective,
    }
    for name, features in collections.items():
        _atomic_json(
            output_files[name],
            {
                "type": "FeatureCollection",
                "name": f"{fire_name} {name.replace('_', ' ')}",
                "features": features,
            },
        )
    summary_count = _write_filtered_summary(
        paths["summary"],
        output_files["progression_summary"],
        fire_name=fire_name,
        fire_year=fire_year,
    )

    component_paths: list[Path] = []
    for path in paths.values():
        component_paths.append(path)
        if path.suffix == ".shp":
            component_paths.extend(path.with_suffix(suffix) for suffix in (".shx", ".dbf", ".prj"))
    source_record: dict[str, Any] = {
        "product": "GOES-Observed Fire Event Representation",
        "version": GOFER_VERSION,
        "variant": variant,
        "doi": GOFER_DOI,
        "paper_doi": GOFER_PAPER_DOI,
        "license": "CC-BY-4.0",
        "component_sha256": {
            path.relative_to(source_root).as_posix(): _hash(path)
            for path in sorted(set(component_paths))
        },
    }
    if source_archive is not None:
        archive = Path(source_archive)
        source_record["archive"] = {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "md5": _hash(archive, "md5"),
            "sha256": _hash(archive),
            "expected_v0.2_md5": GOFER_ARCHIVE_MD5,
        }

    spatial = dict(_VARIANTS[variant])
    spatial.pop("directory")
    spatial.pop("prefix")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "GOFER hourly progression observation bundle",
        "software": {
            "importer_module": str(Path(__file__).resolve()),
            "importer_module_sha256": _hash(Path(__file__).resolve()),
        },
        "source": source_record,
        "fire": catalog_record,
        "observation_semantics": {
            "perimeter_type": "cumulative burned extent",
            "acquisition_window": "preceding nominal hour through tUTC",
            "timestamp_role": "end of source hour",
            "availability": "retrospective only; no operational issue time supplied",
            "concurrent_active_line_confidence": concurrent_confidence,
            "source_crs": "EPSG:4326",
            "spatial_validation": spatial,
            "spatial_validation_scope": (
                "population statistics for final perimeters; not a calibrated per-frame Gaussian sigma"
            ),
            "early_progression_warning": (
                "the source paper reports low spatial accuracy before 100 hours and possible "
                "early-perimeter overinflation"
            ),
        },
        "audit": {
            "perimeters": _temporal_audit(perimeters),
            "perimeter_geometry": _geometry_audit(perimeters),
            "concurrent_active_line_records": len(concurrent),
            "concurrent_active_line_geometry": _geometry_audit(concurrent),
            "concurrent_active_line_dormant_records": sum(
                feature["properties"].get("source_active_state") == 0.0
                for feature in concurrent
            ),
            "retrospective_active_line_records": len(retrospective),
            "retrospective_active_line_geometry": _geometry_audit(retrospective),
            "progression_summary_records": summary_count,
        },
        "assets": {},
    }
    manifest_path = output / "observation_manifest.json"
    for name, path in output_files.items():
        manifest["assets"][name] = {
            "href": path.name,
            "bytes": path.stat().st_size,
            "sha256": _hash(path),
        }
    _atomic_json(manifest_path, manifest)
    return manifest
