"""Versioned, STAC-compatible incident bundles for simulation and replay."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aeolus.data.bundle import ScenarioBundle, load_bundle, write_bundle

INCIDENT_SCHEMA_VERSION = 2
STAC_VERSION = "1.1.0"


def _iso_utc(value: datetime | str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_asset(root: Path, href: str) -> Path:
    candidate = (root / href).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"incident asset escapes bundle root: {href}")
    return candidate


@dataclass(frozen=True)
class IncidentBundle:
    """A portable incident directory described by one STAC Item."""

    root: Path
    item: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> IncidentBundle:
        bundle_root = Path(root)
        with (bundle_root / "item.json").open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        bundle = cls(bundle_root, item)
        bundle.validate()
        return bundle

    @property
    def incident_id(self) -> str:
        return str(self.item["id"])

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        values = self.item["bbox"]
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    def asset_path(self, key: str, *, required: bool = True) -> Path | None:
        asset = self.item.get("assets", {}).get(key)
        if asset is None:
            if required:
                raise KeyError(f"incident asset is missing: {key}")
            return None
        return _safe_asset(self.root, str(asset["href"]))

    def scenario_bundle(self) -> ScenarioBundle:
        path = self.asset_path("simulator-landscape")
        assert path is not None
        return load_bundle(path)

    def perimeter_collection(self) -> dict[str, Any]:
        path = self.asset_path("observed-perimeters")
        assert path is not None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("type") != "FeatureCollection":
            raise ValueError("observed-perimeters must be a GeoJSON FeatureCollection")
        return value

    def validate(self) -> None:
        if self.item.get("type") != "Feature":
            raise ValueError("incident item must be a STAC Feature")
        if self.item.get("stac_version") != STAC_VERSION:
            raise ValueError(f"unsupported STAC version: {self.item.get('stac_version')}")
        if self.item.get("properties", {}).get("aeolus:schema_version") != INCIDENT_SCHEMA_VERSION:
            raise ValueError("unsupported incident bundle schema")
        bbox = self.item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("incident bbox must contain [west, south, east, north]")
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError("incident bbox is empty or reversed")
        for key, asset in self.item.get("assets", {}).items():
            if "href" not in asset:
                raise ValueError(f"incident asset {key!r} has no href")
            path = _safe_asset(self.root, str(asset["href"]))
            if not path.exists():
                raise FileNotFoundError(path)
        self.scenario_bundle().validate()
        self.perimeter_collection()


def write_incident_bundle(
    root: str | Path,
    *,
    incident_id: str,
    bbox: tuple[float, float, float, float],
    start_datetime: datetime | str,
    end_datetime: datetime | str,
    scenario_bundle: ScenarioBundle,
    perimeter_collection: dict[str, Any],
    source_landscape: str | Path | None = None,
    weather_path: str | Path | None = None,
    title: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> IncidentBundle:
    """Write a complete, relocatable incident bundle."""

    destination = Path(root)
    landscape_dir = destination / "landscape"
    observation_dir = destination / "observations"
    forcing_dir = destination / "forcing"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)

    simulator_path = landscape_dir / "simulator.npz"
    write_bundle(simulator_path, scenario_bundle)
    perimeter_path = observation_dir / "perimeters.geojson"
    perimeter_path.write_text(
        json.dumps(perimeter_collection, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    assets: dict[str, dict[str, Any]] = {
        "simulator-landscape": {
            "href": "landscape/simulator.npz",
            "type": "application/x-npz",
            "roles": ["data", "simulation"],
            "title": "Simulator-ready landscape arrays",
        },
        "observed-perimeters": {
            "href": "observations/perimeters.geojson",
            "type": "application/geo+json",
            "roles": ["data", "observations"],
            "title": "Time-indexed observed fire perimeters",
        },
    }
    if source_landscape is not None:
        source = Path(source_landscape)
        landscape_tif = landscape_dir / "landscape.tif"
        if source.resolve() != landscape_tif.resolve():
            shutil.copy2(source, landscape_tif)
        assets["landscape"] = {
            "href": "landscape/landscape.tif",
            "type": "image/tiff; application=geotiff",
            "roles": ["data"],
            "title": "Source terrain and fuel landscape",
        }
    if weather_path is not None:
        forcing_dir.mkdir(parents=True, exist_ok=True)
        weather_source = Path(weather_path)
        weather_destination = forcing_dir / "weather.nc"
        if weather_source.resolve() != weather_destination.resolve():
            shutil.copy2(weather_source, weather_destination)
        assets["weather"] = {
            "href": "forcing/weather.nc",
            "type": "application/x-netcdf",
            "roles": ["data", "forcing"],
            "title": "Time-indexed weather forcing",
        }

    west, south, east, north = (float(value) for value in bbox)
    item = {
        "stac_version": STAC_VERSION,
        "stac_extensions": [],
        "type": "Feature",
        "id": incident_id,
        "bbox": [west, south, east, north],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[west, south], [east, south], [east, north], [west, north], [west, south]]
            ],
        },
        "properties": {
            "datetime": None,
            "start_datetime": _iso_utc(start_datetime),
            "end_datetime": _iso_utc(end_datetime),
            "title": title or incident_id,
            "aeolus:schema_version": INCIDENT_SCHEMA_VERSION,
            "aeolus:sources": sources or [],
        },
        "links": [],
        "assets": assets,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "item.json").write_text(
        json.dumps(item, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return IncidentBundle.load(destination)
