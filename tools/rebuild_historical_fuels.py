#!/usr/bin/env python3
"""Rebuild prepared incidents with time-admissible LANDFIRE fuels."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from aeolus.data import (
    IncidentBundle,
    reconstruct_historical_landscape,
    select_historical_landfire_version,
    write_incident_bundle,
)
from aeolus.evaluation.validity import assess_historical_fuel_provenance


def _incident_directories(root: Path) -> list[Path]:
    return sorted(
        {item.parent for pattern in ("*/item.json", "incidents/*/item.json") for item in root.glob(pattern)}
    )


def _replace_landfire_source(
    sources: list[dict[str, Any]],
    replacement: dict[str, Any],
) -> list[dict[str, Any]]:
    retained = [dict(source) for source in sources if "landfire" not in str(source.get("name", "")).lower()]
    return [*retained, replacement]


def _rebuild_one(source_root: Path, destination_root: Path) -> dict[str, Any]:
    incident = IncidentBundle.load(source_root)
    properties = incident.item["properties"]
    start = str(properties["start_datetime"])
    selected = select_historical_landfire_version(
        start,
        require_streamable=True,
    )
    preferred = select_historical_landfire_version(
        start,
        require_streamable=False,
    )
    destination = destination_root / source_root.name
    checkpoint = destination / "fuel_rebuild_record.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    provenance = destination / "provenance" / selected.version_id
    source_landscape = incident.asset_path("landscape")
    weather = incident.asset_path("weather", required=False)
    if source_landscape is None:
        raise KeyError(f"{incident.incident_id} has no source landscape")
    rebuilt, statistics = reconstruct_historical_landscape(
        incident.scenario_bundle(),
        source_landscape,
        version=selected,
        provenance_directory=provenance,
        output_landscape_path=provenance / "landscape.tif",
    )
    fuel_source = dict(rebuilt.metadata["sources"][-1])
    item_sources = _replace_landfire_source(
        list(properties.get("aeolus:sources", [])),
        {
            "name": selected.display_name,
            "version_id": selected.version_id,
            "disturbance_through_year": selected.disturbance_through_year,
            "effective_condition_year": selected.effective_condition_year,
            "completion_year": selected.completion_year,
            "data_cutoff": fuel_source["data_cutoff"],
            "access_status": selected.access_status,
            "evidence_urls": list(selected.evidence_urls),
        },
    )
    output = write_incident_bundle(
        destination,
        incident_id=incident.incident_id,
        bbox=incident.bbox,
        start_datetime=start,
        end_datetime=str(properties["end_datetime"]),
        scenario_bundle=rebuilt,
        perimeter_collection=incident.perimeter_collection(),
        source_landscape=provenance / "landscape.tif",
        weather_path=weather,
        title=str(properties.get("title", incident.incident_id)),
        sources=item_sources,
    )
    assessment = assess_historical_fuel_provenance(
        output.scenario_bundle(),
        incident_start=start,
    )
    record = {
        "incident_id": incident.incident_id,
        "source_bundle": str(source_root.resolve()),
        "output_bundle": str(output.root.resolve()),
        "incident_start": start,
        "preferred_version": preferred.as_dict(),
        "selected_streamable_version": selected.as_dict(),
        "preferred_version_is_streamable": preferred.streamable,
        "archive_substitution_remaining": (preferred.version_id != selected.version_id),
        "fuel_provenance_assessment": assessment.as_dict(),
        "statistics": statistics,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".partial.json")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(checkpoint)
    return record


def rebuild_corpus(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    workers: int,
) -> dict[str, Any]:
    source = Path(source_root)
    destination = Path(destination_root)
    if source.resolve() == destination.resolve():
        raise ValueError("historical fuel reconstruction requires a separate output root")
    incidents = _incident_directories(source)
    if not incidents:
        raise FileNotFoundError(f"no incident bundles beneath {source}")
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(incidents)))) as pool:
        futures = {pool.submit(_rebuild_one, incident, destination): incident for incident in incidents}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": record["incident_id"],
                        "fuel_model_changed_fraction": record["statistics"]["fuel_model_changed_fraction"],
                    }
                ),
                flush=True,
            )
    records.sort(key=lambda record: str(record["incident_id"]))
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["fuel_provenance_assessment"]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": 1,
        "study": "time-admissible historical fuel reconstruction",
        "source_root": str(source.resolve()),
        "destination_root": str(destination.resolve()),
        "incident_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "gate_passes": (set(status_counts) == {"historically_admissible_by_product_date"}),
        "selection_rule": (
            "disturbance_through_year is earlier than the incident year; "
            "effective_condition_year is no later than the incident year; "
            "the most recent currently streamable qualifying vintage is used"
        ),
        "incidents": records,
    }
    (destination / "fuel_rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    prepared = {
        "schema_version": 1,
        "source_prepared_root": str(source.resolve()),
        "incidents": [
            {
                "incident_id": record["incident_id"],
                "bundle": record["output_bundle"],
            }
            for record in records
        ],
    }
    (destination / "prepared.json").write_text(
        json.dumps(prepared, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = rebuild_corpus(
        args.source_root,
        args.out,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
