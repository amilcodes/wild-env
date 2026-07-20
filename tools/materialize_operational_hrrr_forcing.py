#!/usr/bin/env python3
"""Materialize pre-issue HRRR forecast forcing for a frozen benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from aeolus.data import (
    IncidentBundle,
    WeatherForcing,
    fetch_hrrr_forecast,
    initialize_causal_forecast_moisture,
    scenario_lonlat_grid,
    write_weather_forcing,
)
from aeolus.evaluation.frozen_benchmark import (
    _pairs,
    _slug,
    audit_frozen_contract,
    load_frozen_contract,
)
from aeolus.evaluation.historical import PerimeterSeries
from aeolus.evaluation.validity import assess_forcing_availability


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requested_pairs(
    specification: dict[str, Any],
    split: str,
) -> list[tuple[int, int]]:
    pairs = _pairs(specification, split)
    # Canonical physics reconstructs the arrival state from two perimeters.
    return [pair for pair in pairs if pair[0] >= 1]


def materialize(
    contract_path: Path,
    prepared_root: Path,
    destination: Path,
    *,
    assumed_availability_lag_hours: float,
    splits: tuple[str, ...],
    limit: int | None,
) -> dict[str, Any]:
    contract = load_frozen_contract(contract_path)
    audit = audit_frozen_contract(contract)
    if not audit["valid"]:
        raise ValueError(f"frozen contract failed audit: {audit}")
    base = contract["resolved_base_manifest"]
    specifications = {str(item["incident_code"]): item for item in base["incidents"]}
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    requested = 0
    for split in splits:
        for code in audit["assignments"][split]:
            specification = specifications[code]
            bundle = IncidentBundle.load(prepared_root / _slug(code))
            scenario = bundle.scenario_bundle()
            latitude, longitude = scenario_lonlat_grid(
                scenario.metadata,
                scenario.elevation_m.shape,
            )
            series = PerimeterSeries.from_incident(bundle)
            background_path = bundle.asset_path("weather")
            assert background_path is not None
            background = WeatherForcing.load(background_path)
            incident_root = destination / _slug(code)
            cache_root = destination / ".hrrr-cache" / _slug(code)
            for start_index, target_index in _requested_pairs(specification, split):
                if limit is not None and requested >= limit:
                    break
                requested += 1
                start = series.frames[start_index].timestamp
                target = series.frames[target_index].timestamp
                path = incident_root / f"pair_{start_index:03d}_{target_index:03d}.nc"
                if path.exists():
                    forcing = WeatherForcing.load(path)
                else:
                    atmospheric = fetch_hrrr_forecast(
                        latitude,
                        longitude,
                        start,
                        target,
                        cache_directory=cache_root,
                        assumed_availability_lag_hours=(assumed_availability_lag_hours),
                    )
                    forcing = initialize_causal_forecast_moisture(
                        atmospheric,
                        background,
                        issue_time=start,
                    )
                    forcing = replace(
                        forcing,
                        metadata={
                            **forcing.metadata,
                            "incident_code": code,
                            "benchmark_split": split,
                            "perimeter_start_index": start_index,
                            "perimeter_target_index": target_index,
                            "forecast_target_time": target.isoformat().replace("+00:00", "Z"),
                        },
                    )
                    temporary = path.with_suffix(".partial.nc")
                    write_weather_forcing(
                        temporary,
                        forcing,
                        start_datetime=forcing.time_origin,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary.replace(path)
                availability = assess_forcing_availability(
                    forcing,
                    forecast_start=start,
                    forecast_end=target,
                )
                if not availability["supports_operational_forecast"]:
                    raise RuntimeError(
                        f"operational forcing audit failed for {code} "
                        f"{start_index}->{target_index}: {availability}"
                    )
                records.append(
                    {
                        "incident_code": code,
                        "split": split,
                        "start_index": start_index,
                        "target_index": target_index,
                        "issue_time": start.isoformat(),
                        "target_time": target.isoformat(),
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "forecast_reference_time": forcing.metadata.get("forecast_reference_time"),
                        "availability": availability,
                        "moisture_initialization_time": forcing.metadata.get(
                            "fuel_moisture_initialization_time"
                        ),
                        "live_fuel_default_used": forcing.metadata.get("live_fuel_default_used"),
                    }
                )
                print(
                    f"[operational-forcing] {len(records)} transitions complete: "
                    f"{code} {start_index}->{target_index}",
                    file=sys.stderr,
                    flush=True,
                )
            if limit is not None and requested >= limit:
                break
        if limit is not None and requested >= limit:
            break
    result = {
        "schema_version": 1,
        "study": "frozen historical operational HRRR transition forcing",
        "contract_path": str(contract_path.resolve()),
        "contract_base_manifest_sha256": audit["base_manifest_sha256"],
        "prepared_root": str(prepared_root.resolve()),
        "assumed_availability_lag_hours": assumed_availability_lag_hours,
        "requested_splits": list(splits),
        "transition_count": len(records),
        "complete": limit is None,
        "all_operationally_available": all(
            item["availability"]["supports_operational_forecast"] for item in records
        ),
        "records": records,
    }
    manifest = destination / "operational_forcing_manifest.json"
    temporary_manifest = manifest.with_suffix(".partial.json")
    temporary_manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--availability-lag-hours", type=float, default=2.0)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "development", "test"),
        default=("train", "development", "test"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = materialize(
        args.contract,
        args.prepared_root,
        args.destination,
        assumed_availability_lag_hours=args.availability_lag_hours,
        splits=tuple(args.splits),
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "transition_count": result["transition_count"],
                "complete": result["complete"],
                "all_operationally_available": result["all_operationally_available"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
