#!/usr/bin/env python3
"""Build a reproducible, stratified NIROPS historical benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
import yaml


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _state(code: str) -> str:
    match = re.match(r"^([A-Z]{2})(?:[-_])", code)
    return match.group(1) if match else "UNKNOWN"


def inventory(path: Path) -> list[dict[str, Any]]:
    incidents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reader = shapefile.Reader(str(path))
    for record in reader.iterRecords():
        values = record.as_dict()
        incidents[str(values["Incident_C"])].append(values)
    rows: list[dict[str, Any]] = []
    for code, observations in incidents.items():
        observations.sort(key=lambda item: _timestamp(item["UTC"]))
        unique: list[dict[str, Any]] = []
        seen: set[datetime] = set()
        for observation in observations:
            time = _timestamp(observation["UTC"])
            if time not in seen:
                seen.add(time)
                unique.append(observation)
        times = [_timestamp(item["UTC"]) for item in unique]
        acres = np.asarray([max(0.0, float(item["Acres"])) for item in unique])
        intervals = np.asarray(
            [
                (right - left).total_seconds() / 3600.0
                for left, right in zip(times[:-1], times[1:], strict=True)
            ],
            dtype=np.float64,
        )
        usable = (intervals >= 6.0) & (intervals <= 36.0)
        best_start = -1
        best_length = 0
        run_start = 0
        for index, is_usable in enumerate(usable):
            if is_usable:
                length = index - run_start + 1
                if length > best_length:
                    best_start, best_length = run_start, length
            else:
                run_start = index + 1
        rows.append(
            {
                "incident_code": code,
                "state": _state(code),
                "year": times[0].year,
                "observation_count": len(unique),
                "raw_feature_count": len(observations),
                "start": times[0].isoformat(),
                "end": times[-1].isoformat(),
                "duration_days": (times[-1] - times[0]).total_seconds() / 86_400.0,
                "maximum_reported_acres": float(np.max(acres)),
                "area_monotonic_fraction": float(
                    np.mean(np.diff(acres) >= -np.maximum(5.0, 0.01 * acres[:-1]))
                )
                if len(acres) > 1
                else 1.0,
                "usable_transition_count": int(usable.sum()),
                "longest_usable_transition_run": int(best_length),
                "benchmark_window_start_index": int(best_start),
            }
        )
    return sorted(rows, key=lambda item: item["incident_code"])


def select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in rows
        if item["longest_usable_transition_run"] >= 5
        and item["maximum_reported_acres"] >= 100.0
        and item["area_monotonic_fraction"] >= 0.75
    ]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} incidents satisfy benchmark eligibility")
    acreage = np.asarray([item["maximum_reported_acres"] for item in eligible])
    boundaries = np.quantile(np.log1p(acreage), [0.25, 0.50, 0.75])
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        size_bin = int(np.searchsorted(boundaries, np.log1p(item["maximum_reported_acres"])))
        item["size_bin"] = size_bin
        groups[(str(item["state"]), size_bin)].append(item)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                -item["longest_usable_transition_run"],
                -item["observation_count"],
                item["incident_code"],
            )
        )
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def write_outputs(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    manifest_path: Path,
    inventory_path: Path,
) -> None:
    selected_codes = {item["incident_code"] for item in selected}
    fields = list(rows[0]) + ["selected"]
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({**item, "selected": item["incident_code"] in selected_codes})
    incidents = []
    for item in selected:
        start = int(item["benchmark_window_start_index"])
        incidents.append(
            {
                "incident_code": item["incident_code"],
                "calibration_pair": [start, start + 1],
                "validation_pairs": [[start + offset, start + offset + 1] for offset in range(1, 5)],
                "stratum": {
                    "state": item["state"],
                    "year": item["year"],
                    "size_bin": item["size_bin"],
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "title": "NIROPS expanded stratified historical spread validation",
        "seed": 20260730,
        "selection": {
            "source_incident_count": len(rows),
            "selected_incident_count": len(selected),
            "criteria": {
                "consecutive_usable_transitions": 5,
                "transition_duration_hours": [6, 36],
                "minimum_maximum_reported_acres": 100,
                "minimum_area_monotonic_fraction": 0.75,
                "stratification": "state and quartile of log maximum reported acres",
            },
            "inventory_csv": str(inventory_path),
        },
        "grid_size": 128,
        "weather_spinup_hours": 1440,
        "incident_weather_source": "hrrr-analysis",
        "buffer_m": 4500.0,
        "spotting_rate": 0.0,
        "ensemble_particles": 12,
        "perimeter_localization_sigma_m": 350.0,
        "parallel_workers": 8,
        "spread_candidates": [
            0.001,
            0.003,
            0.008,
            0.021,
            0.055,
            0.144,
            0.377,
            0.610,
            0.987,
            1.5,
            2.75,
            5.0,
        ],
        "incidents": incidents,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    summary_path = inventory_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "source_incident_count": len(rows),
                "eligible_incident_count": sum(
                    item["longest_usable_transition_run"] >= 5
                    and item["maximum_reported_acres"] >= 100.0
                    and item["area_monotonic_fraction"] >= 0.75
                    for item in rows
                ),
                "selected_incident_count": len(selected),
                "states": sorted({item["state"] for item in selected}),
                "years": sorted({item["year"] for item in selected}),
                "selected_incidents": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--count", type=int, default=36)
    args = parser.parse_args()
    rows = inventory(args.shapefile)
    selected = select(rows, args.count)
    write_outputs(
        rows,
        selected,
        manifest_path=args.manifest,
        inventory_path=args.inventory,
    )
    print(
        json.dumps(
            {
                "inventory": len(rows),
                "selected": len(selected),
                "manifest": str(args.manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
