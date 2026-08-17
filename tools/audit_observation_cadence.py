#!/usr/bin/env python3
"""Audit timestamp cadence and duplicate-scene semantics across perimeter sources."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import shapefile
import yaml

from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import PerimeterSeries


def _time(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _summary(intervals: list[float]) -> dict[str, Any]:
    values = np.asarray(intervals, dtype=np.float64)
    return {
        "transition_count": int(len(values)),
        "median_hours": float(np.median(values)),
        "q10_hours": float(np.quantile(values, 0.10)),
        "q90_hours": float(np.quantile(values, 0.90)),
        "fraction_at_most_18_hours": float(np.mean(values <= 18.0)),
        "fraction_at_most_30_hours": float(np.mean(values <= 30.0)),
        "fraction_over_36_hours": float(np.mean(values > 36.0)),
    }


def audit(
    shapefile_path: Path,
    manifest_path: Path,
    feds_bundle_path: Path,
) -> dict[str, Any]:
    selected = {
        str(item["incident_code"])
        for item in yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["incidents"]
    }
    timestamps: dict[str, set[datetime]] = defaultdict(set)
    reader = shapefile.Reader(str(shapefile_path))
    raw_nirops_features = 0
    for record in reader.iterRecords():
        properties = record.as_dict()
        timestamps[str(properties["Incident_C"])].add(_time(properties["UTC"]))
        raw_nirops_features += 1
    all_intervals: list[float] = []
    selected_intervals: list[float] = []
    for code, times in timestamps.items():
        ordered = sorted(times)
        intervals = [
            (right - left).total_seconds() / 3600.0
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        all_intervals.extend(intervals)
        if code in selected:
            selected_intervals.extend(intervals)

    feds = IncidentBundle.load(feds_bundle_path)
    raw_feds = feds.perimeter_collection()["features"]
    series = PerimeterSeries.from_incident(feds)
    feds_intervals = [
        (right.timestamp - left.timestamp).total_seconds() / 3600.0
        for left, right in zip(series.frames[:-1], series.frames[1:], strict=True)
    ]
    return {
        "schema_version": 1,
        "nirops": {
            "raw_feature_count": raw_nirops_features,
            "incident_count": len(timestamps),
            "all_incident_transitions": _summary(all_intervals),
            "current_six_incident_transitions": _summary(selected_intervals),
        },
        "feds_case": {
            "incident_id": feds.incident_id,
            "raw_feature_count": len(raw_feds),
            "unique_nonempty_rasterized_frames": len(series.frames),
            "coalesced_duplicate_features": sum(
                frame.properties["coalesced_duplicate_features"] for frame in series.frames
            ),
            "cadence": _summary(feds_intervals),
            "interval_hours": feds_intervals,
            "time_semantics": "FEDS 12-hour local-solar-time bins",
        },
        "interpretation_constraints": [
            (
                "NIROPS acquisition intervals are irregular and observation times "
                "do not equal spread event times."
            ),
            "FEDS perimeters are VIIRS-derived modeled products, not independent high-resolution truth.",
            "Features in one FEDS timestamp are scene fragments and are unioned before evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("feds_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.shapefile, args.manifest, args.feds_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
