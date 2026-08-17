#!/usr/bin/env python3
"""Measure spatial association between hindcast errors and archived firelines."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyogrio
import rasterio.features
from affine import Affine
from pyproj import Transformer
from shapely import from_wkb
from shapely.ops import transform as transform_geometry

from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import PerimeterSeries


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_firelines(
    dataset: str,
    irwin_id: str,
) -> list[dict[str, Any]]:
    normalized_irwin = "{" + irwin_id.strip("{}").upper() + "}"
    _, table = pyogrio.read_arrow(
        dataset,
        layer="Firelines_Engagement_17_24",
        where=f"IRWINID = '{normalized_irwin}'",
        columns=[
            "IncidentName",
            "IRWINID",
            "LineDateTime",
            "FirelineEngagement",
            "LineLengthGeodesicKM",
        ],
    )
    geometry_name = "Shape"
    return [
        {
            **{name: row[name] for name in table.column_names if name != geometry_name},
            "geometry": from_wkb(row[geometry_name]) if row[geometry_name] is not None else None,
        }
        for row in table.to_pylist()
    ]


def _rasterize(
    lines: list[dict[str, Any]],
    incident: IncidentBundle,
    *,
    buffer_m: float,
    target_time: datetime | None = None,
    engagement: str | None = None,
) -> np.ndarray:
    landscape = incident.scenario_bundle()
    metadata = landscape.metadata
    transform = Affine(*[float(value) for value in metadata["transform"]])
    transformer = Transformer.from_crs(
        "EPSG:4326",
        str(metadata["crs"]),
        always_xy=True,
    )
    geometries = []
    for line in lines:
        geometry = line["geometry"]
        if geometry is None or geometry.is_empty:
            continue
        if engagement is not None and str(line.get("FirelineEngagement")) != engagement:
            continue
        timestamp = _utc(line.get("LineDateTime"))
        if target_time is not None and (timestamp is None or timestamp > target_time):
            continue
        projected = transform_geometry(transformer.transform, geometry)
        geometries.append((projected.buffer(buffer_m), 1))
    if not geometries:
        return np.zeros(landscape.elevation_m.shape, dtype=np.bool_)
    return rasterio.features.rasterize(
        geometries,
        out_shape=landscape.elevation_m.shape,
        transform=transform,
        all_touched=True,
        dtype=np.uint8,
    ).astype(np.bool_)


def _fraction(mask: np.ndarray, context: np.ndarray) -> float | None:
    count = int(mask.sum())
    return float((mask & context).sum() / count) if count else None


def analyze(
    results_path: Path,
    examples_path: Path,
    prepared_root: Path,
    fireline_dataset: str,
    *,
    buffer_m: float,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    specifications = {str(item["incident_code"]): item for item in results["manifest"]["incidents"]}
    incidents: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    with np.load(examples_path, allow_pickle=False) as examples:
        for code, specification in specifications.items():
            incident = IncidentBundle.load(prepared_root / _slug(code))
            series = PerimeterSeries.from_incident(incident)
            name = str(series.frames[0].properties["incident_name"])
            irwin_id = str(series.frames[0].properties["irwin_id"])
            lines = _read_firelines(fireline_dataset, irwin_id)
            all_context = _rasterize(lines, incident, buffer_m=buffer_m)
            engagement_counts = Counter(str(line.get("FirelineEngagement") or "unknown") for line in lines)
            timed = sum(_utc(line.get("LineDateTime")) is not None for line in lines)
            incidents.append(
                {
                    "incident_code": code,
                    "incident_name": name,
                    "irwin_id": irwin_id,
                    "archived_line_features": len(lines),
                    "features_with_line_datetime": timed,
                    "timestamp_coverage_fraction": timed / max(len(lines), 1),
                    "engagement_counts": dict(sorted(engagement_counts.items())),
                    "retrospective_buffer_area_cells": int(all_context.sum()),
                }
            )
            for start_index, target_index in specification["validation_pairs"]:
                start = series.frames[int(start_index)]
                target = series.frames[int(target_index)]
                probability = examples[
                    f"{_key(code)}_{start_index}_{target_index}_history_ensemble_probability"
                ]
                predicted_growth = (probability >= 0.5) & ~start.mask
                observed_growth = target.mask & ~start.mask
                false_positive = predicted_growth & ~target.mask
                true_positive = predicted_growth & observed_growth
                missed_growth = observed_growth & ~predicted_growth
                timed_context = _rasterize(
                    lines,
                    incident,
                    buffer_m=buffer_m,
                    target_time=target.timestamp,
                )
                held_context = _rasterize(
                    lines,
                    incident,
                    buffer_m=buffer_m,
                    engagement="Held",
                )
                burned_context = _rasterize(
                    lines,
                    incident,
                    buffer_m=buffer_m,
                    engagement="Burned Over",
                )
                transitions.append(
                    {
                        "incident_code": code,
                        "start_index": int(start_index),
                        "target_index": int(target_index),
                        "start_time": start.timestamp.isoformat(),
                        "target_time": target.timestamp.isoformat(),
                        "interval_hours": (target.timestamp - start.timestamp).total_seconds() / 3600.0,
                        "predicted_growth_cells": int(predicted_growth.sum()),
                        "observed_growth_cells": int(observed_growth.sum()),
                        "false_positive_growth_cells": int(false_positive.sum()),
                        "true_positive_growth_cells": int(true_positive.sum()),
                        "missed_growth_cells": int(missed_growth.sum()),
                        "false_positive_near_any_archived_line_fraction": _fraction(
                            false_positive,
                            all_context,
                        ),
                        "false_positive_near_timestamped_line_by_target_fraction": _fraction(
                            false_positive,
                            timed_context,
                        ),
                        "false_positive_near_held_line_fraction": _fraction(
                            false_positive,
                            held_context,
                        ),
                        "false_positive_near_burned_over_line_fraction": _fraction(
                            false_positive,
                            burned_context,
                        ),
                        "observed_growth_near_any_archived_line_fraction": _fraction(
                            observed_growth,
                            all_context,
                        ),
                        "missed_growth_near_any_archived_line_fraction": _fraction(
                            missed_growth,
                            all_context,
                        ),
                    }
                )
    retrospective = [
        item["false_positive_near_any_archived_line_fraction"]
        for item in transitions
        if item["false_positive_near_any_archived_line_fraction"] is not None
    ]
    timestamped = [
        item["false_positive_near_timestamped_line_by_target_fraction"]
        for item in transitions
        if item["false_positive_near_timestamped_line_by_target_fraction"] is not None
    ]
    return {
        "schema_version": 1,
        "purpose": "suppression-confounding context audit for unsuppressed historical hindcasts",
        "buffer_m": buffer_m,
        "incidents": incidents,
        "transitions": transitions,
        "summary": {
            "incident_count": len(incidents),
            "transition_count": len(transitions),
            "archived_line_feature_count": sum(item["archived_line_features"] for item in incidents),
            "features_with_line_datetime": sum(item["features_with_line_datetime"] for item in incidents),
            "mean_false_positive_near_any_archived_line_fraction": (
                float(np.mean(retrospective)) if retrospective else None
            ),
            "mean_false_positive_near_timestamped_line_by_target_fraction": (
                float(np.mean(timestamped)) if timestamped else None
            ),
        },
        "interpretation_constraints": [
            "The all-line overlap is retrospective spatial context, not a causal suppression estimate.",
            "LineDateTime is absent for most features and does not establish construction completion time.",
            (
                "Archived line geometry omits water and retardant drops and may "
                "include proposed or indirect line."
            ),
            (
                "A spatial association between forecast error and line does not "
                "identify suppression effectiveness."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("examples", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("fireline_dataset")
    parser.add_argument("output", type=Path)
    parser.add_argument("--buffer-m", type=float, default=300.0)
    args = parser.parse_args()
    analysis = analyze(
        args.results,
        args.examples,
        args.prepared_root,
        args.fireline_dataset,
        buffer_m=args.buffer_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2))


if __name__ == "__main__":
    main()
