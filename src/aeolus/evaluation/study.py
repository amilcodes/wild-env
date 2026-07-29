"""Reproducible multi-incident historical validation study."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle, write_incident_bundle, write_weather_forcing
from aeolus.data.importers import (
    build_landscape_from_services,
    fetch_nasa_power_hourly,
    geojson_bbox,
    load_nirops_perimeters,
)
from aeolus.evaluation.historical import (
    PerimeterSeries,
    boundary_distance_metrics,
    calibrate_spread_adjustment,
    perimeter_metrics,
    run_hindcast,
    tolerance_metrics,
)
from aeolus.policies import no_aerial_action
from aeolus.workflows import scenario_from_incident


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("historical study manifest requires schema_version 1")
    if not value.get("incidents"):
        raise ValueError("historical study manifest contains no incidents")
    return value


def prepare_study(
    manifest_path: str | Path,
    source_shapefile: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Build aligned NIROPS, terrain, fuel, canopy, and weather bundles."""

    manifest = load_manifest(manifest_path)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for specification in manifest["incidents"]:
        code = str(specification["incident_code"])
        bundle_root = root / _slug(code)
        if (bundle_root / "item.json").exists():
            bundle = IncidentBundle.load(bundle_root)
        else:
            perimeters = load_nirops_perimeters(source_shapefile, code)
            bbox = geojson_bbox(perimeters)
            landscape, landscape_path = build_landscape_from_services(
                bbox,
                bundle_root / "provenance",
                size=(int(manifest["grid_size"]), int(manifest["grid_size"])),
                buffer_m=float(manifest["buffer_m"]),
                split="evaluation",
            )
            start = perimeters["features"][0]["properties"]["observed_at"]
            end = perimeters["features"][-1]["properties"]["observed_at"]
            west, south, east, north = bbox
            weather = fetch_nasa_power_hourly(
                (south + north) / 2.0,
                (west + east) / 2.0,
                start,
                end,
            )
            weather_path = write_weather_forcing(
                bundle_root / "provenance" / "nasa_power_weather.nc",
                weather,
                start_datetime=start,
            )
            source = perimeters["aeolus:source"]
            bundle = write_incident_bundle(
                bundle_root,
                incident_id=f"nirops-{_slug(code)}",
                bbox=bbox,
                start_datetime=start,
                end_datetime=end,
                scenario_bundle=landscape,
                perimeter_collection=perimeters,
                source_landscape=landscape_path,
                weather_path=weather_path,
                title=code,
                sources=[
                    {"name": "USFS NIROPS curated progression", **source},
                    {"name": "NASA POWER hourly MERRA-2 meteorology"},
                    {"name": "USGS 3DEP"},
                    {"name": "LANDFIRE 2025"},
                ],
            )
        series = PerimeterSeries.from_incident(bundle)
        prepared.append(
            {
                "incident_code": code,
                "bundle": str(bundle.root.resolve()),
                "frames": len(series.frames),
                "cell_size_m": series.cell_size_m,
                "bbox": list(bundle.bbox),
            }
        )
    result = {
        "schema_version": 1,
        "manifest": str(Path(manifest_path).resolve()),
        "source_shapefile": str(Path(source_shapefile).resolve()),
        "incidents": prepared,
    }
    (root / "prepared.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _persistence_result(
    series: PerimeterSeries,
    start_index: int,
    target_index: int,
) -> dict[str, Any]:
    start = series.frames[start_index]
    target = series.frames[target_index]
    predicted = start.mask
    observed_growth = target.mask & ~start.mask
    no_growth = np.zeros_like(start.mask)
    return {
        "start_time": start.timestamp.isoformat(),
        "target_time": target.timestamp.isoformat(),
        "requested_minutes": round(
            (target.timestamp - start.timestamp).total_seconds() / 60.0
        ),
        "metrics": perimeter_metrics(predicted, target.mask, series.cell_size_m),
        "growth_metrics": perimeter_metrics(
            no_growth, observed_growth, series.cell_size_m
        ),
        "perimeter_tolerance_1_cell": tolerance_metrics(
            predicted, target.mask, radius_cells=1
        ),
        "growth_tolerance_1_cell": tolerance_metrics(
            no_growth, observed_growth, radius_cells=1
        ),
        "boundary": boundary_distance_metrics(
            predicted, target.mask, series.cell_size_m
        ),
    }


def _compact_forecast(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"episode", "prediction_mask"}
    }


def _metric(record: dict[str, Any], key: str) -> float:
    value: Any = record["forecast"]
    for part in key.split("."):
        value = value[part]
    return float(value)


def _summary(
    records: list[dict[str, Any]],
    metric: str,
    *,
    seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, float]:
    values = np.asarray([_metric(record, metric) for record in records])
    incident_names = sorted({str(record["incident_code"]) for record in records})
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    by_incident = {
        name: [record for record in records if record["incident_code"] == name]
        for name in incident_names
    }
    for index in range(bootstrap_samples):
        sampled = rng.choice(incident_names, size=len(incident_names), replace=True)
        sample_values = [
            _metric(record, metric)
            for name in sampled
            for record in by_incident[str(name)]
        ]
        means[index] = np.mean(sample_values)
    return {
        "n_forecasts": int(values.size),
        "n_incidents": len(incident_names),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def run_study(
    manifest_path: str | Path,
    prepared_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Calibrate once per incident and score only later observation intervals."""

    manifest = load_manifest(manifest_path)
    root = Path(prepared_root)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    for incident_index, specification in enumerate(manifest["incidents"]):
        code = str(specification["incident_code"])
        incident = IncidentBundle.load(root / _slug(code))
        series = PerimeterSeries.from_incident(incident)
        calibration_start, calibration_target = (
            int(value) for value in specification["calibration_pair"]
        )
        validation_pairs = [
            (int(pair[0]), int(pair[1]))
            for pair in specification["validation_pairs"]
        ]
        candidates = sorted(
            {
                *(float(value) for value in manifest["spread_candidates"]),
                *(
                    float(value)
                    for value in specification.get(
                        "additional_spread_candidates", []
                    )
                ),
            }
        )
        max_minutes = max(
            round(
                (
                    series.frames[target].timestamp
                    - series.frames[start].timestamp
                ).total_seconds()
                / 60.0
            )
            for start, target in [
                (calibration_start, calibration_target),
                *validation_pairs,
            ]
        )
        config = scenario_from_incident(
            incident,
            seed=int(manifest["seed"]) + incident_index * 7919,
            horizon_min=max_minutes + 3,
            spotting_rate=float(manifest["spotting_rate"]),
        )
        calibration = calibrate_spread_adjustment(
            config,
            series,
            no_aerial_action,
            start_index=calibration_start,
            target_index=calibration_target,
            candidates=candidates,
        )
        adjustment = float(calibration["selected_spread_adjustment"])
        calibrated_config = replace(
            config,
            fire=replace(
                config.fire,
                surface_spread_adjustment=adjustment,
                crown_spread_adjustment=adjustment,
            ),
            terminate_on_escape=False,
        )
        raw_config = replace(config, terminate_on_escape=False)
        calibrations.append(
            {
                "incident_code": code,
                "calibration_pair": [calibration_start, calibration_target],
                "selected_spread_adjustment": adjustment,
                "calibration_metrics": calibration["calibration_metrics"],
                "calibration_growth_metrics": calibration[
                    "calibration_growth_metrics"
                ],
                "candidate_trials": calibration["candidate_trials"],
            }
        )
        for pair_index, (start_index, target_index) in enumerate(validation_pairs):
            persistence = _persistence_result(series, start_index, target_index)
            raw = run_hindcast(
                AeolusSimulator(raw_config),
                series,
                no_aerial_action,
                start_index=start_index,
                target_index=target_index,
                return_prediction=pair_index == len(validation_pairs) - 1,
            )
            calibrated = run_hindcast(
                AeolusSimulator(calibrated_config),
                series,
                no_aerial_action,
                start_index=start_index,
                target_index=target_index,
                return_prediction=pair_index == len(validation_pairs) - 1,
            )
            for method, forecast in (
                ("persistence", persistence),
                ("raw_physics", raw),
                ("calibrated_physics", calibrated),
            ):
                records.append(
                    {
                        "incident_code": code,
                        "method": method,
                        "start_index": start_index,
                        "target_index": target_index,
                        "spread_adjustment": (
                            adjustment if method == "calibrated_physics" else None
                        ),
                        "forecast": _compact_forecast(forecast),
                    }
                )
            if pair_index == len(validation_pairs) - 1:
                key = _slug(code).replace("-", "_")
                landscape = incident.scenario_bundle()
                examples[f"{key}_elevation"] = landscape.elevation_m
                examples[f"{key}_fuel_model"] = landscape.fuel_model_number
                examples[f"{key}_start"] = series.frames[start_index].mask
                examples[f"{key}_observed"] = series.frames[target_index].mask
                examples[f"{key}_raw"] = raw.pop("prediction_mask")
                examples[f"{key}_calibrated"] = calibrated.pop("prediction_mask")

    metrics = (
        "metrics.iou",
        "growth_metrics.iou",
        "growth_tolerance_1_cell.f1",
        "boundary.mean_symmetric_distance_m",
        "boundary.hausdorff_95_m",
        "metrics.symmetric_difference_km2",
    )
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    active_growth_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for method in ("persistence", "raw_physics", "calibrated_physics"):
        method_records = [record for record in records if record["method"] == method]
        active_records = [
            record
            for record in method_records
            if record["forecast"]["growth_metrics"]["observed_area_km2"] > 0.0
        ]
        summaries[method] = {
            metric: _summary(
                method_records,
                metric,
                seed=int(manifest["seed"])
                + sum((index + 1) * byte for index, byte in enumerate(metric.encode())),
            )
            for metric in metrics
        }
        active_growth_summaries[method] = {
            metric: _summary(
                active_records,
                metric,
                seed=int(manifest["seed"])
                + 100_000
                + sum(
                    (index + 1) * byte
                    for index, byte in enumerate(metric.encode())
                ),
            )
            for metric in (
                "growth_metrics.iou",
                "growth_tolerance_1_cell.f1",
            )
        }
    result = {
        "schema_version": 1,
        "study": "NIROPS multi-incident held-out spread hindcast",
        "manifest": manifest,
        "calibrations": calibrations,
        "forecasts": records,
        "summaries": summaries,
        "active_growth_summaries": active_growth_summaries,
        "interpretation_constraints": [
            "each forecast starts from the observed perimeter at its own start time",
            "only the designated earlier interval is used to select the spread adjustment",
            "NASA POWER is coarse reanalysis forcing rather than incident-station weather",
            "observed growth includes unobserved historical suppression effects",
            "no historical aerial-drop sequence is available for causal policy scoring",
        ],
    }
    (output / "historical_validation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output / "historical_validation_examples.npz", **examples)
    return result
