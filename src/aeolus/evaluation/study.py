"""Reproducible multi-incident historical validation study."""

from __future__ import annotations

import json
import multiprocessing
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeolus.data import (
    IncidentBundle,
    derive_dead_fuel_moisture,
    derive_live_fuel_moisture,
    downscale_weather_to_topography,
    fetch_hrrr_analysis,
    overlay_hrrr_analysis,
    scenario_lonlat_grid,
    trim_weather_forcing,
    write_incident_bundle,
    write_weather_forcing,
)
from aeolus.data.importers import (
    build_landscape_from_services,
    fetch_nasa_power_hourly,
    geojson_bbox,
    load_nirops_perimeters,
)
from aeolus.evaluation.ensemble import (
    calibrate_particle_ensemble,
    run_ensemble_hindcast,
)
from aeolus.evaluation.historical import (
    HindcastJob,
    PerimeterSeries,
    boundary_distance_metrics,
    calibrate_spread_adjustment,
    evaluation_worker_count,
    execute_hindcast_job,
    perimeter_metrics,
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


def _prepare_historical_weather(
    manifest: dict[str, Any],
    landscape: Any,
    *,
    latitude: float,
    longitude: float,
    weather_start: datetime,
    incident_start: datetime | str,
    incident_end: datetime | str,
    cache_directory: Path,
):
    forcing = downscale_weather_to_topography(
        fetch_nasa_power_hourly(
            latitude,
            longitude,
            weather_start,
            incident_end,
        ),
        landscape.elevation_m,
    )
    incident_weather_source = str(manifest.get("incident_weather_source", "nasa-power")).lower()
    if incident_weather_source == "hrrr-analysis":
        target_latitude, target_longitude = scenario_lonlat_grid(
            landscape.metadata,
            landscape.elevation_m.shape,
        )
        hrrr = fetch_hrrr_analysis(
            target_latitude,
            target_longitude,
            incident_start,
            incident_end,
            cache_directory=cache_directory,
        )
        forcing = overlay_hrrr_analysis(
            forcing,
            hrrr,
            background_start=weather_start,
        )
    elif incident_weather_source != "nasa-power":
        raise ValueError("incident_weather_source must be nasa-power or hrrr-analysis")
    forcing = derive_dead_fuel_moisture(forcing)
    forcing = derive_live_fuel_moisture(
        forcing,
        latitude_deg=latitude,
        longitude_deg=longitude,
        start_datetime=weather_start,
    )
    incident_start_datetime = (
        datetime.fromisoformat(incident_start.replace("Z", "+00:00"))
        if isinstance(incident_start, str)
        else incident_start
    )
    incident_end_datetime = (
        datetime.fromisoformat(incident_end.replace("Z", "+00:00"))
        if isinstance(incident_end, str)
        else incident_end
    )
    trim_start_minute = (incident_start_datetime - weather_start).total_seconds() / 60.0
    trim_end_minute = (incident_end_datetime - weather_start).total_seconds() / 60.0
    return trim_weather_forcing(
        forcing,
        start_minute=trim_start_minute,
        end_minute=trim_end_minute,
        rebase=True,
    )


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
            weather_start = datetime.fromisoformat(start.replace("Z", "+00:00")) - timedelta(
                hours=float(manifest.get("weather_spinup_hours", 336.0))
            )
            west, south, east, north = bbox
            latitude = (south + north) / 2.0
            longitude = (west + east) / 2.0
            weather = _prepare_historical_weather(
                manifest,
                landscape,
                latitude=latitude,
                longitude=longitude,
                weather_start=weather_start,
                incident_start=start,
                incident_end=end,
                cache_directory=bundle_root / "provenance" / "hrrr-cache",
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
                    {
                        "name": "NASA POWER hourly MERRA-2 meteorology",
                        "weather_spinup_hours": float(manifest.get("weather_spinup_hours", 336.0)),
                        "dead_fuel_moisture_model": ("WRF-SFIRE-compatible equilibrium time-lag"),
                        "live_fuel_moisture_model": "NFDRS-v4-style growing-season index",
                        "microclimate_model": ("terrain lapse-rate and conserved vapor pressure"),
                        "incident_weather_source": str(manifest.get("incident_weather_source", "nasa-power")),
                    },
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
    (root / "prepared.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def refresh_prepared_weather(
    manifest_path: str | Path,
    prepared_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Clone prepared incidents and rebuild forcing with moisture spin-up."""

    manifest = load_manifest(manifest_path)
    source_root = Path(prepared_root)
    output_root = Path(destination)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prepared-data destination: {output_root}")
    shutil.copytree(source_root, output_root)
    prepared: list[dict[str, Any]] = []
    spinup_hours = float(manifest.get("weather_spinup_hours", 336.0))
    for specification in manifest["incidents"]:
        code = str(specification["incident_code"])
        bundle = IncidentBundle.load(output_root / _slug(code))
        properties = bundle.item["properties"]
        start = datetime.fromisoformat(str(properties["start_datetime"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(properties["end_datetime"]).replace("Z", "+00:00"))
        weather_start = start - timedelta(hours=spinup_hours)
        west, south, east, north = bundle.bbox
        latitude = (south + north) / 2.0
        longitude = (west + east) / 2.0
        landscape = bundle.scenario_bundle()
        forcing = _prepare_historical_weather(
            manifest,
            landscape,
            latitude=latitude,
            longitude=longitude,
            weather_start=weather_start,
            incident_start=start,
            incident_end=end,
            cache_directory=bundle.root / "provenance" / "hrrr-cache",
        )
        weather_path = bundle.asset_path("weather")
        if weather_path is None:
            raise KeyError(f"prepared incident {code} has no weather asset")
        write_weather_forcing(
            weather_path,
            forcing,
            start_datetime=start,
        )
        item = dict(bundle.item)
        item_properties = dict(item["properties"])
        sources = list(item_properties.get("aeolus:sources", []))
        sources.append(
            {
                "name": "Aeolus historical-forcing refresh",
                "background": "NASA POWER hourly MERRA-2 meteorology",
                "weather_spinup_hours": spinup_hours,
                "dead_fuel_moisture_model": ("WRF-SFIRE-compatible equilibrium time-lag"),
                "live_fuel_moisture_model": "NFDRS-v4-style growing-season index",
                "microclimate_model": ("terrain lapse-rate and conserved vapor pressure; wind unchanged"),
                "incident_weather_source": str(manifest.get("incident_weather_source", "nasa-power")),
            }
        )
        item_properties["aeolus:sources"] = sources
        item["properties"] = item_properties
        (bundle.root / "item.json").write_text(
            json.dumps(item, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        refreshed = IncidentBundle.load(bundle.root)
        series = PerimeterSeries.from_incident(refreshed)
        prepared.append(
            {
                "incident_code": code,
                "bundle": str(refreshed.root.resolve()),
                "frames": len(series.frames),
                "cell_size_m": series.cell_size_m,
                "bbox": list(refreshed.bbox),
                "weather_spinup_hours": spinup_hours,
                "weather_samples": len(forcing.minute),
            }
        )
    result = {
        "schema_version": 1,
        "manifest": str(Path(manifest_path).resolve()),
        "source_prepared_root": str(source_root.resolve()),
        "incidents": prepared,
    }
    (output_root / "prepared.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
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
        "requested_minutes": round((target.timestamp - start.timestamp).total_seconds() / 60.0),
        "metrics": perimeter_metrics(predicted, target.mask, series.cell_size_m),
        "growth_metrics": perimeter_metrics(no_growth, observed_growth, series.cell_size_m),
        "perimeter_tolerance_1_cell": tolerance_metrics(predicted, target.mask, radius_cells=1),
        "growth_tolerance_1_cell": tolerance_metrics(no_growth, observed_growth, radius_cells=1),
        "boundary": boundary_distance_metrics(predicted, target.mask, series.cell_size_m),
    }


def _compact_forecast(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "episode",
            "prediction_mask",
            "arrival_time_min",
            "probability",
            "arrival_time_mean",
            "arrival_time_std",
        }
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
        name: [record for record in records if record["incident_code"] == name] for name in incident_names
    }
    for index in range(bootstrap_samples):
        sampled = rng.choice(incident_names, size=len(incident_names), replace=True)
        sample_values = [_metric(record, metric) for name in sampled for record in by_incident[str(name)]]
        means[index] = np.mean(sample_values)
    return {
        "n_forecasts": int(values.size),
        "n_incidents": len(incident_names),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


_PROBABILISTIC_METRICS = (
    "probabilistic_metrics.brier_score",
    "probabilistic_metrics.balanced_brier_score",
    "probabilistic_metrics.log_score",
    "active_domain_probabilistic_metrics.brier_score",
    "active_domain_probabilistic_metrics.balanced_brier_score",
    "persistence_probabilistic_metrics.brier_score",
    "persistence_probabilistic_metrics.balanced_brier_score",
    "persistence_active_domain_probabilistic_metrics.brier_score",
    "persistence_active_domain_probabilistic_metrics.balanced_brier_score",
)


def _probabilistic_summary_blocks(
    ensemble_records: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float | str],
]:
    active_records = [
        record
        for record in ensemble_records
        if record["forecast"]["growth_metrics"]["observed_area_km2"] > 0.0
    ]

    def summarize(
        records: list[dict[str, Any]],
        seed_offset: int,
    ) -> dict[str, dict[str, float]]:
        return {
            metric: _summary(
                records,
                metric,
                seed=seed
                + seed_offset
                + sum((index + 1) * byte for index, byte in enumerate(metric.encode())),
            )
            for metric in _PROBABILISTIC_METRICS
        }

    all_summaries = summarize(ensemble_records, 200_000)
    active_summaries = summarize(active_records, 300_000)

    def aggregate_skill(
        block: dict[str, dict[str, float]],
        model_key: str,
        persistence_key: str,
    ) -> float:
        model = float(block[model_key]["mean"])
        reference = float(block[persistence_key]["mean"])
        return float(1.0 - model / reference)

    skill = {
        "whole_domain_brier_skill": aggregate_skill(
            active_summaries,
            "probabilistic_metrics.brier_score",
            "persistence_probabilistic_metrics.brier_score",
        ),
        "whole_domain_balanced_brier_skill": aggregate_skill(
            active_summaries,
            "probabilistic_metrics.balanced_brier_score",
            "persistence_probabilistic_metrics.balanced_brier_score",
        ),
        "active_domain_brier_skill": aggregate_skill(
            active_summaries,
            "active_domain_probabilistic_metrics.brier_score",
            "persistence_active_domain_probabilistic_metrics.brier_score",
        ),
        "active_domain_balanced_brier_skill": aggregate_skill(
            active_summaries,
            "active_domain_probabilistic_metrics.balanced_brier_score",
            ("persistence_active_domain_probabilistic_metrics.balanced_brier_score"),
        ),
        "definition": (
            "1 - aggregate posterior score / aggregate persistence score; "
            "computed over intervals with observed growth"
        ),
    }
    return all_summaries, active_summaries, skill


def refresh_study_summaries(path: str | Path) -> dict[str, Any]:
    """Recompute derived probability summaries without rerunning hindcasts."""

    result_path = Path(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ensemble_methods = (
        "calibrated_ensemble",
        "history_calibrated_ensemble",
    )
    ensemble_records_by_method = {
        method: [record for record in result["forecasts"] if record["method"] == method]
        for method in ensemble_methods
    }
    ensemble_records_by_method = {
        method: records for method, records in ensemble_records_by_method.items() if records
    }
    for records in ensemble_records_by_method.values():
        for record in records:
            forecast = record["forecast"]
            if "requested_minutes" not in forecast:
                start = datetime.fromisoformat(forecast["start_time"])
                target = datetime.fromisoformat(forecast["target_time"])
                forecast["requested_minutes"] = max(
                    1,
                    round((target - start).total_seconds() / 60.0),
                )
    probability_by_method: dict[str, Any] = {}
    active_probability_by_method: dict[str, Any] = {}
    skill_by_method: dict[str, Any] = {}
    for method, records in ensemble_records_by_method.items():
        probability, active_probability, skill = _probabilistic_summary_blocks(
            records,
            seed=int(result["manifest"]["seed"]),
        )
        probability_by_method[method] = probability
        active_probability_by_method[method] = active_probability
        skill_by_method[method] = skill
    baseline_method = "calibrated_ensemble"
    result["probabilistic_summaries_by_method"] = probability_by_method
    result["probabilistic_active_growth_summaries_by_method"] = active_probability_by_method
    result["probabilistic_skill_against_persistence_by_method"] = skill_by_method
    result["probabilistic_summaries"] = probability_by_method[baseline_method]
    result["probabilistic_active_growth_summaries"] = active_probability_by_method[baseline_method]
    result["probabilistic_skill_against_persistence"] = skill_by_method[baseline_method]
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def run_study(
    manifest_path: str | Path,
    prepared_root: str | Path,
    destination: str | Path,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run a study and always close its worker processes."""

    manifest = load_manifest(manifest_path)
    parallel_workers = evaluation_worker_count(
        1024,
        (int(workers) if workers is not None else int(manifest.get("parallel_workers", 8))),
    )
    process_executor = ProcessPoolExecutor(
        max_workers=parallel_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        return _run_study(
            manifest_path,
            prepared_root,
            destination,
            process_executor=process_executor,
            parallel_workers=parallel_workers,
        )
    finally:
        process_executor.shutdown(wait=True, cancel_futures=True)


def _run_study(
    manifest_path: str | Path,
    prepared_root: str | Path,
    destination: str | Path,
    *,
    process_executor: ProcessPoolExecutor,
    parallel_workers: int,
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
        print(
            f"[study] incident {incident_index + 1}/{len(manifest['incidents'])}: {code}",
            file=sys.stderr,
            flush=True,
        )
        incident = IncidentBundle.load(root / _slug(code))
        series = PerimeterSeries.from_incident(incident)
        calibration_start, calibration_target = (int(value) for value in specification["calibration_pair"])
        validation_pairs = [(int(pair[0]), int(pair[1])) for pair in specification["validation_pairs"]]
        candidates = sorted(
            {
                *(float(value) for value in manifest["spread_candidates"]),
                *(float(value) for value in specification.get("additional_spread_candidates", [])),
            }
        )
        max_minutes = max(
            round((series.frames[target].timestamp - series.frames[start].timestamp).total_seconds() / 60.0)
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
            parallel_workers=parallel_workers,
            executor=process_executor,
        )
        ensemble_calibration = calibrate_particle_ensemble(
            config,
            series,
            no_aerial_action,
            start_index=calibration_start,
            target_index=calibration_target,
            spread_candidates=candidates,
            particle_count=int(manifest.get("ensemble_particles", 12)),
            seed=int(manifest["seed"]) + incident_index * 15485863,
            localization_sigma_m=(
                float(manifest["perimeter_localization_sigma_m"])
                if "perimeter_localization_sigma_m" in manifest
                else None
            ),
            parallel_workers=parallel_workers,
            executor=process_executor,
        )
        print(
            (
                f"[study] calibrated {code}: scalar="
                f"{calibration['selected_spread_adjustment']:.4g}, "
                f"ensemble_ess={ensemble_calibration['effective_sample_size']:.2f}"
            ),
            file=sys.stderr,
            flush=True,
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
                "calibration_growth_metrics": calibration["calibration_growth_metrics"],
                "candidate_trials": calibration["candidate_trials"],
                "ensemble": ensemble_calibration,
            }
        )
        for pair_index, (start_index, target_index) in enumerate(validation_pairs):
            print(
                (
                    f"[study] {code} held-out interval "
                    f"{pair_index + 1}/{len(validation_pairs)} "
                    f"({start_index}->{target_index})"
                ),
                file=sys.stderr,
                flush=True,
            )
            persistence = _persistence_result(series, start_index, target_index)
            capture_prediction = pair_index == len(validation_pairs) - 1
            raw_future = process_executor.submit(
                execute_hindcast_job,
                HindcastJob(
                    config=raw_config,
                    series=series,
                    policy=no_aerial_action,
                    start_index=start_index,
                    target_index=target_index,
                    return_prediction=capture_prediction,
                ),
            )
            calibrated_future = process_executor.submit(
                execute_hindcast_job,
                HindcastJob(
                    config=calibrated_config,
                    series=series,
                    policy=no_aerial_action,
                    start_index=start_index,
                    target_index=target_index,
                    return_prediction=capture_prediction,
                ),
            )
            history_raw_future = process_executor.submit(
                execute_hindcast_job,
                HindcastJob(
                    config=raw_config,
                    series=series,
                    policy=no_aerial_action,
                    start_index=start_index,
                    target_index=target_index,
                    return_prediction=capture_prediction,
                    use_arrival_history=True,
                ),
            )
            history_calibrated_future = process_executor.submit(
                execute_hindcast_job,
                HindcastJob(
                    config=calibrated_config,
                    series=series,
                    policy=no_aerial_action,
                    start_index=start_index,
                    target_index=target_index,
                    return_prediction=capture_prediction,
                    use_arrival_history=True,
                ),
            )
            ensemble = run_ensemble_hindcast(
                config,
                series,
                no_aerial_action,
                start_index=start_index,
                target_index=target_index,
                particles=ensemble_calibration["particles"],
                weights=ensemble_calibration["posterior_weights"],
                return_probability=True,
                parallel_workers=parallel_workers,
                executor=process_executor,
            )
            history_ensemble = run_ensemble_hindcast(
                config,
                series,
                no_aerial_action,
                start_index=start_index,
                target_index=target_index,
                particles=ensemble_calibration["particles"],
                weights=ensemble_calibration["posterior_weights"],
                return_probability=True,
                use_arrival_history=True,
                parallel_workers=parallel_workers,
                executor=process_executor,
            )
            raw = raw_future.result()
            calibrated = calibrated_future.result()
            history_raw = history_raw_future.result()
            history_calibrated = history_calibrated_future.result()
            key = _slug(code).replace("-", "_")
            probability_key = f"{key}_{start_index}_{target_index}_ensemble_probability"
            examples[probability_key] = ensemble["probability"]
            history_probability_key = f"{key}_{start_index}_{target_index}_history_ensemble_probability"
            examples[history_probability_key] = history_ensemble["probability"]
            for method, forecast in (
                ("persistence", persistence),
                ("raw_physics", raw),
                ("history_raw_physics", history_raw),
                ("calibrated_physics", calibrated),
                ("history_calibrated_physics", history_calibrated),
                ("calibrated_ensemble", ensemble),
                ("history_calibrated_ensemble", history_ensemble),
            ):
                records.append(
                    {
                        "incident_code": code,
                        "method": method,
                        "start_index": start_index,
                        "target_index": target_index,
                        "spread_adjustment": (adjustment if "calibrated" in method else None),
                        "forecast": _compact_forecast(forecast),
                    }
                )
            if pair_index == len(validation_pairs) - 1:
                landscape = incident.scenario_bundle()
                examples[f"{key}_elevation"] = landscape.elevation_m
                examples[f"{key}_fuel_model"] = landscape.fuel_model_number
                examples[f"{key}_start"] = series.frames[start_index].mask
                examples[f"{key}_observed"] = series.frames[target_index].mask
                examples[f"{key}_raw"] = raw.pop("prediction_mask")
                raw.pop("arrival_time_min")
                examples[f"{key}_calibrated"] = calibrated.pop("prediction_mask")
                calibrated.pop("arrival_time_min")
                examples[f"{key}_history_raw"] = history_raw.pop("prediction_mask")
                history_raw.pop("arrival_time_min")
                examples[f"{key}_history_calibrated"] = history_calibrated.pop("prediction_mask")
                history_calibrated.pop("arrival_time_min")
                examples[f"{key}_ensemble_probability"] = ensemble.pop("probability")
                examples[f"{key}_ensemble_arrival_mean"] = ensemble.pop("arrival_time_mean")
                examples[f"{key}_ensemble_arrival_std"] = ensemble.pop("arrival_time_std")
                examples[f"{key}_history_ensemble_probability"] = history_ensemble.pop("probability")
                examples[f"{key}_history_ensemble_arrival_mean"] = history_ensemble.pop("arrival_time_mean")
                examples[f"{key}_history_ensemble_arrival_std"] = history_ensemble.pop("arrival_time_std")
            else:
                ensemble.pop("probability")
                ensemble.pop("arrival_time_mean")
                ensemble.pop("arrival_time_std")
                history_ensemble.pop("probability")
                history_ensemble.pop("arrival_time_mean")
                history_ensemble.pop("arrival_time_std")

    process_executor.shutdown(wait=True)
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
    methods = (
        "persistence",
        "raw_physics",
        "history_raw_physics",
        "calibrated_physics",
        "history_calibrated_physics",
        "calibrated_ensemble",
        "history_calibrated_ensemble",
    )
    for method in methods:
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
                + sum((index + 1) * byte for index, byte in enumerate(metric.encode())),
            )
            for metric in (
                "growth_metrics.iou",
                "growth_tolerance_1_cell.f1",
            )
        }
    probability_by_method: dict[str, Any] = {}
    active_probability_by_method: dict[str, Any] = {}
    probabilistic_skill_by_method: dict[str, Any] = {}
    for method in ("calibrated_ensemble", "history_calibrated_ensemble"):
        ensemble_records = [record for record in records if record["method"] == method]
        (
            probability_by_method[method],
            active_probability_by_method[method],
            probabilistic_skill_by_method[method],
        ) = _probabilistic_summary_blocks(
            ensemble_records,
            seed=int(manifest["seed"]),
        )
    probabilistic_summaries = probability_by_method["calibrated_ensemble"]
    probabilistic_active_growth_summaries = active_probability_by_method["calibrated_ensemble"]
    probabilistic_skill = probabilistic_skill_by_method["calibrated_ensemble"]
    result = {
        "schema_version": 1,
        "study": "NIROPS multi-incident held-out spread hindcast",
        "manifest": manifest,
        "calibrations": calibrations,
        "forecasts": records,
        "summaries": summaries,
        "active_growth_summaries": active_growth_summaries,
        "probabilistic_summaries": probabilistic_summaries,
        "probabilistic_active_growth_summaries": (probabilistic_active_growth_summaries),
        "probabilistic_skill_against_persistence": probabilistic_skill,
        "probabilistic_summaries_by_method": probability_by_method,
        "probabilistic_active_growth_summaries_by_method": (active_probability_by_method),
        "probabilistic_skill_against_persistence_by_method": (probabilistic_skill_by_method),
        "interpretation_constraints": [
            "each forecast starts from the observed perimeter at its own start time",
            "historical branches use only the perimeter preceding forecast initialization",
            "only the designated earlier interval is used to select the spread adjustment",
            "NASA POWER is coarse reanalysis forcing rather than incident-station weather",
            "observed growth includes unobserved historical suppression effects",
            "no historical aerial-drop sequence is available for causal policy scoring",
        ],
    }
    (output / "historical_validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output / "historical_validation_examples.npz", **examples)
    return result
