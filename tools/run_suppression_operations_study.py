"""Run paired suppression experiments and held-out arrival-history hindcasts."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeolus.config import DEFAULT_RESOURCES, ResourceSpec, ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import (
    HindcastJob,
    PerimeterSeries,
    execute_hindcast_job,
)
from aeolus.policies import joint_assignment, no_aerial_action
from aeolus.workflows import scenario_from_incident


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _crew_resources() -> tuple[ResourceSpec, ResourceSpec]:
    return (
        ResourceSpec(
            "crew_31",
            "crew",
            1.25,
            0.0,
            0,
            18,
            720,
            line_length_m=600.0,
            line_width_m=1.2,
            line_production_m_min=5.0,
            max_operating_wind_m_s=30.0,
            max_direct_intensity_kw_m=2100.0,
        ),
        ResourceSpec(
            "dozer_14",
            "crew",
            1.0,
            0.0,
            0,
            25,
            720,
            line_length_m=900.0,
            line_width_m=3.2,
            line_production_m_min=9.0,
            max_operating_wind_m_s=30.0,
            max_direct_intensity_kw_m=1500.0,
        ),
    )


def _run_suppression_trial(
    payload: tuple[str, int, float],
) -> dict[str, Any]:
    strategy, seed, wind_speed = payload
    crew_resources = _crew_resources()
    if strategy == "uncontrolled":
        resources = DEFAULT_RESOURCES
        ground_arrival = 10_000
        policy = no_aerial_action
    elif strategy == "aerial_only":
        resources = DEFAULT_RESOURCES
        ground_arrival = 10_000
        policy = joint_assignment
    elif strategy == "integrated_operations":
        resources = (*DEFAULT_RESOURCES, *crew_resources)
        ground_arrival = 10_000
        policy = joint_assignment
    else:  # pragma: no cover
        raise ValueError(strategy)
    config = ScenarioConfig(
        seed=seed,
        width=64,
        height=64,
        cell_size_m=45.0,
        horizon_min=180,
        decision_interval_min=3,
        max_tasks=48,
        wind_speed_m_s=wind_speed,
        wind_direction_deg=35.0,
        wind_variability=0.18,
        air_temperature_c=31.0,
        relative_humidity_pct=21.0,
        spotting_rate=0.0,
        ground_arrival_min=ground_arrival,
        terminate_on_escape=False,
        resources=resources,
        suppression=replace(
            ScenarioConfig().suppression,
            base_reload_bays=1,
        ),
    )
    simulator = AeolusSimulator(config)
    while not simulator.state.terminated and not simulator.state.truncated:
        simulator.decision_step(policy(simulator))
    events = simulator.state.events

    def event_count(name: str) -> int:
        return sum(item["kind"] == name for item in events)

    return {
        "strategy": strategy,
        "seed": seed,
        "wind_speed_m_s": wind_speed,
        "weighted_loss": simulator.episode_record()["weighted_loss"],
        "burned_fraction": float(simulator.state.truth.observed_burned.mean()),
        "active_fraction": float((simulator.state.truth.phase == 1).mean()),
        "escaped": simulator.state.escaped,
        "contained": simulator.state.contained,
        "minute": simulator.state.minute,
        "cumulative_cost": simulator.state.cumulative_cost,
        "cumulative_exposure": simulator.state.cumulative_exposure,
        "blocked_actions": simulator.state.blocked_actions,
        "water_drops": event_count("water_drop"),
        "retardant_drops": event_count("retardant_drop"),
        "line_starts": event_count("line_started"),
        "line_completions": event_count("line_complete"),
        "line_breach_events": event_count("line_breached"),
        "reload_queue_entries": event_count("reload_queued"),
        "constructed_line_cells": int((simulator.state.truth.constructed_line > 0.0).sum()),
        "held_line_cells": int((simulator.state.truth.line_status == 2).sum()),
        "breached_line_cells": int((simulator.state.truth.line_status == 3).sum()),
    }


def _cluster_bootstrap_delta(
    trials: list[dict[str, Any]],
    treatment: str,
    metric: str,
    *,
    seed: int,
    samples: int = 4000,
) -> dict[str, float]:
    pairs: dict[tuple[int, float], float] = {}
    for item in trials:
        key = (int(item["seed"]), float(item["wind_speed_m_s"]))
        sign = 1.0 if item["strategy"] == treatment else -1.0
        if item["strategy"] in {treatment, "uncontrolled"}:
            pairs[key] = pairs.get(key, 0.0) + sign * float(item[metric])
    seeds = sorted({key[0] for key in pairs})
    by_seed = {
        value: [delta for (seed_value, _), delta in pairs.items() if seed_value == value] for value in seeds
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        bootstrap[index] = np.mean(
            [delta for selected_seed in selected for delta in by_seed[int(selected_seed)]]
        )
    point = float(np.mean(list(pairs.values())))
    return {
        "mean_treatment_minus_uncontrolled": point,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "relative_change": point
        / max(
            np.mean([float(item[metric]) for item in trials if item["strategy"] == "uncontrolled"]),
            1.0e-12,
        ),
    }


def _history_jobs(
    manifest: dict[str, Any],
    incident_root: Path,
    baseline: dict[str, Any],
) -> tuple[list[HindcastJob], list[dict[str, Any]]]:
    calibration = {
        item["incident_code"]: float(item["selected_spread_adjustment"]) for item in baseline["calibrations"]
    }
    jobs: list[HindcastJob] = []
    metadata: list[dict[str, Any]] = []
    for incident_index, specification in enumerate(manifest["incidents"]):
        code = str(specification["incident_code"])
        incident = IncidentBundle.load(incident_root / _slug(code))
        series = PerimeterSeries.from_incident(incident)
        validation_pairs = [(int(pair[0]), int(pair[1])) for pair in specification["validation_pairs"]]
        max_minutes = max(
            round((series.frames[target].timestamp - series.frames[start].timestamp).total_seconds() / 60.0)
            for start, target in validation_pairs
        )
        config = scenario_from_incident(
            incident,
            seed=int(manifest["seed"]) + incident_index * 7919,
            horizon_min=max_minutes + 3,
            spotting_rate=float(manifest["spotting_rate"]),
        )
        adjustment = calibration[code]
        config = replace(
            config,
            fire=replace(
                config.fire,
                surface_spread_adjustment=adjustment,
                crown_spread_adjustment=adjustment,
            ),
            terminate_on_escape=False,
        )
        for pair_index, (start, target) in enumerate(validation_pairs):
            jobs.append(
                HindcastJob(
                    config=config,
                    series=series,
                    policy=no_aerial_action,
                    start_index=start,
                    target_index=target,
                    return_prediction=pair_index == len(validation_pairs) - 1,
                    use_arrival_history=True,
                )
            )
            metadata.append(
                {
                    "incident_code": code,
                    "start_index": start,
                    "target_index": target,
                    "spread_adjustment": adjustment,
                }
            )
    return jobs, metadata


def _run_arrival_history_study(
    manifest_path: Path,
    incident_root: Path,
    baseline_path: Path,
    workers: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    jobs, metadata = _history_jobs(manifest, incident_root, baseline)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        forecasts = list(executor.map(execute_hindcast_job, jobs))
    baseline_records = {
        (
            record["incident_code"],
            int(record["start_index"]),
            int(record["target_index"]),
        ): record["forecast"]
        for record in baseline["forecasts"]
        if record["method"] == "calibrated_physics"
    }
    records: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    metric_paths = {
        "perimeter_iou": ("metrics", "iou"),
        "growth_iou": ("growth_metrics", "iou"),
        "growth_tolerance_f1": ("growth_tolerance_1_cell", "f1"),
        "mean_boundary_distance_m": ("boundary", "mean_symmetric_distance_m"),
        "hausdorff_95_m": ("boundary", "hausdorff_95_m"),
    }
    for item, forecast in zip(metadata, forecasts, strict=True):
        key = (
            item["incident_code"],
            item["start_index"],
            item["target_index"],
        )
        baseline_forecast = baseline_records[key]
        values: dict[str, Any] = {**item}
        for name, path in metric_paths.items():
            history_value = float(forecast[path[0]][path[1]])
            baseline_value = float(baseline_forecast[path[0]][path[1]])
            values[f"history_{name}"] = history_value
            values[f"baseline_{name}"] = baseline_value
            values[f"delta_{name}"] = history_value - baseline_value
        values["initialization"] = forecast["initialization"]
        records.append(values)
        if "prediction_mask" in forecast:
            slug = _slug(item["incident_code"]).replace("-", "_")
            examples[f"{slug}_history_prediction"] = forecast.pop("prediction_mask")
            examples[f"{slug}_history_arrival_time"] = forecast.pop("arrival_time_min")
            series = jobs[len(records) - 1].series
            examples[f"{slug}_history_earlier"] = series.frames[item["start_index"] - 1].mask
            examples[f"{slug}_history_start"] = series.frames[item["start_index"]].mask
            examples[f"{slug}_history_observed"] = series.frames[item["target_index"]].mask

    rng = np.random.default_rng(int(manifest["seed"]) + 9929)
    incident_codes = sorted({item["incident_code"] for item in records})
    summary: dict[str, Any] = {}
    for metric in metric_paths:
        deltas = np.asarray([item[f"delta_{metric}"] for item in records])
        bootstrap = np.empty(4000)
        grouped = {
            code: [item for item in records if item["incident_code"] == code] for code in incident_codes
        }
        for index in range(len(bootstrap)):
            selected = rng.choice(
                incident_codes,
                size=len(incident_codes),
                replace=True,
            )
            bootstrap[index] = np.mean(
                [item[f"delta_{metric}"] for code in selected for item in grouped[str(code)]]
            )
        summary[metric] = {
            "n_forecasts": len(records),
            "n_incidents": len(incident_codes),
            "history_mean": float(np.mean([item[f"history_{metric}"] for item in records])),
            "baseline_mean": float(np.mean([item[f"baseline_{metric}"] for item in records])),
            "paired_delta_mean": float(deltas.mean()),
            "paired_delta_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "paired_delta_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "improved_forecasts": int(
                (
                    deltas > 0.0
                    if metric
                    in {
                        "perimeter_iou",
                        "growth_iou",
                        "growth_tolerance_f1",
                    }
                    else deltas < 0.0
                ).sum()
            ),
        }
    return (
        {
            "method": "two-perimeter harmonic arrival history with advancing-front localization",
            "baseline": str(baseline_path.resolve()),
            "records": records,
            "paired_summary": summary,
        },
        examples,
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {key for record in records for key, value in record.items() if not isinstance(value, (dict, list))}
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in record.items() if key in fieldnames} for record in records
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arrival, examples = _run_arrival_history_study(
        args.manifest,
        args.incidents,
        args.baseline,
        args.workers,
    )
    trial_payloads = [
        (strategy, 20260800 + seed, wind)
        for seed in range(args.seeds)
        for wind in (2.0, 6.0, 10.0)
        for strategy in (
            "uncontrolled",
            "aerial_only",
            "integrated_operations",
        )
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        suppression_trials = list(executor.map(_run_suppression_trial, trial_payloads))
    suppression_summary = {
        strategy: {
            metric: float(
                np.mean([record[metric] for record in suppression_trials if record["strategy"] == strategy])
            )
            for metric in (
                "weighted_loss",
                "burned_fraction",
                "escaped",
                "contained",
                "cumulative_cost",
                "cumulative_exposure",
                "water_drops",
                "retardant_drops",
                "line_completions",
                "reload_queue_entries",
                "held_line_cells",
                "breached_line_cells",
            )
        }
        for strategy in (
            "uncontrolled",
            "aerial_only",
            "integrated_operations",
        )
    }
    paired_effects = {
        strategy: {
            metric: _cluster_bootstrap_delta(
                suppression_trials,
                strategy,
                metric,
                seed=20260728
                + sum((index + 1) * value for index, value in enumerate(f"{strategy}:{metric}".encode())),
            )
            for metric in ("weighted_loss", "burned_fraction")
        }
        for strategy in ("aerial_only", "integrated_operations")
    }
    result = {
        "schema_version": 1,
        "study": "suppression operations and coupled-state initialization",
        "configuration": {
            "suppression_seeds": args.seeds,
            "wind_regimes_m_s": [2.0, 6.0, 10.0],
            "workers": args.workers,
            "scenario": asdict(
                ScenarioConfig(
                    seed=20260800,
                    width=64,
                    height=64,
                    cell_size_m=45.0,
                    horizon_min=180,
                    decision_interval_min=3,
                    max_tasks=48,
                    wind_speed_m_s=6.0,
                    wind_direction_deg=35.0,
                    wind_variability=0.18,
                    air_temperature_c=31.0,
                    relative_humidity_pct=21.0,
                    spotting_rate=0.0,
                    ground_arrival_min=10_000,
                    terminate_on_escape=False,
                    resources=(*DEFAULT_RESOURCES, *_crew_resources()),
                    suppression=replace(
                        ScenarioConfig().suppression,
                        base_reload_bays=1,
                    ),
                )
            ),
            "strategy_resources": {
                "uncontrolled": [item.resource_id for item in DEFAULT_RESOURCES],
                "aerial_only": [item.resource_id for item in DEFAULT_RESOURCES],
                "integrated_operations": [
                    item.resource_id for item in (*DEFAULT_RESOURCES, *_crew_resources())
                ],
            },
        },
        "arrival_history": arrival,
        "suppression_operations": {
            "trials": suppression_trials,
            "summary": suppression_summary,
            "paired_effects": paired_effects,
            "interpretation_constraints": [
                "suppression trials are controlled synthetic mechanism experiments",
                "joint assignment is a doctrine-inspired comparator, not an optimal policy",
                "held and breached line are raster engagement outcomes at 45 m resolution",
                "historical incident bundles do not include complete suppression action logs",
            ],
        },
    }
    (args.out / "operations_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        args.out / "suppression_trials.csv",
        suppression_trials,
    )
    _write_csv(
        args.out / "arrival_history_forecasts.csv",
        arrival["records"],
    )
    np.savez_compressed(
        args.out / "arrival_history_examples.npz",
        **examples,
    )
    print(json.dumps({"output": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
