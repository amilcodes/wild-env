#!/usr/bin/env python3
"""Run local mechanism studies for the non-compute-bound P1 controls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from aeolus.config import AirspaceVolumeSpec, ResourceSpec, ScenarioConfig
from aeolus.core.aviation import evaluate_leg_performance
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import ResourceRuntime
from aeolus.data import IncidentBundle, WeatherForcing
from aeolus.evaluation.observation import (
    AcquisitionWindow,
    RasterObservationModel,
    expected_raster_observation_probability,
)
from aeolus.evaluation.protocol import paired_policy_summary
from aeolus.evaluation.validity import (
    assess_fast_kernel_validity,
    assess_historical_fuel_provenance,
)
from aeolus.policies import (
    joint_assignment,
    no_aerial_action,
    rollout_lookahead_diagnostics,
)


def _observation_study(seed: int, samples: int = 800) -> dict[str, Any]:
    yy, xx = np.mgrid[:101, :101]
    arrival = (np.hypot(xx - 50.0, yy - 50.0) - 12.0) * 6.0
    acquisition = AcquisitionWindow(0.0, 120.0, 165.0)
    model = RasterObservationModel(
        acquisition=acquisition,
        localization_sigma_m=0.0,
    )
    acquisition_probability = expected_raster_observation_probability(
        arrival,
        model=model,
        cell_size_m=30.0,
    )
    endpoint_probability = (arrival <= acquisition.end_minute).astype(np.float32)
    midpoint_probability = (arrival <= 0.5 * (acquisition.start_minute + acquisition.end_minute)).astype(
        np.float32
    )
    evaluation = (arrival >= -30.0) & (arrival <= 150.0)
    rng = np.random.default_rng(seed)
    times = rng.uniform(
        acquisition.start_minute,
        acquisition.end_minute,
        samples,
    )
    brier = {
        "acquisition_window": [],
        "end_timestamp": [],
        "midpoint_timestamp": [],
    }
    for minute in times:
        observed = arrival <= minute
        target = observed[evaluation].astype(np.float64)
        for name, probability in (
            ("acquisition_window", acquisition_probability),
            ("end_timestamp", endpoint_probability),
            ("midpoint_timestamp", midpoint_probability),
        ):
            brier[name].append(float(np.mean((probability[evaluation].astype(np.float64) - target) ** 2)))
    summary = {
        name: {
            "mean_brier": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }
        for name, values in brier.items()
    }
    summary["relative_brier_reduction_vs_end_timestamp"] = float(
        1.0 - summary["acquisition_window"]["mean_brier"] / summary["end_timestamp"]["mean_brier"]
    )
    return {
        "monte_carlo_acquisitions": samples,
        "evaluation_cells": int(evaluation.sum()),
        "window": asdict(acquisition),
        "scores": summary,
        "interpretation": (
            "Observed masks were sampled at a uniformly unknown time within "
            "the acquisition window. Brier score is evaluated on the temporal "
            "ambiguity band."
        ),
    }


def _incident_directories(root: Path) -> list[Path]:
    candidates = {
        path.parent for pattern in ("*/item.json", "incidents/*/item.json") for path in root.glob(pattern)
    }
    return sorted(candidates)


def _fuel_audit(prepared_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for directory in _incident_directories(prepared_root):
        incident = IncidentBundle.load(directory)
        start = incident.item["properties"]["start_datetime"]
        assessment = assess_historical_fuel_provenance(
            incident.scenario_bundle(),
            incident_start=start,
        )
        records.append(
            {
                "incident_id": incident.incident_id,
                **assessment.as_dict(),
            }
        )
    statuses = Counter(record["status"] for record in records)
    return {
        "prepared_root": str(prepared_root),
        "incident_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "incidents": records,
        "gate_passes": bool(records) and set(statuses) == {"historically_admissible_by_product_date"},
        "interpretation": (
            "A later product year is a potential leakage flag. Pixel-level "
            "disturbance history is still required to determine whether the "
            "landscape actually changed after each incident."
        ),
    }


def _forcing_regime_audit(prepared_root: Path) -> dict[str, Any]:
    electra = next(
        (
            directory
            for directory in _incident_directories(prepared_root)
            if "electra" in directory.name.lower()
        ),
        None,
    )
    if electra is None:
        return {"available": False}
    incident = IncidentBundle.load(electra)
    landscape = incident.scenario_bundle()
    weather_path = incident.asset_path("weather")
    assert weather_path is not None
    weather = WeatherForcing.load(weather_path)
    cell_size = float(landscape.metadata["cell_size_m"])
    slope_y, slope_x = np.gradient(landscape.elevation_m, cell_size)
    audit = assess_fast_kernel_validity(
        wind_speed_m_s=weather.wind_speed_m_s,
        slope_tan=np.hypot(slope_x, slope_y),
        moisture_dead_1h=(weather.moisture_dead_1h if weather.moisture_dead_1h is not None else 0.075),
        moisture_live_herbaceous=(
            weather.moisture_live_herbaceous if weather.moisture_live_herbaceous is not None else 0.75
        ),
        moisture_live_woody=(
            weather.moisture_live_woody if weather.moisture_live_woody is not None else 0.60
        ),
    )
    return {
        "available": True,
        "incident_id": incident.incident_id,
        **audit,
    }


def _aviation_study(
    performance_surface: Path,
) -> dict[str, Any]:
    spec = ResourceSpec(
        "research_rotorcraft",
        "water",
        55.0,
        2500.0,
        5,
        0,
        120,
        performance_surface_path=str(performance_surface.resolve()),
        maximum_crosswind_m_s=12.0,
    )
    volume = AirspaceVolumeSpec(
        "study-volume",
        ((8.0, 0.0), (12.0, 0.0), (12.0, 20.0), (8.0, 20.0)),
        0.0,
        4000.0,
        start_minute=0,
        end_minute=60,
    )
    elevations = (0.0, 1000.0, 2000.0, 3000.0)
    records: list[dict[str, Any]] = []
    for elevation in elevations:
        runtime = ResourceRuntime(spec, 2.0, 10.0, payload_fraction=0.5)
        leg = evaluate_leg_performance(
            runtime,
            start_xy=(2.0, 10.0),
            end_xy=(7.0, 10.0),
            cell_size_m=100.0,
            elevation_m=np.full((20, 20), elevation),
            air_temperature_c=30.0,
            wind_speed_m_s=6.0,
            wind_from_direction_deg=270.0,
            minute=0.0,
        )
        records.append(
            {
                "elevation_m_msl": elevation,
                **asdict(leg),
                "feasible": leg.feasible,
            }
        )
    crossing_runtime = ResourceRuntime(spec, 2.0, 10.0, payload_fraction=0.5)
    crossing = evaluate_leg_performance(
        crossing_runtime,
        start_xy=(2.0, 10.0),
        end_xy=(18.0, 10.0),
        cell_size_m=100.0,
        elevation_m=np.zeros((20, 20)),
        air_temperature_c=20.0,
        wind_speed_m_s=4.0,
        wind_from_direction_deg=270.0,
        minute=0.0,
        airspace_volumes=(volume,),
    )
    return {
        "performance_surface": str(performance_surface),
        "altitude_sweep": records,
        "airspace_crossing": {
            **asdict(crossing),
            "feasible": crossing.feasible,
        },
        "data_status": (
            "The bundled surface verifies the interface and is explicitly "
            "synthetic. Vehicle-specific use requires reviewed flight-manual data."
        ),
    }


def _run_episode(
    config: ScenarioConfig,
    policy_name: str,
) -> dict[str, Any]:
    simulator = AeolusSimulator(config)
    while not simulator.state.terminated and not simulator.state.truncated:
        if policy_name == "no_aerial":
            actions = no_aerial_action(simulator)
        elif policy_name == "joint_assignment":
            actions = joint_assignment(simulator)
        elif policy_name == "rollout_lookahead":
            actions = rollout_lookahead_diagnostics(
                simulator,
                horizon_decisions=2,
            )["actions"]
        else:  # pragma: no cover
            raise KeyError(policy_name)
        simulator.decision_step(actions)
    weighted_loss = simulator._weighted_loss()
    loss_cost_objective = (
        weighted_loss
        + 0.02 / config.reward_loss_scale * simulator.state.cumulative_cost
        + 0.01 / config.reward_loss_scale * simulator.state.blocked_actions
    )
    return {
        "weighted_loss": weighted_loss,
        "cumulative_cost": simulator.state.cumulative_cost,
        "loss_cost_objective": loss_cost_objective,
        "blocked_actions": simulator.state.blocked_actions,
        "escaped": simulator.state.escaped,
        "contained": simulator.state.contained,
        "minute": simulator.state.minute,
    }


def _planning_study(seed: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    policies = ("no_aerial", "joint_assignment", "rollout_lookahead")
    cases: list[tuple[str, ScenarioConfig]] = []
    for wind in (3.0, 7.0):
        for offset in (0, 1):
            case_seed = seed + int(wind * 1000) + offset * 7919
            config = ScenarioConfig(
                scenario_id=f"planning-wind-{wind:g}-seed-{offset}",
                title="Local planning comparator mechanism case",
                seed=case_seed,
                width=24,
                height=24,
                max_tasks=16,
                horizon_min=24,
                decision_interval_min=3,
                wind_speed_m_s=wind,
                spotting_rate=0.0,
                terminate_on_escape=False,
            )
            cases.append((config.scenario_id, config))
    for case_id, config in cases:
        for policy in policies:
            outcome = _run_episode(replace(config), policy)
            records.append(
                {
                    "case_id": case_id,
                    "seed": config.seed,
                    "policy": policy,
                    **outcome,
                }
            )
    loss_comparison = paired_policy_summary(
        records,
        candidate_policy="rollout_lookahead",
        baseline_policy="joint_assignment",
        metric="weighted_loss",
        lower_is_better=True,
        bootstrap_samples=2000,
        seed=seed + 17,
    )
    objective_comparison = paired_policy_summary(
        records,
        candidate_policy="rollout_lookahead",
        baseline_policy="joint_assignment",
        metric="loss_cost_objective",
        lower_is_better=True,
        bootstrap_samples=2000,
        seed=seed + 23,
    )
    no_action_comparisons = {
        metric: paired_policy_summary(
            records,
            candidate_policy="rollout_lookahead",
            baseline_policy="no_aerial",
            metric=metric,
            lower_is_better=True,
            bootstrap_samples=2000,
            seed=seed + offset,
        )
        for metric, offset in (
            ("weighted_loss", 29),
            ("loss_cost_objective", 31),
        )
    }
    return {
        "cases": len(cases),
        "policies": list(policies),
        "records": records,
        "lookahead_vs_joint_assignment": {
            "terminal_weighted_loss": loss_comparison,
            "configured_loss_cost_objective": objective_comparison,
        },
        "lookahead_vs_no_action": no_action_comparisons,
        "loss_cost_objective_definition": (
            "weighted_loss + (0.02 / reward_loss_scale) * cumulative_cost "
            "+ (0.01 / reward_loss_scale) * blocked_actions; terminal escape "
            "and containment terms are absent because neither occurred"
        ),
        "interpretation": (
            "The paired objectives expose reward calibration separately from "
            "terminal fire loss. Small synthetic mechanism cases do not "
            "establish policy quality or operational effect."
        ),
    }


def _benchmark_protocol_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    incidents = manifest.get("incidents", [])
    required = ("split", "fuel_family", "weather_regime", "ecoregion")
    missing = {
        field: sum(
            field not in incident and field not in incident.get("stratum", {}) for incident in incidents
        )
        for field in required
    }
    return {
        "manifest": str(manifest_path),
        "incident_count": len(incidents),
        "missing_required_fields": missing,
        "states": sorted(
            {
                incident.get("stratum", {}).get("state")
                for incident in incidents
                if incident.get("stratum", {}).get("state") is not None
            }
        ),
        "years": sorted(
            {
                int(incident.get("stratum", {}).get("year"))
                for incident in incidents
                if incident.get("stratum", {}).get("year") is not None
            }
        ),
        "passes_partition_gate": bool(incidents) and all(count == 0 for count in missing.values()),
    }


def _render(result: dict[str, Any], destination: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#F8FAFC",
            "savefig.facecolor": "#F8FAFC",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    figure.subplots_adjust(
        left=0.075,
        right=0.97,
        bottom=0.10,
        top=0.87,
        hspace=0.38,
        wspace=0.27,
    )

    observation = result["observation"]["scores"]
    names = ("acquisition_window", "midpoint_timestamp", "end_timestamp")
    axes[0, 0].bar(
        np.arange(3),
        [observation[name]["mean_brier"] for name in names],
        color=("#246B91", "#8A5B91", "#B8663B"),
    )
    axes[0, 0].set_xticks(np.arange(3))
    axes[0, 0].set_xticklabels(
        ("Window likelihood", "Midpoint", "Bin end"),
    )
    axes[0, 0].set_ylabel("Mean Brier score")
    axes[0, 0].set_title(
        "A  Acquisition-time uncertainty",
        loc="left",
        fontweight="bold",
    )

    status_counts = result["fuel_provenance"]["status_counts"]
    status_names = list(status_counts)
    axes[0, 1].barh(
        np.arange(len(status_names)),
        [status_counts[name] for name in status_names],
        color="#C45B3C",
    )
    axes[0, 1].set_yticks(np.arange(len(status_names)))
    axes[0, 1].set_yticklabels([name.replace("_", " ") for name in status_names])
    axes[0, 1].set_xlabel("Prepared incidents")
    axes[0, 1].set_title(
        "B  Historical fuel provenance",
        loc="left",
        fontweight="bold",
    )

    altitude = result["aviation"]["altitude_sweep"]
    density = [item["density_altitude_m"] for item in altitude]
    payload = [item["maximum_payload_fraction"] for item in altitude]
    speed = [item["true_airspeed_m_s"] for item in altitude]
    axes[1, 0].plot(
        density,
        payload,
        marker="o",
        color="#246B91",
        label="Maximum payload fraction",
    )
    axes[1, 0].set_xlabel("Density altitude (m)")
    axes[1, 0].set_ylabel("Maximum payload fraction")
    twin = axes[1, 0].twinx()
    twin.plot(
        density,
        speed,
        marker="s",
        color="#B8663B",
        label="True airspeed",
    )
    twin.set_ylabel("True airspeed (m/s)")
    lines = axes[1, 0].lines + twin.lines
    axes[1, 0].legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="lower left",
    )
    axes[1, 0].set_title(
        "C  Tactical performance surface",
        loc="left",
        fontweight="bold",
    )

    planning_records = result["planning"]["records"]
    policy_order = ("no_aerial", "joint_assignment", "rollout_lookahead")
    case_ids = sorted({record["case_id"] for record in planning_records})
    x = np.arange(len(policy_order))
    for case_id in case_ids:
        values = [
            next(
                record["weighted_loss"]
                for record in planning_records
                if record["case_id"] == case_id and record["policy"] == policy
            )
            for policy in policy_order
        ]
        axes[1, 1].plot(x, values, marker="o", alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        ("No action", "Joint assignment", "Lookahead"),
    )
    axes[1, 1].set_ylabel("Terminal fire-weighted loss")
    axes[1, 1].set_title(
        "D  Local planning comparator: fire loss",
        loc="left",
        fontweight="bold",
    )
    objective = result["planning"]["lookahead_vs_joint_assignment"]["configured_loss_cost_objective"]
    axes[1, 1].text(
        0.02,
        0.97,
        (
            "Configured loss + cost: "
            f"assignment {objective['baseline_mean']:.2f}, "
            f"lookahead {objective['candidate_mean']:.2f}"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        color="#475569",
        fontsize=8,
    )

    figure.suptitle(
        "Non-compute-bound P1 control study",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.925,
        (
            "Mechanism and audit evidence. Aviation values use an explicit "
            "synthetic interface-verification surface; historical fuel status "
            "is a leakage screen."
        ),
        color="#475569",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def run(
    *,
    prepared_root: Path,
    expanded_manifest: Path,
    performance_surface: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "non-compute-bound P1 mechanism and validity controls",
        "seed": seed,
        "observation": _observation_study(seed),
        "fuel_provenance": _fuel_audit(prepared_root),
        "fire_regime": _forcing_regime_audit(prepared_root),
        "aviation": _aviation_study(performance_surface),
        "planning": _planning_study(seed),
        "benchmark_protocol": _benchmark_protocol_audit(expanded_manifest),
        "interpretation_constraints": [
            "Synthetic observation trials verify likelihood semantics, not sensor calibration.",
            "Fuel-product chronology flags potential leakage without proving pixel-level change.",
            "The aviation surface is synthetic and cannot support vehicle-performance claims.",
            "Planning trials are small internal mechanism cases without learned policies.",
            "Crown and spotting remain mechanism-only regimes until independent calibration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("../outputs/historical-validation-v4-electra"),
    )
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        default=Path("configs/historical_validation_expanded.yaml"),
    )
    parser.add_argument(
        "--performance-surface",
        type=Path,
        default=Path("configs/aviation/generic_research_rotorcraft_v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/noncompute_p1/noncompute_p1_study.json"),
    )
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    result = run(
        prepared_root=args.prepared_root,
        expanded_manifest=args.expanded_manifest,
        performance_surface=args.performance_surface,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    destination = args.figure if args.figure is not None else args.out.with_suffix(".png")
    _render(result, destination)


if __name__ == "__main__":
    main()
