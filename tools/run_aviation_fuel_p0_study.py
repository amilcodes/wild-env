#!/usr/bin/env python3
"""Summarize aviation closure and paired historical-fuel effects."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from aeolus.config import load_config
from aeolus.data import audit_vehicle_catalog, load_vehicle_catalog

METRICS = {
    "perimeter_iou": ("metrics.iou", 1.0),
    "growth_tolerance_f1": ("growth_tolerance_1_cell.f1", 1.0),
    "boundary_mean_distance_m": (
        "boundary.mean_symmetric_distance_m",
        -1.0,
    ),
    "symmetric_difference_km2": (
        "metrics.symmetric_difference_km2",
        -1.0,
    ),
}
METHODS = (
    "raw_physics",
    "fixed_calibrated_physics",
    "calibrated_physics",
    "calibrated_ensemble",
    "history_calibrated_ensemble",
)


def _nested(value: dict[str, Any], path: str) -> float:
    current: Any = value
    for component in path.split("."):
        current = current[component]
    return float(current)


def _paired_comparison(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    before_records = {
        (
            str(record["incident_code"]),
            str(record["method"]),
            int(record["start_index"]),
            int(record["target_index"]),
        ): record
        for record in before["forecasts"]
    }
    after_records = {
        (
            str(record["incident_code"]),
            str(record["method"]),
            int(record["start_index"]),
            int(record["target_index"]),
        ): record
        for record in after["forecasts"]
    }
    rng = np.random.default_rng(seed)
    methods: dict[str, Any] = {}
    for method in METHODS:
        keys = sorted(key for key in before_records.keys() & after_records.keys() if key[1] == method)
        if not keys:
            continue
        metric_results: dict[str, Any] = {}
        for name, (path, direction) in METRICS.items():
            old = np.asarray(
                [_nested(before_records[key]["forecast"], path) for key in keys],
                dtype=np.float64,
            )
            new = np.asarray(
                [_nested(after_records[key]["forecast"], path) for key in keys],
                dtype=np.float64,
            )
            delta = new - old
            if len(delta):
                bootstrap = np.mean(
                    rng.choice(
                        delta,
                        size=(5000, len(delta)),
                        replace=True,
                    ),
                    axis=1,
                )
                ci = np.quantile(bootstrap, (0.025, 0.975))
            else:
                ci = np.asarray([np.nan, np.nan])
            metric_results[name] = {
                "path": path,
                "direction": "higher_is_better" if direction > 0 else "lower_is_better",
                "paired_forecasts": len(keys),
                "before_mean": float(np.mean(old)) if len(old) else None,
                "after_mean": float(np.mean(new)) if len(new) else None,
                "mean_delta_after_minus_before": (float(np.mean(delta)) if len(delta) else None),
                "paired_improvement_fraction": (
                    float(np.mean(direction * delta > 0.0)) if len(delta) else None
                ),
                "paired_delta_ci95": [
                    float(ci[0]),
                    float(ci[1]),
                ],
            }
        methods[method] = {
            "paired_forecasts": len(keys),
            "metrics": metric_results,
        }
    return {
        "design": (
            "Paired by incident, method, start perimeter, and target perimeter. "
            "The manifest, code, weather, observations, and random seeds are "
            "held fixed; the landscape fuel/canopy vintage changes."
        ),
        "front_solver": after.get(
            "front_solver",
            ("adaptive_huygens" if "fixed-parameter" in str(after.get("study", "")) else "unspecified"),
        ),
        "calibration_semantics": after.get(
            "calibration_semantics",
            "read from source result",
        ),
        "methods": methods,
    }


def run(
    *,
    catalog_path: Path,
    operations_config: Path,
    fuel_manifest_path: Path,
    before_results_path: Path,
    after_results_path: Path,
    seed: int,
) -> dict[str, Any]:
    catalog = load_vehicle_catalog(catalog_path)
    fleet = load_config(operations_config).scenario.resources
    fuel_manifest = json.loads(fuel_manifest_path.read_text(encoding="utf-8"))
    before = json.loads(before_results_path.read_text(encoding="utf-8"))
    after = json.loads(after_results_path.read_text(encoding="utf-8"))
    profile_ids = {profile.profile_id for profile in catalog.profiles}
    missing_profiles = sorted(
        {resource.vehicle_profile_id for resource in fleet if resource.vehicle_profile_id not in profile_ids}
    )
    fuel_records = fuel_manifest["incidents"]
    return {
        "schema_version": 1,
        "study": "operational aviation closure and historical-fuel P0",
        "seed": seed,
        "aviation_catalog": audit_vehicle_catalog(catalog),
        "reference_scenario": {
            "config": str(operations_config.resolve()),
            "resource_count": len(fleet),
            "resource_kind_counts": dict(sorted(Counter(resource.kind for resource in fleet).items())),
            "autonomy_counts": dict(sorted(Counter(resource.autonomy_level for resource in fleet).items())),
            "missing_catalog_profiles": missing_profiles,
            "all_resources_traceable": not missing_profiles
            and all(resource.vehicle_profile_id for resource in fleet),
            "field_performance_ready": all(
                resource.performance_evidence_grade in {"flight_manual", "engineering_validated"}
                and resource.performance_surface_path is not None
                for resource in fleet
            ),
        },
        "historical_fuel_reconstruction": {
            "manifest": str(fuel_manifest_path.resolve()),
            "incident_count": len(fuel_records),
            "gate_passes": bool(fuel_manifest["gate_passes"]),
            "mean_fuel_model_changed_fraction": float(
                np.mean([record["statistics"]["fuel_model_changed_fraction"] for record in fuel_records])
            ),
            "mean_burnability_changed_fraction": float(
                np.mean([record["statistics"]["burnability_changed_fraction"] for record in fuel_records])
            ),
            "archive_substitutions_remaining": int(
                sum(bool(record["archive_substitution_remaining"]) for record in fuel_records)
            ),
            "incidents": [
                {
                    "incident_id": record["incident_id"],
                    "fuel_model_changed_fraction": record["statistics"]["fuel_model_changed_fraction"],
                    "burnability_changed_fraction": record["statistics"]["burnability_changed_fraction"],
                    "preferred_version": record["preferred_version"]["version_id"],
                    "selected_version": record["selected_streamable_version"]["version_id"],
                    "archive_substitution_remaining": record["archive_substitution_remaining"],
                }
                for record in fuel_records
            ],
        },
        "paired_hindcast_comparison": _paired_comparison(
            before,
            after,
            seed=seed,
        ),
        "claim_boundary": {
            "supported": [
                "selected aircraft and UAS are tied to current public operator or interagency sources",
                "nominal published values and modeling assumptions are separable",
                (
                    "the reference scenario exercises crewed, remotely "
                    "piloted, and supervised-autonomy resources"
                ),
                "historical fuels pass a pre-incident disturbance-cutoff gate",
            ],
            "not_supported": [
                "vehicle dispatch or payload feasibility under field conditions",
                "approval of autonomous suppressant-dropping aircraft",
                "causal attribution of observed historical spread to fuels alone",
                "operational validation of a learned policy",
            ],
        },
    }


def render(result: dict[str, Any], destination: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#F7F5EF",
            "axes.facecolor": "#F7F5EF",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    audit = result["aviation_catalog"]
    kinds = result["reference_scenario"]["resource_kind_counts"]
    axes[0, 0].bar(
        list(kinds),
        list(kinds.values()),
        color=("#B8663B", "#246B91", "#66765A"),
    )
    axes[0, 0].set_ylabel("Resources")
    axes[0, 0].set_title("A  Selected reference fleet", loc="left")
    axes[0, 0].text(
        0.98,
        0.95,
        "9 traceable profiles\n0 flight-manual closed",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        color="#475569",
    )

    basis = audit["parameter_basis_counts"]
    basis_names = [name for name, count in basis.items() if count]
    axes[0, 1].barh(
        basis_names,
        [basis[name] for name in basis_names],
        color="#8A5B91",
    )
    axes[0, 1].set_xlabel("Simulator parameters")
    axes[0, 1].set_title("B  Parameter evidence basis", loc="left")

    incidents = result["historical_fuel_reconstruction"]["incidents"]
    labels = [item["incident_id"].replace("nirops-", "").split("-")[0].upper() for item in incidents]
    x = np.arange(len(incidents))
    axes[1, 0].bar(
        x - 0.18,
        [item["fuel_model_changed_fraction"] for item in incidents],
        width=0.36,
        label="FBFM40 code",
        color="#B8663B",
    )
    axes[1, 0].bar(
        x + 0.18,
        [item["burnability_changed_fraction"] for item in incidents],
        width=0.36,
        label="Burnability",
        color="#246B91",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel("Changed fraction")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title(
        "C  2025 → time-admissible landscape change",
        loc="left",
    )

    comparison = result["paired_hindcast_comparison"]["methods"]
    method_labels = []
    old_iou = []
    new_iou = []
    for method in (name for name in METHODS if name in comparison):
        metric = comparison[method]["metrics"]["perimeter_iou"]
        if metric["paired_forecasts"]:
            method_labels.append(method.replace("_", "\n"))
            old_iou.append(metric["before_mean"])
            new_iou.append(metric["after_mean"])
    x = np.arange(len(method_labels))
    axes[1, 1].plot(x, old_iou, marker="o", label="2025 fuel vintage")
    axes[1, 1].plot(
        x,
        new_iou,
        marker="s",
        label="Time-admissible fuels",
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(method_labels, fontsize=8)
    axes[1, 1].set_ylabel("Mean held-out perimeter IoU")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title(
        "D  Fixed-parameter screening hindcasts",
        loc="left",
    )

    figure.suptitle(
        "Wildfire aviation closure and historical-fuel validity",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.935,
        (
            "Operational identity is public-source traceable; flight-manual "
            "performance remains an explicit closure item. Fuel effects are paired."
        ),
        color="#475569",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.91))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/aviation/us_wildfire_reference_fleet_v1.json"),
    )
    parser.add_argument(
        "--operations-config",
        type=Path,
        default=Path("configs/aviation/us_wildfire_reference_operations.yaml"),
    )
    parser.add_argument("--fuel-manifest", type=Path, required=True)
    parser.add_argument("--before-results", type=Path, required=True)
    parser.add_argument("--after-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    result = run(
        catalog_path=args.catalog,
        operations_config=args.operations_config,
        fuel_manifest_path=args.fuel_manifest,
        before_results_path=args.before_results,
        after_results_path=args.after_results,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render(result, args.figure or args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
