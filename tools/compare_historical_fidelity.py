#!/usr/bin/env python3
"""Compare two historical-study outputs on identical incident forecast pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = (
    "persistence",
    "raw_physics",
    "history_raw_physics",
    "calibrated_physics",
    "history_calibrated_physics",
    "calibrated_ensemble",
    "history_calibrated_ensemble",
)
METHOD_LABELS = {
    "persistence": "Persistence",
    "raw_physics": "Raw",
    "history_raw_physics": "History raw",
    "calibrated_physics": "Calibrated",
    "history_calibrated_physics": "History calibrated",
    "calibrated_ensemble": "Ensemble",
    "history_calibrated_ensemble": "History ensemble",
}
METRICS = {
    "cumulative_iou": ("metrics", "iou"),
    "advancing_front_tolerance_f1_observed_growth": (
        "growth_tolerance_1_cell",
        "f1",
    ),
    "boundary_mean_symmetric_distance_m": (
        "boundary",
        "mean_symmetric_distance_m",
    ),
    "symmetric_difference_km2": ("metrics", "symmetric_difference_km2"),
}


def _nested(record: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _records(
    study: Mapping[str, Any],
    incident_code: str,
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    return {
        (
            str(item["method"]),
            int(item["start_index"]),
            int(item["target_index"]),
        ): item["forecast"]
        for item in study["forecasts"]
        if item["incident_code"] == incident_code
    }


def _calibration(
    study: Mapping[str, Any],
    incident_code: str,
) -> Mapping[str, Any]:
    matches = [item for item in study["calibrations"] if item["incident_code"] == incident_code]
    if len(matches) != 1:
        raise ValueError(f"expected one calibration for {incident_code}, found {len(matches)}")
    match = matches[0]
    return {
        "calibration_pair": match["calibration_pair"],
        "selected_spread_adjustment": match["selected_spread_adjustment"],
        "ensemble_effective_sample_size": match["ensemble"]["effective_sample_size"],
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(np.mean(values))


def compare(
    baseline_path: Path,
    candidate_path: Path,
    incident_code: str,
) -> dict[str, Any]:
    baseline_study = _load(baseline_path)
    candidate_study = _load(candidate_path)
    baseline = _records(baseline_study, incident_code)
    candidate = _records(candidate_study, incident_code)
    common = baseline.keys() & candidate.keys()
    if not common:
        raise ValueError(f"no common forecasts for {incident_code}")

    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        keys = sorted(key for key in common if key[0] == method)
        if not keys:
            continue
        metric_summary: dict[str, Any] = {}
        for metric, path in METRICS.items():
            metric_keys = keys
            if metric == "advancing_front_tolerance_f1_observed_growth":
                metric_keys = [
                    key
                    for key in keys
                    if _nested(
                        candidate[key],
                        ("growth_metrics", "observed_area_km2"),
                    )
                    > 0.0
                ]
            old_values = [_nested(baseline[key], path) for key in metric_keys]
            new_values = [_nested(candidate[key], path) for key in metric_keys]
            old_mean = _mean(old_values)
            new_mean = _mean(new_values)
            metric_summary[metric] = {
                "baseline_mean": old_mean,
                "candidate_mean": new_mean,
                "candidate_minus_baseline": new_mean - old_mean,
                "paired_forecasts": len(metric_keys),
            }
        methods[method] = metric_summary

    probabilistic: dict[str, Any] = {}
    for method in ("calibrated_ensemble", "history_calibrated_ensemble"):
        keys = [
            key
            for key in sorted(common)
            if key[0] == method and _nested(candidate[key], ("growth_metrics", "observed_area_km2")) > 0.0
        ]
        if not keys:
            continue
        for label, path in (
            (
                "active_domain_brier",
                ("active_domain_probabilistic_metrics", "brier_score"),
            ),
            (
                "persistence_active_domain_brier",
                (
                    "persistence_active_domain_probabilistic_metrics",
                    "brier_score",
                ),
            ),
        ):
            baseline_mean = _mean([_nested(baseline[key], path) for key in keys])
            candidate_mean = _mean([_nested(candidate[key], path) for key in keys])
            probabilistic.setdefault(method, {})[label] = {
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "candidate_minus_baseline": candidate_mean - baseline_mean,
                "paired_forecasts_with_observed_growth": len(keys),
            }

    return {
        "schema_version": 1,
        "incident_code": incident_code,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "common_forecast_records": len(common),
        "calibration": {
            "baseline": _calibration(baseline_study, incident_code),
            "candidate": _calibration(candidate_study, incident_code),
        },
        "methods": methods,
        "probabilistic": probabilistic,
        "interpretation_constraints": [
            "The comparison is paired on method, start perimeter, and target perimeter.",
            "The candidate combines HRRR analysis, terrain-conditioned thermodynamics, "
            "spatial dead-fuel moisture, and dynamic live-fuel moisture; this ablation "
            "does not identify the contribution of any one component.",
            "Electra is one incident with four held-out transitions, so differences are "
            "diagnostic rather than population-level estimates.",
            "Calibration uses an earlier perimeter pair from the same incident.",
            "Advancing-front F1 is restricted to transitions with observed growth, "
            "avoiding perfect empty-set scores in no-growth intervals.",
            "Higher cumulative IoU and advancing-front F1 are favorable; lower "
            "boundary distance, symmetric difference, and Brier score are favorable.",
        ],
    }


def render(comparison: Mapping[str, Any], destination: Path) -> None:
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
    methods = [method for method in METHOD_ORDER if method in comparison["methods"]]
    positions = np.arange(len(methods))
    width = 0.36
    panels = (
        ("cumulative_iou", "A  Cumulative perimeter", "IoU", True),
        (
            "advancing_front_tolerance_f1_observed_growth",
            "B  Advancing front",
            "1-cell-tolerant F1\n(observed-growth intervals)",
            True,
        ),
        (
            "boundary_mean_symmetric_distance_m",
            "C  Boundary location",
            "Mean symmetric distance (m)",
            False,
        ),
        (
            "symmetric_difference_km2",
            "D  Cumulative disagreement",
            "Symmetric difference (km²)",
            False,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.18,
        top=0.87,
        hspace=0.38,
        wspace=0.25,
    )
    for axis, (metric, title, ylabel, higher_is_better) in zip(
        axes.flat,
        panels,
        strict=True,
    ):
        baseline = [comparison["methods"][method][metric]["baseline_mean"] for method in methods]
        candidate = [comparison["methods"][method][metric]["candidate_mean"] for method in methods]
        axis.bar(
            positions - width / 2,
            baseline,
            width,
            color="#B8663B",
            label="v3 point forcing",
        )
        axis.bar(
            positions + width / 2,
            candidate,
            width,
            color="#246B91",
            label="v4 historical forcing",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [METHOD_LABELS[method] for method in methods],
            rotation=30,
            ha="right",
        )
        direction = "Higher is better" if higher_is_better else "Lower is better"
        axis.text(
            0.99,
            0.96,
            direction,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#64748B",
        )
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    incident = str(comparison["incident_code"]).split("_")[-1]
    figure.suptitle(
        f"{incident}: paired historical-fidelity ablation",
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
            "Four held-out perimeter transitions per method. Candidate changes are "
            "combined; this comparison does not isolate individual forcing components."
        ),
        color="#475569",
        fontsize=9,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("incident_code")
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    comparison = compare(args.baseline, args.candidate, args.incident_code)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    if args.figure is not None:
        render(comparison, args.figure)


if __name__ == "__main__":
    main()
