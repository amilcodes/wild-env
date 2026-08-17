#!/usr/bin/env python3
"""Analyze and visualize the corrected historical-accuracy study."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch

from aeolus.data import IncidentBundle, WeatherForcing
from aeolus.evaluation.historical import PerimeterSeries

METHODS = (
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
    "history_raw_physics": "Raw + history",
    "calibrated_physics": "Calibrated",
    "history_calibrated_physics": "Calibrated + history",
    "calibrated_ensemble": "Ensemble",
    "history_calibrated_ensemble": "Ensemble + history",
}
METHOD_COLORS = {
    "persistence": "#687381",
    "raw_physics": "#C66A3D",
    "history_raw_physics": "#E39A73",
    "calibrated_physics": "#397B96",
    "history_calibrated_physics": "#72AFC1",
    "calibrated_ensemble": "#675A9C",
    "history_calibrated_ensemble": "#9A8BC4",
}
INCIDENT_LABELS = {
    "CA-AEU-017769_Electra": "Electra, CA",
    "OR-MAF-022199_CrocketsKnob": "Crockets Knob, OR",
    "AZ-SCA-001418_DryLake": "Dry Lake, AZ",
    "ID-IPF-000447_RidgeCreek": "Ridge Creek, ID",
    "NM-GNF-000382_Davis": "Davis, NM",
    "UT-VLD-000127_Bear": "Bear, UT",
}
PAIRS = (
    ("raw_physics", "history_raw_physics"),
    ("calibrated_physics", "history_calibrated_physics"),
    ("calibrated_ensemble", "history_calibrated_ensemble"),
)
METRICS = (
    "metrics.iou",
    "growth_metrics.iou",
    "growth_tolerance_1_cell.f1",
    "boundary.mean_symmetric_distance_m",
    "boundary.hausdorff_95_m",
    "metrics.symmetric_difference_km2",
    "metrics.area_bias_km2",
)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )


def _metric(record: dict[str, Any], key: str) -> float:
    value: Any = record["forecast"]
    for part in key.split("."):
        value = value[part]
    return float(value)


def _records(results: dict[str, Any], method: str) -> list[dict[str, Any]]:
    return [record for record in results["forecasts"] if record["method"] == method]


def _keyed(results: dict[str, Any], method: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (
            str(record["incident_code"]),
            int(record["start_index"]),
            int(record["target_index"]),
        ): record
        for record in _records(results, method)
    }


def _cluster_bootstrap_delta(
    results: dict[str, Any],
    baseline: str,
    treatment: str,
    metric: str,
    *,
    seed: int,
    samples: int = 5000,
) -> dict[str, Any]:
    left = _keyed(results, baseline)
    right = _keyed(results, treatment)
    if left.keys() != right.keys():
        raise ValueError("paired methods do not share forecast keys")
    incidents = sorted({key[0] for key in left})
    deltas_by_incident = {
        incident: np.asarray(
            [_metric(right[key], metric) - _metric(left[key], metric) for key in left if key[0] == incident],
            dtype=np.float64,
        )
        for incident in incidents
    }
    deltas = np.concatenate(list(deltas_by_incident.values()))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(incidents, size=len(incidents), replace=True)
        draws[index] = np.mean(np.concatenate([deltas_by_incident[str(incident)] for incident in selected]))
    higher_is_better = "iou" in metric or metric.endswith(".f1")
    improved = (
        deltas > 0.0
        if higher_is_better
        else (
            np.abs(np.asarray([_metric(right[key], metric) for key in left]))
            < np.abs(np.asarray([_metric(left[key], metric) for key in left]))
            if metric.endswith("area_bias_km2")
            else deltas < 0.0
        )
    )
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": metric,
        "n_pairs": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "improved_pairs": int(improved.sum()),
    }


def build_analysis(results: dict[str, Any]) -> dict[str, Any]:
    seed = int(results["manifest"]["seed"])
    incidents = [str(item["incident_code"]) for item in results["calibrations"]]
    paired = {
        f"{baseline}__to__{treatment}": {
            metric: _cluster_bootstrap_delta(
                results,
                baseline,
                treatment,
                metric,
                seed=seed + sum((index + 1) * value for index, value in enumerate(metric.encode())),
            )
            for metric in METRICS
        }
        for baseline, treatment in PAIRS
    }
    incident_summary: dict[str, Any] = {}
    for incident in incidents:
        incident_summary[incident] = {}
        for method in METHODS:
            records = [record for record in _records(results, method) if record["incident_code"] == incident]
            incident_summary[incident][method] = {
                metric: float(np.mean([_metric(record, metric) for record in records])) for metric in METRICS
            }
    candidates = [float(value) for value in results["manifest"]["spread_candidates"]]
    calibration = []
    for item in results["calibrations"]:
        selected = float(item["selected_spread_adjustment"])
        calibration.append(
            {
                "incident_code": item["incident_code"],
                "selected_spread_adjustment": selected,
                "at_lower_search_boundary": bool(np.isclose(selected, min(candidates))),
                "at_upper_search_boundary": bool(np.isclose(selected, max(candidates))),
                "raw_effective_sample_size": float(item["ensemble"]["raw_effective_sample_size"]),
                "tempered_effective_sample_size": float(item["ensemble"]["effective_sample_size"]),
                "likelihood_tempering_beta": float(item["ensemble"]["likelihood_tempering_beta"]),
            }
        )
    physics_records = [record for record in results["forecasts"] if record["method"] != "persistence"]
    time_errors = [
        int(record["forecast"]["simulated_minutes"]) - int(record["forecast"]["requested_minutes"])
        for record in physics_records
        if "simulated_minutes" in record["forecast"]
    ]
    return {
        "schema_version": 1,
        "study": results["study"],
        "forecast_records": len(results["forecasts"]),
        "held_out_intervals": len(_records(results, "persistence")),
        "incidents": len(incidents),
        "aggregate_summaries": results["summaries"],
        "active_growth_summaries": results["active_growth_summaries"],
        "probabilistic_skill_against_persistence": (
            results["probabilistic_skill_against_persistence_by_method"]
        ),
        "paired_arrival_history_ablation": paired,
        "incident_summary": incident_summary,
        "calibration_diagnostics": calibration,
        "protocol_checks": {
            "records_per_method": {method: len(_records(results, method)) for method in METHODS},
            "maximum_time_overshoot_min": max(time_errors),
            "minimum_time_overshoot_min": min(time_errors),
            "all_physics_forecasts_reached_requested_time": all(value >= 0 for value in time_errors),
            "all_history_forecasts_are_causal": all(
                int(record["start_index"]) >= 1
                for method in METHODS
                if "history" in method
                for record in _records(results, method)
            ),
        },
    }


def write_interval_table(results: dict[str, Any], destination: Path) -> None:
    fields = (
        "incident_code",
        "method",
        "start_index",
        "target_index",
        "start_time",
        "target_time",
        "requested_minutes",
        *METRICS,
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in results["forecasts"]:
            forecast = record["forecast"]
            writer.writerow(
                {
                    "incident_code": record["incident_code"],
                    "method": record["method"],
                    "start_index": record["start_index"],
                    "target_index": record["target_index"],
                    "start_time": forecast["start_time"],
                    "target_time": forecast["target_time"],
                    "requested_minutes": forecast["requested_minutes"],
                    **{metric: _metric(record, metric) for metric in METRICS},
                }
            )


def _bar_panel(
    ax: plt.Axes,
    results: dict[str, Any],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    means = [float(results["summaries"][method][metric]["mean"]) for method in METHODS]
    lows = [float(results["summaries"][method][metric]["ci95_low"]) for method in METHODS]
    highs = [float(results["summaries"][method][metric]["ci95_high"]) for method in METHODS]
    error = np.asarray(
        [
            [mean - low for mean, low in zip(means, lows, strict=True)],
            [high - mean for mean, high in zip(means, highs, strict=True)],
        ]
    )
    x = np.arange(len(METHODS))
    ax.bar(
        x,
        means,
        color=[METHOD_COLORS[method] for method in METHODS],
        yerr=error,
        capsize=2.5,
        width=0.74,
        error_kw={"elinewidth": 0.9, "ecolor": "#26313B"},
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS[method].replace(" + ", "\n+") for method in METHODS],
        rotation=25,
        ha="right",
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)


def build_summary_figure(results: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    _bar_panel(axes[0, 0], results, "metrics.iou", "A. Cumulative extent", "IoU")
    _bar_panel(
        axes[0, 1],
        results,
        "growth_tolerance_1_cell.f1",
        "B. Advancing-front localization",
        "1-cell-tolerance F1",
    )
    _bar_panel(
        axes[1, 0],
        results,
        "boundary.mean_symmetric_distance_m",
        "C. Boundary displacement",
        "Mean symmetric distance (m)",
    )
    _bar_panel(
        axes[1, 1],
        results,
        "metrics.symmetric_difference_km2",
        "D. Extent disagreement",
        "Symmetric difference (km²)",
    )
    figure.suptitle(
        "Corrected held-out historical hindcasts",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.955,
        "Six incidents, 24 daily transitions. Error bars are incident-cluster bootstrap 95% intervals.",
        color="#475462",
        fontsize=8.8,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.92), h_pad=2.2, w_pad=1.8)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def build_history_ablation_figure(analysis: dict[str, Any], destination: Path) -> None:
    metrics = (
        ("metrics.iou", "Cumulative IoU", 1.0),
        ("growth_tolerance_1_cell.f1", "Growth F1", 1.0),
        ("boundary.mean_symmetric_distance_m", "Boundary distance", -1.0),
        ("metrics.symmetric_difference_km2", "Symmetric difference", -1.0),
    )
    pair_keys = [f"{left}__to__{right}" for left, right in PAIRS]
    labels = ("Raw", "Calibrated", "Ensemble")
    figure, axes = plt.subplots(1, 4, figsize=(12.6, 3.7))
    for ax, (metric, title, sign) in zip(axes, metrics, strict=True):
        values = [
            sign * float(analysis["paired_arrival_history_ablation"][pair_key][metric]["mean_delta"])
            for pair_key in pair_keys
        ]
        lows = [
            sign
            * float(
                analysis["paired_arrival_history_ablation"][pair_key][metric][
                    "ci95_low" if sign > 0 else "ci95_high"
                ]
            )
            for pair_key in pair_keys
        ]
        highs = [
            sign
            * float(
                analysis["paired_arrival_history_ablation"][pair_key][metric][
                    "ci95_high" if sign > 0 else "ci95_low"
                ]
            )
            for pair_key in pair_keys
        ]
        error = np.asarray(
            [
                [value - low for value, low in zip(values, lows, strict=True)],
                [high - value for value, high in zip(values, highs, strict=True)],
            ]
        )
        x = np.arange(3)
        ax.bar(
            x,
            values,
            color=("#D8875E", "#5794AA", "#806FAF"),
            yerr=error,
            capsize=3,
            error_kw={"elinewidth": 0.9, "ecolor": "#26313B"},
        )
        ax.axhline(0.0, color="#26313B", linewidth=0.9)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Improvement from arrival history")
    figure.suptitle(
        "Paired contribution of two-perimeter arrival-history initialization",
        x=0.055,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(w_pad=1.6)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def build_incident_figure(analysis: dict[str, Any], destination: Path) -> None:
    incidents = list(analysis["incident_summary"])
    labels = [INCIDENT_LABELS[incident] for incident in incidents]
    persistence = [analysis["incident_summary"][incident]["persistence"] for incident in incidents]
    model = [analysis["incident_summary"][incident]["history_calibrated_ensemble"] for incident in incidents]
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.8))
    panels = (
        ("metrics.iou", "Cumulative IoU", True),
        ("growth_tolerance_1_cell.f1", "Growth localization F1", True),
        ("boundary.mean_symmetric_distance_m", "Boundary distance (m)", False),
    )
    y = np.arange(len(incidents))
    for ax, (metric, title, _) in zip(axes, panels, strict=True):
        left = [float(item[metric]) for item in persistence]
        right = [float(item[metric]) for item in model]
        ax.plot(left, y, "o", color=METHOD_COLORS["persistence"], label="Persistence")
        ax.plot(
            right,
            y,
            "o",
            color=METHOD_COLORS["history_calibrated_ensemble"],
            label="History ensemble",
        )
        for first, second, row in zip(left, right, y, strict=True):
            ax.plot((first, second), (row, row), color="#B7BFC7", linewidth=1)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", visible=False)
    axes[-1].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Incident-level performance remains heterogeneous",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.93), w_pad=1.5)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _key(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", code.lower()).strip("_")


def build_atlas(
    results: dict[str, Any],
    examples_path: Path,
    destination: Path,
) -> None:
    codes = [str(item["incident_code"]) for item in results["calibrations"]]
    cmap = ListedColormap(["#00000000", "#334155D9", "#3E94C5E8", "#65A85AEB", "#D65C49EB"])
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.3))
    with np.load(examples_path) as values:
        for ax, code in zip(axes.flat, codes, strict=True):
            key = _key(code)
            elevation = values[f"{key}_elevation"]
            start = values[f"{key}_start"].astype(bool)
            observed = values[f"{key}_observed"].astype(bool)
            probability = values[f"{key}_history_ensemble_probability"]
            predicted = probability >= 0.5
            observed_growth = observed & ~start
            predicted_growth = predicted & ~start
            classes = np.zeros(start.shape, dtype=np.uint8)
            classes[start] = 1
            classes[observed_growth & ~predicted_growth] = 2
            classes[observed_growth & predicted_growth] = 3
            classes[predicted_growth & ~observed_growth] = 4
            hillshade = LightSource(azdeg=315, altdeg=35).hillshade(
                elevation.astype(float),
                vert_exag=1.8,
            )
            ax.imshow(hillshade, cmap="gray", vmin=0.15, vmax=1.0)
            ax.imshow(classes, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            record = [
                record
                for record in _records(results, "history_calibrated_ensemble")
                if record["incident_code"] == code
            ][-1]
            forecast = record["forecast"]
            ax.set_title(
                f"{INCIDENT_LABELS[code]}\n"
                f"IoU {forecast['metrics']['iou']:.2f} | "
                f"growth F1 {forecast['growth_tolerance_1_cell']['f1']:.2f}",
                loc="left",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.set_yticks([])
    legend = (
        Patch(facecolor="#334155", label="Initial perimeter"),
        Patch(facecolor="#3E94C5", label="Observed growth only"),
        Patch(facecolor="#65A85A", label="Matched growth"),
        Patch(facecolor="#D65C49", label="Predicted growth only"),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "Final held-out transition by incident: history ensemble at p >= 0.5",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.03, 0.06, 0.99, 0.94), h_pad=1.5, w_pad=1.1)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def build_moisture_figure(
    results: dict[str, Any],
    prepared_root: Path,
    destination: Path,
) -> None:
    codes = [str(item["incident_code"]) for item in results["calibrations"]]
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.7), sharey=True)
    for ax, code in zip(axes.flat, codes, strict=True):
        incident = IncidentBundle.load(prepared_root / re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-"))
        forcing = WeatherForcing.load(incident.asset_path("weather"))
        series = PerimeterSeries.from_incident(incident)
        origin = forcing.time_origin
        assert origin is not None
        days = forcing.minute / (60.0 * 24.0)
        for values, label, color in (
            (forcing.moisture_dead_1h, "1 h", "#D36B44"),
            (forcing.moisture_dead_10h, "10 h", "#477E9B"),
            (forcing.moisture_dead_100h, "100 h", "#6B5C9E"),
        ):
            assert values is not None
            ax.plot(days, np.asarray(values).reshape(len(days), -1).mean(axis=1), label=label, color=color)
        for frame in series.frames:
            offset = (frame.timestamp - origin).total_seconds() / (60.0 * 60.0 * 24.0)
            ax.axvline(offset, color="#303944", linewidth=0.55, alpha=0.35)
        incident_start = (
            datetime.fromisoformat(str(incident.item["properties"]["start_datetime"]).replace("Z", "+00:00"))
            - origin
        ).total_seconds() / (60.0 * 60.0 * 24.0)
        ax.axvspan(days[0], incident_start, color="#D9DEE4", alpha=0.35)
        ax.set_title(INCIDENT_LABELS[code], loc="left", fontweight="bold")
        ax.set_xlabel("Days since forcing origin")
        ax.set_ylabel("Dead-fuel moisture (kg/kg)")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    figure.suptitle(
        "Prognostic fuel-moisture spin-up and incident-period state",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.955,
        "Gray: 14-day spin-up. Vertical lines: NIROPS perimeter timestamps.",
        fontsize=8.6,
        color="#475462",
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.92), h_pad=1.7, w_pad=1.4)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    analysis = build_analysis(results)
    (args.out / "historical_accuracy_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    write_interval_table(results, args.out / "historical_accuracy_intervals.csv")
    build_summary_figure(results, figures / "historical_accuracy_summary.png")
    build_history_ablation_figure(
        analysis,
        figures / "arrival_history_ablation.png",
    )
    build_incident_figure(analysis, figures / "incident_performance.png")
    build_atlas(
        results,
        args.examples,
        figures / "history_ensemble_atlas.png",
    )
    build_moisture_figure(
        results,
        args.prepared_root,
        figures / "fuel_moisture_spinup.png",
    )
    print(
        json.dumps(
            {
                "analysis": str((args.out / "historical_accuracy_analysis.json").resolve()),
                "intervals": str((args.out / "historical_accuracy_intervals.csv").resolve()),
                "figures": sorted(str(path.resolve()) for path in figures.glob("*.png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
