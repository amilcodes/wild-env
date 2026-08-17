#!/usr/bin/env python3
"""Render observation, suppression, cadence, and sample-size audit results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def render(context_root: Path, inventory_path: Path, destination: Path) -> None:
    cadence = json.loads((context_root / "observation_cadence.json").read_text(encoding="utf-8"))
    uncertainty = json.loads((context_root / "observation_uncertainty.json").read_text(encoding="utf-8"))
    suppression = json.loads((context_root / "suppression_confounding.json").read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

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
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.09,
        top=0.86,
        hspace=0.38,
        wspace=0.28,
    )

    cadence_axis = axes[0, 0]
    categories = ("All NIROPS", "Current six", "FEDS case")
    blocks = (
        cadence["nirops"]["all_incident_transitions"],
        cadence["nirops"]["current_six_incident_transitions"],
        cadence["feds_case"]["cadence"],
    )
    x = np.arange(3)
    cadence_axis.bar(
        x - 0.18,
        [item["median_hours"] for item in blocks],
        width=0.36,
        color="#397B96",
        label="Median",
    )
    cadence_axis.bar(
        x + 0.18,
        [item["q90_hours"] for item in blocks],
        width=0.36,
        color="#9AB7C5",
        label="90th percentile",
    )
    cadence_axis.set_xticks(x, categories)
    cadence_axis.set_ylabel("Observation interval (hours)")
    cadence_axis.set_title("A  Observation cadence", loc="left", fontweight="bold")
    cadence_axis.legend(frameon=False)

    suppression_axis = axes[0, 1]
    by_incident: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"all": [], "timed": []})
    for item in suppression["transitions"]:
        code = str(item["incident_code"]).split("_")[-1]
        all_value = item["false_positive_near_any_archived_line_fraction"]
        timed_value = item["false_positive_near_timestamped_line_by_target_fraction"]
        if all_value is not None:
            by_incident[code]["all"].append(float(all_value))
        if timed_value is not None:
            by_incident[code]["timed"].append(float(timed_value))
    names = list(by_incident)
    positions = np.arange(len(names))
    suppression_axis.bar(
        positions - 0.18,
        [np.mean(by_incident[name]["all"]) for name in names],
        width=0.36,
        color="#C66A3D",
        label="Any archived line",
    )
    suppression_axis.bar(
        positions + 0.18,
        [np.mean(by_incident[name]["timed"]) if by_incident[name]["timed"] else 0.0 for name in names],
        width=0.36,
        color="#675A9C",
        label="Timestamp-qualified",
    )
    suppression_axis.set_xticks(positions, names, rotation=25, ha="right")
    suppression_axis.set_ylim(0.0, 1.0)
    suppression_axis.set_ylabel("False-positive growth near line")
    suppression_axis.set_title(
        "B  Suppression-confounding context",
        loc="left",
        fontweight="bold",
    )
    suppression_axis.legend(frameon=False)

    uncertainty_axis = axes[1, 0]
    selected_labels = ("hard", "declared_150_m", "declared_350_m", "declared_700_m")
    label_text = ("0", "150", "350", "700")
    summary = {(item["observation_sigma_label"], item["metric"]): item for item in uncertainty["summary"]}
    soft_iou = [summary[(label, "soft_iou")]["mean"] for label in selected_labels]
    coverage = [summary[(label, "predicted_boundary_envelope_coverage")]["mean"] for label in selected_labels]
    uncertainty_axis.plot(
        label_text,
        soft_iou,
        marker="o",
        color="#397B96",
        label="Soft IoU",
    )
    uncertainty_axis.plot(
        label_text,
        coverage,
        marker="o",
        color="#5D9364",
        label="Boundary envelope coverage",
    )
    uncertainty_axis.set_ylim(0.0, 1.0)
    uncertainty_axis.set_xlabel("Declared localization sigma (m)")
    uncertainty_axis.set_ylabel("Mean over 24 transitions")
    uncertainty_axis.set_title(
        "C  Perimeter-error sensitivity",
        loc="left",
        fontweight="bold",
    )
    uncertainty_axis.legend(frameon=False)

    sample_axis = axes[1, 1]
    selected = inventory["selected_incidents"]
    states = Counter(item["state"] for item in selected)
    years = Counter(str(item["year"]) for item in selected)
    state_names = sorted(states)
    state_positions = np.arange(len(state_names))
    sample_axis.bar(
        state_positions,
        [states[name] for name in state_names],
        color="#5C7C91",
        label="Incidents by state",
    )
    sample_axis.set_xticks(state_positions, state_names)
    sample_axis.set_ylabel("Selected incidents")
    sample_axis.set_title("D  Expanded benchmark coverage", loc="left", fontweight="bold")
    inset = sample_axis.inset_axes([0.54, 0.52, 0.43, 0.42])
    year_names = sorted(years)
    year_positions = np.arange(len(year_names))
    inset.bar(
        year_positions,
        [years[name] for name in year_names],
        color="#A97651",
    )
    inset.set_xticks(year_positions, year_names)
    inset.set_title("By source year", fontsize=8)
    inset.tick_params(axis="x", labelrotation=45, labelsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.grid(alpha=0.12)

    figure.suptitle(
        "Historical fidelity: evidence and remaining observation limits",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.93,
        (
            "Cadence and archive coverage are measured; localization sigma is a sensitivity "
            "parameter; archived-line overlap is not a causal suppression estimate."
        ),
        color="#475569",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context_root", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.context_root, args.inventory, args.output)


if __name__ == "__main__":
    main()
