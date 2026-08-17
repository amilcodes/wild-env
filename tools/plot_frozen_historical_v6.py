#!/usr/bin/env python3
"""Plot the frozen metric-grid historical benchmark and validity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "persistence": "#495057",
    "retrospective": "#3a86ff",
    "operational_extent": "#8338ec",
    "operational_front": "#ff6b35",
    "raw": "#d62828",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(result: dict, method: str, metric: str) -> float:
    return float(result["summaries"]["test"][method][metric]["incident_weighted_mean"])


def plot(
    retrospective_path: Path,
    operational_path: Path,
    reprojection_path: Path,
    output: Path,
) -> None:
    retrospective = _load(retrospective_path)
    operational = _load(operational_path)
    projection = _load(reprojection_path)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    figure.suptitle(
        "Frozen historical-skill pilot — metric grids and incident holdout",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    incidents = projection["incidents"]
    labels = [item["incident_id"].split("-")[-1].title() for item in incidents]
    ratios = [item["before"]["approximate_ground_area_per_map_area"] for item in incidents]
    ax.bar(labels, ratios, color="#457b9d")
    ax.axhline(1.0, color="#202020", linewidth=1.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("ground area / former map area")
    ax.set_title("A. Projection defect removed")
    ax.tick_params(axis="x", rotation=25)
    ax.text(
        0.02,
        0.04,
        "Former physical areas were 43–122% too large\nwhen map area was read as ground area.",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax = axes[0, 1]
    for result, label, color, marker in (
        (retrospective, "retrospective forcing", COLORS["retrospective"], "o"),
        (operational, "pre-issue HRRR", COLORS["operational_front"], "s"),
    ):
        trials = result["fit"]["trials"]
        x = np.asarray([item["spread_adjustment"] for item in trials])
        iou = np.asarray([item["train_incident_weighted_cumulative_iou"] for item in trials])
        ax.plot(x, iou, marker=marker, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("global spread adjustment")
    ax.set_ylabel("train cumulative IoU")
    ax.set_title("B. Train-only extent selection")
    ax.legend(frameon=True, fontsize=9)
    ax.text(
        0.02,
        0.04,
        "Operational extent selection collapses to 0.01\n(the same forecast as persistence).",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax = axes[1, 0]
    methods = (
        ("Persistence", operational, "persistence", COLORS["persistence"]),
        (
            "Retrospective\nselected",
            retrospective,
            "history_global_extent",
            COLORS["retrospective"],
        ),
        (
            "Operational\nextent",
            operational,
            "history_global_extent",
            COLORS["operational_extent"],
        ),
        (
            "Operational\nfront",
            operational,
            "history_global_front",
            COLORS["operational_front"],
        ),
        ("Operational\nraw", operational, "history_raw_physics", COLORS["raw"]),
    )
    names = [item[0] for item in methods]
    values = [_metric(item[1], item[2], "metrics.iou") for item in methods]
    bars = ax.bar(names, values, color=[item[3] for item in methods])
    ax.set_ylim(0.5, 0.87)
    ax.set_ylabel("test cumulative IoU")
    ax.set_title("C. Unseen 2023 incidents (8 transitions)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    ax = axes[1, 1]
    tradeoff_methods = (
        (
            "Persistence / operational extent",
            operational,
            "persistence",
            COLORS["persistence"],
            (7, -4),
        ),
        (
            "Retrospective selected",
            retrospective,
            "history_global_extent",
            COLORS["retrospective"],
            (7, 8),
        ),
        (
            "Operational front",
            operational,
            "history_global_front",
            COLORS["operational_front"],
            (6, 5),
        ),
        (
            "Operational raw",
            operational,
            "history_raw_physics",
            COLORS["raw"],
            (6, 5),
        ),
    )
    for label, result, method, color, offset in tradeoff_methods:
        boundary = _metric(result, method, "boundary.mean_symmetric_distance_m")
        front = _metric(result, method, "growth_tolerance_1_cell.f1")
        ax.scatter(boundary, front, s=85, color=color, edgecolor="white", zorder=3)
        ax.annotate(
            label,
            (boundary, front),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("test mean symmetric boundary distance (m) ↓")
    ax.set_ylabel("test advancing-front F1 ↑")
    ax.set_title("D. Extent–front tradeoff")
    ax.set_xlim(left=90)
    ax.set_ylim(bottom=-0.01)
    ax.text(
        0.98,
        0.04,
        "Desired direction: upper left",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrospective", type=Path)
    parser.add_argument("operational", type=Path)
    parser.add_argument("reprojection_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(
        args.retrospective,
        args.operational,
        args.reprojection_manifest,
        args.output,
    )


if __name__ == "__main__":
    main()
