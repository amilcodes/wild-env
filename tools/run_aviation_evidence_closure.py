#!/usr/bin/env python3
"""Audit selected wildfire aircraft and render the evidence-closure matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from aeolus.data.aerial_delivery import delivery_geometry, load_aerial_delivery_surface
from aeolus.data.aviation_catalog import load_vehicle_catalog
from aeolus.data.aviation_evidence import (
    audit_aviation_evidence,
    load_aviation_evidence_registry,
    required_evidence_domains,
)

STATUS_VALUE = {"open": 0, "proxy": 1, "partial": 2, "closed": 3}
STATUS_COLOR = ["#D7D9DC", "#E8B45B", "#5F8FB7", "#438A72"]


def _figure(audit: dict[str, object], path: Path) -> None:
    profiles = audit["profiles"]
    domains = sorted(
        {domain for profile in profiles for domain in required_evidence_domains(profile["resource_kind"])}
    )
    matrix = np.full((len(profiles), len(domains)), np.nan)
    for row, profile in enumerate(profiles):
        for column, domain in enumerate(domains):
            evidence = profile["domains"].get(domain)
            if evidence is not None:
                matrix[row, column] = STATUS_VALUE[evidence["status"]]

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(4.7, 1.5))
    ax = fig.add_subplot(grid[0, 0])
    masked = np.ma.masked_invalid(matrix)
    cmap = ListedColormap(STATUS_COLOR)
    cmap.set_bad("#FFFFFF")
    ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([item.replace("_", "\n") for item in domains], fontsize=9)
    ax.set_yticks(np.arange(len(profiles)))
    ax.set_yticklabels([profile["display_name"] for profile in profiles], fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(profiles), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("A. Exact-configuration evidence closure", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Required model domain", labelpad=14)
    for value, label in enumerate(("Open", "Proxy", "Partial", "Closed")):
        ax.scatter([], [], marker="s", s=120, color=STATUS_COLOR[value], label=label)
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
    )

    side = fig.add_subplot(grid[0, 1])
    scores = np.asarray([profile["closure_score"] for profile in profiles])
    positions = np.arange(len(profiles))
    side.barh(positions, scores, color="#476F91", height=0.64)
    side.set_yticks([])
    side.set_xlim(0.0, 1.0)
    side.invert_yaxis()
    side.set_xlabel("Research evidence score")
    side.set_title("B. Coverage, not field qualification", loc="left", fontsize=15, weight="bold")
    side.grid(axis="x", alpha=0.22)
    for position, score in zip(positions, scores, strict=True):
        side.text(min(score + 0.025, 0.92), position, f"{score:.2f}", va="center", fontsize=9)
    side.text(
        0.0,
        1.075,
        f"{audit['document_count']} public sources reviewed\n"
        f"{audit['field_closed_profile_count']} / {audit['profile_count']} profiles field-closed",
        transform=side.transAxes,
        va="bottom",
        fontsize=11,
    )
    side.text(
        0.0,
        -0.16,
        "A closed cell requires exact-current aircraft and mission-system evidence\n"
        "from an approved flight manual or controlled engineering validation.",
        transform=side.transAxes,
        va="top",
        fontsize=9,
        color="#333333",
    )
    for spine in ("top", "right", "left"):
        side.spines[spine].set_visible(False)
    fig.suptitle(
        "Wildfire aviation evidence closure — public-source acquisition, 31 July 2026",
        fontsize=18,
        weight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/aviation/us_wildfire_reference_fleet_v1.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/aviation/evidence_registry_v1.json"),
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path("configs/aviation/delivery_surfaces/calfire_s2t_mtdc_2006_gum_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aviation_evidence"),
    )
    args = parser.parse_args()
    catalog = load_vehicle_catalog(args.catalog)
    registry = load_aviation_evidence_registry(args.registry, catalog=catalog)
    audit = audit_aviation_evidence(registry, catalog)
    surface = load_aerial_delivery_surface(args.surface)
    geometry = delivery_geometry(
        surface,
        requested_coverage_gpc=3.0,
        payload_l=surface.nominal_payload_l,
    )
    audit["implemented_public_evidence"] = {
        "s2t_coverage_level_3": {
            "line_length_m": geometry.line_length_m,
            "effective_width_m": geometry.effective_width_m,
            "flow_rate_l_s": geometry.flow_rate_l_s,
            "controller_setting": geometry.controller_setting,
            "prior_generic_length_m": 650.0,
            "prior_generic_width_m": 70.0,
            "prior_to_measured_equivalent_bounding_area_ratio": (
                650.0 * 70.0 / (geometry.line_length_m * geometry.effective_width_m)
            ),
        },
        "firehawk_refill": {
            "published_upper_bound_s": 45.0,
            "canonical_simulator_ticks": 1,
            "tick_duration_s": 60.0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "aviation_evidence_closure.json"
    figure_path = args.output_dir / "aviation_evidence_closure.png"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _figure(audit, figure_path)
    print(json.dumps({"audit": str(json_path), "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
