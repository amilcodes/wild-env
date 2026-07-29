"""Build publication figures and flat metric tables for the NIROPS study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS = ("persistence", "raw_physics", "calibrated_physics")
METHOD_LABELS = {
    "persistence": "Persistence",
    "raw_physics": "Raw physics",
    "calibrated_physics": "Calibrated physics",
}
METHOD_COLORS = {
    "persistence": "#6D7785",
    "raw_physics": "#C96732",
    "calibrated_physics": "#2C718E",
}
INCIDENT_LABELS = {
    "CA-AEU-017769_Electra": "Electra, CA",
    "OR-MAF-022199_CrocketsKnob": "Crockets Knob, OR",
    "AZ-SCA-001418_DryLake": "Dry Lake, AZ",
    "ID-IPF-000447_RidgeCreek": "Ridge Creek, ID",
    "NM-GNF-000382_Davis": "Davis, NM",
    "UT-VLD-000127_Bear": "Bear, UT",
}


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
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


def _summary_bar(
    ax: plt.Axes,
    summaries: dict[str, Any],
    metric: str,
    title: str,
    ylabel: str,
    *,
    scale: float = 1.0,
) -> None:
    means = [summaries[method][metric]["mean"] * scale for method in METHODS]
    lows = [summaries[method][metric]["ci95_low"] * scale for method in METHODS]
    highs = [summaries[method][metric]["ci95_high"] * scale for method in METHODS]
    errors = np.asarray(
        [
            [mean - low for mean, low in zip(means, lows, strict=True)],
            [high - mean for mean, high in zip(means, highs, strict=True)],
        ]
    )
    x = np.arange(len(METHODS))
    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=[METHOD_COLORS[method] for method in METHODS],
        yerr=errors,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#27313B"},
    )
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}" if value < 10 else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_summary_figure(results: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    _summary_bar(
        axes[0, 0],
        results["summaries"],
        "metrics.iou",
        "A. Cumulative burned-extent IoU",
        "IoU (higher is better)",
    )
    _summary_bar(
        axes[0, 1],
        results["active_growth_summaries"],
        "growth_tolerance_1_cell.f1",
        "B. Advancing-front localization, active-growth intervals",
        "1-cell-tolerance F1 (higher is better)",
    )
    _summary_bar(
        axes[1, 0],
        results["summaries"],
        "boundary.mean_symmetric_distance_m",
        "C. Perimeter boundary displacement",
        "Mean symmetric distance (m; lower is better)",
    )
    _summary_bar(
        axes[1, 1],
        results["summaries"],
        "metrics.symmetric_difference_km2",
        "D. Burned-extent disagreement",
        "Symmetric difference (km²; lower is better)",
    )
    fig.suptitle(
        "Held-out historical hindcasts: six incidents, 24 forecast intervals",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.955,
        "Bars are incident-cluster bootstrap means with 95% intervals; each forecast "
        "is initialized from an observed NIROPS perimeter.",
        ha="left",
        va="top",
        color="#45515E",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.92), h_pad=2.2, w_pad=2.2)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _incident_records(results: dict[str, Any], code: str, method: str) -> list[dict[str, Any]]:
    return [
        item
        for item in results["forecasts"]
        if item["incident_code"] == code and item["method"] == method
    ]


def build_transfer_figure(results: dict[str, Any], destination: Path) -> None:
    codes = [item["incident_code"] for item in results["calibrations"]]
    labels = [INCIDENT_LABELS[code] for code in codes]
    adjustments = [
        float(item["selected_spread_adjustment"])
        for item in results["calibrations"]
    ]
    cumulative_deltas = []
    active_growth_deltas = []
    for code in codes:
        raw = _incident_records(results, code, "raw_physics")
        calibrated = _incident_records(results, code, "calibrated_physics")
        cumulative_deltas.append(
            np.mean([item["forecast"]["metrics"]["iou"] for item in calibrated])
            - np.mean([item["forecast"]["metrics"]["iou"] for item in raw])
        )
        active = [
            (raw_item, calibrated_item)
            for raw_item, calibrated_item in zip(raw, calibrated, strict=True)
            if raw_item["forecast"]["growth_metrics"]["observed_area_km2"] > 0
        ]
        active_growth_deltas.append(
            np.mean(
                [
                    calibrated_item["forecast"]["growth_tolerance_1_cell"]["f1"]
                    - raw_item["forecast"]["growth_tolerance_1_cell"]["f1"]
                    for raw_item, calibrated_item in active
                ]
            )
            if active
            else 0.0
        )

    y = np.arange(len(codes))
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.2),
        gridspec_kw={"width_ratios": (0.9, 1.25)},
    )
    left.scatter(adjustments, y, s=65, color="#2C718E", zorder=3)
    left.axvline(1.0, color="#66717D", linewidth=1, linestyle="--")
    left.set_xscale("log")
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlabel("Selected spread multiplier (log scale)")
    left.set_title("A. Fit on one earlier interval", loc="left", fontweight="bold")
    for value, y_value in zip(adjustments, y, strict=True):
        left.text(value * 1.08, y_value, f"{value:g}", va="center", fontsize=8)

    height = 0.34
    right.barh(
        y - height / 2,
        cumulative_deltas,
        height,
        label="Cumulative IoU",
        color="#2C718E",
    )
    right.barh(
        y + height / 2,
        active_growth_deltas,
        height,
        label="Active-growth tolerance F1",
        color="#D39C43",
    )
    right.axvline(0, color="#27313B", linewidth=1)
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.set_xlabel("Calibrated minus raw score")
    right.set_title("B. Transfer to four later intervals", loc="left", fontweight="bold")
    right.legend(frameon=False, loc="lower right")
    right.text(
        0.01,
        0.01,
        "Positive is improvement. Bear is the regime-transfer failure.",
        transform=right.transAxes,
        fontsize=8,
        color="#45515E",
    )
    fig.suptitle(
        "One-interval scalar calibration does not transfer consistently",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.92), w_pad=2.5)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _hillshade(elevation: np.ndarray) -> np.ndarray:
    finite = np.asarray(elevation, dtype=float)
    if np.isnan(finite).any():
        finite = np.nan_to_num(finite, nan=float(np.nanmedian(finite)))
    return LightSource(azdeg=315, altdeg=35).hillshade(
        finite, vert_exag=1.8, dx=1.0, dy=1.0
    )


def build_atlas(
    results: dict[str, Any],
    examples_path: Path,
    prepared: dict[str, Any],
    destination: Path,
) -> None:
    prepared_by_code = {
        item["incident_code"]: item for item in prepared["incidents"]
    }
    codes = [item["incident_code"] for item in results["calibrations"]]
    records = {
        code: _incident_records(results, code, "calibrated_physics")[-1]
        for code in codes
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.6))
    categorical = ListedColormap(
        ["#00000000", "#334155D9", "#3996C6E8", "#66A85CEB", "#D65745EB"]
    )
    with np.load(examples_path) as values:
        for ax, code in zip(axes.flat, codes, strict=True):
            key = code.lower().replace("-", "_")
            elevation = values[f"{key}_elevation"]
            start = values[f"{key}_start"].astype(bool)
            observed = values[f"{key}_observed"].astype(bool)
            predicted = values[f"{key}_calibrated"].astype(bool)
            observed_growth = observed & ~start
            predicted_growth = predicted & ~start
            classes = np.zeros(start.shape, dtype=np.uint8)
            classes[start] = 1
            classes[observed_growth & ~predicted_growth] = 2
            classes[observed_growth & predicted_growth] = 3
            classes[predicted_growth & ~observed_growth] = 4
            ax.imshow(_hillshade(elevation), cmap="gray", vmin=0.25, vmax=1.0)
            ax.imshow(classes, cmap=categorical, vmin=0, vmax=4, interpolation="nearest")
            forecast = records[code]["forecast"]
            cell_size = float(prepared_by_code[code]["cell_size_m"])
            width_km = start.shape[1] * cell_size / 1000.0
            duration_h = float(forecast["requested_minutes"]) / 60.0
            ax.set_title(
                f"{INCIDENT_LABELS[code]}\n"
                f"{duration_h:.1f} h | IoU {forecast['metrics']['iou']:.2f} | "
                f"growth F1 {forecast['growth_tolerance_1_cell']['f1']:.2f}",
                loc="left",
                fontweight="bold",
                fontsize=9.5,
            )
            bar_km = max(1, round(width_km / 5))
            pixels = bar_km * 1000.0 / cell_size
            x0 = 7
            y0 = start.shape[0] - 8
            ax.plot([x0, x0 + pixels], [y0, y0], color="white", lw=4, solid_capstyle="butt")
            ax.text(
                x0 + pixels / 2,
                y0 - 3,
                f"{bar_km} km",
                color="white",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#CFD5DB")
    handles = [
        Patch(facecolor="#334155", label="Observed at initialization"),
        Patch(facecolor="#3996C6", label="Observed growth missed"),
        Patch(facecolor="#66A85C", label="Observed growth predicted"),
        Patch(facecolor="#D65745", label="Predicted growth outside observation"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8.5,
    )
    fig.suptitle(
        "Final held-out interval for each incident: calibrated forecast error anatomy",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.955,
        "Terrain is hillshaded. Blue and red expose directional disagreement that "
        "cumulative extent scores can hide.",
        ha="left",
        va="top",
        color="#45515E",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.025, 0.07, 0.99, 0.93), h_pad=1.6, w_pad=1.1)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def build_study_map(prepared: dict[str, Any], destination: Path) -> None:
    state_positions = {
        "WA": (-120.7, 47.3),
        "OR": (-120.7, 44.1),
        "CA": (-119.6, 37.1),
        "ID": (-114.3, 45.6),
        "NV": (-116.8, 39.0),
        "UT": (-111.8, 39.2),
        "AZ": (-111.8, 34.4),
        "NM": (-106.0, 34.3),
        "MT": (-110.8, 47.0),
        "WY": (-107.5, 43.0),
        "CO": (-105.8, 39.0),
    }
    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    ax.set_facecolor("#F3F0E8")
    label_offsets = {
        "CA-AEU-017769_Electra": (7, 7),
        "OR-MAF-022199_CrocketsKnob": (7, -13),
        "AZ-SCA-001418_DryLake": (7, -14),
        "ID-IPF-000447_RidgeCreek": (7, -12),
        "NM-GNF-000382_Davis": (7, 9),
        "UT-VLD-000127_Bear": (7, -13),
    }
    for state, (lon, lat) in state_positions.items():
        ax.text(lon, lat, state, color="#C1B8AA", fontsize=13, fontweight="bold", ha="center")
    for index, item in enumerate(prepared["incidents"], start=1):
        west, south, east, north = item["bbox"]
        lon = (west + east) / 2
        lat = (south + north) / 2
        ax.scatter(lon, lat, s=58, color="#C9563F", edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(
            f"{index}  {INCIDENT_LABELS[item['incident_code']]}",
            (lon, lat),
            xytext=label_offsets[item["incident_code"]],
            textcoords="offset points",
            fontsize=8.5,
            color="#26313A",
        )
    ax.set_xlim(-124.5, -103.0)
    ax.set_ylim(31.0, 49.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.suptitle(
        "Six-fire held-out study sample",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.91,
        "NIROPS airborne infrared progressions, 2020-2023",
        fontsize=9,
        color="#45515E",
        va="top",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _plot_geometry(ax: plt.Axes, geometry: Any, **kwargs: Any) -> None:
    if geometry.geom_type in {"LineString", "LinearRing"}:
        x, y = geometry.xy
        ax.plot(x, y, **kwargs)
    elif geometry.geom_type == "Polygon":
        x, y = geometry.exterior.xy
        ax.plot(x, y, **kwargs)
        for interior in geometry.interiors:
            x, y = interior.xy
            ax.plot(x, y, **kwargs)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            _plot_geometry(ax, part, **kwargs)


def build_fireline_case(
    fireline_gdb: Path,
    crockets_bundle: Path,
    audit: dict[str, Any],
    destination: Path,
) -> None:
    try:
        import pyogrio
        from shapely import from_wkb
        from shapely.geometry import shape
    except ImportError:
        return
    _, table = pyogrio.read_arrow(
        fireline_gdb,
        layer="Firelines_Engagement_17_24",
        where="IncidentName = 'Crockets Knob'",
        columns=["IncidentName", "FirelineEngagement", "LineDateTime"],
    )
    item = json.loads((crockets_bundle / "item.json").read_text(encoding="utf-8"))
    perimeter_path = crockets_bundle / item["assets"]["observed-perimeters"]["href"]
    perimeter_data = json.loads(perimeter_path.read_text(encoding="utf-8"))
    final_perimeter = shape(perimeter_data["features"][-1]["geometry"])
    colors = {
        "Held": "#31836B",
        "Burned Over": "#D04A3A",
        "Not Engaged": "#7A8794",
    }
    engagements = table["FirelineEngagement"].to_pylist()
    geometries = from_wkb(table["Shape"].to_numpy())
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    _plot_geometry(ax, final_perimeter, color="#171D22", linewidth=1.6, zorder=4)
    for engagement in ("Not Engaged", "Held", "Burned Over"):
        for geometry, value in zip(geometries, engagements, strict=True):
            if value == engagement:
                _plot_geometry(
                    ax,
                    geometry,
                    color=colors[engagement],
                    linewidth=0.45 if engagement == "Not Engaged" else 0.7,
                    alpha=0.55 if engagement == "Not Engaged" else 0.8,
                    zorder=2 if engagement == "Not Engaged" else 3,
                )
    west, south, east, north = final_perimeter.bounds
    dx = max((east - west) * 0.12, 0.015)
    dy = max((north - south) * 0.12, 0.015)
    ax.set_xlim(west - dx, east + dx)
    ax.set_ylim(south - dy, north + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.suptitle(
        "Crockets Knob: spatial fireline outcomes are available",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    case = audit["crockets_knob"]
    ax.text(
        0.08,
        0.925,
        f"{case['features']:,} line features; only "
        f"{case['line_datetime_present_fraction'] * 100:.2f}% carry LineDateTime",
        transform=fig.transFigure,
        va="top",
        fontsize=9,
        color="#45515E",
    )
    handles = [
        Line2D([0], [0], color="#171D22", linewidth=1.8, label="Final NIROPS perimeter"),
        *[
            Line2D([0], [0], color=colors[name], linewidth=2.2, label=name)
            for name in ("Held", "Burned Over", "Not Engaged")
        ],
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.91))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def export_forecasts(results: dict[str, Any], destination: Path) -> None:
    fields = [
        "incident_code",
        "method",
        "start_index",
        "target_index",
        "start_time",
        "target_time",
        "duration_h",
        "cumulative_iou",
        "growth_iou",
        "growth_tolerance_f1",
        "boundary_mean_m",
        "hausdorff_95_m",
        "symmetric_difference_km2",
        "observed_growth_km2",
        "predicted_growth_km2",
        "spread_adjustment",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results["forecasts"]:
            forecast = item["forecast"]
            writer.writerow(
                {
                    "incident_code": item["incident_code"],
                    "method": item["method"],
                    "start_index": item["start_index"],
                    "target_index": item["target_index"],
                    "start_time": forecast["start_time"],
                    "target_time": forecast["target_time"],
                    "duration_h": forecast["requested_minutes"] / 60.0,
                    "cumulative_iou": forecast["metrics"]["iou"],
                    "growth_iou": forecast["growth_metrics"]["iou"],
                    "growth_tolerance_f1": forecast["growth_tolerance_1_cell"]["f1"],
                    "boundary_mean_m": forecast["boundary"]["mean_symmetric_distance_m"],
                    "hausdorff_95_m": forecast["boundary"]["hausdorff_95_m"],
                    "symmetric_difference_km2": forecast["metrics"]["symmetric_difference_km2"],
                    "observed_growth_km2": forecast["growth_metrics"]["observed_area_km2"],
                    "predicted_growth_km2": forecast["growth_metrics"]["predicted_area_km2"],
                    "spread_adjustment": item["spread_adjustment"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--fireline-gdb", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    _set_style()
    build_summary_figure(results, args.out / "aggregate_metrics.png")
    build_transfer_figure(results, args.out / "calibration_transfer.png")
    build_atlas(results, args.examples, prepared, args.out / "incident_atlas.png")
    build_study_map(prepared, args.out / "incident_locations.png")
    if args.fireline_gdb:
        crockets = Path(
            next(
                item["bundle"]
                for item in prepared["incidents"]
                if "CrocketsKnob" in item["incident_code"]
            )
        )
        build_fireline_case(
            args.fireline_gdb,
            crockets,
            audit,
            args.out / "crockets_knob_firelines.png",
        )
    export_forecasts(results, args.out.parent / "forecast_metrics.csv")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
