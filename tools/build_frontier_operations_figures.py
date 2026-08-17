"""Build publication figures from the frozen frontier-operations results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aeolus.data import (
    StationObservation,
    WeatherForcing,
    analyze_incident_forcing,
)


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "#0f1418",
            "axes.facecolor": "#141b20",
            "axes.edgecolor": "#52606a",
            "axes.labelcolor": "#dce6ec",
            "axes.titlecolor": "#f1f5f7",
            "xtick.color": "#b8c5cc",
            "ytick.color": "#b8c5cc",
            "text.color": "#e8eef2",
            "grid.color": "#33414a",
            "font.size": 9,
            "savefig.facecolor": "#0f1418",
        }
    )


def _short_incident(key: str) -> str:
    mapping = {
        "ca_aeu_017769_electra": "Electra",
        "or_maf_022199_crocketsknob": "Crockets Knob",
        "az_sca_001418_drylake": "Dry Lake",
        "id_ipf_000447_ridgecreek": "Ridge Creek",
        "nm_gnf_000382_davis": "Davis",
        "ut_vld_000127_bear": "Bear",
    }
    return mapping.get(key, key.replace("_", " ").title())


def arrival_atlas(payload: np.lib.npyio.NpzFile, destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    prefixes = sorted(
        key.removesuffix("_history_prediction")
        for key in payload.files
        if key.endswith("_history_prediction")
    )
    figure, axes = plt.subplots(
        len(prefixes),
        5,
        figsize=(12.5, 2.35 * len(prefixes)),
        layout="constrained",
    )
    arrival_cmap = LinearSegmentedColormap.from_list(
        "arrival",
        ["#3b1d5a", "#5f5bc4", "#25a4b8", "#8ed081", "#f1d35f"],
    )
    for row, prefix in enumerate(prefixes):
        earlier = payload[f"{prefix}_history_earlier"].astype(bool)
        start = payload[f"{prefix}_history_start"].astype(bool)
        observed = payload[f"{prefix}_history_observed"].astype(bool)
        predicted = payload[f"{prefix}_history_prediction"].astype(bool)
        arrival = payload[f"{prefix}_history_arrival_time"].astype(float)
        reconstructed = np.where(start, arrival, np.nan)
        panels = (
            (earlier, "earlier perimeter", "gray"),
            (start, "forecast start", "gray"),
            (reconstructed, "reconstructed arrival", arrival_cmap),
            (predicted, "history forecast", "gray"),
            (observed, "held-out observed", "gray"),
        )
        finite = reconstructed[np.isfinite(reconstructed)]
        vmin = float(np.min(finite)) if finite.size else -1.0
        for column, (values, title, cmap) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(
                values,
                origin="upper",
                cmap=cmap,
                vmin=vmin if column == 2 else 0,
                vmax=0 if column == 2 else 1,
                interpolation="nearest",
            )
            if column in (3, 4):
                axis.contour(
                    start,
                    levels=[0.5],
                    colors=["#55d6e8"],
                    linewidths=0.65,
                )
            if row == 0:
                axis.set_title(title, fontsize=10)
            if column == 0:
                axis.set_ylabel(_short_incident(prefix), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Two-perimeter coupled-state initialization and held-out forecasts",
        fontsize=15,
        x=0.50,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def suppression_outcomes(result: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    trials = result["suppression_operations"]["trials"]
    strategies = ("uncontrolled", "aerial_only", "integrated_operations")
    labels = ("uncontrolled", "aerial only", "integrated")
    colors = ("#85919a", "#43a8d8", "#69d58a")
    winds = (2.0, 6.0, 10.0)
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.3), layout="constrained")
    x = np.arange(len(winds))
    width = 0.24
    for strategy_index, (strategy, label, color) in enumerate(zip(strategies, labels, colors, strict=True)):
        records = [item for item in trials if item["strategy"] == strategy]
        loss = [
            np.mean([item["weighted_loss"] for item in records if item["wind_speed_m_s"] == wind])
            for wind in winds
        ]
        burned = [
            100.0 * np.mean([item["burned_fraction"] for item in records if item["wind_speed_m_s"] == wind])
            for wind in winds
        ]
        escaped = [
            100.0 * np.mean([item["escaped"] for item in records if item["wind_speed_m_s"] == wind])
            for wind in winds
        ]
        offset = (strategy_index - 1) * width
        axes[0, 0].bar(x + offset, loss, width, color=color, label=label)
        axes[0, 1].bar(x + offset, burned, width, color=color)
        axes[1, 0].bar(x + offset, escaped, width, color=color)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, [f"{wind:g}" for wind in winds])
        axis.set_xlabel("10 m wind speed (m/s)")
        axis.grid(axis="y", alpha=0.35)
    axes[0, 0].set_title("Value-weighted fire loss")
    axes[0, 0].set_ylabel("loss units")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].set_title("Burned fraction")
    axes[0, 1].set_ylabel("domain (%)")
    axes[1, 0].set_title("Finite-domain escape rate")
    axes[1, 0].set_ylabel("trials (%)")

    integrated = [item for item in trials if item["strategy"] == "integrated_operations"]
    operation_metrics = (
        ("retardant_drops", "retardant drops"),
        ("water_drops", "water drops"),
        ("line_completions", "completed lines"),
        ("held_line_cells", "held line cells"),
        ("breached_line_cells", "breached line cells"),
        ("reload_queue_entries", "reload queues"),
    )
    values = [np.mean([item[key] for item in integrated]) for key, _ in operation_metrics]
    axes[1, 1].barh(
        np.arange(len(values)),
        values,
        color=["#d74d91", "#43a8d8", "#74d98c", "#64e885", "#ef675f", "#d9b45f"],
    )
    axes[1, 1].set_yticks(
        np.arange(len(values)),
        [label for _, label in operation_metrics],
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Integrated operational workload")
    axes[1, 1].set_xlabel("mean per 180 min trial")
    axes[1, 1].grid(axis="x", alpha=0.35)
    figure.suptitle(
        "Matched-seed suppression experiments (8 seeds x 3 wind regimes)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def forcing_analysis_figure(destination: Path) -> None:
    import matplotlib.pyplot as plt

    minute = np.array([0.0, 60.0])
    background = WeatherForcing(
        minute=minute,
        wind_speed_m_s=np.array([5.0, 5.0], dtype=np.float32),
        wind_direction_deg=np.array([270.0, 270.0], dtype=np.float32),
        air_temperature_c=np.array([28.0, 28.0], dtype=np.float32),
        relative_humidity_pct=np.array([28.0, 28.0], dtype=np.float32),
        metadata={"source": "synthetic gridded background"},
    )
    coordinate = np.linspace(0.0, 40_000.0, 81)
    grid_x, grid_y = np.meshgrid(coordinate, coordinate)
    stations = [
        StationObservation(
            0.0,
            11_000.0,
            12_000.0,
            wind_speed_m_s=9.0,
            wind_from_direction_deg=245.0,
            air_temperature_c=33.0,
            relative_humidity_pct=17.0,
            moisture_dead_1h=0.045,
            station_id="RAWS-A",
        ),
        StationObservation(
            0.0,
            31_000.0,
            27_000.0,
            wind_speed_m_s=3.0,
            wind_from_direction_deg=305.0,
            air_temperature_c=25.0,
            relative_humidity_pct=42.0,
            moisture_dead_1h=0.12,
            station_id="RAWS-B",
        ),
    ]
    analysis = analyze_incident_forcing(
        background,
        stations,
        grid_x,
        grid_y,
        length_scale_m=10_000.0,
        time_window_min=10.0,
    )
    sample = analysis.forcing.at_minute(0.0)
    panels = (
        (sample["wind_speed_m_s"], "analyzed wind speed", "m/s", "viridis"),
        (
            sample["moisture_dead_1h"] * 100.0,
            "analyzed 1 h fuel moisture",
            "%",
            "YlGnBu",
        ),
        (
            analysis.wind_correction_std_m_s[0],
            "posterior wind uncertainty",
            "normalized SD",
            "magma",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), layout="constrained")
    for axis, (values, title, units, cmap) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values,
            origin="lower",
            extent=(0, 40, 0, 40),
            cmap=cmap,
        )
        axis.scatter(
            [station.x_m / 1000.0 for station in stations],
            [station.y_m / 1000.0 for station in stations],
            marker="^",
            s=65,
            color="#f7f3e8",
            edgecolor="#101316",
        )
        axis.set_title(title)
        axis.set_xlabel("east (km)")
        axis.set_ylabel("north (km)")
        figure.colorbar(image, ax=axis, label=units, shrink=0.84)
    figure.suptitle(
        "Station-conditioned incident forcing by Gaussian optimum interpolation",
        fontsize=14,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _style()
    result = json.loads((args.results / "operations_results.json").read_text(encoding="utf-8"))
    with np.load(
        args.results / "arrival_history_examples.npz",
        allow_pickle=False,
    ) as examples:
        arrival_atlas(examples, args.out / "arrival_history_atlas.png")
    suppression_outcomes(result, args.out / "suppression_outcomes.png")
    forcing_analysis_figure(args.out / "forcing_analysis.png")


if __name__ == "__main__":
    main()
