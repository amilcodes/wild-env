#!/usr/bin/env python3
"""Render an incident-forcing diagnostic for the historical fidelity iteration."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from aeolus.data import IncidentBundle, WeatherForcing
from aeolus.evaluation.historical import PerimeterSeries


def _times(forcing: WeatherForcing) -> list:
    origin = forcing.time_origin
    if origin is None:
        raise ValueError("weather forcing has no absolute time origin")
    return [origin + timedelta(minutes=float(value)) for value in forcing.minute]


def _spatial_statistic(values: np.ndarray, statistic: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    flattened = array.reshape(len(array), -1)
    if statistic == "mean":
        return flattened.mean(axis=1)
    if statistic == "p10":
        return np.quantile(flattened, 0.10, axis=1)
    if statistic == "p90":
        return np.quantile(flattened, 0.90, axis=1)
    raise ValueError(statistic)


def render(old_root: Path, new_root: Path, destination: Path) -> None:
    old_incident = IncidentBundle.load(old_root)
    new_incident = IncidentBundle.load(new_root)
    old = WeatherForcing.load(old_incident.asset_path("weather"))
    new = WeatherForcing.load(new_incident.asset_path("weather"))
    series = PerimeterSeries.from_incident(new_incident)
    start, end = series.frames[0].timestamp, series.frames[-1].timestamp
    old_time, new_time = _times(old), _times(new)
    new_in_incident = np.asarray(
        [start <= value <= end for value in new_time],
        dtype=np.bool_,
    )
    indices = np.flatnonzero(new_in_incident)
    peak_index = int(indices[np.argmax(_spatial_statistic(new.wind_speed_m_s, "mean")[indices])])

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
    figure = plt.figure(figsize=(13.0, 9.0))
    figure.subplots_adjust(
        left=0.065,
        right=0.965,
        bottom=0.07,
        top=0.875,
        hspace=0.42,
        wspace=0.34,
    )
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.25))
    wind_axis = figure.add_subplot(grid[0, :])
    component_axis = figure.add_subplot(grid[1, 0], sharex=wind_axis)
    moisture_axis = figure.add_subplot(grid[1, 1], sharex=wind_axis)
    map_axis = figure.add_subplot(grid[2, 0])
    live_axis = figure.add_subplot(grid[2, 1], sharex=wind_axis)

    wind_mean = _spatial_statistic(new.wind_speed_m_s, "mean")
    wind_p10 = _spatial_statistic(new.wind_speed_m_s, "p10")
    wind_p90 = _spatial_statistic(new.wind_speed_m_s, "p90")
    wind_axis.fill_between(
        new_time,
        wind_p10,
        wind_p90,
        color="#6BA6C8",
        alpha=0.22,
        label="HRRR spatial 10–90%",
    )
    wind_axis.plot(new_time, wind_mean, color="#246B91", linewidth=1.5, label="HRRR spatial mean")
    wind_axis.plot(
        old_time,
        _spatial_statistic(old.wind_speed_m_s, "mean"),
        color="#B8663B",
        linewidth=1.0,
        alpha=0.85,
        label="NASA POWER point",
    )
    wind_axis.set_ylabel("10-m wind (m/s)")
    wind_axis.set_title("A  Incident-period wind", loc="left", fontweight="bold")
    wind_axis.legend(frameon=False, ncol=3, loc="upper left")

    def components(forcing: WeatherForcing) -> tuple[np.ndarray, np.ndarray]:
        speed = np.asarray(forcing.wind_speed_m_s)
        direction = np.deg2rad(np.asarray(forcing.wind_direction_deg))
        return (
            _spatial_statistic(-speed * np.sin(direction), "mean"),
            _spatial_statistic(-speed * np.cos(direction), "mean"),
        )

    old_u, old_v = components(old)
    new_u, new_v = components(new)
    component_axis.plot(new_time, new_u, color="#246B91", label="HRRR U")
    component_axis.plot(new_time, new_v, color="#4A9271", label="HRRR V")
    component_axis.plot(old_time, old_u, color="#B8663B", alpha=0.65, label="POWER U")
    component_axis.plot(old_time, old_v, color="#8A5B91", alpha=0.65, label="POWER V")
    component_axis.axhline(0.0, color="#334155", linewidth=0.6)
    component_axis.set_ylabel("Wind component (m/s)")
    component_axis.set_title("B  Cartesian wind components", loc="left", fontweight="bold")
    component_axis.legend(frameon=False, ncol=2, loc="upper left")

    for values, label, color in (
        (new.moisture_dead_1h, "1 h", "#C45B3C"),
        (new.moisture_dead_10h, "10 h", "#3C7C9B"),
        (new.moisture_dead_100h, "100 h", "#675A9C"),
    ):
        if values is not None:
            moisture_axis.plot(
                new_time,
                _spatial_statistic(values, "mean"),
                label=label,
                color=color,
            )
    moisture_axis.set_ylabel("Dead moisture (kg/kg)")
    moisture_axis.set_title("C  Prognostic dead-fuel state", loc="left", fontweight="bold")
    moisture_axis.legend(frameon=False, ncol=3, loc="upper left")

    landscape = new_incident.scenario_bundle()
    elevation = landscape.elevation_m
    speed = new.wind_speed_m_s[peak_index]
    direction = np.deg2rad(new.wind_direction_deg[peak_index])
    map_image = map_axis.imshow(speed, cmap="magma", interpolation="nearest")
    map_axis.contour(elevation, levels=9, colors="white", linewidths=0.35, alpha=0.45)
    step = 10
    yy, xx = np.mgrid[0 : speed.shape[0] : step, 0 : speed.shape[1] : step]
    map_axis.quiver(
        xx,
        yy,
        -np.sin(direction[::step, ::step]),
        np.cos(direction[::step, ::step]),
        color="white",
        alpha=0.8,
        scale=22,
        width=0.003,
    )
    map_axis.set_title(
        (f"D  HRRR field at peak mean wind\n{new_time[peak_index]:%Y-%m-%d %H:%M UTC}"),
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    map_axis.set_xticks([])
    map_axis.set_yticks([])
    colorbar = figure.colorbar(map_image, ax=map_axis, shrink=0.85)
    colorbar.set_label("Wind speed (m/s)")

    if new.moisture_live_herbaceous is not None:
        live_axis.plot(
            new_time,
            _spatial_statistic(new.moisture_live_herbaceous, "mean"),
            color="#558B5F",
            label="Live herbaceous",
        )
    if new.moisture_live_woody is not None:
        live_axis.plot(
            new_time,
            _spatial_statistic(new.moisture_live_woody, "mean"),
            color="#94613F",
            label="Live woody",
        )
    live_axis.axhline(1.20, color="#687381", linestyle="--", linewidth=0.8, label="Curing onset")
    live_axis.set_ylabel("Live moisture (kg/kg)")
    live_axis.set_title("E  NFDRS-v4-style live-fuel state", loc="left", fontweight="bold")
    live_axis.legend(frameon=False, loc="upper left")

    for axis in (wind_axis, component_axis, moisture_axis, live_axis):
        axis.set_xlim(start, end)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
        for frame in series.frames:
            axis.axvline(frame.timestamp, color="#1F2937", linewidth=0.45, alpha=0.25)

    coverage = float(new.metadata.get("hrrr_analysis_coverage_fraction", float("nan")))
    figure.suptitle(
        "Electra historical forcing fidelity",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.945,
        (
            f"HRRR direct analysis coverage {coverage:.2%}; vertical marks are NIROPS acquisitions. "
            "Spatial range is native-HRRR sampling plus thermodynamic terrain projection."
        ),
        color="#475569",
        fontsize=9,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)

    old_origin = old.time_origin
    if old_origin is None:
        raise ValueError("comparison forcing has no time origin")
    comparison_times = [new_time[index] for index in indices]
    old_samples = [
        old.at_minute((timestamp - old_origin).total_seconds() / 60.0) for timestamp in comparison_times
    ]
    old_speed = np.asarray([sample["wind_speed_m_s"] for sample in old_samples])
    old_direction = np.asarray([sample["wind_direction_deg"] for sample in old_samples])
    new_speed = wind_mean[indices]
    new_direction = np.rad2deg(np.arctan2(-new_u[indices], -new_v[indices])) % 360.0
    direction_error = ((new_direction - old_direction + 180.0) % 360.0) - 180.0
    wind_spatial_range = wind_p90[indices] - wind_p10[indices]
    summary = {
        "schema_version": 1,
        "incident": new_incident.incident_id,
        "comparison_sample_count": len(indices),
        "hrrr_analysis_coverage_fraction": new.metadata.get("hrrr_analysis_coverage_fraction"),
        "hrrr_missing_analysis_hours": new.metadata.get(
            "hrrr_missing_analysis_hours",
            [],
        ),
        "hrrr_vs_power": {
            "wind_speed_mean_absolute_difference_m_s": float(np.mean(np.abs(new_speed - old_speed))),
            "wind_direction_mean_absolute_difference_deg": float(np.mean(np.abs(direction_error))),
            "mean_hrrr_spatial_p90_p10_wind_range_m_s": float(np.mean(wind_spatial_range)),
            "maximum_hrrr_spatial_p90_p10_wind_range_m_s": float(np.max(wind_spatial_range)),
        },
        "derived_state": {
            "dead_1h_spatial_standard_deviation_final_kg_kg": (
                float(np.std(new.moisture_dead_1h[-1])) if new.moisture_dead_1h is not None else None
            ),
            "live_herbaceous_minimum_kg_kg": (
                float(np.min(new.moisture_live_herbaceous))
                if new.moisture_live_herbaceous is not None
                else None
            ),
            "live_herbaceous_maximum_kg_kg": (
                float(np.max(new.moisture_live_herbaceous))
                if new.moisture_live_herbaceous is not None
                else None
            ),
        },
        "interpretation": (
            "Differences quantify forcing products and spatial sampling; they "
            "are not wind errors because no independent station reference is used."
        ),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_incident", type=Path)
    parser.add_argument("new_incident", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.old_incident, args.new_incident, args.output)


if __name__ == "__main__":
    main()
