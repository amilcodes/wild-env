"""Shared operational-map drawing for desktop and headless rendering."""

from __future__ import annotations

from math import atan2, degrees
from typing import Any

import numpy as np

from aeolus.viewer.config import ViewerConfig
from aeolus.viewer.imagery import ImageryLayer
from aeolus.viewer.model import ReplayModel

BACKGROUND = "#0b1014"
FOREGROUND = "#e8edf0"
MUTED = "#8f9ba3"
GRID = "#6d7880"

RESOURCE_COLOR = {
    "retardant": "#f4d58d",
    "water": "#57b9ff",
    "sensor": "#d9c75c",
    "crew": "#76d98c",
}
RESOURCE_MARKER = {
    "retardant": "^",
    "water": "o",
    "sensor": "D",
    "crew": "s",
}


def _rgba(mask: np.ndarray, color: tuple[float, float, float], alpha: np.ndarray | float) -> np.ndarray:
    output = np.zeros((*mask.shape, 4), dtype=np.float32)
    output[..., :3] = color
    output[..., 3] = np.where(mask, alpha, 0.0)
    return output


def _nice_scale_length(domain_m: float) -> float:
    target = domain_m * 0.16
    magnitude = 10.0 ** np.floor(np.log10(max(target, 1.0)))
    normalized = target / magnitude
    multiplier = 1.0 if normalized < 2.0 else 2.0 if normalized < 5.0 else 5.0
    return float(multiplier * magnitude)


def _resource_heading(model: ReplayModel, frame: int, resource_index: int) -> float | None:
    if frame <= 0:
        return None
    x = np.asarray(model.states["resources/x"][max(0, frame - 2) : frame + 1, resource_index])
    y = np.asarray(model.states["resources/y"][max(0, frame - 2) : frame + 1, resource_index])
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    if abs(dx) + abs(dy) < 1.0e-4:
        return None
    return degrees(atan2(dy, dx))


def _label_positions(resources: list[dict[str, Any]], width: int, height: int) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    candidates = ((1.8, -2.2), (1.8, 2.8), (-8.2, -2.2), (-8.2, 2.8), (1.8, 5.4))
    for resource in resources:
        best = None
        best_cost = float("inf")
        for dx, dy in candidates:
            x = float(np.clip(resource["x"] + dx, 0.5, width - 12.0))
            y = float(np.clip(resource["y"] + dy, 1.0, height - 1.0))
            cost = sum(max(0.0, 10.0 - abs(x - px)) * max(0.0, 3.0 - abs(y - py)) for px, py in positions)
            if cost < best_cost:
                best = (x, y)
                best_cost = cost
        assert best is not None
        positions.append(best)
    return positions


def draw_operational_map(
    axis: Any,
    model: ReplayModel,
    config: ViewerConfig,
    frame: int,
    *,
    selected_resource: str | None = None,
    imagery: ImageryLayer | None = None,
) -> None:
    """Draw one replay frame into an existing Matplotlib axis."""

    from matplotlib import colormaps
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    axis.clear()
    axis.set_facecolor(BACKGROUND)
    states = model.states
    elevation = model.static("static/elevation_m")
    height, width = elevation.shape
    phase = model.field("truth/phase", frame)
    intensity = model.field("truth/intensity_kw_m", frame)
    layers = config.layers
    light = LightSource(azdeg=315, altdeg=42)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "aeolus-terrain",
        ["#102119", "#23402a", "#42583b", "#69674c", "#8d7657", "#c1b29a"],
    )
    terrain = light.shade(
        elevation,
        cmap=terrain_cmap,
        vert_exag=config.camera.vertical_exaggeration,
        blend_mode="soft",
    )
    if layers.imagery and imagery is not None:
        axis.imshow(terrain, origin="upper", interpolation="bilinear", zorder=0)
        axis.imshow(
            imagery.rgb,
            origin="upper",
            interpolation="bilinear",
            alpha=config.imagery.opacity,
            zorder=1,
        )
        if layers.hillshade:
            gray = light.hillshade(
                elevation,
                vert_exag=config.camera.vertical_exaggeration,
            )
            axis.imshow(gray, origin="upper", cmap="gray", alpha=0.25, zorder=2)
    elif layers.hillshade:
        axis.imshow(terrain, origin="upper", interpolation="bilinear", zorder=0)
    else:
        axis.imshow(np.full((*elevation.shape, 3), 0.08), origin="upper", zorder=0)

    if layers.fuels and model.has("static/fuel_model_number"):
        axis.imshow(
            model.static("static/fuel_model_number"),
            origin="upper",
            cmap="viridis",
            alpha=0.36,
            interpolation="nearest",
            zorder=2,
        )
    if layers.contours and float(np.ptp(elevation)) > 20.0:
        levels = np.linspace(float(elevation.min()), float(elevation.max()), 12)[1:-1]
        axis.contour(
            elevation,
            levels=levels,
            colors="#d5d0bf",
            linewidths=0.35,
            alpha=0.22,
            zorder=3,
        )
    if layers.burned_area:
        burn_age_alpha = np.where(phase == 2, 0.70, 0.0)
        axis.imshow(
            _rgba(phase == 2, (0.08, 0.055, 0.045), burn_age_alpha),
            origin="upper",
            zorder=4,
        )
    if layers.active_fire:
        normalized = np.clip(np.log1p(intensity) / np.log(20_001.0), 0.0, 1.0)
        fire_rgba = colormaps["inferno"](normalized)
        fire_rgba[..., 3] = np.where(phase == 1, 0.94, 0.0)
        axis.imshow(fire_rgba, origin="upper", zorder=8)
        if np.any(phase == 1):
            axis.contour(
                phase == 1,
                levels=[0.5],
                colors=["#ffd173"],
                linewidths=0.8,
                alpha=0.95,
                zorder=9,
            )
    if layers.fire_type and model.has("truth/fire_type"):
        fire_type = model.field("truth/fire_type", frame)
        if np.any(fire_type >= 2):
            axis.contour(
                fire_type >= 2,
                levels=[0.5],
                colors=["#f4f0ff"],
                linewidths=1.2,
                linestyles="dashdot",
                zorder=10,
            )
    if layers.belief_uncertainty and model.has("belief/burn_probability"):
        probability = model.field("belief/burn_probability", frame)
        uncertain = (probability >= 0.20) & (probability <= 0.80)
        axis.imshow(
            _rgba(uncertain, (0.50, 0.86, 0.96), 0.20),
            origin="upper",
            zorder=11,
        )
    if layers.belief_perimeter:
        belief = model.field("belief/intensity_mean", frame)
        if np.any(belief >= 20.0):
            axis.contour(
                belief >= 20.0,
                levels=[0.5],
                colors=["#aeeafa"],
                linewidths=1.15,
                linestyles="dashed",
                alpha=0.78,
                zorder=12,
            )
    if layers.water:
        water_name = (
            "treatment/water_coverage_gpc" if model.has("treatment/water_coverage_gpc") else "treatment/water"
        )
        water = model.field(water_name, frame)
        axis.imshow(
            _rgba(
                water > 0.025,
                (0.06, 0.55, 0.96),
                np.clip(water / max(float(np.nanpercentile(water, 98)), 0.15) * 0.70, 0.0, 0.70),
            ),
            origin="upper",
            zorder=13,
        )
    if layers.retardant:
        retardant_name = (
            "treatment/retardant_coverage_gpc"
            if model.has("treatment/retardant_coverage_gpc")
            else "treatment/retardant"
        )
        retardant = model.field(retardant_name, frame)
        axis.imshow(
            _rgba(
                retardant > 0.025,
                (0.88, 0.10, 0.47),
                np.clip(
                    retardant / max(float(np.nanpercentile(retardant, 98)), 0.15) * 0.76,
                    0.0,
                    0.76,
                ),
            ),
            origin="upper",
            zorder=14,
        )
    if layers.constructed_line and model.has("treatment/line_status"):
        status = model.field("treatment/line_status", frame)
        for value, color in ((1, "#64cfff"), (2, "#58e078"), (3, "#ff4f52")):
            line_y, line_x = np.where(status == value)
            if line_x.size:
                axis.scatter(
                    line_x,
                    line_y,
                    marker="s",
                    s=12,
                    color=color,
                    linewidths=0,
                    alpha=0.95,
                    zorder=16,
                )
    if layers.assets:
        assets = model.static("static/asset_value")
        if np.any(assets > 0):
            axis.contourf(
                assets,
                levels=[0.05, float(assets.max()) + 1.0],
                colors=["#f1d36b"],
                alpha=0.12,
                zorder=5,
            )
            axis.contour(
                assets,
                levels=[0.05],
                colors=["#f1d36b"],
                linewidths=1.1,
                zorder=17,
            )
    if layers.service_sites:
        marker = {
            "airport": "P",
            "retardant_base": "s",
            "helibase": "H",
            "dip_site": "v",
            "scoopable_water": "V",
            "temporary_tank": "h",
        }
        for site in model.service_sites:
            axis.scatter(
                [site["x"]],
                [site["y"]],
                s=86,
                marker=marker.get(site["kind"], "P"),
                facecolor="#d7e1e6",
                edgecolor=BACKGROUND,
                linewidth=1.0,
                zorder=19,
            )
            axis.text(
                site["x"] + 1.4,
                site["y"] + 1.8,
                site["site_id"],
                color="#c9d3d8",
                fontsize=7.0,
                zorder=20,
                path_effects=[],
            )

    resources = model.resources(frame)
    label_positions = _label_positions(resources, width, height)
    for resource_index, resource in enumerate(resources):
        kind = str(resource["kind"])
        color = RESOURCE_COLOR[kind]
        if layers.vehicle_tracks:
            start_minute = int(model.minutes[frame]) - config.playback.trail_minutes
            start = int(np.searchsorted(model.minutes, start_minute))
            track_x = np.asarray(states["resources/x"][start : frame + 1, resource_index])
            track_y = np.asarray(states["resources/y"][start : frame + 1, resource_index])
            axis.plot(
                track_x,
                track_y,
                color=color,
                linewidth=1.15,
                alpha=0.72,
                zorder=21,
            )
        if layers.targets and np.isfinite(resource["target_x"]) and np.isfinite(resource["target_y"]):
            axis.plot(
                [resource["x"], resource["target_x"]],
                [resource["y"], resource["target_y"]],
                color=color,
                linewidth=0.65,
                linestyle=(0, (3, 3)),
                alpha=0.54,
                zorder=20,
            )
            axis.scatter(
                [resource["target_x"]],
                [resource["target_y"]],
                marker="x",
                s=30,
                color=color,
                linewidth=0.9,
                zorder=21,
            )
        heading = _resource_heading(model, frame, resource_index)
        resource_marker: Any = RESOURCE_MARKER[kind]
        if heading is not None and kind != "crew":
            resource_marker = MarkerStyle(RESOURCE_MARKER[kind]).transformed(
                Affine2D().rotate_deg(heading - 90.0)
            )
        selected = resource["id"] == selected_resource
        if selected:
            axis.scatter(
                [resource["x"]],
                [resource["y"]],
                s=190,
                facecolor="none",
                edgecolor="#ffffff",
                linewidth=1.3,
                zorder=23,
            )
        axis.scatter(
            [resource["x"]],
            [resource["y"]],
            marker=resource_marker,
            s=82,
            facecolor=color,
            edgecolor=BACKGROUND,
            linewidth=1.0,
            zorder=24,
        )
        if layers.vehicle_labels:
            label_x, label_y = label_positions[resource_index]
            axis.text(
                label_x,
                label_y,
                resource["id"],
                color=FOREGROUND,
                fontsize=7.3,
                weight="medium",
                zorder=25,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": BACKGROUND,
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )

    if layers.wind:
        fire_y, fire_x = np.where(phase == 1)
        sample_x = float(fire_x.mean()) if fire_x.size else width / 2.0
        sample_y = float(fire_y.mean()) if fire_y.size else height / 2.0
        conditions = model.conditions(frame, sample_x, sample_y)
        direction = np.deg2rad(conditions["wind_direction_deg"])
        length = max(width, height) * 0.075
        dx = -np.sin(direction) * length
        dy = np.cos(direction) * length
        origin_x, origin_y = width * 0.91, height * 0.11
        axis.annotate(
            "",
            xy=(origin_x + dx, origin_y + dy),
            xytext=(origin_x, origin_y),
            arrowprops={"arrowstyle": "-|>", "color": FOREGROUND, "lw": 1.2},
            zorder=30,
        )
        axis.text(
            origin_x,
            origin_y - 3.0,
            f"WIND {conditions['wind_direction_deg']:.0f}° / {conditions['wind_speed_m_s']:.1f} m s⁻¹",
            color=FOREGROUND,
            fontsize=7.2,
            ha="right",
            zorder=30,
        )

    scale_m = _nice_scale_length(width * model.cell_size_m)
    scale_cells = scale_m / model.cell_size_m
    x0, y0 = width * 0.055, height * 0.935
    axis.plot([x0, x0 + scale_cells], [y0, y0], color=FOREGROUND, linewidth=2.2, zorder=30)
    axis.plot([x0, x0], [y0 - 0.8, y0 + 0.8], color=FOREGROUND, linewidth=1.0, zorder=30)
    axis.plot(
        [x0 + scale_cells, x0 + scale_cells],
        [y0 - 0.8, y0 + 0.8],
        color=FOREGROUND,
        linewidth=1.0,
        zorder=30,
    )
    scale_label = f"{scale_m / 1000:.1f} km" if scale_m >= 1000 else f"{scale_m:.0f} m"
    axis.text(
        x0 + scale_cells / 2,
        y0 - 1.8,
        scale_label,
        color=FOREGROUND,
        fontsize=7.2,
        ha="center",
        va="top",
        zorder=30,
    )
    axis.annotate(
        "N",
        xy=(width * 0.055, height * 0.055),
        xytext=(width * 0.055, height * 0.115),
        color=FOREGROUND,
        fontsize=8.0,
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": FOREGROUND, "lw": 1.15},
        zorder=30,
    )

    if layers.coordinate_grid:
        tick_x = np.linspace(0, width - 1, 6)
        tick_y = np.linspace(0, height - 1, 6)
        axis.set_xticks(tick_x)
        axis.set_yticks(tick_y)
        axis.grid(color=GRID, linewidth=0.35, alpha=0.20)
        axis.tick_params(colors=MUTED, labelsize=7, length=0)
        axis.set_xticklabels([f"{value * model.cell_size_m / 1000:.1f}" for value in tick_x])
        axis.set_yticklabels([f"{value * model.cell_size_m / 1000:.1f}" for value in tick_y])
        axis.set_xlabel("domain east [km]", color=MUTED, fontsize=7.5)
        axis.set_ylabel("domain south [km]", color=MUTED, fontsize=7.5)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_color("#344048")
        spine.set_linewidth(0.8)
    if imagery is not None and layers.imagery and imagery.attribution:
        axis.text(
            0.995,
            0.008,
            imagery.attribution,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=FOREGROUND,
            fontsize=6.5,
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": BACKGROUND,
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=31,
        )
