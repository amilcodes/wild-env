"""Deterministic publication and video rendering from replay bundles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.replay.recorder import ReplayBundle
from aeolus.viewer.config import ViewerConfig, load_viewer_config
from aeolus.viewer.imagery import load_imagery
from aeolus.viewer.model import ReplayModel
from aeolus.viewer.scene2d import (
    BACKGROUND,
    FOREGROUND,
    MUTED,
    RESOURCE_COLOR,
    draw_operational_map,
)


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] to render replays") from exc
    return plt


def _frame_index(replay: ReplayBundle, frame: int) -> int:
    value = frame if frame >= 0 else replay.frame_count + frame
    if not 0 <= value < replay.frame_count:
        raise IndexError(f"frame {frame} is outside replay with {replay.frame_count} frames")
    return value


def _config(value: ViewerConfig | str | Path | None) -> ViewerConfig:
    return value if isinstance(value, ViewerConfig) else load_viewer_config(value)


def _incident_clock(model: ReplayModel, minute: int) -> str:
    return model.clock_label(minute)


def _draw_status_panel(
    axis: Any,
    model: ReplayModel,
    frame: int,
    selected_resource: str | None,
) -> None:
    axis.clear()
    axis.set_facecolor(BACKGROUND)
    axis.axis("off")
    minute = int(model.minutes[frame])
    phase = model.field("truth/phase", frame)
    intensity = model.field("truth/intensity_kw_m", frame)
    fire_y, fire_x = np.where(phase == 1)
    sample_x = float(fire_x.mean()) if fire_x.size else model.shape[1] / 2
    sample_y = float(fire_y.mean()) if fire_y.size else model.shape[0] / 2
    conditions = model.conditions(frame, sample_x, sample_y)
    resources = model.resources(frame)
    selected = (
        next((item for item in resources if item["id"] == selected_resource), None)
        if selected_resource
        else None
    )

    axis.text(0.0, 0.985, model.title.upper(), color=MUTED, fontsize=8.3, va="top")
    axis.text(
        0.0,
        0.946,
        _incident_clock(model, minute),
        color=FOREGROUND,
        fontsize=13.0,
        weight="medium",
        va="top",
    )
    axis.text(
        0.0,
        0.902,
        f"frame {frame + 1:,} / {model.frame_count:,}   ·   policy {model.metadata['policy_name']}",
        color=MUTED,
        fontsize=7.5,
        va="top",
    )

    active_cells = int((phase == 1).sum())
    burned_cells = int((phase == 2).sum())
    cell_area_ha = model.cell_size_m**2 / 10_000.0
    axis.text(0.0, 0.850, "FIRE STATE", color=MUTED, fontsize=8.0, va="top")
    axis.text(
        0.0,
        0.813,
        f"{active_cells * cell_area_ha:,.1f} ha active",
        color=FOREGROUND,
        fontsize=11.0,
        va="top",
    )
    axis.text(
        0.0,
        0.778,
        f"{burned_cells * cell_area_ha:,.1f} ha burned",
        color=FOREGROUND,
        fontsize=10.0,
        va="top",
    )
    axis.text(
        0.0,
        0.744,
        f"maximum intensity {float(intensity.max()):,.0f} kW m⁻¹",
        color=MUTED,
        fontsize=7.5,
        va="top",
    )
    if model.has("truth/flame_length_m"):
        flame = model.field("truth/flame_length_m", frame)
        axis.text(
            0.0,
            0.716,
            f"maximum flame length {float(flame.max()):.1f} m",
            color=MUTED,
            fontsize=7.5,
            va="top",
        )

    axis.text(0.0, 0.665, "LOCAL CONDITIONS AT FIRE", color=MUTED, fontsize=8.0, va="top")
    axis.text(
        0.0,
        0.627,
        (f"wind {conditions['wind_direction_deg']:.0f}° from / {conditions['wind_speed_m_s']:.1f} m s⁻¹"),
        color=FOREGROUND,
        fontsize=8.6,
        va="top",
    )
    axis.text(
        0.0,
        0.596,
        (
            f"{conditions['air_temperature_c']:.1f} °C   ·   "
            f"RH {conditions['relative_humidity_pct']:.0f}%   ·   "
            f"1 h moisture {conditions['dead_fuel_moisture']:.1%}"
        ),
        color=FOREGROUND,
        fontsize=7.8,
        va="top",
    )
    axis.text(
        0.0,
        0.566,
        f"precipitation {conditions['precipitation_rate_mm_h']:.2f} mm h⁻¹",
        color=MUTED,
        fontsize=7.5,
        va="top",
    )

    if selected is not None:
        axis.text(0.0, 0.520, "SELECTED VEHICLE", color=MUTED, fontsize=8.0, va="top")
        axis.text(
            0.0,
            0.482,
            selected["id"],
            color=RESOURCE_COLOR[selected["kind"]],
            fontsize=10.0,
            weight="medium",
            va="top",
        )
        endurance = selected["endurance_remaining_min"]
        endurance_text = "n/a" if np.isnan(endurance) else f"{endurance:.0f} min"
        axis.text(
            0.0,
            0.448,
            (f"{selected['status_name']}   ·   {selected['task_name']}   ·   ETA {selected['eta_min']} min"),
            color=FOREGROUND,
            fontsize=7.8,
            va="top",
        )
        axis.text(
            0.0,
            0.419,
            (f"payload {selected['payload_fraction']:.0%}   ·   endurance {endurance_text}"),
            color=FOREGROUND,
            fontsize=7.8,
            va="top",
        )
        axis.text(
            0.0,
            0.390,
            f"site {selected['service_site'] or selected['current_site'] or '—'}",
            color=MUTED,
            fontsize=7.5,
            va="top",
        )
        resources_top = 0.342
    else:
        resources_top = 0.520

    axis.text(0.0, resources_top, "FLEET", color=MUTED, fontsize=8.0, va="top")
    available_height = resources_top - 0.12
    row_step = min(0.035, available_height / max(len(resources), 1))
    for row, resource in enumerate(resources):
        y = resources_top - 0.034 - row * row_step
        if y < 0.105:
            break
        axis.scatter(
            [0.018],
            [y],
            s=22,
            marker="o",
            color=RESOURCE_COLOR[resource["kind"]],
            transform=axis.transAxes,
        )
        axis.text(
            0.055,
            y,
            (f"{resource['id']}  ·  {resource['status_name']}  ·  {resource['payload_fraction']:.0%}"),
            color=RESOURCE_COLOR[resource["kind"]],
            fontsize=7.1,
            va="center",
            transform=axis.transAxes,
        )

    events = model.events_near(minute)
    if events:
        event_names = ", ".join(event.kind for event in events[:3])
        if len(events) > 3:
            event_names += f", +{len(events) - 3}"
        axis.text(0.0, 0.065, "EVENTS THIS MINUTE", color=MUTED, fontsize=7.2, va="top")
        axis.text(
            0.0,
            0.035,
            event_names,
            color=FOREGROUND,
            fontsize=6.9,
            va="top",
            wrap=True,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)


def render_frame_2d(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    frame: int = -1,
    dpi: int | None = None,
    viewer_config: ViewerConfig | str | Path | None = None,
    selected_resource: str | None = None,
) -> Path:
    """Render an export-quality operational map and inspection panel."""

    plt = _pyplot()
    config = _config(viewer_config)
    model = ReplayModel(replay)
    index = _frame_index(replay, frame)
    imagery = load_imagery(model, config.imagery)
    export_dpi = config.export.dpi if dpi is None else dpi
    figure = plt.figure(
        figsize=(
            config.export.width / export_dpi,
            config.export.height / export_dpi,
        ),
        facecolor=BACKGROUND,
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(5.35, 1.42))
    map_axis = figure.add_subplot(grid[0, 0])
    panel_axis = figure.add_subplot(grid[0, 1])
    draw_operational_map(
        map_axis,
        model,
        config,
        index,
        selected_resource=selected_resource,
        imagery=imagery,
    )
    minute = int(model.minutes[index])
    map_axis.set_title(
        f"Operational replay  ·  {_incident_clock(model, minute)}",
        loc="left",
        color=FOREGROUND,
        fontsize=13.0,
        weight="medium",
        pad=10,
    )
    _draw_status_panel(panel_axis, model, index, selected_resource)
    figure.subplots_adjust(
        left=0.04,
        right=0.992,
        top=0.94,
        bottom=0.055,
        wspace=0.055,
    )
    figure.text(
        0.012,
        0.007,
        (
            f"Replay schema {model.metadata['schema_version']}  ·  "
            f"{model.spatial_reference.get('crs') or 'local grid'}  ·  "
            f"{model.cell_size_m:g} m cell  ·  "
            "truth, belief and treatment layers shown separately"
        ),
        color=MUTED,
        fontsize=6.7,
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination_path,
        dpi=export_dpi,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return destination_path


def render_frame_3d(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    frame: int = -1,
    dpi: int | None = None,
    viewer_config: ViewerConfig | str | Path | None = None,
    selected_resource: str | None = None,
) -> Path:
    """Render a terrain view; vehicle height is a documented display offset."""

    plt = _pyplot()
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    config = _config(viewer_config)
    model = ReplayModel(replay)
    index = _frame_index(replay, frame)
    elevation = model.static("static/elevation_m")
    phase = model.field("truth/phase", index)
    intensity = model.field("truth/intensity_kw_m", index)
    imagery = load_imagery(model, config.imagery)
    height, width = elevation.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    stride = max(1, max(height, width) // 128)
    light = LightSource(azdeg=315, altdeg=42)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "aeolus-terrain",
        ["#102119", "#23402a", "#42583b", "#69674c", "#8d7657", "#c1b29a"],
    )
    colors = (
        imagery.rgb.copy()
        if config.layers.imagery and imagery is not None
        else light.shade(
            elevation,
            cmap=terrain_cmap,
            vert_exag=config.camera.vertical_exaggeration,
        )[..., :3]
    )
    colors = np.concatenate(
        [colors[..., :3], np.ones((*colors.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    normalized = np.clip(np.log1p(intensity) / np.log(20_001.0), 0.0, 1.0)
    fire_colors = plt.get_cmap("inferno")(normalized)
    active = phase == 1
    colors[active] = fire_colors[active]
    colors[phase == 2, :3] *= 0.24
    export_dpi = config.export.dpi if dpi is None else dpi
    figure = plt.figure(
        figsize=(config.export.width / export_dpi, config.export.height / export_dpi),
        facecolor=BACKGROUND,
        layout="constrained",
    )
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor(BACKGROUND)
    z = (elevation - float(elevation.min())) * config.camera.vertical_exaggeration
    axis.plot_surface(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        z[::stride, ::stride],
        facecolors=colors[::stride, ::stride],
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    active_y, active_x = np.where(active)
    if active_x.size:
        axis.scatter(
            active_x,
            active_y,
            z[active_y, active_x] + max(float(np.ptp(z)) * 0.025, 3.0),
            c=np.clip(intensity[active_y, active_x], 20.0, 10_000.0),
            cmap="inferno",
            norm="log",
            s=8,
            alpha=0.92,
            depthshade=False,
        )
    if config.layers.water:
        water = model.field("treatment/water_coverage_gpc", index)
        water_y, water_x = np.where(water > 0.05)
        axis.scatter(
            water_x,
            water_y,
            z[water_y, water_x] + 1.2,
            color="#168fe5",
            s=5,
            alpha=0.62,
            depthshade=False,
        )
    if config.layers.retardant:
        retardant = model.field("treatment/retardant_coverage_gpc", index)
        retardant_y, retardant_x = np.where(retardant > 0.05)
        axis.scatter(
            retardant_x,
            retardant_y,
            z[retardant_y, retardant_x] + 1.5,
            color="#d91b78",
            s=5,
            alpha=0.68,
            depthshade=False,
        )
    resources = model.resources(index)
    for resource_index, resource in enumerate(resources):
        start_minute = int(model.minutes[index]) - config.playback.trail_minutes
        start = int(np.searchsorted(model.minutes, start_minute))
        trail_x = np.asarray(model.states["resources/x"][start : index + 1, resource_index])
        trail_y = np.asarray(model.states["resources/y"][start : index + 1, resource_index])
        ix = np.clip(np.rint(trail_x).astype(int), 0, width - 1)
        iy = np.clip(np.rint(trail_y).astype(int), 0, height - 1)
        display_offset = 2.0 if resource["kind"] == "crew" else max(float(np.ptp(z)) * 0.06, 12.0)
        trail_z = z[iy, ix] + display_offset
        color = RESOURCE_COLOR[resource["kind"]]
        axis.plot(trail_x, trail_y, trail_z, color=color, linewidth=1.5, alpha=0.85)
        axis.scatter(
            [trail_x[-1]],
            [trail_y[-1]],
            [trail_z[-1]],
            color=color,
            s=56 if resource["id"] == selected_resource else 34,
            edgecolor="#ffffff" if resource["id"] == selected_resource else BACKGROUND,
            linewidth=0.9,
            depthshade=False,
        )
    for site in model.service_sites:
        x, y = int(site["x"]), int(site["y"])
        axis.scatter(
            [x],
            [y],
            [z[y, x] + 2.0],
            color="#d7e1e6",
            marker="P",
            s=42,
            edgecolor=BACKGROUND,
            linewidth=0.7,
            depthshade=False,
        )
    minute = int(model.minutes[index])
    axis.set_title(
        f"Terrain replay  ·  {_incident_clock(model, minute)}",
        loc="left",
        color=FOREGROUND,
        fontsize=13.0,
        weight="medium",
        pad=10,
    )
    axis.view_init(
        elev=config.camera.elevation_deg,
        azim=config.camera.azimuth_deg,
    )
    axis.set_box_aspect((width, height, max(width, height) * 0.35))
    axis.set_axis_off()
    figure.text(
        0.012,
        0.012,
        (
            "Terrain elevation is vertically exaggerated. Aircraft track height is a "
            "display-separation offset; the simulator does not resolve altitude."
        ),
        color=MUTED,
        fontsize=7.0,
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination_path,
        dpi=export_dpi,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return destination_path


def render_video(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    fps: int | None = None,
    max_frames: int = 240,
    viewer_config: ViewerConfig | str | Path | None = None,
    view: str = "operational_2d",
    selected_resource: str | None = None,
) -> Path:
    """Render a deterministic H.264-compatible replay video."""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] to render replay videos") from exc
    import tempfile

    config = _config(viewer_config)
    output_fps = config.export.fps if fps is None else fps
    selected = np.unique(
        np.linspace(
            0,
            replay.frame_count - 1,
            min(replay.frame_count, max_frames),
            dtype=int,
        )
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    render = render_frame_3d if view == "terrain_3d" else render_frame_2d
    video_config = replace(
        config,
        export=replace(config.export, dpi=100),
    )
    with tempfile.TemporaryDirectory(prefix="aeolus-replay-") as temporary:
        temp_root = Path(temporary)
        with imageio.get_writer(
            destination_path,
            fps=output_fps,
            codec=config.export.codec,
            quality=8,
            macro_block_size=2,
        ) as writer:
            for output_index, frame_index in enumerate(selected):
                frame_path = temp_root / f"frame-{output_index:05d}.png"
                render(
                    replay,
                    frame_path,
                    frame=int(frame_index),
                    viewer_config=video_config,
                    selected_resource=selected_resource,
                )
                writer.append_data(imageio.imread(frame_path))
    return destination_path
