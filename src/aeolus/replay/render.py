"""High-resolution 2D and terrain-aware 3D replay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aeolus.replay.recorder import ReplayBundle


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] to render replays") from exc
    return plt


def _frame_index(replay: ReplayBundle, frame: int) -> int:
    count = replay.frame_count
    value = frame if frame >= 0 else count + frame
    if not 0 <= value < count:
        raise IndexError(f"frame {frame} is outside replay with {count} frames")
    return value


def _rgba_overlay(mask: np.ndarray, color: tuple[float, float, float], alpha: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def render_frame_2d(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    frame: int = -1,
    dpi: int = 180,
) -> Path:
    plt = _pyplot()
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    index = _frame_index(replay, frame)
    states = replay.states
    elevation = np.asarray(states["static/elevation_m"])
    phase = np.asarray(states["truth/phase"][index])
    intensity = np.asarray(states["truth/intensity_kw_m"][index])
    belief = np.asarray(states["belief/intensity_mean"][index])
    water = np.asarray(states["treatment/water"][index])
    retardant = np.asarray(states["treatment/retardant"][index])
    ground = np.asarray(states["treatment/ground_hold"][index])
    assets = np.asarray(states["static/asset_value"])
    minute = int(states["time/minute"][index])

    figure = plt.figure(figsize=(14.4, 8.4), facecolor="#101316", layout="constrained")
    grid = figure.add_gridspec(1, 2, width_ratios=(4.8, 1.25))
    axis = figure.add_subplot(grid[0, 0])
    status = figure.add_subplot(grid[0, 1])
    for item in (axis, status):
        item.set_facecolor("#101316")

    light = LightSource(azdeg=315, altdeg=38)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "incident-terrain",
        ["#17251d", "#29452d", "#496343", "#77745a", "#a38f70", "#d4c9b4"],
    )
    terrain = light.shade(
        elevation,
        cmap=terrain_cmap,
        vert_exag=1.5,
        blend_mode="soft",
    )
    axis.imshow(terrain, origin="upper")
    axis.imshow(
        _rgba_overlay(phase == 2, (0.10, 0.07, 0.06), np.full(phase.shape, 0.70)),
        origin="upper",
    )
    active_normalized = np.clip((intensity - 18.0) / 62.0, 0.0, 1.0)
    active_rgba = plt.get_cmap("autumn")(active_normalized)
    active_rgba[..., :3] = np.maximum(active_rgba[..., :3], (0.98, 0.20, 0.02))
    active_rgba[..., 3] = np.where(phase == 1, 0.90, 0.0)
    axis.imshow(active_rgba, origin="upper")
    if np.any(phase == 1):
        axis.contour(
            phase == 1,
            levels=[0.5],
            colors=["#ffd166"],
            linewidths=0.65,
            alpha=0.9,
        )
    axis.imshow(
        _rgba_overlay(water > 0.05, (0.10, 0.62, 0.95), np.clip(water * 0.72, 0.0, 0.72)),
        origin="upper",
    )
    axis.imshow(
        _rgba_overlay(
            retardant > 0.05,
            (0.90, 0.18, 0.55),
            np.clip(retardant * 0.78, 0.0, 0.78),
        ),
        origin="upper",
    )
    axis.imshow(
        _rgba_overlay(ground > 0.05, (0.35, 0.90, 0.48), np.clip(ground * 0.62, 0.0, 0.62)),
        origin="upper",
    )
    if np.any(belief >= 20.0):
        axis.contour(
            belief >= 20.0,
            levels=[0.5],
            colors=["#b8efff"],
            linewidths=1.1,
            linestyles="dashed",
            alpha=0.62,
        )
    if np.any(assets > 0):
        axis.contour(assets, levels=[0.1], colors=["#f8d675"], linewidths=1.2)

    resource_ids = replay.metadata["resource_ids"]
    resource_kinds = replay.metadata["resource_kinds"]
    x_values = np.asarray(states["resources/x"][index])
    y_values = np.asarray(states["resources/y"][index])
    marker_by_kind = {"retardant": "^", "water": "o", "sensor": "D"}
    color_by_kind = {"retardant": "#f7f0db", "water": "#72c9ff", "sensor": "#f4d35e"}
    for resource_index, (resource_id, kind) in enumerate(zip(resource_ids, resource_kinds, strict=True)):
        start = max(0, index - 28)
        trail_x = np.asarray(states["resources/x"][start : index + 1, resource_index])
        trail_y = np.asarray(states["resources/y"][start : index + 1, resource_index])
        axis.plot(trail_x, trail_y, color=color_by_kind[kind], linewidth=1.1, alpha=0.68)
        axis.scatter(
            [x_values[resource_index]],
            [y_values[resource_index]],
            marker=marker_by_kind[kind],
            s=76,
            color=color_by_kind[kind],
            edgecolors="#101316",
            linewidths=0.9,
            zorder=8,
        )
        axis.text(
            x_values[resource_index] + 1.4 + 3.5 * (resource_index // 4),
            y_values[resource_index] - 4.6 + 2.6 * (resource_index % 4),
            resource_id,
            color="#f2f5f7",
            fontsize=7.5,
            weight="medium",
            zorder=9,
        )

    axis.set_title(
        f"Wildfire initial attack  •  T+{minute:03d} min",
        loc="left",
        color="#f2f5f7",
        fontsize=15,
        weight="medium",
        pad=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#39434b")

    active_cells = int((phase == 1).sum())
    burned_cells = int((phase == 2).sum())
    status.axis("off")
    status.text(0.0, 0.97, "INCIDENT STATE", color="#97a5af", fontsize=8.5, transform=status.transAxes)
    status.text(
        0.0,
        0.90,
        f"{active_cells:,}",
        color="#f2f5f7",
        fontsize=24,
        weight="medium",
        transform=status.transAxes,
    )
    status.text(0.0, 0.855, "active cells", color="#97a5af", fontsize=9, transform=status.transAxes)
    status.text(
        0.0,
        0.76,
        f"{burned_cells:,}",
        color="#f2f5f7",
        fontsize=19,
        weight="medium",
        transform=status.transAxes,
    )
    status.text(0.0, 0.715, "burned cells", color="#97a5af", fontsize=9, transform=status.transAxes)
    status.text(0.0, 0.64, "RESOURCES", color="#97a5af", fontsize=8.5, transform=status.transAxes)
    statuses = np.asarray(states["resources/status"][index])
    payload = np.asarray(states["resources/payload_fraction"][index])
    status_names = ["ready", "outbound", "returning", "reloading", "withdrawn"]
    resource_step = min(0.095, 0.36 / max(len(resource_ids) - 1, 1))
    for row, resource_id in enumerate(resource_ids):
        y = 0.58 - row * resource_step
        status.text(0.0, y, resource_id, color="#f2f5f7", fontsize=9, transform=status.transAxes)
        status.text(
            0.0,
            y - 0.026,
            f"{status_names[int(statuses[row])]}  ·  payload {payload[row]:.0%}",
            color="#97a5af",
            fontsize=7.5,
            transform=status.transAxes,
        )
    status.text(0.0, 0.16, "MAP KEY", color="#97a5af", fontsize=8.5, transform=status.transAxes)
    legend = [
        ("active fire", "#ff8a2b"),
        ("policy belief edge", "#b8efff"),
        ("water", "#1f9bea"),
        ("retardant", "#e62e8c"),
        ("ground hold", "#59d77a"),
    ]
    for row, (label, color) in enumerate(legend):
        y = 0.125 - row * 0.030
        status.scatter([0.025], [y], s=28, color=color, transform=status.transAxes)
        status.text(0.075, y - 0.007, label, color="#dce3e8", fontsize=8, transform=status.transAxes)

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return destination_path


def render_frame_3d(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    frame: int = -1,
    dpi: int = 180,
) -> Path:
    plt = _pyplot()
    from matplotlib.colors import LightSource, LinearSegmentedColormap

    index = _frame_index(replay, frame)
    states = replay.states
    elevation = np.asarray(states["static/elevation_m"])
    phase = np.asarray(states["truth/phase"][index])
    intensity = np.asarray(states["truth/intensity_kw_m"][index])
    minute = int(states["time/minute"][index])
    height, width = elevation.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    stride = max(1, max(height, width) // 96)
    light = LightSource(azdeg=315, altdeg=38)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "incident-terrain",
        ["#17251d", "#29452d", "#496343", "#77745a", "#a38f70", "#d4c9b4"],
    )
    colors = light.shade(elevation, cmap=terrain_cmap, vert_exag=1.6)
    fire_colors = plt.get_cmap("autumn")(np.clip((intensity - 18.0) / 62.0, 0.0, 1.0))
    fire_colors[..., :3] = np.maximum(fire_colors[..., :3], (0.98, 0.18, 0.01))
    active = phase == 1
    colors[active] = fire_colors[active]
    burned = phase == 2
    colors[burned, :3] = colors[burned, :3] * 0.28

    figure = plt.figure(figsize=(13.6, 8.2), facecolor="#101316", layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#101316")
    axis.plot_surface(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        elevation[::stride, ::stride],
        facecolors=colors[::stride, ::stride],
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    active_y, active_x = np.where(active)
    if active_x.size:
        active_z = elevation[active_y, active_x] + 14.0
        axis.scatter(
            active_x,
            active_y,
            active_z,
            c=np.clip(intensity[active_y, active_x], 20.0, 80.0),
            cmap="autumn",
            vmin=20.0,
            vmax=80.0,
            s=7,
            alpha=0.92,
            depthshade=False,
        )
    resource_ids = replay.metadata["resource_ids"]
    resource_kinds = replay.metadata["resource_kinds"]
    for resource_index, (resource_id, kind) in enumerate(zip(resource_ids, resource_kinds, strict=True)):
        trail_start = max(0, index - 45)
        trail_x = np.asarray(states["resources/x"][trail_start : index + 1, resource_index])
        trail_y = np.asarray(states["resources/y"][trail_start : index + 1, resource_index])
        ix = np.clip(np.rint(trail_x).astype(int), 0, width - 1)
        iy = np.clip(np.rint(trail_y).astype(int), 0, height - 1)
        altitude = elevation[iy, ix] + {"retardant": 410.0, "water": 270.0, "sensor": 560.0}[kind]
        color = {"retardant": "#f7f0db", "water": "#72c9ff", "sensor": "#f4d35e"}[kind]
        axis.plot(trail_x, trail_y, altitude, color=color, linewidth=1.7)

    axis.set_title(
        f"Terrain replay  •  T+{minute:03d} min",
        loc="left",
        color="#f2f5f7",
        fontsize=15,
        weight="medium",
        pad=10,
    )
    axis.view_init(elev=45, azim=-128)
    axis.set_box_aspect((width, height, max(width, height) * 0.34))
    axis.set_axis_off()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return destination_path


def render_video(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 120,
) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] to render replay videos") from exc
    import tempfile

    count = replay.frame_count
    selected = np.unique(np.linspace(0, count - 1, min(count, max_frames), dtype=int))
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aeolus-replay-") as temporary:
        temp_root = Path(temporary)
        with imageio.get_writer(
            destination_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for output_index, frame_index in enumerate(selected):
                frame_path = temp_root / f"frame-{output_index:05d}.png"
                render_frame_2d(replay, frame_path, frame=int(frame_index), dpi=110)
                writer.append_data(imageio.imread(frame_path))
    return destination_path
