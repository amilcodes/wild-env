"""Render a paired rollout from the fire-coupled tensor MARL environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from torch import Tensor

from aeolus.config import load_config
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.policies import incident_risk_greedy

# Palette and map symbology follow the native viewer (aeolus.viewer.scene2d).
BACKGROUND = "#0b1014"
FOREGROUND = "#e8edf0"
MUTED = "#8f9ba3"
SPINE = "#344048"
GRID = "#1d272e"
TERRAIN = ["#102119", "#23402a", "#42583b", "#69674c", "#8d7657", "#c1b29a"]
SITE = "#d7e1e6"
SITE_LABEL = "#c9d3d8"
ASSET = "#f1d36b"
BURNED = "#140e0b"
BURN_EDGE = "#ff8a3d"
WATER_FILL = "#0f8cf5"
HOLD = "#b4bdc3"
GREEDY = "#57b9ff"
SITE_MARKER = {
    "airport": "P",
    "retardant_base": "s",
    "helibase": "H",
    "dip_site": "v",
    "scoopable_water": "v",
    "temporary_tank": "h",
}
HOLD_LABEL = "No dispatch"
GREEDY_LABEL = "Greedy baseline"

# Figure layout in inches: two maps over two time series, with the per-world
# comparison in a third column.
FIGURE_SIZE = (13.6, 9.3)
MAP_SIZE = 4.55
COLUMN_X = (0.62, 5.82, 10.66)
CHART_BOTTOM = 0.62
CHART_HEIGHT = 1.95
MAP_BOTTOM = 3.42
WORLD_WIDTH = 2.3


@dataclass
class Rollout:
    env: TensorIncidentEnv
    initial: dict[str, np.ndarray]
    final: dict[str, np.ndarray]
    tracks: np.ndarray
    minutes: np.ndarray
    expected_loss: np.ndarray
    burned_fraction: np.ndarray
    delivered_l: np.ndarray
    wasted_l: np.ndarray
    returns: np.ndarray


def _array(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _snapshot(env: TensorIncidentEnv) -> dict[str, np.ndarray]:
    state = env.state
    return {
        name: _array(getattr(state, name))
        for name in (
            "fuel_factor",
            "asset_value",
            "barrier",
            "burning",
            "burned",
            "belief_burning",
            "belief_uncertainty",
            "water_line_strength",
            "retardant_line_strength",
            "resource_xy",
        )
    }


@torch.no_grad()
def _rollout(
    config_path: Path,
    *,
    policy: str,
    batch_size: int,
    grid_size: int,
    segments: int,
    steps: int,
    seed: int,
) -> Rollout:
    scenario = load_config(config_path).scenario
    env = TensorIncidentEnv(
        scenario,
        batch_size=batch_size,
        max_segments=segments,
        grid_size=grid_size,
        fire_substeps=2,
        device="cpu",
        terminate_on_completion=False,
        terminate_on_escape=False,
    )
    env.reset(seed=seed)
    initial = _snapshot(env)
    tracks = [_array(env.state.resource_xy)]
    minutes = [0.0]
    loss, burned, _ = env._outcome_metrics(env.state)
    expected_loss = [_array(loss)]
    burned_fraction = [_array(burned)]
    delivered = [np.zeros(batch_size)]
    wasted = [np.zeros(batch_size)]
    returns = [np.zeros(batch_size)]

    delivered_total = torch.zeros(batch_size)
    wasted_total = torch.zeros(batch_size)
    return_total = torch.zeros(batch_size)
    for _ in range(steps):
        if policy == "hold":
            actions = torch.zeros((batch_size, env.num_resources), dtype=torch.long)
        else:
            actions = incident_risk_greedy(env)
        transition = env.step(actions)
        delivered_total += transition.delivered_l
        wasted_total += transition.wasted_l
        return_total += transition.reward
        tracks.append(_array(env.state.resource_xy))
        minutes.append(float(env.state.minute[0]))
        expected_loss.append(_array(transition.expected_loss))
        burned_fraction.append(_array(transition.burned_fraction))
        delivered.append(_array(delivered_total))
        wasted.append(_array(wasted_total))
        returns.append(_array(return_total))

    return Rollout(
        env=env,
        initial=initial,
        final=_snapshot(env),
        tracks=np.stack(tracks),
        minutes=np.asarray(minutes),
        expected_loss=np.stack(expected_loss),
        burned_fraction=np.stack(burned_fraction),
        delivered_l=np.stack(delivered),
        wasted_l=np.stack(wasted),
        returns=np.stack(returns),
    )


def _rgba(color: str, alpha: np.ndarray) -> np.ndarray:
    result = np.empty((*alpha.shape, 4), dtype=float)
    result[..., :3] = to_rgb(color)
    result[..., 3] = np.clip(alpha, 0.0, 1.0)
    return result


def _sorties(tracks: np.ndarray, site_xy: np.ndarray, *, radius: float = 0.75) -> np.ndarray:
    """Count departures from each service site for one world.

    ``tracks`` holds one position per decision step and resource, so a sortie is
    a step at a site followed by a step away from every site.
    """

    counts = np.zeros(len(site_xy), dtype=int)
    for resource_track in np.moveaxis(tracks, 1, 0):
        distance = np.linalg.norm(resource_track[:, None, :] - site_xy[None], axis=-1)
        at_site = np.where(distance.min(axis=1) < radius, distance.argmin(axis=1), -1)
        departed = (at_site[:-1] >= 0) & (at_site[1:] < 0)
        np.add.at(counts, at_site[:-1][departed], 1)
    return counts


def _site_label_offset(
    text: str,
    xy: np.ndarray,
    avoid: np.ndarray,
    *,
    fontsize: float,
) -> tuple[float, float, str, str]:
    """Pick the label corner around a site marker that overlaps ``avoid`` the least."""

    size = avoid.shape[0]
    width = len(text) * 0.6 * fontsize / 72.0 / (MAP_SIZE / size)
    height = 1.3 * fontsize / 72.0 / (MAP_SIZE / size)
    candidates = ((1.4, 1.8, "left", "top"), (1.4, -1.2, "left", "bottom"), (-1.4, 1.8, "right", "top"))
    best = candidates[0]
    best_cost = float("inf")
    for dx, dy, ha, va in candidates:
        x0 = xy[0] + dx - (width if ha == "right" else 0.0)
        y0 = xy[1] + dy - (height if va == "bottom" else 0.0)
        if x0 < 0.0 or x0 + width > size - 1 or y0 < 0.0 or y0 + height > size - 1:
            continue
        cost = avoid[int(y0) : int(np.ceil(y0 + height)), int(x0) : int(np.ceil(x0 + width))].sum()
        if cost < best_cost:
            best, best_cost = (dx, dy, ha, va), cost
    return best


def _label_box(alpha: float = 0.75) -> dict[str, object]:
    return {"boxstyle": "round,pad=0.25", "facecolor": BACKGROUND, "edgecolor": "none", "alpha": alpha}


def _map(
    ax: plt.Axes,
    rollout: Rollout,
    *,
    world: int,
    title: str,
    show_sorties: bool,
    label_context: bool,
) -> None:
    env = rollout.env
    snapshot = rollout.final
    size = env.grid_size
    cell_ha = float(env.cell_area_m2) / 10_000.0

    fuel = snapshot["fuel_factor"][world]
    terrain = LinearSegmentedColormap.from_list("aeolus-terrain", TERRAIN)
    span = np.ptp(fuel)
    shade = terrain((fuel - fuel.min()) / span if span > 0 else np.zeros_like(fuel))[..., :3]
    shade = 0.74 * shade + 0.26 * np.asarray(to_rgb(BACKGROUND))
    ax.imshow(shade, origin="upper", interpolation="bilinear", zorder=0)

    assets = snapshot["asset_value"][world]
    if assets.max() > 0.45:
        ax.contourf(assets, levels=[0.45, assets.max() + 1.0], colors=[ASSET], alpha=0.10, zorder=2)
        ax.contour(assets, levels=[0.45], colors=[ASSET], linewidths=1.0, alpha=0.9, zorder=3)

    burned = snapshot["burned"][world]
    ax.imshow(_rgba(BURNED, 0.85 * burned), origin="upper", interpolation="bilinear", zorder=4)
    burning = snapshot["burning"][world]
    if burning.max() > 0.05:
        ax.imshow(_rgba(BURN_EDGE, 0.9 * burning), origin="upper", interpolation="bilinear", zorder=5)

    water = snapshot["water_line_strength"][world]
    if water.max() > 0.025:
        alpha = np.where(water > 0.025, np.clip(water / max(water.max(), 0.15) * 0.9, 0.25, 0.9), 0.0)
        ax.imshow(_rgba(WATER_FILL, alpha), origin="upper", interpolation="nearest", zorder=6)

    if burned.max() > 0.5:
        ax.contour(burned, levels=[0.5], colors=[BURN_EDGE], linewidths=1.3, zorder=7)
    ignition = rollout.initial["burning"][world]
    ax.contour(
        ignition,
        levels=[0.5],
        colors=[FOREGROUND],
        linewidths=0.9,
        linestyles=[(0, (2.0, 1.6))],
        zorder=8,
    )

    rows, cols = np.nonzero(burned > 0.5)

    site_xy = _array(env.site_xy)
    if show_sorties:
        if water.max() > 0.025:
            weights = water / water.sum()
            hub = np.array([(weights * _array(env.grid_x)).sum(), (weights * _array(env.grid_y)).sum()])
            water_rows, water_cols = np.nonzero(water > 0.1)
            ax.text(
                water_cols.min() - 1.0,
                water_rows.min() + 0.5,
                "water drops",
                color=GREEDY,
                fontsize=9,
                ha="right",
                va="center",
                bbox=_label_box(),
                zorder=13,
            )
        else:
            hub = np.array([cols.mean(), rows.mean()]) if rows.size else np.full(2, size / 2.0)
        counts = _sorties(rollout.tracks[:, world], site_xy)
        for xy, count in zip(site_xy, counts):
            if count == 0:
                continue
            ax.plot(
                [xy[0], hub[0]],
                [xy[1], hub[1]],
                color=GREEDY,
                linewidth=0.9 + 0.5 * np.sqrt(count),
                alpha=0.85,
                solid_capstyle="round",
                zorder=9,
            )
            label_xy = xy + 0.4 * (hub - xy)
            ax.text(
                label_xy[0],
                label_xy[1],
                f"{count}",
                color=FOREGROUND,
                fontsize=8.5,
                ha="center",
                va="center",
                bbox=_label_box(0.9),
                zorder=13,
            )
        delivered = rollout.delivered_l[-1, world] / 1000.0
        wasted = rollout.wasted_l[-1, world] / 1000.0
        ax.text(
            0.03,
            0.03,
            f"Numbers are sorties flown from each site.\n"
            f"{delivered:.0f} kL dropped, about {wasted:.0f} kL estimated off target.",
            transform=ax.transAxes,
            color=SITE_LABEL,
            fontsize=8.5,
            linespacing=1.5,
            va="bottom",
            bbox=_label_box(0.85),
            zorder=13,
        )

    for site, xy in zip(env.sites, site_xy):
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=70,
            marker=SITE_MARKER.get(site.kind, "P"),
            facecolor=SITE,
            edgecolor=BACKGROUND,
            linewidth=1.0,
            zorder=11,
        )
        if label_context:
            dx, dy, ha, va = _site_label_offset(site.site_id, xy, assets > 0.45, fontsize=8)
            ax.text(
                xy[0] + dx,
                xy[1] + dy,
                site.site_id,
                color=SITE_LABEL,
                fontsize=8,
                ha=ha,
                va=va,
                zorder=12,
            )

    if label_context:
        asset_rows, asset_cols = np.nonzero(assets > 0.45)
        if asset_rows.size:
            left = int(np.argmin(asset_cols))
            ax.text(
                asset_cols[left] - 1.0,
                asset_rows[left],
                "assets",
                color=ASSET,
                fontsize=9,
                ha="right",
                va="center",
                zorder=12,
            )
        if rows.size:
            ax.text(
                cols.mean(),
                rows.max() + 1.6,
                "burned area",
                color=BURN_EDGE,
                fontsize=9,
                va="top",
                zorder=12,
            )
        ignition_rows, ignition_cols = np.nonzero(ignition > 0.5)
        if ignition_rows.size:
            ax.annotate(
                "ignition",
                xy=(ignition_cols.min() - 0.6, ignition_rows.min() - 0.6),
                xytext=(ignition_cols.min() - 5.5, ignition_rows.min() - 5.0),
                color=FOREGROUND,
                fontsize=9,
                ha="right",
                va="center",
                arrowprops={"arrowstyle": "-", "color": FOREGROUND, "lw": 0.7},
                zorder=13,
            )
        scale_km = 2.0
        scale_cells = scale_km * 1000.0 / float(env.surrogate_cell_m)
        x1, y0 = size - 3.0, size - 3.5
        ax.plot([x1 - scale_cells, x1], [y0, y0], color=FOREGROUND, linewidth=2.0, zorder=12)
        ax.text(
            x1 - scale_cells / 2.0,
            y0 - 1.0,
            f"{scale_km:.0f} km",
            color=FOREGROUND,
            fontsize=8.5,
            ha="center",
            va="bottom",
            zorder=12,
        )

    ax.set_title(title, loc="left", fontsize=13, color=FOREGROUND, pad=8, fontweight="semibold")
    ax.set_title(
        f"T+{rollout.minutes[-1]:.0f} min, {burned.sum() * cell_ha:.0f} ha burned",
        loc="right",
        fontsize=10.5,
        color=MUTED,
        pad=8,
    )
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def _style_chart(ax: plt.Axes, title: str, subtitle: str, *, spines: tuple[str, ...]) -> None:
    ax.set_title(title, loc="left", fontsize=13, color=FOREGROUND, pad=24, fontweight="semibold")
    ax.annotate(
        subtitle,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        color=MUTED,
        fontsize=9.5,
        va="bottom",
    )
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_color(SPINE)


def _series_panel(
    ax: plt.Axes,
    hold: Rollout,
    greedy: Rollout,
    *,
    field: str,
    title: str,
    ylabel: str,
) -> None:
    worlds = getattr(hold, field).shape[1]
    top = 0.0
    for rollout, label, color in ((hold, HOLD_LABEL, HOLD), (greedy, GREEDY_LABEL, GREEDY)):
        mean = getattr(rollout, field).mean(axis=1) * 100.0
        top = max(top, float(mean.max()))
        ax.plot(rollout.minutes, mean, color=color, linewidth=2.2, solid_capstyle="round")
        ax.annotate(
            label,
            xy=(rollout.minutes[-1], mean[-1]),
            xytext=(0, 6),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            ha="right",
            va="bottom",
        )
    ax.set_xlim(0, hold.minutes[-1])
    ax.set_ylim(0, 1.18 * top)
    ax.set_xticks(np.arange(0, hold.minutes[-1] + 1, 30))
    ax.set_xlabel("minutes", color=MUTED, fontsize=9.5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style_chart(ax, title, f"Mean of {worlds} worlds", spines=("left", "bottom"))


def _world_panel(ax: plt.Axes, hold: Rollout, greedy: Rollout, *, world: int) -> None:
    hold_loss = hold.expected_loss[-1] * 100.0
    greedy_loss = greedy.expected_loss[-1] * 100.0
    order = np.argsort(hold_loss)
    rows = np.arange(len(order))
    for row, index in zip(rows, order):
        if index == world:
            ax.axhspan(row - 0.5, row + 0.5, color="#151e24", linewidth=0, zorder=0)
        ax.plot(
            [greedy_loss[index], hold_loss[index]],
            [row, row],
            color="#4b5a64",
            linewidth=1.8,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(hold_loss[order], rows, s=34, color=HOLD, edgecolor=BACKGROUND, linewidth=0.6, zorder=3)
    ax.scatter(greedy_loss[order], rows, s=34, color=GREEDY, edgecolor=BACKGROUND, linewidth=0.6, zorder=3)

    # Label the two dots of the top row: one beside its dot, one above it.
    top = order[-1]
    ax.annotate(
        GREEDY_LABEL,
        xy=(greedy_loss[top], rows[-1]),
        xytext=(-8, 0),
        textcoords="offset points",
        color=GREEDY,
        fontsize=9.5,
        ha="right",
        va="center",
    )
    ax.annotate(
        HOLD_LABEL,
        xy=(hold_loss[top], rows[-1]),
        xytext=(4, 9),
        textcoords="offset points",
        color=HOLD,
        fontsize=9.5,
        ha="right",
        va="bottom",
    )
    ax.set_yticks(rows, [f"world {index}" for index in order])
    ax.yaxis.tick_right()
    for tick, index in zip(ax.get_yticklabels(), order):
        if index == world:
            tick.set_color(FOREGROUND)
            tick.set_fontweight("semibold")
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(-0.06 * hold_loss.max(), 1.12 * hold_loss.max())
    ax.set_xlabel("% of asset value", color=MUTED, fontsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    lower = int((greedy_loss < hold_loss).sum())
    _style_chart(
        ax,
        "Expected loss by world",
        f"T+{hold.minutes[-1]:.0f} min, lower in {lower} of {len(order)}",
        spines=("bottom",),
    )


def _axes(figure: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    figure_width, figure_height = FIGURE_SIZE
    return figure.add_axes(
        (left / figure_width, bottom / figure_height, width / figure_width, height / figure_height)
    )


def render(args: argparse.Namespace) -> None:
    hold = _rollout(
        args.config,
        policy="hold",
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        segments=args.segments,
        steps=args.steps,
        seed=args.seed,
    )
    greedy = _rollout(
        args.config,
        policy="greedy",
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        segments=args.segments,
        steps=args.steps,
        seed=args.seed,
    )
    loss_reduction = hold.expected_loss[-1] - greedy.expected_loss[-1]
    world = int(np.argmax(loss_reduction)) if args.world is None else args.world
    if not 0 <= world < args.batch_size:
        raise ValueError("world index is outside the rollout batch")

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": "DejaVu Sans",
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=200)
    map_top = MAP_BOTTOM + MAP_SIZE
    hold_ax = _axes(figure, COLUMN_X[0], MAP_BOTTOM, MAP_SIZE, MAP_SIZE)
    greedy_ax = _axes(figure, COLUMN_X[1], MAP_BOTTOM, MAP_SIZE, MAP_SIZE)
    loss_ax = _axes(figure, COLUMN_X[0], CHART_BOTTOM, MAP_SIZE, CHART_HEIGHT)
    burn_ax = _axes(figure, COLUMN_X[1], CHART_BOTTOM, MAP_SIZE, CHART_HEIGHT)
    # The chart title sits 16 pt higher than a map title; drop the axes to match.
    world_ax = _axes(figure, COLUMN_X[2], CHART_BOTTOM, WORLD_WIDTH, map_top - 16 / 72 - CHART_BOTTOM)

    _map(
        hold_ax,
        hold,
        world=world,
        title=HOLD_LABEL,
        show_sorties=False,
        label_context=True,
    )
    _map(
        greedy_ax,
        greedy,
        world=world,
        title=GREEDY_LABEL,
        show_sorties=True,
        label_context=False,
    )
    _series_panel(
        loss_ax,
        hold,
        greedy,
        field="expected_loss",
        title="Expected asset loss",
        ylabel="% of asset value",
    )
    _series_panel(
        burn_ax,
        hold,
        greedy,
        field="burned_fraction",
        title="Burned area",
        ylabel="% of burnable area",
    )
    _world_panel(world_ax, hold, greedy, world=world)

    figure_height = FIGURE_SIZE[1]
    figure.text(
        COLUMN_X[0] / FIGURE_SIZE[0],
        (map_top + 0.95) / figure_height,
        "Paired rollout in the tensor incident environment",
        color=FOREGROUND,
        fontsize=20,
        fontweight="semibold",
        va="baseline",
    )
    selection = (
        f"Maps show world {world}, where the greedy baseline reduced expected loss the most."
        if args.world is None
        else f"Maps show world {world}."
    )
    figure.text(
        COLUMN_X[0] / FIGURE_SIZE[0],
        (map_top + 0.58) / figure_height,
        (
            f"{args.batch_size} worlds with {greedy.env.num_resources} aircraft each, "
            f"{args.steps} decisions over {greedy.minutes[-1]:.0f} minutes, seed {args.seed}. {selection}"
        ),
        color=MUTED,
        fontsize=11,
        va="baseline",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.3, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cluster_tensor_incident.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/rl/tensor-incident-rollout.png"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7701)
    parser.add_argument("--world", type=int)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
