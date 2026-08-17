"""Render a paired rollout from the fire-coupled tensor MARL environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from torch import Tensor

from aeolus.config import load_config
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.policies import incident_risk_greedy

BACKGROUND = "#0d1217"
FOREGROUND = "#e7edf2"
MUTED = "#96a5b3"
HOLD = "#d6a85f"
GREEDY = "#48b9e6"
WATER = "#38a9e5"
RETARDANT = "#ea3b8b"


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
    rgb = np.asarray(plt.matplotlib.colors.to_rgb(color))
    result = np.empty((*alpha.shape, 4), dtype=float)
    result[..., :3] = rgb
    result[..., 3] = np.clip(alpha, 0.0, 1.0)
    return result


def _map(
    ax: plt.Axes,
    snapshot: dict[str, np.ndarray],
    *,
    world: int,
    title: str,
    env: TensorIncidentEnv,
    tracks: np.ndarray | None = None,
) -> None:
    fuel = snapshot["fuel_factor"][world]
    terrain_cmap = LinearSegmentedColormap.from_list(
        "fuel",
        ["#101a16", "#243a27", "#526747", "#8a7c58", "#b6a980"],
    )
    ax.imshow(fuel, origin="upper", cmap=terrain_cmap, vmin=0.35, vmax=1.55)
    ax.imshow(_rgba("#080a0c", 0.70 * snapshot["burned"][world]), origin="upper")
    ax.imshow(
        _rgba("#ff6b1a", np.clip(snapshot["burning"][world] * 1.7, 0.0, 0.95)),
        origin="upper",
    )
    water = snapshot["water_line_strength"][world]
    retardant = snapshot["retardant_line_strength"][world]
    if water.max() > 0.05:
        ax.contour(water, levels=[0.12, 0.45], colors=[WATER, "#a8e7ff"], linewidths=[1.0, 1.8])
    if retardant.max() > 0.05:
        ax.contour(
            retardant,
            levels=[0.12, 0.45],
            colors=[RETARDANT, "#ffabd1"],
            linewidths=[1.0, 1.8],
        )
    belief = snapshot["belief_burning"][world]
    if belief.max() > 0.05:
        ax.contour(belief, levels=[0.12], colors="#b9efff", linewidths=1.0, linestyles="--")
    assets = snapshot["asset_value"][world]
    if assets.max() > 0.25:
        ax.contour(assets, levels=[0.45, 0.75], colors=["#d7ba61", "#ffe17c"], linewidths=0.8)

    site_xy = _array(env.site_xy)
    ax.scatter(
        site_xy[:, 0],
        site_xy[:, 1],
        marker="s",
        s=30,
        facecolors=BACKGROUND,
        edgecolors="#dfe7ed",
        linewidths=0.9,
        zorder=7,
    )
    if tracks is not None:
        for resource_index, kind in enumerate(env.resource_kind_names):
            color = WATER if kind == "water" else RETARDANT
            xy = tracks[:, world, resource_index]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.8, alpha=0.52, zorder=5)
        current = snapshot["resource_xy"][world]
        colors = [WATER if kind == "water" else RETARDANT for kind in env.resource_kind_names]
        ax.scatter(
            current[:, 0],
            current[:, 1],
            c=colors,
            marker="^",
            s=32,
            edgecolors=BACKGROUND,
            linewidths=0.8,
            zorder=8,
        )

    ax.set_title(title, loc="left", fontsize=12, color=FOREGROUND, pad=8, fontweight="semibold")
    ax.set_xlim(0, env.grid_size - 1)
    ax.set_ylim(env.grid_size - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#3b4852")


def _line_panel(
    ax: plt.Axes,
    hold: Rollout,
    greedy: Rollout,
    *,
    field: str,
    title: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    for rollout, label, color in ((hold, "hold", HOLD), (greedy, "greedy baseline", GREEDY)):
        values = getattr(rollout, field) * scale
        mean = values.mean(axis=1)
        low = np.quantile(values, 0.10, axis=1)
        high = np.quantile(values, 0.90, axis=1)
        ax.fill_between(rollout.minutes, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(rollout.minutes, mean, color=color, linewidth=2.0, label=label)
    ax.set_title(title, loc="left", fontsize=11, color=FOREGROUND, fontweight="semibold")
    ax.set_xlabel("incident minute", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color="#29343d", linewidth=0.7, alpha=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3b4852")


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
    figure = plt.figure(figsize=(16, 9), dpi=160)
    grid = figure.add_gridspec(2, 6, height_ratios=[1.65, 1.0], hspace=0.32, wspace=0.34)
    initial_ax = figure.add_subplot(grid[0, 0:2])
    hold_ax = figure.add_subplot(grid[0, 2:4])
    greedy_ax = figure.add_subplot(grid[0, 4:6])
    loss_ax = figure.add_subplot(grid[1, 0:2])
    burn_ax = figure.add_subplot(grid[1, 2:4])
    operations_ax = figure.add_subplot(grid[1, 4:6])

    final_minute = int(greedy.minutes[-1])
    _map(initial_ax, greedy.initial, world=world, title="INITIAL STATE  ·  T+0", env=greedy.env)
    _map(
        hold_ax,
        hold.final,
        world=world,
        title=f"HOLD COMPARATOR  ·  T+{final_minute}",
        env=hold.env,
    )
    _map(
        greedy_ax,
        greedy.final,
        world=world,
        title=f"BELIEF-ONLY GREEDY  ·  T+{final_minute}",
        env=greedy.env,
        tracks=greedy.tracks,
    )

    _line_panel(
        loss_ax,
        hold,
        greedy,
        field="expected_loss",
        title="Asset-weighted expected loss",
        ylabel="affected asset mass [%]",
        scale=100.0,
    )
    _line_panel(
        burn_ax,
        hold,
        greedy,
        field="burned_fraction",
        title="Cumulative burned area",
        ylabel="burnable domain [%]",
        scale=100.0,
    )

    x = np.arange(2)
    delivered = greedy.delivered_l[-1].mean() / 1000.0
    wasted = greedy.wasted_l[-1].mean() / 1000.0
    operations_ax.bar(x, [delivered, wasted], color=[GREEDY, "#d9665b"], width=0.55)
    operations_ax.set_xticks(x, ["delivered", "off-target estimate"])
    operations_ax.set_ylabel("mean volume per world [kL]", color=MUTED)
    operations_ax.set_title(
        "Aerial workload",
        loc="left",
        fontsize=11,
        color=FOREGROUND,
        fontweight="semibold",
    )
    operations_ax.tick_params(colors=MUTED, labelsize=8)
    operations_ax.grid(axis="y", color="#29343d", linewidth=0.7, alpha=0.7)
    for spine in operations_ax.spines.values():
        spine.set_color("#3b4852")

    figure.suptitle(
        "Fire-coupled tensor MARL environment",
        x=0.055,
        y=0.982,
        ha="left",
        color=FOREGROUND,
        fontsize=22,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.945,
        (
            f"Paired {args.batch_size}-world CPU rollout  ·  {greedy.env.num_resources} agents/world  ·  "
            f"{args.steps} decisions  ·  seed {args.seed}  ·  detailed world {world}"
        ),
        color=MUTED,
        fontsize=10,
    )
    legend = [
        Line2D([0], [0], color="#ff6b1a", lw=3, label="active fire truth"),
        Line2D([0], [0], color="#b9efff", lw=1.5, ls="--", label="delayed belief front"),
        Line2D([0], [0], color=WATER, lw=2, label="water / water agents"),
        Line2D([0], [0], color=RETARDANT, lw=2, label="retardant / retardant agents"),
        Line2D([0], [0], color="#ffe17c", lw=1.5, label="asset contours"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        labelcolor=MUTED,
        fontsize=8,
    )
    figure.text(
        0.945,
        0.028,
        "Comparator visualization; no learned-policy claim",
        color="#75838e",
        fontsize=7.5,
        ha="right",
    )
    figure.subplots_adjust(top=0.89, bottom=0.10, left=0.055, right=0.97)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
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
