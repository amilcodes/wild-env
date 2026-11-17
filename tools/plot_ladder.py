"""Render the ladder result figure from a run's summary.json (static PNG for the README).

    python tools/plot_ladder.py results/ladder/evaluation-v1/summary.json docs/assets/ladder/ladder-scores.png

Left: overall score (equal weight per instance) with stratified bootstrap 95% CI, sorted.
Right: score by tier, one facet per model family, baselines dashed in gray.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8d8c86"
# Validated categorical slots 1-4 (light), assigned by role so a color means the same role in both facets.
ROLE_COLOR = {"small": "#2a78d6", "mid": "#eb6834", "frontier-a": "#1baf7a", "frontier-b": "#eda100"}
FAMILIES = {
    "Claude": ["haiku-5.5", "sonnet-5.5", "opus-5.5", "fable-5.2"],
    "GPT": ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra"],
}
ROLES = ["small", "mid", "frontier-a", "frontier-b"]
BASELINES = {"baseline:search": "search (256 samples)", "baseline:greedy": "greedy"}
TIER_BASELINES = {"baseline:search": "search"}


def main(summary_path: str, out_path: str) -> None:
    summary = json.loads(Path(summary_path).read_text())
    rows = {r["policy"]: r for r in summary["leaderboard"]}
    tiers = [int(t) for t in summary["tiers"]]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )
    fig = plt.figure(figsize=(13, 5.2), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.32)

    # Overall dot plot.
    ax = fig.add_subplot(grid[0])
    ax.set_facecolor(SURFACE)
    shown = [
        r
        for r in summary["leaderboard"]
        if r["kind"] == "model" or r["policy"] in BASELINES or r["policy"] == "reference:open_loop"
    ]
    shown.sort(key=lambda r: r["mean_score"])
    for y, r in enumerate(shown):
        lo, hi = r["ci95"]
        model = r["kind"] == "model"
        color = INK if model else NEUTRAL
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", alpha=0.55 if model else 0.8)
        ax.plot(r["mean_score"], y, "o", ms=8, color=color, mec=SURFACE, mew=2)
        ax.text(hi + 0.6, y, f"{r['mean_score']:.1f}", va="center", fontsize=9, color=INK_2)
    labels = [r["policy"].replace("baseline:", "").replace("reference:", "") for r in shown]
    ax.set_yticks(range(len(shown)), labels)
    for tick, r in zip(ax.get_yticklabels(), shown, strict=True):
        tick.set_color(INK if r["kind"] == "model" else NEUTRAL)
    ax.set_xlabel("% of proven optimum (mean, 95% CI)")
    ax.set_title("Overall", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    left = min(r["ci95"][0] for r in shown)
    ax.set_xlim(max(0, left - 3), 104)

    # By tier, faceted by family.
    low = min(
        min(r["by_tier"].values())
        for r in rows.values()
        if r["kind"] == "model" or r["policy"] in TIER_BASELINES
    )
    for k, (family, names) in enumerate(FAMILIES.items()):
        ax = fig.add_subplot(grid[k + 1])
        ax.set_facecolor(SURFACE)
        for name, label in TIER_BASELINES.items():
            if name in rows:
                ys = [rows[name]["by_tier"].get(str(t), float("nan")) for t in tiers]
                ax.plot(tiers, ys, color=NEUTRAL, lw=1.5, ls=(0, (4, 3)))
                ax.text(tiers[-1] + 0.15, ys[-1], label, color=NEUTRAL, fontsize=8, va="center")
        for role, name in zip(ROLES, names, strict=True):
            if name not in rows:
                continue
            ys = [rows[name]["by_tier"].get(str(t), float("nan")) for t in tiers]
            ax.plot(
                tiers, ys, color=ROLE_COLOR[role], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=name
            )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.45, -0.1),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            labelcolor=INK,
            handlelength=1.6,
        )
        ax.axvspan(5.5, tiers[-1] + 0.4, color=GRID, alpha=0.35, lw=0)
        ax.text(5.6, max(low - 4, 0) + 1, "packing tiers", color=INK_2, fontsize=8)
        ax.set_xticks(tiers, [f"T{t}" for t in tiers])
        ax.set_xlim(tiers[0] - 0.3, tiers[-1] + 1.1)
        ax.set_ylim(max(low - 4, 0), 101.5)
        ax.set_title(f"{family} models by tier", loc="left")
        ax.set_ylabel("% of optimum" if k == 0 else "")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        f"Wildfire ops ladder ({summary['instances']} evaluation instances, one call per model per instance)",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
