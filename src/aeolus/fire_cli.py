"""Inspection, validation, and throughput tools for the fire-behavior core."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aeolus.config import FireBehaviorConfig
from aeolus.core.fire_behavior import fire_behavior_lookup
from aeolus.core.front import advance_level_set
from aeolus.core.state import FirePhase, FireType
from aeolus.core.tensor_fire import TensorFireKernel, make_synthetic_batch


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _point(args: argparse.Namespace) -> dict[str, Any]:
    shape = (1, 1)

    def full(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    behavior = fire_behavior_lookup().resolve_numpy(
        fuel_model_number=np.full(shape, args.fuel_model, dtype=np.int16),
        moisture_dead_1h=full(args.moisture),
        moisture_live_herbaceous=full(args.live_herbaceous_moisture),
        moisture_live_woody=full(args.live_woody_moisture),
        wind_speed_10m_m_s=args.wind,
        wind_from_direction_deg=args.wind_from,
        terrain_slope_x=full(args.slope),
        terrain_slope_y=full(0.0),
        canopy_cover=full(args.canopy_cover),
        canopy_height_m=full(args.canopy_height),
        canopy_base_height_m=full(args.canopy_base_height),
        canopy_bulk_density_kg_m3=full(args.canopy_bulk_density),
        foliar_moisture=full(args.foliar_moisture),
        config=FireBehaviorConfig(),
    )
    type_name = {
        FireType.SURFACE: "surface",
        FireType.PASSIVE_CROWN: "passive_crown",
        FireType.ACTIVE_CROWN: "active_crown",
    }[FireType(int(behavior.fire_type.item()))]
    return {
        "inputs": {
            "fuel_model_number": args.fuel_model,
            "dead_1h_moisture_fraction": args.moisture,
            "live_herbaceous_moisture_fraction": args.live_herbaceous_moisture,
            "live_woody_moisture_fraction": args.live_woody_moisture,
            "wind_speed_10m_m_s": args.wind,
            "wind_from_direction_deg": args.wind_from,
            "slope_rise_run": args.slope,
            "canopy_cover_fraction": args.canopy_cover,
            "canopy_height_m": args.canopy_height,
            "canopy_base_height_m": args.canopy_base_height,
            "canopy_bulk_density_kg_m3": args.canopy_bulk_density,
        },
        "behavior": {
            "fire_type": type_name,
            "heading_spread_rate_m_min": float(behavior.spread_rate_m_min.item()),
            "fireline_intensity_kw_m": float(behavior.fireline_intensity_kw_m.item()),
            "flame_length_m": float(behavior.flame_length_m.item()),
            "heading_unit_xy": [
                float(behavior.head_x.item()),
                float(behavior.head_y.item()),
            ],
            "ellipse_eccentricity": float(behavior.eccentricity.item()),
        },
        "reference_table": fire_behavior_lookup().provenance,
    }


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    state = make_synthetic_batch(
        batch_size=args.batch,
        height=args.height,
        width=args.width,
        device=device,
        moisture_dead_1h=args.moisture,
        cell_size_m=args.cell_size,
    )
    settings = FireBehaviorConfig(enable_spotting=args.spotting)
    kernel = TensorFireKernel(cell_size_m=args.cell_size, config=settings)
    wind = torch.linspace(
        args.wind * 0.75,
        args.wind * 1.25,
        args.batch,
        device=device,
    )
    for minute in range(1, args.warmup + 1):
        kernel.step(
            state,
            minute=minute,
            wind_speed_m_s=wind,
            wind_from_direction_deg=args.wind_from,
        )
    _synchronize(device)
    started = time.perf_counter()
    for offset in range(args.steps):
        kernel.step(
            state,
            minute=args.warmup + offset + 1,
            wind_speed_m_s=wind,
            wind_from_direction_deg=args.wind_from,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    cell_steps = args.batch * args.height * args.width * args.steps
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "batch": args.batch,
        "grid": [args.height, args.width],
        "steps": args.steps,
        "elapsed_s": elapsed,
        "million_cell_steps_s": cell_steps / elapsed / 1e6,
        "environment_steps_s": args.batch * args.steps / elapsed,
        "simulated_fire_minutes_s": args.batch * args.steps / elapsed,
        "resident_state_mib": sum(
            value.numel() * value.element_size()
            for value in vars(state).values()
            if isinstance(value, torch.Tensor)
        )
        / (1024**2),
        "active_or_burned_cells": int((state.phase != 0).sum().item()),
        "spotting": args.spotting,
    }


def _validation_batch(
    device: torch.device,
    size: int,
    minutes: int,
) -> tuple[Any, list[np.ndarray], list[int], torch.Tensor]:
    state = make_synthetic_batch(
        batch_size=4,
        height=size,
        width=size,
        device=device,
        moisture_dead_1h=0.07,
        cell_size_m=30.0,
    )
    state.fuel_model_number[2] = 145
    # A planar 30% east-facing slope for the terrain-coupling case.
    x = torch.arange(size, device=device, dtype=torch.float32)
    state.elevation_m[2] = x[None, :] * 9.0
    state.canopy_cover[3] = 0.72
    state.canopy_height_m[3] = 18.0
    state.canopy_base_height_m[3] = 2.2
    state.canopy_bulk_density_kg_m3[3] = 0.18
    wind = torch.tensor([0.0, 7.0, 5.0, 9.0], device=device)
    direction = torch.tensor([270.0, 270.0, 270.0, 245.0], device=device)
    settings = FireBehaviorConfig(
        enable_spotting=True,
        spotting_embers_per_source_min=0.05,
    )
    kernel = TensorFireKernel(cell_size_m=30.0, config=settings)
    snapshots: list[np.ndarray] = []
    snapshot_minutes: list[int] = []
    capture = {1, max(1, minutes // 3), max(1, 2 * minutes // 3), minutes}
    generator = torch.Generator(device=device)
    generator.manual_seed(20260728)
    for minute in range(1, minutes + 1):
        kernel.step(
            state,
            minute=minute,
            wind_speed_m_s=wind,
            wind_from_direction_deg=direction,
            air_temperature_c=torch.tensor([24.0, 31.0, 34.0, 36.0], device=device),
            relative_humidity_pct=torch.tensor([38.0, 24.0, 18.0, 14.0], device=device),
            generator=generator,
        )
        if minute in capture:
            snapshots.append(state.phase.detach().cpu().numpy().copy())
            snapshot_minutes.append(minute)
    return state, snapshots, snapshot_minutes, wind


def _render_validation_atlas(
    destination: Path,
    state: Any,
    snapshots: list[np.ndarray],
    snapshot_minutes: list[int],
    wind: torch.Tensor,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] for validation graphics") from exc

    labels = [
        "Calm / flat / GS2",
        "Wind-driven / GS2",
        "Wind + 30% slope / SH5",
        "Active crown + spotting / GS2",
    ]
    colors = ListedColormap(["#17241d", "#ff7a19", "#281a17"])
    figure = plt.figure(figsize=(16, 11), facecolor="#0e1215", layout="constrained")
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.88))
    final = state.phase.detach().cpu().numpy()
    fire_type = state.fire_type.detach().cpu().numpy()
    spread = state.spread_rate_m_min.detach().cpu().numpy()
    intensity = state.intensity_kw_m.detach().cpu().numpy()
    for column in range(4):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(final[column], cmap=colors, vmin=0, vmax=2)
        if np.any(fire_type[column] >= int(FireType.PASSIVE_CROWN)):
            axis.contour(
                fire_type[column] >= int(FireType.PASSIVE_CROWN),
                levels=[0.5],
                colors="#f1e6ff",
                linewidths=0.8,
            )
        axis.set_title(labels[column], color="#f1f4f6", fontsize=11, loc="left")
        axis.text(
            0.01,
            -0.08,
            f"{int((final[column] != 0).sum()):,} reached cells  ·  max ROS {spread[column].max():.1f} m/min",
            transform=axis.transAxes,
            color="#9ba8b1",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#34414a")

    timeline_axis = figure.add_subplot(grid[1, :2])
    reached = np.asarray([[(snapshot[case] != 0).sum() for snapshot in snapshots] for case in range(4)])
    for case, label in enumerate(labels):
        timeline_axis.plot(
            snapshot_minutes,
            reached[case],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    timeline_axis.set_title("Resolved growth through time", color="#f1f4f6", fontsize=12, loc="left")
    timeline_axis.set_xlabel("simulation minute", color="#aab4bb")
    timeline_axis.set_ylabel("cells reached", color="#aab4bb")
    timeline_axis.legend(frameon=False, fontsize=8, labelcolor="#dce2e6")

    behavior_axis = figure.add_subplot(grid[1, 2:])
    wind_grid = np.linspace(0.0, 20.0, 101)
    for fuel, label in (
        (101, "GR1"),
        (122, "GS2"),
        (145, "SH5"),
        (183, "TL3"),
        (202, "SB2"),
    ):
        values = []
        for wind_value in wind_grid:
            point_args = argparse.Namespace(
                fuel_model=fuel,
                moisture=0.07,
                wind=float(wind_value),
                wind_from=270.0,
                slope=0.0,
                canopy_cover=0.0,
                canopy_height=0.0,
                canopy_base_height=0.0,
                canopy_bulk_density=0.0,
                foliar_moisture=1.0,
            )
            values.append(_point(point_args)["behavior"]["heading_spread_rate_m_min"])
        behavior_axis.plot(wind_grid, values, linewidth=2.0, label=label)
    behavior_axis.set_yscale("log")
    behavior_axis.set_title("Reference-table heading spread", color="#f1f4f6", fontsize=12, loc="left")
    behavior_axis.set_xlabel("10 m wind speed (m/s)", color="#aab4bb")
    behavior_axis.set_ylabel("heading ROS (m/min, log scale)", color="#aab4bb")
    behavior_axis.legend(frameon=False, ncol=3, fontsize=8, labelcolor="#dce2e6")

    diagnostic = figure.add_subplot(grid[2, :])
    diagnostic.axis("off")
    final_counts = (final != 0).reshape(4, -1).sum(axis=1)
    active_crown = (fire_type == int(FireType.ACTIVE_CROWN)).reshape(4, -1).sum(axis=1)
    lines = [
        "BEHAVIOR DIAGNOSTICS",
        "",
        *[
            f"{labels[index]:<34}  wind {float(wind[index]):>4.1f} m/s"
            f"  · reached {int(final_counts[index]):>6,d}"
            f"  · active-crown cells {int(active_crown[index]):>5,d}"
            f"  · peak intensity {float(intensity[index].max()):>9,.0f} kW/m"
            for index in range(4)
        ],
        "",
        "Surface values: Pyretechnics-derived Rothermel + Scott/Burgan table. "
        "Coupling: vector wind/slope ellipse. Crown: Van Wagner initiation + Cruz spread. "
        "Embers: stochastic lognormal downwind transport.",
    ]
    diagnostic.text(
        0.0,
        0.96,
        "\n".join(lines),
        va="top",
        family="monospace",
        color="#d6dde2",
        fontsize=9.2,
        linespacing=1.5,
    )
    for axis in figure.axes:
        axis.set_facecolor("#11171b")
        axis.tick_params(colors="#8f9da7")
        if hasattr(axis, "spines"):
            for spine in axis.spines.values():
                spine.set_color("#34414a")
    figure.suptitle(
        "Aeolus fire-behavior validation atlas",
        color="#f4f6f7",
        fontsize=18,
        x=0.012,
        ha="left",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def _suite(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    state, snapshots, snapshot_minutes, wind = _validation_batch(device, args.size, args.minutes)
    _synchronize(device)
    atlas_path = destination / "fire_validation_atlas.png"
    _render_validation_atlas(atlas_path, state, snapshots, snapshot_minutes, wind)
    phase = state.phase.detach().cpu().numpy()
    fire_type = state.fire_type.detach().cpu().numpy()
    report = {
        "device": str(device),
        "grid": [args.size, args.size],
        "minutes": args.minutes,
        "cases": [
            {
                "name": name,
                "wind_speed_m_s": float(wind[index].item()),
                "reached_cells": int((phase[index] != int(FirePhase.UNBURNED)).sum()),
                "active_crown_cells": int((fire_type[index] == int(FireType.ACTIVE_CROWN)).sum()),
                "max_spread_rate_m_min": float(state.spread_rate_m_min[index].max().item()),
                "max_intensity_kw_m": float(state.intensity_kw_m[index].max().item()),
            }
            for index, name in enumerate(
                (
                    "calm_flat_gs2",
                    "wind_flat_gs2",
                    "wind_slope_sh5",
                    "active_crown_spotting_gs2",
                )
            )
        ],
        "reference_table": fire_behavior_lookup().provenance,
        "artifacts": {"atlas": str(atlas_path.resolve())},
    }
    report_path = destination / "fire_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["artifacts"]["report"] = str(report_path.resolve())
    return report


def _radial_front_case(
    *,
    cell_size_m: float,
    solver: str,
    duration_min: float = 30.0,
    initial_radius_m: float = 240.0,
    speed_m_min: float = 4.0,
) -> dict[str, float | int | str]:
    domain_radius_m = 720.0
    size = int(np.ceil(2.0 * domain_radius_m / cell_size_m)) + 1
    if size % 2 == 0:
        size += 1
    coordinates = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * cell_size_m
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(x, y)
    phi = (radius - initial_radius_m).astype(np.float32)
    speed = np.full_like(phi, speed_m_min)
    one = np.ones_like(phi)
    zero = np.zeros_like(phi)
    steps = max(
        1,
        int(np.ceil(duration_min * speed_m_min / (0.30 * cell_size_m))),
    )
    dt = duration_min / steps
    for _ in range(steps):
        phi, _ = advance_level_set(
            phi,
            head_rate_m_min=speed,
            head_x=one,
            head_y=zero,
            eccentricity=zero,
            burnable=np.ones_like(phi, dtype=np.bool_),
            cell_size_m=cell_size_m,
            dt_min=dt,
            solver=solver,
            band_width_cells=16.0,
        )
    burned = phi <= 0.0
    expected_radius = initial_radius_m + speed_m_min * duration_min
    equivalent_radius = float(np.sqrt(burned.sum() * cell_size_m**2 / np.pi))
    maximum_radius = float(radius[burned].max())
    return {
        "solver": solver,
        "cell_size_m": cell_size_m,
        "grid_size": size,
        "steps": steps,
        "expected_radius_m": expected_radius,
        "equivalent_radius_m": equivalent_radius,
        "equivalent_radius_error_m": equivalent_radius - expected_radius,
        "maximum_radius_m": maximum_radius,
        "maximum_radius_error_m": maximum_radius - expected_radius,
    }


def _rotation_cases() -> list[dict[str, float]]:
    cell_size_m = 15.0
    size = 101
    initial_radius_m = 120.0
    speed_m_min = 5.0
    duration_min = 24.0
    coordinates = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * cell_size_m
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    initial = (np.hypot(x, y) - initial_radius_m).astype(np.float32)
    speed = np.full_like(initial, speed_m_min)
    eccentricity = np.full_like(initial, 0.65)
    steps = int(np.ceil(duration_min * speed_m_min / (0.30 * cell_size_m)))
    dt = duration_min / steps
    cases: list[dict[str, float]] = []
    for heading_deg in np.arange(0.0, 360.0, 45.0):
        radians = np.deg2rad(heading_deg)
        head_x = np.full_like(initial, np.cos(radians))
        head_y = np.full_like(initial, np.sin(radians))
        phi = initial.copy()
        for _ in range(steps):
            phi, _ = advance_level_set(
                phi,
                head_rate_m_min=speed,
                head_x=head_x,
                head_y=head_y,
                eccentricity=eccentricity,
                burnable=np.ones_like(phi, dtype=np.bool_),
                cell_size_m=cell_size_m,
                dt_min=dt,
                solver="weno5",
                band_width_cells=16.0,
            )
        reached = phi <= 0.0
        projection = x * np.cos(radians) + y * np.sin(radians)
        cases.append(
            {
                "heading_deg": float(heading_deg),
                "heading_extent_m": float(projection[reached].max()),
                "area_km2": float(reached.sum() * cell_size_m**2 / 1_000_000.0),
            }
        )
    return cases


def _render_front_verification(
    destination: Path,
    radial_cases: list[dict[str, float | int | str]],
    rotation_cases: list[dict[str, float]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[render] for verification graphics") from exc

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        facecolor="#0e1215",
        layout="constrained",
    )
    for solver, color in (("godunov", "#e7a75d"), ("weno5", "#67c6d4")):
        selected = [case for case in radial_cases if case["solver"] == solver]
        axes[0].plot(
            [float(case["cell_size_m"]) for case in selected],
            [abs(float(case["equivalent_radius_error_m"])) for case in selected],
            "o-",
            linewidth=2.0,
            label=solver.upper(),
            color=color,
        )
    axes[0].invert_xaxis()
    axes[0].set_xlabel("cell size (m)")
    axes[0].set_ylabel("absolute equivalent-radius error (m)")
    axes[0].set_title(
        "Circular-front grid refinement",
        loc="left",
        color="#f1f4f6",
    )
    axes[0].legend(frameon=False, labelcolor="#dce2e6")

    angles = np.deg2rad([case["heading_deg"] for case in rotation_cases] + [rotation_cases[0]["heading_deg"]])
    extents = [case["heading_extent_m"] for case in rotation_cases] + [rotation_cases[0]["heading_extent_m"]]
    polar = figure.add_subplot(1, 2, 2, projection="polar")
    axes[1].remove()
    polar.plot(angles, extents, "o-", color="#ff7a19", linewidth=2.0)
    polar.fill(angles, extents, color="#ff7a19", alpha=0.18)
    polar.set_title(
        "Heading extent under rotated forcing",
        loc="left",
        color="#f1f4f6",
    )
    for axis in (axes[0], polar):
        axis.set_facecolor("#11171b")
        axis.tick_params(colors="#aab4bb")
        axis.xaxis.label.set_color("#aab4bb")
        axis.yaxis.label.set_color("#aab4bb")
        for spine in axis.spines.values():
            spine.set_color("#34414a")
    figure.suptitle(
        "Level-set numerical verification",
        color="#f4f6f7",
        fontsize=17,
        x=0.01,
        ha="left",
    )
    figure.savefig(destination, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def _verify_front(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    radial_cases = [
        _radial_front_case(cell_size_m=cell, solver=solver)
        for solver in ("godunov", "weno5")
        for cell in (60.0, 30.0, 15.0)
    ]
    rotation_cases = _rotation_cases()
    extents = np.asarray([case["heading_extent_m"] for case in rotation_cases])
    areas = np.asarray([case["area_km2"] for case in rotation_cases])
    figure_path = destination / "front_numerics.png"
    _render_front_verification(figure_path, radial_cases, rotation_cases)
    report = {
        "front_equation": "anisotropic Hamilton-Jacobi",
        "spatial_discretization": "Jiang-Shu WENO5 with Godunov comparator",
        "time_integration": "SSP-RK3",
        "radial_grid_refinement": radial_cases,
        "rotation_invariance": {
            "cases": rotation_cases,
            "heading_extent_coefficient_of_variation": float(extents.std() / extents.mean()),
            "area_coefficient_of_variation": float(areas.std() / areas.mean()),
        },
        "artifacts": {"figure": str(figure_path.resolve())},
    }
    report_path = destination / "front_numerics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["artifacts"]["report"] = str(report_path.resolve())
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and validate the Aeolus fire-behavior core")
    subparsers = parser.add_subparsers(dest="command", required=True)
    point = subparsers.add_parser("point", help="resolve one local fire-behavior case")
    point.add_argument("--fuel-model", type=int, default=122)
    point.add_argument("--moisture", type=float, default=0.07)
    point.add_argument("--live-herbaceous-moisture", type=float, default=0.75)
    point.add_argument("--live-woody-moisture", type=float, default=0.60)
    point.add_argument("--wind", type=float, default=6.0)
    point.add_argument("--wind-from", type=float, default=270.0)
    point.add_argument("--slope", type=float, default=0.0)
    point.add_argument("--canopy-cover", type=float, default=0.0)
    point.add_argument("--canopy-height", type=float, default=0.0)
    point.add_argument("--canopy-base-height", type=float, default=0.0)
    point.add_argument("--canopy-bulk-density", type=float, default=0.0)
    point.add_argument("--foliar-moisture", type=float, default=1.0)

    benchmark = subparsers.add_parser("benchmark", help="measure batched tensor-kernel throughput")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--batch", type=int, default=64)
    benchmark.add_argument("--height", type=int, default=128)
    benchmark.add_argument("--width", type=int, default=128)
    benchmark.add_argument("--cell-size", type=float, default=30.0)
    benchmark.add_argument("--steps", type=int, default=40)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--wind", type=float, default=7.0)
    benchmark.add_argument("--wind-from", type=float, default=270.0)
    benchmark.add_argument("--moisture", type=float, default=0.07)
    benchmark.add_argument("--spotting", action="store_true")
    benchmark.add_argument(
        "--output",
        type=Path,
        help="optional path for the machine-readable benchmark result",
    )

    suite = subparsers.add_parser("validate", help="run idealized cases and render a validation atlas")
    suite.add_argument("--device", default="auto")
    suite.add_argument("--size", type=int, default=128)
    suite.add_argument("--minutes", type=int, default=90)
    suite.add_argument("--output", required=True)
    verification = subparsers.add_parser(
        "verify-front",
        help="run grid-refinement and rotational-invariance checks",
    )
    verification.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "point":
        result = _point(args)
    elif args.command == "benchmark":
        result = _benchmark(args)
    elif args.command == "verify-front":
        result = _verify_front(args)
    else:
        result = _suite(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.command == "benchmark" and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
