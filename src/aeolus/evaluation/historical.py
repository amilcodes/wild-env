"""Historical perimeter assimilation and hindcast evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import numpy as np

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.data.incident import IncidentBundle

Policy = Callable[[AeolusSimulator], dict[str, int]]


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class PerimeterFrame:
    timestamp: datetime
    mask: np.ndarray
    properties: dict[str, Any]


@dataclass(frozen=True)
class PerimeterSeries:
    frames: tuple[PerimeterFrame, ...]
    cell_size_m: float

    @classmethod
    def from_incident(cls, incident: IncidentBundle) -> PerimeterSeries:
        try:
            import rasterio.features
            from affine import Affine
            from pyproj import Transformer
            from shapely.geometry import shape
            from shapely.ops import transform as transform_geometry
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] to rasterize incident perimeters") from exc

        landscape = incident.scenario_bundle()
        metadata = landscape.metadata
        transform_values = metadata.get("transform")
        if not isinstance(transform_values, list) or len(transform_values) != 6:
            if not isinstance(transform_values, tuple) or len(transform_values) != 6:
                raise ValueError("scenario bundle metadata requires a six-value affine transform")
        affine = Affine(*[float(value) for value in transform_values])
        target_crs = str(metadata["crs"])
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        frames: list[PerimeterFrame] = []
        for feature in incident.perimeter_collection()["features"]:
            properties = dict(feature.get("properties", {}))
            observed_at = properties.get("observed_at")
            if not isinstance(observed_at, str):
                raise ValueError("perimeter feature has no normalized observed_at timestamp")
            geometry = transform_geometry(transformer.transform, shape(feature["geometry"]))
            mask = rasterio.features.rasterize(
                [(geometry, 1)],
                out_shape=landscape.elevation_m.shape,
                transform=affine,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            ).astype(np.bool_)
            if mask.any():
                frames.append(PerimeterFrame(_parse_datetime(observed_at), mask, properties))
        frames.sort(key=lambda item: item.timestamp)
        if len(frames) < 2:
            raise ValueError("historical evaluation requires at least two non-empty perimeter frames")
        return cls(tuple(frames), float(metadata["cell_size_m"]))


def perimeter_metrics(predicted: np.ndarray, observed: np.ndarray, cell_size_m: float) -> dict[str, float]:
    predicted_mask = predicted.astype(np.bool_)
    observed_mask = observed.astype(np.bool_)
    intersection = int((predicted_mask & observed_mask).sum())
    union = int((predicted_mask | observed_mask).sum())
    predicted_count = int(predicted_mask.sum())
    observed_count = int(observed_mask.sum())
    cell_area_km2 = cell_size_m**2 / 1_000_000.0
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(predicted_count, 1),
        "recall": intersection / max(observed_count, 1),
        "symmetric_difference_km2": (union - intersection) * cell_area_km2,
        "predicted_area_km2": predicted_count * cell_area_km2,
        "observed_area_km2": observed_count * cell_area_km2,
        "area_bias_km2": (predicted_count - observed_count) * cell_area_km2,
    }


def tolerance_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    radius_cells: int = 1,
) -> dict[str, float]:
    """Precision/recall after a finite raster localization tolerance."""

    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed masks must share a grid")
    if radius_cells < 0:
        raise ValueError("radius_cells must be non-negative")

    def dilate(mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(mask, dtype=np.bool_)
        padded = np.pad(mask.astype(np.bool_), radius_cells)
        height, width = mask.shape
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                y0, x0 = radius_cells + dy, radius_cells + dx
                result |= padded[y0 : y0 + height, x0 : x0 + width]
        return result

    predicted_mask = predicted.astype(np.bool_)
    observed_mask = observed.astype(np.bool_)
    predicted_count = int(predicted_mask.sum())
    observed_count = int(observed_mask.sum())
    matched_predicted = int((predicted_mask & dilate(observed_mask)).sum())
    matched_observed = int((observed_mask & dilate(predicted_mask)).sum())
    precision = (
        matched_predicted / predicted_count
        if predicted_count
        else float(observed_count == 0)
    )
    recall = (
        matched_observed / observed_count
        if observed_count
        else float(predicted_count == 0)
    )
    return {
        "radius_cells": float(radius_cells),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def run_hindcast(
    simulator: AeolusSimulator,
    series: PerimeterSeries,
    policy: Policy,
    *,
    start_index: int = 0,
    target_index: int = 1,
) -> dict[str, Any]:
    """Initialize at one observation and forecast toward a later perimeter."""

    if not (0 <= start_index < target_index < len(series.frames)):
        raise IndexError("hindcast indices must satisfy 0 <= start < target < frame count")
    start = series.frames[start_index]
    target = series.frames[target_index]
    requested_minutes = max(1, round((target.timestamp - start.timestamp).total_seconds() / 60.0))
    simulator.reset(simulator.config.seed)
    simulator.initialize_from_observed_perimeter(start.mask, source="historical-hindcast-start")
    # An operational episode may terminate when fire reaches the finite raster
    # boundary. A hindcast still has to integrate for the full observation
    # interval so its area error cannot improve merely by stopping early.
    while simulator.state.minute < requested_minutes and not simulator.state.truncated:
        if simulator.state.terminated:
            if simulator.state.escaped:
                simulator.state.terminated = False
            else:
                # A contained fire is a stable absorbing state. There is no
                # reason to execute empty decision steps through the remaining
                # wall-clock interval.
                break
        simulator.decision_step(policy(simulator))
    predicted = simulator.state.truth.phase != 0
    predicted_growth = predicted & ~start.mask
    observed_growth = target.mask & ~start.mask
    return {
        "start_time": start.timestamp.isoformat(),
        "target_time": target.timestamp.isoformat(),
        "requested_minutes": requested_minutes,
        "simulated_minutes": simulator.state.minute,
        "terminated": simulator.state.terminated,
        "truncated": simulator.state.truncated,
        "escaped": simulator.state.escaped,
        "metrics": perimeter_metrics(predicted, target.mask, series.cell_size_m),
        "growth_metrics": perimeter_metrics(
            predicted_growth,
            observed_growth,
            series.cell_size_m,
        ),
        "perimeter_tolerance_1_cell": tolerance_metrics(
            predicted, target.mask, radius_cells=1
        ),
        "growth_tolerance_1_cell": tolerance_metrics(
            predicted_growth, observed_growth, radius_cells=1
        ),
        "episode": simulator.episode_record(),
    }


def calibrate_spread_adjustment(
    config: ScenarioConfig,
    series: PerimeterSeries,
    policy: Policy,
    *,
    start_index: int = 0,
    target_index: int = 1,
    validation_target_index: int | None = None,
    candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Fit one effective spread multiplier and evaluate the next interval.

    This is deliberately a low-dimensional calibration. It is useful when
    local weather or fuel conditioning is absent, while avoiding the false
    identifiability of jointly fitting wind, moisture, fuel, and suppression
    from one observed perimeter.
    """

    values = candidates or np.geomspace(0.005, 1.0, 16).tolist()
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("spread-adjustment candidates must be positive")
    trials: list[dict[str, Any]] = []
    for adjustment in values:
        fire = replace(
            config.fire,
            surface_spread_adjustment=float(adjustment),
            crown_spread_adjustment=float(adjustment),
        )
        trial_config = replace(
            config,
            fire=fire,
            terminate_on_escape=False,
        )
        result = run_hindcast(
            AeolusSimulator(trial_config),
            series,
            policy,
            start_index=start_index,
            target_index=target_index,
        )
        metrics = result["metrics"]
        growth_metrics = result["growth_metrics"]
        area_ratio = max(growth_metrics["predicted_area_km2"], 1e-9) / max(
            growth_metrics["observed_area_km2"], 1e-9
        )
        score = float(
            growth_metrics["iou"] - 0.15 * abs(np.log(area_ratio))
        )
        trials.append(
            {
                "spread_adjustment": float(adjustment),
                "score": score,
                "metrics": metrics,
                "growth_metrics": growth_metrics,
                "growth_tolerance_1_cell": result[
                    "growth_tolerance_1_cell"
                ],
            }
        )
    best = max(trials, key=lambda item: item["score"])
    calibrated_fire = replace(
        config.fire,
        surface_spread_adjustment=best["spread_adjustment"],
        crown_spread_adjustment=best["spread_adjustment"],
    )
    calibrated_config = replace(
        config,
        fire=calibrated_fire,
        terminate_on_escape=False,
    )
    validation = None
    if validation_target_index is not None:
        if validation_target_index <= target_index:
            raise ValueError("validation target must follow the calibration target")
        validation = run_hindcast(
            AeolusSimulator(calibrated_config),
            series,
            policy,
            start_index=target_index,
            target_index=validation_target_index,
        )
    return {
        "method": "one-parameter log-grid calibration",
        "fitted_parameter": "surface_and_crown_spread_adjustment",
        "calibration_interval": {
            "start_index": start_index,
            "target_index": target_index,
        },
        "selected_spread_adjustment": best["spread_adjustment"],
        "calibration_metrics": best["metrics"],
        "calibration_growth_metrics": best["growth_metrics"],
        "candidate_trials": trials,
        "validation": validation,
        "limitations": [
            "effective multiplier conflates unresolved weather, fuel conditioning, and suppression",
            "a single perimeter interval cannot identify those factors separately",
            "validation interval is not used during selection",
        ],
    }


def run_shadow_replay(
    simulator: AeolusSimulator,
    series: PerimeterSeries,
    policy: Policy,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, Any]:
    """Replay historical perimeter updates into belief without changing truth.

    Actions are generated and logged as they would have been at the time, while
    observed perimeters are assimilated only after their timestamp. This mode
    evaluates decision behavior and information timing; it does not claim a
    causal suppression outcome against the historical fire.
    """

    if simulator.config.terminate_on_escape:
        simulator = AeolusSimulator(replace(simulator.config, terminate_on_escape=False))
    final_index = len(series.frames) - 1 if end_index is None else end_index
    if not (0 <= start_index < final_index < len(series.frames)):
        raise IndexError("shadow indices must satisfy 0 <= start < end < frame count")
    start = series.frames[start_index]
    final = series.frames[final_index]
    requested_minutes = max(1, round((final.timestamp - start.timestamp).total_seconds() / 60.0))
    schedule = [
        (
            max(0, round((frame.timestamp - start.timestamp).total_seconds() / 60.0)),
            frame,
        )
        for frame in series.frames[start_index + 1 : final_index + 1]
    ]
    simulator.reset(simulator.config.seed)
    simulator.initialize_from_observed_perimeter(start.mask, source="historical-shadow-start")
    assimilated = 0
    while (
        simulator.state.minute < requested_minutes
        and not simulator.state.terminated
        and not simulator.state.truncated
    ):
        simulator.decision_step(policy(simulator))
        while assimilated < len(schedule) and schedule[assimilated][0] <= simulator.state.minute:
            _, frame = schedule[assimilated]
            simulator.assimilate_observed_perimeter(
                frame.mask,
                source=f"historical-shadow:{frame.timestamp.isoformat()}",
            )
            assimilated += 1
    return {
        "start_time": start.timestamp.isoformat(),
        "end_time": final.timestamp.isoformat(),
        "requested_minutes": requested_minutes,
        "simulated_minutes": simulator.state.minute,
        "assimilated_perimeters": assimilated,
        "episode": simulator.episode_record(),
    }


def compare_counterfactual_policies(
    config: ScenarioConfig,
    series: PerimeterSeries,
    policies: dict[str, Policy],
    seeds: list[int],
    *,
    start_index: int = 0,
    target_index: int = 1,
) -> dict[str, Any]:
    """Run paired policy branches from one historical perimeter and seed set."""

    if not (0 <= start_index < target_index < len(series.frames)):
        raise IndexError("comparison indices must satisfy 0 <= start < target < frame count")
    start = series.frames[start_index]
    target = series.frames[target_index]
    horizon = max(1, round((target.timestamp - start.timestamp).total_seconds() / 60.0))
    branch_config = replace(config, horizon_min=max(config.horizon_min, horizon))
    records: dict[str, list[dict[str, Any]]] = {}
    for name, policy in policies.items():
        records[name] = []
        for seed in seeds:
            simulator = AeolusSimulator(replace(branch_config, seed=seed))
            simulator.initialize_from_observed_perimeter(
                start.mask,
                source="historical-counterfactual-start",
            )
            while (
                simulator.state.minute < horizon
                and not simulator.state.terminated
                and not simulator.state.truncated
            ):
                simulator.decision_step(policy(simulator))
            predicted = simulator.state.truth.phase != 0
            records[name].append(
                {
                    "seed": seed,
                    "metrics_to_observed": perimeter_metrics(
                        predicted,
                        target.mask,
                        series.cell_size_m,
                    ),
                    "episode": simulator.episode_record(),
                }
            )
    summary = {
        name: {
            "episodes": len(items),
            "mean_weighted_loss": float(
                np.mean([item["episode"]["weighted_loss"] for item in items])
            ),
            "escape_rate": float(np.mean([item["episode"]["escaped"] for item in items])),
            "containment_rate": float(
                np.mean([item["episode"]["contained"] for item in items])
            ),
            "mean_iou_to_observed": float(
                np.mean([item["metrics_to_observed"]["iou"] for item in items])
            ),
        }
        for name, items in records.items()
    }
    return {
        "schema_version": 1,
        "mode": "paired-counterfactual",
        "start_time": start.timestamp.isoformat(),
        "target_time": target.timestamp.isoformat(),
        "horizon_min": horizon,
        "seeds": seeds,
        "summary": summary,
        "episodes": records,
    }
