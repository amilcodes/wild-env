"""Probabilistic perimeter forecasting and parameter-state calibration.

The ensemble deliberately carries a small set of interpretable epistemic
uncertainties: effective spread, wind exposure, wind direction, and dead-fuel
moisture.  Perimeter observations update particle weights through a
localization-aware boundary likelihood.  Held-out forecasts are then scored
as probability fields, avoiding the false precision of a single calibrated
burn mask.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.evaluation.historical import (
    HindcastJob,
    PerimeterSeries,
    boundary_distance_metrics,
    execute_hindcast_jobs,
    perimeter_metrics,
    tolerance_metrics,
)

Policy = Callable[[AeolusSimulator], dict[str, int]]


@dataclass(frozen=True)
class FireParameterParticle:
    """One physically interpretable member of the fire forecast ensemble."""

    spread_adjustment: float
    wind_speed_adjustment: float
    wind_direction_bias_deg: float
    dead_fuel_moisture_bias: float
    seed_offset: int

    def apply(self, config: ScenarioConfig) -> ScenarioConfig:
        fire = replace(
            config.fire,
            surface_spread_adjustment=self.spread_adjustment,
            crown_spread_adjustment=self.spread_adjustment,
            wind_speed_adjustment=self.wind_speed_adjustment,
            wind_direction_bias_deg=self.wind_direction_bias_deg,
            dead_fuel_moisture_bias=self.dead_fuel_moisture_bias,
        )
        return replace(
            config,
            seed=config.seed + self.seed_offset,
            fire=fire,
            terminate_on_escape=False,
        )


def sample_parameter_particles(
    spread_candidates: Sequence[float],
    *,
    particle_count: int,
    seed: int,
) -> tuple[FireParameterParticle, ...]:
    """Construct a stratified ensemble from explicit, auditable priors."""

    candidates = np.asarray(spread_candidates, dtype=np.float64)
    if particle_count < 2:
        raise ValueError("particle ensemble requires at least two members")
    if candidates.size == 0 or np.any(candidates <= 0.0):
        raise ValueError("spread candidates must be positive")
    candidates.sort()
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(particle_count, dtype=np.float64) + 0.5) / particle_count
    candidate_indices = np.minimum(
        (quantiles * candidates.size).astype(np.int64),
        candidates.size - 1,
    )
    rng.shuffle(candidate_indices)
    spread = candidates[candidate_indices] * np.exp(rng.normal(0.0, 0.06, particle_count))
    wind_speed = np.clip(
        np.exp(rng.normal(0.0, 0.18, particle_count)),
        0.60,
        1.55,
    )
    wind_direction = np.clip(
        rng.normal(0.0, 14.0, particle_count),
        -40.0,
        40.0,
    )
    moisture = np.clip(
        rng.normal(0.0, 0.015, particle_count),
        -0.045,
        0.045,
    )
    return tuple(
        FireParameterParticle(
            spread_adjustment=float(spread[index]),
            wind_speed_adjustment=float(wind_speed[index]),
            wind_direction_bias_deg=float(wind_direction[index]),
            dead_fuel_moisture_bias=float(moisture[index]),
            seed_offset=104729 * (index + 1),
        )
        for index in range(particle_count)
    )


def _boundary_residual_m(
    predicted: np.ndarray,
    observed: np.ndarray,
    cell_size_m: float,
) -> float:
    if not predicted.any() and not observed.any():
        return 0.0
    metrics = boundary_distance_metrics(predicted, observed, cell_size_m)
    value = float(metrics["mean_symmetric_distance_m"])
    if np.isfinite(value):
        return value
    diagonal = float(np.hypot(*predicted.shape) * cell_size_m)
    return diagonal


def perimeter_log_likelihood(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    cell_size_m: float,
    localization_sigma_m: float,
    area_log_sigma: float = 0.70,
) -> dict[str, float]:
    """Robust pseudo-likelihood for uncertain remotely sensed perimeters."""

    if localization_sigma_m <= 0.0:
        raise ValueError("localization sigma must be positive")
    predicted_mask = predicted.astype(np.bool_)
    observed_mask = observed.astype(np.bool_)
    boundary_residual = _boundary_residual_m(predicted_mask, observed_mask, cell_size_m)
    predicted_area = max(int(predicted_mask.sum()), 0.5)
    observed_area = max(int(observed_mask.sum()), 0.5)
    log_area_ratio = float(np.log(predicted_area / observed_area))
    boundary_term = -0.5 * (boundary_residual / localization_sigma_m) ** 2
    area_term = -0.5 * (log_area_ratio / area_log_sigma) ** 2
    return {
        "log_likelihood": float(boundary_term + area_term),
        "boundary_residual_m": boundary_residual,
        "log_area_ratio": log_area_ratio,
        "boundary_log_likelihood": float(boundary_term),
        "area_log_likelihood": float(area_term),
    }


def incremental_growth_log_likelihood(
    predicted_growth: np.ndarray,
    observed_growth: np.ndarray,
    *,
    area_log_sigma: float = 1.50,
    tolerance_f1_sigma: float = 0.45,
) -> dict[str, float]:
    """Continuous growth likelihood that remains finite for empty predictions."""

    predicted = np.asarray(predicted_growth, dtype=np.bool_)
    observed = np.asarray(observed_growth, dtype=np.bool_)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed growth must share a grid")
    predicted_count = int(predicted.sum())
    observed_count = int(observed.sum())
    log_area_ratio = float(np.log((predicted_count + 0.5) / (observed_count + 0.5)))
    tolerance_f1 = float(tolerance_metrics(predicted, observed, radius_cells=1)["f1"])
    area_term = -0.5 * (log_area_ratio / area_log_sigma) ** 2
    localization_term = -0.5 * ((1.0 - tolerance_f1) / tolerance_f1_sigma) ** 2
    return {
        "log_likelihood": float(area_term + localization_term),
        "log_area_ratio": log_area_ratio,
        "tolerance_f1": tolerance_f1,
        "area_log_likelihood": float(area_term),
        "localization_log_likelihood": float(localization_term),
    }


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log weights must be a non-empty vector")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.size, 1.0 / values.size)
    return weights / total


def tempered_log_weights(
    log_weights: np.ndarray,
    *,
    minimum_ess_fraction: float = 0.35,
) -> tuple[np.ndarray, float, float]:
    """Adapt likelihood power to avoid finite-ensemble particle collapse."""

    values = np.asarray(log_weights, dtype=np.float64)
    if not 0.0 < minimum_ess_fraction <= 1.0:
        raise ValueError("minimum ESS fraction must be within (0, 1]")
    raw_weights = normalize_log_weights(values)
    raw_ess = float(1.0 / np.sum(raw_weights**2))
    target_ess = minimum_ess_fraction * values.size
    if raw_ess >= target_ess:
        return raw_weights, 1.0, raw_ess
    lower, upper = 0.0, 1.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        weights = normalize_log_weights(midpoint * values)
        ess = 1.0 / np.sum(weights**2)
        if ess >= target_ess:
            lower = midpoint
        else:
            upper = midpoint
    tempered = normalize_log_weights(lower * values)
    return tempered, float(lower), raw_ess


def systematic_resample(
    weights: Sequence[float],
    *,
    seed: int,
) -> np.ndarray:
    """Low-variance particle-filter resampling indices."""

    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    count = probabilities.size
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).astype(np.int64)


def probability_metrics(
    probability: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int = 10,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Proper scores and reliability diagnostics for a burn probability map."""

    forecast = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    truth = np.asarray(observed, dtype=np.bool_)
    if forecast.shape != truth.shape:
        raise ValueError("probability and observed fields must share a grid")
    selected = (
        np.ones_like(truth, dtype=np.bool_)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=np.bool_)
    )
    if selected.shape != truth.shape or not selected.any():
        raise ValueError("evaluation mask must select at least one grid cell")
    p = forecast[selected]
    y = truth[selected].astype(np.float64)
    event = y == 1.0
    nonevent = ~event
    brier = float(np.mean((p - y) ** 2))
    event_brier = float(np.mean((p[event] - 1.0) ** 2)) if event.any() else float("nan")
    nonevent_brier = float(np.mean(p[nonevent] ** 2)) if nonevent.any() else float("nan")
    finite_class_scores = [value for value in (event_brier, nonevent_brier) if np.isfinite(value)]
    log_score = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    reliability: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = bin_edges[index], bin_edges[index + 1]
        member = (p >= lower) & (p <= upper if index == bins - 1 else p < upper)
        count = int(member.sum())
        if count == 0:
            continue
        mean_probability = float(p[member].mean())
        observed_frequency = float(y[member].mean())
        expected_calibration_error += count / p.size * abs(mean_probability - observed_frequency)
        reliability.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
            }
        )
    return {
        "brier_score": brier,
        "balanced_brier_score": float(np.mean(finite_class_scores)),
        "event_brier_score": event_brier,
        "nonevent_brier_score": nonevent_brier,
        "log_score": log_score,
        "expected_calibration_error": float(expected_calibration_error),
        "mean_predictive_variance": float(np.mean(p * (1.0 - p))),
        "mean_probability": float(p.mean()),
        "observed_frequency": float(y.mean()),
        "evaluated_cells": int(p.size),
        "reliability": reliability,
    }


def calibrate_particle_ensemble(
    config: ScenarioConfig,
    series: PerimeterSeries,
    policy: Policy,
    *,
    start_index: int,
    target_index: int,
    spread_candidates: Sequence[float],
    particle_count: int = 16,
    seed: int = 0,
    localization_sigma_m: float | None = None,
    parallel_workers: int | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Update an interpretable parameter ensemble from one perimeter interval."""

    particles = sample_parameter_particles(
        spread_candidates,
        particle_count=particle_count,
        seed=seed,
    )
    sigma = (
        max(2.0 * series.cell_size_m, 250.0) if localization_sigma_m is None else float(localization_sigma_m)
    )
    target = series.frames[target_index]
    start = series.frames[start_index]
    jobs = [
        HindcastJob(
            config=particle.apply(config),
            series=series,
            policy=policy,
            start_index=start_index,
            target_index=target_index,
            return_prediction=True,
        )
        for particle in particles
    ]
    results = execute_hindcast_jobs(
        jobs,
        parallel_workers=parallel_workers,
        executor=executor,
    )
    outcomes: list[tuple[dict[str, Any], float]] = []
    for particle, result in zip(particles, results, strict=True):
        prediction = result.pop("prediction_mask")
        cumulative_likelihood = perimeter_log_likelihood(
            prediction,
            target.mask,
            cell_size_m=series.cell_size_m,
            localization_sigma_m=sigma,
        )
        predicted_growth = prediction & ~start.mask
        observed_growth = target.mask & ~start.mask
        growth_likelihood = incremental_growth_log_likelihood(
            predicted_growth,
            observed_growth,
        )
        log_likelihood = cumulative_likelihood["log_likelihood"] + growth_likelihood["log_likelihood"]
        likelihood = {
            "log_likelihood": float(log_likelihood),
            "cumulative": cumulative_likelihood,
            "incremental_growth": growth_likelihood,
        }
        outcomes.append(
            (
                {
                    "particle": asdict(particle),
                    "likelihood": likelihood,
                    "metrics": result["metrics"],
                    "growth_metrics": result["growth_metrics"],
                },
                likelihood["log_likelihood"],
            )
        )
    trials = [outcome[0] for outcome in outcomes]
    log_weights = np.asarray(
        [outcome[1] for outcome in outcomes],
        dtype=np.float64,
    )
    weights, tempering_beta, raw_effective_sample_size = tempered_log_weights(log_weights)
    for trial, weight in zip(trials, weights, strict=True):
        trial["posterior_weight"] = float(weight)
    effective_sample_size = float(1.0 / np.sum(weights**2))
    return {
        "method": "localization-aware parameter particle filter",
        "calibration_interval": {
            "start_index": start_index,
            "target_index": target_index,
        },
        "localization_sigma_m": sigma,
        "particles": [asdict(particle) for particle in particles],
        "posterior_weights": weights.tolist(),
        "effective_sample_size": effective_sample_size,
        "raw_effective_sample_size": raw_effective_sample_size,
        "likelihood_tempering_beta": tempering_beta,
        "entropy": float(-np.sum(weights * np.log(np.maximum(weights, 1e-300)))),
        "trials": trials,
        "identifiability_note": (
            "weights represent predictive adequacy of joint parameter particles; "
            "they do not identify causal errors in wind, moisture, or fuels"
        ),
    }


def run_ensemble_hindcast(
    config: ScenarioConfig,
    series: PerimeterSeries,
    policy: Policy,
    *,
    start_index: int,
    target_index: int,
    particles: Sequence[FireParameterParticle | dict[str, Any]],
    weights: Sequence[float],
    return_probability: bool = False,
    use_arrival_history: bool = False,
    parallel_workers: int | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Forecast a held-out perimeter and retain the full burn probability."""

    members = tuple(
        particle if isinstance(particle, FireParameterParticle) else FireParameterParticle(**particle)
        for particle in particles
    )
    probabilities = np.asarray(weights, dtype=np.float64)
    if len(members) != probabilities.size or probabilities.size == 0:
        raise ValueError("particles and weights must have the same non-zero length")
    probabilities = probabilities / probabilities.sum()
    jobs = [
        HindcastJob(
            config=particle.apply(config),
            series=series,
            policy=policy,
            start_index=start_index,
            target_index=target_index,
            return_prediction=True,
            use_arrival_history=use_arrival_history,
        )
        for particle in members
    ]
    results = execute_hindcast_jobs(
        jobs,
        parallel_workers=parallel_workers,
        executor=executor,
    )
    outcomes: list[tuple[np.ndarray, np.ndarray, float]] = []
    for result in results:
        prediction = np.asarray(result["prediction_mask"], dtype=np.bool_)
        outcomes.append(
            (
                prediction,
                np.asarray(result["arrival_time_min"], dtype=np.float32),
                float(prediction.sum() * series.cell_size_m**2 / 1_000_000.0),
            )
        )
    member_masks = [outcome[0] for outcome in outcomes]
    member_arrivals = [outcome[1] for outcome in outcomes]
    member_areas = [outcome[2] for outcome in outcomes]
    probability = np.tensordot(
        probabilities,
        np.stack(member_masks).astype(np.float32),
        axes=(0, 0),
    ).astype(np.float32)
    arrival_stack = np.stack(member_arrivals).astype(np.float64)
    finite_arrival = np.isfinite(arrival_stack)
    cell_weight = probabilities[:, None, None] * finite_arrival
    arrival_weight = cell_weight.sum(axis=0)
    safe_arrival = np.where(finite_arrival, arrival_stack, 0.0)
    arrival_mean = np.divide(
        np.sum(cell_weight * safe_arrival, axis=0),
        arrival_weight,
        out=np.full(probability.shape, np.inf, dtype=np.float64),
        where=arrival_weight > 0.0,
    )
    centred_arrival = np.zeros_like(safe_arrival)
    np.subtract(
        safe_arrival,
        arrival_mean[None, :, :],
        out=centred_arrival,
        where=finite_arrival & np.isfinite(arrival_mean)[None, :, :],
    )
    arrival_variance = np.divide(
        np.sum(cell_weight * centred_arrival**2, axis=0),
        arrival_weight,
        out=np.full(probability.shape, np.inf, dtype=np.float64),
        where=arrival_weight > 0.0,
    )
    arrival_std = np.sqrt(np.maximum(arrival_variance, 0.0))
    thresholded = probability >= 0.5
    start = series.frames[start_index].mask
    observed = series.frames[target_index].mask
    observed_growth = observed & ~start
    growth_probability = np.where(start, 0.0, probability)
    growth_thresholded = thresholded & ~start
    # Proper scores over the whole raster are dominated by easy unburned
    # background. Define the active domain only from observations, so a model
    # cannot improve its score by changing which cells enter evaluation.
    try:
        from scipy.ndimage import binary_dilation, binary_erosion

        active_domain = binary_dilation(
            observed_growth,
            iterations=2,
        )
        if not active_domain.any():
            active_domain = binary_dilation(start, iterations=2) & ~binary_erosion(start, iterations=2)
    except ImportError:  # pragma: no cover
        active_domain = observed_growth
        if not active_domain.any():
            active_domain = np.ones_like(start, dtype=np.bool_)
    ensemble_probability_metrics = probability_metrics(probability, observed)
    active_probability_metrics = probability_metrics(
        growth_probability,
        observed_growth,
        evaluation_mask=active_domain,
    )
    persistence_probability_metrics = probability_metrics(
        start.astype(np.float32),
        observed,
    )
    persistence_active_probability_metrics = probability_metrics(
        np.zeros_like(probability),
        observed_growth,
        evaluation_mask=active_domain,
    )

    def skill_score(score: float, reference: float) -> float:
        return float(1.0 - score / reference) if reference > 0.0 else float("nan")

    result: dict[str, Any] = {
        "method": "posterior parameter ensemble",
        "start_time": series.frames[start_index].timestamp.isoformat(),
        "target_time": series.frames[target_index].timestamp.isoformat(),
        "requested_minutes": max(
            1,
            round(
                (series.frames[target_index].timestamp - series.frames[start_index].timestamp).total_seconds()
                / 60.0
            ),
        ),
        "member_count": len(members),
        "effective_sample_size": float(1.0 / np.sum(probabilities**2)),
        "metrics": perimeter_metrics(thresholded, observed, series.cell_size_m),
        "growth_metrics": perimeter_metrics(growth_thresholded, observed_growth, series.cell_size_m),
        "perimeter_tolerance_1_cell": tolerance_metrics(thresholded, observed, radius_cells=1),
        "growth_tolerance_1_cell": tolerance_metrics(growth_thresholded, observed_growth, radius_cells=1),
        "boundary": boundary_distance_metrics(thresholded, observed, series.cell_size_m),
        "probabilistic_metrics": ensemble_probability_metrics,
        "active_domain_probabilistic_metrics": active_probability_metrics,
        "persistence_probabilistic_metrics": persistence_probability_metrics,
        "persistence_active_domain_probabilistic_metrics": (persistence_active_probability_metrics),
        "probabilistic_skill_against_persistence": {
            "brier_skill_score": skill_score(
                ensemble_probability_metrics["brier_score"],
                persistence_probability_metrics["brier_score"],
            ),
            "balanced_brier_skill_score": skill_score(
                ensemble_probability_metrics["balanced_brier_score"],
                persistence_probability_metrics["balanced_brier_score"],
            ),
            "active_domain_brier_skill_score": skill_score(
                active_probability_metrics["brier_score"],
                persistence_active_probability_metrics["brier_score"],
            ),
            "active_domain_balanced_brier_skill_score": skill_score(
                active_probability_metrics["balanced_brier_score"],
                persistence_active_probability_metrics["balanced_brier_score"],
            ),
        },
        "member_area_km2": {
            "weighted_mean": float(np.average(member_areas, weights=probabilities)),
            "minimum": float(np.min(member_areas)),
            "maximum": float(np.max(member_areas)),
            "p10": float(np.quantile(member_areas, 0.10)),
            "p50": float(np.quantile(member_areas, 0.50)),
            "p90": float(np.quantile(member_areas, 0.90)),
        },
    }
    if return_probability:
        result["probability"] = probability
        result["arrival_time_mean"] = arrival_mean.astype(np.float32)
        result["arrival_time_std"] = arrival_std.astype(np.float32)
    return result
