"""Observation models for uncertain historical fire perimeters.

The functions in this module do not reinterpret an observed perimeter as
ground truth at sub-pixel precision.  They expose geometry uncertainty as a
declared sensitivity parameter and preserve the hard-mask scores alongside
the uncertainty-aware scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerimeterUncertainty:
    """A declared isotropic one-sigma localization uncertainty."""

    sigma_m: float
    confidence: float = 0.95

    def validate(self) -> None:
        if self.sigma_m < 0.0:
            raise ValueError("sigma_m must be non-negative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be within (0, 1)")


@dataclass(frozen=True)
class AcquisitionWindow:
    """Relative acquisition and availability times for one observation.

    The source may integrate information throughout ``[start_minute,
    end_minute]`` and become available later.  Times are relative to the
    forecast origin, allowing historical studies to distinguish sensing time
    from processing latency.
    """

    start_minute: float
    end_minute: float
    available_minute: float

    def validate(self) -> None:
        values = np.asarray(
            (self.start_minute, self.end_minute, self.available_minute),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("acquisition-window times must be finite")
        if self.end_minute < self.start_minute:
            raise ValueError("acquisition end cannot precede acquisition start")
        if self.available_minute < self.end_minute:
            raise ValueError("observation availability cannot precede acquisition end")

    @property
    def duration_minute(self) -> float:
        return float(self.end_minute - self.start_minute)

    @property
    def processing_latency_minute(self) -> float:
        return float(self.available_minute - self.end_minute)


@dataclass(frozen=True)
class RasterObservationModel:
    """Declared sensor terms for a cumulative burned-area observation.

    Detection and false-alarm probabilities apply after temporal integration
    and spatial localization.  ``obscured_probability`` may be a scalar or a
    raster supplied to the scoring function.  Obscured cells carry less
    information; they are not interpreted as observed non-fire.
    """

    acquisition: AcquisitionWindow
    localization_sigma_m: float
    detection_probability: float = 1.0
    false_alarm_probability: float = 0.0

    def validate(self) -> None:
        self.acquisition.validate()
        if self.localization_sigma_m < 0.0:
            raise ValueError("localization sigma must be non-negative")
        if not 0.0 < self.detection_probability <= 1.0:
            raise ValueError("detection probability must be within (0, 1]")
        if not 0.0 <= self.false_alarm_probability < 1.0:
            raise ValueError("false-alarm probability must be within [0, 1)")
        if self.false_alarm_probability >= self.detection_probability:
            raise ValueError("false-alarm probability must be below detection probability")


def acquisition_window_burn_probability(
    arrival_minute: np.ndarray,
    acquisition: AcquisitionWindow,
) -> np.ndarray:
    """Probability a cell had burned at an unknown acquisition instant.

    Acquisition time is treated as uniform over the declared window.  This is
    a transparent default for binned or composite products; a source-specific
    scan-time distribution can replace it when available.
    """

    acquisition.validate()
    arrival = np.asarray(arrival_minute, dtype=np.float64)
    if arrival.ndim != 2:
        raise ValueError("arrival time must be a two-dimensional raster")
    duration = acquisition.duration_minute
    if duration == 0.0:
        probability = arrival <= acquisition.end_minute
    else:
        probability = np.clip(
            (acquisition.end_minute - arrival) / duration,
            0.0,
            1.0,
        )
    probability = np.where(np.isfinite(arrival), probability, 0.0)
    return np.asarray(probability, dtype=np.float32)


def expected_raster_observation_probability(
    arrival_minute: np.ndarray,
    *,
    model: RasterObservationModel,
    cell_size_m: float,
) -> np.ndarray:
    """Map simulated arrival time to expected observed occupancy."""

    model.validate()
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    burn_probability = acquisition_window_burn_probability(
        arrival_minute,
        model.acquisition,
    ).astype(np.float64)
    if model.localization_sigma_m > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] for observation localization") from exc
        burn_probability = gaussian_filter(
            burn_probability,
            sigma=model.localization_sigma_m / cell_size_m,
            mode="nearest",
        )
    observed_probability = model.false_alarm_probability + (
        model.detection_probability - model.false_alarm_probability
    ) * np.clip(burn_probability, 0.0, 1.0)
    return np.asarray(np.clip(observed_probability, 0.0, 1.0), dtype=np.float32)


def acquisition_aware_observation_metrics(
    arrival_minute: np.ndarray,
    observed_mask: np.ndarray,
    *,
    model: RasterObservationModel,
    cell_size_m: float,
    obscured_probability: float | np.ndarray = 0.0,
    evaluation_mask: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Score an observation without assigning an exact sensing instant.

    Cross entropy and Brier score are information-weighted by the probability
    that the cell was observable.  The returned log likelihood is the
    unnormalized weighted Bernoulli likelihood for particle filtering; its
    mean is also reported for comparisons across grids.
    """

    observed = np.asarray(observed_mask, dtype=np.bool_)
    probability = expected_raster_observation_probability(
        arrival_minute,
        model=model,
        cell_size_m=cell_size_m,
    ).astype(np.float64)
    if probability.shape != observed.shape:
        raise ValueError("arrival time and observed mask must share a grid")
    obscured = np.broadcast_to(
        np.asarray(obscured_probability, dtype=np.float64),
        observed.shape,
    )
    if np.any(~np.isfinite(obscured)) or np.any((obscured < 0.0) | (obscured > 1.0)):
        raise ValueError("obscured_probability must be finite within [0, 1]")
    selected = (
        np.ones_like(observed, dtype=np.bool_)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=np.bool_)
    )
    if selected.shape != observed.shape or not selected.any():
        raise ValueError("evaluation mask must select at least one cell")
    weights = (1.0 - obscured)[selected]
    if float(weights.sum()) <= 0.0:
        raise ValueError("obscuration leaves no effective observation support")
    p = np.clip(probability[selected], epsilon, 1.0 - epsilon)
    y = observed[selected].astype(np.float64)
    log_terms = y * np.log(p) + (1.0 - y) * np.log1p(-p)
    residual = (p - y) ** 2
    weighted_log_likelihood = float(np.sum(weights * log_terms))
    effective_cells = float(weights.sum())
    temporal_probability = acquisition_window_burn_probability(
        arrival_minute,
        model.acquisition,
    )
    ambiguous = (temporal_probability > 0.0) & (temporal_probability < 1.0)
    return {
        "log_likelihood": weighted_log_likelihood,
        "mean_log_score": float(-weighted_log_likelihood / effective_cells),
        "brier_score": float(np.sum(weights * residual) / effective_cells),
        "effective_observed_cells": effective_cells,
        "selected_cells": float(selected.sum()),
        "temporally_ambiguous_cells": float((ambiguous & selected).sum()),
        "acquisition_duration_min": model.acquisition.duration_minute,
        "processing_latency_min": model.acquisition.processing_latency_minute,
        "localization_sigma_m": float(model.localization_sigma_m),
        "detection_probability": float(model.detection_probability),
        "false_alarm_probability": float(model.false_alarm_probability),
    }


def signed_distance_m(mask: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Return positive distance inside a perimeter and negative outside."""

    if mask.ndim != 2:
        raise ValueError("perimeter mask must be two-dimensional")
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for observation uncertainty") from exc

    observed = mask.astype(np.bool_)
    inside = distance_transform_edt(observed)
    outside = distance_transform_edt(~observed)
    return ((inside - outside) * cell_size_m).astype(np.float32)


def soft_perimeter_probability(
    mask: np.ndarray,
    *,
    sigma_m: float,
    cell_size_m: float,
) -> np.ndarray:
    """Convert a hard perimeter to occupancy probability under Gaussian error.

    ``sigma_m`` is a sensitivity parameter unless calibrated for a particular
    perimeter product.  A value of zero returns the original binary mask.
    """

    if sigma_m < 0.0:
        raise ValueError("sigma_m must be non-negative")
    if sigma_m == 0.0:
        return mask.astype(np.float32)
    try:
        from scipy.special import ndtr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for observation uncertainty") from exc
    probability = ndtr(signed_distance_m(mask, cell_size_m) / sigma_m)
    return np.asarray(probability, dtype=np.float32)


def perimeter_observation_envelope(
    mask: np.ndarray,
    *,
    sigma_m: float,
    cell_size_m: float,
    confidence: float = 0.95,
) -> np.ndarray:
    """Return cells consistent with the observed boundary at a confidence level."""

    model = PerimeterUncertainty(sigma_m=sigma_m, confidence=confidence)
    model.validate()
    if sigma_m == 0.0:
        return np.zeros_like(mask, dtype=np.bool_)
    try:
        from scipy.special import ndtri
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for observation uncertainty") from exc
    half_width_m = float(ndtri(0.5 + confidence / 2.0)) * sigma_m
    return np.abs(signed_distance_m(mask, cell_size_m)) <= half_width_m


def _boundary(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] for observation uncertainty") from exc
    binary = mask.astype(np.bool_)
    return binary & ~binary_erosion(binary, border_value=0)


def uncertainty_aware_perimeter_metrics(
    prediction_probability: np.ndarray,
    observed_mask: np.ndarray,
    *,
    sigma_m: float,
    cell_size_m: float,
    confidence: float = 0.95,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Score a burn-probability field against an uncertain perimeter.

    Soft IoU uses probabilistic intersection and union.  Brier and cross
    entropy are reported against the occupancy probability induced by the
    declared perimeter localization uncertainty.  Boundary coverage is the
    fraction of a thresholded predicted boundary inside the corresponding
    observation envelope.
    """

    probability = np.asarray(prediction_probability, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=np.bool_)
    if probability.shape != observed.shape:
        raise ValueError("prediction and observation must share a grid")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("prediction_probability must be finite within [0, 1]")

    target = soft_perimeter_probability(
        observed,
        sigma_m=sigma_m,
        cell_size_m=cell_size_m,
    ).astype(np.float64)
    intersection = np.minimum(probability, target).sum()
    union = np.maximum(probability, target).sum()
    brier = np.mean((probability - target) ** 2)
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    cross_entropy = -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))

    predicted_boundary = _boundary(probability >= 0.5)
    envelope = perimeter_observation_envelope(
        observed,
        sigma_m=sigma_m,
        cell_size_m=cell_size_m,
        confidence=confidence,
    )
    boundary_count = int(predicted_boundary.sum())
    if sigma_m == 0.0:
        envelope = _boundary(observed)
    boundary_coverage = (
        float((predicted_boundary & envelope).sum()) / boundary_count
        if boundary_count
        else float(not observed.any())
    )
    return {
        "observation_sigma_m": float(sigma_m),
        "observation_confidence": float(confidence),
        "soft_iou": float(intersection / max(union, epsilon)),
        "soft_brier_score": float(brier),
        "soft_cross_entropy": float(cross_entropy),
        "predicted_boundary_envelope_coverage": boundary_coverage,
    }


def interval_censored_arrival_score(
    arrival_minute: np.ndarray,
    start_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    interval_minutes: float,
    time_scale_minutes: float = 60.0,
) -> dict[str, float]:
    """Score arrival times where observations only bracket the event time.

    Newly observed cells have a valid arrival interval ``(0, interval]``.
    Cells still outside the target perimeter are right-censored.  The loss is
    a normalized hinge distance to those intervals; already-burned start
    cells are excluded because their ignition times predate the interval.
    """

    arrival = np.asarray(arrival_minute, dtype=np.float64)
    start = np.asarray(start_mask, dtype=np.bool_)
    target = np.asarray(target_mask, dtype=np.bool_)
    if arrival.shape != start.shape or arrival.shape != target.shape:
        raise ValueError("arrival and perimeter masks must share a grid")
    if interval_minutes <= 0.0 or time_scale_minutes <= 0.0:
        raise ValueError("interval and time scale must be positive")

    finite = np.isfinite(arrival)
    new_burn = target & ~start
    right_censored = ~target & ~start
    new_loss = np.where(
        ~finite,
        interval_minutes,
        np.maximum(-arrival, 0.0) + np.maximum(arrival - interval_minutes, 0.0),
    )
    right_loss = np.where(finite, np.maximum(interval_minutes - arrival, 0.0), 0.0)
    losses = np.concatenate((new_loss[new_burn], right_loss[right_censored]))
    return {
        "interval_censored_arrival_mae_scaled": (
            float(np.mean(losses) / time_scale_minutes) if losses.size else 0.0
        ),
        "newly_observed_cell_count": float(new_burn.sum()),
        "right_censored_cell_count": float(right_censored.sum()),
    }
