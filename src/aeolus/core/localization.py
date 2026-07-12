"""Localization operators for sequential perimeter correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeolus.core.front import signed_distance


@dataclass(frozen=True)
class FrontCorrection:
    corrected_level_set_m: np.ndarray
    level_set_increment_m: np.ndarray
    localization_weight: np.ndarray
    diagnostics: dict[str, float | int]


def localize_front_correction(
    forecast_level_set_m: np.ndarray,
    observed_perimeter_mask: np.ndarray,
    cell_size_m: float,
    *,
    localization_radius_m: float,
    gain: float = 0.90,
    maximum_displacement_m: float | None = None,
) -> FrontCorrection:
    """Correct a level set only in a moving band around the observed front.

    The innovation is expressed in signed-distance coordinates.  A compact
    Gaussian taper prevents the perimeter observation from directly rewriting
    atmosphere, fuels, or fire state far behind and ahead of the active front.
    """

    if forecast_level_set_m.shape != observed_perimeter_mask.shape or forecast_level_set_m.ndim != 2:
        raise ValueError("forecast and observed perimeter must share a 2-D grid")
    if cell_size_m <= 0.0 or localization_radius_m <= 0.0:
        raise ValueError("cell size and localization radius must be positive")
    if not 0.0 <= gain <= 1.0:
        raise ValueError("localization gain must be within [0, 1]")
    observed_distance = signed_distance(
        observed_perimeter_mask.astype(np.bool_),
        cell_size_m,
    ).astype(np.float64)
    forecast = np.asarray(forecast_level_set_m, dtype=np.float64)
    distance_to_advancing_front = np.minimum(
        np.abs(observed_distance),
        np.abs(forecast),
    )
    weight = np.exp(-0.5 * (distance_to_advancing_front / localization_radius_m) ** 2)
    weight[distance_to_advancing_front > 3.0 * localization_radius_m] = 0.0
    innovation = observed_distance - forecast
    displacement_limit = (
        3.0 * localization_radius_m if maximum_displacement_m is None else float(maximum_displacement_m)
    )
    if displacement_limit <= 0.0:
        raise ValueError("maximum displacement must be positive")
    increment = (
        gain
        * weight
        * np.clip(
            innovation,
            -displacement_limit,
            displacement_limit,
        )
    )
    corrected = forecast + increment
    support = weight > 0.01
    return FrontCorrection(
        corrected_level_set_m=corrected.astype(np.float32),
        level_set_increment_m=increment.astype(np.float32),
        localization_weight=weight.astype(np.float32),
        diagnostics={
            "support_cells": int(support.sum()),
            "mean_absolute_increment_m": (
                float(np.mean(np.abs(increment[support]))) if support.any() else 0.0
            ),
            "maximum_absolute_increment_m": float(np.max(np.abs(increment))),
            "localization_radius_m": float(localization_radius_m),
            "gain": float(gain),
        },
    )
