"""Measured aerial-delivery envelopes and simulator geometry transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

GPC_L_M2 = 3.785411784 / 9.290304


@dataclass(frozen=True)
class AerialDeliverySurface:
    """One controlled drop-test series for a declared delivery system."""

    surface_id: str
    platform_configuration: str
    material: str
    nominal_payload_l: float
    coverage_level_gpc: np.ndarray
    flow_rate_l_s: np.ndarray
    controller_setting: tuple[str, ...]
    longest_line_m: np.ndarray
    test_airspeed_m_s: tuple[float, float]
    drop_height_m: tuple[float, float]
    transform_coverage_gpc: tuple[float, float]
    metadata: dict[str, Any]

    def validate(self) -> None:
        coverage = np.asarray(self.coverage_level_gpc, dtype=np.float64)
        flow = np.asarray(self.flow_rate_l_s, dtype=np.float64)
        line = np.asarray(self.longest_line_m, dtype=np.float64)
        if not self.surface_id.strip() or not self.platform_configuration.strip():
            raise ValueError("delivery surface requires an identity and configuration")
        if self.material not in {"water", "gum_thickened_retardant", "water_enhancing_gel"}:
            raise ValueError("delivery surface has an unsupported material")
        if coverage.ndim != 1 or coverage.size < 2 or np.any(np.diff(coverage) <= 0.0):
            raise ValueError("coverage levels must be a strictly increasing vector")
        if flow.shape != coverage.shape or line.shape != coverage.shape:
            raise ValueError("delivery response vectors must match the coverage axis")
        if len(self.controller_setting) != coverage.size:
            raise ValueError("delivery controller settings must match the coverage axis")
        if self.nominal_payload_l <= 0.0 or np.any(flow <= 0.0) or np.any(line <= 0.0):
            raise ValueError("delivery payload, flow, and line length must be positive")
        for name, bounds in (
            ("airspeed", self.test_airspeed_m_s),
            ("drop height", self.drop_height_m),
            ("transform coverage", self.transform_coverage_gpc),
        ):
            if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
                raise ValueError(f"delivery {name} domain is invalid")
        if self.transform_coverage_gpc[0] < coverage[0] or self.transform_coverage_gpc[1] > coverage[-1]:
            raise ValueError("delivery transform domain must lie within measured coverage")
        if not str(self.metadata.get("source_url", "")).strip():
            raise ValueError("delivery surface requires source provenance")
        if not str(self.metadata.get("configuration_applicability", "")).strip():
            raise ValueError("delivery surface requires an applicability statement")


@dataclass(frozen=True)
class DeliveryGeometry:
    """Volume-conserving simulator proxy derived from a measured line table."""

    surface_id: str
    requested_coverage_gpc: float
    line_length_m: float
    effective_width_m: float
    flow_rate_l_s: float
    controller_setting: str
    payload_fraction: float
    extrapolated: bool


def _surface_from_payload(payload: dict[str, Any]) -> AerialDeliverySurface:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported aerial-delivery schema")
    domain = payload["test_domain"]
    response = payload["response"]
    surface = AerialDeliverySurface(
        surface_id=str(payload["surface_id"]),
        platform_configuration=str(payload["platform_configuration"]),
        material=str(payload["material"]),
        nominal_payload_l=float(payload["nominal_payload_l"]),
        coverage_level_gpc=np.asarray(response["coverage_level_gpc"], dtype=np.float64),
        flow_rate_l_s=np.asarray(response["flow_rate_l_s"], dtype=np.float64),
        controller_setting=tuple(str(value) for value in response["controller_setting"]),
        longest_line_m=np.asarray(response["longest_line_m"], dtype=np.float64),
        test_airspeed_m_s=tuple(float(value) for value in domain["airspeed_m_s"]),
        drop_height_m=tuple(float(value) for value in domain["drop_height_m"]),
        transform_coverage_gpc=tuple(
            float(value) for value in payload["simulator_transform_domain"]["coverage_gpc"]
        ),
        metadata=dict(payload["metadata"]),
    )
    surface.validate()
    return surface


@lru_cache(maxsize=32)
def load_aerial_delivery_surface(path: str | Path) -> AerialDeliverySurface:
    """Load a measured aerial-delivery surface from JSON."""

    return _surface_from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def delivery_geometry(
    surface: AerialDeliverySurface,
    *,
    requested_coverage_gpc: float,
    payload_l: float,
) -> DeliveryGeometry:
    """Map a drop-test line table into a volume-conserving 2-D footprint.

    The source reports the longest measured line at each coverage contour, not
    a complete gridded deposition field.  The returned width is therefore an
    explicit uniform-coverage equivalent: payload divided by requested areal
    coverage and measured line length.  It is a simulator transform, not an
    additional measurement.
    """

    if requested_coverage_gpc <= 0.0 or payload_l <= 0.0:
        raise ValueError("delivery coverage and payload must be positive")
    axis = surface.coverage_level_gpc
    transform_min, transform_max = surface.transform_coverage_gpc
    clipped = float(np.clip(requested_coverage_gpc, transform_min, transform_max))
    extrapolated = not bool(transform_min <= requested_coverage_gpc <= transform_max)
    nominal_line_m = float(np.interp(clipped, axis, surface.longest_line_m))
    flow_l_s = float(np.interp(clipped, axis, surface.flow_rate_l_s))
    payload_fraction = float(payload_l / surface.nominal_payload_l)
    line_m = nominal_line_m * payload_fraction
    area_m2 = payload_l / (clipped * GPC_L_M2)
    width_m = area_m2 / max(line_m, 1e-9)
    setting_index = int(np.argmin(np.abs(axis - clipped)))
    return DeliveryGeometry(
        surface_id=surface.surface_id,
        requested_coverage_gpc=clipped,
        line_length_m=line_m,
        effective_width_m=float(width_m),
        flow_rate_l_s=flow_l_s,
        controller_setting=surface.controller_setting[setting_index],
        payload_fraction=payload_fraction,
        extrapolated=extrapolated,
    )
