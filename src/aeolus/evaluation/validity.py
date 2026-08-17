"""Declared validity controls for historical fuels and fast fire behavior."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import cos, radians
from typing import Any

import numpy as np

from aeolus.core.fire_behavior import FireBehaviorLookup
from aeolus.core.state import FireType
from aeolus.data.bundle import ScenarioBundle
from aeolus.data.weather import WeatherForcing


def assess_metric_crs(
    scenario: ScenarioBundle,
    *,
    latitude_deg: float | None = None,
) -> dict[str, Any]:
    """Audit whether raster map units are suitable as physical ground metres.

    ``cell_size_m`` drives spread, distance, area, aviation, and suppression
    calculations. A projected CRS with metre axes is necessary. Web Mercator
    is explicitly rejected because its map scale varies with latitude.
    """

    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[geo] to audit scenario CRS") from exc

    crs = CRS.from_user_input(str(scenario.metadata["crs"]))
    authority = crs.to_authority()
    epsg = crs.to_epsg()
    axes_in_metres = bool(crs.axis_info) and all(
        str(axis.unit_name).lower() in {"metre", "meter"} for axis in crs.axis_info[:2]
    )
    web_mercator = epsg in {3857, 3785, 900913, 102100, 102113}
    projected_metric = bool(crs.is_projected and axes_in_metres and not web_mercator)
    bbox = scenario.metadata.get("source_bbox_wgs84")
    inferred_latitude = (
        0.5 * (float(bbox[1]) + float(bbox[3])) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
    )
    latitude = float(latitude_deg) if latitude_deg is not None else inferred_latitude
    ground_scale = cos(radians(latitude)) if web_mercator else None
    return {
        "crs": crs.to_string(),
        "authority": list(authority) if authority is not None else None,
        "epsg": epsg,
        "is_projected": bool(crs.is_projected),
        "axes_in_metres": axes_in_metres,
        "web_mercator": web_mercator,
        "supports_physical_distance_claims": projected_metric,
        "audit_latitude_deg": latitude,
        "approximate_ground_metres_per_map_metre": ground_scale,
        "approximate_ground_area_per_map_area": (ground_scale**2 if ground_scale is not None else None),
        "failure_reason": (
            None
            if projected_metric
            else (
                "Web Mercator map units have latitude-dependent scale"
                if web_mercator
                else "scenario CRS is not a projected two-axis metre CRS"
            )
        ),
    }


def assess_forcing_availability(
    forcing: WeatherForcing,
    *,
    forecast_start: datetime | str,
    forecast_end: datetime | str,
) -> dict[str, Any]:
    """Separate retrospective analysis-forced and operational forecast skill."""

    forcing.validate()
    start = _utc(forecast_start)
    end = _utc(forecast_end)
    if end <= start:
        raise ValueError("forecast end must follow forecast start")
    origin = forcing.time_origin
    if origin is None:
        return {
            "forcing_class": "unknown",
            "supports_retrospective_hindcast": False,
            "supports_operational_forecast": False,
            "failure_reason": "forcing has no absolute time origin",
        }
    valid_start = origin + timedelta(minutes=float(forcing.minute[0]))
    valid_end = origin + timedelta(minutes=float(forcing.minute[-1]))
    covers = bool(valid_start <= start and valid_end >= end)
    kind = str(forcing.metadata.get("analysis_or_forecast", "unknown")).lower()
    if kind == "forecast":
        reference_value = forcing.metadata.get("forecast_reference_time")
        reference = _utc(str(reference_value)) if reference_value is not None else None
        lag = float(forcing.metadata.get("assumed_availability_lag_hours", 0.0))
        available = reference is not None and (reference.timestamp() + lag * 3600.0 <= start.timestamp())
        return {
            "forcing_class": "archived_operational_forecast",
            "forecast_reference_time": (reference.isoformat() if reference is not None else None),
            "assumed_availability_lag_hours": lag,
            "valid_time_coverage": covers,
            "available_by_forecast_start": bool(available),
            "uses_verifying_future_analysis": False,
            "supports_retrospective_hindcast": bool(covers),
            "supports_operational_forecast": bool(covers and available),
            "failure_reason": (
                None if covers and available else "forecast cycle availability or valid-time coverage failed"
            ),
        }
    return {
        "forcing_class": "retrospective_analysis_or_reanalysis",
        "valid_time_coverage": covers,
        "uses_verifying_future_analysis": bool(valid_end > start),
        "supports_retrospective_hindcast": bool(covers),
        "supports_operational_forecast": False,
        "failure_reason": ("future analysis/reanalysis fields were not available at forecast issue time"),
    }


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_product_year(source: dict[str, Any]) -> int | None:
    for key in ("product_year", "vintage_year", "version_year"):
        if key in source:
            try:
                return int(source[key])
            except (TypeError, ValueError):
                return None
    match = re.search(r"\b(19|20)\d{2}\b", str(source.get("name", "")))
    return int(match.group(0)) if match else None


@dataclass(frozen=True)
class FuelProvenanceAssessment:
    """Historical-use audit for landscape fuel products."""

    incident_start: str
    fuel_sources: tuple[str, ...]
    product_years: tuple[int, ...]
    disturbance_through_years: tuple[int, ...]
    missing_product_year_sources: tuple[str, ...]
    post_incident_product_sources: tuple[str, ...]
    incident_or_later_disturbance_sources: tuple[str, ...]
    same_year_ambiguous_sources: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_historical_fuel_provenance(
    scenario: ScenarioBundle,
    *,
    incident_start: datetime | str,
) -> FuelProvenanceAssessment:
    """Detect potential use of future landscape information.

    A nationally updated fuel product can include disturbance or succession
    information unavailable at the incident time.  Product year is therefore
    treated as a leakage guard rather than proof that any particular pixel was
    changed.  Same-year products remain ambiguous unless an explicit
    ``data_cutoff`` precedes the incident.
    """

    incident = _utc(incident_start)
    sources = [
        dict(source)
        for source in scenario.metadata.get("sources", [])
        if any(token in str(source.get("name", "")).lower() for token in ("landfire", "fuel", "vegetation"))
    ]
    names: list[str] = []
    years: list[int] = []
    disturbance_years: list[int] = []
    missing: list[str] = []
    future: list[str] = []
    future_disturbance: list[str] = []
    ambiguous: list[str] = []
    for source in sources:
        name = str(source.get("name", "<unnamed fuel source>"))
        names.append(name)
        if "disturbance_through_year" in source:
            try:
                disturbance_year = int(source["disturbance_through_year"])
                disturbance_years.append(disturbance_year)
                if disturbance_year >= incident.year:
                    future_disturbance.append(name)
            except (TypeError, ValueError):
                missing.append(name)
        year = _source_product_year(source)
        if year is None:
            missing.append(name)
            continue
        years.append(year)
        if year > incident.year:
            future.append(name)
            continue
        if year == incident.year:
            cutoff_value = source.get("data_cutoff")
            if cutoff_value is None:
                ambiguous.append(name)
                continue
            try:
                cutoff = _utc(str(cutoff_value))
            except ValueError:
                ambiguous.append(name)
                continue
            if cutoff > incident:
                future.append(name)
    if future or future_disturbance:
        status = "potential_post_incident_information"
    elif ambiguous:
        status = "same_year_cutoff_unresolved"
    elif missing or not sources:
        status = "incomplete_provenance"
    else:
        status = "historically_admissible_by_product_date"
    return FuelProvenanceAssessment(
        incident_start=incident.isoformat(),
        fuel_sources=tuple(names),
        product_years=tuple(years),
        disturbance_through_years=tuple(disturbance_years),
        missing_product_year_sources=tuple(missing),
        post_incident_product_sources=tuple(future),
        incident_or_later_disturbance_sources=tuple(future_disturbance),
        same_year_ambiguous_sources=tuple(ambiguous),
        status=status,
    )


def _outside_count(values: np.ndarray, lower: float, upper: float) -> int:
    array = np.asarray(values, dtype=np.float64)
    return int(((array < lower) | (array > upper) | ~np.isfinite(array)).sum())


def assess_fast_kernel_validity(
    *,
    wind_speed_m_s: np.ndarray | float,
    slope_tan: np.ndarray | float,
    moisture_dead_1h: np.ndarray | float,
    moisture_live_herbaceous: np.ndarray | float,
    moisture_live_woody: np.ndarray | float,
    fire_type: np.ndarray | None = None,
    spot_ignition_count: int = 0,
) -> dict[str, Any]:
    """Report lookup clipping and mechanism-only fire regimes.

    Surface behavior inside the packaged table is the current empirical
    validity envelope.  Crown and spotting calculations are retained for
    mechanism studies but are marked separately because the project has not
    calibrated those transitions against independent incidents.
    """

    lookup = FireBehaviorLookup()
    fields = {
        "wind_speed_m_s": (
            np.asarray(wind_speed_m_s),
            float(lookup.wind_grid[0]),
            float(lookup.wind_grid[-1]),
        ),
        "slope_tan": (
            np.asarray(slope_tan),
            float(lookup.slope_grid[0]),
            float(lookup.slope_grid[-1]),
        ),
        "moisture_dead_1h": (
            np.asarray(moisture_dead_1h),
            float(lookup.moisture_grid[0]),
            float(lookup.moisture_grid[-1]),
        ),
        "moisture_live_herbaceous": (
            np.asarray(moisture_live_herbaceous),
            float(lookup.live_herbaceous_grid[0]),
            float(lookup.live_herbaceous_grid[-1]),
        ),
        "moisture_live_woody": (
            np.asarray(moisture_live_woody),
            float(lookup.live_woody_grid[0]),
            float(lookup.live_woody_grid[-1]),
        ),
    }
    domain: dict[str, Any] = {}
    outside_total = 0
    evaluated_total = 0
    for name, (values, lower, upper) in fields.items():
        outside = _outside_count(values, lower, upper)
        count = int(values.size)
        outside_total += outside
        evaluated_total += count
        domain[name] = {
            "minimum": lower,
            "maximum": upper,
            "evaluated_values": count,
            "outside_values": outside,
            "outside_fraction": outside / max(count, 1),
        }

    passive_crown = 0
    active_crown = 0
    if fire_type is not None:
        types = np.asarray(fire_type)
        passive_crown = int((types == int(FireType.PASSIVE_CROWN)).sum())
        active_crown = int((types == int(FireType.ACTIVE_CROWN)).sum())
    mechanism_only = passive_crown + active_crown > 0 or spot_ignition_count > 0
    if outside_total:
        classification = "outside_lookup_domain"
    elif mechanism_only:
        classification = "mechanism_only_unvalidated_regime"
    else:
        classification = "surface_lookup_domain"
    return {
        "classification": classification,
        "supports_current_accuracy_claim": classification == "surface_lookup_domain",
        "lookup_domain": domain,
        "outside_value_count": outside_total,
        "evaluated_value_count": evaluated_total,
        "passive_crown_cells": passive_crown,
        "active_crown_cells": active_crown,
        "spot_ignition_count": int(spot_ignition_count),
        "interpretation": (
            "Lookup-domain checks detect numerical clipping. Crown and spotting "
            "are marked mechanism-only until independently calibrated."
        ),
    }
