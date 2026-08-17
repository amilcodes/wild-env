"""Provenance-graded wildfire aviation vehicle profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeolus.config import ResourceSpec

PARAMETER_BASES = {
    "published",
    "unit_conversion",
    "modeling_assumption",
    "role_mapping",
}
EVIDENCE_GRADES = {
    "scenario_assumption",
    "public_specification",
    "flight_manual",
    "engineering_validated",
}
AUTONOMY_LEVELS = {
    "crewed",
    "remotely_piloted",
    "supervised_autonomy",
    "research_autonomy",
}
RESOURCE_KINDS = {"water", "retardant", "sensor", "crew"}
REQUIRED_SIMULATOR_PARAMETERS = {
    "cruise_speed_m_s",
    "payload_l",
    "reload_min",
    "dispatch_latency_min",
    "endurance_min",
}
RESOURCE_PARAMETER_NAMES = {
    "cruise_speed_m_s",
    "payload_l",
    "reload_min",
    "dispatch_latency_min",
    "endurance_min",
    "water_radius_m",
    "retardant_length_m",
    "retardant_width_m",
    "line_length_m",
    "line_width_m",
    "line_production_m_min",
    "max_operating_wind_m_s",
    "max_direct_intensity_kw_m",
    "target_coverage_level_gpc",
    "reserve_endurance_min",
    "drop_speed_m_s",
    "minimum_drop_length_m",
    "maximum_drop_length_m",
    "cruise_altitude_agl_m",
    "minimum_terrain_clearance_m",
    "maximum_operating_altitude_m_msl",
    "maximum_crosswind_m_s",
    "minimum_service_depth_m",
    "minimum_service_length_m",
}
INTEGER_RESOURCE_PARAMETERS = {
    "reload_min",
    "dispatch_latency_min",
    "endurance_min",
}


@dataclass(frozen=True)
class VehicleParameter:
    """One simulator value and the evidence used to obtain it."""

    value: float
    unit: str
    basis: str
    source_url: str | None = None
    source_value: str | None = None
    note: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VehicleParameter:
        parameter = cls(
            value=float(value["value"]),
            unit=str(value["unit"]),
            basis=str(value["basis"]),
            source_url=(str(value["source_url"]) if value.get("source_url") is not None else None),
            source_value=(str(value["source_value"]) if value.get("source_value") is not None else None),
            note=(str(value["note"]) if value.get("note") is not None else None),
        )
        parameter.validate()
        return parameter

    def validate(self) -> None:
        if not self.unit.strip():
            raise ValueError("vehicle parameter unit cannot be empty")
        if self.basis not in PARAMETER_BASES:
            raise ValueError(f"unknown vehicle parameter basis: {self.basis}")
        if self.basis in {"published", "unit_conversion"} and not self.source_url:
            raise ValueError("published and converted parameters require a source URL")


@dataclass(frozen=True)
class VehicleProfile:
    """An operational identity plus simulator parameterization."""

    profile_id: str
    display_name: str
    operator_reference: str
    resource_kind: str
    autonomy_level: str
    operational_roles: tuple[str, ...]
    operational_status: str
    identity_sources: tuple[str, ...]
    simulation_evidence_grade: str
    performance_surface_path: str | None
    delivery_evidence_grade: str
    delivery_surface_path: str | None
    parameters: dict[str, VehicleParameter]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VehicleProfile:
        profile = cls(
            profile_id=str(value["profile_id"]),
            display_name=str(value["display_name"]),
            operator_reference=str(value["operator_reference"]),
            resource_kind=str(value["resource_kind"]),
            autonomy_level=str(value["autonomy_level"]),
            operational_roles=tuple(str(role) for role in value["operational_roles"]),
            operational_status=str(value["operational_status"]),
            identity_sources=tuple(str(source) for source in value["identity_sources"]),
            simulation_evidence_grade=str(value["simulation_evidence_grade"]),
            performance_surface_path=(
                str(value["performance_surface_path"])
                if value.get("performance_surface_path") is not None
                else None
            ),
            delivery_evidence_grade=str(value.get("delivery_evidence_grade", "scenario_assumption")),
            delivery_surface_path=(
                str(value["delivery_surface_path"])
                if value.get("delivery_surface_path") is not None
                else None
            ),
            parameters={
                str(name): VehicleParameter.from_dict(parameter)
                for name, parameter in value["simulator_parameters"].items()
            },
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        for name, value in (
            ("profile_id", self.profile_id),
            ("display_name", self.display_name),
            ("operator_reference", self.operator_reference),
            ("operational_status", self.operational_status),
        ):
            if not value.strip():
                raise ValueError(f"vehicle {name} cannot be empty")
        if self.resource_kind not in RESOURCE_KINDS:
            raise ValueError("vehicle profile has an invalid resource kind")
        if self.autonomy_level not in AUTONOMY_LEVELS:
            raise ValueError("vehicle profile has an invalid autonomy level")
        if self.simulation_evidence_grade not in EVIDENCE_GRADES:
            raise ValueError("vehicle profile has an invalid evidence grade")
        if self.delivery_evidence_grade not in EVIDENCE_GRADES:
            raise ValueError("vehicle profile has an invalid delivery evidence grade")
        if not self.operational_roles or not self.identity_sources:
            raise ValueError("vehicle profile requires roles and identity sources")
        missing = REQUIRED_SIMULATOR_PARAMETERS.difference(self.parameters)
        if missing:
            raise ValueError(f"vehicle profile is missing simulator parameters: {sorted(missing)}")
        if self.parameters["cruise_speed_m_s"].value <= 0.0:
            raise ValueError("vehicle cruise speed must be positive")
        if self.parameters["payload_l"].value < 0.0:
            raise ValueError("vehicle payload cannot be negative")
        if self.parameters["endurance_min"].value <= 0.0:
            raise ValueError("vehicle endurance must be positive")
        if (
            self.simulation_evidence_grade in {"flight_manual", "engineering_validated"}
            and self.performance_surface_path is None
        ):
            raise ValueError("field-grade vehicle evidence requires a performance surface")
        if (
            self.delivery_evidence_grade in {"flight_manual", "engineering_validated"}
            and self.delivery_surface_path is None
        ):
            raise ValueError("field-grade delivery evidence requires a delivery surface")

    @property
    def field_performance_ready(self) -> bool:
        return (
            self.simulation_evidence_grade in {"flight_manual", "engineering_validated"}
            and self.performance_surface_path is not None
        )


@dataclass(frozen=True)
class VehicleCatalog:
    schema_version: int
    catalog_id: str
    researched_at: str
    profiles: tuple[VehicleProfile, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported vehicle catalog schema")
        if not self.catalog_id.strip() or not self.profiles:
            raise ValueError("vehicle catalog requires an identifier and profiles")
        identifiers = [profile.profile_id for profile in self.profiles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("vehicle catalog profile identifiers must be unique")

    def profile(self, profile_id: str) -> VehicleProfile:
        return next(
            (profile for profile in self.profiles if profile.profile_id == profile_id),
            None,
        ) or _missing_profile(profile_id)


def _missing_profile(profile_id: str) -> VehicleProfile:
    raise KeyError(f"unknown vehicle profile: {profile_id}")


def load_vehicle_catalog(path: str | Path) -> VehicleCatalog:
    """Load and validate a wildfire aviation catalog."""

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate vehicle-catalog key: {key}")
            value[key] = item
        return value

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    catalog = VehicleCatalog(
        schema_version=int(payload["schema_version"]),
        catalog_id=str(payload["catalog_id"]),
        researched_at=str(payload["researched_at"]),
        profiles=tuple(VehicleProfile.from_dict(profile) for profile in payload["profiles"]),
    )
    catalog.validate()
    return catalog


def resource_spec_from_profile(
    profile: VehicleProfile,
    *,
    resource_id: str,
    overrides: dict[str, Any] | None = None,
) -> ResourceSpec:
    """Materialize a simulator resource while retaining evidence status."""

    values: dict[str, Any] = {
        name: (int(parameter.value) if name in INTEGER_RESOURCE_PARAMETERS else parameter.value)
        for name, parameter in profile.parameters.items()
        if name in RESOURCE_PARAMETER_NAMES
    }
    values.update(
        {
            "resource_id": resource_id,
            "kind": profile.resource_kind,
            "vehicle_profile_id": profile.profile_id,
            "performance_evidence_grade": (profile.simulation_evidence_grade),
            "autonomy_level": profile.autonomy_level,
            "operational_roles": profile.operational_roles,
            "performance_surface_path": profile.performance_surface_path,
            "delivery_surface_path": profile.delivery_surface_path,
            "delivery_evidence_grade": profile.delivery_evidence_grade,
        }
    )
    values.update(overrides or {})
    return ResourceSpec(**values)


def audit_vehicle_catalog(catalog: VehicleCatalog) -> dict[str, Any]:
    """Summarize evidence, autonomy, role, and field-readiness coverage."""

    basis_counts = {basis: 0 for basis in sorted(PARAMETER_BASES)}
    autonomy_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    assumption_profiles: list[str] = []
    field_ready: list[str] = []
    delivery_ready: list[str] = []
    for profile in catalog.profiles:
        autonomy_counts[profile.autonomy_level] = autonomy_counts.get(profile.autonomy_level, 0) + 1
        for role in profile.operational_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        if any(parameter.basis == "modeling_assumption" for parameter in profile.parameters.values()):
            assumption_profiles.append(profile.profile_id)
        if profile.field_performance_ready:
            field_ready.append(profile.profile_id)
        if (
            profile.delivery_evidence_grade in {"flight_manual", "engineering_validated"}
            and profile.delivery_surface_path is not None
        ):
            delivery_ready.append(profile.profile_id)
        for parameter in profile.parameters.values():
            basis_counts[parameter.basis] += 1
    return {
        "catalog_id": catalog.catalog_id,
        "profile_count": len(catalog.profiles),
        "autonomy_counts": dict(sorted(autonomy_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "parameter_basis_counts": basis_counts,
        "profiles_with_modeling_assumptions": sorted(assumption_profiles),
        "field_performance_ready_profiles": sorted(field_ready),
        "field_performance_ready_count": len(field_ready),
        "delivery_evidence_ready_profiles": sorted(delivery_ready),
        "delivery_evidence_ready_count": len(delivery_ready),
        "interpretation": (
            "Current-use identity and nominal public specifications are "
            "separate from flight-manual or engineering-validated performance."
        ),
    }
