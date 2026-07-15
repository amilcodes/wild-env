from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.data.aviation_catalog import (
    VehicleProfile,
    audit_vehicle_catalog,
    load_vehicle_catalog,
    resource_spec_from_profile,
)

CATALOG_PATH = Path(__file__).parents[1] / "configs" / "aviation" / "us_wildfire_reference_fleet_v1.json"


def test_reference_catalog_is_complete_and_honest_about_evidence() -> None:
    catalog = load_vehicle_catalog(CATALOG_PATH)
    audit = audit_vehicle_catalog(catalog)

    assert len(catalog.profiles) == 9
    assert audit["autonomy_counts"] == {
        "crewed": 6,
        "remotely_piloted": 2,
        "supervised_autonomy": 1,
    }
    assert audit["field_performance_ready_count"] == 0
    assert audit["delivery_evidence_ready_profiles"] == ["calfire-s2t-retardant"]
    assert audit["profiles_with_modeling_assumptions"]
    assert all(profile.identity_sources for profile in catalog.profiles)


def test_catalog_profile_materializes_a_traceable_resource() -> None:
    catalog = load_vehicle_catalog(CATALOG_PATH)
    profile = catalog.profile("calfire-s70i-water")
    resource = resource_spec_from_profile(
        profile,
        resource_id="firehawk-01",
        overrides={
            "home_site_id": "helibase",
            "service_modes": ("land", "hover_fill"),
        },
    )

    assert resource.kind == "water"
    assert resource.payload_l == pytest.approx(3785.411784)
    assert resource.vehicle_profile_id == profile.profile_id
    assert resource.performance_evidence_grade == "scenario_assumption"
    assert resource.autonomy_level == "crewed"
    assert resource.home_site_id == "helibase"


def test_field_grade_claim_requires_performance_surface() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["profiles"][0]
    payload["profile_id"] = "invalid-field-grade-claim"
    payload["simulation_evidence_grade"] = "flight_manual"
    payload["performance_surface_path"] = None

    with pytest.raises(ValueError, match="performance surface"):
        VehicleProfile.from_dict(payload)


def test_field_grade_delivery_claim_requires_delivery_surface() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["profiles"][1]
    payload["profile_id"] = "invalid-field-grade-delivery-claim"
    payload["delivery_evidence_grade"] = "engineering_validated"
    payload["delivery_surface_path"] = None

    with pytest.raises(ValueError, match="delivery surface"):
        VehicleProfile.from_dict(payload)


def test_catalog_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_vehicle_catalog(path)
