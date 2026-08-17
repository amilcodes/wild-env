from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.data.aviation_catalog import load_vehicle_catalog
from aeolus.data.aviation_evidence import (
    audit_aviation_evidence,
    load_aviation_evidence_registry,
)

ROOT = Path(__file__).parents[1]
CATALOG_PATH = ROOT / "configs" / "aviation" / "us_wildfire_reference_fleet_v1.json"
REGISTRY_PATH = ROOT / "configs" / "aviation" / "evidence_registry_v1.json"


def test_registry_covers_catalog_and_keeps_variant_boundaries() -> None:
    catalog = load_vehicle_catalog(CATALOG_PATH)
    registry = load_aviation_evidence_registry(REGISTRY_PATH, catalog=catalog)
    audit = audit_aviation_evidence(registry, catalog)

    assert audit["profile_count"] == 9
    assert audit["document_count"] >= 19
    assert audit["field_closed_profile_count"] == 0
    by_profile = {item["profile_id"]: item for item in audit["profiles"]}
    c130 = by_profile["calfire-c130h-retardant"]["domains"]
    assert c130["environmental_performance"]["status"] == "proxy"
    assert c130["environmental_performance"]["applicability"] == "base_airframe_only"
    s2t = by_profile["calfire-s2t-retardant"]["domains"]
    assert s2t["delivery_pattern"]["status"] == "partial"
    assert s2t["delivery_pattern"]["evidence_grade"] == "engineering_validated"


def test_registry_rejects_false_closed_claim(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    claim = payload["profiles"][0]["domains"]["mission_mobility"]
    claim["status"] = "closed"
    path = tmp_path / "false-closure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact-current"):
        load_aviation_evidence_registry(path)
