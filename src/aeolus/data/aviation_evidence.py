"""Configuration-specific evidence registry for wildfire aircraft."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeolus.data.aviation_catalog import VehicleCatalog

EVIDENCE_STATUSES = {"open", "proxy", "partial", "closed"}
EVIDENCE_GRADES = {
    "scenario_assumption",
    "public_specification",
    "flight_manual",
    "engineering_validated",
}
APPLICABILITY_LEVELS = {
    "exact_current_configuration",
    "exact_historical_configuration",
    "base_airframe_only",
    "operational_standard",
    "role_only",
    "none",
}
COMMON_DOMAINS = {
    "configuration_identity",
    "mission_mobility",
    "mass_balance",
    "environmental_performance",
    "endurance_reserve",
    "dispatch_turnaround",
}
SUPPRESSANT_DOMAINS = {"delivery_system", "delivery_pattern"}
WATER_DOMAINS = {"refill_site_envelope"}
SENSOR_DOMAINS = {"mission_payload"}


@dataclass(frozen=True)
class EvidenceDocument:
    document_id: str
    title: str
    publisher: str
    issued: str
    url: str
    document_class: str
    local_cache_path: str | None
    sha256: str | None
    access_note: str | None


@dataclass(frozen=True)
class DomainEvidence:
    status: str
    evidence_grade: str
    applicability: str
    document_ids: tuple[str, ...]
    finding: str

    def validate(self, document_ids: set[str]) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"unknown aviation evidence status: {self.status}")
        if self.evidence_grade not in EVIDENCE_GRADES:
            raise ValueError(f"unknown aviation evidence grade: {self.evidence_grade}")
        if self.applicability not in APPLICABILITY_LEVELS:
            raise ValueError(f"unknown aviation evidence applicability: {self.applicability}")
        missing = set(self.document_ids).difference(document_ids)
        if missing:
            raise ValueError(f"aviation evidence references unknown documents: {sorted(missing)}")
        if self.status != "open" and not self.document_ids:
            raise ValueError("non-open aviation evidence requires a document")
        if self.status == "closed" and (
            self.applicability != "exact_current_configuration"
            or self.evidence_grade not in {"flight_manual", "engineering_validated"}
        ):
            raise ValueError(
                "closed aviation domains require exact-current flight-manual or engineering evidence"
            )


@dataclass(frozen=True)
class ProfileEvidence:
    profile_id: str
    domains: dict[str, DomainEvidence]


@dataclass(frozen=True)
class AviationEvidenceRegistry:
    registry_id: str
    researched_at: str
    documents: tuple[EvidenceDocument, ...]
    profiles: tuple[ProfileEvidence, ...]

    def validate(self, catalog: VehicleCatalog | None = None) -> None:
        if not self.registry_id.strip() or not self.documents or not self.profiles:
            raise ValueError("aviation evidence registry requires identity, documents, and profiles")
        document_ids = [item.document_id for item in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("aviation evidence document identifiers must be unique")
        profile_ids = [item.profile_id for item in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("aviation evidence profile identifiers must be unique")
        for profile in self.profiles:
            for evidence in profile.domains.values():
                evidence.validate(set(document_ids))
        if catalog is not None:
            catalog_ids = {profile.profile_id for profile in catalog.profiles}
            if set(profile_ids) != catalog_ids:
                raise ValueError("aviation evidence registry must cover every catalog profile exactly once")


def load_aviation_evidence_registry(
    path: str | Path,
    *,
    catalog: VehicleCatalog | None = None,
) -> AviationEvidenceRegistry:
    """Load and validate the aircraft evidence/closure registry."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported aviation evidence registry schema")
    registry = AviationEvidenceRegistry(
        registry_id=str(payload["registry_id"]),
        researched_at=str(payload["researched_at"]),
        documents=tuple(
            EvidenceDocument(
                document_id=str(item["document_id"]),
                title=str(item["title"]),
                publisher=str(item["publisher"]),
                issued=str(item["issued"]),
                url=str(item["url"]),
                document_class=str(item["document_class"]),
                local_cache_path=(
                    str(item["local_cache_path"]) if item.get("local_cache_path") is not None else None
                ),
                sha256=(str(item["sha256"]) if item.get("sha256") is not None else None),
                access_note=(str(item["access_note"]) if item.get("access_note") is not None else None),
            )
            for item in payload["documents"]
        ),
        profiles=tuple(
            ProfileEvidence(
                profile_id=str(item["profile_id"]),
                domains={
                    str(domain): DomainEvidence(
                        status=str(value["status"]),
                        evidence_grade=str(value["evidence_grade"]),
                        applicability=str(value["applicability"]),
                        document_ids=tuple(str(doc) for doc in value.get("document_ids", [])),
                        finding=str(value["finding"]),
                    )
                    for domain, value in item["domains"].items()
                },
            )
            for item in payload["profiles"]
        ),
    )
    registry.validate(catalog)
    return registry


def required_evidence_domains(resource_kind: str) -> set[str]:
    domains = set(COMMON_DOMAINS)
    if resource_kind in {"water", "retardant"}:
        domains.update(SUPPRESSANT_DOMAINS)
    if resource_kind == "water":
        domains.update(WATER_DOMAINS)
    if resource_kind == "sensor":
        domains.update(SENSOR_DOMAINS)
    return domains


def audit_aviation_evidence(
    registry: AviationEvidenceRegistry,
    catalog: VehicleCatalog,
) -> dict[str, Any]:
    """Return a strict per-domain closure audit for every selected vehicle."""

    registry.validate(catalog)
    evidence_by_profile = {item.profile_id: item for item in registry.profiles}
    status_weight = {"open": 0.0, "proxy": 0.25, "partial": 0.5, "closed": 1.0}
    profiles: list[dict[str, Any]] = []
    status_counts = {status: 0 for status in sorted(EVIDENCE_STATUSES)}
    for vehicle in catalog.profiles:
        profile = evidence_by_profile[vehicle.profile_id]
        required = sorted(required_evidence_domains(vehicle.resource_kind))
        domain_rows: dict[str, Any] = {}
        for domain in required:
            evidence = profile.domains.get(
                domain,
                DomainEvidence(
                    status="open",
                    evidence_grade="scenario_assumption",
                    applicability="none",
                    document_ids=(),
                    finding="No configuration-specific public evidence located.",
                ),
            )
            status_counts[evidence.status] += 1
            domain_rows[domain] = {
                "status": evidence.status,
                "evidence_grade": evidence.evidence_grade,
                "applicability": evidence.applicability,
                "document_ids": list(evidence.document_ids),
                "finding": evidence.finding,
            }
        closed = [name for name, value in domain_rows.items() if value["status"] == "closed"]
        unresolved = [name for name, value in domain_rows.items() if value["status"] != "closed"]
        score = sum(status_weight[value["status"]] for value in domain_rows.values()) / len(required)
        profiles.append(
            {
                "profile_id": vehicle.profile_id,
                "display_name": vehicle.display_name,
                "resource_kind": vehicle.resource_kind,
                "closure_score": float(score),
                "field_closed": not unresolved,
                "closed_domains": closed,
                "unresolved_domains": unresolved,
                "domains": domain_rows,
            }
        )
    return {
        "registry_id": registry.registry_id,
        "researched_at": registry.researched_at,
        "profile_count": len(profiles),
        "document_count": len(registry.documents),
        "field_closed_profile_count": sum(item["field_closed"] for item in profiles),
        "domain_status_counts": status_counts,
        "profiles": profiles,
        "interpretation": (
            "Closed means exact-current configuration evidence from an approved flight manual "
            "or controlled engineering validation. Partial and proxy evidence can constrain "
            "research experiments but cannot support dispatch or flight-safety decisions."
        ),
    }
