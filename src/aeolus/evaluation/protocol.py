"""Frozen-split and paired-policy evaluation controls."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    incident_id: str
    geography: str
    year: int
    fuel_family: str
    weather_regime: str
    split: str

    def validate(self) -> None:
        for name in (
            "case_id",
            "incident_id",
            "geography",
            "fuel_family",
            "weather_regime",
            "split",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"evaluation case {name} cannot be empty")
        if self.split not in {"train", "development", "test"}:
            raise ValueError("evaluation split must be train, development, or test")
        if not 1900 <= int(self.year) <= 2200:
            raise ValueError("evaluation case year is invalid")


def audit_evaluation_partitions(
    cases: Sequence[EvaluationCase],
    *,
    exclusive_fields: tuple[str, ...] = ("incident_id",),
) -> dict[str, Any]:
    """Audit frozen case partitions for group leakage and coverage."""

    if not cases:
        raise ValueError("evaluation partition requires at least one case")
    for case in cases:
        case.validate()
    case_ids = [case.case_id for case in cases]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    leakage: dict[str, list[dict[str, Any]]] = {}
    for field in exclusive_fields:
        values: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            if not hasattr(case, field):
                raise ValueError(f"unknown evaluation partition field: {field}")
            values[str(getattr(case, field))].add(case.split)
        leakage[field] = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(values.items())
            if len(splits) > 1
        ]

    split_summary: dict[str, Any] = {}
    for split in ("train", "development", "test"):
        members = [case for case in cases if case.split == split]
        split_summary[split] = {
            "cases": len(members),
            "incidents": len({case.incident_id for case in members}),
            "geographies": sorted({case.geography for case in members}),
            "years": sorted({case.year for case in members}),
            "fuel_families": sorted({case.fuel_family for case in members}),
            "weather_regimes": sorted({case.weather_regime for case in members}),
        }
    leakage_count = sum(len(items) for items in leakage.values())
    empty_splits = [split for split, summary in split_summary.items() if summary["cases"] == 0]
    return {
        "case_count": len(cases),
        "duplicate_case_ids": duplicates,
        "exclusive_fields": list(exclusive_fields),
        "group_leakage": leakage,
        "group_leakage_count": leakage_count,
        "empty_splits": empty_splits,
        "split_summary": split_summary,
        "valid": not duplicates and leakage_count == 0 and not empty_splits,
        "cases": [asdict(case) for case in cases],
    }


def _records_by_pair(
    records: Iterable[Mapping[str, Any]],
    *,
    policy: str,
    metric: str,
) -> dict[tuple[str, int], float]:
    selected: dict[tuple[str, int], float] = {}
    for record in records:
        if str(record["policy"]) != policy:
            continue
        key = (str(record["case_id"]), int(record["seed"]))
        if key in selected:
            raise ValueError(f"duplicate policy evaluation record: {policy} {key}")
        value = float(record[metric])
        if not np.isfinite(value):
            raise ValueError(f"policy metric {metric} must be finite")
        selected[key] = value
    return selected


def paired_policy_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_policy: str,
    baseline_policy: str,
    metric: str,
    lower_is_better: bool,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired effect with case-cluster bootstrap uncertainty.

    Seeds are paired within cases.  Bootstrap resampling occurs at case level,
    preserving within-incident dependence across seeds and scenario variants.
    Positive ``improvement`` always favors the candidate.
    """

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    candidate = _records_by_pair(
        records,
        policy=candidate_policy,
        metric=metric,
    )
    baseline = _records_by_pair(
        records,
        policy=baseline_policy,
        metric=metric,
    )
    common = sorted(candidate.keys() & baseline.keys())
    if not common:
        raise ValueError("candidate and baseline have no paired records")
    missing_candidate = sorted(baseline.keys() - candidate.keys())
    missing_baseline = sorted(candidate.keys() - baseline.keys())
    raw_delta = np.asarray(
        [candidate[key] - baseline[key] for key in common],
        dtype=np.float64,
    )
    improvement = -raw_delta if lower_is_better else raw_delta
    by_case: dict[str, list[float]] = defaultdict(list)
    for (case_id, _), value in zip(common, improvement, strict=True):
        by_case[case_id].append(float(value))
    case_ids = sorted(by_case)
    case_means = np.asarray(
        [np.mean(by_case[case_id]) for case_id in case_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        selected = rng.integers(0, len(case_ids), len(case_ids))
        samples[index] = float(np.mean(case_means[selected]))
    interval = np.quantile(samples, (0.025, 0.975))
    mean_improvement = float(case_means.mean())
    return {
        "candidate_policy": candidate_policy,
        "baseline_policy": baseline_policy,
        "metric": metric,
        "lower_is_better": bool(lower_is_better),
        "paired_records": len(common),
        "case_clusters": len(case_ids),
        "candidate_mean": float(np.mean([candidate[key] for key in common])),
        "baseline_mean": float(np.mean([baseline[key] for key in common])),
        "raw_candidate_minus_baseline": float(raw_delta.mean()),
        "mean_improvement": mean_improvement,
        "median_paired_improvement": float(np.median(improvement)),
        "ci95_improvement_low": float(interval[0]),
        "ci95_improvement_high": float(interval[1]),
        "probability_improvement_positive": float(np.mean(samples > 0.0)),
        "missing_candidate_pairs": [
            {"case_id": case_id, "seed": item_seed} for case_id, item_seed in missing_candidate
        ],
        "missing_baseline_pairs": [
            {"case_id": case_id, "seed": item_seed} for case_id, item_seed in missing_baseline
        ],
        "passes_positive_cluster_interval": bool(interval[0] > 0.0),
        "definition": (
            "case-cluster bootstrap over within-case paired-seed mean effects; "
            "positive improvement favors the candidate"
        ),
    }
