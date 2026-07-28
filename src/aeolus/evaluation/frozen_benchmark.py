"""Chronological incident-holdout benchmark with globally fitted baselines."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeolus.data import IncidentBundle, WeatherForcing
from aeolus.evaluation.baselines import (
    isotropic_spread_prediction,
    recent_equivalent_radius_rate_m_min,
    wind_ellipse_prediction,
)
from aeolus.evaluation.historical import (
    PerimeterSeries,
    boundary_distance_metrics,
    perimeter_metrics,
    tolerance_metrics,
)
from aeolus.evaluation.validity import (
    assess_forcing_availability,
    assess_historical_fuel_provenance,
    assess_metric_crs,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("frozen historical contract requires schema_version 1")
    base_path = (contract_path.parent / str(contract["base_manifest"])).resolve()
    expected = str(contract["base_manifest_sha256"])
    actual = _sha256(base_path)
    if actual != expected:
        raise ValueError(f"base manifest digest changed: expected {expected}, received {actual}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return {**contract, "resolved_base_manifest": base, "base_manifest_path": str(base_path)}


def _partition(contract: dict[str, Any], specification: dict[str, Any]) -> str:
    code = str(specification["incident_code"])
    explicit = contract.get("incident_assignments", {})
    for split in ("train", "development", "test"):
        if code in explicit.get(split, []):
            return split
    year = int(specification.get("stratum", {}).get("year", -1))
    for split, years in contract.get("year_partition", {}).items():
        if year in [int(value) for value in years]:
            return str(split)
    raise ValueError(f"incident {code} has no frozen partition assignment")


def audit_frozen_contract(contract: dict[str, Any]) -> dict[str, Any]:
    base = contract["resolved_base_manifest"]
    assignments: dict[str, list[str]] = defaultdict(list)
    duplicates: list[str] = []
    seen: set[str] = set()
    for specification in base["incidents"]:
        code = str(specification["incident_code"])
        if code in seen:
            duplicates.append(code)
        seen.add(code)
        assignments[_partition(contract, specification)].append(code)
    empty = [split for split in ("train", "development", "test") if not assignments.get(split)]
    assignment_overlap = sorted(
        code
        for code in seen
        if sum(
            code in contract.get("incident_assignments", {}).get(split, [])
            for split in ("train", "development", "test")
        )
        > 1
    )
    forbidden_test_tuning = []
    for specification in base["incidents"]:
        if _partition(contract, specification) == "test" and specification.get(
            "additional_spread_candidates"
        ):
            forbidden_test_tuning.append(str(specification["incident_code"]))
    ignores_nontrain_candidates = bool(
        contract.get("ignore_incident_specific_candidates_outside_train", False)
    )
    valid = (
        not duplicates
        and not empty
        and not assignment_overlap
        and (not forbidden_test_tuning or ignores_nontrain_candidates)
    )
    return {
        "valid": valid,
        "base_manifest_sha256": contract["base_manifest_sha256"],
        "partition_unit": "incident",
        "assignments": dict(assignments),
        "counts": {key: len(value) for key, value in assignments.items()},
        "duplicate_incidents": sorted(set(duplicates)),
        "empty_splits": empty,
        "assignment_overlap": assignment_overlap,
        "forbidden_test_incident_specific_candidates": forbidden_test_tuning,
        "test_incident_specific_candidates_ignored": ignores_nontrain_candidates,
        "test_targets_used_for_fitting": False,
    }


def _pairs(specification: dict[str, Any], split: str) -> list[tuple[int, int]]:
    pairs = [tuple(int(value) for value in pair) for pair in specification["validation_pairs"]]
    if split == "train" and "calibration_pair" in specification:
        calibration = tuple(int(value) for value in specification["calibration_pair"])
        if calibration not in pairs:
            pairs = [calibration, *pairs]
    return pairs


def _issue_wind_direction(
    bundle: IncidentBundle,
    timestamp: datetime,
) -> tuple[float, dict[str, Any]]:
    weather_path = bundle.asset_path("weather", required=False)
    if weather_path is None:
        return 0.0, {"forcing_class": "missing", "supports_operational_forecast": False}
    weather = WeatherForcing.load(weather_path)
    origin = weather.time_origin
    if origin is None:
        return 0.0, {"forcing_class": "unknown", "supports_operational_forecast": False}
    minute = (timestamp - origin).total_seconds() / 60.0
    sample = weather.at_minute(minute)
    direction = np.asarray(sample["wind_direction_deg"], dtype=np.float64)
    radians = np.deg2rad(direction)
    mean = float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360.0)
    return mean, dict(weather.metadata)


def _predict(
    method: str,
    parameter: float,
    series: PerimeterSeries,
    specification: dict[str, Any],
    bundle: IncidentBundle,
    start_index: int,
    target_index: int,
) -> np.ndarray:
    start = series.frames[start_index]
    target = series.frames[target_index]
    duration = (target.timestamp - start.timestamp).total_seconds() / 60.0
    burnable = ~bundle.scenario_bundle().barrier
    if method in {"persistence", "global_isotropic"}:
        rate = 0.0 if method == "persistence" else parameter
        return isotropic_spread_prediction(
            start.mask,
            rate_m_min=rate,
            duration_min=duration,
            cell_size_m=series.cell_size_m,
            burnable_mask=burnable,
        )
    if method == "last_growth_trend":
        if start_index < 1:
            return start.mask.copy()
        previous = series.frames[start_index - 1]
        elapsed = (start.timestamp - previous.timestamp).total_seconds() / 60.0
        rate = recent_equivalent_radius_rate_m_min(
            previous.mask,
            start.mask,
            elapsed_min=elapsed,
            cell_size_m=series.cell_size_m,
        )
        return isotropic_spread_prediction(
            start.mask,
            rate_m_min=parameter * rate,
            duration_min=duration,
            cell_size_m=series.cell_size_m,
            burnable_mask=burnable,
        )
    if method == "wind_ellipse":
        direction, _ = _issue_wind_direction(bundle, start.timestamp)
        return wind_ellipse_prediction(
            start.mask,
            head_rate_m_min=parameter,
            duration_min=duration,
            cell_size_m=series.cell_size_m,
            wind_from_direction_deg=direction,
            flank_ratio=float(specification.get("flank_ratio", 0.35)),
            backing_ratio=float(specification.get("backing_ratio", 0.15)),
            burnable_mask=burnable,
        )
    raise KeyError(f"unknown baseline method: {method}")


def _score_prediction(
    predicted: np.ndarray,
    series: PerimeterSeries,
    start_index: int,
    target_index: int,
) -> dict[str, Any]:
    start = series.frames[start_index].mask
    target = series.frames[target_index].mask
    observed_growth = target & ~start
    predicted_growth = predicted & ~start
    return {
        "metrics": perimeter_metrics(predicted, target, series.cell_size_m),
        "growth_metrics": perimeter_metrics(
            predicted_growth,
            observed_growth,
            series.cell_size_m,
        ),
        "perimeter_tolerance_1_cell": tolerance_metrics(
            predicted,
            target,
            radius_cells=1,
        ),
        "growth_tolerance_1_cell": tolerance_metrics(
            predicted_growth,
            observed_growth,
            radius_cells=1,
        ),
        "boundary": boundary_distance_metrics(
            predicted,
            target,
            series.cell_size_m,
        ),
    }


def _metric(record: dict[str, Any], path: str) -> float:
    value: Any = record["forecast"]
    for key in path.split("."):
        value = value[key]
    return float(value)


def _incident_weighted_mean(records: list[dict[str, Any]], metric: str) -> float:
    incidents = sorted({str(record["incident_code"]) for record in records})
    return float(
        np.mean(
            [
                np.mean(
                    [_metric(record, metric) for record in records if record["incident_code"] == incident]
                )
                for incident in incidents
            ]
        )
    )


def _fit_parameter(
    method: str,
    candidates: list[float],
    training_cases: list[tuple[dict[str, Any], IncidentBundle, PerimeterSeries, int, int]],
    *,
    selection_metric: str,
    observed_growth_only: bool,
) -> dict[str, Any]:
    trials = []
    for candidate in candidates:
        records = []
        for specification, bundle, series, start_index, target_index in training_cases:
            if method == "last_growth_trend" and start_index < 1:
                continue
            predicted = _predict(
                method,
                candidate,
                series,
                specification,
                bundle,
                start_index,
                target_index,
            )
            scored = _score_prediction(
                predicted,
                series,
                start_index,
                target_index,
            )
            if observed_growth_only and scored["growth_metrics"]["observed_area_km2"] <= 0.0:
                continue
            records.append(
                {
                    "incident_code": specification["incident_code"],
                    "forecast": scored,
                }
            )
        objective = _incident_weighted_mean(records, selection_metric)
        trials.append(
            {
                "parameter": float(candidate),
                "train_incident_weighted_objective": objective,
                "training_transitions": len(records),
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["train_incident_weighted_objective"],
            -item["parameter"],
        ),
    )
    return {
        "method": method,
        "selection_metric": selection_metric,
        "observed_growth_only": observed_growth_only,
        "incident_weighted": True,
        "selected_parameter": selected["parameter"],
        "trials": trials,
    }


def _summary(records: list[dict[str, Any]], method: str, metric: str) -> dict[str, Any]:
    selected = [record for record in records if record["method"] == method]
    values = np.asarray([_metric(record, metric) for record in selected])
    return {
        "transitions": len(selected),
        "incidents": len({record["incident_code"] for record in selected}),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "incident_weighted_mean": _incident_weighted_mean(selected, metric),
    }


def _paired_improvement(
    records: list[dict[str, Any]],
    method: str,
    metric: str,
    *,
    higher_is_better: bool,
    seed: int,
    samples: int = 5000,
) -> dict[str, Any]:
    keyed = {
        (record["incident_code"], record["start_index"], record["target_index"], record["method"]): record
        for record in records
    }
    incidents = sorted({record["incident_code"] for record in records})
    by_incident: dict[str, float] = {}
    for incident in incidents:
        keys = sorted(
            (start, target)
            for code, start, target, candidate in keyed
            if code == incident and candidate == method
        )
        deltas = []
        for start, target in keys:
            candidate = _metric(keyed[(incident, start, target, method)], metric)
            persistence = _metric(
                keyed[(incident, start, target, "persistence")],
                metric,
            )
            delta = candidate - persistence
            deltas.append(delta if higher_is_better else -delta)
        by_incident[incident] = float(np.mean(deltas))
    incident_values = np.asarray(list(by_incident.values()))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean(rng.choice(incident_values, size=len(incident_values), replace=True))
            for _ in range(samples)
        ]
    )
    return {
        "method": method,
        "reference": "persistence",
        "metric": metric,
        "positive_favors_candidate": True,
        "incident_mean_improvements": by_incident,
        "mean_improvement": float(incident_values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_improvement_positive": float(np.mean(draws > 0.0)),
        "passes_positive_incident_cluster_interval": bool(np.quantile(draws, 0.025) > 0.0),
    }


def run_frozen_baseline_benchmark(
    contract_path: str | Path,
    prepared_root: str | Path,
) -> dict[str, Any]:
    contract = load_frozen_contract(contract_path)
    audit = audit_frozen_contract(contract)
    if not audit["valid"]:
        raise ValueError(f"frozen historical contract failed audit: {audit}")
    base = contract["resolved_base_manifest"]
    root = Path(prepared_root)
    prepared: dict[str, tuple[dict[str, Any], IncidentBundle, PerimeterSeries]] = {}
    realism: list[dict[str, Any]] = []
    for specification in base["incidents"]:
        code = str(specification["incident_code"])
        bundle_path = root / _slug(code)
        if not (bundle_path / "item.json").exists():
            continue
        bundle = IncidentBundle.load(bundle_path)
        series = PerimeterSeries.from_incident(bundle)
        prepared[code] = (specification, bundle, series)
        weather_path = bundle.asset_path("weather", required=False)
        forcing = WeatherForcing.load(weather_path) if weather_path is not None else None
        forcing_audits = []
        if forcing is not None:
            for start_index, target_index in _pairs(
                specification,
                _partition(contract, specification),
            ):
                forcing_audits.append(
                    assess_forcing_availability(
                        forcing,
                        forecast_start=series.frames[start_index].timestamp,
                        forecast_end=series.frames[target_index].timestamp,
                    )
                )
        realism.append(
            {
                "incident_code": code,
                "split": _partition(contract, specification),
                "metric_crs": assess_metric_crs(bundle.scenario_bundle()),
                "historical_fuels": assess_historical_fuel_provenance(
                    bundle.scenario_bundle(),
                    incident_start=bundle.item["properties"]["start_datetime"],
                ).as_dict(),
                "forcing_audits": forcing_audits,
            }
        )
    missing = [code for codes in audit["assignments"].values() for code in codes if code not in prepared]

    training_cases = []
    for code in audit["assignments"]["train"]:
        if code not in prepared:
            continue
        specification, bundle, series = prepared[code]
        for start_index, target_index in _pairs(specification, "train"):
            training_cases.append((specification, bundle, series, start_index, target_index))
    baseline = contract["baselines"]
    fits = {
        method: {
            "extent": _fit_parameter(
                method,
                [float(value) for value in baseline[method]["candidates"]],
                training_cases,
                selection_metric="metrics.iou",
                observed_growth_only=False,
            ),
            "front": _fit_parameter(
                method,
                [float(value) for value in baseline[method]["candidates"]],
                training_cases,
                selection_metric="growth_tolerance_1_cell.f1",
                observed_growth_only=True,
            ),
        }
        for method in ("global_isotropic", "last_growth_trend", "wind_ellipse")
    }
    models = [
        {"label": "persistence", "family": "persistence", "parameter": 0.0},
        *[
            {
                "label": f"{method}_{objective}",
                "family": method,
                "parameter": float(result["selected_parameter"]),
            }
            for method, objectives in fits.items()
            for objective, result in objectives.items()
        ],
    ]
    records: list[dict[str, Any]] = []
    for split in ("development", "test"):
        for code in audit["assignments"][split]:
            if code not in prepared:
                continue
            specification, bundle, series = prepared[code]
            for start_index, target_index in _pairs(specification, split):
                for model in models:
                    label = str(model["label"])
                    method = str(model["family"])
                    parameter = float(model["parameter"])
                    predicted = _predict(
                        method,
                        parameter,
                        series,
                        {**specification, **baseline.get(method, {})},
                        bundle,
                        start_index,
                        target_index,
                    )
                    records.append(
                        {
                            "incident_code": code,
                            "split": split,
                            "start_index": start_index,
                            "target_index": target_index,
                            "method": label,
                            "method_family": method,
                            "fitted_parameter": parameter,
                            "forecast": _score_prediction(
                                predicted,
                                series,
                                start_index,
                                target_index,
                            ),
                        }
                    )
    metrics = (
        "metrics.iou",
        "metrics.symmetric_difference_km2",
        "boundary.mean_symmetric_distance_m",
        "growth_tolerance_1_cell.f1",
    )
    summaries = {
        split: {
            method: {
                metric: _summary(
                    [record for record in records if record["split"] == split],
                    method,
                    metric,
                )
                for metric in metrics
            }
            for method in [str(model["label"]) for model in models]
        }
        for split in ("development", "test")
        if any(record["split"] == split for record in records)
    }
    test_records = [record for record in records if record["split"] == "test"]
    improvements = {
        method: {
            "cumulative_iou": _paired_improvement(
                test_records,
                method,
                "metrics.iou",
                higher_is_better=True,
                seed=int(base["seed"]) + index * 101,
            ),
            "boundary_distance_m": _paired_improvement(
                test_records,
                method,
                "boundary.mean_symmetric_distance_m",
                higher_is_better=False,
                seed=int(base["seed"]) + index * 101 + 1,
            ),
        }
        for index, method in enumerate(str(model["label"]) for model in models)
        if method != "persistence" and test_records
    }
    return {
        "schema_version": 1,
        "study": "frozen chronological incident-holdout geometric baseline benchmark",
        "contract_path": str(Path(contract_path).resolve()),
        "contract_audit": audit,
        "prepared_incidents": sorted(prepared),
        "missing_incidents": sorted(missing),
        "realism_audit": realism,
        "fit": fits,
        "records": records,
        "summaries": summaries,
        "test_improvement_against_persistence": improvements,
        "claim_gate": {
            "complete_frozen_corpus": not missing,
            "all_prepared_incidents_metric": all(
                item["metric_crs"]["supports_physical_distance_claims"] for item in realism
            ),
            "all_prepared_incidents_historical_fuels_admissible": all(
                item["historical_fuels"]["status"] == "historically_admissible_by_product_date"
                for item in realism
            ),
            "all_evaluated_forcing_operationally_available": all(
                audit_item["supports_operational_forecast"]
                for item in realism
                for audit_item in item["forcing_audits"]
            ),
            "physics_evaluated": False,
        },
    }
