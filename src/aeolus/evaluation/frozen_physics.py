"""Canonical WENO5 evaluation under a frozen incident-holdout contract."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from aeolus.data import IncidentBundle
from aeolus.evaluation.frozen_benchmark import (
    _incident_weighted_mean,
    _paired_improvement,
    _pairs,
    _score_prediction,
    _slug,
    _summary,
    audit_frozen_contract,
    load_frozen_contract,
)
from aeolus.evaluation.historical import (
    HindcastJob,
    PerimeterSeries,
    execute_hindcast_job,
)
from aeolus.policies import no_aerial_action
from aeolus.workflows import scenario_from_incident

_CACHE_SCHEMA_VERSION = 2


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"episode", "prediction_mask", "arrival_time_min"}
    }


def _execute_cached_jobs(
    jobs: list[HindcastJob],
    keys: list[tuple[Any, ...]],
    *,
    cache_directory: Path,
    executor: ProcessPoolExecutor,
    phase: str,
) -> list[dict[str, Any]]:
    """Execute independent hindcasts with atomic, resumable local checkpoints."""

    import hashlib

    if len(jobs) != len(keys):
        raise ValueError("cached hindcast jobs and keys must have equal length")
    phase_root = cache_directory / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending = {}
    for index, (job, key) in enumerate(zip(jobs, keys, strict=True)):
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        path = phase_root / f"{digest}.pickle"
        if path.exists():
            with path.open("rb") as stream:
                payload = pickle.load(stream)  # noqa: S301 - trusted local cache only
            if payload.get("key") != key:
                raise RuntimeError(f"hindcast cache key collision: {path}")
            results[index] = payload["result"]
            continue
        pending[executor.submit(execute_hindcast_job, job)] = (index, key, path)
    completed = len(jobs) - len(pending)
    if completed:
        print(
            f"[frozen-physics] resumed {completed}/{len(jobs)} {phase} hindcasts",
            file=sys.stderr,
            flush=True,
        )
    for future in as_completed(pending):
        index, key, path = pending[future]
        result = future.result()
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            pickle.dump({"key": key, "result": result}, stream)
        temporary.replace(path)
        results[index] = result
        completed += 1
        if completed % 8 == 0 or completed == len(jobs):
            print(
                f"[frozen-physics] {phase} {completed}/{len(jobs)} complete",
                file=sys.stderr,
                flush=True,
            )
    if any(result is None for result in results):
        raise RuntimeError("one or more cached hindcasts did not complete")
    return [result for result in results if result is not None]


def _config(
    bundle: IncidentBundle,
    series: PerimeterSeries,
    pairs: list[tuple[int, int]],
    *,
    seed: int,
    spread_adjustment: float,
    weather_forcing: Path | None = None,
):
    max_minutes = max(
        round((series.frames[target].timestamp - series.frames[start].timestamp).total_seconds() / 60.0)
        for start, target in pairs
    )
    base = scenario_from_incident(
        bundle,
        seed=seed,
        horizon_min=max_minutes + 3,
        max_tasks=16,
        spotting_rate=0.0,
    )
    return replace(
        base,
        weather_forcing=(str(weather_forcing.resolve()) if weather_forcing is not None else None),
        terminate_on_escape=False,
        fire=replace(
            base.fire,
            surface_spread_adjustment=float(spread_adjustment),
            crown_spread_adjustment=float(spread_adjustment),
            enable_spotting=False,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_operational_forcing_index(
    root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[tuple[str, int, int], tuple[Path, str]], dict[str, Any]]:
    manifest_path = root / "operational_forcing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_base_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("operational forcing corpus belongs to a different contract")
    if not manifest.get("complete") or not manifest.get("all_operationally_available"):
        raise ValueError("operational forcing corpus is incomplete or failed audit")
    index: dict[tuple[str, int, int], tuple[Path, str]] = {}
    for record in manifest["records"]:
        key = (
            str(record["incident_code"]),
            int(record["start_index"]),
            int(record["target_index"]),
        )
        path = Path(record["path"])
        expected = str(record["sha256"])
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"operational forcing artifact failed digest audit: {path}")
        if key in index:
            raise ValueError(f"duplicate operational forcing transition: {key}")
        index[key] = (path, expected)
    return index, manifest


def _forcing_for_transition(
    bundle: IncidentBundle,
    code: str,
    start_index: int,
    target_index: int,
    operational_index: dict[tuple[str, int, int], tuple[Path, str]] | None,
) -> tuple[Path | None, str]:
    if operational_index is not None:
        key = (code, start_index, target_index)
        if key not in operational_index:
            raise FileNotFoundError(f"missing operational transition forcing: {key}")
        return operational_index[key]
    path = bundle.asset_path("weather", required=False)
    return (path, _sha256(path) if path is not None else "no-weather-forcing")


def _fit_global_adjustment(
    candidates: list[float],
    prepared: dict[str, tuple[dict[str, Any], IncidentBundle, PerimeterSeries]],
    train_codes: list[str],
    *,
    seed: int,
    process_executor: ProcessPoolExecutor,
    cache_directory: Path,
    operational_index: dict[tuple[str, int, int], tuple[Path, str]] | None,
) -> dict[str, Any]:
    jobs: list[HindcastJob] = []
    keys: list[tuple[float, str, int, int, str]] = []
    for incident_index, code in enumerate(train_codes):
        specification, bundle, series = prepared[code]
        pairs = [pair for pair in _pairs(specification, "train") if pair[0] >= 1]
        for candidate in candidates:
            for start_index, target_index in pairs:
                forcing_path, forcing_digest = _forcing_for_transition(
                    bundle,
                    code,
                    start_index,
                    target_index,
                    operational_index,
                )
                config = _config(
                    bundle,
                    series,
                    [pair for pair in pairs if pair == (start_index, target_index)],
                    seed=seed + incident_index * 7919,
                    spread_adjustment=candidate,
                    weather_forcing=forcing_path,
                )
                jobs.append(
                    HindcastJob(
                        config=config,
                        series=series,
                        policy=no_aerial_action,
                        start_index=start_index,
                        target_index=target_index,
                        use_arrival_history=True,
                    )
                )
                keys.append(
                    (
                        candidate,
                        code,
                        start_index,
                        target_index,
                        forcing_digest,
                    )
                )
    print(
        f"[frozen-physics] fitting {len(candidates)} adjustments on {len(jobs)} train hindcasts",
        file=sys.stderr,
        flush=True,
    )
    cache_keys = [
        (
            _CACHE_SCHEMA_VERSION,
            "train",
            candidate,
            code,
            start_index,
            target_index,
            True,
            forcing_digest,
        )
        for candidate, code, start_index, target_index, forcing_digest in keys
    ]
    results = _execute_cached_jobs(
        jobs,
        cache_keys,
        cache_directory=cache_directory,
        executor=process_executor,
        phase="fit",
    )
    records = [
        {
            "candidate": candidate,
            "incident_code": code,
            "start_index": start_index,
            "target_index": target_index,
            "forecast": _compact(result),
        }
        for (candidate, code, start_index, target_index, _forcing_digest), result in zip(
            keys,
            results,
            strict=True,
        )
    ]
    trials = []
    for candidate in candidates:
        selected = [record for record in records if record["candidate"] == candidate]
        active = [
            record for record in selected if record["forecast"]["growth_metrics"]["observed_area_km2"] > 0.0
        ]
        trials.append(
            {
                "spread_adjustment": candidate,
                "train_incident_weighted_cumulative_iou": _incident_weighted_mean(
                    selected,
                    "metrics.iou",
                ),
                "train_incident_weighted_advancing_front_f1": _incident_weighted_mean(
                    active,
                    "growth_tolerance_1_cell.f1",
                ),
                "transitions": len(selected),
            }
        )
    extent = max(
        trials,
        key=lambda item: (
            item["train_incident_weighted_cumulative_iou"],
            -item["spread_adjustment"],
        ),
    )
    front = max(
        trials,
        key=lambda item: (
            item["train_incident_weighted_advancing_front_f1"],
            -item["spread_adjustment"],
        ),
    )
    return {
        "parameter": "surface_and_crown_spread_adjustment",
        "initialization": "causal two-perimeter arrival history",
        "selection_unit": "incident-weighted train transitions",
        "extent_selected_adjustment": float(extent["spread_adjustment"]),
        "front_selected_adjustment": float(front["spread_adjustment"]),
        "trials": trials,
    }


def run_frozen_physics_benchmark(
    contract_path: str | Path,
    prepared_root: str | Path,
    *,
    parallel_workers: int = 8,
    cache_directory: str | Path | None = None,
    operational_forcing_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = load_frozen_contract(contract_path)
    audit = audit_frozen_contract(contract)
    if not audit["valid"]:
        raise ValueError(f"frozen historical contract failed audit: {audit}")
    base = contract["resolved_base_manifest"]
    root = Path(prepared_root)
    prepared: dict[str, tuple[dict[str, Any], IncidentBundle, PerimeterSeries]] = {}
    for specification in base["incidents"]:
        code = str(specification["incident_code"])
        path = root / _slug(code)
        if (path / "item.json").exists():
            bundle = IncidentBundle.load(path)
            prepared[code] = (
                specification,
                bundle,
                PerimeterSeries.from_incident(bundle),
            )
    required_codes = [
        code for split in ("train", "development", "test") for code in audit["assignments"][split]
    ]
    missing = sorted(set(required_codes) - prepared.keys())
    if missing:
        raise FileNotFoundError(f"frozen physics corpus is incomplete: {missing}")
    candidates = [float(value) for value in contract["physics_candidates"]]
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("physics candidates must be positive")
    cache_root = (
        Path(cache_directory) if cache_directory is not None else root / ".aeolus-frozen-physics-cache"
    )
    operational_index: dict[tuple[str, int, int], tuple[Path, str]] | None = None
    operational_manifest: dict[str, Any] | None = None
    if operational_forcing_root is not None:
        operational_index, operational_manifest = _load_operational_forcing_index(
            Path(operational_forcing_root),
            expected_manifest_sha256=str(audit["base_manifest_sha256"]),
        )

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=parallel_workers,
        mp_context=context,
    ) as executor:
        fit = _fit_global_adjustment(
            candidates,
            prepared,
            audit["assignments"]["train"],
            seed=int(base["seed"]),
            process_executor=executor,
            cache_directory=cache_root,
            operational_index=operational_index,
        )
        requested_models = (
            ("history_raw_physics", 1.0, True),
            (
                "history_global_extent",
                float(fit["extent_selected_adjustment"]),
                True,
            ),
            (
                "history_global_front",
                float(fit["front_selected_adjustment"]),
                True,
            ),
        )
        models_list: list[tuple[str, float, bool]] = []
        aliases: dict[str, str] = {}
        signatures: dict[tuple[float, bool], str] = {}
        for method, adjustment, use_history in requested_models:
            signature = (adjustment, use_history)
            if signature in signatures:
                aliases[method] = signatures[signature]
            else:
                signatures[signature] = method
                models_list.append((method, adjustment, use_history))
        models = tuple(models_list)
        jobs: list[HindcastJob] = []
        keys: list[tuple[str, str, str, int, int, float, bool, str]] = []
        records: list[dict[str, Any]] = []
        for split in ("development", "test"):
            for incident_index, code in enumerate(audit["assignments"][split]):
                specification, bundle, series = prepared[code]
                pairs = _pairs(specification, split)
                for start_index, target_index in pairs:
                    records.append(
                        {
                            "incident_code": code,
                            "split": split,
                            "start_index": start_index,
                            "target_index": target_index,
                            "method": "persistence",
                            "spread_adjustment": 0.0,
                            "forecast": _score_prediction(
                                series.frames[start_index].mask,
                                series,
                                start_index,
                                target_index,
                            ),
                        }
                    )
                    for method, adjustment, use_history in models:
                        forcing_path, forcing_digest = _forcing_for_transition(
                            bundle,
                            code,
                            start_index,
                            target_index,
                            operational_index,
                        )
                        config = _config(
                            bundle,
                            series,
                            [(start_index, target_index)],
                            seed=int(base["seed"]) + incident_index * 7919,
                            spread_adjustment=adjustment,
                            weather_forcing=forcing_path,
                        )
                        jobs.append(
                            HindcastJob(
                                config=config,
                                series=series,
                                policy=no_aerial_action,
                                start_index=start_index,
                                target_index=target_index,
                                use_arrival_history=use_history,
                            )
                        )
                        keys.append(
                            (
                                split,
                                code,
                                method,
                                start_index,
                                target_index,
                                adjustment,
                                use_history,
                                forcing_digest,
                            )
                        )
        print(
            f"[frozen-physics] evaluating {len(jobs)} development/test hindcasts",
            file=sys.stderr,
            flush=True,
        )
        cache_keys = [
            (
                _CACHE_SCHEMA_VERSION,
                "evaluation",
                split,
                code,
                method,
                start_index,
                target_index,
                adjustment,
                use_history,
                forcing_digest,
            )
            for (
                split,
                code,
                method,
                start_index,
                target_index,
                adjustment,
                use_history,
                forcing_digest,
            ) in keys
        ]
        results = _execute_cached_jobs(
            jobs,
            cache_keys,
            cache_directory=cache_root,
            executor=executor,
            phase="evaluation",
        )

    records.extend(
        {
            "incident_code": code,
            "split": split,
            "start_index": start_index,
            "target_index": target_index,
            "method": method,
            "spread_adjustment": adjustment,
            "forecast": _compact(result),
        }
        for (
            split,
            code,
            method,
            start_index,
            target_index,
            adjustment,
            _use_history,
            _forcing_digest,
        ), result in zip(keys, results, strict=True)
    )
    for alias, source in aliases.items():
        records.extend({**record, "method": alias} for record in list(records) if record["method"] == source)
    metrics = (
        "metrics.iou",
        "metrics.symmetric_difference_km2",
        "boundary.mean_symmetric_distance_m",
        "growth_tolerance_1_cell.f1",
    )
    methods = ("persistence", *(item[0] for item in requested_models))
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
            for method in methods
        }
        for split in ("development", "test")
    }
    test_records = [record for record in records if record["split"] == "test"]
    improvements = {
        method: {
            "cumulative_iou": _paired_improvement(
                test_records,
                method,
                "metrics.iou",
                higher_is_better=True,
                seed=int(base["seed"]) + index * 103,
            ),
            "boundary_distance_m": _paired_improvement(
                test_records,
                method,
                "boundary.mean_symmetric_distance_m",
                higher_is_better=False,
                seed=int(base["seed"]) + index * 103 + 1,
            ),
        }
        for index, method in enumerate(methods)
        if method != "persistence"
    }
    return {
        "schema_version": 1,
        "study": "frozen chronological incident-holdout canonical physics benchmark",
        "contract_path": str(Path(contract_path).resolve()),
        "prepared_root": str(root.resolve()),
        "cache_directory": str(cache_root.resolve()),
        "forcing_mode": (
            "archived_operational_forecast"
            if operational_index is not None
            else "retrospective_analysis_or_reanalysis"
        ),
        "operational_forcing_manifest": operational_manifest,
        "contract_audit": audit,
        "fit": fit,
        "records": records,
        "summaries": summaries,
        "test_improvement_against_persistence": improvements,
        "interpretation_constraints": [
            "All spread adjustments are selected only on train incidents.",
            "Development and test incidents receive no incident-specific spread calibration.",
            (
                "Archived forecasts use one cycle available before each issue "
                "time; operational availability is assumed from the declared lag."
                if operational_index is not None
                else "Verifying analysis/reanalysis forcing is a retrospective "
                "upper-bound input, not an operational forecast."
            ),
            (
                "Public NIROPS perimeters include unobserved suppression effects "
                "that the no-action hindcasts cannot identify."
            ),
            (
                "Spotting is disabled because the current historical observation "
                "set does not identify spotting parameters."
            ),
        ],
    }
