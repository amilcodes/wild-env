#!/usr/bin/env python3
"""Run a paired, fixed-parameter historical-fuel sensitivity study."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import (
    HindcastJob,
    PerimeterSeries,
    execute_hindcast_job,
)
from aeolus.evaluation.study import load_manifest
from aeolus.policies import no_aerial_action
from aeolus.workflows import scenario_from_incident

METHOD_ADJUSTMENTS = {
    "raw_physics": None,
    "fixed_calibrated_physics": "reference",
}
METRICS = {
    "perimeter_iou": ("metrics.iou", 1.0),
    "growth_tolerance_f1": ("growth_tolerance_1_cell.f1", 1.0),
    "boundary_mean_distance_m": (
        "boundary.mean_symmetric_distance_m",
        -1.0,
    ),
    "symmetric_difference_km2": (
        "metrics.symmetric_difference_km2",
        -1.0,
    ),
}


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _nested(value: dict[str, Any], path: str) -> float:
    current: Any = value
    for component in path.split("."):
        current = current[component]
    return float(current)


def _paired_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    def keyed(records):
        return {
            (
                record["incident_code"],
                record["method"],
                record["start_index"],
                record["target_index"],
            ): record
            for record in records
        }

    old = keyed(before["forecasts"])
    new = keyed(after["forecasts"])
    rng = np.random.default_rng(seed)
    methods: dict[str, Any] = {}
    for method in METHOD_ADJUSTMENTS:
        keys = sorted(key for key in old.keys() & new.keys() if key[1] == method)
        metrics: dict[str, Any] = {}
        for name, (path, direction) in METRICS.items():
            old_values = np.asarray([_nested(old[key]["forecast"], path) for key in keys])
            new_values = np.asarray([_nested(new[key]["forecast"], path) for key in keys])
            delta = new_values - old_values
            bootstrap = np.mean(
                rng.choice(
                    delta,
                    size=(5000, len(delta)),
                    replace=True,
                ),
                axis=1,
            )
            metrics[name] = {
                "path": path,
                "direction": ("higher_is_better" if direction > 0 else "lower_is_better"),
                "before_mean": float(np.mean(old_values)),
                "after_mean": float(np.mean(new_values)),
                "mean_delta_after_minus_before": float(np.mean(delta)),
                "paired_improvement_fraction": float(np.mean(direction * delta > 0.0)),
                "paired_delta_ci95": np.quantile(
                    bootstrap,
                    (0.025, 0.975),
                ).tolist(),
            }
        methods[method] = {
            "paired_forecasts": len(keys),
            "metrics": metrics,
        }
    return {
        "design": (
            "The source bundle and time-admissible bundle share elevation, "
            "weather, observations, initialization, code, parameter values, "
            "and random seeds. Only fuel and canopy state change."
        ),
        "solver": (
            "adaptive_huygens computational-screening ablation; primary "
            "WENO5 hindcasts remain a cluster evaluation item"
        ),
        "methods": methods,
    }


def run(
    *,
    manifest_path: Path,
    before_root: Path,
    after_root: Path,
    reference_results: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    reference = json.loads(reference_results.read_text(encoding="utf-8"))
    adjustment = {
        item["incident_code"]: float(item["selected_spread_adjustment"]) for item in reference["calibrations"]
    }
    jobs: list[HindcastJob] = []
    metadata: list[dict[str, Any]] = []
    for incident_index, specification in enumerate(manifest["incidents"]):
        code = str(specification["incident_code"])
        roots = {
            "before": before_root / _slug(code),
            "after": after_root / _slug(code),
        }
        incidents = {name: IncidentBundle.load(root) for name, root in roots.items()}
        series = {name: PerimeterSeries.from_incident(incident) for name, incident in incidents.items()}
        pairs = [(int(pair[0]), int(pair[1])) for pair in specification["validation_pairs"]]
        max_minutes = max(
            round(
                (
                    series["before"].frames[target].timestamp - series["before"].frames[start].timestamp
                ).total_seconds()
                / 60.0
            )
            for start, target in pairs
        )
        for corpus_name in ("before", "after"):
            timestamps = [frame.timestamp for frame in series[corpus_name].frames]
            if timestamps != [frame.timestamp for frame in series["before"].frames]:
                raise ValueError(f"{code} perimeter timestamps differ")
            base = scenario_from_incident(
                incidents[corpus_name],
                seed=int(manifest["seed"]) + incident_index * 7919,
                horizon_min=max_minutes + 3,
                spotting_rate=float(manifest["spotting_rate"]),
            )
            for method, selected in METHOD_ADJUSTMENTS.items():
                spread = 1.0 if selected is None else adjustment[code]
                config = replace(
                    base,
                    fire=replace(
                        base.fire,
                        front_solver="adaptive_huygens",
                        surface_spread_adjustment=spread,
                        crown_spread_adjustment=spread,
                    ),
                    terminate_on_escape=False,
                )
                for start_index, target_index in pairs:
                    jobs.append(
                        HindcastJob(
                            config=config,
                            series=series[corpus_name],
                            policy=no_aerial_action,
                            start_index=start_index,
                            target_index=target_index,
                        )
                    )
                    metadata.append(
                        {
                            "corpus": corpus_name,
                            "incident_code": code,
                            "method": method,
                            "start_index": start_index,
                            "target_index": target_index,
                            "spread_adjustment": spread,
                        }
                    )

    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        forecasts = list(executor.map(execute_hindcast_job, jobs))
    records = [
        {
            **record,
            "forecast": forecast,
        }
        for record, forecast in zip(metadata, forecasts, strict=True)
    ]
    corpora = {
        name: {
            "schema_version": 1,
            "study": "fixed-parameter historical fuel sensitivity",
            "front_solver": "adaptive_huygens",
            "calibration_semantics": ("fixed reference values; no parameter refit after fuel change"),
            "manifest": manifest,
            "forecasts": [
                {key: value for key, value in record.items() if key != "corpus"}
                for record in records
                if record["corpus"] == name
            ],
        }
        for name in ("before", "after")
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in corpora.items():
        (output / f"{name}_results.json").write_text(
            json.dumps(value, indent=2),
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "study": "paired fixed-parameter historical fuel sensitivity",
        "before_root": str(before_root.resolve()),
        "after_root": str(after_root.resolve()),
        "reference_calibration": str(reference_results.resolve()),
        "forecast_count_per_corpus": len(corpora["before"]["forecasts"]),
        "paired_comparison": _paired_summary(
            corpora["before"],
            corpora["after"],
            seed=int(manifest["seed"]),
        ),
        "artifacts": {
            "before": str((output / "before_results.json").resolve()),
            "after": str((output / "after_results.json").resolve()),
        },
        "interpretation_constraints": [
            "The legacy adaptive-Huygens solver is used for computational screening.",
            "Fixed reference calibration values isolate fuel/canopy sensitivity.",
            "This result does not replace the WENO5 posterior-ensemble benchmark.",
            "Observed growth still includes unobserved historical suppression.",
        ],
    }
    (output / "paired_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/historical_validation.yaml"),
    )
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("results/frontier_fire/historical_validation_v3/historical_validation_results.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        before_root=args.before_root,
        after_root=args.after_root,
        reference_results=args.reference_results,
        output=args.out,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
