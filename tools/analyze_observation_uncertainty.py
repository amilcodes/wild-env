#!/usr/bin/env python3
"""Sensitivity analysis for historical perimeter localization uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import PerimeterSeries
from aeolus.evaluation.observation import uncertainty_aware_perimeter_metrics


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def analyze(
    results_path: Path,
    examples_path: Path,
    prepared_root: Path,
    *,
    sigma_m: list[float],
) -> dict[str, Any]:
    study = json.loads(results_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    with np.load(examples_path, allow_pickle=False) as examples:
        for specification in study["manifest"]["incidents"]:
            code = str(specification["incident_code"])
            incident = IncidentBundle.load(prepared_root / _slug(code))
            series = PerimeterSeries.from_incident(incident)
            sigmas = [
                ("hard", 0.0),
                ("one_cell", series.cell_size_m),
                ("two_cells", 2.0 * series.cell_size_m),
                *[(f"declared_{float(value):g}_m", float(value)) for value in sigma_m],
            ]
            for start_index, target_index in specification["validation_pairs"]:
                probability = examples[
                    f"{_key(code)}_{start_index}_{target_index}_history_ensemble_probability"
                ]
                target = series.frames[int(target_index)]
                for label, sigma in sigmas:
                    metrics = uncertainty_aware_perimeter_metrics(
                        probability,
                        target.mask,
                        sigma_m=sigma,
                        cell_size_m=series.cell_size_m,
                    )
                    records.append(
                        {
                            "incident_code": code,
                            "start_index": int(start_index),
                            "target_index": int(target_index),
                            "cell_size_m": series.cell_size_m,
                            "observation_sigma_label": label,
                            **metrics,
                        }
                    )
    summary: list[dict[str, Any]] = []
    labels = list(dict.fromkeys(item["observation_sigma_label"] for item in records))
    for label in labels:
        selected = [item for item in records if item["observation_sigma_label"] == label]
        for metric in (
            "soft_iou",
            "soft_brier_score",
            "soft_cross_entropy",
            "predicted_boundary_envelope_coverage",
        ):
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            incident_means = [
                np.mean([item[metric] for item in selected if item["incident_code"] == incident])
                for incident in sorted({item["incident_code"] for item in selected})
            ]
            summary.append(
                {
                    "observation_sigma_label": label,
                    "observation_sigma_m_mean": float(
                        np.mean([item["observation_sigma_m"] for item in selected])
                    ),
                    "metric": metric,
                    "transition_count": len(values),
                    "incident_count": len(incident_means),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "incident_mean_standard_deviation": float(np.std(incident_means)),
                }
            )
    return {
        "schema_version": 1,
        "purpose": "declared perimeter-localization sensitivity analysis",
        "records": records,
        "summary": summary,
        "interpretation_constraints": [
            (
                "Sigma values are sensitivity parameters; the NIROPS release "
                "does not provide incident-specific localization variances."
            ),
            (
                "The soft target models isotropic Gaussian boundary displacement "
                "and does not represent interpretation or acquisition-time error."
            ),
            "Hard-mask metrics remain the primary directly observed scores.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("examples", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma-m", type=float, action="append")
    args = parser.parse_args()
    analysis = analyze(
        args.results,
        args.examples,
        args.prepared_root,
        sigma_m=args.sigma_m or [350.0],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["records"][0]))
        writer.writeheader()
        writer.writerows(analysis["records"])
    print(json.dumps(analysis["summary"], indent=2))


if __name__ == "__main__":
    main()
