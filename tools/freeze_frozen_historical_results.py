#!/usr/bin/env python3
"""Create a digest-locked claim manifest for the frozen historical pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("metric_manifest", type=Path)
    parser.add_argument("operational_forcing_manifest", type=Path)
    parser.add_argument("baseline_results", type=Path)
    parser.add_argument("retrospective_results", type=Path)
    parser.add_argument("operational_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = {
        "contract": args.contract,
        "metric_reprojection": args.metric_manifest,
        "operational_forcing": args.operational_forcing_manifest,
        "geometric_baselines": args.baseline_results,
        "retrospective_physics": args.retrospective_results,
        "operational_physics": args.operational_results,
    }
    metric = _load(args.metric_manifest)
    forcing = _load(args.operational_forcing_manifest)
    operational = _load(args.operational_results)
    test = operational["summaries"]["test"]
    persistence = test["persistence"]
    candidate = test["history_global_front"]
    improvement = operational["test_improvement_against_persistence"]["history_global_front"]
    iou_gate = improvement["cumulative_iou"]
    boundary_gate = improvement["boundary_distance_m"]
    front_f1 = candidate["growth_tolerance_1_cell.f1"]["incident_weighted_mean"]
    result = {
        "schema_version": 1,
        "study": "frozen metric-grid incident-holdout historical skill pilot",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "worktree_dirty": bool(_git(["status", "--porcelain"])),
            "python": sys.version,
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "validity": {
            "metric_crs_all_incidents": metric["gate"]["all_metric_crs"],
            "maximum_raster_area_fractional_error": metric["gate"][
                "maximum_absolute_fractional_raster_area_error"
            ],
            "operational_forcing_transition_count": forcing["transition_count"],
            "operational_forcing_complete": forcing["complete"],
            "operational_forcing_all_available_by_issue": forcing["all_operationally_available"],
            "partition_unit": operational["contract_audit"]["partition_unit"],
            "partition_counts": operational["contract_audit"]["counts"],
            "test_target_used_for_fitting": operational["contract_audit"]["test_targets_used_for_fitting"],
            "test_incident_count": test["persistence"]["metrics.iou"]["incidents"],
            "test_transition_count": test["persistence"]["metrics.iou"]["transitions"],
        },
        "test_result": {
            "persistence": {
                "cumulative_iou": persistence["metrics.iou"]["incident_weighted_mean"],
                "boundary_distance_m": persistence["boundary.mean_symmetric_distance_m"][
                    "incident_weighted_mean"
                ],
                "advancing_front_f1": persistence["growth_tolerance_1_cell.f1"]["incident_weighted_mean"],
            },
            "operational_front_selected": {
                "spread_adjustment": operational["fit"]["front_selected_adjustment"],
                "cumulative_iou": candidate["metrics.iou"]["incident_weighted_mean"],
                "boundary_distance_m": candidate["boundary.mean_symmetric_distance_m"][
                    "incident_weighted_mean"
                ],
                "advancing_front_f1": front_f1,
            },
        },
        "claim_gate": {
            "positive_incident_cluster_iou_interval": iou_gate["passes_positive_incident_cluster_interval"],
            "positive_incident_cluster_boundary_interval": boundary_gate[
                "passes_positive_incident_cluster_interval"
            ],
            "advancing_front_signal_above_persistence": (
                front_f1 > persistence["growth_tolerance_1_cell.f1"]["incident_weighted_mean"]
            ),
            "held_out_historical_skill_passed": bool(
                iou_gate["passes_positive_incident_cluster_interval"]
                and boundary_gate["passes_positive_incident_cluster_interval"]
                and front_f1 > persistence["growth_tolerance_1_cell.f1"]["incident_weighted_mean"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["claim_gate"], indent=2))


if __name__ == "__main__":
    main()
