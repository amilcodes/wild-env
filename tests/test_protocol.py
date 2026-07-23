from __future__ import annotations

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.evaluation.protocol import (
    EvaluationCase,
    audit_evaluation_partitions,
    paired_policy_summary,
)
from aeolus.policies import rollout_lookahead_diagnostics


def test_partition_audit_detects_incident_leakage() -> None:
    valid_cases = (
        EvaluationCase(
            "train-a",
            "incident-a",
            "sierra",
            2021,
            "grass",
            "moderate-wind",
            "train",
        ),
        EvaluationCase(
            "dev-b",
            "incident-b",
            "basin",
            2022,
            "shrub",
            "high-wind",
            "development",
        ),
        EvaluationCase(
            "test-c",
            "incident-c",
            "cascades",
            2023,
            "timber",
            "low-wind",
            "test",
        ),
    )
    assert audit_evaluation_partitions(valid_cases)["valid"]
    leaked = (
        *valid_cases,
        EvaluationCase(
            "test-a",
            "incident-a",
            "sierra",
            2021,
            "grass",
            "moderate-wind",
            "test",
        ),
    )
    audit = audit_evaluation_partitions(leaked)
    assert not audit["valid"]
    assert audit["group_leakage"]["incident_id"][0]["value"] == "incident-a"


def test_paired_policy_summary_bootstraps_case_clusters() -> None:
    records = []
    for case_index in range(5):
        for seed in range(4):
            baseline = 100.0 + 2.0 * case_index + seed
            records.extend(
                (
                    {
                        "case_id": f"case-{case_index}",
                        "seed": seed,
                        "policy": "baseline",
                        "loss": baseline,
                    },
                    {
                        "case_id": f"case-{case_index}",
                        "seed": seed,
                        "policy": "candidate",
                        "loss": baseline - 8.0 - case_index,
                    },
                )
            )
    summary = paired_policy_summary(
        records,
        candidate_policy="candidate",
        baseline_policy="baseline",
        metric="loss",
        lower_is_better=True,
        bootstrap_samples=1000,
        seed=12,
    )
    assert summary["paired_records"] == 20
    assert summary["case_clusters"] == 5
    assert summary["mean_improvement"] == 10.0
    assert summary["ci95_improvement_low"] > 0.0
    assert summary["passes_positive_cluster_interval"]


def test_rollout_lookahead_is_deterministic_and_does_not_mutate_source() -> None:
    simulator = AeolusSimulator(
        ScenarioConfig(
            width=24,
            height=24,
            max_tasks=16,
            horizon_min=12,
            decision_interval_min=2,
            spotting_rate=0.0,
        )
    )
    before_minute = simulator.state.minute
    before_events = len(simulator.state.events)
    first = rollout_lookahead_diagnostics(
        simulator,
        horizon_decisions=2,
    )
    second = rollout_lookahead_diagnostics(
        simulator,
        horizon_decisions=2,
    )
    assert first["actions"] == second["actions"]
    assert first["selected_proposal"] == second["selected_proposal"]
    assert len(first["trials"]) >= 2
    assert simulator.state.minute == before_minute
    assert len(simulator.state.events) == before_events
    observations = simulator.observations()
    for resource_id, action in first["actions"].items():
        assert observations[resource_id]["action_mask"][action]
