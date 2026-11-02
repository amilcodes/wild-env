"""Audit completed agent evidence and export plain episode and paired-result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from aeolus.agents.benchmark import write_json
from aeolus.agents.contract import digest
from aeolus.evaluation.protocol import paired_policy_summary


def audit(root, prior_path=None):
    run = root / "benchmark"
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    replays = json.loads((run / "replay-verification.json").read_text())
    assert summary["complete"] and len(summary["records"]) == summary["expected_episodes"]
    assert len(replays) == len(summary["records"]) and all(x["verified"] for x in replays)
    replay_by_trace = {x["trace"]: x for x in replays}
    initial, scenarios, latency = defaultdict(set), defaultdict(set), defaultdict(list)
    policies = defaultdict(
        lambda: {
            "decisions": 0,
            "valid_decisions": 0,
            "model_identity_verified": 0,
            "codex_effort_verified": 0,
            "non_hold_decisions": 0,
            "nonempty_memory": 0,
            "disallowed_tools": 0,
            "model_rounds": 0,
            "cached_input_tokens": 0,
            "unknown_usage_calls": 0,
            "input_tokens_reported": 0,
            "output_tokens_reported": 0,
            "task_assignments": Counter(),
            "failures": Counter(),
            "usage_by_model": defaultdict(Counter),
        }
    )
    failure_details = []
    for record in summary["records"]:
        path = run / record["trace"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["trace_sha256"]
        entries = [json.loads(line) for line in path.open()]
        header, decisions = entries[0], entries[1:-1]
        assert entries[-1]["type"] == "result"
        expected = {
            k: v for k, v in record.items() if k not in ("trace_sha256", "source_run", "source_trace")
        }
        expected["trace"] = record.get("source_trace", record["trace"])
        assert entries[-1]["result"] == expected
        assert len(decisions) == record["decisions"] == replay_by_trace[path.name]["decisions"]
        assert 0 < record["minute"] <= header["scenario"]["horizon_min"]
        assert (
            record["minute"] == header["scenario"]["horizon_min"] or record["contained"] or record["escaped"]
        )
        assert header["agent"] == next(x for x in manifest["agents"] if x["name"] == record["policy"])
        assert header["case_id"] == record["case_id"] and header["seed"] == record["seed"]
        assert digest(header["scenario"]) == record["scenario_sha256"]
        assert sum(not step["receipt"]["valid"] for step in decisions) == record["failed_decisions"]
        key = (record["case_id"], record["seed"])
        scenarios[key].add(record["scenario_sha256"])
        initial[key].add(decisions[0]["observation_sha256"])
        policy = policies[record["policy"]]
        for step in decisions:
            obs, receipt = step["observation"], step["receipt"]
            meta, usage = step.get("adapter_metadata") or {}, step.get("usage") or {}
            assert digest(obs) == step["observation_sha256"]
            policy["decisions"] += 1
            policy["valid_decisions"] += int(receipt["valid"])
            policy["non_hold_decisions"] += int(any(receipt["actions"].values()))
            policy["nonempty_memory"] += int(bool(obs["memory"]))
            policy["disallowed_tools"] += len(meta.get("tools_used", []))
            policy["model_rounds"] += meta.get("model_rounds", 0)
            policy["cached_input_tokens"] += meta.get("cached_input_tokens") or 0
            policy["unknown_usage_calls"] += int(
                any(usage.get(k) is None for k in ("input_tokens", "output_tokens"))
            )
            for token in ("input_tokens", "output_tokens"):
                policy[token + "_reported"] += usage.get(token) or 0
            for model, counts in meta.get("usage_by_model", {}).items():
                policy["usage_by_model"][model].update(counts)
            if not receipt["valid"]:
                policy["failures"][receipt["error"]] += 1
                failure_details.append(
                    {
                        "policy": record["policy"],
                        "case_id": record["case_id"],
                        "seed": record["seed"],
                        "decision_id": receipt["decision_id"],
                        "error": receipt["error"],
                        "trace": path.name,
                        "adapter_error": step["adapter_error"],
                    }
                )
            provider = header["agent"]["provider"]
            model_matches = step.get("model") == header["agent"]["model"]
            if provider == "codex_cli":
                policy["model_identity_verified"] += int(
                    model_matches and meta.get("model_identity_source") == "codex_turn_context"
                )
                policy["codex_effort_verified"] += int(
                    meta.get("reasoning_effort") == header["agent"]["reasoning_effort"]
                )
            elif provider == "claude_cli":
                policy["model_identity_verified"] += int(
                    model_matches and meta.get("reported_models") == [header["agent"]["model"]]
                )
            if meta:
                assert (
                    meta["effective_system_sha256"]
                    == manifest["adapter_runtime"][record["policy"]]["effective_system_sha256"]
                )
            latency[record["policy"]].append(step["latency_s"])
            kinds = {t["id"]: t["kind"] for t in obs["tasks"]}
            for action in receipt["actions"].values():
                policy["task_assignments"][kinds[action]] += 1
            for forbidden in (
                "weighted_loss",
                "reward_return",
                "escaped",
                "contained",
                "rng_state",
                "cumulative_cost",
                "exposure_units",
            ):
                assert forbidden not in obs
            assert obs["observation_age_basis"] == "minutes_since_delivery_to_belief"
    assert all(len(v) == 1 for v in initial.values())
    assert all(len(v) == 1 for v in scenarios.values())
    assert sum(x["decisions"] for x in policies.values()) == sum(x["decisions"] for x in replays)
    for name, rows in policies.items():
        rows["mean_latency_s"] = statistics.mean(latency[name])
        rows["median_latency_s"] = statistics.median(latency[name])
        rows["p95_latency_s"] = sorted(latency[name])[int(0.95 * (len(latency[name]) - 1))]
    comparisons = []
    for candidate in manifest["agents"]:
        if candidate["provider"] == "baseline":
            continue
        for baseline in [x for x in manifest["agents"] if x["provider"] == "baseline"]:
            comparison = paired_policy_summary(
                summary["records"],
                candidate_policy=candidate["name"],
                baseline_policy=baseline["name"],
                metric="loss_growth",
                lower_is_better=True,
                bootstrap_samples=2000,
                seed=0,
            )
            comparison["eligible"] = all(
                x["eligible"]
                for x in summary["records"]
                if x["policy"] in {candidate["name"], baseline["name"]}
            )
            comparison["interval_is_descriptive"] = True
            comparison["failed_episodes_included"] = True
            comparisons.append(comparison)
    checked, drift, mismatched = 0, [], set()
    if prior_path is not None:
        prior = json.loads(prior_path.read_text())
        lookup = {(x["policy"], x["case_id"], x["seed"]): x for x in prior["records"]}
        for row in summary["records"]:
            key = (row["policy"], row["case_id"], row["seed"])
            if row["provider"] != "baseline" or key not in lookup:
                continue
            checked += 1
            for field in (
                "loss_growth",
                "burned_area_ha",
                "resource_cost_units",
                "exposure_units",
                "decisions",
            ):
                if abs(row[field] - lookup[key][field]) > 1e-9:
                    drift.append({"key": list(key), "field": field})
                    mismatched.add(key)
    model_specs = [x for x in manifest["agents"] if x["provider"] != "baseline"]
    verified = all(
        policies[x["name"]]["model_identity_verified"] == policies[x["name"]]["decisions"]
        for x in model_specs
    )
    result = {
        "schema_version": 1,
        "suite_sha256": summary["suite_sha256"],
        "source": manifest["source"],
        "protocol": manifest["protocol"],
        "episodes": len(summary["records"]),
        "conditions": len(initial),
        "decisions": sum(x["decisions"] for x in policies.values()),
        "all_episodes_eligible": all(x["eligible"] for x in summary["records"]),
        "all_traces_checksum_verified": True,
        "all_traces_replay_verified": True,
        "same_initial_observations_for_all_policies": True,
        "same_case_seed_scenarios": True,
        "all_cli_model_identities_verified": verified,
        "identity_basis": (
            "Codex session turn_context; Claude init/assistant events. Not provider-signed snapshots. "
            "Claude effort is requested, not independently attested."
        ),
        "disallowed_tool_events": sum(x["disallowed_tools"] for x in policies.values()),
        "baseline_reference_episodes_checked": checked,
        "baseline_reference_episodes_matching": checked - len(mismatched),
        "baseline_metric_drift": drift,
        "policies": dict(policies),
        "failures": failure_details,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Synthetic incident-desk control benchmark. Validated execution and observation boundaries "
            "do not establish operational wildfire validity or broad frontier capability."
        ),
    }
    write_json(root / "validity-audit.json", result)
    write_json(root / "paired-comparisons.json", comparisons)
    fields = [
        "policy",
        "provider",
        "case_id",
        "seed",
        "requested_model",
        "minute",
        "loss_growth",
        "burned_area_ha",
        "resource_cost_units",
        "exposure_units",
        "agent_calls",
        "failed_decisions",
        "eligible",
        "agent_latency_s",
        "trace_sha256",
    ]
    with (root / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["records"])
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
                "decisions": result["decisions"],
                "all_episodes_eligible": result["all_episodes_eligible"],
                "model_identity_verified": verified,
                "baseline_reference_matches": checked - len(mismatched),
                "failures": failure_details,
                "leaderboard": summary["leaderboard"],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-reference", type=Path)
    args = parser.parse_args()
    audit(args.run, args.baseline_reference)
