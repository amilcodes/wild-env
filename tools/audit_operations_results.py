"""Audit live provenance and report task-level dispersion without changing the frozen score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from aeolus.agents.benchmark import write_json
from aeolus.agents.contract import digest


def audit(root):
    run = root / "benchmark"
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    replays = json.loads((run / "replay-verification.json").read_text())
    assert summary["complete"] and len(summary["records"]) == manifest["expected_episodes"]
    assert len(replays) == len(summary["records"]) and all(x["verified"] for x in replays)
    replay_by_trace = {x["trace"]: x for x in replays}
    specs = {x["name"]: x for x in manifest["agents"]}
    initial, groups, latencies = defaultdict(set), defaultdict(list), defaultdict(list)
    regret_parts = defaultdict(list)
    counts, failures, mistakes = defaultdict(Counter), [], []
    auxiliary = defaultdict(lambda: defaultdict(Counter))
    for record in summary["records"]:
        path = run / record["trace"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["trace_sha256"]
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        header, steps = entries[0], entries[1:-1]
        assert entries[-1]["result"] == {k: v for k, v in record.items() if k != "trace_sha256"}
        assert len(steps) == record["decisions"] == replay_by_trace[path.name]["decisions"]
        assert digest(header["instance"]) == record["instance_sha256"]
        assert header["world"] == record["world"] and header["repeat"] == record["repeat"]
        assert header["policy"] == record["policy"]
        policy, identity = record["policy"], record["instance_sha256"]
        initial[identity].add(steps[0]["observation_sha256"])
        groups[(policy, identity)].append(record)
        assert header["agent"] == specs.get(policy)
        valid_regret, invalid_regret = 0.0, 0.0
        for step in steps:
            obs, receipt = step["observation"], step["receipt"]
            meta, usage = step.get("adapter_metadata") or {}, step.get("usage") or {}
            assert digest(obs) == step["observation_sha256"]
            assert not {"world", "seed", "family", "evaluation", "net_reward", "optimal_value"} & obs.keys()
            counts[policy]["decisions"] += 1
            counts[policy]["invalid_dispatches"] += int(not receipt["valid"])
            counts[policy]["disallowed_tool_events"] += len(meta.get("tools_used", []))
            counts[policy]["infrastructure_errors"] += int(step["adapter_error"] is not None)
            loss = step["evaluation"]["regret"]
            if receipt["valid"]:
                valid_regret += loss
            else:
                invalid_regret += loss
            if loss > 1e-8:
                counts[policy]["decisions_with_positive_regret"] += 1
                mistakes.append(
                    {
                        "policy": policy,
                        "family": record["family"],
                        "seed": record["seed"],
                        "level": record["level"],
                        "world": record["world"],
                        "repeat": record["repeat"],
                        "tick": obs["tick"],
                        "regret": loss,
                        "normalized_regret_points": 100 * loss / record["optimal_expected_reward"],
                        "forecast_known": obs["forecast"] != "unknown",
                        "valid": receipt["valid"],
                        "trace": path.name,
                    }
                )
            if not receipt["valid"]:
                failures.append(
                    {
                        "policy": policy,
                        "trace": path.name,
                        "tick": obs["tick"],
                        "error": receipt["error"],
                        "adapter_error": step["adapter_error"],
                    }
                )
            if policy not in specs:
                continue
            spec = specs[policy]
            counts[policy]["live_calls"] += 1
            matches = step["model"] == spec["model"] and meta.get("reported_models") == [spec["model"]]
            if spec["provider"] == "codex_cli":
                matches = matches and meta.get("model_identity_source") == "codex_turn_context"
                counts[policy]["effort_verified"] += int(
                    meta.get("reasoning_effort") == spec["reasoning_effort"]
                )
            counts[policy]["model_identity_verified"] += int(matches)
            counts[policy]["prompt_verified"] += int(
                meta.get("effective_system_sha256")
                == manifest["adapter_runtime"][policy]["effective_system_sha256"]
            )
            counts[policy]["unknown_usage_calls"] += int(
                any(usage.get(k) is None for k in ("input_tokens", "output_tokens"))
            )
            for key in ("input_tokens", "output_tokens"):
                counts[policy][key] += usage.get(key) or 0
            counts[policy]["cached_input_tokens"] += meta.get("cached_input_tokens") or 0
            counts[policy]["model_rounds"] += meta.get("model_rounds") or 0
            for model, values in meta.get("usage_by_model", {}).items():
                auxiliary[policy][model].update({k: v for k, v in values.items() if v is not None})
            latencies[policy].append(step["latency_s"])
        assert sum(not s["receipt"]["valid"] for s in steps) == record["invalid_decisions"]
        scale = 100 / record["optimal_expected_reward"]
        assert abs((valid_regret + invalid_regret) * scale - record["normalized_regret_pct"]) < 1e-8
        regret_parts[(policy, identity)].append((valid_regret * scale, invalid_regret * scale))
    assert all(len(x) == 1 for x in initial.values())
    task_rows = []
    for (policy, identity), records in sorted(groups.items()):
        by_repeat = defaultdict(list)
        for r in records:
            by_repeat[r["repeat"]].append(r["normalized_regret_pct"])
        means = [statistics.mean(values) for _, values in sorted(by_repeat.items())]
        task_rows.append(
            {
                "policy": policy,
                "instance_sha256": identity,
                "family": records[0]["family"],
                "level": records[0]["level"],
                "seed": records[0]["seed"],
                "mean_regret_pct": statistics.mean(means),
                "repeat_spread_points": max(means) - min(means),
                "repeat_means": means,
                "episodes": len(records),
            }
        )
    dispersion = []
    for policy in sorted(counts):
        tasks = [t for t in task_rows if t["policy"] == policy]
        values = np.array([t["mean_regret_pct"] for t in tasks])
        rng = np.random.default_rng(0)
        boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        row = {
            "policy": policy,
            "mean_regret_pct": float(values.mean()),
            "instance_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "median_instance_regret_pct": float(np.median(values)),
            "worst_instance_regret_pct": float(values.max()),
            "optimal_instances": int(np.sum(values < 1e-8)),
            "mean_repeat_spread_points": statistics.mean(t["repeat_spread_points"] for t in tasks),
            "valid_decision_regret_points": statistics.mean(
                statistics.mean(x[0] for x in regret_parts[(policy, t["instance_sha256"])]) for t in tasks
            ),
            "invalid_dispatch_regret_points": statistics.mean(
                statistics.mean(x[1] for x in regret_parts[(policy, t["instance_sha256"])]) for t in tasks
            ),
            "family_mean_regret_pct": {
                f: statistics.mean(t["mean_regret_pct"] for t in tasks if t["family"] == f)
                for f in sorted({t["family"] for t in tasks})
            },
        }
        if latencies[policy]:
            row["latency_median_s"] = statistics.median(latencies[policy])
            row["latency_p95_s"] = float(np.quantile(latencies[policy], 0.95))
        dispersion.append(row)
    verified = all(counts[p]["model_identity_verified"] == counts[p]["live_calls"] for p in specs)
    prompts = all(counts[p]["prompt_verified"] == counts[p]["live_calls"] for p in specs)
    effort = all(
        counts[p]["effort_verified"] == counts[p]["live_calls"]
        for p, s in specs.items()
        if s["provider"] == "codex_cli"
    )
    sensitivity = []
    for left, right in itertools.combinations(sorted(specs), 2):
        lhs = {t["instance_sha256"]: t for t in task_rows if t["policy"] == left}
        rhs = {t["instance_sha256"]: t for t in task_rows if t["policy"] == right}
        assert lhs.keys() == rhs.keys()
        differences = [rhs[k]["mean_regret_pct"] - lhs[k]["mean_regret_pct"] for k in lhs]
        families = sorted({t["family"] for t in lhs.values()})
        family_means = {
            f: statistics.mean(
                rhs[k]["mean_regret_pct"] - lhs[k]["mean_regret_pct"] for k in lhs if lhs[k]["family"] == f
            )
            for f in families
        }
        repeated = [
            statistics.mean(rhs[k]["repeat_means"][r] - lhs[k]["repeat_means"][r] for k in lhs)
            for r in range(manifest["suite"]["repeats"])
        ]
        omitted = {
            f: statistics.mean(
                rhs[k]["mean_regret_pct"] - lhs[k]["mean_regret_pct"] for k in lhs if lhs[k]["family"] != f
            )
            for f in families
            if len(families) > 1
        }
        sensitivity.append(
            {
                "left": left,
                "right": right,
                "positive_favors_left": True,
                "instance_wins_left": sum(x > 1e-8 for x in differences),
                "instance_wins_right": sum(x < -1e-8 for x in differences),
                "instance_ties": sum(abs(x) <= 1e-8 for x in differences),
                "repeat_advantages_left": repeated,
                "family_advantages_left": family_means,
                "leave_one_family_out_advantages_left": omitted,
            }
        )
    result = {
        "protocol": manifest["protocol"],
        "source": manifest["source"],
        "suite_sha256": manifest["suite_sha256"],
        "episodes": len(summary["records"]),
        "all_model_identities_verified": verified,
        "all_effective_prompts_verified": prompts,
        "all_codex_efforts_verified": effort,
        "claude_effort_note": (
            "Requested high effort; CLI events do not independently attest effective effort."
        ),
        "identical_initial_observations_across_policies_worlds_repeats": True,
        "all_replays_and_trace_checksums_verified": True,
        "all_episodes_eligible": all(r["eligible"] for r in summary["records"]),
        "policies": dict(counts),
        "auxiliary_model_usage": dict(auxiliary),
        "failures": failures,
        "dispersion": sorted(dispersion, key=lambda r: r["mean_regret_pct"]),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inference_note": "Descriptive intervals; use the frozen paired Holm tests for model separation.",
    }
    write_json(root / "validity-audit.json", result)
    write_json(root / "instance-scores.json", task_rows)
    write_json(
        root / "sensitivity.json",
        {
            "note": "Descriptive stability checks only; the frozen paired tests determine separation.",
            "model_pairs": sensitivity,
        },
    )
    write_json(root / "decision-errors.json", sorted(mistakes, key=lambda r: -r["normalized_regret_points"]))
    with (root / "instance-scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[k for k in task_rows[0] if k != "repeat_means"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(task_rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "episodes",
                    "all_model_identities_verified",
                    "all_effective_prompts_verified",
                    "all_codex_efforts_verified",
                    "all_episodes_eligible",
                )
            },
            indent=2,
        )
    )
