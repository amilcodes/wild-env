"""Plain, replayable operations evaluations with exact decision-regret scoring."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from aeolus.agents.adapters import AgentFailure, AgentSpec
from aeolus.agents.benchmark import source_identity, write_json
from aeolus.agents.cli_adapters import CliAdapter
from aeolus.agents.contract import DISPATCH_SCHEMA, canonical_json, digest
from aeolus.agents.operations import (
    PROMPT,
    PROTOCOL,
    OperationsDesk,
    Planner,
    ShortPlanner,
    dispatch,
    generate,
    greedy_action,
    initial_state,
    instance_from_dict,
    joint_actions,
    payoff,
)

BASELINES = ("hold", "myopic", "deadline", "two-step", "optimal")


def load_suite(path):
    raw = yaml.safe_load(Path(path).read_text())
    if set(raw) != {"schema_version", "suite_id", "split", "repeats", "cases"} or raw["schema_version"] != 2:
        raise ValueError("operations suite requires schema_version 2 and exact declared fields")
    if (
        raw["split"] not in {"development", "evaluation"}
        or type(raw["repeats"]) is not int
        or not 1 <= raw["repeats"] <= 8
    ):
        raise ValueError("invalid split or repeats")
    instances = [generate(**case) for case in raw["cases"]]
    if not instances or len({x.identity for x in instances}) != len(instances):
        raise ValueError("instances must be nonempty and unique")
    return raw, instances


def worlds(instance):
    return (0, 1) if any(job.values[0] != job.values[1] for job in instance.jobs) else (0,)


def episode(instance, world, repeat, policy, adapter, planner, short, trace):
    desk = OperationsDesk(instance, world)
    optimum = planner.value(initial_state(instance))
    regret, invalid, infrastructure, calls, latency = 0.0, 0, 0, 0, 0.0
    opportunities, optimal_choices = 0, 0
    tokens = {"input_tokens": 0, "output_tokens": 0}
    unknown_usage = 0
    header = {
        "type": "episode",
        "protocol": PROTOCOL,
        "instance": asdict(instance),
        "world": world,
        "repeat": repeat,
        "policy": policy,
        "agent": adapter.spec.public_dict() if adapter else None,
    }
    with trace.open("x") as stream:
        stream.write(canonical_json(header) + "\n")
        while not desk.done:
            obs, before = desk.observation(), desk.state
            response, error, reply = None, None, None
            started = time.monotonic()
            if adapter:
                calls += 1
                try:
                    reply = adapter.act(json.loads(canonical_json(obs)))
                    response, error = reply.dispatch, reply.error
                except AgentFailure as exc:
                    error = str(exc)
                if reply:
                    for key in tokens:
                        tokens[key] += reply.usage.get(key) or 0
                    unknown_usage += int(any(reply.usage.get(k) is None for k in tokens))
                    if reply.model != adapter.spec.model:
                        error = error or "model_identity_unverified"
                else:
                    unknown_usage += 1
                infrastructure += int(error is not None)
            else:
                if policy == "hold":
                    selected = tuple(-1 for _ in instance.resources)
                elif policy == "optimal":
                    selected = planner.action(before)
                elif policy == "two-step":
                    selected = short.action(before)
                else:
                    selected = greedy_action(instance, before, deadline=policy == "deadline")
                response = dispatch(instance, before, obs, selected)
            elapsed = time.monotonic() - started
            latency += elapsed
            action, receipt = desk.step(response, failure=error)
            loss = planner.regret(before, action)
            choice = sum(1 for _ in joint_actions(instance, before)) > 1
            opportunities += int(choice)
            optimal_choices += int(choice and loss < 1e-8)
            regret += loss
            invalid += int(not receipt["valid"])
            entry = {
                "type": "decision",
                "observation": obs,
                "observation_sha256": digest(obs),
                "response": response,
                "adapter_error": error,
                "receipt": receipt,
                "model": reply.model if reply else policy,
                "usage": reply.usage if reply else None,
                "adapter_metadata": reply.metadata if reply else None,
                "response_id": reply.response_id if reply else None,
                "latency_s": elapsed,
                "evaluation": {
                    "regret": loss,
                    "optimal_value_before": planner.value(before),
                    "state_after_sha256": digest(asdict(desk.state)),
                },
            }
            stream.write(canonical_json(entry) + "\n")
            stream.flush()
        reward = payoff(instance, desk.state, world)
        result = {
            "policy": policy,
            "provider": adapter.spec.provider if adapter else "operations_baseline",
            "model": adapter.spec.model if adapter else policy,
            "instance_sha256": instance.identity,
            "family": instance.family,
            "seed": instance.seed,
            "level": instance.level,
            "world": world,
            "repeat": repeat,
            "decisions": desk.state.tick,
            "agent_calls": calls,
            "net_reward": reward,
            "optimal_expected_reward": optimum,
            "decision_regret": regret,
            "normalized_regret_pct": 100.0 * regret / max(1.0, optimum),
            "reward_capture_pct": 100.0 * reward / max(1.0, optimum),
            "opportunities": opportunities,
            "optimal_choices": optimal_choices,
            "invalid_decisions": invalid,
            "infrastructure_errors": infrastructure,
            "eligible": infrastructure == 0,
            "latency_s": latency,
            "tokens_reported": tokens,
            "unknown_usage_calls": unknown_usage,
            "trace": trace.name,
        }
        stream.write(canonical_json({"type": "result", "result": result}) + "\n")
    result["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    return result


def summarize(records):
    rows = []
    for policy in sorted({r["policy"] for r in records}):
        selected = [r for r in records if r["policy"] == policy]
        # Each generated instance gets equal weight, regardless of its number of worlds/repeats.
        groups = {}
        for r in selected:
            groups.setdefault(r["instance_sha256"], []).append(r)
        row = {
            "policy": policy,
            "episodes": len(selected),
            "instances": len(groups),
            "eligible": all(r["eligible"] for r in selected),
        }
        for metric in ("normalized_regret_pct", "reward_capture_pct", "net_reward"):
            row["mean_" + metric] = statistics.mean(
                statistics.mean(r[metric] for r in group) for group in groups.values()
            )
        for field in (
            "decisions",
            "agent_calls",
            "invalid_decisions",
            "infrastructure_errors",
            "opportunities",
            "optimal_choices",
            "latency_s",
        ):
            row[field] = sum(r[field] for r in selected)
        rows.append(row)
    return sorted(rows, key=lambda r: (not r["eligible"], r["mean_normalized_regret_pct"], r["policy"]))


def run(suite_path, agents_path, out, workers=4, baselines=True):
    raw, instances = load_suite(suite_path)
    if not 1 <= workers <= 8 or out.exists():
        raise ValueError("workers must be 1..8 and output must be new")
    specs = (
        []
        if agents_path is None
        else [
            AgentSpec(**s)
            for s in yaml.safe_load(Path(agents_path).read_text())["agents"]
            if s["provider"] != "baseline"
        ]
    )
    if any(s.provider not in {"codex_cli", "claude_cli"} for s in specs):
        raise ValueError("operations live benchmark currently uses authenticated CLI adapters")
    if any(s.max_calls < max(i.horizon for i in instances) for s in specs):
        raise ValueError("call budget must cover every decision; no hidden budget holds")
    names = [s.name for s in specs]
    if len(set(names)) != len(names) or set(names) & set(BASELINES):
        raise ValueError("duplicate or reserved policy name")
    adapters = {s.name: CliAdapter(s, system_prompt=PROMPT) for s in specs}
    planners, shorts, certificates = {}, {}, []
    # Solve every instance before issuing any model request. A budget failure aborts the suite.
    for instance in instances:
        planner = Planner(instance)
        value = planner.value(initial_state(instance))
        planners[instance.identity], shorts[instance.identity] = planner, ShortPlanner(instance)
        certificates.append(
            {
                "instance_sha256": instance.identity,
                "optimal_expected_reward": value,
                "states_solved": planner.states,
                "exact": True,
            }
        )
    out.mkdir(parents=True)
    policies = ([*BASELINES] if baselines else []) + names
    if not policies:
        raise ValueError("no policies selected")
    jobs = [
        (i, w, repeat, policy)
        for i in instances
        for policy in policies
        for repeat in range(1 if policy in BASELINES else raw["repeats"])
        for w in worlds(i)
    ]
    manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "suite": raw,
        "suite_sha256": digest(raw),
        "system_prompt": PROMPT,
        "prompt_sha256": digest(PROMPT),
        "dispatch_schema": DISPATCH_SCHEMA,
        "source": source_identity(),
        "agents": [s.public_dict() for s in specs],
        "baselines": list(BASELINES) if baselines else [],
        "adapter_runtime": {name: a.identity for name, a in adapters.items()},
        "instances": [asdict(i) for i in instances],
        "certificates": certificates,
        "expected_episodes": len(jobs),
        "workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "Normalized Bellman regret, equally weighted per instance; lower is better.",
        "failure_rule": (
            "Invalid actions hold for one tick and remain scored. "
            "Infrastructure/identity errors make comparisons ineligible."
        ),
        "limitations": [
            "Synthetic scheduling abstraction, without continuous fire physics or expert-calibrated loss.",
            "The optimum uses public forecast belief, without the hidden world before reveal.",
            "Repeated samples are not independent instances; provider scaffolding differs.",
            "No output-token cap is exposed by CLI adapters.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    completed = {}

    def execute(index):
        i, w, repeat, policy = jobs[index]
        return episode(
            i,
            w,
            repeat,
            policy,
            adapters.get(policy),
            planners[i.identity],
            shorts[i.identity],
            out / f"episode-{index:04d}.jsonl",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, index): index for index in range(len(jobs))}
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            records = [completed[k] for k in sorted(completed)]
            summary = {
                "schema_version": 2,
                "suite_sha256": digest(raw),
                "complete": len(records) == len(jobs),
                "expected_episodes": len(jobs),
                "records": records,
                "leaderboard": summarize(records),
            }
            write_json(out / "summary.json", summary)
            print(f"[{len(records)}/{len(jobs)}] {completed[index]['policy']}", flush=True)
    return summary


def replay(path, planner=None):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines()]
    header = entries[0]
    if header["protocol"] != PROTOCOL or entries[-1]["type"] != "result":
        raise ValueError("incomplete trace or incompatible protocol")
    instance = instance_from_dict(header["instance"])
    desk = OperationsDesk(instance, header["world"])
    planner = planner or Planner(instance)
    if planner.instance != instance:
        raise ValueError("reference planner belongs to a different instance")
    regret = 0.0
    for entry in entries[1:-1]:
        if (
            digest(desk.observation()) != entry["observation_sha256"]
            or digest(entry["observation"]) != entry["observation_sha256"]
        ):
            raise ValueError("observation drift")
        before = desk.state
        action, receipt = desk.step(entry["response"], failure=entry["adapter_error"])
        loss = planner.regret(before, action)
        evaluation = {
            "regret": loss,
            "optimal_value_before": planner.value(before),
            "state_after_sha256": digest(asdict(desk.state)),
        }
        if receipt != entry["receipt"] or evaluation != entry["evaluation"]:
            raise ValueError("transition or exact-regret drift")
        regret += loss
    result = entries[-1]["result"]
    optimum = planner.value(initial_state(instance))
    if (
        not desk.done
        or payoff(instance, desk.state, header["world"]) != result["net_reward"]
        or abs(regret - result["decision_regret"]) > 1e-9
        or desk.invalid != result["invalid_decisions"]
        or optimum != result["optimal_expected_reward"]
        or abs(100 * regret / max(1, optimum) - result["normalized_regret_pct"]) > 1e-9
        or desk.state.tick != result["decisions"]
    ):
        raise ValueError("terminal result drift")
    return {"trace": Path(path).name, "verified": True, "decisions": desk.state.tick}


def analyze(out, bootstrap_samples=10000):
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if not summary["complete"] or len(summary["records"]) != manifest["expected_episodes"]:
        raise ValueError("cannot analyze an incomplete evaluation")
    checks = []
    planners = {i.identity: Planner(i) for i in (instance_from_dict(raw) for raw in manifest["instances"])}
    for record in summary["records"]:
        if hashlib.sha256((out / record["trace"]).read_bytes()).hexdigest() != record["trace_sha256"]:
            raise ValueError("trace checksum mismatch")
        terminal = json.loads((out / record["trace"]).read_text().splitlines()[-1])["result"]
        if terminal != {k: v for k, v in record.items() if k != "trace_sha256"}:
            raise ValueError("summary differs from original trace result")
        checks.append(replay(out / record["trace"], planners[record["instance_sha256"]]))
    write_json(out / "replay-verification.json", checks)
    records = summary["records"]
    policies = sorted({r["policy"] for r in records})
    expected = set()
    for raw in manifest["instances"]:
        instance = instance_from_dict(raw)
        for policy in [*manifest["baselines"], *[s["name"] for s in manifest["agents"]]]:
            for repeat in range(1 if policy in manifest["baselines"] else manifest["suite"]["repeats"]):
                for world in worlds(instance):
                    expected.add((instance.identity, policy, repeat, world))
    actual = [(r["instance_sha256"], r["policy"], r["repeat"], r["world"]) for r in records]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("missing or duplicated frozen evaluation conditions")
    grouped = {}
    for policy in policies:
        grouped[policy] = {
            i: statistics.mean(
                r["normalized_regret_pct"]
                for r in records
                if r["policy"] == policy and r["instance_sha256"] == i
            )
            for i in {r["instance_sha256"] for r in records if r["policy"] == policy}
        }
    comparisons = []
    model_names = {s["name"] for s in manifest["agents"]}
    for left, right in itertools.combinations(policies, 2):
        keys = sorted(set(grouped[left]) & set(grouped[right]))
        delta = np.array([grouped[right][k] - grouped[left][k] for k in keys])
        # Cluster at the independently generated instance; do not inflate N with worlds or repeated calls.
        rng = np.random.default_rng(0)
        samples = delta[rng.integers(0, len(delta), (bootstrap_samples, len(delta)))].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        if len(delta) <= 18:
            signs = np.array(list(itertools.product((-1, 1), repeat=len(delta))), dtype=np.int8)
            null = (signs * delta).mean(axis=1)
            pvalue = float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))
        else:
            signs = rng.choice((-1, 1), size=(100000, len(delta)))
            null = (signs * delta).mean(axis=1)
            pvalue = float((1 + np.sum(np.abs(null) >= abs(delta.mean()) - 1e-12)) / (1 + len(null)))
        comparisons.append(
            {
                "left": left,
                "right": right,
                "instances": len(keys),
                "mean_advantage_left_regret_points": float(delta.mean()),
                "ci95": [float(low), float(high)],
                "paired_sign_flip_p": pvalue,
                "model_comparison": left in model_names and right in model_names,
                "eligible": all(r["eligible"] for r in records if r["policy"] in {left, right}),
            }
        )
    model_pairs = sorted(
        [c for c in comparisons if c["model_comparison"]], key=lambda c: c["paired_sign_flip_p"]
    )
    running = 0.0
    for rank, comparison in enumerate(model_pairs):
        running = max(running, (len(model_pairs) - rank) * comparison["paired_sign_flip_p"])
        comparison["holm_adjusted_p"] = min(1.0, running)
        comparison["separated_after_multiple_comparisons"] = bool(
            comparison["eligible"]
            and comparison["holm_adjusted_p"] < 0.05
            and (comparison["ci95"][0] > 0 or comparison["ci95"][1] < 0)
        )
    write_json(out / "comparisons.json", comparisons)
    fields = [k for k, v in records[0].items() if not isinstance(v, (dict, list))]
    with (out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    audit = {
        "protocol": PROTOCOL,
        "episodes": len(records),
        "decisions": sum(r["decisions"] for r in records),
        "live_model_requests": sum(r["agent_calls"] for r in records),
        "all_replays_verified": all(x["verified"] for x in checks),
        "all_trace_checksums_verified": True,
        "invalid_decisions": sum(r["invalid_decisions"] for r in records),
        "infrastructure_errors": sum(r["infrastructure_errors"] for r in records),
        "independent_instances": len(manifest["instances"]),
        "inference_note": (
            "Intervals resample generated instances, collapsing worlds and repeated calls. "
            "Paired sign-flip tests assume exchangeability; Holm correction covers model pairs. "
            "The small number of families limits generalization."
        ),
    }
    write_json(out / "audit.json", audit)
    return {"audit": audit, "leaderboard": summary["leaderboard"], "comparisons": comparisons}
