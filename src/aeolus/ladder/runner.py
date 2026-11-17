"""Resumable benchmark runner: references, baselines, model calls, summary.

Output directory layout (all plain JSON / JSONL / CSV / text):

- ``suite.json``        resolved suite, instance keys and SHA-256 identities, provenance
- ``instances.json``    every full instance (reloadable with ``instance_from_dict``)
- ``references.json``   exact optimum (information-matched, clairvoyant, open-loop) and baselines
- ``calls.jsonl``       one line per model call: response, errors, usage, latency, score
- ``summary.json``      per-model and per-tier scores, CIs, paired comparisons
- ``scores.csv``        one row per (policy, instance)
- ``report.txt``        human-readable leaderboard and separations
"""

from __future__ import annotations

import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from aeolus.agents.contract import canonical_json, digest
from aeolus.ladder import stats
from aeolus.ladder.baselines import BASELINES, baseline_plan
from aeolus.ladder.generator import GENERATOR_VERSION, TIERS, generate
from aeolus.ladder.llm import ModelSpec, PlanModel
from aeolus.ladder.plan import evaluate_plan
from aeolus.ladder.prompt import PROTOCOL, observation
from aeolus.ladder.solver import solve

# Errors that say nothing about the model's planning; retried, then reported as failures.
INFRASTRUCTURE = {
    "timeout",
    "cli_usage_limit",
    "cli_authentication_failed",
    "cli_model_unavailable",
    "cli_request_failed",
    "cli_invalid_event_stream",
    "model_mismatch",
}
SOLVED = 99.999


def provenance() -> dict[str, Any]:
    """Git commit and a hash of the ladder sources, so a run can be tied to exact code."""
    import hashlib
    import subprocess

    root = Path(__file__).resolve().parent
    source = hashlib.sha256()
    for file in sorted(root.glob("*.py")) + [root.parent / "agents" / "operations.py"]:
        source.update(file.name.encode() + b"\0" + file.read_bytes())
    info: dict[str, Any] = {"ladder_source_sha256": source.hexdigest()}
    try:

        def git(*args):
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

        info["git_commit"] = git("rev-parse", "HEAD").strip()
        info["git_dirty"] = bool(git("status", "--porcelain", "--", str(root)).strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def load_suite(path) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    raw = yaml.safe_load(Path(path).read_text())
    expected = {"schema_version", "suite_id", "split", "seed_offset", "tiers", "repeats"}
    if set(raw) != expected or raw["schema_version"] != 3:
        raise ValueError(f"suite must have exactly {sorted(expected)} with schema_version 3")
    if raw["split"] not in {"development", "evaluation"}:
        raise ValueError("split must be development or evaluation")
    instances = []
    for tier, count in sorted(raw["tiers"].items()):
        if int(tier) not in TIERS:
            raise ValueError(f"unknown tier {tier}")
        for i in range(int(count)):
            seed = int(raw["seed_offset"]) + i
            instances.append((f"t{int(tier)}-s{seed}", generate(int(tier), seed)))
    return raw, instances


def load_models(path) -> list[ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text())
    specs = [ModelSpec(**{**m, "command": tuple(m.get("command", ()))}) for m in raw["models"]]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("model names must be unique")
    return specs


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def references(instances, out: Path) -> dict[str, Any]:
    path = out / "references.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    changed = False
    for key, instance in instances:
        if key in cached and cached[key]["instance_sha256"] == instance.identity:
            continue
        info = solve(instance, "information")
        entry = {
            "instance_sha256": instance.identity,
            "tier": instance.level,
            "seed": instance.seed,
            "optimum": info.value,
            "optimum_detail": info.public(),
            "optimal_plan": info.plan,
            "clairvoyant": solve(instance, "clairvoyant").value,
            "open_loop": solve(instance, "open_loop").value,
            "baselines": {},
        }
        for name in BASELINES:
            outcome = evaluate_plan(instance, baseline_plan(instance, name))
            entry["baselines"][name] = {
                "expected_value": outcome.expected_value,
                "score": 100.0 * outcome.expected_value / info.value,
            }
        cached[key] = entry
        changed = True
    if changed:
        _write_json(path, cached)
    return cached


def _completed(calls_path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    done = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["model"], row["instance"], row["repeat"])] = row
    return done


def run(suite_path, models_path, out: Path, workers: int = 4, max_attempts: int = 2, only=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    suite, instances = load_suite(suite_path)
    by_key = dict(instances)
    _write_json(
        out / "suite.json",
        {
            "protocol": PROTOCOL,
            "generator_version": GENERATOR_VERSION,
            "provenance": provenance(),
            "suite": suite,
            "instances": {k: {"tier": i.level, "seed": i.seed, "sha256": i.identity} for k, i in instances},
        },
    )
    _write_json(out / "instances.json", {key: asdict(instance) for key, instance in instances})
    refs = references(instances, out)
    specs = load_models(models_path) if models_path else []
    if only:
        specs = [s for s in specs if s.name in only]
    calls_path = out / "calls.jsonl"
    done = _completed(calls_path)
    lock = threading.Lock()
    tasks = [
        (spec, key, repeat)
        for repeat in range(int(suite["repeats"]))
        for key, _ in instances
        for spec in specs
        if (prior := done.get((spec.name, key, repeat))) is None
        or (prior["infrastructure_error"] and prior["attempt"] < max_attempts)
    ]
    models = {spec.name: PlanModel(spec) for spec in specs}

    def execute(spec: ModelSpec, key: str, repeat: int):
        instance = by_key[key]
        obs = observation(instance)
        call = models[spec.name].act(obs)
        reply = call.reply
        error = call.error
        outcome = evaluate_plan(instance, reply.dispatch if reply.dispatch is not None else {"plan": []})
        prior = done.get((spec.name, key, repeat))
        row = {
            "model": spec.name,
            "model_id": spec.model,
            "provider": spec.provider,
            "effort": spec.effort,
            "instance": key,
            "tier": instance.level,
            "instance_sha256": instance.identity,
            "repeat": repeat,
            "attempt": (prior["attempt"] + 1) if prior else 1,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "observation_sha256": digest(obs),
            "system_sha256": models[spec.name].identity["system_sha256"],
            "error": error,
            "infrastructure_error": error in INFRASTRUCTURE or (error or "").startswith("transport"),
            "reported_model": reply.model,
            "reported_models": (reply.metadata or {}).get("reported_models"),
            "tools_used": (reply.metadata or {}).get("tools_used"),
            "usage": reply.usage,
            "latency_s": round(call.latency_s, 2),
            "response": reply.dispatch,
            "outcome": outcome.public(),
            "score": 100.0 * outcome.expected_value / refs[key]["optimum"],
        }
        with lock:
            with calls_path.open("a") as stream:
                stream.write(canonical_json(row) + "\n")
            done[(spec.name, key, repeat)] = row
        return row

    if tasks:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, *task) for task in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                flag = f" ERROR={row['error']}" if row["error"] else ""
                print(
                    f"[{n}/{len(tasks)}] {row['model']:<14} {row['instance']:<10} "
                    f"score={row['score']:6.1f} latency={row['latency_s']:6.1f}s{flag}",
                    flush=True,
                )
    summary = summarize(out, instances, refs, specs)
    return summary


def summarize(out: Path, instances, refs, specs) -> dict[str, Any]:
    calls_path = out / "calls.jsonl"
    rows = list(_completed(calls_path).values())
    tiers = {key: inst.level for key, inst in instances}
    keys = [key for key, _ in instances]
    # policy -> instance -> score (models: mean over repeats)
    scores: dict[str, dict[str, float]] = {}
    failures: dict[str, int] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name in BASELINES:
        scores[f"baseline:{name}"] = {k: refs[k]["baselines"][name]["score"] for k in keys}
    scores["reference:open_loop"] = {k: 100 * refs[k]["open_loop"] / refs[k]["optimum"] for k in keys}
    scores["reference:clairvoyant"] = {k: 100 * refs[k]["clairvoyant"] / refs[k]["optimum"] for k in keys}
    per_repeat: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["instance"] not in tiers:
            continue
        name = row["model"]
        failures[name] = failures.get(name, 0) + int(row["infrastructure_error"])
        per_repeat.setdefault(name, {}).setdefault(row["instance"], []).append(row["score"])
        m = meta.setdefault(
            name,
            {
                "calls": 0,
                "violations": 0,
                "parse_errors": 0,
                "latency_s": 0.0,
                "output_tokens": 0,
                "entries": 0,
            },
        )
        m["calls"] += 1
        m["violations"] += len(row["outcome"]["violations"])
        m["parse_errors"] += int(row["outcome"]["parse_error"] is not None or row["response"] is None)
        m["latency_s"] += row["latency_s"]
        m["output_tokens"] += (row["usage"] or {}).get("output_tokens") or 0
        m["entries"] += row["outcome"]["entries"]
    for name, by_instance in per_repeat.items():
        scores[name] = {k: stats.mean(v) for k, v in by_instance.items()}
    spec_by_name = {s.name: s for s in specs}

    leaderboard = []
    for name, by_instance in scores.items():
        shared = [k for k in keys if k in by_instance]
        values = [by_instance[k] for k in shared]
        row = {
            "policy": name,
            "kind": "model" if ":" not in name else name.split(":")[0],
            "label": spec_by_name[name].tier if name in spec_by_name else None,
            "instances": len(shared),
            "complete": len(shared) == len(keys),
            "mean_score": stats.mean(values),
            "ci95": stats.bootstrap_ci(values, [tiers[k] for k in shared]),
            "solved": sum(v >= SOLVED for v in values),
            "by_tier": {
                str(t): stats.mean([by_instance[k] for k in shared if tiers[k] == t])
                for t in sorted({tiers[k] for k in shared})
            },
            "by_regime": {
                regime: stats.mean([by_instance[k] for k in shared if regime_of(tiers[k]) == regime])
                for regime in ("windows", "packing")
                if any(regime_of(tiers[k]) == regime for k in shared)
            },
        }
        if name in meta:
            m = meta[name]
            row.update(
                {
                    "calls": m["calls"],
                    "failed_calls": failures.get(name, 0),
                    "parse_errors": m["parse_errors"],
                    "violations_per_call": m["violations"] / max(1, m["calls"]),
                    "mean_latency_s": m["latency_s"] / max(1, m["calls"]),
                    "mean_output_tokens": m["output_tokens"] / max(1, m["calls"]),
                }
            )
        leaderboard.append(row)
    leaderboard.sort(key=lambda r: -r["mean_score"])
    model_scores = {n: s for n, s in scores.items() if ":" not in n}
    comparisons = stats.pairwise(model_scores, tiers) if len(model_scores) > 1 else []
    baseline_vs = []
    for name in model_scores:
        for ref in ("baseline:search", "baseline:greedy"):
            baseline_vs.extend(c for c in stats.pairwise({name: model_scores[name], ref: scores[ref]}, tiers))
    summary = {
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instances": len(keys),
        "tiers": {str(t): sum(1 for k in keys if tiers[k] == t) for t in sorted(set(tiers.values()))},
        "models": {s.name: s.public() for s in specs},
        "leaderboard": leaderboard,
        "pairwise_models": comparisons,
        "models_vs_baselines": baseline_vs,
    }
    _write_json(out / "summary.json", summary)
    with (out / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["policy", "instance", "tier", "score"])
        for name, by_instance in sorted(scores.items()):
            for k in keys:
                if k in by_instance:
                    writer.writerow([name, k, tiers[k], f"{by_instance[k]:.4f}"])
    (out / "report.txt").write_text(report(summary))
    return summary


def regime_of(tier: int) -> str:
    return "packing" if TIERS[tier].overload else "windows"


def report(summary: dict[str, Any]) -> str:
    tiers = list(summary["tiers"])
    regimes = [r for r in ("windows", "packing") if any(regime_of(int(t)) == r for t in tiers)]
    lines = [
        f"WILDFIRE OPS LADDER ({summary['protocol']}) - {summary['instances']} instances, tiers "
        + ", ".join(f"{t}:{n}" for t, n in summary["tiers"].items()),
        "Score = 100 x expected net value / proven information-matched optimum (equal weight per instance).",
        "windows = tiers 1-5; packing = capacity-bound tiers 6-8.",
        "",
        f"{'policy':<24}{'mean':>7}  {'95% CI':<17}{'solved':>7}  "
        + "".join(f"{r:>9}" for r in regimes)
        + "  "
        + "".join(f"{'T' + t:>7}" for t in tiers),
    ]
    for row in summary["leaderboard"]:
        lo, hi = row["ci95"]
        lines.append(
            f"{row['policy']:<24}{row['mean_score']:7.1f}  [{lo:5.1f}, {hi:5.1f}]   "
            f"{row['solved']:>4}/{row['instances']:<3}"
            + "".join(f"{row.get('by_regime', {}).get(r, float('nan')):9.1f}" for r in regimes)
            + "  "
            + "".join(f"{row['by_tier'].get(t, float('nan')):7.1f}" for t in tiers)
            + ("" if row["complete"] else "  (incomplete)")
        )
    model_rows = [r for r in summary["leaderboard"] if r["kind"] == "model"]
    if model_rows:
        lines += [
            "",
            f"{'model':<24}{'calls':>6}{'failed':>7}{'unparsed':>9}"
            f"{'skips/call':>11}{'latency':>9}{'out tok':>9}",
        ]
        for r in model_rows:
            lines.append(
                f"{r['policy']:<24}{r['calls']:>6}{r['failed_calls']:>7}{r['parse_errors']:>9}"
                f"{r['violations_per_call']:>11.2f}{r['mean_latency_s']:>8.0f}s{r['mean_output_tokens']:>9.0f}"
            )
    if summary["pairwise_models"]:
        sep = [c for c in summary["pairwise_models"] if c["separated"]]
        lines += [
            "",
            f"PAIRED MODEL COMPARISONS (sign-flip test, Holm over {len(summary['pairwise_models'])} pairs; "
            f"separated = Holm p<0.05 and bootstrap CI excludes 0): {len(sep)} separated",
        ]
        for c in sorted(summary["pairwise_models"], key=lambda c: -abs(c["mean_difference"])):
            a, b, d, (lo, hi) = c["left"], c["right"], c["mean_difference"], c["ci95"]
            wins, losses = c["left_better_instances"], c["right_better_instances"]
            if d < 0:
                a, b, d, lo, hi, wins, losses = b, a, -d, -hi, -lo, losses, wins
            lines.append(
                f"  {'*' if c['separated'] else ' '} {a} > {b}: {d:5.1f} pts CI [{lo:5.1f}, {hi:5.1f}]  "
                f"Holm p={c['p_holm']:.4f}  better/worse on {wins}/{losses} of {c['instances']}"
            )
    return "\n".join(lines) + "\n"
