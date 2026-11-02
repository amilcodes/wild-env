"""Matched-seed episodes, complete transcripts, provenance, and deterministic replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aeolus.agents.adapters import AgentAdapter, AgentFailure, AgentSpec
from aeolus.agents.contract import DISPATCH_SCHEMA, PROTOCOL_VERSION, SYSTEM_PROMPT, canonical_json, digest
from aeolus.agents.desk import IncidentDesk
from aeolus.config import ScenarioConfig, _scenario
from aeolus.evaluation.protocol import paired_policy_summary


@dataclass(frozen=True)
class Case:
    case_id: str
    capability: str
    seeds: tuple[int, ...]
    scenario: ScenarioConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_suite(path: str | Path) -> tuple[dict[str, Any], list[Case]]:
    raw = yaml.safe_load(Path(path).read_text())
    required = {"schema_version", "suite_id", "split", "description", "scenario", "cases"}
    if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
        raise ValueError("suite requires schema_version=1, suite_id, split, description, scenario, cases")
    if raw["split"] not in {"development", "evaluation"}:
        raise ValueError("suite split must be development or evaluation")
    if not isinstance(raw["suite_id"], str) or not raw["suite_id"].strip():
        raise ValueError("suite_id must be nonempty")
    cases = []
    ids: set[str] = set()
    for item in raw["cases"]:
        if set(item) != {"id", "capability", "seeds", "overrides"}:
            raise ValueError("each case requires id, capability, seeds, overrides")
        case_id = item["id"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be nonempty and unique")
        ids.add(case_id)
        seeds = item["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("case seeds must be unique nonnegative integers")
        config = _scenario(_merge(raw["scenario"], item["overrides"]))
        if config.horizon_min < 1 or config.horizon_min % config.decision_interval_min:
            raise ValueError("horizon must be positive and divisible by the decision interval")
        resource_ids = [resource.resource_id for resource in config.resources]
        if not resource_ids or len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource IDs must be nonempty and unique")
        # V1 is a self-contained synthetic contract. Do not silently omit external input hashes.
        if (
            config.landscape_bundle
            or config.weather_forcing
            or any(
                resource.performance_surface_path or resource.delivery_surface_path
                for resource in config.resources
            )
        ):
            raise ValueError(
                "desk v1 suites are self-contained; external incident/performance files are unsupported"
            )
        canonical_json(asdict(config))  # Reject nonfinite manifest inputs.
        cases.append(Case(case_id, item["capability"], tuple(seeds), config))
    if not cases:
        raise ValueError("suite needs at least one case")
    return raw, cases


def source_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.suffix in {".py", ".npz", ".html"}
    }
    result: dict[str, Any] = {
        "source_sha256": digest(files),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.system(),
    }
    result["dependencies"] = {}
    for name in ("numpy", "scipy", "PyYAML", "gymnasium", "pettingzoo", "torch"):
        try:
            result["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][name] = None
    try:
        root = package.parents[1]
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_episode(case: Case, seed: int, spec: AgentSpec, adapter: Any, trace_path: Path) -> dict[str, Any]:
    desk = IncidentDesk(case.scenario, seed)
    usage = {"input_tokens": 0, "output_tokens": 0}
    usage_unknown = 0
    calls, failures, consecutive_errors = 0, 0, 0
    latency_s = 0.0
    returned_models: set[str] = set()
    header = {
        "type": "episode",
        "protocol": PROTOCOL_VERSION,
        "case_id": case.case_id,
        "seed": seed,
        "scenario": asdict(case.scenario),
        "agent": spec.public_dict(),
    }
    with trace_path.open("x") as trace:
        trace.write(canonical_json(header) + "\n")
        while not desk.done:
            observation = desk.observation()
            response = None
            error = None
            reply = None
            started = time.monotonic()
            if calls >= spec.max_calls:
                error = "call_budget_exhausted"
            elif consecutive_errors >= 3:
                error = "agent_circuit_open"
            else:
                calls += 1
                try:
                    # Serialization enforces the public JSON boundary for in-process adapters too.
                    reply = adapter.act(json.loads(canonical_json(observation)))
                    response, error = reply.dispatch, reply.error
                except AgentFailure as exc:
                    error = str(exc)
            elapsed = time.monotonic() - started
            latency_s += elapsed
            if reply is not None:
                if reply.model:
                    returned_models.add(reply.model)
                for key in usage:
                    usage[key] += reply.usage.get(key) or 0
                usage_unknown += int(any(reply.usage.get(key) is None for key in usage))
            elif error not in {"call_budget_exhausted", "agent_circuit_open"}:
                usage_unknown += 1
            receipt = desk.step(response, failure=error)
            consecutive_errors = consecutive_errors + 1 if not receipt["valid"] else 0
            failures += int(not receipt["valid"])
            entry = {
                "type": "decision",
                "observation": observation,
                "observation_sha256": digest(observation),
                "response": response,
                "adapter_error": error,
                "receipt": receipt,
                "latency_s": elapsed,
                "usage": reply.usage if reply else None,
                "response_id": reply.response_id if reply else None,
                "model": reply.model if reply else None,
                "adapter_metadata": reply.metadata if reply else None,
                "metrics": desk.metrics(),
            }
            trace.write(canonical_json(entry) + "\n")
            trace.flush()
        result = {
            "policy": spec.name,
            "provider": spec.provider,
            "requested_model": spec.model,
            "returned_models": sorted(returned_models),
            "case_id": case.case_id,
            "capability": case.capability,
            "seed": seed,
            **desk.metrics(),
            "agent_calls": calls,
            "failed_decisions": failures,
            "eligible": failures == 0 and desk.sim.state.blocked_actions == 0,
            "agent_latency_s": latency_s,
            "tokens_reported": usage,
            "calls_with_unknown_usage": usage_unknown,
            "trace": trace_path.name,
            "scenario_sha256": digest(asdict(case.scenario)),
        }
        trace.write(canonical_json({"type": "result", "result": result}) + "\n")
    result["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    return result


def summarize(records: list[dict[str, Any]], specs: list[AgentSpec], baseline: str) -> dict[str, Any]:
    baseline_rows = {(row["case_id"], row["seed"]): row for row in records if row["policy"] == baseline}
    leaderboard = []
    comparisons = []
    for spec in specs:
        rows = [row for row in records if row["policy"] == spec.name]
        if not rows:
            continue
        normalized = []
        for row in rows:
            ref = baseline_rows.get((row["case_id"], row["seed"]))
            if ref and ref["loss_growth"] > 1e-9:
                normalized.append(100.0 * (ref["loss_growth"] - row["loss_growth"]) / ref["loss_growth"])
        leaderboard.append(
            {
                "policy": spec.name,
                "provider": spec.provider,
                "episodes": len(rows),
                "eligible": all(row["eligible"] for row in rows),
                "mean_loss_growth": float(np.mean([row["loss_growth"] for row in rows])),
                "mean_burned_area_ha": float(np.mean([row["burned_area_ha"] for row in rows])),
                "mean_resource_cost_units": float(np.mean([row["resource_cost_units"] for row in rows])),
                "mean_loss_reduction_pct": float(np.mean(normalized)) if normalized else None,
                "normalized_pairs": len(normalized),
                "failed_decisions": sum(row["failed_decisions"] for row in rows),
                "decisions": sum(row["decisions"] for row in rows),
                "agent_latency_s": sum(row["agent_latency_s"] for row in rows),
                "input_tokens_reported": sum(row["tokens_reported"]["input_tokens"] for row in rows),
                "output_tokens_reported": sum(row["tokens_reported"]["output_tokens"] for row in rows),
                "calls_with_unknown_usage": sum(row["calls_with_unknown_usage"] for row in rows),
            }
        )
        if spec.name != baseline and any((row["case_id"], row["seed"]) in baseline_rows for row in rows):
            comparison = paired_policy_summary(
                records,
                candidate_policy=spec.name,
                baseline_policy=baseline,
                metric="loss_growth",
                lower_is_better=True,
                bootstrap_samples=2000,
                seed=0,
            )
            comparison["eligible"] = all(row["eligible"] for row in rows) and all(
                row["eligible"] for row in baseline_rows.values()
            )
            comparison["interval_is_descriptive"] = True
            comparisons.append(comparison)
    return {
        "baseline": baseline,
        "leaderboard": sorted(
            leaderboard, key=lambda row: (not row["eligible"], row["mean_loss_growth"], row["policy"])
        ),
        "comparisons": comparisons,
    }


def run_benchmark(
    suite_path: str | Path,
    specs: list[AgentSpec],
    out: Path,
    *,
    progress: Callable[[str], None] = print,
    workers: int = 1,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 to 8")
    raw, cases = load_suite(suite_path)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("agent names must be unique")
    hold_specs = [spec for spec in specs if spec.provider == "baseline" and spec.model == "hold"]
    if len(hold_specs) != 1:
        raise ValueError("include exactly one hold baseline for paired scoring")
    # Fail configuration/credentials checks before creating a run or charging a model.
    adapters = {spec.name: AgentAdapter(spec) for spec in specs}
    if out.exists():
        raise ValueError("output directory already exists; choose a fresh run directory")
    out.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_VERSION,
        "suite": raw,
        "suite_sha256": digest(raw),
        "source": source_identity(),
        "system_prompt": SYSTEM_PROMPT,
        "prompt_sha256": digest(SYSTEM_PROMPT),
        "dispatch_schema": DISPATCH_SCHEMA,
        "agents": [spec.public_dict() for spec in specs],
        "adapter_runtime": {
            name: adapter.cli.identity if adapter.cli is not None else None
            for name, adapter in adapters.items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_episodes": sum(len(case.seeds) for case in cases) * len(specs),
        "episode_workers": workers,
        "limitations": [
            "Public synthetic benchmark; no human operator comparator or real-world efficacy claim.",
            "Same initial seed is paired; policy-dependent RNG consumption can diverge after dispatch.",
            "Bootstrap intervals describe these scenario clusters, not broad frontier-model capability.",
            "Failures advance with hold; failed runs remain visible but are ineligible for ranking.",
            "Resource costs and asset-weighted loss are simulator units, not dollars or lives.",
            "HTTP timeouts are socket timeouts; CLI timeouts are wall-clock subprocess limits.",
            "CLI output-token caps are unavailable (null); calls and wall time are bounded.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    jobs = [(case, seed, spec) for case in cases for seed in case.seeds for spec in specs]
    completed: dict[int, dict[str, Any]] = {}

    def execute(index: int) -> dict[str, Any]:
        case, seed, spec = jobs[index]
        progress(f"[start {index + 1}/{len(jobs)}] {spec.name} / {case.case_id} / {seed}")
        return run_episode(case, seed, spec, adapters[spec.name], out / f"episode-{index:04d}.jsonl")

    # Episode state and CLI sessions are independent. Only this main thread writes the summary.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, index): index for index in range(len(jobs))}
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            records = [completed[key] for key in sorted(completed)]
            progress(f"[complete {len(records)}/{len(jobs)}] {completed[index]['policy']}")
            summary = {
                "schema_version": 1,
                "suite_id": raw["suite_id"],
                "suite_sha256": digest(raw),
                "split": raw["split"],
                "complete": len(records) == manifest["expected_episodes"],
                "expected_episodes": manifest["expected_episodes"],
                "records": records,
                **summarize(records, specs, hold_specs[0].name),
            }
            write_json(out / "summary.json", summary)
    return summary


def replay_trace(path: str | Path) -> dict[str, Any]:
    """Re-execute recorded dispatches without calling a model; fail on observation/score drift."""
    entries = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not entries or entries[0].get("protocol") != PROTOCOL_VERSION or entries[-1].get("type") != "result":
        raise ValueError("trace is incomplete or has an unsupported protocol")
    header = entries[0]
    desk = IncidentDesk(_scenario(header["scenario"]), header["seed"])
    for index, entry in enumerate(entries[1:-1]):
        if entry.get("type") != "decision" or digest(desk.observation()) != entry["observation_sha256"]:
            raise ValueError(f"observation drift at decision {index}")
        if digest(entry["observation"]) != entry["observation_sha256"]:
            raise ValueError(f"transcript observation checksum mismatch at decision {index}")
        receipt = desk.step(entry["response"], failure=entry["adapter_error"])
        if digest(receipt) != digest(entry["receipt"]) or digest(desk.metrics()) != digest(entry["metrics"]):
            raise ValueError(f"transition drift at decision {index}")
    if not desk.done:
        raise ValueError("trace ends before the episode completes")
    for key, value in desk.metrics().items():
        if value != entries[-1]["result"].get(key):
            raise ValueError(f"terminal metric drift: {key}")
    return {"verified": True, "decisions": desk.decision, "metrics": desk.metrics()}
