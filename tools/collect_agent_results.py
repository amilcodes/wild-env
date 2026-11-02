"""Combine completed runs without changing trace bytes or selecting successful seeds.

Run from a checkout containing all recorded source commits. Only an adapter-only
source revision is permitted; the environment, contract and evaluation must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aeolus.agents.adapters import AgentSpec
from aeolus.agents.benchmark import summarize, write_json

ADAPTER_FILES = {"src/aeolus/agents/cli_adapters.py", "tests/agents/test_cli_adapters.py"}
VOLATILE_RECORD_FIELDS = {"agent_latency_s", "trace", "trace_sha256"}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(sources, output):
    if output.exists():
        raise ValueError("choose a new output directory")
    manifests, summaries, checks = {}, {}, {}
    for label, root in sources.items():
        if not label.replace("-", "").isalnum():
            raise ValueError("source labels must be alphanumeric with optional hyphens")
        if int((root / "exit_code").read_text()) not in (0, 2):
            raise ValueError(f"{label}: runner did not finish normally")
        run = root / "benchmark"
        manifests[label] = read_json(run / "manifest.json")
        summaries[label] = read_json(run / "summary.json")
        checks[label] = {x["trace"]: x for x in read_json(run / "replay-verification.json")}
        summary = summaries[label]
        if not summary["complete"] or len(summary["records"]) != summary["expected_episodes"]:
            raise ValueError(f"{label}: incomplete summary")
        if len(checks[label]) != len(summary["records"]):
            raise ValueError(f"{label}: incomplete replay verification")
        if manifests[label]["source"]["git_dirty"]:
            raise ValueError(f"{label}: source was dirty")
    first = next(iter(manifests.values()))
    differences = {}
    for label, manifest in manifests.items():
        for key in ("protocol", "suite", "suite_sha256", "system_prompt", "prompt_sha256", "dispatch_schema"):
            if manifest[key] != first[key]:
                raise ValueError(f"{label}: different {key}")
        for key in ("python", "numpy", "platform", "dependencies"):
            if manifest["source"][key] != first["source"][key]:
                raise ValueError(f"{label}: different numerical runtime {key}")
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", first["source"]["git_commit"], manifest["source"]["git_commit"]],
            text=True,
        ).splitlines()
        if set(changed) - ADAPTER_FILES:
            raise ValueError(f"{label}: source changes outside CLI adapter/tests: {changed}")
        differences[label] = changed

    agents, runtime, records, replays = {}, {}, [], []
    seen, duplicates, catalog = {}, [], {}
    # Validate everything before creating output. Duplicates may only be deterministic baselines.
    for label, root in sources.items():
        manifest = manifests[label]
        for spec in manifest["agents"]:
            name = spec["name"]
            if name in agents and agents[name] != spec:
                raise ValueError(f"conflicting configuration for {name}")
            agents[name] = spec
            value = manifest["adapter_runtime"][name]
            if name in runtime and runtime[name] != value:
                raise ValueError(f"conflicting runtime for {name}")
            runtime[name] = value
        catalog[label] = {
            "run_name": root.name,
            "source": manifest["source"],
            "exit_code": int((root / "exit_code").read_text()),
            "manifest_sha256": sha256(root / "benchmark/manifest.json"),
            "summary_sha256": sha256(root / "benchmark/summary.json"),
            "episodes": len(summaries[label]["records"]),
        }
        for original in summaries[label]["records"]:
            path = root / "benchmark" / original["trace"]
            replay = checks[label][original["trace"]]
            if sha256(path) != original["trace_sha256"] or not replay["verified"]:
                raise ValueError(f"{label}/{path.name}: checksum or replay failure")
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            recorded = {k: v for k, v in original.items() if k != "trace_sha256"}
            if entries[-1]["result"] != recorded:
                raise ValueError(f"{label}/{path.name}: summary differs from trace result")
            key = (original["policy"], original["case_id"], original["seed"])
            stable = {k: v for k, v in original.items() if k not in VOLATILE_RECORD_FIELDS}
            observations = [x["observation_sha256"] for x in entries[1:-1]]
            if key in seen:
                old = seen[key]
                if (
                    original["provider"] != "baseline"
                    or stable != old["stable"]
                    or observations != old["observations"]
                ):
                    raise ValueError(f"cannot deduplicate differing or stochastic episode {key}")
                duplicates.append(
                    {
                        "key": list(key),
                        "retained_source": old["source"],
                        "duplicate_source": label,
                        "duplicate_trace": path.name,
                        "duplicate_trace_sha256": original["trace_sha256"],
                        "all_metrics_and_observations_match": True,
                    }
                )
                continue
            seen[key] = {"stable": stable, "observations": observations, "source": label}
            name = f"{label}-{path.name}"
            records.append({**original, "trace": name, "source_run": label, "source_trace": path.name})
            replays.append({**replay, "trace": name, "source_run": label})
    effective_prompts = {x["effective_system_sha256"] for x in runtime.values() if x}
    if len(effective_prompts) > 1:
        raise ValueError("CLI policies used different effective task prompts")
    conditions = {(x["case_id"], x["seed"]) for x in records}
    planned_conditions = {
        (case["id"], seed) for case in first["suite"]["cases"] for seed in case["seeds"]
    }
    if conditions != planned_conditions:
        raise ValueError("recorded conditions differ from the frozen suite")
    for name in agents:
        if {(x["case_id"], x["seed"]) for x in records if x["policy"] == name} != conditions:
            raise ValueError(f"missing conditions for {name}")

    run = output / "benchmark"
    run.mkdir(parents=True)
    for label, root in sources.items():
        dest = output / "source-runs" / label
        dest.mkdir(parents=True)
        for name in ("manifest.json", "summary.json", "replay-verification.json"):
            shutil.copyfile(root / "benchmark" / name, dest / name)
        for name in ("exit_code", "tests.txt", "lint.txt", "progress.txt"):
            if (root / name).exists():
                shutil.copyfile(root / name, dest / name)
    for record in records:
        shutil.copyfile(
            sources[record["source_run"]] / "benchmark" / record["source_trace"], run / record["trace"]
        )
    summary = {
        **{
            k: v
            for k, v in next(iter(summaries.values())).items()
            if k not in ("records", "leaderboard", "comparisons")
        },
        "complete": True,
        "expected_episodes": len(records),
        "records": records,
        **summarize(records, [AgentSpec(**x) for x in agents.values()], "hold"),
    }
    manifest = {
        **first,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "combined_completed_runs",
        "agents": list(agents.values()),
        "adapter_runtime": runtime,
        "expected_episodes": len(records),
        "episode_workers": None,
        "source": {
            "source_runs": catalog,
            "environment_reference_commit": first["source"]["git_commit"],
            "source_changes_from_reference": differences,
            "environment_contract_and_scoring_unchanged": True,
        },
        "aggregation": {
            "trace_bytes_unchanged": True,
            "all_unique_episodes_retained": True,
            "duplicate_baselines": duplicates,
            "replays": (
                "Original source runners replayed each trace; copied trace bytes were checksum-verified."
            ),
            "collector_sha256": sha256(Path(__file__)),
        },
    }
    write_json(run / "manifest.json", manifest)
    write_json(run / "summary.json", summary)
    write_json(run / "replay-verification.json", replays)
    print(
        json.dumps({"episodes": len(records), "duplicates_verified": len(duplicates), "output": str(output)})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="label=run-directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = [value.split("=", 1) for value in args.source]
    if len({x[0] for x in pairs}) != len(pairs):
        raise ValueError("source labels must be unique")
    collect({label: Path(path) for label, path in pairs}, args.out)
