from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aeolus.agents.adapters import AgentAdapter, AgentReply, AgentSpec
from aeolus.agents.benchmark import Case, load_suite, replay_trace, run_benchmark, run_episode
from aeolus.agents.contract import canonical_json
from aeolus.agents.operations_benchmark import load_suite as load_operations_suite
from aeolus.config import ScenarioConfig


def small_suite(tmp_path):
    value = {
        "schema_version": 1,
        "suite_id": "test-suite",
        "split": "development",
        "description": "Test fixture",
        "scenario": {"width": 32, "height": 32, "horizon_min": 6, "max_tasks": 24, "spotting_rate": 0.0},
        "cases": [{"id": "attack", "capability": "dispatch", "seeds": [17], "overrides": {}}],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(value))
    return path, value


def test_complete_run_replays_and_refuses_overwrite(tmp_path):
    suite, _ = small_suite(tmp_path)
    specs = [AgentSpec("hold"), AgentSpec("greedy", model="greedy")]
    out = tmp_path / "run"
    result = run_benchmark(suite, specs, out, progress=lambda message: None)
    assert result["complete"] and len(result["records"]) == 2
    assert all(row["eligible"] for row in result["records"])
    assert result["comparisons"][0]["paired_records"] == 1
    for row in result["records"]:
        replay = replay_trace(out / row["trace"])
        assert replay["verified"] and replay["decisions"] == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source"]["source_sha256"]
    assert manifest["prompt_sha256"]
    assert not list(out.glob("*.html"))
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark(suite, specs, out)


def test_replay_detects_tampered_observation_and_metrics(tmp_path):
    suite, _ = small_suite(tmp_path)
    out = tmp_path / "run"
    run_benchmark(suite, [AgentSpec("hold")], out, progress=lambda message: None)
    trace = out / "episode-0000.jsonl"
    original = trace.read_text()
    for field in ("observation", "metrics"):
        entries = [json.loads(line) for line in original.splitlines()]
        entries[1][field]["minute"] = 999
        trace.write_text("\n".join(canonical_json(entry) for entry in entries))
        with pytest.raises(ValueError, match="drift|checksum"):
            replay_trace(trace)
    trace.write_text("\n".join(original.splitlines()[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        replay_trace(trace)


def test_budget_and_bad_agent_do_not_disappear_from_scores(tmp_path):
    case = Case("test", "dispatch", (17,), ScenarioConfig(width=32, height=32, horizon_min=12, max_tasks=24))
    spec = AgentSpec("limited", max_calls=1)
    result = run_episode(case, 17, spec, AgentAdapter(spec), tmp_path / "budget.jsonl")
    assert result["agent_calls"] == 1
    assert result["failed_decisions"] == 3 and not result["eligible"]
    assert replay_trace(tmp_path / "budget.jsonl")["verified"]

    class InvalidAgent:
        def act(self, observation):
            return AgentReply({"not": "a dispatch"})

    case = replace(case, scenario=replace(case.scenario, horizon_min=15))
    result = run_episode(case, 17, AgentSpec("bad"), InvalidAgent(), tmp_path / "invalid.jsonl")
    assert result["agent_calls"] == 3  # Circuit opens, but the incident still finishes.
    assert result["failed_decisions"] == 5 and result["minute"] == 15
    assert replay_trace(tmp_path / "invalid.jsonl")["verified"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["cases"][0].update(seeds=[1, 1]),
        lambda raw: raw["cases"][0].update(seeds=[True]),
        lambda raw: raw["cases"].append(raw["cases"][0]),
        lambda raw: raw.update(cases=[]),
        lambda raw: raw["scenario"].update(horizon_min=7),
        lambda raw: raw["scenario"].update(weather_forcing="unhashed.csv"),
        lambda raw: raw["scenario"].update(wind_speed_m_s=float("nan")),
        lambda raw: raw["scenario"].update(unknown_setting=1),
    ],
)
def test_invalid_suite_rejected(tmp_path, mutation):
    suite, raw = small_suite(tmp_path)
    mutation(raw)
    suite.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        load_suite(suite)


def test_shipped_suites_are_valid():
    for path in Path("configs/agents").glob("*.yaml"):
        if ".example." not in path.name:
            raw = yaml.safe_load(path.read_text())
            if "agents" in raw:
                assert set(raw) == {"agents"}
                assert [AgentSpec(**spec) for spec in raw["agents"]]
            else:
                loader = load_operations_suite if raw.get("schema_version") == 2 else load_suite
                _, cases = loader(path)
                assert cases


def test_parallel_episodes_match_serial_metrics_and_stable_trace_order(tmp_path):
    suite, _ = small_suite(tmp_path)
    specs = [AgentSpec("hold"), AgentSpec("greedy", model="greedy")]
    serial = run_benchmark(suite, specs, tmp_path / "serial", progress=lambda _: None)
    parallel = run_benchmark(suite, specs, tmp_path / "parallel", progress=lambda _: None, workers=2)
    for a, b in zip(serial["records"], parallel["records"]):
        for key in ("policy", "seed", "loss_growth", "burned_area_ha", "decisions", "trace", "eligible"):
            assert a[key] == b[key]
        assert replay_trace(tmp_path / "parallel" / b["trace"])["verified"]


def test_partial_summary_waits_for_a_matching_baseline(tmp_path):
    from aeolus.agents.benchmark import summarize

    suite, _ = small_suite(tmp_path)
    specs = [AgentSpec("hold"), AgentSpec("greedy", model="greedy")]
    full = run_benchmark(suite, specs, tmp_path / "full", progress=lambda _: None)
    hold, candidate = full["records"]
    candidate = {**candidate, "case_id": "other-case"}
    partial = summarize([hold, candidate], specs, "hold")
    assert partial["comparisons"] == []
