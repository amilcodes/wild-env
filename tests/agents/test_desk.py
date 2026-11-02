from __future__ import annotations

import copy
import json

import pytest

from aeolus.agents.adapters import BASELINES, baseline_dispatch
from aeolus.agents.contract import canonical_json, parse_json, validate_dispatch
from aeolus.agents.desk import IncidentDesk
from aeolus.config import ResourceSpec, ScenarioConfig
from aeolus.core.tasks import generate_tasks


@pytest.fixture
def desk():
    return IncidentDesk(
        ScenarioConfig(
            width=32, height=32, horizon_min=12, max_tasks=24, spotting_rate=0.0, terminate_on_escape=False
        ),
        17,
    )


def test_hidden_fire_and_evaluator_scores_do_not_change_observation(desk):
    before = canonical_json(desk.observation())
    truth = desk.sim.state.truth
    truth.phase[:] = 2
    truth.intensity_kw_m[:] = 9999
    truth.observed_burned[:] = 1
    truth.fuel_remaining[:] = 0
    truth.arrival_time_min[:] = 123
    desk.total_reward = -9999
    desk.sim.state.cumulative_exposure = 9999
    desk.sim.tasks = generate_tasks(desk.sim)
    assert canonical_json(desk.observation()) == before
    assert desk.metrics()["weighted_loss"] > 0


def test_receipt_does_not_leak_evaluator_metrics(desk):
    observation = desk.observation()
    desk.step(baseline_dispatch(observation, "greedy"))
    packet = desk.observation()
    text = canonical_json(packet)
    for forbidden in (
        "weighted_loss",
        "reward_return",
        "rng_state",
        "escaped",
        "contained",
        "exposure_units",
    ):
        assert forbidden not in text
    assert packet["last_dispatch"]["valid"]
    assert packet["minute"] == 3
    assert any(sector["mean_observation_age_min"] is None for sector in packet["sectors"])


def test_observation_is_detached_and_json_serializable(desk):
    packet = json.loads(canonical_json(desk.observation()))
    packet["resources"][0]["payload_l"] = 0
    packet["tasks"].clear()
    assert desk.observation()["resources"][0]["payload_l"] > 0
    assert len(desk.observation()["tasks"]) > 1


@pytest.mark.parametrize("policy", BASELINES)
def test_baselines_use_legal_capacity_aware_dispatches(desk, policy):
    while not desk.done:
        packet = desk.observation()
        response = baseline_dispatch(packet, policy)
        validate_dispatch(response, packet)
        assert desk.step(response)["valid"]
    assert desk.metrics()["invalid_dispatches"] == 0
    with pytest.raises(RuntimeError, match="complete"):
        desk.step(response)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response: response.update(decision_id="stale"),
        lambda response: response["assignments"].pop(),
        lambda response: response["assignments"][0].update(task_id=True),
        lambda response: response["assignments"][0].update(task_id=1.0),
        lambda response: response["assignments"][0].update(task_id=-1),
        lambda response: response["assignments"][0].update(resource_id="unknown"),
        lambda response: response["assignments"][0].update(extra="field"),
        lambda response: response.update(extra=True),
        lambda response: response.update(memory="x" * 2001),
        lambda response: response.update(rationale=3),
        lambda response: response.update(assignments=None),
    ],
)
def test_invalid_dispatch_is_atomic_and_consumes_interval(desk, mutation):
    response = baseline_dispatch(desk.observation(), "greedy")
    mutation(response)
    receipt = desk.step(response)
    assert not receipt["valid"]
    assert set(receipt["actions"].values()) == {0}
    assert desk.sim.state.minute == 3
    assert desk.metrics()["invalid_dispatches"] == 1
    assert not any(event["kind"] == "assignment" for event in desk.sim.state.events)


def test_duplicate_resource_and_joint_capacity_rejected(desk):
    response = baseline_dispatch(desk.observation(), "hold")
    response["assignments"][1] = copy.deepcopy(response["assignments"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_dispatch(response, desk.observation())
    fleet = (
        ResourceSpec("water_a", "water", 45, 2000, 5, 1, 120),
        ResourceSpec("water_b", "water", 45, 2000, 5, 1, 120),
    )
    pair = IncidentDesk(ScenarioConfig(width=32, height=32, resources=fleet), 17)
    packet = pair.observation()
    task_id = next(task["id"] for task in packet["tasks"] if task["kind"] == "WATER")
    response = baseline_dispatch(packet, "hold")
    for assignment in response["assignments"]:
        assignment["task_id"] = task_id
    with pytest.raises(ValueError, match="capacity"):
        validate_dispatch(response, packet)


def test_memory_persists_only_after_valid_dispatch(desk):
    response = baseline_dispatch(desk.observation(), "hold")
    response["memory"] = "Recheck the eastern sector."
    desk.step(response)
    assert desk.observation()["memory"] == response["memory"]
    response["memory"] = "Should not be saved: stale decision ID."
    desk.step(response)
    assert desk.observation()["memory"] == "Recheck the eastern sector."


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "```json\n{}\n```"])
def test_strict_json_rejects_ambiguous_or_repaired_output(raw):
    with pytest.raises(ValueError):
        parse_json(raw)
