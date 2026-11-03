import json
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus.agents.contract import canonical_json, digest
from aeolus.agents.operations import (
    Instance,
    Job,
    OperationsDesk,
    Planner,
    dispatch,
    generate,
    greedy_action,
    initial_state,
    payoff,
    with_forecast,
)


def play(instance, world, policy):
    desk, planner = OperationsDesk(instance, world), Planner(instance)
    regret = 0.0
    while not desk.done:
        before = desk.state
        action = planner.action(before) if policy == "optimal" else greedy_action(instance, before)
        desk.step(dispatch(instance, before, desk.observation(), action))
        regret += planner.regret(before, action)
    return payoff(instance, desk.state, world), regret


def test_exact_planner_matches_manually_solved_opportunity_cost():
    jobs = (
        Job(11, "long", "PROTECT", (3,), 0, 3, (), 0, 0, (30, 30)),
        Job(12, "early", "PROTECT", (1,), 0, 1, (), 0, 0, (18, 18)),
        Job(13, "later", "PROTECT", (2,), 1, 3, (), 0, 0, (20, 20)),
    )
    instance = Instance("manual", 1, 1, 3, ("team",), 0, 3, jobs)
    planner = Planner(instance)
    assert planner.value(initial_state(instance)) == 38
    assert instance.jobs[planner.action(initial_state(instance))[0]].id == 12
    assert play(instance, 0, "optimal") == (38, 0)


def test_information_matched_optimum_requires_pay_for_observation():
    jobs = (
        Job(11, "survey", "SURVEY", (1,), 0, 1, (), 0, 3, (0, 0)),
        Job(12, "alpha", "PROTECT", (1,), 0, 2, (), 1, 0, (100, 0)),
        Job(13, "beta", "PROTECT", (1,), 0, 2, (), 1, 0, (0, 100)),
    )
    instance = Instance("manual", 1, 1, 2, ("team",), 1, 2, jobs)
    planner = Planner(instance)
    assert planner.value(initial_state(instance)) == 97
    assert planner.value(with_forecast(initial_state(instance), 0)) == 100
    assert planner.action(initial_state(instance)) == (0,)
    assert play(instance, 0, "optimal") == (97, 0)
    assert play(instance, 1, "optimal") == (97, 0)


@pytest.mark.parametrize(
    "family", ("deadline", "prerequisite", "water", "information", "forecast", "compound")
)
def test_bellman_regret_equals_expected_reward_shortfall(family):
    instance = generate(family, 29)
    planner = Planner(instance)
    optimum = planner.value(initial_state(instance))
    for policy in ("optimal", "greedy"):
        outcomes = [play(instance, w, policy) for w in (0, 1)]
        reward = sum(x[0] for x in outcomes) / 2
        regret = sum(x[1] for x in outcomes) / 2
        assert regret == pytest.approx(optimum - reward)
        if policy == "optimal":
            assert regret == pytest.approx(0)


def test_hidden_forecast_never_changes_packet_before_reveal():
    instance = generate("information", 17)
    alpha, beta = OperationsDesk(instance, 0), OperationsDesk(instance, 1)
    assert canonical_json(alpha.observation()) == canonical_json(beta.observation())
    alpha.hidden_forecast = 1
    assert canonical_json(alpha.observation()) == canonical_json(beta.observation())
    alpha.hidden_forecast = 0
    survey = next(j for j, job in enumerate(instance.jobs) if job.kind == "SURVEY")
    action = (survey, -1)
    for desk in (alpha, beta):
        desk.step(dispatch(instance, desk.state, desk.observation(), action))
    assert alpha.observation()["forecast"] == "alpha"
    assert beta.observation()["forecast"] == "beta"
    assert "net_reward" not in canonical_json(alpha.observation())


def test_joint_water_failure_is_atomic_and_advances_time():
    jobs = (
        Job(11, "a", "PROTECT", (1, 1), 0, 2, (), 2, 1, (10, 10)),
        Job(12, "b", "PROTECT", (1, 1), 0, 2, (), 2, 1, (10, 10)),
    )
    instance = Instance("manual", 1, 1, 2, ("a", "b"), 2, 2, jobs)
    desk = OperationsDesk(instance, 0)
    _, receipt = desk.step(dispatch(instance, desk.state, desk.observation(), (0, 1)))
    assert not receipt["valid"] and desk.state.tick == 1
    assert desk.state.water == 2 and desk.state.cost == 0 and desk.state.completed == 0
    assert set(receipt["actions"].values()) == {0}


def test_deadline_and_prerequisite_are_checked_at_start():
    jobs = (
        Job(11, "access", "ACCESS", (1,), 0, 1, (), 0, 1, (0, 0)),
        Job(12, "protect", "PROTECT", (1,), 0, 2, (11,), 0, 1, (20, 20)),
    )
    instance = Instance("manual", 1, 1, 2, ("team",), 0, 2, jobs)
    desk = OperationsDesk(instance, 0)
    assert 12 not in desk.observation()["resources"][0]["legal_tasks"]
    desk.step(dispatch(instance, desk.state, desk.observation(), (0,)))
    assert 12 in desk.observation()["resources"][0]["legal_tasks"]
    desk.step(dispatch(instance, desk.state, desk.observation(), (1,)))
    assert desk.done and payoff(instance, desk.state, 0) == 18


def test_relabeling_does_not_change_reference_value():
    instance = generate("compound", 17)
    renamed = replace(
        instance,
        jobs=tuple(
            replace(j, id=j.id + 300, label="renamed", prerequisites=tuple(p + 300 for p in j.prerequisites))
            for j in instance.jobs
        ),
    )
    assert Planner(instance).value(initial_state(instance)) == Planner(renamed).value(initial_state(renamed))


def test_solver_budget_fails_instead_of_claiming_optimality():
    instance = generate("compound", 17)
    with pytest.raises(RuntimeError, match="state budget"):
        Planner(instance, max_states=2).value(initial_state(instance))


def test_challenge_extension_preserves_every_frozen_level_one_and_two_instance():
    manifest = json.loads(Path("results/agent_benchmark/operations_v2_evaluation_manifest.json").read_text())
    for original in manifest["instances"]:
        assert generate(original["family"], original["seed"], original["level"]).identity == digest(original)


def test_challenge_has_a_deeper_dependency_join_and_preserves_hidden_forecast_boundary():
    instance = generate("compound", 2042, 3)
    jobs = {j.id: j for j in instance.jobs}

    def depth(task):
        return 1 + max((depth(p) for p in jobs[task].prerequisites), default=0)

    assert len(jobs) == 16 and max(depth(j) for j in jobs) == 4
    assert any(len(j.prerequisites) == 2 for j in jobs.values())
    assert canonical_json(OperationsDesk(instance, 0).observation()) == canonical_json(
        OperationsDesk(instance, 1).observation()
    )
    with pytest.raises(ValueError, match="only the compound"):
        generate("deadline", 2042, 3)


def test_run_replays_and_rejects_edited_summary(tmp_path):
    from aeolus.agents.operations_benchmark import analyze, run

    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "schema_version: 2\nsuite_id: test\nsplit: development\nrepeats: 2\n"
        "cases:\n  - {family: information, seed: 17, level: 1}\n"
    )
    out = tmp_path / "run"
    summary = run(suite, None, out, workers=2)
    assert summary["complete"] and len(summary["records"]) == 10
    audit = analyze(out, bootstrap_samples=100)
    assert audit["audit"]["all_replays_verified"]
    optimal = next(row for row in summary["leaderboard"] if row["policy"] == "optimal")
    assert optimal["mean_normalized_regret_pct"] == 0
    summary["records"][0]["net_reward"] += 1
    (out / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary differs"):
        analyze(out, bootstrap_samples=100)
