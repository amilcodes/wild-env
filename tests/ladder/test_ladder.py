import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from aeolus.agents.operations import FAMILIES, Planner, initial_state
from aeolus.agents.operations import generate as generate_v2
from aeolus.ladder import TIERS, evaluate_plan, generate, solve
from aeolus.ladder.baselines import BASELINES, baseline_plan
from aeolus.ladder.env import LadderEnv
from aeolus.ladder.plan import PLAN_SCHEMA, WORLDS
from aeolus.ladder.prompt import observation
from aeolus.ladder.stats import holm, pairwise, sign_flip_p


def test_generator_is_deterministic_and_seed_sensitive():
    for tier in TIERS:
        a, b, c = generate(tier, 7), generate(tier, 7), generate(tier, 8)
        assert a.identity == b.identity
        assert a.identity != c.identity
        spec = TIERS[tier]
        assert len(a.resources) == spec.resources
        if spec.overload:  # packing: free-standing work exceeds team capacity
            work = sum(min(d for d in j.durations if d > 0) for j in a.jobs if not j.prerequisites)
            assert work >= spec.overload * spec.resources * spec.horizon
        else:
            assert len(a.jobs) == spec.protect + spec.access + spec.surveys
        assert all(any(d > 0 for d in job.durations) for job in a.jobs)


@pytest.mark.parametrize("family", FAMILIES)
def test_mip_matches_v2_bellman_planner(family):
    for level in (1, 2):
        instance = generate_v2(family, 11, level)
        assert solve(instance).value == pytest.approx(
            Planner(instance).value(initial_state(instance)), abs=1e-9
        )


@pytest.mark.parametrize("tier,seed", [(1, 0), (1, 3), (2, 1), (2, 2)])
def test_mip_matches_bellman_on_ladder_instances(tier, seed):
    instance = generate(tier, seed)
    expected = Planner(instance, max_states=2_000_000).value(initial_state(instance))
    assert solve(instance).value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_reference_ordering_and_exact_replay(tier):
    instance = generate(tier, 1)
    info = solve(instance)  # solve() itself asserts the optimal plan replays to the MIP value
    clairvoyant = solve(instance, "clairvoyant").value
    open_loop = solve(instance, "open_loop").value
    assert open_loop <= info.value + 1e-9 <= clairvoyant + 2e-9
    outcome = evaluate_plan(instance, info.plan)
    assert outcome.violations == []
    assert outcome.expected_value == pytest.approx(info.value)


@pytest.mark.parametrize("tier", [2, 4, 6])
def test_no_admissible_plan_beats_the_optimum(tier):
    instance = generate(tier, 5)
    optimum = solve(instance).value
    rng = random.Random(0)
    for _ in range(300):
        plan = []
        for _ in range(rng.randint(1, 3 * len(instance.jobs))):
            job = rng.choice(instance.jobs)
            plan.append(
                {
                    "resource": rng.choice(instance.resources),
                    "job": job.id,
                    "start": rng.randint(0, instance.horizon),
                    "branch": rng.choice(["both", "both", *WORLDS]),
                }
            )
        assert evaluate_plan(instance, {"rationale": "", "plan": plan}).expected_value <= optimum + 1e-9


def test_world_specific_entries_before_reveal_are_rejected():
    instance = generate(3, 2)
    info = solve(instance)
    leaked = [dict(e) for e in info.plan["plan"]]
    early = [e for e in leaked if e["branch"] == "both"]
    assert early, "expected shared entries before the reveal"
    early[0]["branch"] = "alpha"
    outcome = evaluate_plan(instance, {"rationale": "", "plan": leaked})
    assert any(v["reason"] == "forecast-specific entry before reveal" for v in outcome.violations)


def test_malformed_plans_are_scored_not_repaired():
    instance = generate(2, 0)
    assert evaluate_plan(instance, None).parse_error
    assert evaluate_plan(instance, {"plan": "x"}).expected_value == 0
    bad = {"rationale": "", "plan": [{"resource": "nope", "job": 1, "start": 0, "branch": "both"}]}
    outcome = evaluate_plan(instance, bad)
    assert outcome.expected_value == 0 and outcome.violations[0]["reason"] == "unknown resource"


@pytest.mark.parametrize("tier", [1, 3, 5, 7])
def test_baselines_are_admissible_and_bounded(tier):
    instance = generate(tier, 4)
    optimum = solve(instance).value
    for name in BASELINES:
        outcome = evaluate_plan(instance, baseline_plan(instance, name))
        assert outcome.violations == [], name
        assert outcome.expected_value <= optimum + 1e-9


def test_observation_hides_reference_and_world():
    instance = generate(4, 3)
    text = json.dumps(observation(instance))
    for forbidden in ("optimum", "optimal", "hidden", "actual_world", "regret"):
        assert forbidden not in text
    assert set(PLAN_SCHEMA["required"]) == {"rationale", "plan"}


def _plan_actions(env, plan, world):
    """Per-tick actions that execute the optimal plan's starts in one world."""
    instance = env.instance
    slot = {job.id: j for j, job in enumerate(instance.jobs)}
    names = {name: r for r, name in enumerate(instance.resources)}
    by_tick = {}
    for e in plan["plan"]:
        if e["branch"] in ("both", WORLDS[world]):
            by_tick.setdefault(e["start"], {})[names[e["resource"]]] = slot[e["job"]] + 1
    return by_tick


@pytest.mark.parametrize("tier", [1, 3, 6])
def test_env_return_matches_plan_value_and_never_leaks(tier):
    instance = generate(tier, 2)
    info = solve(instance)
    outcome = evaluate_plan(instance, info.plan)
    env = LadderEnv(instance=instance)
    for world in (0, 1):
        obs, _ = env.reset(seed=0, options={"world": world})
        assert env.observation_space.contains(obs)
        actions = _plan_actions(env, info.plan, world)
        total, done, rewards = 0.0, False, []
        while not done:
            action = np.zeros(env.max_resources, dtype=np.int64)
            for r, a in actions.get(env.state.tick, {}).items():
                action[r] = a
            obs, reward, done, _, step_info = env.step(action)
            total += reward
            rewards.append(reward)
            assert step_info["conflicts"] == 0
        assert total == pytest.approx(outcome.world_values[world])
        assert step_info["realized_value"] == pytest.approx(outcome.world_values[world])
    # Rewards before the reveal are identical across worlds (no leak of the hidden forecast).
    pre = []
    for world in (0, 1):
        env.reset(seed=0, options={"world": world})
        seq = []
        actions = _plan_actions(env, info.plan, world)
        while env.state.forecast < 0 and env.state.tick < instance.horizon:
            action = np.zeros(env.max_resources, dtype=np.int64)
            for r, a in actions.get(env.state.tick, {}).items():
                action[r] = a
            seq.append(env.step(action)[1])
        pre.append(seq[:-1])
    assert pre[0] == pytest.approx(pre[1])


def test_env_passes_gymnasium_checker_and_random_play_is_bounded():
    from gymnasium.utils.env_checker import check_env

    env = LadderEnv(tiers=(1, 2, 3))
    check_env(env, skip_render_check=True)
    rng = np.random.default_rng(0)
    for episode in range(5):
        obs, _ = env.reset(seed=episode)
        # A single realized world is bounded by that world's clairvoyant optimum <= clairvoyant / prior.
        bound = solve(env.instance, "clairvoyant").value / min(env.instance.prior)
        done, total = False, 0.0
        while not done:
            mask = obs["action_mask"]
            action = np.array([rng.choice(np.flatnonzero(mask[r])) for r in range(env.max_resources)])
            obs, reward, done, _, _ = env.step(action)
            total += reward
        assert total <= bound + 1e-6


def test_statistics_helpers():
    assert sign_flip_p([1.0] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_p([0.0, 0.0]) == 1.0
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["b"] >= adjusted["c"]
    tiers = {f"i{k}": 1 for k in range(12)}
    rows = pairwise({"x": {k: 90.0 for k in tiers}, "y": {k: 80.0 for k in tiers}}, tiers)
    assert rows[0]["separated"] and rows[0]["mean_difference"] == pytest.approx(10.0)


def test_command_adapter_end_to_end(tmp_path):
    """A command agent that returns the optimal plan scores exactly 100 through the full runner."""
    from aeolus.ladder.runner import run

    script = tmp_path / "oracle.py"
    script.write_text(
        "import json, sys\n"
        "req = json.load(sys.stdin)\n"
        "inst = json.loads(open(sys.argv[1]).read())\n"
        "obs = req['observation']\n"
        "key = str(sorted(j['id'] for j in obs['jobs']))\n"
        "print(json.dumps(inst[key]))\n"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "schema_version: 3\nsuite_id: t\nsplit: development\nseed_offset: 0\n"
        "tiers: {1: 2, 2: 1}\nrepeats: 1\n"
    )
    plans = {}
    for tier, seed in ((1, 0), (1, 1), (2, 0)):
        inst = generate(tier, seed)
        plans[str(sorted(j.id for j in inst.jobs))] = solve(inst).plan
    table = tmp_path / "plans.json"
    table.write_text(json.dumps(plans))
    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n  - name: oracle\n    provider: command\n    model: oracle\n"
        f"    command: [{json.dumps(sys.executable)}, {json.dumps(str(script))}, {json.dumps(str(table))}]\n"
    )
    summary = run(suite, models, tmp_path / "out", workers=2)
    row = next(r for r in summary["leaderboard"] if r["policy"] == "oracle")
    assert row["mean_score"] == pytest.approx(100.0) and row["solved"] == 3
    assert (tmp_path / "out" / "report.txt").exists()
    assert Path(tmp_path / "out" / "calls.jsonl").read_text().count("\n") == 3
