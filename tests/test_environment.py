from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from aeolus.config import ResourceSpec, ScenarioConfig
from aeolus.core.fire import rothermel_ros_m_min
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import critic_global_features, generate_tasks
from aeolus.envs.parallel import AeolusParallelEnv
from aeolus.policies import greedy_value, joint_assignment, no_aerial_action


def small_config(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = {
        "seed": 17,
        "width": 32,
        "height": 32,
        "horizon_min": 36,
        "max_tasks": 24,
        "decision_interval_min": 3,
        "wind_speed_m_s": 3.0,
        "spotting_rate": 0.0,
    }
    values.update(overrides)
    return ScenarioConfig(**values)


def test_seeded_heuristic_rollout_is_reproducible() -> None:
    config = small_config()
    first, second = AeolusSimulator(config), AeolusSimulator(config)
    first.reset(77)
    second.reset(77)
    for _ in range(6):
        first.decision_step(greedy_value(first))
        second.decision_step(greedy_value(second))
    assert first.episode_record()["weighted_loss"] == second.episode_record()["weighted_loss"]
    assert np.array_equal(first.state.truth.phase, second.state.truth.phase)
    assert np.array_equal(first.state.belief.intensity_mean, second.state.belief.intensity_mean)


def test_observation_is_not_a_truth_state_alias() -> None:
    sim = AeolusSimulator(small_config())
    sim.reset(91)
    before = sim.state.belief.intensity_mean.copy()
    sim.decision_step(no_aerial_action(sim))
    # The truth kernel advances, while belief only changes where an explicitly
    # delivered observation has been received.
    assert not np.array_equal(sim.state.truth.intensity_kw_m, before)
    stale = sim.state.belief.observed_at < sim.state.minute
    assert stale.any()
    assert np.all(sim.state.belief.intensity_std[stale] >= 1.0)


def test_hidden_truth_changes_do_not_change_actor_observation() -> None:
    sim = AeolusSimulator(small_config())
    before = sim.reset(91)
    before_critic = critic_global_features(sim).copy()
    sim.state.truth.intensity_kw_m *= 0.0
    sim.state.truth.phase[:] = 2
    sim.state.truth.observed_burned[:] = 1.0
    sim.tasks = generate_tasks(sim)
    after = sim.observations()
    after_critic = critic_global_features(sim)
    for agent in sim.agent_ids:
        for field in ("resource", "tasks", "action_mask", "task_valid", "global"):
            assert np.array_equal(before[agent][field], after[agent][field])
    assert not np.array_equal(before_critic, after_critic)


def test_action_masks_are_resource_specific() -> None:
    sim = AeolusSimulator(small_config())
    observations = sim.reset(7)
    tanker_mask = observations["tanker_12"]["action_mask"]
    helicopter_mask = observations["heli_07"]["action_mask"]
    scout_mask = observations["ir_scout"]["action_mask"]
    assert tanker_mask[0] and helicopter_mask[0] and scout_mask[0]
    assert tanker_mask.sum() > 1 and helicopter_mask.sum() > 1 and scout_mask.sum() > 1
    # Resource-specific masks differ because only compatible mission types are
    # legal at an available resource.
    assert not np.array_equal(tanker_mask, helicopter_mask)
    assert not np.array_equal(helicopter_mask, scout_mask)


def test_conflicting_same_task_is_logged_and_blocked() -> None:
    water_a = ResourceSpec("water_a", "water", 45.0, 2500.0, 6, 1, 90)
    water_b = ResourceSpec("water_b", "water", 45.0, 2500.0, 6, 1, 90)
    sensor = ResourceSpec("sensor", "sensor", 35.0, 0.0, 0, 1, 90)
    sim = AeolusSimulator(small_config(resources=(water_a, water_b, sensor)))
    sim.reset(11)
    water_task = next(task.index for task in sim.tasks if task.kind.name == "WATER")
    sim.decision_step({"water_a": water_task, "water_b": water_task, "sensor": 0})
    assert sim.state.blocked_actions == 1


def test_joint_assignment_avoids_capacity_conflicts_for_same_kind() -> None:
    water_a = ResourceSpec("water_a", "water", 45.0, 2500.0, 6, 1, 90)
    water_b = ResourceSpec("water_b", "water", 45.0, 2500.0, 6, 1, 90)
    sensor = ResourceSpec("sensor", "sensor", 35.0, 0.0, 0, 1, 90)
    sim = AeolusSimulator(small_config(resources=(water_a, water_b, sensor)))
    sim.reset(11)
    actions = joint_assignment(sim)
    non_hold_water = [actions["water_a"], actions["water_b"]]
    assert all(action > 0 for action in non_hold_water)
    assert len(set(non_hold_water)) == 2
    sim.decision_step(actions)
    assert sim.state.blocked_actions == 0


def test_resource_position_interpolates_during_mission() -> None:
    sim = AeolusSimulator(small_config(cell_size_m=300.0))
    sim.reset(11)
    action = next(task.index for task in sim.tasks if task.kind.name == "RETARDANT")
    tanker = sim.state.resources[0]
    start = (tanker.x, tanker.y)
    sim.decision_step({"tanker_12": action, "heli_07": 0, "ir_scout": 0})
    assert (tanker.x, tanker.y) != start
    assert tanker.target_xy is not None
    assert (tanker.x, tanker.y) != tanker.target_xy


def test_inflight_mission_is_independent_of_regenerated_task_list() -> None:
    slow_water = ResourceSpec("slow_water", "water", 30.0, 2500.0, 5, 8, 90)
    sensor = ResourceSpec("sensor", "sensor", 35.0, 0.0, 0, 1, 90)
    sim = AeolusSimulator(
        small_config(resources=(slow_water, sensor), decision_interval_min=3)
    )
    water_task = next(task.index for task in sim.tasks if task.kind.name == "WATER")
    sim.decision_step({"slow_water": water_task, "sensor": 0})
    sim.state.belief.intensity_mean[:] = 0.0
    sim.tasks = generate_tasks(sim)
    assert len(sim.tasks) == 1
    while sim.state.resources[0].status.name == "OUTBOUND":
        sim.decision_step({"slow_water": 0, "sensor": 0})
    assert any(event["kind"] == "water_drop" for event in sim.state.events)


def test_ros_increases_with_wind_and_slope() -> None:
    fuel = small_config().fuel
    calm_flat = rothermel_ros_m_min(fuel, wind_m_s=1.0, slope_tan=0.0)
    windy_flat = rothermel_ros_m_min(fuel, wind_m_s=8.0, slope_tan=0.0)
    windy_slope = rothermel_ros_m_min(fuel, wind_m_s=8.0, slope_tan=0.45)
    assert calm_flat < windy_flat < windy_slope


def test_parallel_env_returns_shared_reward_and_observation_contract() -> None:
    env = AeolusParallelEnv(small_config())
    observations, _ = env.reset(seed=3)
    actions = {agent: 0 for agent in env.agents}
    next_observations, rewards, terminations, truncations, _ = env.step(actions)
    assert set(rewards) == set(env.possible_agents)
    assert len(set(rewards.values())) == 1
    assert all(isinstance(terminations[agent], bool) for agent in env.possible_agents)
    assert all(isinstance(truncations[agent], bool) for agent in env.possible_agents)
    if env.agents:
        assert set(next_observations) == set(env.possible_agents)
        assert env.observation_space("tanker_12").contains(next_observations["tanker_12"])


def test_parallel_env_passes_pettingzoo_api_contract() -> None:
    parallel_api_test(AeolusParallelEnv(small_config(horizon_min=12)), num_cycles=20)


def test_ground_connected_intervention_reduces_loss_in_low_wind_mechanism_case() -> None:
    config = ScenarioConfig(
        seed=3,
        width=48,
        height=48,
        horizon_min=90,
        max_tasks=24,
        wind_speed_m_s=0.6,
        wind_variability=0.10,
        spotting_rate=0.0,
        ground_arrival_min=8,
    )
    uncontrolled = AeolusSimulator(config)
    supported = AeolusSimulator(config)
    uncontrolled.reset(3)
    supported.reset(3)
    while not uncontrolled.state.terminated and not uncontrolled.state.truncated:
        uncontrolled.decision_step(no_aerial_action(uncontrolled))
    while not supported.state.terminated and not supported.state.truncated:
        supported.decision_step(greedy_value(supported))
    assert supported.state.contained
    assert supported.episode_record()["weighted_loss"] < uncontrolled.episode_record()["weighted_loss"]
