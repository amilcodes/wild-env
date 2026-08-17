from __future__ import annotations

import torch

from aeolus.config import ResourceSpec, ScenarioConfig, ServiceSiteSpec
from aeolus.core.simulator import AeolusSimulator
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.policies import incident_risk_greedy
from aeolus.training.networks import EntityAttentionActorCritic
from aeolus.training.rollout import TensorIncidentCollector


def incident_config() -> ScenarioConfig:
    return ScenarioConfig(
        seed=37,
        width=48,
        height=48,
        cell_size_m=60.0,
        horizon_min=120,
        decision_interval_min=1,
        max_tasks=32,
        wind_speed_m_s=3.0,
        resources=(
            ResourceSpec(
                "water_0",
                "water",
                50.0,
                2500.0,
                5,
                0,
                100,
                home_site_id="base",
                service_modes=("land", "hover_fill"),
            ),
            ResourceSpec(
                "water_1",
                "water",
                48.0,
                2500.0,
                5,
                0,
                100,
                home_site_id="base",
                service_modes=("land", "hover_fill"),
            ),
            ResourceSpec(
                "retardant_0",
                "retardant",
                62.0,
                6000.0,
                9,
                0,
                150,
                home_site_id="base",
            ),
        ),
        service_sites=(
            ServiceSiteSpec(
                "base",
                "airport",
                4,
                42,
                ("water", "retardant", "fuel"),
                "land",
                bays=2,
                approach_capacity=3,
                refill_rate_l_min=1800.0,
                fixed_turnaround_min=3.0,
                available_volume_l=100_000.0,
                manually_verified=True,
            ),
            ServiceSiteSpec(
                "dip",
                "dip_site",
                31,
                27,
                ("water",),
                "hover_fill",
                bays=1,
                approach_capacity=2,
                refill_rate_l_min=5000.0,
                fixed_turnaround_min=1.0,
                available_volume_l=40_000.0,
                manually_verified=True,
            ),
        ),
    )


def test_tensor_incident_shapes_and_probability_conservation() -> None:
    env = TensorIncidentEnv(
        incident_config(),
        batch_size=5,
        max_segments=7,
        grid_size=24,
    )
    observation = env.observations()
    assert observation.resource.shape == (5, 3, 17)
    assert observation.tasks.shape == (5, 10, 21)
    assert observation.action_mask.shape == (5, 3, 10)
    transition = env.step(incident_risk_greedy(env))
    phase_sum = env.state.unburned + env.state.burning + env.state.burned
    assert torch.allclose(
        phase_sum.masked_select(~env.state.barrier),
        torch.ones_like(phase_sum.masked_select(~env.state.barrier)),
        atol=1.0e-5,
    )
    assert torch.isfinite(transition.reward).all()
    assert torch.isfinite(transition.constraint_costs).all()


def test_actor_observation_excludes_hidden_fire_truth() -> None:
    env = TensorIncidentEnv(
        incident_config(),
        batch_size=3,
        max_segments=6,
        grid_size=20,
    )
    public_before = env._observation_tensors(env.state)
    altered = env.state._replace(
        burning=torch.zeros_like(env.state.burning),
        burned=(~env.state.barrier).to(env.dtype),
        unburned=torch.zeros_like(env.state.unburned),
    )
    public_after = env._observation_tensors(altered)
    for before, after in zip(public_before[:5], public_after[:5], strict=True):
        assert torch.equal(before, after)
    assert not torch.equal(public_before[5], public_after[5])


def test_completed_drop_conserves_liquid_volume_on_surrogate_grid() -> None:
    env = TensorIncidentEnv(
        incident_config(),
        batch_size=1,
        max_segments=5,
        grid_size=20,
    )
    actions = torch.tensor([[1, 0, 0]])
    delivered = 0.0
    for _ in range(4):
        transition = env.step(actions)
        delivered += float(transition.delivered_l.item())
        actions.zero_()
        if delivered > 0.0:
            break
    integrated_water_l = float(
        (env.state.water_coverage_gpc * env.cell_area_m2 * env.config.suppression.gpc_l_m2).sum()
    )
    assert delivered == env.resource_payload_l[0].item()
    assert abs(integrated_water_l - delivered) / delivered < 1.0e-5
    assert env.state.resource_payload_fraction[0, 0] == 0.0


def test_service_reservations_preserve_stock_and_capacity() -> None:
    env = TensorIncidentEnv(
        incident_config(),
        batch_size=1,
        max_segments=4,
        grid_size=20,
    )
    state = env.state._replace(
        resource_payload_fraction=torch.zeros_like(env.state.resource_payload_fraction)
    )
    env.state = state
    env._refresh_observation()
    base_action = 1 + env.max_segments
    before = env.state.site_remaining_l[0, 0].clone()
    transition = env.step(torch.tensor([[base_action, base_action, base_action]]))
    reserved = env.state.resource_reserved_load_l.sum()
    assert transition.blocked_actions.item() == 0
    assert torch.isclose(before - env.state.site_remaining_l[0, 0], reserved)
    assert (env.state.site_slot_available_min[0, 0, :2] > 0.0).all()


def test_public_belief_comparator_changes_fire_outcomes() -> None:
    config = incident_config()
    hold = TensorIncidentEnv(
        config,
        batch_size=16,
        max_segments=10,
        grid_size=24,
    )
    response = TensorIncidentEnv(
        config,
        batch_size=16,
        max_segments=10,
        grid_size=24,
    )
    hold.reset(seed=91)
    response.reset(seed=91)
    hold_return = torch.zeros(16)
    response_return = torch.zeros(16)
    for _ in range(70):
        hold_step = hold.step(torch.zeros((16, 3), dtype=torch.long))
        response_step = response.step(incident_risk_greedy(response))
        hold_return += hold_step.reward
        response_return += response_step.reward
    assert response_step.expected_loss.mean() < 0.70 * hold_step.expected_loss.mean()
    assert response_step.burned_fraction.mean() < hold_step.burned_fraction.mean()
    assert response_return.mean() > hold_return.mean()


def test_transition_is_fullgraph_compilable_and_collector_compatible() -> None:
    config = incident_config()
    env = TensorIncidentEnv(
        config,
        batch_size=2,
        max_segments=4,
        grid_size=16,
        fire_substeps=1,
    )
    env.compile(backend="eager")
    transition = env.step(torch.tensor([[1, 0, 0], [1, 0, 0]]))
    assert torch.isfinite(transition.reward).all()

    collector = TensorIncidentCollector(
        config,
        num_envs=3,
        seed=12,
        device=torch.device("cpu"),
        hidden_dim=32,
        max_segments=4,
        grid_size=16,
        fire_substeps=1,
        observation_period_min=6,
        compile_environment=False,
    )
    model = EntityAttentionActorCritic(
        32,
        attention_heads=4,
        attention_layers=1,
    )
    rollout = collector.collect(model, steps=3)
    assert rollout.resource.shape == (3, 3, 3, 17)
    assert rollout.tasks.shape == (3, 3, 7, 21)
    assert torch.isfinite(rollout.rewards).all()


def test_terminal_world_is_absorbing() -> None:
    env = TensorIncidentEnv(
        incident_config(),
        batch_size=2,
        max_segments=4,
        grid_size=16,
    )
    done = torch.tensor([True, False])
    env.state = env.state._replace(done=done)
    before = tuple(value.clone() for value in env.state)
    transition = env.step(torch.zeros((2, 3), dtype=torch.long))
    for old_value, new_value in zip(before, env.state, strict=True):
        assert torch.equal(old_value[0], new_value[0])
    assert transition.reward[0] == 0.0
    assert transition.delivered_l[0] == 0.0


def test_canonical_snapshot_projection_preserves_shared_fire_state() -> None:
    config = incident_config()
    simulator = AeolusSimulator(config)
    simulator.decision_step({resource_id: 0 for resource_id in simulator.agent_ids})
    env = TensorIncidentEnv(
        config,
        batch_size=4,
        max_segments=5,
        grid_size=20,
    )
    env.initialize_fire_from_canonical(simulator)
    assert (env.state.minute == simulator.state.minute).all()
    assert torch.equal(env.state.burning[0], env.state.burning[3])
    assert torch.equal(env.state.asset_value[0], env.state.asset_value[3])
    assert torch.allclose(
        (env.state.unburned + env.state.burning + env.state.burned).masked_select(~env.state.barrier),
        torch.ones_like(env.state.unburned.masked_select(~env.state.barrier)),
        atol=1.0e-5,
    )
