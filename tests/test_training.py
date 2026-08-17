from __future__ import annotations

import torch

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.evaluation.evaluate import learned_policy
from aeolus.training.networks import TaskPointerActorCritic
from aeolus.training.rollout import SynchronousCollector


def test_collector_and_task_pointer_network_shapes() -> None:
    config = ScenarioConfig(width=32, height=32, horizon_min=24, max_tasks=16, spotting_rate=0.0)
    device = torch.device("cpu")
    model = TaskPointerActorCritic(hidden_dim=48).to(device)
    collector = SynchronousCollector(config, num_envs=2, seed=123, device=device, hidden_dim=48)
    rollout = collector.collect(model, steps=3)
    assert rollout.resource.shape == (3, 2, 3, 13)
    assert rollout.tasks.shape == (3, 2, 16, 11)
    assert rollout.masks.shape == (3, 2, 3, 16)
    assert rollout.actor_global_state.shape == (3, 2, 10)
    assert rollout.critic_global_state.shape == (3, 2, 12)
    assert rollout.actions.shape == (3, 2, 3)
    assert torch.isfinite(rollout.rewards).all()
    assert (rollout.actions >= 0).all() and (rollout.actions < 16).all()


def test_learned_execution_preserves_recurrent_state() -> None:
    config = ScenarioConfig(width=32, height=32, horizon_min=12, max_tasks=16)
    simulator = AeolusSimulator(config)
    model = TaskPointerActorCritic(hidden_dim=48)
    policy = learned_policy(model, torch.device("cpu"))
    first = policy(simulator)
    simulator.decision_step(first)
    second = policy(simulator)
    assert set(first) == set(second) == set(simulator.agent_ids)
    # A reset at minute zero must also be accepted and reset policy memory.
    simulator.reset(7)
    reset_action = policy(simulator)
    assert set(reset_action) == set(simulator.agent_ids)
