"""Run reproducible mechanism and throughput studies for aerial MARL operations."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from aeolus.config import load_config
from aeolus.envs.tensor_operations import TensorOperationsEnv
from aeolus.policies import cycle_time_greedy
from aeolus.training.networks import (
    EntityAttentionActorCritic,
    TaskPointerActorCritic,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def uniform_valid_actions(env: TensorOperationsEnv) -> torch.Tensor:
    observation = env.observations()
    logits = torch.zeros_like(observation.action_mask, dtype=env.dtype)
    actions, _, _ = TaskPointerActorCritic._capacity_aware_actions(
        logits,
        observation.tasks,
        observation.action_mask,
    )
    return actions


def policy_study(
    env: TensorOperationsEnv,
    *,
    policy_name: str,
    steps: int,
) -> dict[str, float | str]:
    env.reset(seed=7701)
    initial_l = env.state.segment_remaining_l.sum(dim=1)
    reward = torch.zeros(env.batch_size, device=env.device)
    wasted_l = torch.zeros_like(reward)
    delivered_l = torch.zeros_like(reward)
    started = time.perf_counter()
    for _ in range(steps):
        actions = cycle_time_greedy(env) if policy_name == "cycle_time_greedy" else uniform_valid_actions(env)
        transition = env.step(actions)
        reward += transition.reward
        wasted_l += transition.wasted_l
        delivered_l += transition.delivered_l
    synchronize(env.device)
    elapsed = time.perf_counter() - started
    completion = 1.0 - env.state.segment_remaining_l.sum(dim=1) / initial_l
    return {
        "policy": policy_name,
        "steps": steps,
        "batch_size": env.batch_size,
        "mean_completion_fraction": float(completion.mean()),
        "complete_episode_fraction": float(env.state.done.to(env.dtype).mean()),
        "mean_return": float(reward.mean()),
        "mean_weighted_delivered_l": float(delivered_l.mean()),
        "mean_wasted_l": float(wasted_l.mean()),
        "environment_decisions_per_second": env.batch_size * steps / elapsed,
        "agent_decisions_per_second": (env.batch_size * env.num_resources * steps / elapsed),
        "elapsed_s": elapsed,
    }


def model_throughput(
    env: TensorOperationsEnv,
    *,
    hidden_dim: int,
    heads: int,
    layers: int,
    iterations: int,
) -> dict[str, float]:
    model = EntityAttentionActorCritic(
        hidden_dim,
        attention_heads=heads,
        attention_layers=layers,
    ).to(env.device)
    observation = env.observations()
    hidden = torch.zeros(
        (env.batch_size, env.num_resources, hidden_dim),
        device=env.device,
    )
    for _ in range(3):
        model.act(
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.global_state,
            env.critic_state(),
            hidden,
        )
    synchronize(env.device)
    started = time.perf_counter()
    for _ in range(iterations):
        _, _, _, _, hidden = model.act(
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.global_state,
            env.critic_state(),
            hidden,
        )
    synchronize(env.device)
    elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "agent_actions_per_second": (env.batch_size * env.num_resources * iterations / elapsed),
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cluster_tensor_operations.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--model-iterations", type=int, default=20)
    parser.add_argument(
        "--out",
        default="results/rl_operations/study.json",
    )
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    experiment = load_config(args.config)
    env = TensorOperationsEnv(
        experiment.scenario,
        batch_size=args.batch_size,
        max_segments=args.segments,
        device=device,
    )
    results = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        },
        "environment": {
            "batch_size": env.batch_size,
            "resources": env.num_resources,
            "service_sites": env.num_sites,
            "segments": env.max_segments,
            "decision_interval_min": env.config.decision_interval_min,
        },
        "policies": [
            policy_study(env, policy_name="uniform_valid", steps=args.steps),
            policy_study(
                env,
                policy_name="cycle_time_greedy",
                steps=args.steps,
            ),
        ],
        "model_throughput": model_throughput(
            env,
            hidden_dim=min(experiment.training.hidden_dim, 128),
            heads=min(experiment.training.attention_heads, 8),
            layers=min(experiment.training.attention_layers, 2),
            iterations=args.model_iterations,
        ),
        "interpretation": (
            "CPU results validate mechanics and the benchmark harness. "
            "Cluster throughput claims require rerunning this exact manifest on "
            "the target accelerator and retaining the emitted artifact."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
