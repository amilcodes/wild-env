"""Benchmark and falsify the fire-coupled tensor MARL environment."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from aeolus.config import load_config
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.policies import incident_risk_greedy
from aeolus.training.networks import (
    EntityAttentionActorCritic,
    TaskPointerActorCritic,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def uniform_valid_actions(env: TensorIncidentEnv) -> torch.Tensor:
    observation = env.observations()
    logits = torch.zeros_like(observation.action_mask, dtype=env.dtype)
    actions, _, _ = TaskPointerActorCritic._capacity_aware_actions(
        logits,
        observation.tasks,
        observation.action_mask,
    )
    return actions


def actions_for(env: TensorIncidentEnv, policy: str) -> torch.Tensor:
    if policy == "hold":
        return torch.zeros(
            (env.batch_size, env.num_resources),
            device=env.device,
            dtype=torch.long,
        )
    if policy == "uniform_valid":
        return uniform_valid_actions(env)
    if policy == "incident_risk_greedy":
        return incident_risk_greedy(env)
    raise ValueError(f"unknown comparator: {policy}")


def policy_study(
    env: TensorIncidentEnv,
    *,
    policy: str,
    steps: int,
    seed: int,
) -> tuple[dict[str, float | int | str], dict[str, torch.Tensor]]:
    env.reset(seed=seed)
    reward = torch.zeros(env.batch_size, device=env.device)
    delivered_l = torch.zeros_like(reward)
    wasted_l = torch.zeros_like(reward)
    blocked = torch.zeros_like(reward)
    costs = torch.zeros(
        (env.batch_size, 4),
        device=env.device,
        dtype=env.dtype,
    )
    initial_loss, initial_burned, _ = env._outcome_metrics(env.state)
    synchronize(env.device)
    started = time.perf_counter()
    final = None
    for _ in range(steps):
        final = env.step(actions_for(env, policy))
        reward += final.reward
        delivered_l += final.delivered_l
        wasted_l += final.wasted_l
        blocked += final.blocked_actions
        costs += final.constraint_costs
    synchronize(env.device)
    elapsed = time.perf_counter() - started
    assert final is not None
    episode_metrics = {
        "return": reward,
        "expected_loss": final.expected_loss,
        "burned_fraction": final.burned_fraction,
        "delivered_l": delivered_l,
        "wasted_l": wasted_l,
        "blocked_actions": blocked,
        "constraint_costs": costs,
    }
    summary: dict[str, float | int | str] = {
        "policy": policy,
        "steps": steps,
        "batch_size": env.batch_size,
        "mean_initial_expected_loss": float(initial_loss.mean()),
        "mean_initial_burned_fraction": float(initial_burned.mean()),
        "mean_final_expected_loss": float(final.expected_loss.mean()),
        "mean_final_burned_fraction": float(final.burned_fraction.mean()),
        "mean_return": float(reward.mean()),
        "mean_delivered_l": float(delivered_l.mean()),
        "mean_wasted_l": float(wasted_l.mean()),
        "mean_blocked_actions": float(blocked.mean()),
        "contained_fraction": float(env.state.contained.to(env.dtype).mean()),
        "escaped_fraction": float(env.state.escaped.to(env.dtype).mean()),
        "terminal_fraction": float(env.state.done.to(env.dtype).mean()),
        "environment_steps_per_second": env.batch_size * steps / elapsed,
        "agent_decisions_per_second": (env.batch_size * env.num_resources * steps / elapsed),
        "elapsed_s": elapsed,
    }
    return summary, episode_metrics


def paired_effect(
    treatment: dict[str, torch.Tensor],
    control: dict[str, torch.Tensor],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("return", "expected_loss", "burned_fraction"):
        difference = treatment[metric] - control[metric]
        result[f"mean_paired_delta_{metric}"] = float(difference.mean())
        result[f"paired_improvement_fraction_{metric}"] = float(
            (difference > 0.0 if metric == "return" else difference < 0.0).to(torch.float32).mean()
        )
    return result


@torch.no_grad()
def end_to_end_throughput(
    env: TensorIncidentEnv,
    *,
    hidden_dim: int,
    heads: int,
    layers: int,
    steps: int,
    seed: int,
) -> dict[str, float | int]:
    model = EntityAttentionActorCritic(
        hidden_dim,
        attention_heads=heads,
        attention_layers=layers,
    ).to(env.device)
    hidden = torch.zeros(
        (env.batch_size, env.num_resources, hidden_dim),
        device=env.device,
    )
    env.reset(seed=seed)
    for _ in range(2):
        observation = env.observations()
        actions, _, _, _, hidden = model.act(
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.global_state,
            env.critic_state(),
            hidden,
        )
        env.step(actions)
    synchronize(env.device)
    started = time.perf_counter()
    for _ in range(steps):
        observation = env.observations()
        actions, _, _, _, hidden = model.act(
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.global_state,
            env.critic_state(),
            hidden,
        )
        transition = env.step(actions)
        hidden = hidden.masked_fill(transition.done[:, None, None], 0.0)
    synchronize(env.device)
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "environment_steps_per_second": env.batch_size * steps / elapsed,
        "agent_decisions_per_second": (env.batch_size * env.num_resources * steps / elapsed),
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cluster_tensor_incident.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--fire-substeps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--model-steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--out",
        default="results/tensor_incident/study.json",
    )
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    experiment = load_config(args.config)
    env = TensorIncidentEnv(
        experiment.scenario,
        batch_size=args.batch_size,
        max_segments=args.segments,
        grid_size=args.grid_size,
        fire_substeps=args.fire_substeps,
        observation_period_min=(experiment.training.tensor_observation_period_min),
        device=device,
        terminate_on_completion=False,
        terminate_on_escape=False,
    )
    if args.compile:
        env.compile()
    summaries = []
    episode_metrics = {}
    for policy in ("hold", "uniform_valid", "incident_risk_greedy"):
        summary, metrics = policy_study(
            env,
            policy=policy,
            steps=args.steps,
            seed=7701,
        )
        summaries.append(summary)
        episode_metrics[policy] = metrics
    results = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "compiled_transition": bool(args.compile),
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
            "grid_size": env.grid_size,
            "fire_substeps": env.fire_substeps,
            "decision_interval_min": env.config.decision_interval_min,
            "parameter_ranges": env.parameter_ranges.__dict__,
        },
        "policies": summaries,
        "paired_greedy_minus_hold": paired_effect(
            episode_metrics["incident_risk_greedy"],
            episode_metrics["hold"],
        ),
        "end_to_end_policy_and_environment": end_to_end_throughput(
            env,
            hidden_dim=min(experiment.training.hidden_dim, 128),
            heads=min(experiment.training.attention_heads, 8),
            layers=min(experiment.training.attention_layers, 2),
            steps=args.model_steps,
            seed=8812,
        ),
        "interpretation": (
            "This local result is a mechanism and CPU-throughput check. "
            "Accelerator throughput and learning claims require rerunning the "
            "same manifest on the target cluster and retaining this artifact."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
