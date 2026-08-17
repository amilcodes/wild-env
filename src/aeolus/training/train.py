"""MAPPO training entry point with AMP, checkpointing, and DDP support."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from aeolus.config import ExperimentConfig, load_config
from aeolus.core.tasks import critic_global_features
from aeolus.training.networks import TaskPointerActorCritic, build_policy_network
from aeolus.training.rollout import (
    Rollout,
    SynchronousCollector,
    TensorIncidentCollector,
    TensorOperationsCollector,
    _stack_observations,
)
from aeolus.workflows import resolve_policy


def _distributed_device(requested: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if requested == "auto":
        # Tiny recurrent policy batches are slower on MPS because simulation is
        # CPU-resident and each decision incurs a device synchronization. MPS
        # remains available when requested explicitly.
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if distributed and requested == "mps":
        raise RuntimeError("multi-process DDP is supported on CUDA or CPU, not MPS")
    if requested == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size, distributed


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gae(rollout: Rollout, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rollout.rewards)
    last_advantage = torch.zeros_like(rollout.bootstrap_value)
    next_value = rollout.bootstrap_value
    for index in range(rollout.rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - rollout.dones[index]
        delta = rollout.rewards[index] + gamma * next_value * nonterminal - rollout.values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
        next_value = rollout.values[index]
    return advantages, advantages + rollout.values


def _ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: ExperimentConfig,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float]:
    train = config.training
    advantages, returns = _gae(rollout, train.gamma, train.gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    time_steps, envs = advantages.shape
    sequence_length = train.recurrent_sequence_length
    if time_steps % sequence_length:
        raise ValueError("rollout length is not divisible by recurrent sequence length")
    sequence_starts = torch.arange(
        0,
        time_steps,
        sequence_length,
        device=device,
    ).repeat_interleave(envs)
    sequence_envs = torch.arange(envs, device=device).repeat(time_steps // sequence_length)
    sequence_count = sequence_starts.numel()
    sequences_per_minibatch = max(1, train.minibatch_size // sequence_length)
    losses: list[tuple[float, float, float]] = []
    amp_enabled = train.use_amp and device.type == "cuda"
    for _ in range(train.epochs_per_update):
        indices = torch.randperm(sequence_count, device=device)
        for start in range(0, sequence_count, sequences_per_minibatch):
            sequence_index = indices[start : start + sequences_per_minibatch]
            starts = sequence_starts[sequence_index]
            selected_envs = sequence_envs[sequence_index]
            recurrent_hidden = rollout.hidden[starts, selected_envs]
            new_logp: list[torch.Tensor] = []
            new_values: list[torch.Tensor] = []
            new_entropy: list[torch.Tensor] = []
            old_logp: list[torch.Tensor] = []
            old_values: list[torch.Tensor] = []
            target_returns: list[torch.Tensor] = []
            sequence_advantages: list[torch.Tensor] = []
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                for offset in range(sequence_length):
                    time_index = starts + offset
                    resource = rollout.resource[time_index, selected_envs]
                    tasks = rollout.tasks[time_index, selected_envs]
                    masks = rollout.masks[time_index, selected_envs]
                    logits, values, next_hidden = model(
                        resource,
                        tasks,
                        masks,
                        rollout.actor_global_state[time_index, selected_envs],
                        rollout.critic_global_state[time_index, selected_envs],
                        recurrent_hidden,
                    )
                    _, logp, sample_entropy = TaskPointerActorCritic._capacity_aware_actions(
                        logits,
                        tasks,
                        masks,
                        actions=rollout.actions[time_index, selected_envs],
                    )
                    new_logp.append(logp)
                    new_values.append(values)
                    new_entropy.append(sample_entropy)
                    old_logp.append(rollout.logp[time_index, selected_envs])
                    old_values.append(rollout.values[time_index, selected_envs])
                    target_returns.append(returns[time_index, selected_envs])
                    sequence_advantages.append(advantages[time_index, selected_envs])
                    recurrent_hidden = next_hidden.masked_fill(
                        rollout.dones[time_index, selected_envs, None, None].bool(),
                        0.0,
                    )
                logp = torch.stack(new_logp)
                values = torch.stack(new_values)
                sample_entropy = torch.stack(new_entropy)
                previous_logp = torch.stack(old_logp)
                previous_values = torch.stack(old_values)
                batch_returns = torch.stack(target_returns)
                batch_advantages = torch.stack(sequence_advantages)
                ratio = (logp - previous_logp).exp()
                expanded_advantage = batch_advantages.unsqueeze(-1)
                unclipped = ratio * expanded_advantage
                clipped = ratio.clamp(1.0 - train.clip_ratio, 1.0 + train.clip_ratio) * expanded_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                clipped_values = previous_values + (values - previous_values).clamp(
                    -train.value_clip_ratio,
                    train.value_clip_ratio,
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - batch_returns).square(),
                        (clipped_values - batch_returns).square(),
                    ).mean()
                )
                entropy = sample_entropy.mean()
                loss = policy_loss + train.value_coef * value_loss - train.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
    summary = np.array(losses, dtype=np.float64).mean(axis=0)
    metrics = {
        "policy_loss": float(summary[0]),
        "value_loss": float(summary[1]),
        "entropy": float(summary[2]),
        "return_mean": float(rollout.rewards.mean()),
    }
    for name, value in rollout.diagnostics.items():
        if name in {"expected_loss", "burned_fraction", "contained", "escaped"}:
            metrics[f"{name}_final"] = float(value[-1].to(torch.float32).mean())
        else:
            metrics[f"{name}_per_env"] = float(value.to(torch.float32).sum(dim=0).mean())
    return metrics


def _expert_warmstart(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    collector: SynchronousCollector,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float] | None:
    steps = config.training.expert_warmstart_steps
    if steps <= 0:
        return None
    expert = resolve_policy(config.training.expert_policy)[0]
    losses: list[float] = []
    core_model = model.module if isinstance(model, DistributedDataParallel) else model
    core_model.train()
    for _ in range(steps):
        resource, tasks, masks, actor_global_state = _stack_observations(
            collector.observations,
            collector.agent_ids,
            device,
        )
        critic_global_state = torch.as_tensor(
            np.stack([critic_global_features(env.sim) for env in collector.envs]),
            device=device,
            dtype=torch.float32,
        )
        target_actions = [expert(env.sim) for env in collector.envs]
        target = torch.as_tensor(
            [[actions[agent] for agent in collector.agent_ids] for actions in target_actions],
            device=device,
            dtype=torch.long,
        )
        logits, _, next_hidden = model(
            resource,
            tasks,
            masks,
            actor_global_state,
            critic_global_state,
            collector.hidden,
        )
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))

        next_observations = []
        for env_index, env in enumerate(collector.envs):
            obs, _, terminations, truncations, _ = env.step(target_actions[env_index])
            done = bool(terminations[collector.agent_ids[0]] or truncations[collector.agent_ids[0]])
            if done:
                collector.episode_index[env_index] += collector.num_envs
                obs, _ = env.reset(seed=collector.seed + int(collector.episode_index[env_index]))
                next_hidden[env_index].zero_()
            next_observations.append(obs)
        collector.observations = next_observations
        collector.hidden = next_hidden.detach()
    return {
        "expert_steps": float(steps),
        "expert_loss_initial": losses[0],
        "expert_loss_final": losses[-1],
    }


def train(experiment: ExperimentConfig) -> None:
    device, rank, world_size, distributed = _distributed_device(experiment.training.device)
    _seed_everything(experiment.training.seed + rank)
    checkpoint_dir = Path(experiment.training.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "config.json").write_text(
            json.dumps(experiment.as_dict(), indent=2), encoding="utf-8"
        )
    if distributed:
        dist.barrier()
    core_model = build_policy_network(experiment.training).to(device)
    if experiment.training.compile_model:
        core_model.compile(
            mode="reduce-overhead",
            fullgraph=False,
        )
    model: nn.Module = core_model
    if distributed:
        model = DistributedDataParallel(
            core_model, device_ids=[device.index] if device.type == "cuda" else None
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.training.learning_rate, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=experiment.training.use_amp and device.type == "cuda")
    if experiment.training.environment_backend in {
        "tensor_operations",
        "tensor_incident",
    }:
        if experiment.training.expert_warmstart_steps:
            raise ValueError("expert warm start currently requires the canonical incident environment")
        if experiment.training.environment_backend == "tensor_operations":
            collector = TensorOperationsCollector(
                experiment.scenario,
                experiment.training.num_envs,
                experiment.training.seed + rank * 100_000,
                device,
                experiment.training.hidden_dim,
                max_segments=experiment.training.tensor_max_segments,
            )
        else:
            collector = TensorIncidentCollector(
                experiment.scenario,
                experiment.training.num_envs,
                experiment.training.seed + rank * 100_000,
                device,
                experiment.training.hidden_dim,
                max_segments=experiment.training.tensor_max_segments,
                grid_size=experiment.training.tensor_grid_size,
                fire_substeps=experiment.training.tensor_fire_substeps,
                observation_period_min=(experiment.training.tensor_observation_period_min),
                compile_environment=experiment.training.compile_environment,
            )
    else:
        collector = SynchronousCollector(
            experiment.scenario,
            experiment.training.num_envs,
            experiment.training.seed + rank * 100_000,
            device,
            experiment.training.hidden_dim,
        )
    warmstart_metrics = _expert_warmstart(
        model,
        optimizer,
        collector,
        experiment,
        device,
    )
    if rank == 0 and warmstart_metrics is not None:
        print(json.dumps({"phase": "expert_warmstart", **warmstart_metrics}), flush=True)
    metrics_path = checkpoint_dir / "metrics.jsonl"
    for update in range(1, experiment.training.updates + 1):
        rollout = collector.collect(core_model, experiment.training.rollout_steps)
        core_model.train()
        metrics = _ppo_update(model, optimizer, rollout, experiment, scaler, device)
        metrics.update({"update": update, "rank": rank, "world_size": world_size})
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            if update % experiment.training.checkpoint_every == 0 or update == experiment.training.updates:
                torch.save(
                    {
                        "schema_version": 2,
                        "update": update,
                        "config": experiment.as_dict(),
                        "policy_architecture": experiment.training.policy_architecture,
                        "environment_backend": experiment.training.environment_backend,
                        "model": core_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                    },
                    checkpoint_dir / "checkpoint.pt",
                )
            print(json.dumps(metrics), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aeolus MAPPO baseline")
    parser.add_argument("--config", required=True, help="YAML experiment manifest")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
