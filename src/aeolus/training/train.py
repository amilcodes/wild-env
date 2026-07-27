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
from aeolus.training.networks import TaskPointerActorCritic
from aeolus.training.rollout import Rollout, SynchronousCollector


def _distributed_device(requested: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    if requested == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
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
    flat = time_steps * envs

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(flat, *value.shape[2:])

    resource = flatten(rollout.resource)
    tasks = flatten(rollout.tasks)
    masks = flatten(rollout.masks)
    global_state = flatten(rollout.global_state)
    hidden = flatten(rollout.hidden)
    actions = flatten(rollout.actions)
    old_logp = flatten(rollout.logp)
    old_values = rollout.values.reshape(flat)
    target_returns = returns.reshape(flat)
    flat_advantages = advantages.reshape(flat)
    indices = torch.randperm(flat, device=device)
    losses: list[tuple[float, float, float]] = []
    amp_enabled = train.use_amp and device.type == "cuda"
    for _ in range(train.epochs_per_update):
        for start in range(0, flat, train.minibatch_size):
            batch_index = indices[start : start + train.minibatch_size]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, values, _ = model(
                    resource[batch_index],
                    tasks[batch_index],
                    masks[batch_index],
                    global_state[batch_index],
                    hidden[batch_index],
                )
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(actions[batch_index])
                ratio = (logp - old_logp[batch_index]).exp()
                expanded_advantage = flat_advantages[batch_index].unsqueeze(-1)
                unclipped = ratio * expanded_advantage
                clipped = ratio.clamp(1.0 - train.clip_ratio, 1.0 + train.clip_ratio) * expanded_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                clipped_values = old_values[batch_index] + (values - old_values[batch_index]).clamp(
                    -train.value_clip_ratio, train.value_clip_ratio
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - target_returns[batch_index]).square(),
                        (clipped_values - target_returns[batch_index]).square(),
                    ).mean()
                )
                entropy = distribution.entropy().mean()
                loss = policy_loss + train.value_coef * value_loss - train.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
    summary = np.array(losses, dtype=np.float64).mean(axis=0)
    return {
        "policy_loss": float(summary[0]),
        "value_loss": float(summary[1]),
        "entropy": float(summary[2]),
        "return_mean": float(rollout.rewards.mean()),
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
    core_model = TaskPointerActorCritic(experiment.training.hidden_dim).to(device)
    model: nn.Module = core_model
    if distributed:
        model = DistributedDataParallel(
            core_model, device_ids=[device.index] if device.type == "cuda" else None
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.training.learning_rate, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=experiment.training.use_amp and device.type == "cuda")
    collector = SynchronousCollector(
        experiment.scenario,
        experiment.training.num_envs,
        experiment.training.seed + rank * 100_000,
        device,
        experiment.training.hidden_dim,
    )
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
                        "schema_version": 1,
                        "update": update,
                        "config": experiment.as_dict(),
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
