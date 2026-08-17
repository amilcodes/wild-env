"""Estimate persistent tensor/rollout memory before accelerator profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aeolus.config import load_config
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.training.networks import build_policy_network

GIB = 1024**3


def tensor_bytes(values) -> int:
    return sum(value.numel() * value.element_size() for value in values if isinstance(value, torch.Tensor))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cluster_tensor_incident.yaml",
    )
    parser.add_argument(
        "--out",
        default="results/tensor_incident/memory_estimate.json",
    )
    args = parser.parse_args()
    experiment = load_config(args.config)
    training = experiment.training
    probe = TensorIncidentEnv(
        experiment.scenario,
        batch_size=1,
        max_segments=training.tensor_max_segments,
        grid_size=training.tensor_grid_size,
        fire_substeps=training.tensor_fire_substeps,
        observation_period_min=training.tensor_observation_period_min,
        device="cpu",
        terminate_on_completion=False,
        terminate_on_escape=False,
    )
    observation = probe.observations()
    state_per_world = tensor_bytes(probe.state)
    observation_per_world = tensor_bytes(
        (
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.task_valid,
            observation.global_state,
            probe.critic_state(),
        )
    )
    hidden_per_world = probe.num_resources * training.hidden_dim * torch.float32.itemsize
    action_per_world = probe.num_resources * torch.int64.itemsize
    logp_per_world = probe.num_resources * torch.float32.itemsize
    scalar_record_per_world = 3 * torch.float32.itemsize
    diagnostics_per_world_step = 10 * torch.float32.itemsize + torch.int64.itemsize
    rollout_per_world_step = (
        observation_per_world
        + hidden_per_world
        + action_per_world
        + logp_per_world
        + scalar_record_per_world
        + diagnostics_per_world_step
    )

    model = build_policy_network(training)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    optimizer_training_bytes = 4 * parameter_bytes
    batch = training.num_envs
    persistent_environment = batch * (state_per_world + observation_per_world)
    rollout_storage = batch * training.rollout_steps * rollout_per_world_step
    lower_bound = persistent_environment + rollout_storage + optimizer_training_bytes
    result = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "batch_size_per_rank": batch,
        "grid_size": training.tensor_grid_size,
        "resources": probe.num_resources,
        "tasks": probe.num_tasks,
        "rollout_steps": training.rollout_steps,
        "recurrent_sequence_length": training.recurrent_sequence_length,
        "bytes": {
            "state_per_world": state_per_world,
            "observation_per_world": observation_per_world,
            "rollout_per_world_step": rollout_per_world_step,
            "diagnostics_per_world_step": diagnostics_per_world_step,
            "persistent_environment": persistent_environment,
            "rollout_storage": rollout_storage,
            "policy_parameters": parameter_bytes,
            "policy_parameters_gradients_adam_estimate": optimizer_training_bytes,
            "accounted_lower_bound": lower_bound,
        },
        "gib": {
            "persistent_environment": persistent_environment / GIB,
            "rollout_storage": rollout_storage / GIB,
            "policy_parameters_gradients_adam_estimate": (optimizer_training_bytes / GIB),
            "accounted_lower_bound": lower_bound / GIB,
        },
        "exclusions": [
            "compiled transition workspace and graph pools",
            "fire and front-extraction intermediate tensors",
            "attention and recurrent autograd activations",
            "PPO temporary tensors and allocator fragmentation",
            "CUDA context, NCCL buffers, and framework caches",
        ],
        "interpretation": (
            "This is a persistent-storage lower bound. Select the actual batch "
            "only from peak allocated/reserved memory measured after compile "
            "and several complete PPO updates on the target accelerator."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
