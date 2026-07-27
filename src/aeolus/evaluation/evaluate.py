"""Common-random-number policy evaluation with structured episode output."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.policies import anchor_flank, greedy_value, nearest_feasible, no_aerial_action
from aeolus.training.networks import TaskPointerActorCritic

Policy = Callable[[AeolusSimulator], dict[str, int]]


def learned_policy(model: TaskPointerActorCritic, device: torch.device) -> Policy:
    @torch.no_grad()
    def act(sim: AeolusSimulator) -> dict[str, int]:
        observations = sim.observations()
        ids = sim.agent_ids
        resource = torch.tensor(
            np.stack([observations[item]["resource"] for item in ids]), device=device
        ).unsqueeze(0)
        tasks = torch.tensor(observations[ids[0]]["tasks"], device=device).unsqueeze(0)
        masks = torch.tensor(
            np.stack([observations[item]["action_mask"] for item in ids]), device=device
        ).unsqueeze(0)
        global_state = torch.tensor(observations[ids[0]]["global"], device=device).unsqueeze(0)
        actions, _, _, _, _ = model.act(
            resource.float(), tasks.float(), masks.bool(), global_state.float(), deterministic=True
        )
        return {agent: int(actions[0, index]) for index, agent in enumerate(ids)}

    return act


def run_episode(sim: AeolusSimulator, policy: Policy, seed: int) -> dict[str, object]:
    sim.reset(seed)
    while not sim.state.terminated and not sim.state.truncated:
        sim.decision_step(policy(sim))
    return sim.episode_record()


def evaluate_pairs(
    policies: dict[str, Policy], simulator: AeolusSimulator, seeds: list[int]
) -> dict[str, object]:
    records = {
        name: [run_episode(simulator, policy, seed) for seed in seeds] for name, policy in policies.items()
    }
    summary = {
        name: {
            "episodes": len(items),
            "mean_weighted_loss": float(np.mean([item["weighted_loss"] for item in items])),
            "escape_rate": float(np.mean([item["escaped"] for item in items])),
            "containment_rate": float(np.mean([item["contained"] for item in items])),
        }
        for name, items in records.items()
    }
    return {"schema_version": 1, "seeds": seeds, "summary": summary, "episodes": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Aeolus policies with paired seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--out", default="runs/evaluation.json")
    args = parser.parse_args()
    config = load_config(args.config)
    simulator = AeolusSimulator(config.scenario)
    policies: dict[str, Policy] = {
        "no_aerial": no_aerial_action,
        "nearest": nearest_feasible,
        "anchor_flank": anchor_flank,
        "greedy_value": greedy_value,
    }
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = TaskPointerActorCritic(config.training.hidden_dim).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        policies["mappo"] = learned_policy(model, device)
    seeds = [config.scenario.seed + index * 7_919 for index in range(args.episodes)]
    result = evaluate_pairs(policies, simulator, seeds)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
