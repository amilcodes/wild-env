"""Common-random-number policy evaluation with structured episode output."""

from __future__ import annotations

import argparse
import json
import weakref
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import CRITIC_GLOBAL_FEATURE_DIM
from aeolus.policies import (
    anchor_flank,
    greedy_value,
    joint_assignment,
    nearest_feasible,
    no_aerial_action,
)
from aeolus.training.networks import TaskPointerActorCritic, build_policy_network

Policy = Callable[[AeolusSimulator], dict[str, int]]


def learned_policy(model: TaskPointerActorCritic, device: torch.device) -> Policy:
    hidden_by_simulator: weakref.WeakKeyDictionary[AeolusSimulator, tuple[int, torch.Tensor]] = (
        weakref.WeakKeyDictionary()
    )

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
        actor_global_state = torch.tensor(observations[ids[0]]["global"], device=device).unsqueeze(0)
        # The value function is irrelevant during execution. A zero privileged
        # state makes the deployment boundary explicit and cannot affect logits.
        critic_global_state = torch.zeros((1, CRITIC_GLOBAL_FEATURE_DIM), device=device, dtype=torch.float32)
        previous = hidden_by_simulator.get(sim)
        hidden = None
        if previous is not None and sim.state.minute > previous[0]:
            hidden = previous[1]
        actions, _, _, _, next_hidden = model.act(
            resource.float(),
            tasks.float(),
            masks.bool(),
            actor_global_state.float(),
            critic_global_state,
            hidden=hidden,
            deterministic=True,
        )
        hidden_by_simulator[sim] = (sim.state.minute, next_hidden.detach())
        return {agent: int(actions[0, index]) for index, agent in enumerate(ids)}

    return act


def run_episode(sim: AeolusSimulator, policy: Policy, seed: int) -> dict[str, object]:
    sim.reset(seed)
    while not sim.state.terminated and not sim.state.truncated:
        sim.decision_step(policy(sim))
    return sim.episode_record()


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 2000,
) -> list[float]:
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def evaluate_pairs(
    policies: dict[str, Policy], simulator: AeolusSimulator, seeds: list[int]
) -> dict[str, object]:
    records = {
        name: [run_episode(simulator, policy, seed) for seed in seeds] for name, policy in policies.items()
    }
    loss = {
        name: np.asarray([item["weighted_loss"] for item in items], dtype=np.float64)
        for name, items in records.items()
    }
    reference_name = "no_aerial" if "no_aerial" in records else next(iter(records))
    summary = {}
    for policy_index, (name, items) in enumerate(records.items()):
        delta = loss[name] - loss[reference_name]
        summary[name] = {
            "episodes": len(items),
            "mean_weighted_loss": float(loss[name].mean()),
            "std_weighted_loss": float(loss[name].std(ddof=1)) if len(items) > 1 else 0.0,
            "weighted_loss_ci95": _bootstrap_mean_interval(
                loss[name],
                seed=93_001 + policy_index,
            ),
            "mean_paired_delta_vs_reference": float(delta.mean()),
            "paired_delta_ci95": _bootstrap_mean_interval(
                delta,
                seed=193_001 + policy_index,
            ),
            "reference_policy": reference_name,
            "escape_rate": float(np.mean([item["escaped"] for item in items])),
            "containment_rate": float(np.mean([item["contained"] for item in items])),
            "mean_blocked_actions": float(np.mean([item["blocked_actions"] for item in items])),
            "mean_flight_min": float(
                np.mean([sum(resource["flight_min"] for resource in item["resource"]) for item in items])
            ),
            "paired_weighted_loss_delta": {
                other_name: {
                    "mean": float((loss[name] - other_loss).mean()),
                    "ci95": _bootstrap_mean_interval(
                        loss[name] - other_loss,
                        seed=293_001 + policy_index * 101 + other_index,
                    ),
                }
                for other_index, (other_name, other_loss) in enumerate(loss.items())
                if other_name != name
            },
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
        "joint_assignment": joint_assignment,
    }
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_policy_network(config.training).to(device)
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
