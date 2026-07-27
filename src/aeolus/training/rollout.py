"""Synchronous environment collection for the included MAPPO baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aeolus.config import ScenarioConfig
from aeolus.envs.parallel import AeolusParallelEnv
from aeolus.training.networks import TaskPointerActorCritic


@dataclass
class Rollout:
    resource: torch.Tensor
    tasks: torch.Tensor
    masks: torch.Tensor
    global_state: torch.Tensor
    hidden: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    bootstrap_value: torch.Tensor


def _stack_observations(
    observations: list[dict[str, dict[str, np.ndarray]]], agent_ids: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resources = np.stack([[obs[agent]["resource"] for agent in agent_ids] for obs in observations])
    tasks = np.stack([obs[agent_ids[0]]["tasks"] for obs in observations])
    masks = np.stack([[obs[agent]["action_mask"] for agent in agent_ids] for obs in observations])
    global_state = np.stack([obs[agent_ids[0]]["global"] for obs in observations])
    return (
        torch.as_tensor(resources, device=device, dtype=torch.float32),
        torch.as_tensor(tasks, device=device, dtype=torch.float32),
        torch.as_tensor(masks, device=device, dtype=torch.bool),
        torch.as_tensor(global_state, device=device, dtype=torch.float32),
    )


class SynchronousCollector:
    def __init__(
        self, config: ScenarioConfig, num_envs: int, seed: int, device: torch.device, hidden_dim: int
    ):
        self.envs = [AeolusParallelEnv(config) for _ in range(num_envs)]
        self.agent_ids = self.envs[0].possible_agents
        self.seed = seed
        self.episode_index = np.arange(num_envs, dtype=np.int64)
        self.device = device
        self.hidden = torch.zeros((num_envs, len(self.agent_ids), hidden_dim), device=device)
        self.observations = [env.reset(seed=seed + int(index))[0] for index, env in enumerate(self.envs)]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @torch.no_grad()
    def collect(self, model: TaskPointerActorCritic, steps: int) -> Rollout:
        records: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "resource",
                "tasks",
                "masks",
                "global_state",
                "hidden",
                "actions",
                "logp",
                "values",
                "rewards",
                "dones",
            )
        }
        model.eval()
        for _ in range(steps):
            resource, tasks, masks, global_state = _stack_observations(
                self.observations, self.agent_ids, self.device
            )
            actions, logp, _, values, next_hidden = model.act(
                resource, tasks, masks, global_state, self.hidden
            )
            action_cpu = actions.detach().cpu().numpy()
            rewards, dones, next_observations = [], [], []
            for env_index, env in enumerate(self.envs):
                action_dict = {
                    agent: int(action_cpu[env_index, index]) for index, agent in enumerate(self.agent_ids)
                }
                obs, reward_dict, terminations, truncations, _ = env.step(action_dict)
                done = bool(terminations[self.agent_ids[0]] or truncations[self.agent_ids[0]])
                rewards.append(float(reward_dict[self.agent_ids[0]]))
                dones.append(done)
                if done:
                    self.episode_index[env_index] += self.num_envs
                    obs, _ = env.reset(seed=self.seed + int(self.episode_index[env_index]))
                    next_hidden[env_index].zero_()
                next_observations.append(obs)
            for key, value in (
                ("resource", resource),
                ("tasks", tasks),
                ("masks", masks),
                ("global_state", global_state),
                ("hidden", self.hidden),
                ("actions", actions),
                ("logp", logp),
                ("values", values),
            ):
                records[key].append(value.detach())
            records["rewards"].append(torch.tensor(rewards, device=self.device, dtype=torch.float32))
            records["dones"].append(torch.tensor(dones, device=self.device, dtype=torch.float32))
            self.observations = next_observations
            self.hidden = next_hidden.detach()
        resource, tasks, masks, global_state = _stack_observations(
            self.observations, self.agent_ids, self.device
        )
        _, bootstrap_value, _ = model(resource, tasks, masks, global_state, self.hidden)
        return Rollout(
            resource=torch.stack(records["resource"]),
            tasks=torch.stack(records["tasks"]),
            masks=torch.stack(records["masks"]),
            global_state=torch.stack(records["global_state"]),
            hidden=torch.stack(records["hidden"]),
            actions=torch.stack(records["actions"]),
            logp=torch.stack(records["logp"]),
            values=torch.stack(records["values"]),
            rewards=torch.stack(records["rewards"]),
            dones=torch.stack(records["dones"]),
            bootstrap_value=bootstrap_value.detach(),
        )
