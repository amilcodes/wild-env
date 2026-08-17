"""Synchronous environment collection for the included MAPPO baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch

from aeolus.config import ScenarioConfig
from aeolus.core.tasks import critic_global_features
from aeolus.envs.parallel import AeolusParallelEnv
from aeolus.envs.tensor_incident import TensorIncidentEnv
from aeolus.envs.tensor_operations import TensorOperationsEnv
from aeolus.training.networks import TaskPointerActorCritic


@dataclass
class Rollout:
    resource: torch.Tensor
    tasks: torch.Tensor
    masks: torch.Tensor
    actor_global_state: torch.Tensor
    critic_global_state: torch.Tensor
    hidden: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    bootstrap_value: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


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
                "actor_global_state",
                "critic_global_state",
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
            resource, tasks, masks, actor_global_state = _stack_observations(
                self.observations, self.agent_ids, self.device
            )
            critic_global_state = torch.as_tensor(
                np.stack([critic_global_features(env.sim) for env in self.envs]),
                device=self.device,
                dtype=torch.float32,
            )
            actions, logp, _, values, next_hidden = model.act(
                resource,
                tasks,
                masks,
                actor_global_state,
                critic_global_state,
                self.hidden,
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
                ("actor_global_state", actor_global_state),
                ("critic_global_state", critic_global_state),
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
        resource, tasks, masks, actor_global_state = _stack_observations(
            self.observations, self.agent_ids, self.device
        )
        critic_global_state = torch.as_tensor(
            np.stack([critic_global_features(env.sim) for env in self.envs]),
            device=self.device,
            dtype=torch.float32,
        )
        _, bootstrap_value, _ = model(
            resource,
            tasks,
            masks,
            actor_global_state,
            critic_global_state,
            self.hidden,
        )
        return Rollout(
            resource=torch.stack(records["resource"]),
            tasks=torch.stack(records["tasks"]),
            masks=torch.stack(records["masks"]),
            actor_global_state=torch.stack(records["actor_global_state"]),
            critic_global_state=torch.stack(records["critic_global_state"]),
            hidden=torch.stack(records["hidden"]),
            actions=torch.stack(records["actions"]),
            logp=torch.stack(records["logp"]),
            values=torch.stack(records["values"]),
            rewards=torch.stack(records["rewards"]),
            dones=torch.stack(records["dones"]),
            bootstrap_value=bootstrap_value.detach(),
            diagnostics={},
        )


class TensorOperationsCollector:
    """End-to-end device-resident collector for operations pretraining."""

    def __init__(
        self,
        config: ScenarioConfig,
        num_envs: int,
        seed: int,
        device: torch.device,
        hidden_dim: int,
        *,
        max_segments: int,
    ):
        self.env = TensorOperationsEnv(
            config,
            batch_size=num_envs,
            max_segments=max_segments,
            device=device,
            terminate_on_completion=False,
        )
        self.device = device
        self.hidden = torch.zeros(
            (num_envs, self.env.num_resources, hidden_dim),
            device=device,
        )
        self.observation = self.env.reset(seed=seed)
        self.episode_decisions = ceil(config.horizon_min / config.decision_interval_min)
        self.decision_in_episode = 0

    @property
    def num_envs(self) -> int:
        return self.env.batch_size

    @torch.no_grad()
    def collect(self, model: TaskPointerActorCritic, steps: int) -> Rollout:
        records: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "resource",
                "tasks",
                "masks",
                "actor_global_state",
                "critic_global_state",
                "hidden",
                "actions",
                "logp",
                "values",
                "rewards",
                "dones",
            )
        }
        diagnostic_records: dict[str, list[torch.Tensor]] = {
            "delivered_l": [],
            "wasted_l": [],
            "blocked_actions": [],
        }
        incident_diagnostics = isinstance(self.env, TensorIncidentEnv)
        if incident_diagnostics:
            diagnostic_records.update(
                {
                    "expected_loss": [],
                    "burned_fraction": [],
                    "constraint_blocked": [],
                    "constraint_exhaustion": [],
                    "constraint_queue": [],
                    "constraint_waste": [],
                    "contained": [],
                    "escaped": [],
                }
            )
        model.eval()
        for _ in range(steps):
            observation = self.observation
            critic_global_state = self.env.critic_state()
            actions, logp, _, values, next_hidden = model.act(
                observation.resource,
                observation.tasks,
                observation.action_mask,
                observation.global_state,
                critic_global_state,
                self.hidden,
            )
            transition = self.env.step(actions)
            diagnostic_records["delivered_l"].append(transition.delivered_l.detach())
            diagnostic_records["wasted_l"].append(transition.wasted_l.detach())
            diagnostic_records["blocked_actions"].append(transition.blocked_actions.detach())
            if incident_diagnostics:
                diagnostic_records["expected_loss"].append(transition.expected_loss.detach())
                diagnostic_records["burned_fraction"].append(transition.burned_fraction.detach())
                for index, name in enumerate(
                    (
                        "constraint_blocked",
                        "constraint_exhaustion",
                        "constraint_queue",
                        "constraint_waste",
                    )
                ):
                    diagnostic_records[name].append(transition.constraint_costs[:, index].detach())
                diagnostic_records["contained"].append(self.env.state.contained.to(torch.float32).detach())
                diagnostic_records["escaped"].append(self.env.state.escaped.to(torch.float32).detach())
            for key, value in (
                ("resource", observation.resource),
                ("tasks", observation.tasks),
                ("masks", observation.action_mask),
                ("actor_global_state", observation.global_state),
                ("critic_global_state", critic_global_state),
                ("hidden", self.hidden),
                ("actions", actions),
                ("logp", logp),
                ("values", values),
                ("rewards", transition.reward),
                ("dones", transition.done.to(torch.float32)),
            ):
                records[key].append(value.detach())
            next_hidden = next_hidden.masked_fill(
                transition.done[:, None, None],
                0.0,
            )
            self.decision_in_episode += 1
            if self.decision_in_episode >= self.episode_decisions:
                self.observation = self.env.reset()
                self.hidden = torch.zeros_like(next_hidden)
                self.decision_in_episode = 0
            else:
                self.observation = transition.observation
                self.hidden = next_hidden.detach()
        observation = self.observation
        _, bootstrap_value, _ = model(
            observation.resource,
            observation.tasks,
            observation.action_mask,
            observation.global_state,
            self.env.critic_state(),
            self.hidden,
        )
        return Rollout(
            resource=torch.stack(records["resource"]),
            tasks=torch.stack(records["tasks"]),
            masks=torch.stack(records["masks"]),
            actor_global_state=torch.stack(records["actor_global_state"]),
            critic_global_state=torch.stack(records["critic_global_state"]),
            hidden=torch.stack(records["hidden"]),
            actions=torch.stack(records["actions"]),
            logp=torch.stack(records["logp"]),
            values=torch.stack(records["values"]),
            rewards=torch.stack(records["rewards"]),
            dones=torch.stack(records["dones"]),
            bootstrap_value=bootstrap_value.detach(),
            diagnostics={name: torch.stack(values) for name, values in diagnostic_records.items()},
        )


class TensorIncidentCollector(TensorOperationsCollector):
    """Device-resident collector for fire-coupled surrogate pretraining."""

    def __init__(
        self,
        config: ScenarioConfig,
        num_envs: int,
        seed: int,
        device: torch.device,
        hidden_dim: int,
        *,
        max_segments: int,
        grid_size: int,
        fire_substeps: int,
        observation_period_min: int,
        compile_environment: bool,
    ):
        self.env = TensorIncidentEnv(
            config,
            batch_size=num_envs,
            max_segments=max_segments,
            grid_size=grid_size,
            fire_substeps=fire_substeps,
            observation_period_min=observation_period_min,
            device=device,
            terminate_on_completion=False,
            terminate_on_escape=False,
        )
        if compile_environment:
            self.env.compile()
        self.device = device
        self.hidden = torch.zeros(
            (num_envs, self.env.num_resources, hidden_dim),
            device=device,
        )
        self.observation = self.env.reset(seed=seed)
        self.episode_decisions = ceil(config.horizon_min / config.decision_interval_min)
        self.decision_in_episode = 0
