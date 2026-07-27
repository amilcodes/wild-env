"""Standards-compliant PettingZoo Parallel API adapter."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import GLOBAL_FEATURE_DIM, RESOURCE_FEATURE_DIM, TASK_FEATURE_DIM


class AeolusParallelEnv(ParallelEnv):
    metadata = {"name": "aeolus_ia_v0", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(self, config: ScenarioConfig | None = None, render_mode: str | None = None):
        self.config = config or ScenarioConfig()
        self.render_mode = render_mode
        self.sim = AeolusSimulator(self.config)
        self.possible_agents = self.sim.agent_ids
        self.agents = self.possible_agents[:]
        self._observation_space = spaces.Dict(
            {
                "resource": spaces.Box(-np.inf, np.inf, shape=(RESOURCE_FEATURE_DIM,), dtype=np.float32),
                "tasks": spaces.Box(
                    -np.inf, np.inf, shape=(self.config.max_tasks, TASK_FEATURE_DIM), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(self.config.max_tasks),
                "task_valid": spaces.MultiBinary(self.config.max_tasks),
                "global": spaces.Box(-np.inf, np.inf, shape=(GLOBAL_FEATURE_DIM,), dtype=np.float32),
            }
        )
        self._action_space = spaces.Discrete(self.config.max_tasks)

    def observation_space(self, agent: str) -> gymnasium.Space:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return self._observation_space

    def action_space(self, agent: str) -> gymnasium.Space:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return self._action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        observations = self.sim.reset(seed)
        self.agents = self.possible_agents[:]
        infos = {
            agent: {"scenario_seed": self.sim.config.seed if seed is None else seed} for agent in self.agents
        }
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called after episode completion; call reset()")
        expected = set(self.agents)
        missing = expected.difference(actions)
        if missing:
            raise ValueError(f"missing action(s) for active agents: {sorted(missing)}")
        observations, reward, terminated, truncated, infos = self.sim.decision_step(actions)
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        if terminated or truncated:
            self.agents = []
            return {}, rewards, terminations, truncations, infos
        return observations, rewards, terminations, truncations, infos

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        state = self.sim.state
        truth = state.truth
        rows: list[str] = []
        for y in range(0, self.config.height, 2):
            row = []
            for x in range(0, self.config.width, 2):
                if truth.barrier[y, x]:
                    row.append("#")
                elif truth.phase[y, x] == 1:
                    row.append("*")
                elif truth.phase[y, x] == 2:
                    row.append(".")
                elif truth.asset_value[y, x] > 0:
                    row.append("A")
                else:
                    row.append(" ")
            rows.append("".join(row))
        return f"minute={state.minute} active={(truth.phase == 1).sum()}\n" + "\n".join(rows)
