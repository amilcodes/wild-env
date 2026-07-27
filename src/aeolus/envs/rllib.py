"""Optional RLlib adapter kept separate from the core dependency set."""

from __future__ import annotations

from typing import Any

from aeolus.config import ScenarioConfig
from aeolus.envs.parallel import AeolusParallelEnv

try:  # Importing Aeolus must not require Ray on a documentation/test install.
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - exercised on optional dependency installs
    MultiAgentEnv = None  # type: ignore[assignment,misc]


if MultiAgentEnv is not None:

    class AeolusRLlibEnv(MultiAgentEnv):
        """RLlib dict-based multi-agent adapter using the same truth simulator."""

        def __init__(self, env_config: dict[str, Any] | None = None):
            super().__init__()
            config = ScenarioConfig(**(env_config or {}))
            self.env = AeolusParallelEnv(config)
            self.possible_agents = self.env.possible_agents
            self.agents = self.env.agents
            self.observation_spaces = {
                agent: self.env.observation_space(agent) for agent in self.possible_agents
            }
            self.action_spaces = {agent: self.env.action_space(agent) for agent in self.possible_agents}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            observations, infos = self.env.reset(seed=seed, options=options)
            self.agents = self.env.agents
            return observations, infos

        def step(self, action_dict: dict[str, int]):
            observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
            episode_done = not self.env.agents
            terminations["__all__"] = episode_done and any(terminations.values())
            truncations["__all__"] = episode_done and any(truncations.values())
            self.agents = self.env.agents
            return observations, rewards, terminations, truncations, infos

else:

    class AeolusRLlibEnv:  # pragma: no cover - helpful error path only
        def __init__(self, *_: object, **__: object):
            raise ImportError("Install Aeolus with `pip install -e '.[rllib]'` to use the RLlib adapter.")
