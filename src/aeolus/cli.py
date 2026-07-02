"""Small diagnostics commands; training and evaluation have dedicated CLIs."""

from __future__ import annotations

import argparse

from aeolus.config import load_config
from aeolus.envs.parallel import AeolusParallelEnv
from aeolus.policies import greedy_value


def rollout_main() -> None:
    parser = argparse.ArgumentParser(description="Run one Aeolus heuristic episode")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    env = AeolusParallelEnv(config.scenario, render_mode="ansi")
    env.reset(seed=args.seed)
    while env.agents:
        actions = greedy_value(env.sim)
        _, rewards, terminations, truncations, _ = env.step(actions)
        print(env.render())
        print({"reward": rewards, "terminated": terminations, "truncated": truncations})
