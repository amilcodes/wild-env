"""Gymnasium environment over the ladder dynamics (one step = one 10-minute tick).

The per-tick environment and the plan benchmark share :func:`transition`, so an
RL policy's return is directly comparable with the exact optimum for the same
instance (:func:`aeolus.ladder.solver.solve`).

Information is handled so the reward cannot leak the hidden forecast: a job
completed before the reveal is credited at its prior-expected value, and the
difference to its realized value is paid at the reveal tick. The undiscounted
return therefore equals the realized net value exactly, and nothing before the
reveal depends on the hidden world.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aeolus.agents.operations import (
    Instance,
    initial_state,
    legal_jobs,
    payoff,
    reveals,
    transition,
    with_forecast,
)
from aeolus.ladder.generator import TIERS, TYPES, generate

KINDS = ("PROTECT", "WETLINE", "HANDLINE", "ACCESS", "SURVEY")
TEAM_TYPES = tuple(TYPES)
VALUE_SCALE = 100.0


class LadderEnv(gym.Env):
    """Multi-resource dispatch with action masks.

    Action: ``MultiDiscrete([max_jobs + 1] * max_resources)``; 0 holds, ``k`` starts job
    slot ``k - 1``. Illegal or conflicting choices (same job twice, joint water
    overrun) resolve to hold in resource order and are counted in ``info["conflicts"]``.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        tiers: tuple[int, ...] = (1, 2, 3),
        seed_range: tuple[int, int] = (1_000_000, 2_000_000),
        instance: Instance | None = None,
        render_mode: str | None = None,
    ):
        if instance is None and not tiers:
            raise ValueError("pass tiers or a fixed instance")
        self.tiers = tuple(tiers) if instance is None else (instance.level,)
        self.seed_range = seed_range
        self.fixed = instance
        self.render_mode = render_mode
        shapes = [TIERS[t] for t in self.tiers if t in TIERS]
        self.max_resources = max([s.resources for s in shapes] + [len(instance.resources) if instance else 0])
        self.max_jobs = max(
            [s.protect + s.access + s.surveys for s in shapes] + [len(instance.jobs) if instance else 0]
        )
        self.max_horizon = max([s.horizon for s in shapes] + [instance.horizon if instance else 0])
        job_features = 1 + len(KINDS) + 11 + self.max_resources
        self.observation_space = spaces.Dict(
            {
                "jobs": spaces.Box(-5.0, 10.0, (self.max_jobs, job_features), np.float32),
                "resources": spaces.Box(0.0, 1.0, (self.max_resources, 3 + len(TEAM_TYPES)), np.float32),
                "global": spaces.Box(0.0, 1.0, (7,), np.float32),
                "action_mask": spaces.Box(0, 1, (self.max_resources, self.max_jobs + 1), np.int8),
            }
        )
        self.action_space = spaces.MultiDiscrete([self.max_jobs + 1] * self.max_resources)
        self.instance: Instance | None = None
        self.world = 0
        self.state = None

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("instance") is not None:
            self.instance = options["instance"]
        elif self.fixed is not None:
            self.instance = self.fixed
        else:
            tier = int(options.get("tier", self.np_random.choice(self.tiers)))
            self.instance = generate(tier, int(self.np_random.integers(*self.seed_range)))
        if len(self.instance.resources) > self.max_resources or len(self.instance.jobs) > self.max_jobs:
            raise ValueError("instance larger than the environment's padded shape")
        world = options.get("world")
        self.world = (
            int(world) if world is not None else int(self.np_random.random() >= self.instance.prior[0])
        )
        self.state = initial_state(self.instance)
        self._credited = 0.0
        if reveals(self.instance, self.state):
            self.state = with_forecast(self.state, self.world)
        self.conflicts = 0
        return self._observation(), self._info()

    def step(self, action):
        if self.state.tick >= self.instance.horizon:
            raise RuntimeError("episode finished; call reset()")
        inst, state = self.instance, self.state
        chosen = [-1] * len(inst.resources)
        taken, water, conflicts = set(), state.water, 0
        for r in range(len(inst.resources)):
            a = int(action[r])
            if a == 0:
                continue
            j = a - 1
            if (
                j < len(inst.jobs)
                and j in legal_jobs(inst, state, r)
                and j not in taken
                and inst.jobs[j].water <= water
            ):
                chosen[r] = j
                taken.add(j)
                water -= inst.jobs[j].water
            else:
                conflicts += 1
        for r in range(len(inst.resources), self.max_resources):
            conflicts += int(int(action[r]) != 0)
        self.conflicts += conflicts
        self.state = transition(inst, state, tuple(chosen))
        if reveals(inst, self.state):
            self.state = with_forecast(self.state, self.world)
        credited = self._credit()
        reward = credited - self._credited
        self._credited = credited
        terminated = self.state.tick >= inst.horizon
        info = self._info(conflicts)
        if terminated:
            info["realized_value"] = float(payoff(inst, self.state, self.world))
            info["world"] = ("alpha", "beta")[self.world]
        return self._observation(), float(reward), terminated, False, info

    def _credit(self) -> float:
        """Net value credited so far: belief-expected until the reveal, realized after it."""
        inst, state = self.instance, self.state
        total = -float(state.cost)
        for j, job in enumerate(inst.jobs):
            if state.completed & (1 << j):
                if state.forecast >= 0:
                    total += job.values[state.forecast]
                else:
                    total += sum(p * v for p, v in zip(inst.prior, job.values, strict=True))
        return total

    # ------------------------------------------------------------ features
    def action_masks(self) -> np.ndarray:
        mask = np.zeros((self.max_resources, self.max_jobs + 1), np.int8)
        mask[:, 0] = 1
        for r in range(len(self.instance.resources)):
            for j in legal_jobs(self.instance, self.state, r):
                if j >= 0:
                    mask[r, j + 1] = 1
        return mask

    def _observation(self):
        inst, state = self.instance, self.state
        T = float(inst.horizon)
        positions = {job.id: j for j, job in enumerate(inst.jobs)}
        unlocks = np.zeros(len(inst.jobs))
        for job in inst.jobs:
            for p in job.prerequisites:
                unlocks[positions[p]] += 1
        jobs = np.zeros(self.observation_space["jobs"].shape, np.float32)
        active = {j for j, _ in state.active if j >= 0}
        for j, job in enumerate(inst.jobs):
            belief = (
                job.values[state.forecast]
                if state.forecast >= 0
                else sum(p * v for p, v in zip(inst.prior, job.values, strict=True))
            )
            row = [1.0]
            row += [float(job.kind == k) for k in KINDS]
            row += [
                job.release / T,
                job.deadline / T,
                (job.deadline - state.tick) / T,
                job.water / max(1.0, inst.water),
                job.cost / VALUE_SCALE,
                job.values[0] / VALUE_SCALE,
                job.values[1] / VALUE_SCALE,
                belief / VALUE_SCALE,
                float(bool(state.completed & (1 << j))),
                float(j in active),
                sum(1 for p in job.prerequisites if not state.completed & (1 << positions[p]))
                + unlocks[j] / 10.0,
            ]
            durations = [d / T for d in job.durations] + [0.0] * (self.max_resources - len(job.durations))
            jobs[j] = row + durations
        resources = np.zeros(self.observation_space["resources"].shape, np.float32)
        for r, name in enumerate(inst.resources):
            busy, finish = state.active[r]
            team = name.rsplit("-", 1)[0]
            resources[r] = [1.0, float(busy >= 0), max(0, finish - state.tick) / T if busy >= 0 else 0.0] + [
                float(team == t) for t in TEAM_TYPES
            ]
        known = state.forecast >= 0
        glob = np.array(
            [
                state.tick / T,
                state.water / max(1.0, inst.water),
                float(known),
                float(known and state.forecast == 0),
                inst.prior[0],
                inst.bulletin_tick / T,
                max(0, inst.bulletin_tick - state.tick) / T,
            ],
            np.float32,
        )
        return {"jobs": jobs, "resources": resources, "global": glob, "action_mask": self.action_masks()}

    def _info(self, conflicts: int = 0) -> dict[str, Any]:
        return {"tick": self.state.tick, "conflicts": conflicts, "instance_sha256": self.instance.identity}

    def render(self):
        if self.render_mode != "ansi":
            return None
        inst, state = self.instance, self.state
        lines = [f"tick {state.tick}/{inst.horizon} water {state.water} forecast {state.forecast}"]
        for r, name in enumerate(inst.resources):
            j, finish = state.active[r]
            lines.append(f"  {name}: " + (f"job {inst.jobs[j].id} until {finish}" if j >= 0 else "idle"))
        return "\n".join(lines)
