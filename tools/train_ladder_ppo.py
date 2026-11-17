"""Learnability check for the ladder RL environment: masked PPO vs greedy, scored against the optimum.

The policy scores every (team, job) pair from shared job/team encoders and samples
teams in order, masking jobs already taken this tick, so joint actions never conflict.
Held-out evaluation uses seeds disjoint from training and reports the episode return as
a percentage of the exact information-matched optimum, averaged over both forecast worlds.

This is a smoke-scale CPU run (minutes), not a training campaign.

    python tools/train_ladder_ppo.py --tiers 1 2 --updates 150 --out runs/ladder/ppo.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aeolus.ladder import generate, solve
from aeolus.ladder.env import LadderEnv


class Policy(nn.Module):
    def __init__(self, job_dim, res_dim, glob_dim, hidden=128):
        super().__init__()
        self.job = nn.Sequential(nn.Linear(job_dim + glob_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.res = nn.Sequential(nn.Linear(res_dim + glob_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pair = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.hold = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, jobs, res, glob):
        # jobs (B,N,Fj) res (B,R,Fr) glob (B,G) -> logits (B,R,N+1), value (B,)
        B, N, _ = jobs.shape
        R = res.shape[1]
        gj = glob[:, None, :].expand(B, N, -1)
        gr = glob[:, None, :].expand(B, R, -1)
        je = self.job(torch.cat([jobs, gj], -1))
        re = self.res(torch.cat([res, gr], -1))
        pooled = (je * jobs[..., :1]).sum(1) / jobs[..., :1].sum(1).clamp(min=1)
        pair = torch.cat(
            [
                je[:, None].expand(B, R, N, -1),
                re[:, :, None].expand(B, R, N, -1),
                pooled[:, None, None].expand(B, R, N, -1),
            ],
            -1,
        )
        logits = self.pair(pair).squeeze(-1)
        hold = self.hold(torch.cat([re, pooled[:, None].expand(B, R, -1)], -1))
        value = self.value(torch.cat([pooled, re.mean(1), re.max(1).values], -1)).squeeze(-1)
        return torch.cat([hold, logits], -1), value


def sequential_sample(logits, mask, greedy=False):
    """Sample teams in order; a job chosen by an earlier team is masked for later ones."""
    B, R, A = logits.shape
    mask = mask.clone()
    actions = torch.zeros(B, R, dtype=torch.long)
    used_masks = torch.zeros_like(mask)
    logp = torch.zeros(B)
    entropy = torch.zeros(B)
    for r in range(R):
        m = mask[:, r].clone()
        used_masks[:, r] = m
        dist = torch.distributions.Categorical(logits=logits[:, r].masked_fill(m == 0, -1e9))
        a = dist.probs.argmax(-1) if greedy else dist.sample()
        actions[:, r] = a
        logp += dist.log_prob(a)
        entropy += dist.entropy()
        for b in torch.nonzero(a > 0).flatten().tolist():
            mask[b, r + 1 :, a[b]] = 0
    return actions, logp, entropy, used_masks


def evaluate_logp(logits, used_masks, actions):
    dists = torch.distributions.Categorical(logits=logits.masked_fill(used_masks == 0, -1e9))
    return dists.log_prob(actions).sum(-1), dists.entropy().sum(-1)


def to_tensors(obs_list):
    return (
        torch.tensor(np.stack([o["jobs"] for o in obs_list])),
        torch.tensor(np.stack([o["resources"] for o in obs_list])),
        torch.tensor(np.stack([o["global"] for o in obs_list])),
        torch.tensor(np.stack([o["action_mask"] for o in obs_list]), dtype=torch.long),
    )


def greedy_action(env):
    """Per-tick greedy over the env's own mask: belief value net of cost per tick of work."""
    inst, state = env.instance, env.state
    action = np.zeros(env.max_resources, dtype=np.int64)
    taken, water = set(), state.water
    mask = env.action_masks()
    for r in range(len(inst.resources)):
        best, best_u = 0, 0.0
        for a in np.flatnonzero(mask[r])[1:]:
            j = a - 1
            job = inst.jobs[j]
            if j in taken or job.water > water:
                continue
            v = (
                job.values[state.forecast]
                if state.forecast >= 0
                else sum(p * x for p, x in zip(inst.prior, job.values, strict=True))
            )
            u = (v - job.cost) / job.durations[r]
            if u > best_u:
                best, best_u = a, u
        if best:
            action[r] = best
            taken.add(best - 1)
            water -= inst.jobs[best - 1].water
    return action


def run_episode(env, instance, world, policy=None, mode="policy"):
    obs, _ = env.reset(options={"instance": instance, "world": world})
    total, done = 0.0, False
    rng = np.random.default_rng(world)
    while not done:
        if mode == "greedy":
            action = greedy_action(env)
        elif mode == "random":
            mask = obs["action_mask"]
            action = np.array([rng.choice(np.flatnonzero(mask[r])) for r in range(env.max_resources)])
        else:
            with torch.no_grad():
                j, r, g, m = to_tensors([obs])
                logits, _ = policy(j, r, g)
                action = sequential_sample(logits, m, greedy=True)[0][0].numpy()
        obs, reward, done, _, _ = env.step(action)
        total += reward
    return total


def held_out_scores(env, tiers, seeds, policy):
    rows = {m: [] for m in ("policy", "greedy", "random")}
    for tier in tiers:
        for seed in seeds:
            instance = generate(tier, seed)
            optimum = solve(instance).value
            for mode in rows:
                value = sum(
                    p * run_episode(env, instance, w, policy, mode) for w, p in enumerate(instance.prior)
                )
                rows[mode].append(100 * value / optimum)
    return {m: float(np.mean(v)) for m, v in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=48, help="episodes per update")
    parser.add_argument("--eval-seeds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/ladder/ppo.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = LadderEnv(tiers=tuple(args.tiers), seed_range=(1_000_000, 9_000_000))
    obs, _ = env.reset(seed=args.seed)
    policy = Policy(obs["jobs"].shape[-1], obs["resources"].shape[-1], obs["global"].shape[-1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    eval_seeds = range(500_000, 500_000 + args.eval_seeds)  # disjoint from training seed_range
    history = []
    began = time.time()
    before = held_out_scores(env, args.tiers, eval_seeds, policy)
    print(f"before training: {before}", flush=True)
    for update in range(args.updates):
        batch = {k: [] for k in ("obs", "act", "mask", "logp", "ret", "val")}
        returns = []
        for _ in range(args.episodes):
            obs, _ = env.reset()
            done, rewards, values = False, [], []
            while not done:
                with torch.no_grad():
                    j, r, g, m = to_tensors([obs])
                    logits, value = policy(j, r, g)
                    action, logp, _, used = sequential_sample(logits, m)
                batch["obs"].append(obs)
                batch["act"].append(action[0])
                batch["mask"].append(used[0])
                batch["logp"].append(logp[0])
                values.append(value[0].item())
                obs, reward, done, _, _ = env.step(action[0].numpy())
                rewards.append(reward / 100.0)
            g_t, ret = 0.0, []
            for reward in reversed(rewards):
                g_t = reward + g_t
                ret.append(g_t)
            batch["ret"].extend(reversed(ret))
            batch["val"].extend(values)
            returns.append(sum(rewards) * 100)
        j, r, g, _ = to_tensors(batch["obs"])
        act = torch.stack(batch["act"])
        used = torch.stack(batch["mask"])
        old_logp = torch.stack(batch["logp"])
        ret = torch.tensor(batch["ret"], dtype=torch.float32)
        adv = ret - torch.tensor(batch["val"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(ret)).split(256):
                logits, value = policy(j[idx], r[idx], g[idx])
                logp, entropy = evaluate_logp(logits, used[idx], act[idx])
                ratio = (logp - old_logp[idx]).exp()
                clipped = torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx])
                loss = -clipped.mean() + 0.5 * (value - ret[idx]).pow(2).mean() - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        history.append(float(np.mean(returns)))
        if update % 10 == 0 or update == args.updates - 1:
            print(
                f"update {update:4d} mean train return {history[-1]:7.1f} ({time.time() - began:.0f}s)",
                flush=True,
            )
    after = held_out_scores(env, args.tiers, eval_seeds, policy)
    print(f"after training:  {after}", flush=True)
    result = {
        "tiers": args.tiers,
        "updates": args.updates,
        "episodes_per_update": args.episodes,
        "held_out_seeds": [eval_seeds.start, eval_seeds.stop],
        "held_out_score_pct_of_optimum": {"before": before, "after": after},
        "train_return_history": history,
        "seconds": time.time() - began,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
