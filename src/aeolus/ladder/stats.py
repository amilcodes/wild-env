"""Instance-level statistics: equal weight per instance, paired tests, Holm correction."""

from __future__ import annotations

import itertools
import math
import random
import statistics


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def bootstrap_ci(values, strata=None, samples=10_000, seed=0, alpha=0.05):
    """Percentile bootstrap of the mean, resampling instances within tier strata."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    groups: dict[object, list[float]] = {}
    for i, v in enumerate(values):
        groups.setdefault(strata[i] if strata else 0, []).append(v)
    total = len(values)
    draws = []
    for _ in range(samples):
        s = 0.0
        for g in groups.values():
            s += sum(g[rng.randrange(len(g))] for _ in g)
        draws.append(s / total)
    draws.sort()
    lo = draws[int(alpha / 2 * samples)]
    hi = draws[min(samples - 1, int((1 - alpha / 2) * samples))]
    return (lo, hi)


def sign_flip_p(differences, samples=20_000, seed=0):
    """Two-sided paired sign-flip permutation p-value for mean difference = 0 (exact when small)."""
    d = [x for x in differences if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    observed = abs(sum(d))
    if n <= 16:
        hits = 0
        for signs in itertools.product((1, -1), repeat=n):
            hits += abs(sum(s * x for s, x in zip(signs, d, strict=True))) >= observed - 1e-12
        return hits / 2**n
    rng = random.Random(seed)
    hits = 1
    for _ in range(samples):
        hits += abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= observed - 1e-12
    return hits / (samples + 1)


def holm(pvalues: dict[object, float]) -> dict[object, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def pairwise(scores: dict[str, dict[str, float]], tiers: dict[str, int], seed=0):
    """scores[model][instance] -> paired comparisons over shared instances."""
    models = sorted(scores)
    raw, rows = {}, {}
    for a, b in itertools.combinations(models, 2):
        shared = sorted(set(scores[a]) & set(scores[b]))
        diffs = [scores[a][k] - scores[b][k] for k in shared]
        strata = [tiers[k] for k in shared]
        raw[(a, b)] = sign_flip_p(diffs, seed=seed)
        rows[(a, b)] = {
            "left": a,
            "right": b,
            "instances": len(shared),
            "mean_difference": mean(diffs),
            "ci95": bootstrap_ci(diffs, strata, seed=seed),
            "left_better_instances": sum(x > 1e-9 for x in diffs),
            "right_better_instances": sum(x < -1e-9 for x in diffs),
            "p_raw": raw[(a, b)],
        }
    adjusted = holm(raw)
    out = []
    for key, row in rows.items():
        row["p_holm"] = adjusted[key]
        lo, hi = row["ci95"]
        row["separated"] = bool(row["p_holm"] < 0.05 and (lo > 0 or hi < 0) and not math.isnan(lo))
        out.append(row)
    return out
