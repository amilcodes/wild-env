"""Tiered incident generator.

Each tier fixes the *shape* of a problem (teams, jobs, horizon, dependency depth,
forecast uncertainty). The seed fixes everything else. Generation never looks at
model or baseline outcomes, so no tier is filtered toward or away from any policy.

Durations come from a small spatial model: every team has a base and a type,
every job has a position and a kind, and a job's duration for a team is its work
time plus the round trip from that team's base. Durations are therefore
correlated in a realistic way (a helicopter is fast everywhere, a hand crew far
from its base is slow) without being a lookup puzzle.

Two regimes share this generator:

- Tiers 1-5 (*windows*): a fixed number of assets with release windows spread over
  the incident, access chains, joins, scarce water and a forecast. Errors here are
  mostly about reading constraints, chains and the forecast correctly.
- Tiers 6-8 (*packing*): work exceeds team capacity (``overload`` x teams x horizon),
  windows are wide and value is roughly proportional to work (+/-15%). Any sensible
  plan fills the teams; the last points require exact assignment of jobs to the
  team that does them fastest, interval packing, and water allocation. This is a
  generalized assignment problem with time windows and is where strong planners
  separate from each other.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from aeolus.agents.operations import Instance, Job

FAMILY = "ladder"


@dataclass(frozen=True)
class TierSpec:
    resources: int
    protect: int  # PROTECT / WETLINE / HANDLINE jobs
    access: int  # ACCESS jobs (value 0, unlock others)
    horizon: int
    chain_depth: int
    uncertain: bool
    surveys: int
    joins: int  # high-value jobs that need two separate access chains
    overload: float = 0.0  # >0: packing regime; free-standing work = overload x teams x horizon


GENERATOR_VERSION = 3  # v1/v2 were used only for development calibration runs

TIERS: dict[int, TierSpec] = {
    1: TierSpec(
        resources=2, protect=7, access=1, horizon=10, chain_depth=1, uncertain=False, surveys=0, joins=0
    ),
    2: TierSpec(
        resources=3, protect=10, access=2, horizon=12, chain_depth=1, uncertain=True, surveys=1, joins=0
    ),
    3: TierSpec(
        resources=3, protect=14, access=3, horizon=16, chain_depth=2, uncertain=True, surveys=1, joins=1
    ),
    4: TierSpec(
        resources=4, protect=19, access=4, horizon=18, chain_depth=2, uncertain=True, surveys=2, joins=1
    ),
    5: TierSpec(
        resources=5, protect=24, access=5, horizon=20, chain_depth=3, uncertain=True, surveys=2, joins=2
    ),
    6: TierSpec(4, 0, 5, 14, chain_depth=3, uncertain=True, surveys=2, joins=0, overload=1.8),
    7: TierSpec(5, 0, 6, 18, chain_depth=3, uncertain=True, surveys=2, joins=1, overload=1.8),
    8: TierSpec(6, 0, 6, 20, chain_depth=3, uncertain=True, surveys=2, joins=1, overload=1.8),
}

# Team types: one-way speed in grid units per tick and work ticks by job kind (None = incapable).
TYPES = {
    "engine": {"speed": 2.0, "PROTECT": 2, "WETLINE": 2, "HANDLINE": None, "ACCESS": None, "SURVEY": 2},
    "crew": {"speed": 1.0, "PROTECT": 2, "WETLINE": None, "HANDLINE": 2, "ACCESS": 2, "SURVEY": None},
    "dozer": {"speed": 1.2, "PROTECT": None, "WETLINE": None, "HANDLINE": 1, "ACCESS": 1, "SURVEY": None},
    "heli": {"speed": 4.0, "PROTECT": None, "WETLINE": 1, "HANDLINE": None, "ACCESS": None, "SURVEY": 1},
}
# Every incident needs every capability; small tiers draw from these guaranteed mixes.
BASE_MIXES = {
    2: [("engine", "crew"), ("engine", "dozer")],
    3: [("engine", "crew", "heli"), ("engine", "dozer", "heli"), ("engine", "crew", "dozer")],
}
LABELS = (
    "orchard substation ranch tower school village relay depot bridge clinic intake shelter mill "
    "station farm plant camp reservoir yard switchyard lodge winery hatchery quarry kiln granary "
    "chapel marina cabin dairy observatory library bakery foundry pumphouse sawmill vineyard airstrip "
    "hospital museum stable silo"
).split()


def _duration(team: str, base, kind: str, position) -> int:
    work = TYPES[team][kind]
    if work is None:
        return 0
    one_way = math.dist(base, position) * 3.0 / TYPES[team]["speed"]
    return max(1, min(8, work + math.ceil(2 * one_way - 0.25)))


def generate(tier: int, seed: int) -> Instance:
    """Deterministic instance for (tier, seed). Same arguments always give the same bytes."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier}; expected one of {sorted(TIERS)}")
    spec = TIERS[tier]
    rng = random.Random(f"wildfire-ops-v3:g{GENERATOR_VERSION}:{tier}:{seed}")
    horizon = spec.horizon
    early, late = (0.35, 0.5) if spec.overload else (0.4, 0.55)
    bulletin = max(2, round(horizon * rng.uniform(early, late))) if spec.uncertain else horizon

    # Teams: guaranteed capability coverage, then random extra types.
    if spec.resources in BASE_MIXES:
        types = list(rng.choice(BASE_MIXES[spec.resources]))
    else:
        types = ["engine", "crew", "dozer", "heli"]
        types += [
            rng.choice(["engine", "crew", "engine", "dozer", "heli"]) for _ in range(spec.resources - 4)
        ]
    rng.shuffle(types)
    counts: dict[str, int] = {}
    names = []
    for kind in types:
        counts[kind] = counts.get(kind, 0) + 1
        names.append(f"{kind}-{counts[kind]}")
    bases = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in types]
    capable = set().union(*({k for k, v in TYPES[t].items() if k != "speed" and v} for t in types))

    ids = rng.sample(range(100, 1000), 400)
    labels = rng.sample(LABELS, len(LABELS)) + [f"site-{k}" for k in range(len(LABELS), 400)]
    jobs: list[Job] = []

    def add(kind, position, *, value=(0, 0), release=0, deadline=None, prerequisites=(), water=0, cost=1):
        durations = tuple(_duration(t, b, kind, position) for t, b in zip(types, bases, strict=True))
        positive = [d for d in durations if d > 0]
        if not positive:
            raise AssertionError("generator produced an impossible job")
        if deadline is None:
            deadline = horizon
        deadline = min(horizon, max(deadline, release + min(positive)))
        j = len(jobs)
        jobs.append(
            Job(
                ids[j],
                labels[j],
                kind,
                durations,
                release,
                deadline,
                tuple(prerequisites),
                water,
                cost,
                value,
            )
        )
        return ids[j], release + min(positive)

    def asset_value(critical=False):
        if critical:
            return rng.randint(140, 240)
        return int(round(math.exp(rng.uniform(math.log(15), math.log(110)))))

    def exposure(value, deadline):
        """Values by forecast world and a deadline. Corridor-specific assets burn later."""
        if not spec.uncertain or rng.random() < (0.5 if spec.overload else 0.4):
            return (value, value), deadline
        residual = int(round(value * rng.uniform(0.0, 0.15)))
        values = (value, residual) if rng.random() < 0.5 else (residual, value)
        return values, min(horizon, max(deadline, bulletin + rng.randint(1, 5)))

    protect_kinds = [k for k in ("PROTECT", "WETLINE", "HANDLINE") if k in capable]

    def shortest(kind, position):
        return min(
            d for d in (_duration(t, b, kind, position) for t, b in zip(types, bases, strict=True)) if d > 0
        )

    def work_value(kind, position, low, high):
        """Packing regime: value proportional to the fastest team's work."""
        return int(round(shortest(kind, position) * rng.uniform(low, high)))

    # Access chains. Each chain is a path of ACCESS jobs; its end unlocks a cluster of assets.
    chain_ends: list[tuple[int, int]] = []  # (job id, earliest completion)
    remaining_access = spec.access
    while remaining_access > 0:
        length = min(remaining_access, rng.randint(1, spec.chain_depth))
        remaining_access -= length
        previous, ready = (), 0
        position = (rng.uniform(0.2, 1), rng.uniform(0.2, 1))
        for _ in range(length):
            job_id, earliest = add(
                "ACCESS",
                position,
                deadline=horizon,
                prerequisites=previous,
                cost=rng.randint(1, 3),
            )
            ready = max(ready, earliest) + (0 if not previous else 1)
            previous = (job_id,)
        chain_ends.append((previous[0], ready))

    # Joins: critical assets needing two different chain ends (falls back to one if only one chain).
    packing = spec.overload > 0
    protect_left = 2 * len(chain_ends) + spec.joins if packing else spec.protect
    for _ in range(spec.joins):
        if protect_left == 0 or not chain_ends:
            break
        picks = rng.sample(chain_ends, min(2, len(chain_ends)))
        kind = rng.choice(protect_kinds)
        position = (rng.uniform(0.3, 1), rng.uniform(0.3, 1))
        value = work_value(kind, position, 20, 28) if packing else asset_value(critical=True)
        water = rng.randint(1, 3) if kind == "WETLINE" else 0
        earliest = max(e for _, e in picks) + 2
        values, deadline = exposure(value, min(horizon, earliest + rng.randint(2, 6)))
        add(
            kind,
            position,
            value=values,
            deadline=deadline,
            prerequisites=tuple(p for p, _ in picks),
            water=water,
            cost=rng.randint(2, 5),
        )
        protect_left -= 1

    # Each chain end unlocks one or two assets (one of them often critical).
    for end, ready in chain_ends:
        for k in range(2 if packing else rng.randint(1, 2)):
            if protect_left == 0:
                break
            kind = rng.choice(protect_kinds)
            position = (rng.uniform(0.2, 1), rng.uniform(0.2, 1))
            critical = k == 0 and rng.random() < 0.6
            value = work_value(kind, position, 16, 22) if packing else asset_value(critical)
            water = rng.randint(1, 3) if kind == "WETLINE" else 0
            values, deadline = exposure(value, min(horizon, ready + 2 + rng.randint(1, 6)))
            add(
                kind,
                position,
                value=values,
                deadline=deadline,
                prerequisites=(end,),
                water=water,
                cost=rng.randint(1, 4),
            )
            protect_left -= 1

    # Free-standing assets. Release windows are spread over the whole incident with little
    # slack, and value scales with the work involved (+/-25%), so ranking by value density
    # does not decide the plan: each team's day is a weighted interval-scheduling problem.
    # Packing regime: wide windows and work beyond team capacity, value ~ work (+/-15%).
    work = 0
    while packing and work < spec.overload * spec.resources * horizon:
        kind = rng.choice(protect_kinds)
        position = (rng.uniform(0, 1), rng.uniform(0, 1))
        fastest = shortest(kind, position)
        release = 0 if rng.random() < 0.5 else rng.randint(0, horizon // 2)
        deadline = min(horizon, release + fastest + rng.randint(2, horizon))
        values, deadline = exposure(int(round(fastest * 12 * rng.uniform(0.85, 1.15))), deadline)
        add(
            kind,
            position,
            value=values,
            release=release,
            deadline=deadline,
            water=rng.randint(1, 3) if kind == "WETLINE" else 0,
            cost=rng.randint(1, 3),
        )
        work += fastest

    for _ in range(protect_left):
        kind = rng.choice(protect_kinds)
        position = (rng.uniform(0, 1), rng.uniform(0, 1))
        durations = [_duration(t, b, kind, position) for t, b in zip(types, bases, strict=True)]
        shortest = min(d for d in durations if d > 0)
        release = 0 if rng.random() < 0.2 else rng.randint(0, max(0, horizon - shortest - 1))
        deadline = min(horizon, release + shortest + rng.choice([0, 1, 1, 2, 2, 3, 4]))
        if rng.random() < 0.08:
            value = asset_value(critical=True)
        else:
            value = int(round(shortest * rng.uniform(9, 15) * rng.uniform(0.8, 1.25)))
        water = rng.randint(1, 3) if kind == "WETLINE" else 0
        values, deadline = exposure(value, deadline)
        add(
            kind,
            position,
            value=values,
            release=release,
            deadline=deadline,
            water=water,
            cost=rng.randint(1, 4),
        )

    for _ in range(spec.surveys):
        add(
            "SURVEY",
            (rng.uniform(0, 1), rng.uniform(0, 1)),
            value=(0, 0),
            deadline=horizon,
            cost=rng.randint(2, 6),
        )

    wet_demand = sum(job.water for job in jobs)
    share = (0.4, 0.5) if packing else (0.4, 0.6)
    water = max(2, math.ceil(wet_demand * rng.uniform(*share))) if wet_demand else 0
    prior = (0.5, 0.5)
    if spec.uncertain:
        p = rng.choice([0.5, 0.6, 0.7])
        prior = (p, round(1 - p, 10)) if rng.random() < 0.5 else (round(1 - p, 10), p)
    order = list(range(len(jobs)))
    rng.shuffle(order)
    return Instance(
        FAMILY,
        seed,
        tier,
        horizon,
        tuple(names),
        water,
        bulletin,
        tuple(jobs[i] for i in order),
        prior,
    )
