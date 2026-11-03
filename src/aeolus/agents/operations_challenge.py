"""Joint-constraint stress tier with a three-stage access chain and a dependency join."""

import random

from aeolus.agents.operations import Instance, Job


def generate_challenge(family, seed):
    if family != "compound":
        raise ValueError("Level 3 currently supports only the compound task family")
    rng = random.Random(seed)
    resources = tuple(rng.sample(["team-north", "team-west", "team-red", "team-blue"], 2))
    ids = rng.sample(range(11, 198), 20)
    labels = rng.sample(
        [
            "orchard",
            "substation",
            "ranch",
            "tower",
            "school",
            "village",
            "relay",
            "depot",
            "bridge",
            "clinic",
            "intake",
            "shelter",
            "mill",
            "station",
            "farm",
            "plant",
            "camp",
            "reservoir",
            "yard",
            "switchyard",
        ],
        20,
    )
    horizon = 9
    jobs = []

    def add(durations, deadline, values, release=0, prerequisites=(), water=0, cost=2, kind="PROTECT"):
        j = len(jobs)
        jobs.append(
            Job(
                ids[j],
                labels[j],
                kind,
                tuple(durations),
                release,
                deadline,
                tuple(prerequisites),
                water,
                cost,
                tuple(values),
            )
        )
        return ids[j]

    a = add((1, 2), 3, (0, 0), cost=3, kind="ACCESS")
    b = add((2, 1), 4, (0, 0), cost=3, kind="ACCESS")
    c = add((2, rng.choice((1, 3))), 6, (0, 0), prerequisites=(a,), cost=2, kind="ACCESS")
    d = add((rng.choice((1, 2)), rng.choice((1, 3))), 7, (0, 0), prerequisites=(c,), cost=2, kind="ACCESS")
    value = rng.randint(55, 85)
    add((2, 3), rng.randint(6, 7), (value, value), prerequisites=(a,), water=2, cost=3)
    value = rng.randint(65, 100)
    add((3, 2), rng.randint(7, 8), (value, value), prerequisites=(b,), water=2, cost=3)
    value = rng.randint(180, 260)
    add(rng.choice(((2, 3), (3, 2))), horizon, (value, value), prerequisites=(d, b), water=2, cost=4)
    for j in range(6):
        durations = [rng.randint(2, 4), rng.randint(2, 4)]
        if j == 0:
            durations[rng.randrange(2)] = 1
        elif j in (2, 4):
            durations[rng.randrange(2)] = 0
        release = rng.randint(0, 3)
        earliest = release + min(d for d in durations if d > 0)
        deadline = rng.randint(earliest, min(horizon, earliest + rng.randint(1, 3)))
        value = rng.randint(28, 85)
        add(
            durations,
            deadline,
            (value, value),
            release=release,
            water=rng.choice((1, 1, 2)),
            cost=rng.randint(1, 6),
        )
    add((1, 2), 3, (0, 0), kind="SURVEY", cost=5)
    value = rng.randint(140, 200)
    deadline = rng.randint(7, 8)
    add((3, 2), deadline, (value, 5), release=1, water=3, cost=4)
    add((2, 3), deadline, (10, value + rng.randint(-5, 10)), release=2, water=3, cost=4)
    if rng.choice((False, True)):
        jobs = [
            Job(
                j.id,
                j.label,
                j.kind,
                j.durations,
                j.release,
                j.deadline,
                j.prerequisites,
                j.water,
                j.cost,
                j.values[::-1],
            )
            for j in jobs
        ]
    if rng.choice((False, True)):
        jobs = [
            Job(
                j.id,
                j.label,
                j.kind,
                j.durations[::-1],
                j.release,
                j.deadline,
                j.prerequisites,
                j.water,
                j.cost,
                j.values,
            )
            for j in jobs
        ]
    rng.shuffle(jobs)
    water = rng.randint(6, 7)
    bulletin = 5
    return Instance(family, seed, 3, horizon, resources, water, bulletin, tuple(jobs))
