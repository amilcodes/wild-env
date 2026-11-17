"""Prototype capacity-bound generator used for the packing probe (superseded by generator v3 tiers 6-8)."""
import math, random, time, statistics as st
from aeolus.agents.operations import Instance, Job
from aeolus.ladder.generator import TYPES, _duration, LABELS
from aeolus.ladder import solve, evaluate_plan
from aeolus.ladder.baselines import baseline_plan

def packing(seed, R=5, T=18, overload=1.8, spread=0.15, chains=2, uncertain=True):
    rng = random.Random(f"pack:{R}:{T}:{overload}:{seed}")
    types = ["engine", "crew", "dozer", "heli"] + [rng.choice(["engine","crew","engine","dozer","heli"]) for _ in range(R-4)]
    rng.shuffle(types)
    counts, names = {}, []
    for k in types:
        counts[k] = counts.get(k, 0) + 1; names.append(f"{k}-{counts[k]}")
    bases = [(rng.uniform(0,1), rng.uniform(0,1)) for _ in types]
    bulletin = round(T * rng.uniform(0.35, 0.5)) if uncertain else T
    jobs, ids = [], rng.sample(range(100, 1000), 200)
    labels = LABELS + [f"site-{i}" for i in range(200)]
    rng.shuffle(labels)
    def add(kind, pos, values, release, deadline, prereq=(), water=0, cost=1):
        d = tuple(_duration(t, b, kind, pos) for t, b in zip(types, bases))
        mn = min(x for x in d if x > 0)
        deadline = min(T, max(deadline, release + mn))
        jobs.append(Job(ids[len(jobs)], labels[len(jobs)], kind, d, release, deadline, tuple(prereq), water, cost, values))
        return ids[len(jobs)-1], release + mn
    def world(v):
        if not uncertain or rng.random() < 0.5: return (v, v), False
        r = int(round(v * rng.uniform(0, 0.15)))
        return ((v, r) if rng.random() < 0.5 else (r, v)), True
    # access chains with valuable payoffs
    for c in range(chains):
        prev, ready = (), 0
        pos = (rng.uniform(0.3,1), rng.uniform(0.3,1))
        for _ in range(rng.randint(2, 3)):
            jid, e = add("ACCESS", pos, (0,0), 0, T, prev, cost=rng.randint(1,3)); prev = (jid,); ready = max(ready, e) + 1
        for _ in range(2):
            kind = rng.choice(["PROTECT","WETLINE","HANDLINE"])
            pos2 = (rng.uniform(0,1), rng.uniform(0,1))
            mn = min(x for x in (_duration(t,b,kind,pos2) for t,b in zip(types,bases)) if x>0)
            vals, spec = world(int(mn * rng.uniform(16, 22)))
            dl = min(T, ready + mn + rng.randint(2, 6))
            if spec: dl = max(dl, min(T, bulletin + mn + rng.randint(0, 4)))
            add(kind, pos2, vals, 0, dl, prev, water=rng.randint(1,2) if kind=="WETLINE" else 0, cost=rng.randint(1,3))
    work = 0
    while work < overload * R * T:
        kind = rng.choice(["PROTECT","WETLINE","HANDLINE"])
        pos = (rng.uniform(0,1), rng.uniform(0,1))
        d = [_duration(t,b,kind,pos) for t,b in zip(types,bases)]
        mn = min(x for x in d if x>0)
        release = 0 if rng.random() < 0.5 else rng.randint(0, T//2)
        deadline = min(T, release + mn + rng.randint(2, T))
        vals, spec = world(int(round(mn * 12 * rng.uniform(1-spread, 1+spread))))
        if spec: deadline = max(deadline, min(T, bulletin + mn + rng.randint(0, 4)))
        add(kind, pos, vals, release, deadline, water=rng.randint(1,3) if kind=="WETLINE" else 0, cost=rng.randint(1,3))
        work += mn
    for _ in range(2 if uncertain else 0):
        add("SURVEY", (rng.uniform(0,1), rng.uniform(0,1)), (0,0), 0, T, cost=rng.randint(2,6))
    wet = sum(j.water for j in jobs)
    p = rng.choice([0.5,0.6,0.7]); prior = (p, round(1-p,10)) if rng.random()<0.5 else (round(1-p,10), p)
    order = list(range(len(jobs))); rng.shuffle(order)
    return Instance("ladder-pack", seed, 0, T, tuple(names), max(2, math.ceil(wet*0.45)), bulletin, tuple(jobs[i] for i in order), prior if uncertain else (0.5,0.5))

if __name__ == "__main__":
    for cfg in [dict(R=4, T=14), dict(R=5, T=18), dict(R=6, T=20)]:
        rows = []
        for seed in range(4):
            inst = packing(seed, **cfg); t = time.time(); ref = solve(inst, time_limit=300); dt = time.time()-t
            o = evaluate_plan(inst, ref.plan)
            util = sum(inst.jobs[j].durations[r] for k,r,j in o.traces[0].starts)/(len(inst.resources)*inst.horizon)
            s = 100*evaluate_plan(inst, baseline_plan(inst,"search")).expected_value/ref.value
            g = 100*evaluate_plan(inst, baseline_plan(inst,"greedy")).expected_value/ref.value
            rows.append((len(inst.jobs), util, g, s, dt))
        print(cfg, "jobs", st.mean(r[0] for r in rows), "util %.2f greedy %.1f search %.1f solve max %.1fs" % (st.mean(r[1] for r in rows), st.mean(r[2] for r in rows), st.mean(r[3] for r in rows), max(r[4] for r in rows)))
