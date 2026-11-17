# Wildfire ops ladder (`wildfire-ops-v3`)

A difficulty-graded planning benchmark and RL environment. Every instance has a
**proven optimum**, so a score is simply "percent of the best achievable expected
value", at every size. Difficulty rises along one ladder (more teams, jobs, horizon,
dependency depth, and forecast uncertainty) instead of a handful of hand-picked
stress cases, so the same suite places small models, mid models and frontier models
on one scale.

It replaces the v2 operations lane as the primary agent benchmark. v2 was exact but
tiny (2 teams, at most 11 jobs, 9 ticks): GPT-6.1 Sol and GPT-6 Astra reached zero
regret on every instance, so it could not order frontier models. The ladder keeps the
v2 dynamics (same `transition` function) and scales them.

## Task

The model is the planning section chief for a synthetic incident. It receives one
public packet and returns one **contingent incident action plan**:

```json
{"rationale": "...",
 "plan": [{"resource": "engine-1", "job": 412, "start": 0, "branch": "both"},
          {"resource": "heli-1",   "job": 377, "start": 6, "branch": "alpha"}]}
```

- Teams have types (engine, crew, dozer, heli). Each job lists a duration per capable
  team (travel + work + return, from a spatial model), a release tick, a deadline, a
  start cost, a water requirement, prerequisites, and a value in each forecast world.
- ACCESS jobs are worth nothing but unlock other jobs (chains up to depth 3, and
  "join" jobs that need two chains). Water is a shared, non-refillable budget.
- The forecast (corridor alpha or beta, public prior) is revealed at a bulletin tick
  or when the first SURVEY job completes. Corridor-specific assets burn later, so
  waiting, surveying early, or committing blind is a real tradeoff.
- `branch: both` entries run in either world; `alpha`/`beta` entries run only in
  that world and must start at or after the reveal. Earlier ones are rejected, which
  makes every plan non-anticipative.

Execution is deterministic and shares v2's tick dynamics. Each team works through its
entries in `start` order; an entry starts at the first tick at or after `start` when
the team is free. If the job is not startable then, the entry is skipped (no cost, no
value) and logged. The exact rules the model sees are in
[`src/aeolus/ladder/prompt.py`](../src/aeolus/ladder/prompt.py).

**Score** = 100 × (prior-weighted net value of the plan over both worlds) / (proven
information-matched optimum). One model call evaluates both worlds, so there is no
sampling noise from the hidden world. Instances get equal weight.

## Tiers

The ladder has two regimes. Small and mid models fall off in the first; frontier
models only separate in the second.

| Tier | Regime | Teams | Jobs | Horizon | Chains | Forecast |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | windows | 2 | 8 | 10 | depth 1 | certain |
| 2 | windows | 3 | 13 | 12 | depth 1 | uncertain, 1 survey |
| 3 | windows | 3 | 18 | 16 | depth 2, 1 join | uncertain, 1 survey |
| 4 | windows | 4 | 25 | 18 | depth 2, 1 join | uncertain, 2 surveys |
| 5 | windows | 5 | 31 | 20 | depth 3, 2 joins | uncertain, 2 surveys |
| 6 | packing | 4 | ~48 | 14 | depth 3 | uncertain, 2 surveys |
| 7 | packing | 5 | ~70 | 18 | depth 3, 1 join | uncertain, 2 surveys |
| 8 | packing | 6 | ~92 | 20 | depth 3, 1 join | uncertain, 2 surveys |

- **Windows (1–5):** assets have release windows spread over the incident with little
  slack; values are asset-like (some critical assets behind access chains). Teams are
  busy about 60–80% of the time in the optimal plan. Points are lost by misreading
  chains, water, deadlines and the forecast.
- **Packing (6–8):** free-standing work is 1.8× team capacity, windows are wide, and value
  is roughly proportional to work (±15%). Teams are busy 97–98% of the time in the
  optimum. Any sensible plan captures ~88% (greedy); the last points need exact
  assignment of jobs to the team that does them fastest, interval packing across
  heterogeneous teams, water allocation and survey timing. This is a generalized
  assignment problem with time windows (strongly NP-hard), still solved exactly in at
  most ~90 s by the MIP.

`generate(tier, seed)` is deterministic. Development seeds are `0–999`; evaluation
suites start at `100000`. Generation never inspects model or baseline outcomes.

### How the ladder was calibrated

All calibration used development seeds and is kept in `results/ladder/`.

1. **Generator v1** (front-loaded releases, tiers 3–6): Astra, Sol, Fable and Opus all
   scored 99.6–99.7, Haiku 96.9, Luna 78.0. Clean frontier-vs-small separation but no
   headroom: in the optimum teams were idle ~45% of the time, so problems decomposed
   into independent local choices.
2. **Generator v2** (spread windows, tiers 5–8 up to 54 jobs): Fable 100.0, Astra 99.9,
   Sol 98.8, Opus 98.7, Haiku 96.1. A first within-frontier gradient, but the top two
   models were pinned at 100; size alone did not create difficulty.
3. **Packing probe** (capacity-bound, 43–62 jobs): Opus fell to 93 on one instance and
   Fable below 100 at 60 jobs, using ~41k output tokens. That became tiers 6–8 of
   generator v3, the version used for evaluation.

## Evaluation v1

Frozen suite [`configs/ladder/evaluation.yaml`](../configs/ladder/evaluation.yaml): 34 instances
on evaluation seeds (tiers 2–5: 18 instances, tiers 6–8: 16), 8 models through the
logged-in CLIs at high effort, one fresh tool-less call per model per instance (272 calls,
zero infrastructure failures, zero unparsed plans). Evidence:
[`results/ladder/evaluation-v1/`](../results/ladder/evaluation-v1/) (every response in
`calls.jsonl`, every instance in `instances.json`, SHA-256 checked against the run).

| Policy | Overall | Windows (T2–5) | Packing (T6–8) | Skipped entries / plan | Mean latency | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Fable 5.2 | 99.6 | 99.8 | 99.4 | 0.00 | 307 s | 31k |
| GPT-6.1 Sol | 99.4 | 99.9 | 98.8 | 0.03 | 588 s | 15k |
| GPT-6 Astra | 99.2 | 99.9 | 98.5 | 0.09 | 440 s | 10k |
| Claude Opus 5.5 | 98.8 | 99.7 | 97.7 | 0.00 | 158 s | 17k |
| Claude Sonnet 5.5 | 95.3 | 98.4 | 91.8 | 0.00 | 90 s | 13k |
| GPT-6.1 Terra | 93.9 | 97.2 | 90.3 | 0.24 | 48 s | 2k |
| Claude Haiku 5.5 | 92.2 | 97.5 | 86.1 | 0.06 | 183 s | 44k |
| GPT-6 Luna | 79.5 | 85.8 | 72.4 | 1.00 | 109 s | 6k |
| search baseline | 89.0 | 87.6 | 90.7 | – | – | – |
| greedy baseline | 72.9 | 58.3 | 89.3 | – | – | – |

**Separation.** Paired sign-flip tests over instances, Holm-corrected over all 28 model
pairs, plus stratified bootstrap intervals. A pair counts as separated only when both
agree. 21 of 28 pairs separate:

- All 16 frontier-versus-rest pairs (Fable, Sol, Astra, Opus vs Sonnet, Terra, Haiku, Luna):
  Holm p = 0.0014, the frontier model better on 28–32 of 34 instances and worse on ≤ 1.
- Sonnet > Haiku (3.2 points); every model > Luna (12.7–20.2 points).
- Fable > Opus (0.8 points; 17 better, 2 worse).
- Not separated: Fable / Sol / Astra (0.2–0.4 points), Sol and Astra vs Opus, and Terra vs Sonnet/Haiku.

**Reading it.** The windows tiers order small vs frontier (Luna and Terra lose points by tier 3,
Sonnet and Haiku by tier 5). The packing tiers carry most of the separation: only the four
frontier models beat a 256-sample randomized search there; Sonnet and Terra roughly tie it,
and Haiku and Luna fall below plain greedy. Within the frontier, the lead is small and
concentrated at tiers 7–8.

## Exact reference and verification

`aeolus.ladder.solver.solve` is a time-indexed MIP over both forecast worlds
(scipy/HiGHS, `mip_rel_gap = 0`). Information is exact: with `R[t]` = 1 once a survey has
completed by tick `t`, the constraint `|x_alpha - x_beta| <= R[t]` for `t <` bulletin
forbids branching before the reveal, and survey timing is itself a MIP decision. One
solve is therefore the optimum over all admissible policies, including the value of
information. The largest tier solves in about a second.

Checks (all in `tests/ladder/test_ladder.py`):

- The MIP equals v2's independent Bellman planner exactly, both on v2 instances of every
  family (levels 1–3) and on ladder tiers 1–2 with uncertain forecasts and surveys.
- The MIP's own optimal plan, replayed through the plan executor, reproduces the MIP value
  with zero skipped entries (asserted inside every solve).
- 300 random plans per tested tier never beat the optimum; baseline plans are admissible.
- open-loop ≤ information-matched ≤ clairvoyant on every tier.
- The Gymnasium environment, driven by the optimal plan, returns exactly the plan's value
  in each world, and its rewards before the reveal are identical across worlds.

## References and baselines reported with every run

| Policy | Meaning |
| --- | --- |
| `reference:clairvoyant` | Knows the world from tick 0 (above 100; information upper bound) |
| `reference:open_loop` | Best plan that never adapts to the forecast |
| `baseline:search` | 256-sample randomized priority-rule search (GRASP-style) over public data |
| `baseline:greedy` / `deadline` | Per-tick value/duration or value/time-to-deadline dispatch |
| `baseline:hold` | Empty plan (0) |

## RL environment

`aeolus.ladder.env.LadderEnv` is a Gymnasium environment over the same dynamics (one
step = one tick). The action is `MultiDiscrete([max_jobs + 1] * max_resources)` with an
`action_mask` in the observation. The reward is credited at the prior-expected value
until the reveal and corrected at the reveal, so the return equals the realized net value
and never leaks the hidden world. `solve(env.instance)` gives the exact normalizer.

```python
from aeolus.ladder.env import LadderEnv
env = LadderEnv(tiers=(1, 2, 3))
obs, info = env.reset(seed=0)
```

**Learnability check** (`tools/train_ladder_ppo.py`, [result](../results/ladder/rl-ppo-tiers123.json)):
a small masked PPO policy (shared job/team encoders, teams sampled in sequence so they
never pick the same job) trained for 150 updates × 48 episodes on tiers 1–3,
about 3 minutes of one CPU core. On 90 held-out instances (seeds disjoint from training)
it reaches **76.5%** of the exact optimum (both worlds), against 68.6% for per-tick
greedy and 44.0% for random legal play. The environment is learnable, and the gap to 100
is measured exactly rather than estimated.

## Running

```bash
pip install -e '.[dev]'           # scipy is required for the reference solver
aeolus-ladder show --tier 4 --seed 3 --system         # exact model input
aeolus-ladder run --suite configs/ladder/evaluation.yaml --out runs/ladder/refs   # references + baselines only
aeolus-ladder run --suite configs/ladder/evaluation.yaml \
  --models configs/ladder/models.yaml --out runs/ladder/eval --workers 8
```

Runs are resumable: rerunning the same command skips completed calls and retries
infrastructure failures (timeouts, usage limits) up to `--max-attempts`. Outputs are
`references.json`, `calls.jsonl` (every response, usage, latency, skipped entry and score),
`scores.csv`, `summary.json`, and `report.txt`.

Model providers: the logged-in `claude` and `codex` CLIs (one fresh, tool-less session
per instance, same prompt, schema and effort request for every model) or a `command`
provider for any client that reads `{system, observation, schema}` JSON on stdin and
prints a plan. Codex's JSON event stream does not report the serving model, so GPT
identities are the requested model IDs; Claude identities are checked against the CLI's
reported model.

Inference runs through the local CLIs and the reference solver needs seconds per
instance, so the ladder runs on the logged-in machine. It does not need the EC2 recipe
used by the canonical simulator and the v1/v2 lanes.

## Limits

- Synthetic scheduling abstraction of incident operations, not a calibrated fire model;
  values are abstract units.
- CLI scaffolding differs between providers and output tokens are not capped; the
  comparison is "model + its CLI at the requested effort", with latency and token use
  reported alongside scores.
- One call per instance per model. Separation is established across instances (paired
  sign-flip tests, Holm-corrected, plus stratified bootstrap intervals), not by repeating
  calls on the same instance.
- Headroom at the very top is thin. Fable, Sol and Astra score 98.7–99.4 even on
  tier 8 and are not separated from each other. Ordering them needs larger packing
  tiers (the MIP handles ~76 jobs and 10 teams in minutes) or more instances; the
  generator and references already support both.
- GPT model identities come from the requested model ID, because the Codex CLI's JSON
  stream does not report the serving model. Claude identities are checked against the
  CLI's reported model. "Frontier", "mid" and "small" are descriptive labels in
  `configs/ladder/models.yaml` and are never used in scoring.
