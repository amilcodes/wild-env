# wild-env

wild-env is a wildfire-operations research environment with two layers:

1. **The ops ladder** (`wildfire-ops-v3`): a planning benchmark and Gymnasium RL
   environment where every instance has a **proven optimum**. It is the primary
   way to evaluate language models and agents here.
2. **The simulator foundation**: a higher-fidelity fire-spread and suppression
   simulator, a GPU tensor surrogate for MARL, historical incident import and a
   replay viewer (described below).

## The ops ladder

A model acts as planning section chief for a synthetic incident. It gets one public
packet (teams, jobs with per-team durations, windows, prerequisites, water, start
costs and forecast-dependent values) and returns one contingent incident action plan:
shared actions plus branches for each forecast world, which may only start once the
forecast is known. The plan is executed exactly in both worlds and scored as a
percentage of the **information-matched optimum**, solved by a two-world MIP and
checked against an independent Bellman planner.

Difficulty climbs one ladder. Tiers 1–5 grow the problem (2→5 teams, 8→31 jobs, deeper
access chains, scarce water, survey-or-wait decisions). Tiers 6–8 make it
capacity-bound (48→92 jobs, 97% team utilization in the optimum): a generalized
assignment problem where only exact packing earns the last points.

### Results (evaluation v1, 34 held-out instances, 8 models)

![Ladder scores overall and by tier for eight models](docs/assets/ladder/ladder-scores.png)

| Policy | Overall | 95% CI | Windows (T2–5) | Packing (T6–8) | Solved |
| --- | ---: | --- | ---: | ---: | ---: |
| Claude Fable 5.2 | **99.6** | 99.4–99.8 | 99.8 | 99.4 | 18/34 |
| GPT-6.1 Sol | **99.4** | 99.2–99.6 | 99.9 | 98.8 | 18/34 |
| GPT-6 Astra | **99.2** | 98.9–99.5 | 99.9 | 98.5 | 17/34 |
| Claude Opus 5.5 | **98.8** | 98.3–99.2 | 99.7 | 97.7 | 12/34 |
| Claude Sonnet 5.5 | 95.3 | 94.4–96.2 | 98.4 | 91.8 | 5/34 |
| GPT-6.1 Terra | 93.9 | 93.0–94.8 | 97.2 | 90.3 | 4/34 |
| Claude Haiku 5.5 | 92.2 | 90.5–93.7 | 97.5 | 86.1 | 6/34 |
| GPT-6 Luna | 79.5 | 76.2–82.7 | 85.8 | 72.4 | 2/34 |
| *search baseline (256 samples)* | 89.0 | 87.0–90.9 | 87.6 | 90.7 | 2/34 |
| *greedy baseline* | 72.9 | 69.2–76.5 | 58.3 | 89.3 | 1/34 |

Score is the percent of the proven optimum, one call per model per instance, high
effort, no tools. **All 16 frontier-versus-rest pairs separate** (paired sign-flip test,
Holm over 28 pairs, p = 0.0014; each frontier model is better on 28–32 of 34 instances
and worse on at most one). 21 of 28 pairs separate overall: Sonnet > Haiku, everyone
> Luna, and Fable > Opus (17 better, 2 worse). Fable, Sol and Astra are not separated
from each other (0.2–0.4 points).

The packing tiers carry the separation. There, only the four frontier models beat a
256-sample randomized search (90.7); Sonnet and Terra roughly tie it, and Haiku and
Luna fall below plain greedy. Every plan from every model was parsed; skipped entries
are rare (Luna about 1 per plan, others ≤ 0.24), so losses are planning quality, not
format. Full report: [`results/ladder/evaluation-v1/report.txt`](results/ladder/evaluation-v1/report.txt).


```bash
pip install -e '.[dev]'
aeolus-ladder show --tier 6 --seed 100000 --system          # the exact model input
aeolus-ladder run --suite configs/ladder/evaluation.yaml --out runs/ladder/refs   # references + baselines
aeolus-ladder run --suite configs/ladder/evaluation.yaml \
  --models configs/ladder/models.yaml --out runs/ladder/eval --workers 8      # models (resumable)
```

```python
from aeolus.ladder.env import LadderEnv      # Gymnasium, action masks, exact normalizer
from aeolus.ladder import generate, solve
env = LadderEnv(tiers=(1, 2, 3)); obs, info = env.reset(seed=0)
optimum = solve(env.instance).value
```

Protocol, verification, calibration history and limits:
[`docs/ladder_benchmark.md`](docs/ladder_benchmark.md). Raw evidence:
[`results/ladder/`](results/ladder/).

### Earlier agent lanes (archived)

These remain runnable and their evidence is unchanged. They are superseded as
model benchmarks by the ladder:

- **Desk v1** ([protocol](docs/frontier_agent_benchmark.md),
  [results](docs/frontier_live_results.md)): per-tick dispatch on the canonical
  simulator. No model clearly beat greedy, and per-episode noise was high.
  Runs on EC2 ([recipe](docs/ec2_agent_benchmarks.md), [CLI bridges](docs/cli_agent_evaluation.md)).
- **Operations v2** ([protocol](docs/operations_benchmark.md),
  [results](results/agent_benchmark/operations_v2_results.txt)): exact per-decision
  regret on small instances (2 teams, ≤11 jobs). Luna separated from the rest; Sol and
  Astra reached zero regret everywhere, so frontier models could not be ordered. The
  ladder keeps v2's dynamics (and its Bellman planner as a cross-check) and scales them.

## Simulator foundation

wild-env is a research environment for wildfire spread, suppression operations,
and multi-agent resource allocation. It combines a higher-fidelity incident
simulator with a fixed-shape PyTorch surrogate for large-batch reinforcement
learning.

The system includes heterogeneous aircraft and ground resources, airports and
water sites, payload and endurance constraints, delayed fire observations,
water and retardant delivery, fireline construction, historical incident
import, paired evaluation, and deterministic replay. The installed package is
`aeolus-ia`; the Python namespace remains `aeolus`.

![Native operational replay showing the fire, suppression lines, service sites, aircraft tracks, and synchronized vehicle and event state](docs/assets/viewer/native-operational.png)

*A recorded canonical-simulator episode in the native Qt viewer. The map,
vehicle table, event log, layers, camera, and timeline read the same immutable
replay bundle.*

![Paired tensor-environment rollout: maps of one world after 120 minutes with no dispatch and with the greedy baseline, mean expected asset loss and burned area over time, and expected loss in each of 16 worlds](docs/assets/rl/tensor-incident-rollout.png)

*A reproducible 16-world CPU rollout from the tensor training environment. It
compares no dispatch, where every aircraft holds, with the greedy baseline,
which acts only on the delayed fire belief. Map shading is fuel load, and blue
lines are sorties between service sites and the fire. It is a mechanism check,
not a learned-policy result.*

## Two simulation paths

| Path | Used for | Main tradeoff |
| --- | --- | --- |
| Canonical simulator | Fire-behavior studies, incident reconstruction, suppression evaluation, replay | More physical and operational detail; lower throughput |
| Tensor incident environment | Thousands of parallel MARL worlds, curriculum generation, policy iteration | Fixed-shape approximation that must be checked against the canonical simulator |

The canonical path uses spatial fuels and weather, wind/slope-coupled surface
fire, crown transition, spotting, dynamic moisture, and a WENO5/RK3 signed
level-set front. Suppression includes coverage-preserving drops, line
production and breach, service queues, finite stocks, dispatch gates, and
two-perimeter arrival-history initialization.

The tensor path keeps the decision state needed by the agents: hidden fire
truth, delayed belief, front tasks, treatment fields, vehicle motion, payload,
endurance, service contention, action masks, and constraint costs. Fire and
sortie transitions remain on the accelerator, and the policy uses recurrent
entity attention with centralized training and decentralized execution.

## Quick start

Python 3.10–3.12 is supported. Install the PyTorch wheel appropriate for the
target CUDA driver before installing the project on a cluster.

```bash
git clone https://github.com/amilcodes/wild-env.git
cd wild-env
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[geo,viewer,dev]'
ruff check src tests tools
pytest
```

### Run and inspect a local incident

Record a deterministic reference episode, then open it in the desktop viewer:

```bash
aeolus-replay \
  --config configs/replay_reference.yaml \
  --policy nearest \
  --horizon-min 12 \
  --out runs/replays/reference

aeolus-view \
  --replay runs/replays/reference \
  --config configs/viewer/operational.yaml
```

Remove `--horizon-min 12` for the complete scenario horizon. The canonical
solver is intentionally much slower than the tensor environment; the full
reference case is an evaluation run rather than a laptop smoke test.

The viewer has synchronized 2D and terrain-3D views, time controls, camera
presets, selectable vehicles and events, layer controls, local GeoTIFF imagery,
still/video export, and ParaView export.

<details>
<summary>Terrain replay example</summary>

![Native terrain replay of the fire front and resource histories](docs/assets/viewer/native-terrain.png)

</details>

### Exercise the RL environment on a CPU

This small run validates mechanics and produces paired baseline measurements.
It is not a substitute for accelerator profiling or full training.

```bash
python tools/run_tensor_incident_study.py \
  --config configs/cluster_tensor_incident.yaml \
  --device cpu \
  --batch-size 32 \
  --steps 40 \
  --model-steps 8 \
  --out runs/tensor-study.json

python tools/render_tensor_incident_rollout.py
```

For a short end-to-end MAPPO check:

```bash
aeolus-train --config configs/smoke.yaml
aeolus-eval \
  --config configs/smoke.yaml \
  --checkpoint runs/smoke/checkpoint.pt \
  --episodes 32 \
  --out runs/evaluation.json
```

### Train at cluster scale

`configs/cluster_tensor_incident.yaml` is the main fire-coupled training
manifest. It targets 2,048 parallel worlds, 128-step rollouts, mixed precision,
compiled transitions and policy networks, recurrent MAPPO, and distributed
launch through Slurm.

```bash
AEOLUS_CONFIG=configs/cluster_tensor_incident.yaml \
  sbatch deploy/slurm/train.sbatch
```

Run the exact throughput harness on the target accelerator before sizing a
training campaign:

```bash
python tools/run_tensor_incident_study.py \
  --config configs/cluster_tensor_incident.yaml \
  --device cuda \
  --compile \
  --batch-size 2048 \
  --grid-size 64 \
  --segments 48 \
  --out results/tensor_incident/gpu-study.json
```

See [`docs/cluster.md`](docs/cluster.md) for container, DDP and Slurm details,
and [`docs/rl_training_execution_plan.md`](docs/rl_training_execution_plan.md)
for the training and transfer gates.

## Public incident data

The incident importer can assemble a timestamped bundle from NASA FEDS
perimeters, USGS 3DEP elevation, and LANDFIRE fuel and canopy products:

```bash
aeolus-incident import-feds \
  --region CONUS \
  --fire-id 61854 \
  --size 128 \
  --buffer-m 5000 \
  --split evaluation \
  --out runs/incidents/feds-61854

aeolus-incident validate runs/incidents/feds-61854
```

The resulting `IncidentBundle` retains source responses, timestamps, spatial
reference, transformations, checksums, and STAC metadata. GOFER hourly
satellite progression and locally normalized perimeter/weather products have
separate import paths described in
[`docs/data_contract.md`](docs/data_contract.md).

Historical runs distinguish hindcast, shadow replay, and counterfactual use:

```bash
aeolus-historical \
  --incident runs/incidents/feds-61854 \
  --mode hindcast \
  --policy no_aerial \
  --start-index 0 \
  --target-index 1 \
  --out runs/hindcast.json
```

The current six-fire held-out result remains close to persistence on cumulative
perimeter skill and weak on advancing-front localization. The larger
36-incident chronological partition is frozen at 22 training, seven
development, and seven test incidents to prevent calibration leakage. Current
results therefore support simulator and policy research; they do not establish
operational predictive skill. Exact protocols and retained artifacts are in
[`docs/historical_validation.md`](docs/historical_validation.md) and
[`docs/heldout_historical_skill_v6.md`](docs/heldout_historical_skill_v6.md).

## Release line

| Version | Main addition |
| --- | --- |
| 0.8 | Ops ladder (`wildfire-ops-v3`): tiered planning benchmark and Gymnasium env with exact two-world MIP optimum |
| 0.7 | Replay schema 2, native Qt/VTK inspection, deterministic 2D/3D/video export |
| 0.6 | Tensor operations and fire-coupled MARL environments, entity-attention MAPPO, service-site logistics |
| 0.5 | Volume-conserving suppression, line production and breach, two-perimeter coupled-state initialization |
| 0.4 | NIROPS held-out evaluation, uncertainty scoring, incident-level partitions and baselines |
| 0.3 | Timestamped public incident bundles, historical modes and scientific replay |
| 0.2 | Operational fire behavior, dynamic moisture, crown fire, spotting and level-set propagation |
| 0.1 | PettingZoo/RLlib environment, constrained task allocation and recurrent MAPPO baseline |

## Research boundaries

- The tensor environment is a calibrated decision surrogate, not a replacement
  for the canonical fire solver.
- Persistence remains a difficult historical spread baseline. Held-out skill,
  observation cadence, incident wind, spatial moisture, and unobserved
  suppression remain active accuracy constraints.
- Example aircraft and UAS capabilities are research assumptions until replaced
  by reviewed performance tables and site-specific operating data.
- Counterfactual suppression effects are simulator results, not estimates of
  real-world treatment effectiveness.

The full gap register and compute/non-compute closure plan are in
[`docs/limits_and_solutions.md`](docs/limits_and_solutions.md).

## Code and documentation map

| Area | Location |
| --- | --- |
| Ops ladder: generator, plan executor, exact MIP, RL env, model runner | `src/aeolus/ladder` |
| Fire, truth/belief and suppression mechanics | `src/aeolus/core` |
| Incident, fuel, weather and progression data | `src/aeolus/data` |
| PettingZoo, RLlib and tensor environments | `src/aeolus/envs` |
| MAPPO networks, rollout and distributed training | `src/aeolus/training` |
| Baselines, paired studies and historical evaluation | `src/aeolus/policies`, `src/aeolus/evaluation` |
| Replay recording, rendering and desktop inspection | `src/aeolus/replay`, `src/aeolus/viewer` |

Start with:

- [`docs/ladder_benchmark.md`](docs/ladder_benchmark.md) — the ops ladder benchmark and RL environment.
- [`docs/architecture.md`](docs/architecture.md) — state, information and execution boundaries.
- [`docs/fire_behavior.md`](docs/fire_behavior.md) — equations, numerical methods and validation limits.
- [`docs/tensor_incident_environment.md`](docs/tensor_incident_environment.md) — surrogate contract and measured throughput.
- [`docs/rl_compute_research.md`](docs/rl_compute_research.md) — policy and compute design.
- [`docs/viewer.md`](docs/viewer.md) — replay schema, controls and rendering.
- [`docs/scenario_configuration.md`](docs/scenario_configuration.md) — scenario and viewer YAML fields.

Repository provenance and artifact-retention policy are recorded in
[`docs/development_record.md`](docs/development_record.md) and
[`docs/repository_artifact_policy.md`](docs/repository_artifact_policy.md).

## License

Apache-2.0. Simulator and policy outputs are research artifacts and are not
certified for operational fire-management decisions.
