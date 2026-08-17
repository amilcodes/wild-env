# Fire-coupled tensor training environment

## Purpose

`TensorIncidentEnv` is the high-throughput curriculum environment for aerial
suppression assignment. It approximates the decision structure of the
canonical incident simulator while keeping a complete batch of worlds on one
accelerator. It is intended for policy pretraining, algorithm comparison, and
large randomized studies. The canonical simulator remains the scientific
reference for transfer evaluation and historical work.

The environment does not emit aircraft controls. Every resource selects one
sortie-level assignment at each decision epoch:

```text
0                       hold
1 ... K                 attack a belief-derived oriented front segment
K + 1 ... K + S         fly to a compatible service site
```

This action boundary matches the existing task-pointer policies and keeps
flight control, separation, contingency handling, and drop-gate control below
the learned policy.

## Design basis

The implementation follows four useful patterns from current simulation and
MARL systems:

- JaxMARL demonstrates the throughput value of JIT compilation, vectorized
  worlds, static shapes, and a device-resident environment/policy loop.
- WarpDrive keeps simulation and learning tensors on the same GPU and avoids a
  host copy at each environment step.
- Madrona and GPUDrive show the value of structure-of-arrays world state,
  fixed-capacity entities, graph-like execution, and measuring the complete
  policy-plus-environment path.
- JaxWildfire shows that a probabilistic cellular fire surrogate can support
  large batched experiments while retaining terrain, fuel, wind, and
  intervention effects. Its published RL demonstration is a much smaller
  single-tanker task; this environment retains multi-resource service and
  assignment dynamics.

PyTorch is used at the experiment boundary because the existing policy,
rollout, fire-table, AMP, and DDP stack already uses it. The transition is a
pure tensor function over a fixed-shape `NamedTuple` and passes
`torch.compile(fullgraph=True)` with the eager validation backend. Python loops
over the eight neighbor directions and the fixed resource list are unrolled at
capture time. There is no `.cpu()`, `.numpy()`, item extraction, or data-
dependent Python branch inside a step.

## State

Each batch member carries the following structure-of-arrays state:

```text
[B, H, W]  unburned, burning, burned probabilities
[B, H, W]  belief probabilities and belief uncertainty
[B, H, W]  fuel factor, asset value, slope, barriers
[B, H, W]  water/retardant coverage and subcell line strength
[B]        wind and latent fire/response parameters
[B, N]     resource status, event times, payload, endurance, counters
[B, N, 2] resource position, leg geometry, target geometry
[B, S]     service-site suppressant stock
[B, S, M] service-bay availability times
```

`B` is world count, `N` aircraft count, `S` service-site count, and `M` the
maximum number of service bays. Raster resolution is deliberately independent
of the canonical landscape grid. A 64 by 64 surrogate grid is the initial
cluster setting.

All tensor dimensions are fixed after construction. Invalid or unused task
slots are masked. Finished worlds are absorbing. The training collector runs
escaped worlds to the common horizon so policies are compared over equal
elapsed time and receives a one-time escape penalty; evaluation can instead
terminate immediately on escape.

## Fire transition

The surrogate uses a continuous probabilistic cellular transition. For cell
`i`, neighbor `j` contributes a directional hazard

```text
h_ji = 0.006 (180 m / delta_x) s q_j d_ji
       exp(c_w u cos(theta_ji - theta_w) + c_s grad(z)_ji)
```

where `q_j` is neighbor burning probability, `s` is a world-level spread
scale, `d_ji` applies the diagonal-distance correction, and `c_w` and `c_s`
are latent wind and slope coefficients. Ignition during substep `dt` is

```text
p_i = 1 - exp(-dt fuel_i treatment_i sum_j h_ji).
```

The base hazard is scaled inversely with surrogate cell size. A projection
study caught the earlier per-cell formulation because a resolution change also
changed physical front speed. The current coefficient was adjusted on six
synthetic canonical teacher cases at the intended 64-by-64 resolution.

Burning mass transfers to burned mass with a sampled residence time. Phase
probabilities are renormalized after every substep and barriers remain zero.
The update is continuous rather than Bernoulli-sampled. This reduces gradient-
irrelevant Monte Carlo noise during large policy studies; stochasticity enters
through world initialization and per-world latent parameters.

Episode initialization randomizes ignition location and radius, correlated
fuel, slope, assets, wind speed/direction, spread scale, wind response, slope
response, residence time, water response, retardant response, and observation
reliability. The ranges are explicit in
`TensorIncidentParameterRanges`. They are provisional training ranges, not
incident posterior claims. A calibration study must replace or narrow them
using canonical and historical traces.

The surrogate intentionally omits crown transition, stochastic spotting,
spatial moisture evolution, atmosphere coupling, and explicit arrival-time
history. Those processes remain in the canonical environment. They enter the
training program through parameter randomization first and learned residual or
teacher calibration later.

## Belief and task construction

Truth and actor belief are separate tensors. Between observations, belief is
propagated through the same coarse fire operator. At a configured cadence, a
blurred truth observation is assimilated with a sampled reliability weight.
Uncertainty drops at assimilation and grows between acquisitions.

Front tasks are regenerated at every transition from belief only:

1. intersect belief burning with a local unburned neighborhood;
2. combine front activity, fuel, local protected value, uncertainty, and
   treatment gap;
3. retain spatial local maxima;
4. select a fixed top `K` and mask empty entries;
5. compute a tangent heading from the belief gradient, with crosswind heading
   as a degenerate fallback.

Actor task tensors and masks do not read hidden fire phase. A dedicated test
replaces truth with a fully burned field while holding belief fixed and
requires every actor tensor to remain bitwise identical. The privileged critic
changes under the same intervention.

## Suppression

A completed assignment deposits an anisotropic Gaussian footprint centered at
the selected belief-front task. Along-track and cross-track scales follow the
resource drop-length and width declarations. The kernel is clipped by barriers
and normalized before application, so

```text
sum_i coverage_i * cell_area * GPC_conversion = delivered volume.
```

Water and retardant coverage are stored separately and decay at their declared
half-lives. A second bounded line-strength field preserves subcell information
when a narrow drop is represented on a coarse cell. Water reduces current
burning residence and spread for a short period. Retardant primarily reduces
future spread and persists longer. Response coefficients are randomized per
world and available to the privileged critic through a compact latent-severity
feature.

This is an intervention-response surrogate. It has not been calibrated against
observed drop polygons, coverage-level measurements, or causal spread changes.
Those are release gates, not assumed properties.

## Aircraft and service events

Flight and service are event-level. State records departure, arrival, service
start, completion, start/end position, target, reserved suppressant, payload,
and endurance. Position is interpolated along the active leg for observations
and replay export.

The hard action mask checks:

- current availability and payload threshold;
- resource and site service-mode compatibility;
- resource and site wind limits;
- site opening time, finite stock, bays, and approach capacity;
- travel, dispatch delay, hover/scoop airborne service, recovery, reserve, and
  remaining endurance;
- front and service assignment capacities.

Site stock is reserved when a service mission is accepted. Service slots are
scheduled in ascending resource order after the capacity-aware joint sampler
has chosen actions. That order is deterministic and remains a possible source
of priority bias for heterogeneous fleets.

## Reward and constraints

The shared reward is the change in normalized outcome state:

```text
-80 delta(value-weighted expected loss)
-8  delta(burned fraction)
+6  reduction in active value-at-risk
+scenario containment bonus
-scenario escape penalty on first escape
-small flight, queue, blocked-action, and wasted-volume terms
```

Expected loss counts burned asset value plus half the value under active fire.
The scale was falsified against hold, uniform-valid, and public-belief greedy
comparators. An earlier cost weighting ranked improved suppression below
holding; it was rejected. Operational terms are deliberately smaller than
asset loss and are also returned as a separate four-component cost vector:

```text
blocked assignments, endurance exhaustion, queue minutes, wasted fraction
```

The current MAPPO trainer records the shared reward. A constrained PPO branch
must add cost critics and dual updates before probabilistic constraints are
optimized rather than merely reported. Deterministic safety constraints remain
hard masks in every algorithm.

PPO minibatches preserve time in fixed recurrent sequences. Hidden state is
initialized from the on-policy rollout at each sequence boundary and reset
after episode termination, so gradients propagate through the configured
sequence length rather than treating every GRU state as an independent sample.
Each update also logs final expected loss, burned fraction, containment,
escape, delivered/wasted liquid, blocked actions, and every constraint-cost
component. Reward can therefore be audited against physical and operational
outcomes during training.

## Measured local result

`results/tensor_incident/cpu_mechanism_study.json` is the retained CPU check:

- 32 matched worlds, 12 aircraft, 6 sites;
- 64 by 64 fire grids, 32 front slots, 2 fire substeps;
- 40 decisions, or 120 incident minutes;
- no compiled transition on the local CPU.

The public-belief greedy comparator reduces final expected loss by 0.0040 and
burned fraction by 0.0020 relative to hold in the paired mean. It improves both
metrics in all 32 matched worlds and improves return in 28. Uniform-valid
contains more worlds under the current threshold but uses about 3.8 times as
much liquid and wastes about 4.7 times as much; this is a useful multiobjective
case for a learned allocator. Eager environment throughput is about 10,900
agent decisions/s for the greedy case and about 9,500 agent decisions/s for
the complete entity-attention policy-plus-environment loop on this machine.

`results/tensor_incident/canonical_teacher_check.json` projects six canonical
snapshots into 32-member surrogate ensembles and compares 36-minute
uncontrolled forecasts. Mean absolute cumulative-fire-fraction delta error is
0.000162 and every canonical delta falls inside the surrogate 5th--95th
percentile interval. The cases are synthetic, share one scenario family, and
have zero asset loss over the short window. They were used to adjust the base
spread scale, so the result is in-sample. This is a calibration smoke test, not
a general validation result.

These are mechanism and local throughput results. They do not establish
historical accuracy, learned-policy performance, GPU throughput, scaling, or
field effectiveness.

The complete transition also lowered and executed through the local CPU
Inductor backend on a two-world smoke case. CUDA lowering and CUDA Graph replay
remain untested until cluster access is available.

## Reproduction

```bash
python tools/run_tensor_incident_study.py \
  --config configs/cluster_tensor_incident.yaml \
  --batch-size 32 \
  --grid-size 64 \
  --segments 32 \
  --steps 40 \
  --out results/tensor_incident/cpu_mechanism_study.json

python tools/evaluate_tensor_incident_fidelity.py \
  --config configs/cluster_tensor_incident.yaml \
  --cases 6 \
  --warmup-steps 4 \
  --forecast-steps 12 \
  --ensemble-size 32 \
  --grid-size 64 \
  --out results/tensor_incident/canonical_teacher_check.json

AEOLUS_CONFIG=configs/cluster_tensor_incident.yaml \
  sbatch deploy/slurm/train.sbatch
```

On the cluster, rerun the study with `--device cuda --compile` before launching
training. Record compilation time, steady-state steps/s, agent decisions/s,
peak allocated/reserved memory, GPU utilization, kernel launch count, graph
break count, and policy/environment/optimizer time fractions.

The preflight memory estimate is generated with:

```bash
python tools/estimate_tensor_incident_memory.py
```

For the reference 2,048-world rank it accounts for 5.16 GiB of persistent
storage before compiler, transition, autograd, allocator, CUDA, and NCCL
overheads.

## Primary technical references

- Rutherford et al., [JaxMARL: Multi-Agent RL Environments and Algorithms in
  JAX](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5aee125f052c90e326dcf6f380df94f6-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024.
- Lan et al., [WarpDrive: Extremely Fast End-to-End Deep Multi-Agent
  Reinforcement Learning on a GPU](https://www.jmlr.org/papers/v23/22-0185.html),
  JMLR 2022.
- Shacklett et al., [Madrona: A Fast, Lightweight, Scalable Platform for
  Functional Simulation](https://madrona-engine.github.io/shacklett_siggraph23.pdf),
  SIGGRAPH 2023.
- Gulino et al., [GPUDrive: Data-driven, multi-agent driving simulation at one
  million FPS](https://proceedings.iclr.cc/paper_files/paper/2025/file/3107ddd4209e5f93c0371425763041a3-Paper-Conference.pdf),
  ICLR 2025.
- Goyal et al., [JaxWildfire](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_80.pdf),
  NeurIPS ML4PS 2025.
- PyTorch, [`torch.compile` programming model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html)
  and [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).
