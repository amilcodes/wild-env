# RL surrogate fidelity and large-scale training plan

Date: 2026-08-12  
Status: execution plan; current tensor environment implemented, broad calibration pending  
Reference simulator: `AeolusSimulator`  
Mass-training environment: `TensorIncidentEnv`

## 1. Research question

The problem is to build a low-cost environment that produces useful aerial
suppression policies without allowing those policies to exploit errors that
disappear in the canonical simulator. There are three separate questions:

1. **Transition fidelity:** from the same projected state and action, does the
   tensor environment predict the canonical fire, belief, resource, service,
   and treatment transition?
2. **Intervention fidelity:** does it predict the incremental effect of water,
   retardant, service routing, and coordinated line construction relative to a
   matched no-action counterfactual?
3. **Decision fidelity:** do policies and planners retain their ordering when
   moved from the tensor environment to the canonical simulator?

Final burned area alone is insufficient. A model can match area while placing
growth on the wrong flank, assigning value to physically infeasible sorties,
or learning an exaggerated treatment response. The release criterion is
therefore transition, intervention, and policy fidelity together.

This work calibrates a fast model to the canonical model. It does not make the
canonical model empirically correct. Historical skill remains a separate gate,
currently being evaluated by the frozen 36-incident benchmark.

## 2. Current implementation and evidence

### 2.1 Fidelity roles

The repository already has the correct basic separation.

| Tier | Implementation | Role | Current material limits |
|---|---|---|---|
| L0 | `TensorOperationsEnv` | Assignment, service, and queue curriculum without a fire | No fire response |
| L1 | `TensorIncidentEnv` | Device-resident, fixed-shape mass training | Coarse probabilistic cellular fire; provisional parameters and simplified aviation |
| L2 | `AeolusSimulator` | Canonical semantics, teacher generation, fine-tuning, evaluation | CPU/NumPy control flow; too expensive for billions of steps |
| L3 | Historical observations and selected coupled-model cases | External validity and model-discrepancy evaluation | Sparse operations records; high cost; observational uncertainty |

L1 currently carries a 64 by 64 probabilistic fire field, hidden truth and
delayed belief, dynamic front tasks, subcell water and retardant line strength,
12 aircraft, six service nodes, finite stock, queue/service events, payload,
endurance, and a capacity-aware task-pointer action. Its transition and policy
loop stay in PyTorch tensors and can be compiled on an accelerator.

L2 carries WENO5/RK3 front propagation, the fire-behavior lookup, dynamic fuel
moisture, crown transition, spotting, line breach, volume-conserving delivery,
time-varying weather, richer belief state, aircraft performance surfaces,
terrain and airspace checks, and complete replay semantics.

### 2.2 Existing fidelity evidence

The current six-case teacher check is a smoke test, not validation. It used one
synthetic scenario family, short uncontrolled forecasts, and the same cases
used to adjust the base spread coefficient.

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Six 36-minute cases: cumulative-fire delta MAE | 0.000162 | Bridge and scale operate in a narrow in-sample regime |
| Six 36-minute cases: nominal 90% coverage | 6/6 | Too few and too broad to assess calibration |
| Four 120-minute cases: cumulative-fire delta MAE | 0.003227 | Error grows materially with horizon |
| Asset-loss delta in both checks | identically zero | Existing check does not test the operational objective |
| Suppression transitions | absent | Treatment fidelity is untested |
| Logistics and action-mask parity | absent | A policy can still exploit feasibility disagreements |
| Policy ranking across fidelities | absent | Transfer quality is unknown |

The next study must be incident- and scenario-held-out, include active
suppression, and report spatial front errors rather than only scalar area.

### 2.3 Measured local cost

The following measurements were taken on the current Apple-arm64 workstation
with the uncompiled CPU path. They are mechanism and budgeting measurements,
not GPU claims.

| Workload | Measured result |
|---|---:|
| L1 environment plus entity-attention policy, 32 worlds | 727 world decisions/s |
| Same workload in aircraft decisions | 8,726 aircraft decisions/s |
| L1 environment with public-belief greedy controller | 843 world decisions/s |
| L2 canonical simulator, 192 by 192 grid, hold controller | 2.59 world decisions/s |
| L2 simulated incident time per wall second | 7.78 minutes/s |

On this scenario the complete CPU tensor loop is about 280 times the
single-world canonical decision rate. This ratio is workload-specific and
does not predict CUDA throughput.

At 2,048 worlds, 64 by 64 cells, 12 resources, 55 task slots, and a 128-step
rollout, the persistent-storage estimator reports:

| Allocation | Lower bound |
|---|---:|
| Environment and observation state | 0.49 GiB |
| Rollout storage | 4.57 GiB |
| Parameters, gradients, and Adam estimate | 0.09 GiB |
| Accounted total | 5.16 GiB |

This excludes compiled workspaces, graph pools, fire/front temporaries,
attention and recurrent activations, PPO temporaries, allocator fragmentation,
CUDA context, and NCCL buffers. It is a planning floor.

## 3. Target architecture for the cheap environment

The first target is a calibrated structured model, with a learned residual
added only if the structured model fails held-out gates.

### 3.1 Preserve the state and action contract

L1 and L2 must share the following semantics before fire calibration is
interpretable:

- identical task identifiers and capacity semantics;
- identical payload, stock, service, queue, endurance, and reserve accounting;
- a common decision clock and event ordering;
- a conservative action mask, where an L1 false-valid action is a safety
  failure;
- volume-conserving treatment projection;
- identical containment, escape, and episode-horizon definitions; and
- actor observations derived only from information available to the actor.

The tensor path may approximate travel and performance, but it must expose the
approximation and fail closed outside its fitted envelope. Vehicle-performance,
terrain, vector-wind, airspace, and service-geometry parity is a prerequisite
for a policy-transfer claim.

### 3.2 Structured fire core

Retain the local probabilistic hazard update because it is fast, stable,
vectorizable, and naturally enforces nonnegative phase mass. Replace the
current global coefficient ranges with a calibrated conditional rate model:

```text
log hazard = local spread baseline
           + fuel-family term
           + wind-speed/direction term
           + slope-normal term
           + dead/live-moisture term
           + front-curvature term
           + crown/spotting regime term
           + bounded treatment term.
```

The fitted form should initially be monotone splines or a small local MLP with
declared sign constraints. It predicts a hazard correction, not the next fire
mask directly. Phase mass remains normalized; burned mass cannot become
unburned; barriers remain closed; and cumulative burned area remains
monotone.

### 3.3 Subgrid front and spotting

The 64 by 64 state needs explicit subgrid information for narrow drops and
front motion:

- signed fractional front occupancy or distance within each cell;
- front normal and curvature;
- separate surface and crown activity probability;
- fixed-capacity sparse spotting sources and landing kernels; and
- first-arrival or recent-arrival age sufficient to reproduce residence and
  active-front dynamics.

Spotting must be a bounded tensor operation with a fixed maximum event count
per step. Its distance, direction, survival, and ignition distributions are
calibrated against canonical common-random-number traces. A single broadened
local spread coefficient is not an acceptable spotting surrogate because it
changes the value of line placement.

### 3.4 Weather and moisture

L1 should ingest the same compact forcing window as L2:

- 10-m vector wind on the coarse grid;
- temperature, humidity, and precipitation;
- dead 1-h/10-h/100-h moisture or a reduced moisture state;
- live herbaceous/woody moisture; and
- time since the last observation and forcing issue time.

The transition can use coarse fields and reduced moisture equations. Constant
episode wind and one latent fuel factor are inadequate for policies expected
to respond to wind shifts, recovery periods, or changes in treatment value.

### 3.5 Learned residual, if required

An unrestricted image-to-image surrogate is a poor first choice: it can break
mass conservation, invent spread across barriers, and give a policy exploitable
nonlocal effects. If the structured core fails after calibration, add a small
local residual network over the hazard or front-normal velocity.

Preferred order:

1. depthwise/local convolutions over a 5 by 5 to 11 by 11 physical stencil;
2. a dilated residual stencil if the required receptive field is larger;
3. a compact graph operator for irregular barriers and front segments; and
4. a global neural operator only if held-out evidence shows that local models
   cannot reproduce the required correction.

Use an ensemble of residual models or heteroscedastic heads. The environment
must return epistemic uncertainty and an out-of-distribution score. High
uncertainty narrows the allowed training claim and triggers canonical teacher
sampling; it is not hidden by wider domain randomization.

## 4. Canonical teacher corpus

### 4.1 Experimental design

Generate canonical traces from a factorial design rather than random seeds
from one scenario. Factors include:

- fuel family and fuel-load quantile;
- dead and live moisture quantiles;
- flat, rolling, and steep terrain;
- wind speed, direction relative to slope/front, and timed wind shifts;
- surface, crown-transition, and spotting regimes;
- ignition size, age, and perimeter complexity;
- observation cadence, localization error, obscuration, and delay;
- water versus retardant, coverage level, line orientation, and offset;
- fleet composition, payload state, endurance, site topology, stock, closures,
  queue load, and resource loss; and
- short, medium, and full-episode forecast horizons.

Controllers must cover states a learned policy will visit:

1. hold;
2. uniform-valid exploration;
3. current public-belief greedy;
4. exact or rolling-horizon assignment on small cases; and
5. checkpoints from L1 training as they become available.

The fifth source is essential. After initial calibration, use an iterative
teacher-query loop analogous to dataset aggregation: run the current policy in
L1, identify high-uncertainty or high-disagreement states, reconstruct those
states in L2, and append canonical transitions. This directly attacks policy
exploitation and covariate shift.

### 4.2 Initial corpus size and cost

The first corpus should contain 2,000 full six-hour canonical trajectories at
120 decisions each, or 240,000 canonical decisions. Allocate approximately:

- 25% hold and uncontrolled fire;
- 20% uniform-valid exploration;
- 25% public-belief greedy;
- 20% exact or rolling-horizon controls; and
- 10% adversarial/current-policy states once a checkpoint exists.

At the measured 2.59 decisions/s, 240,000 decisions cost about 25.7 CPU-hours
before reset, serialization, and scenario-construction overhead. Budget
40--80 CPU-hours. Sixteen to 32 independent workers should complete the
initial synthetic corpus in roughly 2--6 wall hours if memory and storage
bandwidth scale acceptably.

Store projected 64 by 64 state/action/next-state records in chunked arrays,
plus full canonical checkpoints at trajectory starts and selected event
boundaries. Fifty thousand projected records with approximately 20 float16
raster channels are about 8 GiB before compression; retaining every full
192 by 192 float32 state would be unnecessarily large.

### 4.3 Counterfactual intervention twins

At least 25% of teacher states should produce paired transitions using the
same initial state, weather, and random-number stream:

```text
canonical(state, suppression action) - canonical(state, hold)
```

Generate twins for water, retardant, alternative line headings, line offsets,
multi-aircraft capacity, refill routing, and site closure. These differences
identify treatment response more cleanly than absolute next-state fitting.
Historical perimeters with unobserved suppression cannot provide this causal
label.

### 4.4 Splits and leakage control

Split by complete scenario family and seed, never by transition row:

- 60% calibration/train;
- 20% model-development;
- 20% untouched synthetic test.

Hold out entire combinations of fuel family, wind regime, terrain, and fleet
topology to measure interpolation separately from extrapolation. Historical
incidents retain the frozen 22/7/7 chronological split and are never used to
tune a surrogate after their test outcomes are inspected.

## 5. Fidelity measurements

### 5.1 Projection invariants

Before dynamics fitting, verify canonical-to-tensor projection for:

- burnable area and phase mass;
- fuel mass and asset-value integral;
- water and retardant volume;
- barrier connectivity and narrow-corridor preservation;
- resource payload, endurance, position, and event clock;
- service stock and slot occupancy; and
- belief/fire separation.

Continuous conserved quantities should differ by less than 1%; suppressant
volume should agree to numerical tolerance. Topological barrier errors are
reported separately because a small area error can open a critical path.

### 5.2 One-step transition fidelity

For every paired state/action, report:

| Component | Metrics |
|---|---|
| Fire phase | Brier/CRPS, phase-mass error, cumulative-area error |
| Advancing front | symmetric boundary distance, normal displacement, heading error, active-front precision/recall |
| Spread response | directional rate-of-spread error by wind/slope/fuel/moisture stratum |
| Spotting | event-rate error, landing-distance distribution, ignition probability, downwind/crosswind error |
| Treatment | volume error, footprint overlap, coverage error, causal front-delay and loss-delta error |
| Belief | Brier score, calibration, observation-update error, uncertainty growth |
| Tasks | front coverage, selected-segment recall, priority-rank correlation, heading error |
| Operations | ETA, payload/endurance, queue/service event-time error, stock conservation |
| Mask | false-valid and false-invalid rates, stratified by reason |

False-valid action-mask disagreement has a zero-tolerance release target for
deterministic safety constraints. False-invalid disagreement should remain
below 5% overall and be reported by feasibility reason.

### 5.3 Open-loop error growth

From the same projected canonical state and fixed action sequence, evaluate
1, 4, 12, 40, and 120 decisions: 3, 12, 36, 120, and 360 incident minutes.
Report error growth, not only the final horizon. Initial proposed gates,
frozen before the untouched test is run, are:

- median one-step front-normal error below 0.5 tensor cell and 90th percentile
  below 1.5 cells;
- median one-step area-growth relative error below 10%;
- 120-minute cumulative-area normalized error below 15%;
- 120-minute mean boundary error below two tensor cells;
- intervention-effect sign agreement above 95% and rank correlation above
  0.8;
- nominal 90% predictive intervals with 85--95% empirical coverage overall
  and no major regime below 75%; and
- no systematic violation of phase, volume, or stock conservation.

These thresholds are engineering starting points. They may be changed using
training/development evidence, but must then be re-frozen before test.

### 5.4 Closed-loop decision fidelity

Run identical controllers independently in L1 and L2 on paired cases:

- hold;
- public-belief greedy;
- exact assignment on small cases;
- rolling-horizon planner;
- L1-pretrained policy; and
- canonically fine-tuned policy.

Compare policy ordering and regret on expected loss, burned area, escape,
containment time, liquid use/waste, utilization, queue delay, and constraint
violations. Required gates:

- Kendall policy-rank correlation at least 0.8;
- at least 90% agreement on the sign of paired policy improvements;
- canonical regret of the selected L1 policy within 5% of the best evaluated
  transferable policy on the development set;
- no increase larger than two percentage points in catastrophic escape rate;
  and
- stable ranking across major wind, fuel, observation, and fleet strata.

A transition model with strong pixel metrics but failed policy ranking is not
an RL training environment.

### 5.5 Adversarial surrogate-exploitation test

Search for states and action sequences maximizing the discrepancy

```text
L1 predicted policy advantage - L2 realized policy advantage.
```

Use policy-gradient search in L1, randomized action-sequence search, and
uncertainty-guided scenario selection. Replay candidates in L2. Add discovered
failures to the teacher corpus only through a declared new training revision;
keep an untouched adversarial test bank.

## 6. Calibration and model-selection sequence

1. **Parity first.** Close clock, event, conservation, action-mask, and task
   contract disagreements without fitting fire behavior.
2. **Fit the structured uncontrolled fire.** Use hold traces and minimize a
   weighted phase/front/rate loss. Fit train only, select on development.
3. **Fit belief and task generation.** Match observation updates, uncertainty,
   advancing-front coverage, task heading, and priority ordering.
4. **Fit suppression counterfactuals.** Use paired action/hold twins and keep
   water, retardant, and line-construction response separately identifiable.
5. **Measure multi-step drift.** If the structured model passes, retain it.
6. **Add the smallest residual model that closes a documented failure.** Keep
   physical constraints outside the learned residual.
7. **Calibrate uncertainty and validity envelope.** Use ensemble disagreement,
   conformal residual bounds on development scenarios, and an OOD classifier.
8. **Run untouched synthetic test once.** Freeze model and test artifacts.
9. **Train policies.** Pretrain by planner imitation, then MAPPO and constrained
   MAPPO branches under calibrated domain randomization.
10. **Transfer and fine-tune in L2.** Preserve separate identities for L1
    pretrain, L2 fine-tune, and final selected checkpoint.
11. **Evaluate on held-out incidents.** Use historical cases for shadow-mode
    decisions and within-model counterfactual comparisons. Do not claim field
    suppression effectiveness without intervention records.

Domain randomization should be drawn from fitted posterior or residual ranges,
with deliberate stress tails. Broad arbitrary ranges can hide a biased model
and teach excessively conservative or incoherent policies.

## 7. Large-scale training design

### 7.1 Initial algorithms

Run a controlled sequence:

1. public-belief greedy and exact/rolling-horizon planner baselines;
2. behavior cloning into the entity-attention task pointer;
3. IPPO with shared parameters;
4. recurrent entity-attention MAPPO;
5. constrained MAPPO with cost critics and dual variables; and
6. HAPPO or an autoregressive assignment decoder only if heterogeneous-policy
   interference or assignment conflicts are measured.

Do not expand the algorithm set before one implementation is stable under the
same teacher, throughput, and transfer protocol.

### 7.2 Multi-fidelity training

The default sequence is large L1 pretraining followed by small L2 fine-tuning.
Also evaluate a multi-fidelity policy-gradient/control-variate branch that
combines abundant biased L1 rollouts with scarce L2 rollouts. This provides a
principled alternative when pure pretrain/fine-tune is unstable.

Maintain three checkpoint identities:

- `surrogate_pretrain`;
- `canonical_finetune`; and
- `selected_release_candidate`.

Policy selection uses L2 development cases. L1 return is a training diagnostic,
not the selection objective.

## 8. Compute plan

### 8.1 Declared training volume

The current cluster configuration contains 2,048 worlds per rank, 128 rollout
decisions, and 3,000 updates. On one rank this is:

- 786,432,000 world transitions;
- 9,437,184,000 aircraft decisions; and
- 120 decisions, or six simulated hours, per episode.

With DDP, `num_envs` is per rank. Four ranks at the existing setting generate
four times the samples per update; this is not a fixed-sample speed comparison.
Scaling studies must report both samples and wall time and should first hold
global world count fixed.

### 8.2 GPU memory starting points

| Accelerator memory | Initial worlds/rank | Reasonable profiling range |
|---:|---:|---:|
| 24 GiB | 512 | 256--1,024 |
| 40--48 GiB | 1,024 | 512--2,048 |
| 80 GiB | 2,048 | 1,024--4,096 |

These are launch points, not capacity claims. Sweep batch sizes until steady
end-to-end throughput saturates or peak reserved memory exceeds 85%. Measure
after compilation and at least three complete PPO updates.

### 8.3 Throughput and runtime envelope

No target-GPU measurement exists yet. For planning, use 10,000--100,000 world
transitions/s per high-end GPU for the complete environment/policy collection
path. This conservative bracket is derived from the measured CPU path and the
substantially heavier raster/task transition relative to published toy MARL
benchmarks.

For 786 million transitions, collection alone would take:

| Measured future rate | Collection wall time |
|---:|---:|
| 10,000 transitions/s | 21.8 h |
| 25,000 transitions/s | 8.7 h |
| 50,000 transitions/s | 4.4 h |
| 100,000 transitions/s | 2.2 h |

PPO backpropagation, recurrent sequences, attention, checkpointing, and
evaluation are additional. Budget 12--72 GPU-hours per seed on one A100/H100-
class GPU until profiling narrows the range. Five primary seeds therefore cost
approximately 60--360 GPU-hours. A full eight-configuration, five-seed
algorithm/ablation matrix costs roughly 480--2,880 GPU-hours; it should be
stage-gated rather than launched as one sweep.

Four GPUs are useful after one rank saturates. For a fixed global sample
budget, require at least 70% parallel efficiency at two and four GPUs before
moving to multi-node execution. For the existing per-rank batch configuration,
four GPUs increase sample volume rather than divide the same run.

### 8.4 Required accelerator profile

For eager and compiled modes, sweep 256, 512, 1,024, 2,048, and 4,096 worlds
where memory permits. Record:

- environment-only transitions/s;
- policy plus environment transitions/s;
- full collect plus PPO-update transitions/s;
- aircraft decisions/s;
- compile time and graph breaks;
- peak allocated and reserved memory;
- achieved GPU utilization and kernel occupancy;
- front extraction, fire transition, policy inference, and PPO time fractions;
- reset and checkpoint overhead; and
- DDP/NCCL fraction at 1, 2, 4, and 8 GPUs.

Only measured profiles determine the production batch size. Triton or C++/CUDA
is justified only for a measured kernel bottleneck that PyTorch compilation
does not resolve.

## 9. Execution packages and gates

| Package | Work | Exit evidence |
|---|---|---|
| A. Contract | State/action/event parity specification and projection invariants | Randomized canonical/tensor property tests; no false-valid deterministic actions |
| B. Teacher | Designed synthetic corpus and paired suppression twins | Frozen manifest, scenario-level splits, checksums, throughput/storage report |
| C. Structured fit | Conditional hazard, moisture, belief, and response calibration | Development improvement at all declared horizons; conservation retained |
| D. Residual decision | Ablate local residual against structured core | Add only if held-out front/intervention error materially improves |
| E. Fidelity test | Untouched transition, open-loop, policy-rank, and adversarial tests | All release gates or an explicit narrowed validity envelope |
| F. GPU profile | Full environment-policy-PPO sweep | Stable compiled run, no host synchronization, selected batch/rank |
| G. Policy study | Planner imitation, IPPO, MAPPO, constrained branch | Five-seed paired results and learning/constraint diagnostics |
| H. Transfer | Zero-shot and L2 fine-tuning | Stable L2 policy ranking, bounded tail loss and escape |
| I. Historical shadow | Frozen incident evaluation | Incident-cluster intervals; no target leakage; no field-effect overclaim |

## 10. Immediate work order

While the frozen 36-incident benchmark uses execution source revision 2, do
not change its hashed simulator source. The measurements and this document are
outside that execution closure. After the frozen run completes or is moved to
an explicitly superseding source revision:

1. implement the canonical/tensor state-pair record schema;
2. extend the fidelity harness from scalar no-action outcomes to spatial
   state, action-mask, logistics, and counterfactual suppression metrics;
3. add randomized projection and one-step parity tests;
4. generate the first designed teacher pilot: 100 trajectories across the
   full factor matrix;
5. freeze thresholds and the 2,000-trajectory corpus contract after inspecting
   pilot mechanics, not test outcomes;
6. generate and fit the structured surrogate;
7. run the untouched synthetic test and decide whether a residual network is
   warranted;
8. profile on the first available CUDA node; and
9. begin stage-gated policy training only after fidelity and feasibility gates
   pass.

## 11. Stop conditions

Stop or narrow the use of L1 if any of the following persists after one
declared correction revision:

- policies exploit false-valid actions or conservation errors;
- suppression-effect sign agreement remains below 95%;
- uncertainty does not cover held-out canonical outcomes by regime;
- policy rank correlation remains below 0.8;
- catastrophic escape degradation exceeds two percentage points;
- the residual model improves pixel loss but worsens policy transfer; or
- complete GPU training throughput is too low to provide a material advantage
  over canonical parallelism.

In that event, L1 remains an operations curriculum or proposal generator, and
fire-coupled policy optimization moves to smaller L2 batches or a different
reduced-order model.

## 12. Research basis

- Lan et al., *WarpDrive: Extremely Fast End-to-End Deep Multi-Agent
  Reinforcement Learning on a GPU* (2021): device-resident multi-agent
  simulation and policy execution.
- Rutherford et al., *JaxMARL* (NeurIPS 2024): static, vectorized accelerator
  environments and end-to-end MARL benchmarking.
- Kazemkhani et al., *GPUDrive* (ICLR 2025): fixed-capacity GPU simulation and
  policy learning over many data-derived scenes.
- Çakır et al., *JaxWildfire* (2025): vectorized probabilistic cellular fire
  simulation and differentiable parameter optimization for RL.
- Bolt et al., *An Emulation Framework for Fire Front Spread* (2022): fire-
  front emulation as a fast approximation to a more expensive simulator.
- Liu et al., *A Multi-Fidelity Control Variate Approach for Policy Gradient
  Estimation* (TMLR 2026): combining abundant low-fidelity and scarce high-
  fidelity rollouts without treating the low-fidelity model as unbiased.

