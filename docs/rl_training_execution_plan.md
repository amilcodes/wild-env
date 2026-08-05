# Fire-coupled RL execution plan

## Objective

Train and evaluate policies that coordinate aerial resources across front
attack, reload, refuel, and service contention while responding to a delayed
belief about an evolving fire. Large-scale policy optimization uses the tensor
incident surrogate. Policy selection and scientific claims use the canonical
simulator and held-out incident protocol.

## Current state, 2026-08-02

The first implementation tranche is complete:

- fixed-shape fire, belief, suppression, resource, site, and task state;
- device-resident continuous probabilistic fire transition;
- episode-level parameter randomization;
- belief-only advancing-front extraction;
- water/retardant deposition with exact liquid-volume conservation;
- event-level flight, queue, reload, endurance, reserve, stock, and capacity;
- privileged centralized critic with actor/truth separation;
- absorbing terminal state and equal-horizon escape handling for training;
- entity-attention MAPPO collector and trainer integration;
- recurrent sequence minibatches with truncated backpropagation and hidden
  reset at episode boundaries;
- strict full-graph capture test;
- cluster manifest and Slurm/DDP launcher;
- hold, uniform-valid, and public-belief greedy comparators;
- a retained local mechanism/throughput artifact; and
- a one-update end-to-end optimizer smoke run with a checkpoint.

No trained policy or accelerator performance is claimed.

## Execution sequence

### 1. Freeze the surrogate contract

Status: complete for version 1.

Required gates:

- tensor shapes are static;
- phase probability is conserved;
- delivered liquid is conserved;
- action masks enforce deterministic operating constraints;
- actor outputs are invariant to hidden-truth mutation;
- terminal worlds are absorbing;
- `torch.compile(fullgraph=True)` rejects graph breaks; and
- a fire-responsive comparator separates from hold on matched worlds.

Tests are in `tests/test_tensor_incident.py`. The CPU artifact is
`results/tensor_incident/cpu_mechanism_study.json`.

### 2. Calibrate against canonical teacher traces

Status: in progress. The snapshot projection bridge and first uncontrolled
synthetic check are complete. Six 36-minute teacher cases give mean absolute
cumulative-fire-fraction delta error 0.000162 with 6/6 canonical outcomes
inside the surrogate 90 percent ensemble interval. The check has one scenario
family, zero asset loss in its short forecast window, and was used to adjust
the base spread scale; it does not close this stage.

Generate paired canonical traces over a designed experiment spanning fuel
families, slope, wind speed/direction, moisture, ignition geometry, observation
delay, fleet mix, site geometry, water/retardant placement, and no-action
controls. Project each canonical snapshot to the surrogate grid and retain:

- cumulative burned probability and newly burned area;
- directional front displacement and arrival-time quantiles;
- value-weighted active risk and loss;
- treatment mass, effective coverage, and persistence;
- per-resource cycle, queue, payload, endurance, and utilization statistics;
- candidate-front location, orientation, lifetime, and capacity; and
- actor-observable belief tensors separately from truth targets.

Fit only the exposed `TensorIncidentParameterRanges` and documented reward
normalization. Use incident/scenario group splits. Report errors by regime and
retain residual distributions rather than one global fit.

Initial gates:

- median one-step front displacement error below one surrogate cell;
- cumulative burned-fraction error below 10 percent of the canonical range;
- treatment mass error below numerical tolerance;
- suppression effect direction agrees in at least 95 percent of paired cases;
- median sortie-cycle-time error below one decision interval;
- front-task recall above 0.90 within two surrogate cells; and
- uncertainty intervals cover canonical outcomes at their declared rates.

Failure of a gate narrows the surrogate’s use. It does not justify tuning on
the held-out evaluation set.

### 3. Accelerator profile and batch sizing

Status: blocked only by access to a CUDA/ROCm cluster.

This is the first point at which input from the repository owner is required:

- cluster login and scheduler instructions;
- account/partition/QoS and wall-time limits;
- GPU model and count;
- permitted container or Python environment; and
- durable checkpoint/results path.

Run `tools/run_tensor_incident_study.py --device cuda --compile` at batches 128,
256, 512, 1024, 2048, and until memory or throughput saturates. Profile eager
and compiled transitions separately from policy inference and PPO. Use the
same world, grid, resource, and segment dimensions intended for training.

The static estimator records a 5.16 GiB lower bound for the 2,048-world
manifest, including 4.57 GiB of rollout storage. Compiler workspaces,
transition intermediates, autograd activations, framework caches, and NCCL are
outside that number, so it cannot be used as the final batch-size decision.

Single-rank gates:

- zero graph breaks after warm-up;
- no host transfer in rollout collection;
- stable memory allocation after warm-up;
- at least 70 percent achieved GPU utilization during the environment-policy
  loop, with 90 percent as the target;
- no numerical divergence over ten complete horizons; and
- a measured batch size selected from throughput rather than memory alone.

Scale to DDP only after the single rank is saturated. Record samples/s,
all-reduce fraction, and scaling efficiency at 1, 2, 4, and 8 devices.

### 4. Baselines and curriculum

Status: code path implemented; full runs require the cluster.

Run the following controlled policy set:

1. hold and randomized valid action;
2. `incident_risk_greedy`;
3. recurrent task-pointer IPPO;
4. shared-parameter entity-attention MAPPO;
5. ablation without privileged critic truth;
6. ablation without observation delay/uncertainty; and
7. ablation without service contention.

Curriculum stages:

1. one ignition, one water base, homogeneous fleet, fixed wind;
2. mixed water/retardant fleet and two service modes;
3. site queues, finite stock, reserve, and closure variation;
4. randomized wind, fuel, slope, assets, and observation cadence;
5. latent response parameters and sensor reliability;
6. resource loss, site loss, wind shift, and communications dropout; and
7. canonical trace mixtures near the surrogate validity boundary.

Advance a stage when return, expected loss, tail loss, escape, wasted volume,
and constraint rates are stable across at least five seeds and the policy
beats the public-belief comparator on the selection set.

### 5. Algorithm branches

Status: specified, not implemented in the current trainer.

The controlled first extension is constrained MAPPO with one cost critic per
reported constraint and nonnegative dual variables. Hard masks remain in
force. Cost limits and dual learning rate must be frozen in the manifest.

The second extension is an autoregressive task decoder, because site and front
capacity couple aircraft actions. Compare it against the existing sequential
capacity-conditioned sampler using identical encoders and parameter budgets.

HAPPO is justified only if separate vehicle-class policies outperform shared
parameters under matched data and compute. A hierarchical sector manager is
deferred until the flat assignment policy fails specifically from action-space
or horizon scaling.

The PPO updater now trains through fixed recurrent sequences and resets hidden
state at episode boundaries. It initializes each sequence from the hidden state
recorded on-policy at collection time. A burn-in comparison remains useful for
quantifying hidden-state staleness across PPO epochs before making a strong
claim about long-memory performance.

### 6. Canonical transfer

Status: pending teacher calibration and trained checkpoints.

Evaluate the frozen surrogate checkpoint without optimization in the canonical
environment. Then fine-tune on canonical rollouts at a much smaller batch.
Keep three checkpoint identities: surrogate pretrain, canonical fine-tune, and
final selected model. Never overwrite the pretrained checkpoint.

Measure:

- zero-shot and fine-tuned policy rank;
- action-mask disagreement;
- assignment and service-cycle divergence;
- outcome degradation by wind/fuel/observation/suppression regime;
- policy sensitivity to surrogate parameter range; and
- catastrophic policy failures, not only mean return.

The transfer gate is stable ranking against greedy and optimization baselines
across at least three fidelity levels. A policy that wins only in its training
surrogate does not advance.

### 7. Historical and operational evaluation

Status: pending valid incident corpus, calibrated suppression, and policies.

Use incident-held-out partitions and the existing two-perimeter initialization,
forcing, observation-likelihood, and historical-fuel controls. Historical
evaluation must separate:

- hindcast fire-spread skill without suppression;
- shadow-mode policy decisions on the observed incident;
- within-model counterfactual suppression comparisons; and
- any field-effect claim, which requires intervention records and remains out
  of scope with the current data.

Report paired incident-cluster intervals, median and tail value loss, burned
area, escape, containment time, liquid delivered/wasted, resource utilization,
queue delay, and all constraint violations. Compare persistence, canonical
heuristics, rolling-horizon optimization, pretrained policy, and fine-tuned
policy.

## Release rule

A checkpoint is a research release candidate only when its exact source tree,
manifest, dependency lock, hardware profile, seed set, training metrics,
evaluation artifact, and replay bundle are frozen together. The README may
describe implemented capability. It may claim policy quality only from the
held-out artifact.
