# Aerial MARL and accelerator design, v0.7

## Scope

This increment addresses sortie-level control of a heterogeneous aerial fleet:

- select an oriented water or retardant attack segment;
- select a compatible airport, helibase, retardant base, dip site, scoopable
  water body, or temporary tank;
- conserve payload volume across delivery and refill;
- account for travel, dispatch delay, endurance and reserve;
- contend for named service bays and finite suppressant stock;
- coordinate several aircraft on a segment without exceeding its useful liquid
  demand;
- train the assignment policy at large batch size without a GPU/CPU copy at
  every decision.

The learned action is an operational assignment. Flight-control, collision
avoidance, drop-gate control, and certified contingency management remain lower
layers. A policy output is therefore suitable as a candidate plan for a flight
management system or human supervisor, rather than direct actuator commands.

## Three fidelity levels

### Canonical incident environment

`AeolusSimulator` is the semantic and fire-coupled environment. It retains
truth/belief separation, delayed observations, WENO5/RK3 fire propagation,
spotting, volume-conserving water and retardant, line construction, line
breach, weather forcing, perimeter assimilation, and replay.

Version 0.6 adds explicit service-node state and learned service routing. When
service nodes are configured, an aircraft remains at the drop location with an
empty tank. It must select a service task. A dip site can restore water payload
without restoring endurance. A site offering `fuel` or `charge` restores sortie
endurance at service completion. The action mask verifies a recovery leg and
reserve before an attack is dispatched. Dispatch latency is charged against
endurance and site opening time. Hover-fill and scoop queue/service minutes are
airborne minutes; land-site queue/service minutes are not.

This implementation is deliberately retained as the parity oracle. It contains
Python and NumPy control flow and is unsuitable as the main source of billions
of transitions.

### Tensor operations environment

`TensorOperationsEnv` is a fixed-shape batched environment. Its complete
mutable state is held in PyTorch tensors:

```text
[batch, resource]  position, status, ETA, leg geometry, payload,
                   endurance, target, queue age, reserved load
[batch, segment]   position, heading, suppressant compatibility,
                   priority, required and remaining liquid
[batch, site]      stock, queue/service occupancy
```

The action layout is:

```text
0                       hold
1 ... K                 attack segment
K + 1 ... K + S         service site
```

Action masks, assignment, minute advancement, queue admission, refill
reservation, reward construction, policy inference, GAE storage, and PPO
optimization can remain on one accelerator. There is no per-decision
`.cpu().numpy()` path. Each DDP rank owns one large device batch and one model
replica. Training uses synchronous fixed-horizon batches, so it does not
regenerate a full device batch to reset a few early-complete episodes at every
decision; completion remains observable and rewarded inside the horizon.

This environment remains a useful operations-only ablation. It does not
execute a fire kernel; its attack segments are a generated compact contract.
The tensor incident tier below is the primary fire-responsive pretrainer, and
the canonical environment remains the transfer/evaluation reference.

### Tensor incident environment

`TensorIncidentEnv` is the fire-responsive mass-training tier. It retains the
same fixed action layout and service mechanics, and adds:

- a continuous probabilistic cellular fire on a configurable coarse grid;
- separate hidden truth and delayed actor belief;
- periodic blurred observation assimilation with uncertain reliability;
- dynamic oriented front tasks extracted from belief on the device;
- volume-conserving water and retardant footprints with subcell line state;
- per-world latent spread, wind, slope, residence, suppression, and sensor
  parameters;
- value-loss, burned-area, containment, and escape outcomes; and
- a separate vector of blocked, exhaustion, queue, and waste costs.

The transition uses a fixed-shape tensor `NamedTuple` and passes strict
`torch.compile(fullgraph=True)` capture. The collector, policy, critic, and PPO
path remain on the same device. The canonical WENO5/behavior-table environment
continues to define transfer and scientific validity. Full equations and
limitations are in
[`tensor_incident_environment.md`](tensor_incident_environment.md).

## Service-node contract

`ServiceSiteSpec` records:

- grid location and site identifier;
- site class and service mode (`land`, `hover_fill`, `scoop`);
- water, retardant, fuel, and charge availability;
- service bays and simultaneous approach capacity;
- refill rate and fixed turnaround;
- finite available volume;
- opening and closing minute;
- wind limit;
- minimum water depth and usable length;
- a manual-verification flag.

`load_service_sites_geojson` imports evaluated point features through the
landscape affine transform. It rejects unverified points by default. It does
not infer that an arbitrary polygon is safe. Operational water suitability
also depends on current depth, submerged and overhead obstacles, approach and
egress, other users, ownership, temporary restrictions, and aircraft-specific
performance.

## Aircraft and dispatch feasibility

Each resource now carries:

- home site and supported service modes;
- payload capacity and current payload fraction;
- cruise speed, dispatch latency, and sortie endurance;
- resource-specific and scenario-level reserve;
- wind limit;
- drop dimensions and allowed line length.

For an attack action, the mask evaluates:

```text
current position -> attack segment -> nearest compatible fuel/charge site
```

For a water-only dip action, it evaluates:

```text
current position -> dip site -> nearest compatible fuel/charge site
```

All paths must fit inside remaining endurance minus reserve. Service-site
opening time, wind, mode compatibility, stock, and payload need are also
checked. Approach capacity includes aircraft already inbound, queued, or
servicing, and the airborne-service reserve check conservatively includes work
already committed to hover-fill or scoop sites. A hard mask handles
constraints that should never be explored.

The canonical travel model remains straight-line, but now evaluates sampled
terrain, density-altitude/payload performance, vector-wind groundspeed and
crosswind, derated endurance, terrain ceiling, service depth/length, and
time-active polygon/altitude airspace volumes. It rejects use outside a
vehicle's declared performance surface. The tensor operations environment has
not yet received those calculations. Neither environment finds detours,
separates flight phases, or models shared drop-lane occupancy.

## Coordinated line attack

`AERIAL_LINE` is an oriented attack task compatible with water and retardant
resources. It has capacity greater than one, so several aircraft can be
assigned to the same segment. In the canonical environment, each delivery is
passed through the volume-conserving drop footprint and treatment coverage
accumulates on the same ground line.

The tensor environment holds explicit remaining liquid demand per segment.
Concurrent contributions are summed, accepted delivery is capped by remaining
demand, and excess is recorded as waste. Reward uses protected-value-weighted
accepted delivery and penalizes excess, flying time, queue time, and blocked
actions.

This represents coordination at the assignment and liquid-delivery level. It
does not model formation flight or simultaneous occupancy of a drop lane.

## Included policy and comparator set

### Cycle-time greedy

`cycle_time_greedy` is a device-resident comparator. Loaded resources maximize
marginal protected value per travel minute. Empty resources minimize estimated
travel, queue, and refill time. Sequential assignment subtracts expected
segment demand so later resources are not credited for already-covered work.

### Exact joint assignment

The canonical `joint_assignment` comparator solves the belief-derived
resource/task matching problem with compatibility and capacity. It is useful
for behavior cloning and small-case regret measurements. It is not intended as
the online solver for large fleets.

### Recurrent task-pointer MAPPO

The retained baseline uses parameter sharing, task pointers, a recurrent actor,
hard action masks, and a centralized privileged critic. It is a necessary
reference because well-implemented MAPPO remains a strong cooperative MARL
baseline.

### Entity-attention capacity-aware MAPPO

`EntityAttentionActorCritic` adds permutation-equivariant self-attention over
resources and tasks. The actor can represent fleet telemetry, service nodes,
and front segments without flattening them into an order-dependent vector.
The action head is a masked task pointer.

Joint sampling is sequentially capacity-conditioned. Once an assignment
consumes a unit-capacity task, later resources cannot sample it. Multi-capacity
attack segments and service approaches remain available until their capacity
is consumed. PPO evaluation reconstructs the same conditional distributions.

This is the primary implemented policy for cluster experiments.

## Algorithm research queue

The following are experiment branches, not current performance claims:

1. **HAPPO/HATRPO:** use sequential heterogeneous policy updates when aircraft
   classes require separate policies and shared-parameter MAPPO produces
   destructive updates.
2. **MAT or PARCO-style decoder:** condition each assignment on earlier
   resource actions and learn resource priority. This is a better fit than
   independent categorical heads when task conflicts dominate.
3. **Sable-style retention:** replace quadratic temporal attention/GRU memory
   for very large fleets or long partial-observation histories.
4. **Constrained MAPPO:** maintain separate cost advantages and dual variables
   for probabilistic reserve, communication, drop-clearance, environmental, and
   duty constraints. Hard masks remain in force for deterministic constraints.
5. **Hierarchical control:** a slower sector/site manager chooses incident
   sectors and staging sites; the current policy assigns sorties and attack
   segments. The hierarchy should be introduced only after the flat policy and
   rolling-horizon optimizer are measured on the same cases.
6. **Planner distillation:** generate demonstrations with rolling-horizon
   mixed-integer/min-cost-flow allocation, behavior-clone the task pointer, then
   fine-tune under stochastic fire and queue dynamics.

## Training sequence

1. Mechanism tests on hand-constructed two-site, two-aircraft cases.
2. Behavior cloning from `joint_assignment` and a rolling-horizon operations
   solver on small canonical cases.
3. Tensor-operations pretraining with randomized fleet availability, service
   stock, closures, winds, front geometry, segment priority, and liquid demand.
4. Curriculum from one site/one fire sector to several constrained sites and
   heterogeneous fleets.
5. Fire-coupled canonical fine-tuning with delayed belief state and posterior
   weather/fire ensembles.
6. Paired evaluation against greedy, exact assignment, and rolling-horizon
   planning on unseen incident seeds and withheld historical incidents.
7. Stress evaluation with sensor delay, site closure, resource loss, wind
   shift, communication loss, and incorrect water-site availability.

The optimization unit is a sortie assignment, typically every three minutes.
Training targets should include at least 5–10 seeds per configuration and
report medians, bootstrap intervals, tail loss, constraint violations, and
paired policy differences.

## Cluster topology

`configs/cluster_tensor_operations.yaml` is sized for accelerator training:

- 2,048 environments per DDP rank;
- 12 suppression aircraft;
- 48 attack segments and 6 service nodes;
- 128 rollout decisions;
- entity-attention width 256, 8 heads, 3 layers;
- 4,096 environment-transition PPO minibatches (49,152 resource decisions);
- AMP and model compilation enabled.

These are starting values for an H100/A100-class device, not validated maxima.
Measure memory, simulator decisions/s, policy actions/s, achieved occupancy,
compiler graph breaks, and NCCL fraction. Scale the per-rank environment batch
until inference and stepping saturate the device, then add ranks. Multi-node
scaling before single-rank saturation generally wastes communication.

PyTorch recommends compiling the inner module before wrapping it in DDP; the
trainer follows that ordering.

## Current measured result

`results/rl_operations/cpu_mechanism_study.json` is a retained harness check on
the available machine:

- 32 parallel environments;
- 12 resources, 12 attack segments, 6 sites;
- 40 decisions;
- cycle-time greedy completion fraction 0.987 versus 0.963 for uniform-valid;
- cycle-time greedy complete-episode fraction 0.781 versus 0.625;
- approximately 31,500 agent decisions/s for the full environment on this CPU.

These numbers establish mechanics and comparator behavior. They do not predict
cluster throughput and are not evidence of an effective fire policy. The same
study command must be run on the target GPU and its artifact retained before a
GPU throughput statement is made.

`results/tensor_incident/cpu_mechanism_study.json` records the newer coupled
check on 32 matched worlds, 12 resources, 6 sites, a 64-by-64 grid, 32 dynamic
front slots, and 40 decisions. The public-belief comparator improves expected
loss and burned fraction against hold in every world. Eager CPU throughput is
approximately 10,900 agent decisions/s for that comparator and 9,500 agent
decisions/s for the complete entity-attention policy/environment path. The
strict capture test uses the eager compiler backend locally; the artifact does
not claim CUDA compiler speed.

The initial canonical-teacher projection check covers six synthetic
36-minute forecasts at the intended grid. Its cumulative-fire-fraction delta
MAE is 0.000162 and interval coverage is 6/6. This establishes that the bridge
and scale calibration operate; its narrow regime and zero short-window asset
loss leave suppression response and generalization open.

## Known gaps

The following are still material:

- surrogate parameter ranges have not yet been calibrated against projected
  canonical teacher traces or held-out incident regimes;
- the coupled surrogate omits crown transition, spotting, evolving moisture,
  dynamic wind, arrival-time history, and fire-atmosphere feedback;
- strict graph capture is tested locally, but Inductor/CUDA compile time,
  steady-state performance, memory, utilization, and graph replay have not
  been measured on an accelerator;
- PPO now uses recurrent sequence minibatches and truncated backpropagation;
  burn-in and a hidden-state staleness ablation across PPO epochs remain;
- explicit constraint costs are emitted but the trainer does not yet use cost
  critics or Lagrangian updates;
- site queues do not model taxi/runway sequencing, maintenance, crew duty, fuel
  truck contention, retardant mixing batches, hot-loading rules, or lead-plane
  availability;
- canonical aircraft performance accepts density-altitude/payload tables,
  vector-wind groundspeed, terrain ceiling, and time-active airspace volumes;
  the included table is synthetic, the tensor environment still uses constant
  performance, and weight and balance, pressure altitude, climb/descent, phase-
  specific energy, and reviewed flight-manual data remain absent;
- straight-route airspace blocking exists in the canonical simulator; route
  repair, drop-lane clearance, communications, ATGS authorization, obstacle
  clearance, separation, lost-link paths, and emergency landing remain absent;
- water-body stock is scalar and does not respond to bathymetry, drought,
  contamination, icing, waves, boats, or temporary closure;
- the example uncrewed-aircraft capabilities are research assumptions. They
  require replacement with vehicle-specific, independently reviewed performance
  and certification data;
- the outcome reward and intervention response remain surrogates. They must be
  calibrated against canonical traces, observed drop geometry, and measured
  change in fire spread;
- no trained checkpoint is claimed. Training requires a CUDA/ROCm cluster and a
  frozen evaluation manifest.

The current non-compute-bound closure gates, including canonical/tensor
constraint parity, are maintained in
[`noncompute_p1_remaining_work.md`](noncompute_p1_remaining_work.md).

## Primary references

- Lan et al., [WarpDrive: Extremely Fast End-to-End Deep Multi-Agent
  Reinforcement Learning on a GPU](https://arxiv.org/abs/2108.13976), 2021.
- Shacklett et al., [Madrona: A Fast, Lightweight, Scalable Platform for
  Functional Simulation](https://madrona-engine.github.io/), SIGGRAPH 2023.
- Gulino et al., [GPUDrive: Data-driven, multi-agent driving simulation at one
  million FPS](https://openreview.net/forum?id=ERv8ptegFi), ICLR 2025.
- Rutherford et al., [JaxMARL: Multi-Agent RL Environments and Algorithms in
  JAX](https://arxiv.org/abs/2311.10090), NeurIPS 2024.
- Yu et al., [The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games](https://arxiv.org/abs/2103.01955), NeurIPS 2022.
- Kuba et al., [Trust Region Policy Optimisation in Multi-Agent Reinforcement
  Learning](https://arxiv.org/abs/2109.11251), 2021.
- Wen et al., [Multi-Agent Reinforcement Learning is a Sequence Modeling
  Problem](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html),
  NeurIPS 2022.
- Mahjoub et al., [Sable: a Performant, Efficient and Scalable Sequence Model
  for MARL](https://proceedings.mlr.press/v267/mahjoub25a.html), ICML 2025.
- Berto et al., [PARCO: Parallel AutoRegressive Models for Multi-Agent
  Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f404c0259def290306159047b0b52584-Abstract-Conference.html),
  NeurIPS 2025.
- Al-Husseini et al., [Hierarchical Framework for Optimizing Wildfire
  Surveillance and Suppression using Human-Autonomous
  Teaming](https://arxiv.org/abs/2406.17189), 2024.
- Rodríguez y Silva et al., [Assignment Problems in Wildfire Suppression:
  Models for Optimization of Aerial Resource
  Logistics](https://academic.oup.com/forestscience/article/64/5/504/5001397),
  Forest Science 2018.
- USDA Forest Service, [Standards for Airtanker
  Operations](https://www.fs.usda.gov/sites/default/files/2020-08/fs_standards_for_airtanker_operations_-_final_08192020.pdf),
  2020.
- NWCG, [Dipsite
  definition](https://www.nwcg.gov/publications/pms205/nwcg-glossary-of-wildland-fire-pms-205/dipsite-helicopter-5)
  and [retardant/water drop
  safety](https://www.nwcg.gov/6mfs/aviation/retardant-and-water-drop-safety).
- NIFC, [2026 Interagency Standards for Fire and Fire Aviation
  Operations](https://www.nifc.gov/standards/guides/red-book).
- Wiard-Greene et al., [Investigating the impact of aerial firefighting on rate
  of wildfire spread](https://research.fs.usda.gov/treesearch/80803), 2025.
