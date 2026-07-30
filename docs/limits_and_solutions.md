# Limits and solution program

## Purpose

This document is the standing closure plan for the scientific, computational,
operational, and software limits of Aeolus-IA. It separates work that is
principally constrained by compute from work constrained by data, model
structure, validation, or operational evidence.

The current system supports numerical fire-front research, mechanism studies,
historical perimeter replay, within-simulator policy comparisons, and batched
component benchmarks. It does not yet support operational wildfire prediction,
historical suppression-effectiveness claims, autonomous dispatch, or a claim
that a learned policy improves on incident doctrine.

The three primary scientific gates are:

1. beat persistence on held-out advancing-front metrics;
2. calibrate suppression placement and response against independent data; and
3. train and evaluate a policy in the fully coupled fire-belief-operations
   environment.

## Current execution record

The 2026-07-29 historical-accuracy iteration completed the first implementation
pass against sections 2.1 through 2.3:

- absolute historical forcing alignment;
- 14-day prognostic dead-fuel-moisture spin-up;
- exact FBFM40 loading and bed depth without duplicate spread scaling;
- coarse-grid advancing-front lifecycle correction;
- causal two-perimeter coupled-state initialization;
- seven-method ablation over six incidents and 24 held-out transitions; and
- incident-cluster uncertainty, probabilistic scoring, and spatial failure
  atlases.

The result and remaining validity boundary are recorded in
[Historical accuracy study](historical_accuracy_report.md). Persistence still
wins cumulative extent and boundary metrics. Incident-grade wind, spatial
moisture, observation uncertainty, suppression-aware targets, and a larger
incident-held-out benchmark remain open gates.

The 2026-07-30 non-compute-bound fidelity iteration implemented the next layer
of sections 2.1 through 2.3:

- native-grid NOAA HRRR analysis extraction with cached, explicit missing-hour
  coverage and a NASA POWER spin-up background;
- terrain lapse-rate and conserved-vapor-pressure downscaling to the fire grid;
- 60-day NFDRS-v4-style growing-season-index spin-up for herbaceous and woody
  live-fuel moisture;
- live-herbaceous and live-woody axes in the packaged Pyretechnics behavior
  table, including Scott-Burgan dynamic load transfer;
- duplicate-timestamp perimeter union, a declared Gaussian localization model,
  soft scores, and interval-censored arrival scoring;
- an IRWIN-linked Fireline Engagement archive audit; and
- a deterministic selector for a 36-incident benchmark from all 737 NIROPS
  incident series.

The implementation record, audits, ablation protocol, and remaining validity
boundary are in [Historical fidelity iteration v4](historical_fidelity_v4.md).
These changes close representation defects. They do not yet establish improved
multi-incident forecast skill; that requires running and freezing the expanded
held-out study. The paired Electra diagnostic is mixed: ensemble cumulative
IoU and boundary distance improve substantially, raw-physics scores and
ensemble advancing-front localization deteriorate, and persistence remains the
strongest cumulative baseline.

The 2026-07-30 non-compute-bound P1 control pass added:

- acquisition-window, availability-time, observability, detection, false-alarm,
  and spatial-localization likelihoods for raster fire observations;
- historical fuel-product chronology screening and explicit fast-kernel
  numerical/regime validity classification;
- density-altitude/payload performance surfaces, vector wind groundspeed,
  terrain ceiling, service-geometry, reserve, and time-active airspace gates in
  the canonical simulator;
- incident/group split audits and paired case-cluster bootstrap policy effects;
  and
- a falsification study covering those controls.

The original audit fails the six-incident historical fuel set because every
bundle uses LANDFIRE 2025 for a 2020–2023 incident. A subsequent reconstruction
now supplies six exact-grid LANDFIRE 2016 Remap bundles with pre-incident
disturbance cutoffs, raw source rasters, and checksums. Five incidents retain a
documented request for a closer archived vintage. The expanded 36-incident
partition gate still fails because split, ecoregion, fuel-family, and weather-
regime labels have not been frozen. The planning study finds that the configured
sortie-cost trade favors near-no-action behavior: rollout improves the
loss-plus-cost objective over exact assignment while increasing terminal fire
loss and failing to separate from no action. Full methods and results are in
[Non-compute-bound P1 research](noncompute_p1_research.md). Exact closure gates,
dependencies, and pass criteria are in
[Non-compute-bound P1 remaining work](noncompute_p1_remaining_work.md).

The same iteration selected a traceable operations fleet: CAL FIRE S-2T,
C-130H, S-70i, UH-1H, OV-10A, and King Air platforms plus three
interagency-reference UAS. Nominal public specifications and research
assumptions are now distinct machine-readable fields. None of the profiles is
promoted to field-performance-ready without an authorized flight-manual or
engineering surface. The evidence boundary and closure package are in
[Aviation vehicle closure](aviation_vehicle_closure.md); the fuel rebuild is in
[Historical fuel validity](historical_fuel_p0.md).

The 2026-08-02 RL systems pass implemented a second, fire-coupled accelerator
environment. `TensorIncidentEnv` combines continuous probabilistic spread,
truth/belief separation, periodic observation assimilation, belief-derived
front tasks, volume-conserving water/retardant placement, latent per-world
physics and response, event-level aircraft/service logistics, outcome reward,
and explicit constraint costs in one fixed-shape tensor transition. It is
wired into entity-attention MAPPO, passes full-graph capture and mechanism
tests, and has a retained 32-world CPU falsification artifact. This closes the
architectural gap between static operations pretraining and fire-responsive
mass training. Canonical-teacher calibration, recurrent sequence updates,
constrained PPO, accelerator profiling, full training, transfer, and held-out
evaluation remain open. See [Fire-coupled tensor environment](tensor_incident_environment.md)
and [RL execution plan](rl_training_execution_plan.md).

The 2026-08-04 held-out historical-skill pass found and repaired a physical
coordinate defect in the prepared study: EPSG:3857 map metres had been used as
ground metres, making physical area 43--122% too large across the six incident
latitudes.
The replacement corpus uses local UTM grids and cell-center perimeter
rasterization. A frozen chronological 2/2/2 incident pilot now evaluates
globally selected parameters without incident-specific test calibration. It
also materializes 24 transition-specific archived HRRR forecasts from cycles
available before issue time, with causal fuel-moisture initialization and
artifact digests. On two unseen 2023 incidents, extent selection collapses to
persistence; front selection reaches F1 0.112 but reduces IoU from 0.8465 to
0.8353 and raises boundary error from 127.8 to 144.6 m. The primary historical
skill gate remains open. Methods, results, and the ordered closure program are
in [Held-out historical skill v6](heldout_historical_skill_v6.md).

## Classification

### Compute-bound

- fully device-resident fire, belief, suppression, task, and fleet transition;
- fire-atmosphere teacher simulations and correction-model training;
- large sequential data-assimilation ensembles;
- fire-coupled MARL training;
- multi-GPU and multi-node scaling evidence; and
- large-domain replay and visualization.

### Non-compute-bound

- historical forecast validity and generalization;
- incident wind, moisture, and fuel provenance;
- suppression placement and response calibration;
- aircraft, service-site, crew, and airspace realism;
- observation and communications models;
- decision semantics, objectives, and safety constraints;
- interoperable data contracts;
- experiment provenance and release discipline; and
- operational governance.

### Hybrid

Data assimilation, spotting, crown transition, policy generalization, and
visualization fidelity require both model/data work and enough compute to
evaluate uncertainty.

## Compute-bound program

### Unified accelerator-resident simulation

Use one batched state with explicit environment, raster, resource, site, and
task dimensions. Implement the minute transition as a pure function returning
the next state, observations, costs, and events. The transition must include:

- WENO/RK level-set propagation;
- fuel consumption, intensity, and spotting;
- water, retardant, and fireline state;
- belief prediction and observation assimilation;
- advancing-front extraction and candidate-task generation;
- fleet movement, queues, loading, and assignment; and
- reward, constraints, and terminal conditions.

Keep the existing NumPy simulator as the reference. Use PyTorch compilation
first because the tensor fire and policy implementations already use it. Port
only measured bottlenecks, likely WENO stencils, level-set reinitialization,
spotting, treatment footprints, distance transforms, and front extraction, to
Triton or C++/CUDA.

Use structure-of-arrays resource state, fixed maxima with masks, dense raster
execution for small grids, active-front tiles for larger grids, and
counter-based random streams keyed by episode, time, cell, resource, and event.

Exit gates:

- canonical/tensor parity for every state transition;
- no mandatory host synchronization inside a rollout;
- at least 90% GPU utilization on the target workload;
- at least 10 times canonical-simulator throughput; and
- a full incident replay within declared numerical tolerances.

### Fire-atmosphere and three-dimensional fuel hierarchy

Use three fidelity levels:

1. the fast level-set training model;
2. a learned correction operator trained on coupled-model output; and
3. WRF-SFIRE/QUIC-Fire as high-fidelity teachers and evaluators.

The correction operator should predict residual wind, rate of spread, plume
state, and spotting scale from terrain, three-dimensional fuels, ambient
weather, and the current fire. It must also predict uncertainty and enforce
bounded energy injection, nonnegative spread, and monotone fuel consumption.

Generate teacher cases across terrain, atmospheric stability, wind profiles,
fuel structures, moisture, ignition geometry, and intensity. Use FastFuels or
equivalent inventory/LiDAR-derived fuel realizations for high-fidelity cases.

Exit gates:

- lower held-out wind and spread residuals against coupled teachers;
- rising uncertainty on out-of-distribution cases;
- stable policy rankings across fidelity levels; and
- explicit detection of unsupported plume-dominated conditions.

### Sequential data assimilation at scale

Assimilate arrival-time fields, spread corrections, wind corrections, fuel
moisture, and spotting parameters. Use:

- front registration or morphing before amplitude updates;
- localized ensemble-transform updates;
- particle weights for non-Gaussian histories;
- likelihood tempering;
- rejuvenation following resampling; and
- backward fire-history replay after a perimeter correction.

Satellite detections enter through a sensor likelihood that represents pixel
footprint, cloud, detection probability, false alarms, and geolocation error.
They are not exact burning cells.

Batch ensemble members on the accelerator and localize expensive updates to the
active front. Preserve posterior ancestry, weights, and observation lineage.

Exit gates:

- effective sample size above 25 percent of the ensemble after routine updates;
- calibrated posterior probabilities;
- improvement in the next forecast, not only fit to the current perimeter; and
- held-out advancing-front skill above persistence.

### MARL training

Train in stages:

1. operations pretraining with generated front segments;
2. fire-coupled curriculum over wind, observation delay, spotting,
   heterogeneous fleets, communications, and service failures; and
3. uncertainty training over physical, observation, communications, and
   suppression-response ensembles.

Maintain a controlled algorithm set:

- MAPPO;
- IPPO;
- HAPPO for heterogeneous resources;
- an autoregressive assignment model such as MAT;
- constrained MAPPO-Lagrangian; and
- planner imitation followed by RL fine-tuning.

Train recurrent policies with trajectory-aware sequence minibatches, burn-in,
true episode-boundary resets, and truncated backpropagation through time.

Exit gates:

- at least five independent training seeds;
- frozen evaluation over at least 100 paired seeds per scenario family;
- improvement over doctrine, greedy, no-action, and optimization baselines;
- transfer to incident-, region-, and weather-held-out scenarios; and
- separately reported safety and resource-constraint violations.

### Cluster evidence

Use one process per GPU, environment sharding, DDP, NCCL, asynchronous
checkpointing, and preemption-aware resume. FSDP is only justified if the policy
model becomes the memory bottleneck.

Benchmark:

- 1, 2, 4, and 8 GPUs on one node;
- 2, 4, and 8 nodes;
- 128, 256, and 512 cell domains;
- 64 through 4096 concurrent environments; and
- fire-only, operations-only, and coupled transitions.

Retain environment-minutes/second, decisions/second, active-cell updates/second,
samples/joule, memory, graph breaks, kernel launches, scaling efficiency, and
time to a fixed evaluation score.

A cluster claim requires the exact configuration, container digest, hardware
inventory, profiler trace, and result artifact.

### Replay and visualization scale

Replace the in-memory recorder with appendable time-chunked Zarr arrays, a
bounded asynchronous writer, atomic frame commits, resumable manifests, and
recording profiles. Store homogeneous weather as time series plus correction
fields. Store high-rate vehicle trajectories separately from slower raster
frames.

Reuse VTK actors and array buffers, build terrain once, add raster pyramids,
perform asynchronous chunk loading, and use ParaView/Catalyst for in-situ or
remote analysis of cluster-scale runs.

Exit gates:

- bounded-memory recording of a 512-cell, 24-hour incident;
- readable committed frames after interruption;
- interactive playback on a reference workstation; and
- coordinate/unit agreement between the native viewer and ParaView.

## Non-compute-bound program

### Historical spread validity

Build a preregistered benchmark of at least 30 to 50 incidents spanning four or
more ecoregions, major surface-fuel families, and low through extreme wind
regimes. Record timestamp uncertainty, spatial uncertainty, acquisition method,
and complete provenance for every observation.

Split by incident, geography, and year. Do not use future perimeters during
initialization. Evaluate 1, 3, 6, 12, and 24 hour horizons. Separate cumulative
extent from newly active front. Report boundary displacement, symmetric
difference, front F1, arrival-time error, Brier score, CRPS, and reliability
with incident-cluster uncertainty intervals.

Compare persistence, constant-rate extrapolation, the fast model, an established
spread model, and the posterior ensemble. Censor or explicitly mark intervals
with unidentified historical suppression.

Exit gate: beat persistence on held-out advancing-front metrics with
incident-cluster confidence intervals.

### Incident wind and fuel moisture

Use a forcing hierarchy:

1. NOAA gridded analysis or forecast as the synoptic background;
2. quality-controlled RAWS observations;
3. terrain-aware diagnostic downscaling;
4. optional WRF/QUIC correction fields; and
5. uncertainty carried into fire and policy ensembles.

Analyze wind in Cartesian components and enforce terrain-aware mass consistency.
Maintain explicit 1, 10, 100, and 1000 hour dead-fuel moisture plus herbaceous
and woody live-fuel moisture. Drive dead fuels with time-lag models and
precipitation wetting. Assimilate station fuel-stick observations with bias and
representativeness terms.

Every forcing sample must record source cycle, valid time, forecast lead,
analysis method, station quality flags, and uncertainty.

### Fuel representation

Use a 30 m fire-behavior fuel layer for incident-scale runs and optional
three-dimensional fuel realizations for teacher simulations. Three-dimensional
state includes canopy base height, canopy bulk density, ladder fuels, shrub and
grass strata, dead woody classes, and uncertainty.

Record fuel-product vintage, source disturbance date, known treatments, native
resolution, resampling method, and class mapping. Historical simulations must
not use fuel products containing post-incident disturbances.

Avoid multiplying a Scott/Burgan behavior model by an unrelated family-level
fuel-load proxy. Use model-specific parameters for spread and a separately
defined physical mass field for consumption, emissions, and treatment response.

### Spotting and crown transition

Separate spotting into ember production, lofting, transport, burnout,
interception, landing ignition, ignition delay, and secondary-fire growth.
Calibrate hierarchical distributions by fuel and fire regime.

Validate surface-to-crown initiation, passive crown behavior, active crown
propagation, and crown-to-surface transition separately. Carry parameter
uncertainty into the training distribution.

### Suppression calibration

Separate placement physics from treatment response.

The placement model uses release altitude, speed, heading, tank/door flow,
vertical wind, terrain, canopy, and evaporation to predict coverage and its
uncertainty. The response model estimates spread reduction as a function of
material, coverage, fuel, moisture, intensity, age, and rain.

Fireline construction and holding require resource-, slope-, vegetation-,
intensity-, and time-dependent production, strength, engagement, and burnover
models.

Required records include dispatch, launch, arrival, release, drop polygon,
coverage, objective, turnaround, line geometry and completion, resource
handoffs, observed pre/post fire state, and forcing provenance.

Exit gates:

- placement and response calibrated separately;
- held-out coverage and response predictions with uncertainty; and
- historical claims restricted to incidents with adequate action chronology.

### Aircraft and drone performance

Use precomputed tactical performance surfaces indexed by vehicle, payload,
altitude, temperature, wind, leg, reserve, and flight phase. Validate them
against a six-degree-of-freedom model such as JSBSim.

Use PX4 software-in-the-loop and then hardware-in-the-loop for selected uncrewed
systems. The learned policy emits mission-level intents; a verified guidance
layer owns low-level flight control.

### Airspace and safety

Represent routes as horizontal corridor, altitude band, and time interval.
Model TFRs, terrain clearance, approach corridors, drop-lane occupancy,
supervision authorization, separation, lost-link routes, and alternates.

Use a deterministic external safety shield to reject or repair infeasible
learned actions. Record every intervention. Learned reward does not replace
hard safety constraints.

### Service sites and ground resources

Service sites require geometry, approach direction, dimensions/depth, obstacle
clearance, ownership, current closure, volume, refill rate, environmental
restrictions, weather limits, and timestamped verification.

Ground resources require terrain-network travel, vegetation/slope productivity,
shift limits, fatigue, safety zones, escape routes, trigger points, firing
operations, mop-up, maintenance, and resupply.

### Observation, belief, and communications

Define a generative likelihood for each sensor including footprint, line of
sight, pixel integration, cloud/smoke, saturation, misses, false alarms,
geolocation uncertainty, processing delay, scan schedule, and failure.

Represent network coverage, bandwidth, packet loss, latency, stale messages,
relays, and partitions. The centralized critic may use privileged training
state; deployed actors receive only accessible timestamped information.

Store probabilistic burn, arrival, and intensity state with observation lineage
and uncertainty. Evaluate belief calibration independently of policy return.

### Decision semantics and objectives

Use three control levels:

1. incident objectives, sectors, defended assets, and time budgets;
2. resource assignment and service scheduling; and
3. route, line geometry, and release parameters.

Remove resource-order dependence with permutation-randomized autoregression,
learned agent ordering, matching, auctions, or min-cost flow. Test invariance by
shuffling resource identifiers.

Keep reward and constraints separate. Reward covers expected loss avoided,
containment progress, information, and productivity. Constraints cover reserve,
separation, duty, exposure, site capacity, authority, and communications.

### Baselines and evaluation

Retain no-action, nearest, anchor-and-flank, and greedy baselines. Add exact
assignment for small cases, rolling-horizon MILP or CP-SAT, stochastic MPC,
planner imitation, MAPPO, IPPO, HAPPO, and constrained MAPPO.

Evaluate normal operations plus observation delay, communications loss, site
closure, vehicle failure, spotting burst, wind shift, model misspecification,
out-of-region fuels, and unsupported plume cases.

### Interoperability

Keep `IncidentBundle` as the internal schema and map its assets to:

- STAC Collection and Item metadata;
- Cloud-Optimized GeoTIFF for immutable rasters;
- Zarr with CF-style metadata for time-varying fields;
- GeoParquet for perimeter, line, drop, and site vectors;
- OGC Moving Features for vehicle trajectories; and
- a versioned JSON event stream for decisions and state changes.

Require JSON Schema validation, semantic units, CRS, provenance, checksums, and
explicit schema migrations.

### Visualization semantics

Keep observed perimeter, assimilated belief, simulated state, posterior
probability, policy forecast, treatment, and counterfactual layers visually and
semantically distinct. Every layer carries units, CRS, source, timestamp, and
an observed/inferred/simulated classification.

Add actual altitude, uncertainty, observation age, sensor footprints,
communications, planned/executed routes, airspace reservations, predicted and
realized drop coverage, safety-shield interventions, and synchronized policy
comparisons.

### Recovery, software, and release discipline

Checkpoints contain model, optimizer, scheduler, scaler, all random states,
environment and collector state, scenario sampler, normalization, curriculum,
commit, dirty-tree marker, container digest, configuration hash, and data
checksums.

Release work requires:

- a clean tagged revision;
- CPU CI and scheduled CUDA CI;
- coverage, type, property, numerical, and schema-migration tests;
- full transitive platform locks;
- container images pinned by digest;
- an SBOM and signed result manifest; and
- consistent package and document versions.

### Operational governance

The research boundary remains explicit until an external operational program
owns incident-command authority, aviation approval, certification, human
factors, security, environmental constraints, fail-safe behavior, field
testing, and hazard analysis.

The intended authority chain is:

```text
policy suggestion
    -> deterministic feasibility checks
    -> safety shield
    -> human authorization
    -> mission system
```

Each recommendation records the policy version, input belief, uncertainty,
rejected alternatives, constraint checks, and approving authority.

## Dependency order

1. Clean revision and frozen data contracts.
2. Historical benchmark and incident forcing.
3. Suppression and operations calibration.
4. Unified accelerator-resident simulator.
5. Sequential assimilation and coupled-model corrections.
6. Operational constraints and optimization baselines.
7. MARL curriculum and cluster training.
8. Held-out evaluation and uncertainty analysis.
9. Replay, paper, and release artifacts.
10. Software-in-the-loop and hardware-in-the-loop research.

This order prevents large training runs from optimizing assumptions already
known to require replacement.
