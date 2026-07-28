# Technical architecture

## Research object

The environment is a cooperative semi-Markov Dec-POMDP for initial-attack
resource allocation. Fire and mission dynamics advance at one-minute resolution.
The joint policy acts every `decision_interval_min` by assigning each available
resource to one belief-derived tactical task.

The repository has four separately testable planes:

1. **Incident plane** — a STAC Item indexes source GeoTIFF, simulator NPZ,
   timestamped GeoJSON perimeters, and optional CF-NetCDF weather.
2. **Simulation plane** — hidden fire truth, a delayed/noisy incident belief,
   suppression fields, resource missions, logistics, and event logging.
3. **Decision plane** — PettingZoo/RLlib adapters, exact and heuristic
   comparators, and a task-pointer MAPPO implementation.
4. **Evidence plane** — paired-seed evaluation, historical hindcast/shadow/
   counterfactual modes, and deterministic Zarr/Parquet replay bundles.

## Information boundary

Executing actors receive:

- local resource state, payload, endurance, and assignment history;
- candidate tasks generated from delivered observations;
- a task compatibility mask;
- fleet readiness, public incident time, and aggregate belief diagnostics.

They do not receive true fire phase, true intensity, true burned area, or
unobserved fire-growth events. The centralized critic reads a separate
privileged feature vector during training. Learned execution supplies a zero
critic vector and actor logits are structurally independent of the critic
encoder. Tests mutate hidden truth while holding belief fixed and assert that
actor observations do not change.

Static elevation, barriers, fuel and objective layers are assumed to be known
scenario inputs. They influence candidate construction but are not dynamic
truth observations.

## Task and matching semantics

The task generator extracts exposed belief-front cells and emits compatible
observe, water, retardant, reinforce, and hold alternatives. Each task carries
a target, expected value, uncertainty, ground dependency, and capacity.

Independent actor actions may compete for one task. The simulator resolves
conflicts in a seeded random auction and records attempts, acceptances, and
blocked actions. The `joint_assignment` comparator instead solves the finite
maximum-weight resource/task assignment exactly over the same task graph. It is
the minimum serious non-learning comparison for policy results.

## Fire and intervention kernel

The fast kernel is a stochastic cellular surface-fire approximation:

- Rothermel-dimensional baseline surface ROS from explicit fuel parameters;
- directional wind and local-slope effects;
- per-cell fuel-load and correlated residual multipliers;
- stochastic eight-neighbor ignition and independent short-range spotting;
- fuel consumption, flaming-to-burned transition, and explicit barriers;
- intensity-dependent water effect, oriented retardant footprints, and a
  raster ground-hold field.

Time-varying wind can be interpolated from CF-NetCDF. Temperature and relative
humidity are retained in the forcing record but are not converted into dynamic
dead-fuel moisture in version 0.2; doing that requires a separately calibrated
moisture model.

This kernel is designed for throughput and falsifiability. It is not a
validated fire-behavior model. Per-cell Scott/Burgan fuel parameters, crown
fire, plume dynamics, smoke, long-range spotting, and atmosphere coupling are
outside its current validity envelope.

## Resource dynamics

Dispatch latency, cruise time, mission execution, return, reload, payload,
endurance, and withdrawal are represented explicitly. Resource positions are
interpolated along each mission leg for replay and state inspection. The policy
selects tactical destinations; flight control, deconfliction, base queuing,
maintenance and crew duty rules remain future model components.

## Learning implementation

The included model is parameter-shared recurrent MAPPO:

- task/resource encoders and masked pointer logits;
- a GRU state per resource;
- a centralized value function with a privileged training-only input;
- clipped PPO objective and value update, GAE, entropy regularization,
  gradient clipping, AMP, checkpointed optimizer/config/RNG state;
- optional masked behavior-cloning warm start from an explicit comparator
  policy before on-policy PPO;
- data-parallel training with `torchrun` and DDP.

The actor is decentralized at execution but currently has no learned
inter-agent message channel. Simulation remains CPU-resident; model inference
and optimization run on GPU. A native batched fire kernel becomes justified
when profiling shows rollout simulation, rather than the policy, dominates
wall time.

## Replay contract

Every replay is a directory containing:

- chunked Zarr arrays for truth, belief, treatment, terrain and resource state;
- Parquet events with minute, type and JSON payload;
- JSON metadata containing the exact scenario, policy, episode summary,
  resource schema and checkpoint SHA-256.

Rendering is downstream of replay. The training process remains headless and a
stored episode can be rendered repeatedly in 2D, terrain-aware 3D, or MP4
without rerunning stochastic dynamics.
