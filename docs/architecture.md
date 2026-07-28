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

The version 0.3 kernel resolves per-cell Anderson/Scott–Burgan surface behavior
from a packaged Pyretechnics reference table. Wind and slope factors combine as
vectors, directional spread follows an ellipse, crown initiation/spread uses
Van Wagner and Cruz relations, dead-fuel moisture responds to weather and
precipitation, and spotting uses stochastic downwind/crosswind ember transport.
Adaptive Huygens-style raster front propagation replaces the old probabilistic
neighbor ignition rule.

The canonical NumPy path participates in the complete incident/resource
simulator. A separate `TensorFireKernel` keeps batches resident in PyTorch on
CUDA, ROCm, MPS or CPU. It shares the behavior table and equations, and parity
tests compare local outputs field by field. Replay records fuel model, canopy,
moisture, fire type, spread rate, flame length and fuel consumption.

Water changes current intensity and dead-fuel moisture. Oriented retardant and
ground line change spread conditioning without becoming unconditional
barriers. Intervention response remains an empirical research model.

The full numerical and physical contract is in
[`fire_behavior.md`](fire_behavior.md). Atmosphere coupling, plume dynamics,
smoke and combustion-resolving flow remain outside the model.

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
inter-agent message channel. The interoperable environment remains
NumPy-resident; model inference and optimization run on GPU. High-throughput
fire-only ensembles and future GPU-resident vector environments use the
batched PyTorch kernel.

## Replay contract

Every replay is a directory containing:

- chunked Zarr arrays for fire behavior, truth, belief, treatment, terrain and
  resource state;
- Parquet events with minute, type and JSON payload;
- JSON metadata containing the exact scenario, policy, episode summary,
  resource schema and checkpoint SHA-256.

Rendering is downstream of replay. The training process remains headless and a
stored episode can be rendered repeatedly in 2D, terrain-aware 3D, or MP4
without rerunning stochastic dynamics.
