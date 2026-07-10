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
   counterfactual modes, deterministic Zarr/Parquet replay bundles, and
   read-only native/scientific inspection paths.

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
observe, water, retardant, line-reinforcement, crew/dozer line, and hold
alternatives. Each task carries a target, orientation where applicable,
expected value, uncertainty, ground dependency, and capacity.

Independent actor actions may compete for one task. The simulator resolves
conflicts in a seeded random auction and records attempts, acceptances, and
blocked actions. The `joint_assignment` comparator instead solves the finite
maximum-weight resource/task assignment exactly over the same task graph. It is
the minimum serious non-learning comparison for policy results.

## Fire and intervention kernel

The version 0.5 kernel resolves per-cell Anderson/Scott–Burgan surface behavior
from a packaged Pyretechnics reference table. Wind and slope factors combine as
vectors, directional spread follows an ellipse, crown initiation/spread uses
Van Wagner and Cruz relations, dead-fuel moisture responds to weather and
precipitation, and spotting uses stochastic downwind/crosswind ember transport.
The fireline is the zero contour of a signed-distance field advanced with
WENO5 spatial derivatives and SSP-RK3 time integration. Narrow-band evaluation,
connected support around the represented fire, periodic reinitialization and
sub-step arrival-time interpolation are shared between the NumPy and PyTorch
paths. The earlier adaptive Huygens raster solver remains an ablation.

The canonical NumPy path participates in the complete incident/resource
simulator. A separate `TensorFireKernel` keeps batches resident in PyTorch on
CUDA, ROCm, MPS or CPU. It shares the behavior table, front equation and
forcing adjustments, and parity tests compare local outputs field by field.
Replay records the level set, fuel model, canopy, moisture, fire type, spread
rate, flame length and fuel consumption.

Water and retardant are stored as volume-conserving GPC coverage and effective
treatment. Crew/dozer line accrues over multiple minutes with stochastic
production, explicit width, reinforcement, and unengaged/held/breached states.
Held line is a local zero-normal-spread condition with open flanks; it is not an
unconditional perimeter barrier. Intervention response remains an empirical
research model.

The full numerical and physical contract is in
[`fire_behavior.md`](fire_behavior.md). Atmosphere coupling, plume dynamics,
smoke and combustion-resolving flow remain outside the model.

## Resource dynamics

Dispatch latency, cruise time, multi-minute work, mission execution,
site-specific service queues, finite suppressant stock, payload, sortie
endurance/reserve, and withdrawal are represented explicitly. Airports,
helibases, retardant bases, dip sites, scoopable water, and temporary tanks are
named resources with compatible service modes. Resource positions are
interpolated along each mission leg for replay and state inspection. The policy
selects oriented attack segments and service destinations. Flight control,
deconfliction, maintenance and full crew duty/safety rules remain future model
components.

The coupled-state initializer reconstructs arrival time from two perimeter
observations, derives burn age/fuel/heat memory/recent velocity, and localizes
that velocity to a decaying band around the forecast-start front. Sequential
perimeter correction uses a separately testable signed-distance localization
operator.

## Learning implementation

The included policy set contains recurrent task-pointer MAPPO and an
entity-attention MAPPO model:

- task/resource encoders, set attention, and masked pointer logits;
- a GRU state per resource;
- capacity-conditioned joint sampling for assignment conflicts;
- a centralized value function with a privileged training-only input;
- clipped PPO objective and value update, GAE, entropy regularization,
  gradient clipping, AMP, checkpointed optimizer/config/RNG state;
- optional masked behavior-cloning warm start from an explicit comparator
  policy before on-policy PPO;
- data-parallel training with `torchrun` and DDP.

The entity-attention actor assumes shared fleet telemetry; it does not infer a
radio protocol. The canonical environment remains NumPy-resident.
`tensor_operations` is the static sortie/logistics pretrainer.
`tensor_incident` adds a coarse probabilistic fire, delayed actor belief,
dynamic belief-front localization, volume-conserving water/retardant fields,
per-world latent physics, and outcome reward to the same device-resident
rollout. Its complete transition passes a strict full-graph compilation test.
The surrogate is calibrated and selected against canonical traces; it does not
replace the WENO/behavior-table model for historical evaluation. The detailed
contract is in
[`tensor_incident_environment.md`](tensor_incident_environment.md).

## Replay contract

Every replay is a directory containing:

- chunked Zarr arrays for fire behavior, truth, belief, treatment, terrain and
  resource state;
- Parquet events with minute, type and JSON payload;
- JSON metadata containing the exact scenario, identity/time/spatial
  reference, policy, episode summary, resource schema and checkpoint SHA-256.

Replay schema 2 also records gridded weather and detailed resource mission,
endurance, task/target, service-site and site-stock state. Rendering is
downstream of replay. The training process remains headless and a stored
episode can be inspected repeatedly in a native Qt/VTK application, rendered
to 2D/terrain-aware 3D/MP4, or exported to a ParaView time series without
rerunning stochastic dynamics.

Belief replay includes burn probability, arrival-time mean and standard
deviation, intensity mean and standard deviation, and observation time.
Historical posterior ensembles additionally store forecast probability and
conditional arrival-time moments in their evaluation artifact.
