# Scenario configuration

An experiment manifest contains two records with different lifetimes:

- `scenario` defines the world, forcing, resources, and dynamics that determine
  an episode;
- `training` defines rollout, optimization, checkpointing, and device choices.

Viewer choices live in a separate manifest under `configs/viewer`. Changing a
camera, map layer, or export resolution must not change an episode.

## Validate a manifest

Configuration is loaded into frozen dataclasses. Unknown fields and invalid
values fail before simulator construction.

```bash
python - <<'PY'
from aeolus.config import load_config

config = load_config("configs/replay_reference.yaml")
print(config.scenario.title)
print(len(config.scenario.resources), len(config.scenario.service_sites))
PY
```

`configs/replay_reference.yaml` is a compact, fully specified example for
recording and viewer tests. It is a synthetic local-metric domain. It is not a
field scenario and its service nodes are explicitly marked unverified.

## Identity, place, and time

```yaml
scenario:
  scenario_id: foothill-operations-reference
  title: Foothill Operations Reference
  location_name: Synthetic foothill domain; local metric grid
  time_origin: "2026-08-17T14:00:00-07:00"
```

`scenario_id` is stable and machine-readable. `title` is a human-readable
result label. `location_name` may name a real incident, study area, or clearly
marked synthetic domain. `time_origin` is optional ISO 8601 with an
explicit offset. Replay time is always retained as integer elapsed minutes;
the origin adds a civil-time display without changing simulation integration.

An imported `IncidentBundle` takes precedence for incident ID, title, bounding
box, and timestamp metadata.

## Domain and forcing

```yaml
scenario:
  seed: 260817
  width: 112
  height: 96
  cell_size_m: 45.0
  horizon_min: 150
  decision_interval_min: 3
  observation_delay_min: 3
  wind_speed_m_s: 7.2
  wind_direction_deg: 38.0
  wind_variability: 0.24
  air_temperature_c: 32.0
  relative_humidity_pct: 19.0
  precipitation_rate_mm_h: 0.0
  landscape_bundle: null
  weather_forcing: null
```

`width × height × cell_size_m` fixes the computational extent. Dynamics
advance in one-minute increments; the joint policy acts every
`decision_interval_min`. `observation_delay_min` applies to the policy belief,
not hidden truth.

Scalar weather values define homogeneous fallback forcing. A CF-NetCDF
`weather_forcing` may replace them with time-varying incident-wide or
`(time, y, x)` fields. `wind_direction_deg` is meteorological wind-from
direction. Landscape and forcing contracts are in
[`data_contract.md`](data_contract.md).

## Fire and suppression

The `fuel`, `fire`, and `suppression` sections expose physical/numerical
assumptions in SI units. Omitting a field uses a versioned dataclass default;
reported experiments should still archive the fully materialized scenario
stored in replay metadata.

Important fire controls include front solver, CFL/substeps, surface and crown
adjustments, fuel moistures, and spotting transport. Their equations and
calibration boundary are documented in [`fire_behavior.md`](fire_behavior.md).

Suppression controls include water/retardant decay, rainfall wash, coverage
response, line production variability and breach, dispatch payload/reserve
gates, reload capacity, and aviation wind limits. These are empirical research
parameters. Perimeter-only observations do not identify them; use treatment
chronology and holding outcomes for calibration.

## Service nodes

```yaml
service_sites:
  - site_id: reservoir_alpha
    kind: dip_site
    x: 74
    y: 66
    services: [water]
    service_mode: hover_fill
    bays: 2
    approach_capacity: 2
    refill_rate_l_min: 5200.0
    fixed_turnaround_min: 0.8
    available_volume_l: 280000.0
    open_minute: 0
    close_minute: 1440
    max_operating_wind_m_s: 20.0
    minimum_depth_m: 2.0
    minimum_length_m: 160.0
    manually_verified: false
```

Supported kinds are `airport`, `retardant_base`, `helibase`, `dip_site`,
`scoopable_water`, and `temporary_tank`. `service_mode` is `land`,
`hover_fill`, or `scoop`. Capacity is constrained separately at the service
bays and approach. Finite `available_volume_l` is decremented by delivered
payload. Operating windows, wind limit, geometry limits, and manual
verification remain explicit state.

Coordinates are grid-cell indices. For a real scenario, derive them from the
reviewed incident CRS/transform and retain the source survey or facility
record. `manually_verified: true` is a data assertion, not a display option.

## Resources

Every resource supplies a stable ID, kind, performance envelope, mission
timing, and service compatibility:

```yaml
resources:
  - resource_id: water_uav_00
    kind: water
    cruise_speed_m_s: 48.0
    payload_l: 1400.0
    dispatch_latency_min: 1
    reload_min: 3
    endurance_min: 125
    reserve_endurance_min: 18.0
    home_site_id: west_airport
    service_modes: [land, hover_fill]
    max_operating_wind_m_s: 16.0
    performance_surface_path: configs/aviation/vehicle_configuration.json
    cruise_altitude_agl_m: 150.0
    minimum_terrain_clearance_m: 75.0
    maximum_operating_altitude_m_msl: 4500.0
    maximum_crosswind_m_s: 12.0
    minimum_service_depth_m: 2.0
    minimum_service_length_m: 160.0
    water_radius_m: 55.0
    minimum_drop_length_m: 180.0
    maximum_drop_length_m: 700.0
```

Kinds are `water`, `retardant`, `sensor`, and `crew`. A home site must exist
and offer a compatible mode and service. The simulator rejects dispatch below
reserve or payload gates, routes recovery to compatible nodes, resolves
approach/service queues, and records blocked actions. Crew/dozer resources use
line width, length, production rate, and direct-intensity limits.

Aircraft values in example manifests are research assumptions. Replace them
with reviewed performance tables, loading constraints, duty rules, site
procedures, and weather minima before interpreting operational feasibility.

When `performance_surface_path` is present, the canonical action mask
interpolates true airspeed, endurance multiplier, and maximum payload over
density altitude and payload fraction. It uses terrain plus declared AGL
cruise altitude, vector wind groundspeed/crosswind, outbound/recovery reserve,
and vehicle/site geometry. Outside-surface conditions are infeasible.
Pressure altitude currently uses terrain elevation when no pressure/altimeter
field is available. Climb/descent phases, maintenance, weight and balance,
obstacles, and detailed flight dynamics remain unresolved.

## Airspace volumes

```yaml
airspace_volumes:
  - volume_id: branch-drop-lane
    kind: reserved
    polygon_xy:
      - [44.0, 30.0]
      - [62.0, 30.0]
      - [62.0, 44.0]
      - [44.0, 44.0]
    lower_altitude_m_msl: 800.0
    upper_altitude_m_msl: 1800.0
    start_minute: 30
    end_minute: 95
    allowed_resource_ids: [lead_01, tanker_12]
```

The canonical environment rejects a straight route when its geometry, planned
altitude, and estimated time overlap a volume for which the resource is not
allowed. This is a hard feasibility screen. It does not compute a detour,
separate flight phases, or replace incident airspace authorization and
separation procedures.

## Training

```yaml
training:
  seed: 260817
  device: cuda
  environment_backend: canonical
  policy_architecture: entity_attention
  num_envs: 256
  rollout_steps: 128
  updates: 2000
  minibatch_size: 4096
  hidden_dim: 256
  attention_heads: 8
  attention_layers: 3
  checkpoint_dir: runs/replay-reference
  use_amp: true
```

`canonical` runs the complete NumPy fire/belief/operations environment.
`tensor_operations` keeps the logistics/task-allocation batch resident in
PyTorch for high-throughput pretraining but does not yet couple the full tensor
fire and delayed-belief pipeline. That distinction is material and is retained
in checkpoints. Cluster topology and measured boundaries are documented in
[`cluster.md`](cluster.md) and [`rl_compute_research.md`](rl_compute_research.md).

## Reproducible result layout

Keep configuration and outputs immutable below a run identifier:

```text
runs/<run-id>/
├── scenario.yaml
├── viewer.yaml
├── checkpoint.pt
├── replay/
│   ├── metadata.json
│   ├── states.zarr/
│   └── events.parquet
├── metrics.json
└── figures/
```

The checkpoint digest in replay metadata binds a rendered episode to policy
weights. Also retain the Git revision, container digest, seed manifest,
IncidentBundle, and source retrieval timestamps. Do not use a chosen render as
the selection criterion for a checkpoint or experimental conclusion.
