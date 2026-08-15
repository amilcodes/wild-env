# wild-env

wild-env is a research environment for constrained multi-agent allocation of
heterogeneous aerial and ground resources during wildfire initial attack. It includes a
headless operational-equation fire/resource simulator, delayed belief model,
task-based PettingZoo and RLlib interfaces, recurrent MAPPO baselines, exact
and heuristic comparators, public incident import, historical evaluation, and
deterministic scientific replay.

The installed Python distribution and import namespace remain `aeolus-ia` and
`aeolus` for compatibility with existing experiment manifests. Repository
provenance and the policy for versioned research artifacts are recorded in
[`docs/development_record.md`](docs/development_record.md) and
[`docs/repository_artifact_policy.md`](docs/repository_artifact_policy.md).

The fire core uses a reproducible Pyretechnics-derived Anderson/Scott–Burgan
surface-behavior table, vector wind/slope coupling, crown transition and spread,
dynamic dead-fuel moisture, statistical ember transport and adaptive raster
front propagation with a WENO5/RK3 signed level set. NumPy and
accelerator-resident PyTorch paths share the local-behavior and front
equations. Spatially varying weather and posterior perimeter ensembles are
supported. Its equations and limits are specified in
[`docs/fire_behavior.md`](docs/fire_behavior.md).

Version 0.5 added volume-conserving water/retardant coverage, explicit
crew/dozer line production and breach, reload queues and operational gates,
two-perimeter arrival-history initialization, station-conditioned wind/fuel
moisture analysis, and advancing-front localization. Methods and frozen paired
results are in
[`docs/suppression_operations_research.md`](docs/suppression_operations_research.md).

Version 0.6 adds explicit airports/helibases/dip sites/scoopable water and
retardant bases, site-specific queues and finite stock, payload/endurance
recovery routing, coordinated aerial line tasks, a device-resident batched
operations environment, and an entity-attention capacity-aware MAPPO policy.
The implementation, compute topology, policy research queue, measurements, and
remaining gaps are in
[`docs/rl_compute_research.md`](docs/rl_compute_research.md).

Version 0.7 adds replay schema 2 and a native Qt/VTK inspection application.
Recorded episodes now retain gridded weather, vehicle mission/task/endurance
state, service-node stock, spatial reference, scenario identity and civil-time
origin. The desktop viewer provides synchronized time, vehicle, event, layer,
camera, 2D map and terrain-3D controls. Deterministic still/video rendering and
ParaView time-series export use the same replay contract. See
[`docs/viewer.md`](docs/viewer.md).

The current non-compute-bound P1 pass adds acquisition-window observation
likelihoods, historical fuel chronology screening, fire-regime validity
classification, canonical density-altitude/wind/terrain/airspace feasibility,
frozen-partition audits, and paired case-cluster policy effects. Its control
study identified a post-incident fuel-vintage failure in all six prepared
incidents. The replacement corpus now passes six of six version-level
disturbance-cutoff gates and preserves exact source rasters and checksums; five
incidents still need a closer archived LANDFIRE vintage. The pass also adds a
traceable nine-profile crewed/UAS aviation catalog and reference operations
scenario, while explicitly retaining the need for authorized flight-manual
performance surfaces. The 36-incident chronological partition is frozen at 22
training, seven development, and seven test incidents. Its complete local
benchmark is resumable and source-fingerprinted; live state is recorded under
`results/frontier_fire/historical_validation_frozen_36_local/`. The planning
control still exposes a cost/loss trade that favors near-no-action behavior.
See
[`docs/noncompute_p1_research.md`](docs/noncompute_p1_research.md) for methods
and results and
[`docs/noncompute_p1_remaining_work.md`](docs/noncompute_p1_remaining_work.md)
for the ordered closure program.

Vehicle evidence and the historical-fuel repair are documented in
[`docs/aviation_vehicle_closure.md`](docs/aviation_vehicle_closure.md),
[`docs/aviation_evidence_acquisition.md`](docs/aviation_evidence_acquisition.md),
[`docs/aviation_records_request.md`](docs/aviation_records_request.md), and
[`docs/historical_fuel_p0.md`](docs/historical_fuel_p0.md).

The fixed-parameter 24-interval screening ablation improves mean perimeter IoU
from 0.325 to 0.390 and reduces mean boundary distance from 1,913 m to 1,507 m
when the 2025 landscape is replaced. This is an adaptive-Huygens sensitivity
result; the primary WENO5 ensemble benchmark remains a cluster run.

## Install and verify

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[geo,viewer,dev]'
ruff check src tests
pytest
```

Install the PyTorch wheel matching a cluster's CUDA driver before installing
the project when the default package index is unsuitable.

## Import a timestamped public incident

This command queries NASA FEDS perimeters, USGS 3DEP elevation and LANDFIRE
2025 fuel/canopy services and produces a validated STAC-based IncidentBundle.

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

Source service responses and transformation provenance are retained below the
bundle. Public services can change; archive a bundle used in a result.
`aeolus-incident assemble` builds the same contract from locally normalized
GeoJSON perimeters, an aligned simulator NPZ, and optional GeoTIFF/CF-NetCDF
assets.

## Historical evaluation

```bash
aeolus-historical \
  --incident runs/incidents/feds-61854 \
  --mode hindcast \
  --policy no_aerial \
  --start-index 0 \
  --target-index 1 \
  --out runs/hindcast.json

aeolus-historical \
  --incident runs/incidents/feds-61854 \
  --mode counterfactual \
  --policies no_aerial,anchor_flank,joint_assignment \
  --seeds 32 \
  --start-index 0 \
  --target-index 1 \
  --out runs/counterfactual.json
```

Hindcast, shadow replay and counterfactual branches have deliberately different
causal interpretations; they are described in
[`docs/research_position.md`](docs/research_position.md).

### NIROPS held-out benchmark

The multi-incident study imports analyst-interpreted airborne-infrared
progressions, builds aligned terrain/fuel/weather bundles, calibrates on one
earlier transition per fire, and scores four later transitions per fire:

```bash
aeolus-study prepare \
  --manifest configs/historical_validation.yaml \
  --source-shapefile /path/to/NIROPS_2020_2024_R1_R6.shp \
  --out outputs/historical-validation/incidents

aeolus-study run \
  --manifest configs/historical_validation.yaml \
  --prepared-root outputs/historical-validation/incidents \
  --out outputs/historical-validation/results
```

The frozen v0.4 six-fire result has 24 held-out forecasts. The posterior
ensemble reaches cumulative IoU 0.862 and 167 m mean boundary displacement,
close to the no-growth persistence baseline at 0.873 and 156 m. On the
forecast-independent active-growth domain it improves balanced Brier score by
6.0% relative to persistence, while thresholded advancing-front F1 remains
only 0.096. The result supports uncertainty-aware research use and does not
support an operational spread-accuracy claim. The full protocol, incident
results, and suppression-data audit are in
[`docs/historical_validation.md`](docs/historical_validation.md). Machine-readable
v0.4 outputs are under
[`results/frontier_fire`](results/frontier_fire); the v0.3 archive remains
under [`results/historical_validation`](results/historical_validation).

The current metric-grid incident-holdout pilot corrects a Web Mercator physical
scale defect, freezes whole incidents into chronological train/development/test
partitions, and compares retrospective forcing with 24 pre-issue archived HRRR
forecast windows. It does not beat persistence on the two unseen test
incidents: operational front selection obtains advancing-front F1 0.112 while
reducing cumulative IoU from 0.8465 to 0.8353. See
[`docs/heldout_historical_skill_v6.md`](docs/heldout_historical_skill_v6.md) for
the exact protocol, results, artifacts, and remaining accuracy work.
The incident-wind, spatial/live-moisture, observation-uncertainty,
suppression-confounding, and 36-incident benchmark work is recorded in
[`docs/historical_fidelity_v4.md`](docs/historical_fidelity_v4.md).
A paired Electra ablation improves calibrated-ensemble cumulative IoU from
0.711 to 0.847 and mean boundary distance from 560 m to 244 m, while
observed-growth advancing-front F1 falls from 0.067 to 0.024. Persistence
remains stronger on cumulative extent. The mixed result is diagnostic and
does not establish generalization.

GOFER v0.2 hourly satellite progression can be imported without treating the
retrospective product as an operational observation:

```bash
python tools/import_gofer_progression.py /path/to/GOFER.zip \
  --out outputs/gofer/tamarack-2021 \
  --fire Tamarack \
  --year 2021 \
  --variant combined
```

The importer preserves one-hour acquisition windows, active-line state,
population-level spatial-error evidence, source checksums, and retrospective
availability. `tools/rasterize_progression_observations.py` aligns the vectors
to a metric IncidentBundle grid and writes a compressed, checksum-locked mask
and active-line cube. The current background execution and next accuracy
experiments are specified in
[`docs/background_accuracy_execution.md`](docs/background_accuracy_execution.md).

### Suppression and coupled-state benchmark

```bash
python tools/run_suppression_operations_study.py \
  --manifest configs/historical_validation.yaml \
  --incidents outputs/historical-validation/incidents \
  --baseline results/frontier_fire/historical_validation/historical_validation_results.json \
  --out results/frontier_operations_final \
  --workers 8 \
  --seeds 8
```

The frozen v0.5 study contains 24 two-perimeter held-out hindcasts and 72
matched-seed suppression trials. Two-perimeter initialization increases mean
perimeter IoU from 0.807 to 0.832 and reduces mean boundary error from 299 m to
224 m, while exact new-growth IoU remains lower. In controlled synthetic
trials, integrated air/ground operations reduce mean value-weighted loss by
28.0% and burned fraction by 43.1% relative to uncontrolled runs. These are
simulator mechanism results, not field-effect estimates.

## Train and compare

```bash
aeolus-train --config configs/smoke.yaml
aeolus-eval \
  --config configs/smoke.yaml \
  --checkpoint runs/smoke_v4/checkpoint.pt \
  --episodes 32 \
  --out runs/evaluation.json
```

The larger DDP manifest is `configs/cluster_mappo.yaml`. Slurm/Apptainer launch
contracts are in `deploy/slurm` and [`docs/cluster.md`](docs/cluster.md).
`configs/local_research_mappo.yaml` demonstrates an expert-initialized MAPPO
run; the expert phase and PPO updates are separately reported in stdout and the
full choice is checkpointed in the configuration.

Large-batch sortie and service-routing pretraining uses the accelerator-resident
operations backend:

```bash
AEOLUS_CONFIG=configs/cluster_tensor_operations.yaml \
  sbatch deploy/slurm/train.sbatch
```

The example capabilities in that manifest are research assumptions and must be
replaced with reviewed aircraft performance and evaluated service-site data.
The exact mechanism/throughput harness can be run independently:

```bash
python tools/run_rl_operations_study.py \
  --config configs/cluster_tensor_operations.yaml \
  --device cuda \
  --batch-size 2048 \
  --segments 48 \
  --out results/rl_operations/gpu-study.json
```

Fire-coupled large-batch training uses the tensor incident backend. It keeps a
coarse fire, delayed belief, dynamic front tasks, treatment fields, sorties,
service sites, reward, and policy rollout on the accelerator:

```bash
python tools/run_tensor_incident_study.py \
  --config configs/cluster_tensor_incident.yaml \
  --device cuda \
  --compile \
  --batch-size 2048 \
  --grid-size 64 \
  --segments 48 \
  --out results/tensor_incident/gpu-study.json

AEOLUS_CONFIG=configs/cluster_tensor_incident.yaml \
  sbatch deploy/slurm/train.sbatch
```

The surrogate contract, measured local result, limitations, and calibration
boundary are in
[`docs/tensor_incident_environment.md`](docs/tensor_incident_environment.md).
The ordered training and transfer gates are in
[`docs/rl_training_execution_plan.md`](docs/rl_training_execution_plan.md).

Headless simulator throughput is measured independently of training:

```bash
aeolus-benchmark \
  --config configs/smoke.yaml \
  --policy no_aerial \
  --envs 8 \
  --decisions 512 \
  --out runs/benchmark.json
```

The batched fire core has its own point-query, validation and accelerator
benchmark interface:

```bash
aeolus-fire point --fuel-model 145 --moisture 0.07 --wind 6
aeolus-fire benchmark --device cuda --batch 256 --height 128 --width 128
aeolus-fire validate --device cuda --output runs/fire-validation
aeolus-fire verify-front --output runs/front-verification
```

## Record, inspect, and render

Record a trained or comparison policy without opening a display:

```bash
aeolus-replay \
  --incident runs/incidents/feds-61854 \
  --policy joint_assignment \
  --horizon-min 180 \
  --out runs/replays/feds-61854-joint \
  --frame-2d runs/feds-61854-2d.png \
  --frame-3d runs/feds-61854-3d.png \
  --video runs/feds-61854.mp4
```

Open the recorded episode in the native desktop application:

```bash
aeolus-view \
  --replay runs/replays/feds-61854-joint \
  --config configs/viewer/operational.yaml
```

For a fully local reference case:

```bash
aeolus-replay \
  --config configs/replay_reference.yaml \
  --policy joint_assignment \
  --out runs/replays/reference

aeolus-view \
  --replay runs/replays/reference \
  --config configs/viewer/suppression.yaml
```

The replay directory stores chunked state arrays, typed events, the complete
scenario, spatial/time identity, policy name, episode result and checkpoint
digest. Rendering remains downstream of training and can be repeated without
changing the episode. Viewer/scenario configuration and accuracy limits are in
[`docs/viewer.md`](docs/viewer.md) and
[`docs/scenario_configuration.md`](docs/scenario_configuration.md).

`configs/frontier_suppression.yaml` is the complete explicit air/crew/dozer
operations manifest.

## Package map

- `aeolus.core`: truth/belief, fire, mission and task mechanics.
- `aeolus.data`: scenario, incident, weather/station analysis and public-service import.
- `aeolus.envs`: PettingZoo/RLlib interfaces and tensor operations batches.
- `aeolus.training`: recurrent pointer/entity-attention MAPPO, device rollout and DDP.
- `aeolus.policies`: no-action, heuristic and exact assignment comparators.
- `aeolus.evaluation`: paired seeds and historical evaluation modes.
- `aeolus.replay`: Zarr/Parquet recording and 2D/3D/MP4 rendering.
- `aeolus.viewer`: read-only replay model, native Qt/VTK application, imagery
  alignment, view manifests, and ParaView export.

The exact information boundary and model limitations are documented in
[`docs/architecture.md`](docs/architecture.md); the experiment protocol is in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md), and the measured
C++/CUDA port boundary is in [`docs/native_kernel.md`](docs/native_kernel.md).

## License

Apache-2.0. The simulator and policy outputs are research artifacts and are not
certified for operational fire-management decisions.
