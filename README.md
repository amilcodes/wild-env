# Aeolus-IA

Aeolus-IA is a research environment for constrained multi-agent allocation of
heterogeneous aerial resources during wildfire initial attack. It includes a
headless operational-equation fire/resource simulator, delayed belief model, task-based
PettingZoo and RLlib interfaces, a recurrent MAPPO baseline, exact and heuristic
comparators, public incident import, historical evaluation, and deterministic
2D/3D replay.

The fire core uses a reproducible Pyretechnics-derived Anderson/Scott–Burgan
surface-behavior table, vector wind/slope coupling, crown transition and spread,
dynamic dead-fuel moisture, statistical ember transport and adaptive raster
front propagation. NumPy and accelerator-resident PyTorch paths share the
local-behavior equations. Its equations and limits are specified in
[`docs/fire_behavior.md`](docs/fire_behavior.md).

## Install and verify

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[geo,render,dev]'
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

The frozen six-fire result has 24 held-out forecasts. Calibrated physics
reaches cumulative IoU 0.611 and active-growth one-cell-tolerance F1 0.151;
the no-growth persistence baseline reaches cumulative IoU 0.873. The result
does not support a historical spread-accuracy claim. The full protocol,
incident results, and suppression-data audit are in
[`docs/historical_validation.md`](docs/historical_validation.md). Machine-readable
outputs are under [`results/historical_validation`](results/historical_validation).

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
```

## Record and render

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

The replay directory stores chunked state arrays, events, scenario metadata and
the model checkpoint digest. Rendering is downstream of training and can be
repeated without changing the episode.

## Package map

- `aeolus.core`: truth/belief, fire, mission and task mechanics.
- `aeolus.data`: scenario, incident, weather and public-service import.
- `aeolus.envs`: PettingZoo and optional RLlib interfaces.
- `aeolus.training`: shared recurrent task-pointer MAPPO and DDP.
- `aeolus.policies`: no-action, heuristic and exact assignment comparators.
- `aeolus.evaluation`: paired seeds and historical evaluation modes.
- `aeolus.replay`: Zarr/Parquet recording and 2D/3D/MP4 rendering.

The exact information boundary and model limitations are documented in
[`docs/architecture.md`](docs/architecture.md); the experiment protocol is in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md), and the measured
C++/CUDA port boundary is in [`docs/native_kernel.md`](docs/native_kernel.md).

## License

Apache-2.0. The simulator and policy outputs are research artifacts and are not
certified for operational fire-management decisions.
