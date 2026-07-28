# Aeolus-IA

Aeolus-IA is a research environment for constrained multi-agent allocation of
heterogeneous aerial resources during wildfire initial attack. It includes a
headless stochastic fire/resource simulator, delayed belief model, task-based
PettingZoo and RLlib interfaces, a recurrent MAPPO baseline, exact and heuristic
comparators, public incident import, historical evaluation, and deterministic
2D/3D replay.

The fast fire kernel is intentionally inspectable and has not been calibrated
as an operational fire-behavior model. See
[`docs/research_position.md`](docs/research_position.md) for the fidelity
assessment and required validation evidence.

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
