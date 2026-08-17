# Cluster execution contract

## Training topology

Three training backends are retained.

`canonical` runs independent `AeolusSimulator` instances on the CPU and the
policy/optimizer on the rank GPU. It is the fire-coupled fine-tuning and parity
path.

`tensor_operations` keeps fleet, service-site, queue, payload, endurance,
attack-segment, action-mask, reward, rollout, model, and GAE state on the rank
GPU. It has no per-decision GPU/CPU copy. This is the large-batch operations
pretraining path. `configs/cluster_tensor_operations.yaml` is its reference
manifest.

`tensor_incident` is now the primary large-batch curriculum path. It adds a
64-by-64 probabilistic fire, delayed belief propagation/assimilation,
belief-derived front tasks, treatment fields, outcome reward, escape, and
per-world latent physical/response parameters. Its fixed-shape transition is
compiled with `fullgraph=True`; a graph break is a failed run. The reference
manifest is `configs/cluster_tensor_incident.yaml`.

Each DDP rank owns one independent device batch and one model replica. DDP
all-reduces gradients. The trainer compiles the inner policy before DDP
wrapping. First saturate a single rank by scaling `num_envs`; add ranks after
environment stepping and policy inference have useful occupancy.

The recommended initial allocation is one node with four GPUs, followed by two
nodes after measuring single-node scaling. Profile environment decisions/s,
agent actions/s, GPU utilization, peak memory, compiler graph breaks, PPO time,
and NCCL fraction. The fire-only `TensorFireKernel` keeps WENO5/RK3 batches on
the accelerator for high-fidelity component studies. It is not placed in the
mass-training loop because its numerical work and state size serve a different
fidelity tier.

`tools/estimate_tensor_incident_memory.py` reports a 5.16 GiB persistent-
storage lower bound per rank for the 2,048-world reference manifest. Of that,
4.57 GiB is rollout storage and 0.49 GiB is persistent environment/observation
state. The estimate excludes compiled graph pools, transition intermediates,
autograd activations, PPO temporaries, allocator fragmentation, CUDA context,
and NCCL buffers. It is a planning floor, not a GPU capacity claim.

Historical calibration and ensemble evaluation use independent CPU processes.
Set `parallel_workers` in the study manifest or `AEOLUS_EVAL_WORKERS` for
direct calls. The process pool is persistent across incidents and uses the
spawn start method so it remains safe when accelerator libraries are loaded in
the parent process.

Historical WENO5 fuel-vintage evaluation uses
`deploy/slurm/historical_fuel_weno.sbatch`. It is a two-element array over the
original and time-admissible corpora. Set `AEOLUS_HISTORICAL_DATA_ROOT` to the
host directory containing both corpus subdirectories; it is mounted read-only
at `/data`. The default requests 32 CPUs and 128 GB per task. The study CLI
accepts `--workers` so executor width matches the allocation while the frozen
scientific manifest remains unchanged.

## Run sequence

```bash
cd aeolus_py
docker build -f deploy/Dockerfile -t registry.example/aeolus-ia:0.6 .
apptainer build aeolus-ia.sif docker-daemon://registry.example/aeolus-ia:0.6
export AEOLUS_IMAGE=$PWD/aeolus-ia.sif
export AEOLUS_CONFIG=configs/cluster_tensor_incident.yaml
export PROJECT_DIR=$PWD
python tools/estimate_tensor_incident_memory.py
sbatch deploy/slurm/train.sbatch
```

The scheduler script starts one `torchrun` parent per node and one trainer per
GPU. PyTorch DDP uses NCCL on CUDA. Checkpoints, exact configuration, and JSONL
metrics are written below the configured run directory.

## Required site-specific values before a real cluster run

- Scheduler and launcher policy if the site is not Slurm/Apptainer.
- GPU model, driver, CUDA/NCCL version, and allowable container base image.
- Partition/account/QoS, maximum wall time, and filesystem path for checkpoints.
- Network fabric notes if nondefault NCCL variables are required.
- Data governance location for LANDFIRE/weather/perimeter data and any restricted
  operational-resource records.
- Reviewed aircraft performance tables and evaluated service-site data are
  needed only before vehicle-specific or operational claims. They are not a
  blocker for the current randomized research curriculum.

No owner input is needed for local implementation or CPU testing. Cluster
access first becomes necessary for accelerator profiling, batch selection,
DDP scaling, and full training. The exact sequence and acceptance gates are in
[`rl_training_execution_plan.md`](rl_training_execution_plan.md).

No cloud account, cluster credentials, operational incident data, or certified
aircraft capability data are stored in this repository.
