# Cluster execution contract

## Training topology

Each DDP rank owns independent environment instances and one GPU-resident model
replica. Rollout simulation remains CPU work in this reference build; task
encoder, actor, critic, GAE tensors, and PPO optimization execute on the rank's
GPU. DDP all-reduces model gradients. This is appropriate while the policy is
the dominant accelerator workload. The native-kernel boundary is intentionally
kept separate so scenario semantics do not change when rollout becomes the
bottleneck.

The recommended initial allocation is one or two nodes, four GPUs per node,
and 8–16 CPU cores per GPU. Profile rollout steps/s, GPU utilization, and
all-reduce time before optimizing. If simulation dominates, port the
cell-update/treatment operators to a batched C++/CUDA extension and retain the
Python environment as the parity oracle.

## Run sequence

```bash
cd aeolus_py
docker build -f deploy/Dockerfile -t registry.example/aeolus-ia:0.2 .
apptainer build aeolus-ia.sif docker-daemon://registry.example/aeolus-ia:0.2
export AEOLUS_IMAGE=$PWD/aeolus-ia.sif
export PROJECT_DIR=$PWD
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

No cloud account, cluster credentials, or operational incident data are stored
in this repository.
