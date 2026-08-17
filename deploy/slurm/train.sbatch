#!/usr/bin/env bash
# Aeolus DDP training launcher. Adapt partition, account, time, image path,
# and module names to the target site; do not put credentials in this file.
#SBATCH --job-name=aeolus-mappo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${AEOLUS_IMAGE:?Set AEOLUS_IMAGE to an OCI image or Apptainer SIF path}"
CONFIG="${AEOLUS_CONFIG:-configs/cluster_mappo.yaml}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1

# Expected image behavior: `apptainer exec --nv IMAGE ...` exposes the same
# CUDA/NCCL software stack selected when the image was built.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$IMAGE" \
  bash -lc "cd /workspace && torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=\$SLURM_PROCID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m aeolus.training.train --config $CONFIG"
