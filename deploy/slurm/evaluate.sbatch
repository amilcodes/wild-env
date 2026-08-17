#!/usr/bin/env bash
#SBATCH --job-name=aeolus-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${AEOLUS_IMAGE:?Set AEOLUS_IMAGE}"
CONFIG="${AEOLUS_CONFIG:-configs/cluster_mappo.yaml}"
CHECKPOINT="${AEOLUS_CHECKPOINT:?Set AEOLUS_CHECKPOINT to a checkpoint path under the bound project directory}"
EPISODES="${AEOLUS_EVAL_EPISODES:-512}"

apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$IMAGE" \
  bash -lc "cd /workspace && aeolus-eval --config $CONFIG --checkpoint $CHECKPOINT --episodes $EPISODES --out runs/evaluation-\$SLURM_JOB_ID.json"
