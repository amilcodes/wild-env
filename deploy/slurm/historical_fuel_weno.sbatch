#!/usr/bin/env bash
# Paired primary-solver evaluation. Array 0 uses the original fuel corpus;
# array 1 uses the time-admissible corpus.
#SBATCH --job-name=aeolus-fuel-weno
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${AEOLUS_IMAGE:?Set AEOLUS_IMAGE to an Apptainer SIF path}"
DATA_ROOT="${AEOLUS_HISTORICAL_DATA_ROOT:?Set the host directory containing both prepared corpora}"
MANIFEST="${AEOLUS_HISTORICAL_MANIFEST:-configs/historical_validation.yaml}"
BEFORE_SUBPATH="${AEOLUS_BEFORE_SUBPATH:-historical-validation-v4-electra}"
AFTER_SUBPATH="${AEOLUS_AFTER_SUBPATH:-historical-validation-v5-time-admissible}"
OUTPUT_ROOT="${AEOLUS_OUTPUT_ROOT:-results/aviation_fuel_p0/weno_primary}"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CORPUS="/data/$BEFORE_SUBPATH"
    LABEL="before"
    ;;
  1)
    CORPUS="/data/$AFTER_SUBPATH"
    LABEL="after"
    ;;
  *)
    echo "Unexpected array index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

srun apptainer exec \
  --bind "$PROJECT_DIR:/workspace,$DATA_ROOT:/data:ro" \
  "$IMAGE" \
  bash -lc "cd /workspace && aeolus-study run \
    --manifest $MANIFEST \
    --prepared-root $CORPUS \
    --workers $SLURM_CPUS_PER_TASK \
    --out $OUTPUT_ROOT/$LABEL"
