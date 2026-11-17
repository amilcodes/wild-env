#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
job_root=${1:?usage: run_operations_benchmark.sh RUN_DIRECTORY SUITE_YAML [AGENTS_YAML]}
suite=${2:?suite required}
agents=${3:-}
mkdir -p "$(dirname -- "$job_root")"
mkdir "$job_root"
job_root=$(cd -- "$job_root" && pwd)
finish() { local status=$?; printf '%s\n' "$status" > "$job_root/exit_code"; }
trap finish EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
python_bin=${WILD_ENV_PYTHON:-$repo_root/.venv/bin/python}
"$python_bin" tools/validate_operations_reference.py "$suite" \
  --out "$job_root/reference-verification.json" > "$job_root/reference-log.txt" 2>&1
args=(--suite "$suite" --out "$job_root/benchmark" --workers "${WILD_ENV_WORKERS:-4}")
if [[ -n "$agents" ]]; then args+=(--agents "$agents"); fi
status=0
"$python_bin" -m aeolus.operations_cli run "${args[@]}" > "$job_root/progress.txt" 2>&1 || status=$?
if [[ $status != 0 && $status != 2 ]]; then exit "$status"; fi
"$python_bin" -m aeolus.operations_cli analyze "$job_root/benchmark" > "$job_root/analysis.json" 2>&1
"$python_bin" tools/audit_operations_results.py "$job_root" > "$job_root/audit-log.txt" 2>&1
"$python_bin" tools/validate_operations_states.py "$job_root" > "$job_root/state-verification-log.txt" 2>&1
printf 'Finished: %s\n' "$job_root"
exit "$status"
