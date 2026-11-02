#!/usr/bin/env bash
# Run from a dedicated EC2 checkout; launch under nice as documented.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
job_root=${1:?usage: run_agent_benchmark.sh RUN_DIRECTORY [AGENTS_YAML] [SUITE_YAML]}
agents_file=${2:-}
suite_file=${3:-configs/agents/frontier.yaml}
mkdir -p "$(dirname -- "$job_root")"
mkdir "$job_root"  # Refuse to overwrite an existing job.
job_root=$(cd -- "$job_root" && pwd)
finish() { local status=$?; printf '%s\n' "$status" > "$job_root/exit_code"; }
trap finish EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
python_bin=${WILD_ENV_PYTHON:-$repo_root/.venv/bin/python}
"$python_bin" -m ruff check src tests tools examples > "$job_root/lint.txt" 2>&1
"$python_bin" -m pytest > "$job_root/tests.txt" 2>&1
args=(--suite "$suite_file" --out "$job_root/benchmark")
args+=(--workers "${WILD_ENV_WORKERS:-1}")
if [[ -n "$agents_file" ]]; then args+=(--agents "$agents_file"); fi
benchmark_status=0
"$python_bin" -m aeolus.agent_cli run "${args[@]}" > "$job_root/console.json" 2> "$job_root/progress.txt" || benchmark_status=$?
if [[ $benchmark_status != 0 && $benchmark_status != 2 ]]; then exit "$benchmark_status"; fi
"$python_bin" - "$job_root/benchmark" <<'PY'
import json
import sys
from pathlib import Path
from aeolus.agents.benchmark import replay_trace
run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
assert summary["complete"]
checks = []
for record in summary["records"]:
    check = replay_trace(run / record["trace"])
    checks.append({"trace": record["trace"], "verified": check["verified"], "decisions": check["decisions"]})
(run / "replay-verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(f"Verified {len(checks)} episodes / {sum(check['decisions'] for check in checks)} decisions.")
PY
printf 'Finished: %s\n' "$job_root"
exit "$benchmark_status"
