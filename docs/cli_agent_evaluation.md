# Evaluate subscription-backed agents on EC2

The CLI adapters use the authenticated RSG bridge clients already installed on
EC2. Simulation, scoring, tests and replay execute on EC2. The logged-in Mac only
relays CLI requests to model services. No provider credential is copied to EC2.
The transport is a loopback-only SSH reverse tunnel, authenticated with the
existing bridge token. Keep hostnames, key paths and tokens outside Git.

Use dedicated wild-env ports (for example 47810 and 47811), leaving RSG's 47710
and 47711 and its game/harness processes alone. Start the existing
`scripts/claude_bridge.py serve --port PORT` and
`scripts/codex_bridge.py serve --port PORT` on the logged-in host. Run their
supervisor in its own session with detached standard streams, a reconnecting
SSH tunnel, and `caffeinate -is` for the session lifetime. Do not free remote
ports by killing unrelated sshd processes. Stop only the task's own bridge
process group when the evaluation finishes.

The Codex server's `--codex` argument can point to the executable
`deploy/ec2/codex_provenance.py`. This small shim invokes the installed CLI and
adds a provenance event from the new session's own `turn_context`: selected
model, reasoning effort and read-only sandbox policy. It does not forward
conversation records, other sessions or credentials. This is a CLI configuration
attestation, not a provider-signed model snapshot. Without it, missing returned
model metadata stays unknown instead of being filled from the request.

On EC2, set the loopback addresses and put the bridge clients on PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
export RSG_CLAUDE_BRIDGE=127.0.0.1:47810
export RSG_CODEX_BRIDGE=127.0.0.1:47811
claude auth status
codex --version
```

A successful auth-status probe does not prove inference works: perform a real
request on development cases before evaluation. In this execution, Claude
reported logged in but its first request failed with an expired OAuth session;
that failure is recorded separately and is not a model score. The user completed
`claude auth login`; actual Opus requests then succeeded before evaluation began.

Copy `configs/agents/bridge.example.yaml` and select the actual available model
IDs. Models, effort, timeouts, scenario settings and seeds must be frozen before
reading evaluation outcomes. Each invocation starts a **fresh isolated session**;
only the current public observation, last dispatch receipt and bounded 2,000
character memory carry information across decisions. CLI system scaffolding
remains provider-specific, so results measure the named model plus its recorded
CLI harness, not a bare API model under identical system prompts.

The canonical prompt is extended only to request the exact dispatch-schema JSON
object. The harness applies it once. Claude has built-in tools and settings
sources disabled. Codex uses a read-only empty directory with browser, computer
use and apps disabled; any tool-execution event makes that decision invalid and
the episode ineligible. No shell/file tools are required to play the environment.
Neither adapter repairs, substitutes or retries a dispatch.

The CLI bridges do not expose an enforceable output-token cap. Configuration
therefore requires `max_output_tokens: null`, explicit reasoning effort, a wall
timeout and a call limit. Token counts include the CLI's scaffolding and, where
reported, cache reads/creation. Failed or missing usage remains visible. This is
not a fixed-token-budget comparison. Report call counts, usage, failures and
latency alongside scores; do not interpret subscription usage as free compute.

Use one worker for CPU-heavy baseline runs. Up to four independent episode
workers can overlap the waiting time of remote model calls, under `nice -n 15`
with BLAS/OpenMP threads capped at one. This does not create training workers or
change numerical simulation semantics. A regression test compares serial and
parallel episode metrics and verifies their replays.

```bash
# Run on EC2, first with the development suite, then the frozen evaluation suite.
WILD_ENV_WORKERS=4 nohup setsid nice -n 15 bash deploy/ec2/run_agent_benchmark.sh \
  runs/bridge-development configs/agents/bridge.example.yaml configs/agents/smoke.yaml \
  > runs/bridge-development-launch.log 2>&1 < /dev/null &
WILD_ENV_WORKERS=4 nohup setsid nice -n 15 bash deploy/ec2/run_agent_benchmark.sh \
  runs/bridge-evaluation configs/agents/bridge.example.yaml configs/agents/frontier.yaml \
  > runs/bridge-evaluation-launch.log 2>&1 < /dev/null &
```

Run those two commands sequentially, checking the development result before
starting evaluation. `summary.json` is atomically updated after every completed
episode. Parallel completion order does not change trace indices or final
record order. Partial summaries are explicitly incomplete. The runner replays
complete failed episodes too; exit status 2 retains ineligible results, not
success. No dashboard is produced.

## Environment validity checks

Protocol `wildfire-desk-v1.1` keeps the dispatch schema and simulation mechanics
but makes mission/service behavior explicit. It adds current mission, remaining
crew work, dispatch latency, and site service-rate/turnaround/approach telemetry.
Observation ages are explicitly **time since delivery to belief**, not capture
age. The underlying simulator stores delivery timestamps; it does not expose a
precise per-cell capture timestamp. Historical v1 traces require their recorded
source revision to replay exactly.

All policies see the same serialized packet. Candidate tasks are generated from
delivered fire belief and static geography; their heuristic values are disclosed
as heuristics. Hidden dynamic fire, reward, final scores and future weather never
cross the agent boundary. Existing mutation tests replace hidden fire/evaluator
state, regenerate tasks, and require the public packet to remain identical.
Invalid joint assignments consume time with all resources holding, and busy
resources cannot be overwritten. Shared task capacities are checked atomically.

A valid execution must reach its horizon or a simulator terminal state, preserve all failures,
return real model-generated actions, show no disallowed tools, and replay its
observations/transitions/metrics exactly. Variation in action choices and scores
can establish a useful synthetic control task; it cannot establish operational
wildfire validity. These four public synthetic cases, policy-dependent random
number consumption, and only three seeds per case limit generalization and
paired-effect inference. No human-operator comparison is available.

## Collect completed runs

The runner records replay verification even when invalid model dispatches make
its exit code 2. Keep those episodes in the comparison. To combine separately
executed model groups, use a checkout containing both recorded commits:

```bash
python tools/collect_agent_results.py \
  --source codex=runs/codex-evaluation --source opus=runs/opus-evaluation \
  --out runs/combined-evaluation
python tools/audit_agent_results.py runs/combined-evaluation \
  --baseline-reference results/agent_benchmark/baseline_summary.json
```

The collector rejects incomplete runs, changed protocols, prompts, scenarios,
numerical dependencies, and source differences outside the CLI adapter and its
tests. It copies original trace bytes and keeps each run's provenance. Repeated
baseline episodes are deduplicated only if every observation and stable result
field agrees. It never deduplicates stochastic model episodes. The audit verifies
checksums, completed horizons or terminal states, initial observations, recorded
model identities, usage, and failure accounting. It produces `episodes.csv`,
`validity-audit.json`, and `paired-comparisons.json` with all episodes included.

Claude Code may report auxiliary Haiku usage in addition to Opus dispatches.
`adapter_metadata.usage_by_model` preserves this accounting separately from the
init/assistant model identity. Report it as part of the CLI scaffold; do not
present these results as an identical-budget comparison of bare API models.

CLI reference: [OpenAI non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
and [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference).
