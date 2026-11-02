# Wildfire Desk: frontier model and agent benchmark

Wildfire Desk asks a language model to manage a changing incident, allocate a
heterogeneous fleet, gather information, and work within operational constraints.
It is inspired by the [Scale AI battery trading desk demonstration](https://x.com/earakely/status/2117327354410508302):
a real decision interface, an environment that executes actions, and measurable
outcomes against explicit comparators. Here the outcomes are **simulated wildfire
loss and resource use**, not trading profit or measured real-world firefighting
performance.

## Run it

From the repository root with Python 3.10–3.12:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

# No credentials, no model charges: 12 short episodes.
aeolus-agent-bench run --suite configs/agents/smoke.yaml --out runs/desk-smoke

# 36 episodes: four challenges × three seeds × three baselines.
aeolus-agent-bench run --suite configs/agents/frontier.yaml --out runs/desk-frontier

# View the initial packet or inspect the exact tool schema.
aeolus-agent-bench inspect --suite configs/agents/frontier.yaml --case delayed-recon
aeolus-agent-bench schema

# Verify recorded actions and scores without a model call.
aeolus-agent-bench replay runs/desk-frontier/episode-0002.jsonl
```

Open `runs/desk-frontier/report.html` directly in a browser. It works offline and
contains the outcome table, paired effects, and an episode/decision inspector.
Existing output directories are rejected to protect previous runs. Interrupted
runs retain flushed JSONL traces and a summary explicitly marked incomplete;
start a fresh directory to rerun. `report RUN_DIR` regenerates an interrupted
run's report from its completed episodes.

## Live model adapters

Copy `configs/agents/providers.example.yaml` to a local file under `runs/`.
Replace model placeholders with exact IDs available to your provider account,
remove unused entries, and retain exactly one `hold` baseline. Set the named
credential environment variables locally; never put credentials in the YAML.

```bash
aeolus-agent-bench run \
  --suite configs/agents/frontier.yaml \
  --agents runs/my-providers.yaml \
  --out runs/my-model-comparison
```

| Provider | Wire API | Credential default |
| --- | --- | --- |
| `openai` | Responses API, one forced `submit_dispatch` function, `store: false` | `OPENAI_API_KEY` |
| `anthropic` | Messages API, one forced `submit_dispatch` tool | `ANTHROPIC_API_KEY` |
| `compatible` | Chat Completions tool calls; explicit `base_url` required | `OPENAI_API_KEY` |
| `command` | One JSON request on stdin; one dispatch JSON on stdout | Managed by your command |

Use `api_key_env` to choose a different variable (for example, a gateway key).
The compatible adapter allows unauthenticated loopback inference servers. Remote
URLs require HTTPS and cannot embed credentials or query parameters. The OpenAI
adapter accepts optional `reasoning_effort`; other provider-specific reasoning
settings are deliberately not guessed. Gateway tool support depends on the
selected model. A refusal, missing tool, multiple tool calls, incomplete response,
or malformed JSON is a failed decision rather than a repaired answer.

The native adapter contracts follow the official
[OpenAI function-calling documentation](https://developers.openai.com/api/docs/guides/function-calling)
and [Anthropic tool definitions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools).
HTTP fixture tests exercise request and response formats. No paid model call is
needed for the test suite. Live model availability and access must be verified in
your account; this branch ships no fabricated model scores.

Each episode allows `max_calls` requests and each request caps output tokens.
`timeout_s` is a socket timeout for remote HTTP and a process deadline for command
agents. There are no hidden retries. After three consecutive failed decisions,
the agent circuit opens for the remainder of that episode. Exhausted budgets and
provider errors advance the incident with hold. These results remain visible,
but the affected agent is ineligible for ranking. Token usage is reported as
returned by the provider, including Anthropic cache tokens; missing usage remains
unknown. There is no claimed hard dollar or input-token cap. Use provider-side
spending limits when running a paid campaign.

## Bring an agent framework

```bash
aeolus-agent-bench run --suite configs/agents/smoke.yaml \
  --agents configs/agents/command.example.yaml --out runs/command-smoke
```

`examples/desk_agent.py` is a working greedy fixture, **not an LLM**. Replace its
argv with your own wrapper. The command receives one JSON object containing
`system`, `observation`, and `dispatch_schema`; it writes only the dispatch JSON
to stdout and can log to stderr. The process starts anew each decision. Use the
packet's bounded `memory` field for continuity, or explicitly document any extra
state/scaffolding used by your framework. Command usage tokens are unknown in v1.
Commands are trusted local programs, not sandboxed submissions. For independent
external evaluation, isolate the process/container from simulator files, traces,
network data, and evaluator state. The JSON boundary protects information flow
in the standard harness but is not a hostile-code security boundary.

## Observation and action contract

The agent sees the current minute, current weather domain means, a 4×4 sector
summary of delayed fire belief and static mapped assets, fleet status and payload,
endurance and travel estimates, service stocks, and belief-derived candidate tasks.
Every resource has an explicit legal-task list. Task values are heuristic scores,
not predicted outcomes. Unknown sector ages are `null`. Travel estimates are
straight-line nominal travel times; the simulator's feasibility mask additionally
checks route, wind, payload, recovery, and endurance constraints.

The agent cannot see future weather, RNG state, dynamic fire truth, hidden arrival
history, cumulative loss, terminal scores, or a simulator rollout oracle. Dispatch
receipts use an explicit allowlist because canonical simulator `info` dictionaries
also contain privileged evaluator metrics. Tests mutate hidden fire fields and
confirm that the agent-visible packet remains unchanged.

At each three-minute decision the agent submits:

```json
{
  "decision_id": "current-id-from-observation",
  "assignments": [
    {"resource_id": "each-resource-exactly-once", "task_id": 0}
  ],
  "rationale": "Brief operational explanation, at most 1000 characters.",
  "memory": "Notes to retain, at most 2000 characters."
}
```

A submission must cover every resource exactly once, use current legal task IDs,
and respect shared nonzero task capacity. It is validated atomically. A stale,
incomplete, or illegal submission turns the entire joint dispatch into hold and
consumes one interval. Hold continues any active mission; it does not cancel it.
The simulator handles travel, suppression delivery, return, service queues, reload,
and safety gates. Only the previous receipt and bounded memory carry over to the
next model request. The v1 task is one closed-loop dispatch per decision; it does
not give models an unconstrained shell, free simulations, or unlimited tool turns.

## Cases, metrics, and interpretation

`frontier.yaml` fixes four 90-minute synthetic cases with five resources: a tanker,
two helicopters, a scout, and a ground crew. Three seeds per case yield 12 matched
initial conditions. Challenges are fleet coordination, 12-minute observation delay,
wind restrictions that ground some aircraft, and scarce/closing refill capacity.
The public evaluation label separates this manifest from smoke development; it is
not a secret holdout or a contamination-resistance claim. Do not tune the policy
on these outcomes and describe it as an unseen evaluation. Version new protocols
when changing scenarios, prompts, scoring, or information access.

Primary loss growth is final minus initial canonical `weighted_loss`. This proxy
weights consumed-fuel loss and active intensity by static asset value. It is in
simulator units, not dollars, hectares, lives, or a calibrated damage estimate.
The secondary outcomes are burned area in hectares, resource cost, exposure,
escape/containment flags, and reliability. The report sorts eligible agents by
mean loss growth. It also reports each matched pair's reduction:

`100 × (hold_loss_growth − candidate_loss_growth) / hold_loss_growth`

The report averages these per-pair percentages; it is not a percentage computed
from pooled totals. Pairs whose hold growth is nonpositive have undefined relative
reductions and are omitted from that percentage only; `normalized_pairs` exposes
the denominator. Absolute paired effects use the repository's existing
case-cluster bootstrap (2,000 draws, fixed seed). With four synthetic clusters,
intervals are descriptive, not evidence of broad generalization. Matching initial
seeds does not guarantee identical random disturbances after actions diverge:
reconnaissance and suppression can consume different random numbers.

Baselines `hold`, `nearest`, and `greedy` consume the same serialized observation
as models. Greedy reserves shared task capacity and weighs value, uncertainty,
information gathering, and travel time. These are dispatch heuristics, not human
operators. The existing privileged rollout-lookahead policy is intentionally not
included in the standard comparison.

## Artifacts and reproducibility

- `manifest.json`: expanded suite inputs, suite/prompt hashes, agent configuration,
  code fingerprint, Git commit/dirty state, software versions, and limitations.
- `summary.json`: complete/partial status, every episode, paired comparisons,
  reliability, reported usage, and trace checksums.
- `episode-NNNN.jsonl`: exact agent packet and its checksum, response/tool payload,
  receipt, latency, usage, response ID/model, evaluator-only metrics, and final result.
- `report.html`: self-contained offline audit view. Agent text is HTML escaped.

Replay reconstructs the scenario and seed, re-executes recorded actions, and
compares every observation hash, receipt, intermediate metric, and terminal score.
Reproducibility requires the recorded code and numerical environment. Raw traces
belong in ignored `runs/` or an external artifact store, not Git. Compact measured
summaries are retained under `results/agent_benchmark/`.

V1 rejects external incident/weather/performance-file dependencies rather than
silently omitting their provenance. Historical evaluation needs an additional
versioned contract: hashed input bundles, issue-time availability, frozen incident
splits, reviewed operations, and an independently measured human baseline. The
underlying project supports incident imports, but the current held-out fire-skill
limits still apply. See [historical validation](historical_validation.md).

## Verification

```bash
python -m pip install -e '.[geo,met,render,dev]'
ruff check src tests tools examples
pytest
```

CI also runs a smoke benchmark, deterministic replay, and the native solver smoke
checks. The branch fixes canonical nonterminal escape overriding the episode
horizon and adapts PettingZoo binary masks to Gymnasium's required `int8` dtype.
Neither change modifies fire-spread or suppression equations. The horizon fix
changes outcomes only for episodes that previously continued beyond their stated
horizon, so old overrun results must not be pooled with this protocol.
