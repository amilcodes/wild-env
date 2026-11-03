# Wildfire Desk v1: baseline pilot

This is the original baseline reference. The later
[v1.1 live-agent comparison](frontier_live_results.md) evaluates three real models
against these baselines and verifies that their reference metrics are unchanged.

This is an executed synthetic baseline pilot, not a frontier-model leaderboard.
No provider API keys were available for this run. Native model adapters were
validated against local HTTP fixtures, and the command-agent path was executed
with the explicitly labeled greedy fixture.

The retained run was executed on the existing EC2 CPU host in a dedicated
checkout, under `nice -n 15`, with one benchmark worker and numerical-library
threads capped at one. Lint passed, 189 Python tests passed, one optional PyVista
replay-rendering test was skipped, and the native solver smoke test passed.

The fixed `configs/agents/frontier.yaml` suite contains four cases, three seeds
per case, and a 90-minute horizon with three-minute decisions. Three baseline
policies completed **36 episodes and 1,080 decisions**, with zero invalid
dispatches and zero simulator-blocked actions. The environment includes five
heterogeneous resources, delayed observations, wind dispatch gates, and finite
service stocks. These cases establish an integration/reference result; their
difficulty for frontier models remains unmeasured.

| Observation-only policy | Mean loss growth ↓ | Mean burned area, ha ↓ | Mean paired reduction vs hold ↑ | Failed decisions |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 25.8883 | 7.52 | 10.68% | 0 / 360 |
| Hold | 28.9750 | 9.23 | 0.00% | 0 / 360 |
| Nearest | 29.5605 | 8.43 | −2.08% | 0 / 360 |

Loss growth is the change in the simulator's asset-weighted fire-loss proxy.
Reduction is the mean of matched-seed percentage reductions, not a pooled ratio.
Loss units and resource costs are not dollars or human outcomes. Burned area and
asset-weighted loss can disagree: nearest dispatch reduced burned area while
slightly increasing the weighted-loss proxy in this pilot.

The case-cluster bootstrap gives a mean absolute paired improvement of 3.0867
loss units for greedy, with a descriptive 95% interval [1.4049, 4.9102]. Nearest
has an improvement of −0.5855, interval [−3.3075, 0.9400]. Four public synthetic
clusters are insufficient to infer broad capability or operational effectiveness.
The configuration was not retuned in response to these scores.

## Reproduce and audit

Run these commands in the dedicated EC2 checkout described in the
[remote execution recipe](ec2_agent_benchmarks.md).

```bash
python -m pip install -e '.[dev]'
aeolus-agent-bench run --suite configs/agents/frontier.yaml --out runs/reproduce-desk-v1
# Read runs/reproduce-desk-v1/summary.json.
for trace in runs/reproduce-desk-v1/episode-*.jsonl; do
  aeolus-agent-bench replay "$trace"
done
```

The compact [summary](../results/agent_benchmark/baseline_summary.json) contains
all 36 episode metrics, checksums of the original traces, and paired statistics.
The [manifest](../results/agent_benchmark/baseline_manifest.json) pins the suite,
prompt, source fingerprint, originating Git commit, and numerical dependencies.
The [episode CSV](../results/agent_benchmark/baseline_episodes.csv) provides flat
metrics, and the [replay verification](../results/agent_benchmark/replay_verification.json)
records successful verification of every trace. The original logs remain on EC2
under `runs/ec2-baselines-v1/benchmark/`; all 36 trace checksums were verified
after retrieval.
Timing and trace checksums can vary between reruns because latency is recorded;
observations, dispatches, and simulator metrics replay deterministically in the
same numerical environment. Full JSONL logs are regenerated into ignored `runs/` directories.

Two compatibility problems found during execution were fixed before the retained
pilot: nonterminal escape could override the configured horizon, and the
PettingZoo adapter emitted boolean masks where Gymnasium sampling requires int8.
The first changes previously overrun episodes; no fire or suppression equation
was adjusted. Read the [protocol](frontier_agent_benchmark.md) for the information
boundary, failure handling, budgets, and historical-evaluation requirements.
