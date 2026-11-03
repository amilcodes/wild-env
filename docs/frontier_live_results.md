# Wildfire Desk v1.1: executed frontier-agent pilot

Three real CLI agents completed the frozen EC2 evaluation on November 2, 2026:
`gpt-6.1-sol`, `gpt-6.1-terra`, and `claude-opus-5-5`, all requesting high reasoning
effort. **No model showed a clear improvement over greedy in this pilot.** Terra
and Opus had nearly the same mean loss as greedy; Sol had somewhat higher loss.
All models reduced mean burned area relative to greedy, illustrating that burned
area and asset-weighted loss measure different outcomes.

The comparison contains **72 unique episodes, 2,160 dispatch decisions, and
1,080 live CLI decision requests**. Four public synthetic cases use three seeds
each, five resources, a 90-minute horizon, and three-minute decisions. Every
policy receives the same public observation contract. All episodes, including
failed dispatches, contribute to the following descriptive values.

| Policy | Mean loss growth ↓ | Mean burned area, ha ↓ | Mean paired reduction vs hold ↑ | Invalid decisions |
| --- | ---: | ---: | ---: | ---: |
| Hold | 28.9750 | 9.227 | 0.00% | 0 / 360 |
| Nearest | 29.5605 | 8.427 | −2.08% | 0 / 360 |
| Greedy | 25.8883 | 7.520 | 10.68% | 0 / 360 |
| GPT-6.1 Sol, high | 26.5611 | 6.667 | 9.12% | 0 / 360 |
| GPT-6.1 Terra, high | 25.8819 | 7.040 | 11.35% | 0 / 360 |
| Claude Opus 5.5, high | 25.8831 | 6.987 | 10.91% | 2 / 360 |

Loss growth is a simulator asset-weighted fire-loss proxy, not dollars or lives.
Relative reductions average matched-condition percentages rather than pooled
totals. Opus duplicated the tanker assignment in two responses: initial-attack
seed 211, decision 5; wind-restrictions seed 907, decision 2. Each invalid joint
dispatch held the fleet for one interval. Ten of its twelve episodes were fully
valid; the run is **ineligible for formal ranking** under the frozen failure rule.
Its complete outcome data remain in all descriptive comparisons.

Paired absolute improvements over greedy were Sol −0.6728, Terra +0.0064, and
Opus +0.0052 loss units. Their descriptive case-cluster 95% intervals were
respectively [−3.0239, 1.7123], [−3.0242, 2.7605], and [−1.5089, 2.2281]. All
include zero. Four clusters and one model trajectory per case/seed do not support
a broad capability ranking or a reliable model advantage.

## Execution and verification

Simulation, scoring, tests, and replay ran on the existing EC2 CPU instance at
`nice 15` with numerical threads capped at one. The Mac relayed requests through
the established RSG authenticated CLI bridges using dedicated loopback SSH
tunnels. Existing RSG processes and original bridge ports were left untouched.
The task's temporary relays and wake lock were stopped after inference completed.
Claude's expired OAuth session was resolved during preflight; those authentication
errors are recorded separately from evaluation scores.

All 84 executed evaluation episodes passed deterministic replay, including twelve
repeated hold episodes in the separate Opus run. Those repeats matched every
observation and stable result field and were counted once in the combined set.
All 72 unique trace checksums passed verification again after retrieval. The 36
baseline episodes exactly matched the earlier retained reference metrics. There
were zero disallowed tool events and zero simulator-blocked actions.

The Sol/Terra run used clean commit `6e2cb62762194821d3a42c921a1c0afff7098115`;
Opus used `91b4930ca52b711c8e42eb0d3aee2b48bd6b5d47`. The latter only separates
Claude dispatch identity from auxiliary CLI usage and adds its regression test.
Environment, task prompt, schema, and scoring were identical. Configurations were
frozen before reading evaluation outcomes; no scores were used for policy tuning.
The source runs exited 0 and 2 respectively; exit 2 preserves Opus's ineligible
episodes and does not indicate a replay or infrastructure failure.

The respective EC2 test runs passed 201 and 202 tests with one optional PyVista
test skipped. [Model-code CI](https://github.com/amilcodes/wild-env/actions/runs/41545557702)
and [collection/audit CI](https://github.com/amilcodes/wild-env/actions/runs/41547165045)
passed Python 3.10/3.11 and native smoke checks. Hidden-state mutation, atomic
dispatch validation, capacity, memory, and serial/parallel replay tests passed.
The collector also rejected altered protocols, edited traces, and inconsistent
duplicate baseline metrics.

## Agent configuration and limits

Codex CLI was version 0.165.0; Claude Code was 2.1.291. Each decision started a
fresh session, with only the public packet, previous receipt, and at most 2,000
characters of memory. Each model had a 30-call episode budget and 240-second
per-call deadline. No adapter repaired or retried a response. Reported mean
latencies were Sol 17.05s, Terra 7.05s, and Opus 10.61s, including bridge overhead.

Model identity was verified for all 1,080 requests. Codex model and effort came
from its session `turn_context`; Claude identity came from init/assistant events.
Claude's effort setting was requested but not independently attested. These are
CLI attestations, not provider-signed immutable model snapshots.

This compares named CLI agents, with provider-specific scaffolding and internal
rounds. Claude reported 726 assistant rounds and auxiliary Haiku usage; the audit
retains Opus and Haiku accounting separately. Output-token caps were unavailable,
so this is not an equal-token-budget comparison. All calls reported usage.

The observation boundary, executable actions, failure handling, and replay are
validated. This remains a **synthetic control benchmark**, with public cases,
policy-dependent random-number consumption, no replicated model sampling, and no
human comparator. It does not establish operational wildfire validity or
historical forecasting skill.

## Plain evidence and reproduction

The retained [summary](../results/agent_benchmark/frontier_live_summary.json),
[episode CSV](../results/agent_benchmark/frontier_live_episodes.csv),
[manifest](../results/agent_benchmark/frontier_live_manifest.json),
[validity audit](../results/agent_benchmark/frontier_live_validity_audit.json),
[paired comparisons](../results/agent_benchmark/frontier_live_paired_comparisons.json),
[replay checks](../results/agent_benchmark/frontier_live_replay_verification.json),
[transport record](../results/agent_benchmark/frontier_live_transport.json), and
[test record](../results/agent_benchmark/frontier_live_validation.json) contain the
measured evidence. Raw traces, original source-run logs, and development runs
remain on EC2 and in the delivered trace archive; raw traces are not tracked in Git.

Use [the CLI bridge guide](cli_agent_evaluation.md) and
`configs/agents/bridge.example.yaml` to run all six policies. Replay original
traces using their recorded source revision and numerical dependency versions.
The guide documents collecting separate completed runs without dropping failures
or changing trace bytes. Subsequent live model requests need not reproduce the
same dispatches; replay of recorded dispatches must reproduce the observations
and simulator outcomes.
