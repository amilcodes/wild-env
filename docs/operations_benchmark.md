# Wildfire response operations v2

This lane measures sequential operational planning using explicit asset-protection
jobs, deadlines, heterogeneous duration tables, prerequisites, shared water and
an uncertain wind-corridor forecast. It is a finite scheduling abstraction. It
does not replace the canonical fire simulator or claim calibrated wildfire loss.
The earlier v1.1 results remain unchanged.

The v1 pilot mixed unavoidable fire loss and policy effects, offered pre-ranked
nearby tasks, and used action-dependent consumption of a shared random stream.
A diagnostic hold rollout at development seed 17 found zero mapped-asset burning
at both the original and extended horizons. A wider numerical score range alone
would not resolve those validity issues.

## Task and score

Two response teams have public job durations that include travel, work and
turnaround. Jobs are nonpreemptive. Protection completed by its deadline earns
the declared value under the realized forecast. Starting any job incurs its
declared cost. A completed access job may unlock other jobs. Water is consumed
at dispatch and cannot be oversubscribed by a joint assignment.

The forecast has two fixed possible outcomes with a public prior. The actual
outcome is revealed by an exact scheduled bulletin or a completed survey. Survey
time and cost create a measurable value-of-information decision. The actual
outcome never depends on the policy. Both possible outcomes are evaluated for
uncertain instances; deterministic instances need only one. Numerical IDs, labels,
resource order and task order convey no priority.

An exact Bellman planner uses the same public information as the agent. Before
revelation it averages over the forecast prior; it does not inspect the actual
world. For each action, decision regret is `V*(state) - Q*(state, action)`.
Primary episode regret is the sum of those losses divided by the initial optimal
expected reward, multiplied by 100. Lower is better; zero means every chosen
action preserved an optimal continuation. Values are not clipped to 0–100.
Expected cumulative regret equals the optimal expected reward minus the policy's
expected reward. Tests verify this identity over both possible forecast outcomes.

Realized reward and reward capture are retained separately. A single lucky
trajectory can exceed the initial expected reward; it does not establish a better
policy. Forced holds do not count as contested decisions in the optimal-choice
rate. Invalid model dispatches hold every resource for a tick and remain scored.
Infrastructure, model-identity or disallowed-tool errors make comparisons
ineligible; they are never silently repaired or converted into successful calls.

## Controls and independent verification

Controls are hold, immediate value per duration, deadline-weighted value, two-step
search with a myopic continuation, and the exact information-matched planner.
All use public information. The exact solver has a hard state budget and aborts
instead of reporting an approximate optimum.

An independent time-indexed mixed-integer formulation verifies the fully informed
scheduling optimum for each forecast. Its variables represent job/resource/start
time; constraints separately enforce exclusivity, overlap, prerequisites and
water. Replaying the adaptive reference policy in both forecast worlds verifies
the information-matched expected value. Every planned instance is checked before
any evaluation model request. Checks do not select instances using model scores.

## Frozen evaluation and inference

Development seeds and evaluation seeds are separate. The evaluation fixes six
families at two difficulty levels: deadline tradeoffs, prerequisites, scarce
water, early information acquisition, waiting for forecasts, and combined
constraints. The generator varies values, durations, deadlines, release times,
resource ordering and labels. Each model gets two fresh repetitions per world.
These repetitions are samples from the same task, not extra independent tasks.

The planned model panel includes GPT-6 Luna, Claude Haiku 5.5, GPT-6.1 Sol,
GPT-6.1 Terra, Claude Opus 5.5 and GPT-6 Astra at high requested effort. Smaller
models are controls, not an assumed ordering. Real preflights verify access.
Model identities, CLI versions, effective prompt hashes, usage, latency and full
dispatch traces are retained. The [existing bridge setup](cli_agent_evaluation.md)
keeps simulation and reference solving on EC2. Provider CLI scaffolding differs,
and there is no enforceable output-token cap.

Instances receive equal weight after averaging their forecast worlds and model
repetitions. Bootstrap intervals resample generated instances. Paired sign-flip
tests assume exchangeability; Holm adjustment covers all model-versus-model
comparisons. Separation is reported only when the adjusted test and interval
agree. No particular frontier-model ordering is a design requirement. The small
number of independently generated structures and six families limits broad claims.

## Run on EC2

```bash
# Baseline development and reference checks require no model access.
nice -n 15 bash deploy/ec2/run_operations_benchmark.sh \
  runs/ops-controls configs/agents/operations-development.yaml

# Run development first; inspect infrastructure and contract failures.
WILD_ENV_WORKERS=6 nice -n 15 bash deploy/ec2/run_operations_benchmark.sh \
  runs/ops-development configs/agents/operations-development.yaml \
  configs/agents/operations-models.yaml

# Freeze code, task definition, scoring, model panel and settings before this run.
WILD_ENV_WORKERS=6 nice -n 15 bash deploy/ec2/run_operations_benchmark.sh \
  runs/ops-evaluation configs/agents/operations-evaluation.yaml \
  configs/agents/operations-models.yaml
```

Launch long jobs with the detached EC2 recipe. Outputs are plain JSON, JSONL and
CSV: complete/partial summaries, reference certificates, every model response,
exact decision regret, replay verification and paired comparisons. The optimal
development traces also provide verifiable demonstrations; training gains and
transfer to other environments have not been established.
