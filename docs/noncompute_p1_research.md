# Non-compute-bound P1 research pass

Date: 2026-07-30

## Scope

This pass addressed four validity controls that could be improved without a
large simulation or training allocation:

1. acquisition-time and observability semantics for historical fire
   observations;
2. historical fuel provenance and fast-kernel regime validity;
3. tactical aircraft performance, service-site geometry, terrain, wind, and
   airspace feasibility; and
4. frozen-partition and paired-policy evaluation machinery.

The work changes which evidence the system will accept. It also exposes
several current datasets as insufficient for the claims previously contemplated
for them.

## Current conclusion

| Control | Implementation result | Evidence from this pass | Gate |
|---|---|---|---|
| Acquisition-time uncertainty | Source time is represented as an acquisition interval plus availability time | In a controlled 800-acquisition experiment, interval likelihood reduced mean Brier score by 70.5% relative to treating the bin end as the sensing instant | Mechanism passed; source calibration open |
| Historical fuel chronology | Every fuel source is screened against incident time | All six prepared historical incidents use LANDFIRE 2025 and are flagged for potential post-incident information | Failed |
| Fast surface-behavior domain | Every forcing axis is checked against the actual packaged lookup axes | All 9,781,248 values inspected for the Electra forcing case were inside the numerical table domain | Numerical-domain gate passed; empirical fire-regime gate open |
| Aircraft feasibility | Canonical action masks use density altitude, payload derating, vector wind, terrain ceiling, service geometry, route endurance, and time-active airspace volumes | The reference sweep produces monotonic payload derating and rejects a route crossing a reserved volume | Interface passed; vehicle evidence open |
| Policy comparison | Effects are paired by case and seed and bootstrapped by case cluster | Rollout has 0.521 more terminal fire loss than exact assignment while improving the configured loss-plus-cost objective by 0.999; it does not separate from no action | Objective gate failed |
| Expanded benchmark partition | Required partition and regime fields are audited before evaluation | All 36 selected incidents lack `split`, `fuel_family`, `weather_regime`, and `ecoregion` | Failed |

The complete machine-readable result is
[`results/noncompute_p1/noncompute_p1_study.json`](../results/noncompute_p1/noncompute_p1_study.json).
The summary figure is
[`results/noncompute_p1/noncompute_p1_study.png`](../results/noncompute_p1/noncompute_p1_study.png).

## 1. Observation acquisition and availability

### Scientific issue

Airborne and satellite products often integrate, bin, mosaic, or publish data
over a time interval. The nominal timestamp can denote a bin, a scene, or
publication time. Treating that timestamp as an exact observation time
introduces a systematic scoring error near an advancing front. Cloud and smoke
also reduce observability; an obscured pixel does not provide the same
negative-fire evidence as a clear pixel.

This treatment is consistent with the contextual, cloud-sensitive nature of
375 m VIIRS active-fire detection described in the
[NASA VIIRS 375 m active-fire algorithm document](https://viirsland.gsfc.nasa.gov/PDF/VIIRS_activefire_375m_ATBD.pdf).
The project’s FEDS data source is the
[NASA Fire ESDR product](https://viirsland.gsfc.nasa.gov/Products/NASA/FireESDR.html).
Arrival-history replay after perimeter correction follows the coupled-state
problem described in the
[WRF-SFIRE perimeter-assimilation study](https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2023.1203578/full).

### Implemented observation model

For acquisition interval \([t_0,t_1]\), predicted arrival time \(T\), and a
uniform acquisition-time prior:

```text
P(burned during acquisition | T)
    = clip((t1 - T) / (t1 - t0), 0, 1)
```

The expected raster observation is:

```text
P(observed fire)
    = p_false_alarm
      + (p_detection - p_false_alarm)
        * GaussianLocalize(P(burned during acquisition))
```

Scoring uses information weight \(1-p_\text{obscured}\). The implementation
returns weighted Bernoulli log likelihood, mean log score, Brier score,
effective observable cells, temporally ambiguous cells, acquisition duration,
and processing latency. The likelihood is directly usable for particle
weights. The availability time is kept separate from acquisition time so a
historical replay cannot use an observation before it would have been
available.

Code:

- [`src/aeolus/evaluation/observation.py`](../src/aeolus/evaluation/observation.py)
- [`tests/test_observation.py`](../tests/test_observation.py)

### Controlled experiment

A radial synthetic fire was assigned a 120-minute acquisition window.
Eight hundred exact sensing instants were sampled uniformly inside the window.
Scoring was restricted to 4,148 cells in the temporal ambiguity band.

| Interpretation | Mean Brier | Standard deviation |
|---|---:|---:|
| Acquisition-window likelihood | 0.1127 | 0.0532 |
| Exact midpoint assumption | 0.1713 | 0.0999 |
| Exact bin-end assumption | 0.3815 | 0.1965 |

The result verifies temporal-likelihood semantics under its stated generative
model. Sensor-specific detection, geolocation, scan timing, cloud, and smoke
parameters remain uncalibrated.

## 2. Historical fuels and fire-regime validity

### Provenance gate

LANDFIRE provides current and historical landscape products, disturbance
information, and seasonal fuels through separate product paths
([LANDFIRE data](https://landfire.gov/data),
[public events](https://landfire.gov/reference/publicevents), and
[seasonal fuels](https://landfire.gov/fuel/seasonal_fuels)).
A landscape product released after a historical incident can encode
disturbance or succession information that was unavailable at the incident
time.

The new gate records:

- source name and product year;
- an optional exact data cutoff;
- missing product-year provenance;
- products later than the incident;
- same-year products without a usable cutoff; and
- one explicit admissibility status.

A later product year is a leakage screen. Pixel-level disturbance history is
still required to determine whether the incident landscape actually changed.

The public LANDFIRE importer now records product year 2025, disturbance
coverage through 2024, and a historical-use warning. The audit found that all
six currently prepared incidents, dated 2020–2023, use LANDFIRE 2025. Those
bundles fail the current historical-fuel gate. Results that depend on those
landscapes retain their prior diagnostic value, while the fuel chronology
prevents treating them as clean held-out historical evidence.

Code:

- [`src/aeolus/evaluation/validity.py`](../src/aeolus/evaluation/validity.py)
- [`src/aeolus/data/importers.py`](../src/aeolus/data/importers.py)
- [`tests/test_validity.py`](../tests/test_validity.py)

### Numerical validity envelope

The fast-kernel audit reads the axes from the packaged behavior lookup and
counts forcing values outside each axis before interpolation can clip them. It
separately marks any crown or spotting activity as a mechanism-only regime.

For the prepared Electra forcing:

- wind: 0 of 2,441,216 values outside 0–30 m/s;
- terrain slope tangent: 0 of 16,384 values outside 0–1;
- dead 1 h moisture: 0 of 2,441,216 values outside 0.03–0.40 kg/kg;
- live herbaceous moisture: 0 of 2,441,216 values outside 0.30–2.50 kg/kg; and
- live woody moisture: 0 of 2,441,216 values outside 0.60–2.00 kg/kg.

This is a numerical-domain result. Crown transition and spotting require
independent calibration. The established crown-fire model remains an important
reference for those tests
([Scott and Reinhardt, 2001](https://research.fs.usda.gov/treesearch/4623)).
Spatial live-fuel-moisture evaluation can use the
[Globe-LFMC product description](https://research.fs.usda.gov/download/treesearch/66953.pdf)
as one independent source.

## 3. Tactical aircraft feasibility

### Implemented state and constraints

Each aircraft may now reference a provenance-bearing performance surface with:

- density-altitude and payload-fraction axes;
- true airspeed;
- endurance multiplier; and
- maximum payload fraction.

Canonical leg evaluation computes:

- terrain sampled along the straight route;
- planned MSL altitude from terrain plus declared AGL cruise altitude;
- density altitude from terrain elevation and temperature;
- bilinear true-airspeed and endurance derating;
- vector tailwind, crosswind, and groundspeed;
- leg time and recovery reserve;
- vehicle/site depth and usable-length compatibility;
- maximum operating altitude; and
- intersection with a polygon, altitude band, and active time interval.

Outside-surface use is a hard violation. The action mask checks outbound and
recovery legs. Assignment and replay events carry density altitude,
groundspeed, crosswind, and planned altitude. A return leg that becomes
infeasible is recorded and the resource is withdrawn.

The density-altitude approximation is a screening calculation based on the
[FAA Pilot’s Handbook of Aeronautical Knowledge](https://www.faa.gov/sites/faa.gov/files/pilot_handbook_1.pdf).
Operational procedure requirements remain governed by current interagency and
aviation guidance, including the
[2026 NIFC Red Book](https://www.nifc.gov/standards/guides/red-book) and
[NWCG PMS 520](https://www.nwcg.gov/publications/pms520).

Code and contract:

- [`src/aeolus/core/aviation.py`](../src/aeolus/core/aviation.py)
- [`src/aeolus/core/tasks.py`](../src/aeolus/core/tasks.py)
- [`src/aeolus/core/simulator.py`](../src/aeolus/core/simulator.py)
- [`configs/aviation/generic_research_rotorcraft_v1.json`](../configs/aviation/generic_research_rotorcraft_v1.json)
- [`tests/test_aviation.py`](../tests/test_aviation.py)

### Control result and evidence boundary

At 30 °C, the synthetic reference surface reduced maximum payload fraction
from 0.963 at 549 m density altitude to 0.535 at 4,262 m. A route intersecting
the study airspace volume was rejected. Crosswind, terrain ceiling,
out-of-surface, service-depth, and service-length gates have unit coverage.

The bundled surface is intentionally synthetic and identifies itself as an
interface-verification artifact. Vehicle claims require reviewed flight-manual
tables or another independently accepted performance dataset. Straight-route
blocking currently provides feasibility screening; route finding, flight
phases, obstacle databases, separation, and incident-airspace workflow remain
open.

## 4. Evaluation and planning controls

### Frozen partitions

`EvaluationCase` fixes case, incident, geography, year, fuel family, weather
regime, and split. The auditor rejects:

- duplicate case identifiers;
- one incident or another declared exclusive group appearing in several
  splits;
- invalid split labels; and
- empty train, development, or test sets.

It reports geography, year, fuel, and weather coverage by split. The expanded
36-incident manifest currently has state/year/size strata and lacks the four
required partition/regime fields. It therefore fails before any large
historical run can be interpreted as frozen held-out evidence.

### Paired effects

Policy comparisons pair candidate and baseline on exact `(case, seed)` keys.
Effects are averaged within case, then bootstrapped over case clusters.
Positive improvement always favors the candidate. Missing pairs are retained
in the result rather than silently discarded.

### Rollout and objective falsification

The first receding-horizon comparator evaluates legible first-action proposals
on cloned simulators with identical random state, followed by exact joint
assignment. It is a privileged-model planning diagnostic.

Across four small wind/seed cases:

| Policy | Mean terminal fire loss | Mean configured loss-plus-cost objective |
|---|---:|---:|
| No aerial action | 10.5077 | 10.5077 |
| Exact joint assignment | 9.8976 | 11.4976 |
| Two-decision rollout | 10.4185 | 10.4985 |

On terminal fire loss, rollout is 0.5209 worse than exact assignment; its 95%
case-cluster interval is a 0.1758–0.8661 penalty. On the objective implied by
the configured incremental reward, rollout is 0.9991 better; its 95% interval
is 0.4958–1.4242. The configured objective is:

```text
terminal weighted loss
  + (0.02 / reward_loss_scale) * cumulative sortie cost
  + (0.01 / reward_loss_scale) * blocked actions
```

With `reward_loss_scale = 0.05`, one unit of sortie cost is weighted as 0.4
units of fire loss. Exact assignment spends 4.0 cost units in every case, so
its 1.6 objective penalty exceeds its mean fire-loss advantage. Rollout then
largely chooses no action. Its aligned-objective improvement over no action is
only 0.0092, with a 95% interval of 0–0.0275; the positive-interval gate fails.

This is an objective-calibration failure before it is a policy result. The
current reward does not identify a useful trade between avoided fire loss and
resource cost in these cases. The short rollout also has travel/response
latency and no commitment state. It remains a diagnostic and is excluded from
the policy-quality claim set. A rolling-horizon optimizer needs calibrated
costs, explicit constraints, commitment state, terminal value, and the same
objective used in final evaluation.

Code:

- [`src/aeolus/evaluation/protocol.py`](../src/aeolus/evaluation/protocol.py)
- [`src/aeolus/policies/heuristics.py`](../src/aeolus/policies/heuristics.py)
- [`tests/test_protocol.py`](../tests/test_protocol.py)

Wildfire-resource planning literature supports stochastic availability,
production, and assignment constraints:
[initial-attack simulation and optimization](https://research.fs.usda.gov/treesearch/42818),
[stochastic fireline production](https://research.fs.usda.gov/treesearch/47358),
and
[aerial-resource assignment](https://doi.org/10.1093/forsci/fxy012).

## Reproduction and verification

```bash
python tools/run_noncompute_p1_study.py
pytest -q
ruff check src tests tools
```

Current verification:

- 93 tests passed;
- repository-wide static checks passed;
- patch whitespace validation passed; and
- the study JSON and 2×2 figure were regenerated from the checked code.

## Interpretation boundary

The observation experiment is synthetic, the aircraft performance surface is
synthetic, the planning study is a small internal mechanism test, and the fuel
audit is a chronology screen. These controls reduce the chance of making an
unsupported statement. They do not establish operational forecast accuracy,
vehicle performance, field suppression effect, or policy superiority. The
remaining gates and their exact evidence requirements are maintained in
[`noncompute_p1_remaining_work.md`](noncompute_p1_remaining_work.md).
