# Suppression, operations, and coupled-state initialization

## Scope

Version 0.5 adds an explicit suppression and incident-state layer around the
v0.4 fire kernel. The increment addresses five coupled questions:

1. how liquid drops are represented without inventing treatment area;
2. how line construction, engagement, breach, duty, travel, reload, and
   operational gates enter the semi-Markov decision process;
3. how two observed perimeters initialize arrival time, burn age, residual heat,
   and recent front velocity;
4. how station observations condition gridded wind and fuel moisture; and
5. how perimeter corrections remain localized to the advancing front.

The implementation is an RL research environment. It does not reproduce an
agency dispatch system or certify a suppression tactic.

## Suppression state

### Conserved aerial drop volume

Every aerial drop starts with payload volume `V` in litres and a normalized
wind-displaced ground pattern `p(i,j)`:

`sum(i,j) p(i,j) = 1`

The cell coverage is

`C(i,j) = V p(i,j) / A_cell / 0.407436`

in coverage levels, where one GPC is one US gallon per 100 square feet and
equals 0.407436 L/m2. The Gaussian pattern has a finite along-track and
cross-track support. Wind displaces the centroid and increases dispersion, but
cannot create or destroy volume. Tests reconstruct the payload from the stored
coverage field to a relative tolerance of `1e-5`.

Required coverage level is a fuel-class lookup derived from the USFS coverage
guide, with an explicit fireline-intensity increment. Treatment efficacy is a
saturating response:

`E = 1 - exp(-C / C_required)`

Water reduces current intensity and conditions one-hour dead fuel. It decays
with an 8 minute default half-life. Retardant changes local spread potential,
decays with a 720 minute default half-life, and is washed down by accumulated
rain. Both raw GPC and effective treatment are retained in truth and replay.

This is more identifiable than a binary treatment raster. The response curve
still requires independent calibration against drop telemetry and fireline
effectiveness records.

### Constructed line and engagement

Crew and dozer resources receive a line mission with a center, orientation,
planned length, width, and production distribution. Construction accrues each
minute:

`L(t + 1) = L(t) + q_nominal M`

where `M` is a mean-one lognormal multiplier parameterized by an operational
coefficient of variation. The default CV is 0.38. This represents the observed
variability of realized production without claiming that a single table rate
is an incident forecast.

Each rasterized line cell has:

- constructed width/strength;
- status `unengaged`, `held`, or `breached`;
- colocated retardant coverage; and
- a logged engagement history.

When fire reaches the line, demand is the maximum adjacent fireline intensity.
Capacity combines a base term, constructed width, and retardant reinforcement.
A logistic engagement model produces held or breached outcomes. A held cell has
zero local normal spread while its flanks remain open; a breached cell loses
the line hold. This allows slop-around and end runs instead of declaring a
completed line to be unconditional containment.

### Tactical task construction

The belief-derived task set now contains:

- `OBSERVE`: delayed local measurement;
- `WATER`: direct short-duration intensity/moisture treatment;
- `RETARDANT`: direct or parallel aerial treatment;
- `REINFORCE`: retardant colocated with a planned ground line;
- `LINE`: indirect asset-shielding crew/dozer construction; and
- `HOLD`.

Line centers are placed ahead of the selected belief-front cell. Where assets
are threatened, the approach vector points from the front toward the
value-weighted asset centroid and the line is normal to that vector. Otherwise
the wind-driven head supplies the approach vector. Reinforcement shares the
line geometry. The orientation is stored in mission state; it is not
recomputed after travel.

This is a deterministic doctrine comparator. Learned policies choose among
candidate missions but do not yet choose arbitrary line geometry.

## Resource operations

The resource state machine is:

`AVAILABLE -> OUTBOUND -> WORKING/DROP/OBSERVE -> RETURNING -> QUEUED -> RELOADING -> AVAILABLE`

`WITHDRAWN` is terminal for a resource within the episode.

The model includes:

- dispatch latency and continuous travel interpolation;
- work accrued over multiple minutes;
- payload and volume-specific drop execution;
- finite endurance/duty clock and withdrawal;
- shared reload-bay capacity and first-available queuing;
- aviation wind gates;
- direct crew-attack intensity gates; and
- explicit attempts, accepted tasks, block reasons, cost, and exposure.

The action mask contains operational gates, so a learned actor is not trained
to select a mission that the simulator will always reject. The exact assignment
and heuristic comparators use the same mask.

## Two-perimeter arrival-history reconstruction

### Coupled-state problem

A single observed perimeter does not determine the state needed by a coupled
fire model. Marking the interior burned and the boundary newly flaming assigns
the whole fire an artificial common ignition time. In an atmosphere-coupled
model this produces the wrong heat-flux history and can create an artificial
secondary atmospheric response.

The v0.5 initializer follows the state logic of WRF-SFIRE perimeter replay:
use an earlier perimeter and the forecast-start perimeter to reconstruct a
causal fire-arrival field, then derive the remaining coupled state before free
forecast.

### Constrained harmonic reconstruction

Let the earlier observation be at `t=-Delta` and the forecast-start perimeter
at `t=0`. Non-nested earlier pixels are reported and excluded. Arrival time is
fixed on the two observed fronts and a discrete harmonic field is solved in
the growth band:

`Laplacian(T) = 0`

with Dirichlet values `T=-Delta` at the earlier interface and `T=0` at the
later front. Earlier interior cells receive older arrival times using distance
to the earlier boundary and a robust annular speed. This differs from the
biharmonic spline used in the cited WRF-SFIRE implementation, while solving the
same coupled-state initialization problem with explicit nested-perimeter and
causality constraints.

The arrival field supplies:

- burn age `max(-T, 0)`;
- fuel remaining from the fire residence curve;
- active versus burned phase;
- residual heat-flux memory;
- local recent spread speed `1 / |grad(T)|`;
- outward head vector `grad(T) / |grad(T)|`; and
- a correction-confidence field.

At a rasterized outer edge, the observed normal displacement between the two
fronts is used for speed because a one-sided gradient next to a constant
exterior is biased.

### Advancing-front localization

The recent velocity field is projected from the forecast-start perimeter and
tapered with a Gaussian distance kernel over eight cells. Its temporal
influence decays with a 180 minute half-life. Rate correction is regularized to
the interval `[0.55, 3.0]`; an observed coincident edge therefore cannot stop
the physical forecast.

Sequential perimeter assimilation has a separate signed-distance innovation:

`delta_phi = gain * localization * clip(phi_obs - phi_forecast)`

The increment is zero beyond three localization radii. Only cells whose
corrected level-set sign changes are advanced or retracted. Far-field fuels,
weather, and burned interior are not rewritten.

## Incident forcing and correction fields

`WeatherForcing` now supports aligned `(time, y, x)` fields for:

- wind speed and meteorological direction;
- temperature, relative humidity, and precipitation;
- 1 h, 10 h, and 100 h dead-fuel moisture;
- live herbaceous and live woody moisture; and
- coupled-model `u` and `v` wind corrections.

The CF-NetCDF writer and reader round-trip these fields. Wind corrections are
applied in Cartesian components before conversion back to speed/direction.

`analyze_incident_forcing` fuses a gridded background with projected
RAWS-like station observations using Gaussian optimum interpolation:

`x_a = x_b + B H^T (H B H^T + R)^-1 (y - H x_b)`

Wind is analyzed as `u/v`; circular directions are never arithmetically
averaged. Temperature, humidity, and fuel-moisture innovations use the same
spatial covariance. The analysis returns posterior normalized standard
deviations for ensemble perturbations and records station/time diagnostics.

The contract accepts incident-grade observations. The repository does not
silently label NASA POWER forcing as incident-grade: the frozen historical
bundles still use coarse MERRA-2-derived NASA POWER weather and are reported as
such.

## Evaluation

### Historical paired hindcast

The frozen study uses six NIROPS incidents and 24 held-out perimeter
transitions. Each history forecast uses the perimeter immediately before its
forecast-start perimeter. No target perimeter is used during initialization.
The comparison reuses the v0.4 per-incident scalar spread adjustment and
forecast intervals, isolating the initialization/correction change.

Incident-cluster bootstrap intervals resample fires, not individual perimeter
transitions.

| Metric | v0.4 single perimeter | v0.5 two perimeter | Paired delta (95% CI) |
|---|---:|---:|---:|
| perimeter IoU | 0.807 | 0.832 | +0.024 (+0.004, +0.051) |
| mean boundary distance | 299 m | 224 m | -75 m (-201, -4) |
| 95th-percentile Hausdorff | 1123 m | 840 m | -283 m (-730, -5) |
| growth IoU | 0.042 | 0.031 | -0.011 (-0.018, -0.005) |
| one-cell growth F1 | 0.124 | 0.139 | +0.014 (-0.044, +0.112) |

The initializer improves cumulative location and boundary displacement on these
incidents. It remains conservative on exact new-growth overlap. The
tolerance-based growth result is directionally positive but uncertain. This is
evidence for the coupled-state correction and evidence that future spread-rate
conditioning still needs incident wind/moisture and uncertainty assimilation.

### Suppression mechanism experiment

The suppression experiment uses 8 matched landscape/fire seeds, 3 wind regimes
(2, 6, and 10 m/s), and 3 strategies for 72 total 180 minute trials. All
strategies see the same initial truth seed:

- uncontrolled;
- aerial-only exact task assignment; and
- integrated aerial, crew, and dozer assignment with line reinforcement.

Cluster bootstrap intervals resample seeds across wind regimes.

| Strategy | Mean weighted loss | Mean burned fraction | Escape rate |
|---|---:|---:|---:|
| uncontrolled | 196.1 | 8.30% | 66.7% |
| aerial only | 166.5 | 6.46% | 45.8% |
| integrated | 141.2 | 4.72% | 33.3% |

Against uncontrolled, aerial-only reduced mean loss by 15.1% (paired absolute
delta -29.5, 95% CI -43.7 to -14.2). Integrated operations reduced loss by
28.0% (delta -54.8, 95% CI -74.7 to -36.7) and burned fraction by 43.1%.

These are internal mechanism-validation results, not real-world effect
estimates. The integrated trials average 2.9 completed lines, 6.6 held line
cells, 9.3 breached line cells, and 12.5 reload-queue entries. The nonzero
breach rate is intentional and shows that line is not treated as a perfect
barrier.

## Verification

The frozen version passes 51 automated tests, including:

- analytic radial two-perimeter speed/direction recovery;
- arrival causality, burn age, heat memory, and localization;
- payload-volume conservation and treatment half-life;
- explicit line engagement state;
- station innovation localization and posterior uncertainty;
- band-limited signed-distance correction;
- multi-minute crew construction;
- policy information-boundary checks;
- NumPy/tensor behavior parity and front numerics; and
- PettingZoo and training tensor contracts.

`ruff check src tests tools` passes. The full results, CSV trials, arrival
examples, replay, 2D/3D frames, and MP4 are frozen below
`results/frontier_operations_final`.

## Remaining validity gaps

1. Coverage-response and line-capacity coefficients are mechanism priors, not
   calibrated incident parameters.
2. Historical incident bundles lack complete drop, crew, line, objective, and
   decision logs; historical suppression policy value is therefore not
   identifiable.
3. NASA POWER weather does not resolve incident terrain winds or RAWS
   observations.
4. The fast kernel carries heat-flux memory but does not solve atmospheric
   feedback. WRF-SFIRE/QUIC-Fire replay remains the correction target.
5. Line width below cell resolution is represented as a subgrid strength on a
   raster cell.
6. Crew safety zones, escape routes, shift rules, fatigue, aircraft
   deconfliction, maintenance, and dispatch governance remain incomplete.
7. The doctrine comparator chooses from generated geometries. Continuous
   geometry/control and model-predictive baselines remain future work.

## Primary references

- Kochanski et al., WRF-SFIRE perimeter assimilation and fire replay:
  <https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2023.1203578/full>
- USFS generalized wildfire containment algorithm:
  <https://research.fs.usda.gov/download/treesearch/69196.pdf>
- USFS operational fireline production estimates:
  <https://research.fs.usda.gov/treesearch/44803>
- USFS stochastic fireline production:
  <https://research.fs.usda.gov/treesearch/47358>
- USFS aerial retardant coverage guidance:
  <https://www.fs.usda.gov/t-d/programs/wfcs/pubs/htmlpubs/htm01572808/index.htm>
- USFS retardant effectiveness research:
  <https://research.fs.usda.gov/treesearch/80803>
- USFS Fireline Effectiveness dashboard:
  <https://research.fs.usda.gov/rmrs/products/dataandtools/fireline-effectiveness-fle-dashboard>
- USFS NFDRS and RAWS fuel-moisture system:
  <https://research.fs.usda.gov/firelab/projects/firedangerrating>
- USFS FlamMap fuel conditioning and WindNinja workflow:
  <https://research.fs.usda.gov/firelab/projects/flammap>
- Level-set ensemble Kalman fire-front assimilation:
  <https://publications.iafss.org/publications/fss/11/1443>
- WRF-Fire artificial fire history and moisture estimation:
  <https://arxiv.org/abs/1203.2230>
