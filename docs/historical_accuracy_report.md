# Historical accuracy study

## Corrected coupled-state hindcasts

**Study version:** v3  
**Completed:** 2026-07-29  
**Scope:** six western U.S. incidents, 24 held-out daily transitions  
**Forecasts:** 168 deterministic or ensemble-threshold forecasts  

## Abstract

This study evaluates the current Aeolus-IA fire model against successive
analyst-interpreted NIROPS perimeters. It also evaluates whether causal
two-perimeter arrival-history reconstruction improves the next forecast. The
study corrects four implementation problems found during the audit: historical
weather was indexed from episode-relative time, dead-fuel moisture had no
weather spin-up, FBFM40 loading was applied twice to spread rate, and a
coarse-grid front could be declared contained before crossing into the next
cell.

Arrival-history initialization improves all three physics families in aggregate.
For the calibrated deterministic model, mean cumulative IoU rises by 0.0168
(incident-cluster bootstrap 95% interval 0.0045 to 0.0327) and mean symmetric
boundary displacement falls by 67.8 m (95% interval 13.5 to 153.4 m lower).
The history-conditioned ensemble is the strongest physics forecast: cumulative
IoU 0.7458, active-growth one-cell-tolerance F1 0.1776, and mean symmetric
boundary displacement 396.3 m.

Persistence remains materially stronger on cumulative extent: IoU 0.8732 and
mean symmetric boundary displacement 156.5 m. The history ensemble has positive
probabilistic skill on the active domain but negative ordinary Brier skill over
the whole grid. The current model therefore has useful advancing-front signal
but overpredicts growth across easy background cells. These results support
continued research use; they do not support operational prediction claims.

## Questions

The study addresses four questions:

1. Does the simulator produce nontrivial held-out advancing-front skill?
2. Does a causal estimate of the prior arrival-time field improve the next
   forecast?
3. Does incident-specific scalar calibration transfer beyond its calibration
   interval?
4. Are ensemble probabilities better calibrated than persistence?

It does not evaluate historical suppression effectiveness or learned policy
quality. The public perimeter observations do not contain sufficiently complete
time-indexed suppression actions, resource states, tactics, or airspace
constraints to identify those effects.

## Implementation changes

### Absolute forcing clock

Weather forcing now carries a parsed CF time origin. Every hindcast aligns
episode time zero with the timestamp of its initial observed perimeter before
sampling hourly forcing. Calibration, deterministic validation, ensemble
validation, shadow replay, and counterfactual replay use the same clock path.
The result artifact records the forcing origin and minute offset.

This corrects a consequential error in the archived run: later validation
intervals had replayed weather from the beginning of the incident bundle rather
than weather concurrent with the forecast interval.

### Dead-fuel moisture spin-up

Prepared incident weather now starts 336 hours before the first perimeter.
One-, ten-, and hundred-hour dead-fuel moisture are integrated through that
period with separate Van Wagner-Pickett drying and wetting equilibria, an exact
exponential lag update, a hysteresis deadband, and a precipitation response.
The implementation is shared by data preparation and online model evolution.
Explicit moisture fields in a forcing bundle are not integrated a second time.

This is consistent with the moisture-state structure used by WRF-SFIRE, but the
forcing remains one NASA POWER point per incident. It does not resolve spatial
moisture variation, terrain wind, canopy interception, or fuel-specific
equilibrium corrections.

### FBFM40 loading

The fire-behavior lookup contains Pyretechnics Rothermel behavior by FBFM40
fuel model. The importer now obtains total oven-dry loading and bed depth from
the same FBFM40 definitions. A second family-level fuel-load multiplier was
removed from rate of spread. Fuel loading remains in the state for burnability,
consumption, and heat accounting; it no longer double-scales an already
fuel-model-specific spread lookup.

### Coarse-grid advancing-front lifecycle

The front solver now keeps a burning cell active while it supports an interface
adjacent to unburned fuel. Residence-time expiry cannot retire that support cell
before the continuous level set crosses the next cell boundary. A regression
test exercises a 130 m class cell at a low nominal spread rate.

The diagnostic v2 study exposed this defect: several forecasts stopped at
360 minutes with no raster-neighbor ignition even though the subcell front was
still advancing. All v3 physics forecasts reached their requested observation
time, with integration overshoot between zero and two minutes.

### Causal arrival-history reconstruction

For a forecast beginning at perimeter \(P_t\), the history variant uses only
\(P_{t-1}\), \(P_t\), and their timestamps. It estimates an arrival-time field
over the newly observed band and reconstructs fire age, level-set distance,
fuel remaining, and recently active front state at \(t\). The future target
\(P_{t+1}\) is never used during initialization. Every validation history
forecast passed a causality audit.

This addresses the coupled-state initialization problem identified in
WRF-SFIRE perimeter-assimilation research: inserting a geometric perimeter
without reconstructing compatible fuel and heat history leaves the atmospheric
and fire state dynamically inconsistent.

## Data

The observation reference is the 2026 NIROPS progression dataset by Magstadt
et al., containing analyst-interpreted airborne-infrared observations for
western U.S. fires from 2020 through 2024. The benchmark uses Electra,
Crockets Knob, Dry Lake, Ridge Creek, Davis, and Bear.

| Source | Variables | Use | Current limitation |
|---|---|---|---|
| NIROPS | time-indexed interpreted perimeters | initialization and target | daily cadence; acquisition and interpretation uncertainty not modeled |
| USGS 3DEP | elevation | slope and terrain grid | resolution is resampled to a 128-cell study grid |
| LANDFIRE 2025 | FBFM40, canopy fields | fuel and canopy state | not contemporaneous for every incident; no dynamic curing |
| NASA POWER | hourly wind, temperature, RH, rain | atmosphere and moisture spin-up | coarse point forcing derived from large-scale analysis |

The prepared bundles are retained separately from the archived data. Weather
starts 14 days before each incident's first perimeter and ends after its final
validation target.

## Protocol

Each incident contributes one early calibration transition and four subsequent
held-out transitions. Splitting is by incident trajectory: a validation target
is never used to select that incident's spread multiplier or ensemble weights.
The four transitions from one incident remain statistically dependent, so
uncertainty intervals resample incident clusters rather than individual daily
forecasts.

All methods receive the same terrain, fuels, weather, initial perimeter, and
forecast duration:

| Method | Spread parameter | Initial state | Output |
|---|---|---|---|
| Persistence | none | current perimeter | unchanged extent |
| Raw | 1.0 | current perimeter | deterministic |
| Raw + history | 1.0 | previous and current perimeters | deterministic |
| Calibrated | selected on one prior transition | current perimeter | deterministic |
| Calibrated + history | selected on one prior transition | previous and current perimeters | deterministic |
| Ensemble | weighted on one prior transition | current perimeter | 12-particle probability |
| Ensemble + history | weighted on one prior transition | previous and current perimeters | 12-particle probability |

The calibration search spans effective spread adjustments from 0.0005 to 5.0.
The ensemble varies spread, wind exposure, wind-direction bias, and dead-fuel
moisture. Likelihood tempering targets an effective sample size of 4.2 out of
12 particles.

Metrics separate cumulative burned extent from newly observed growth:

- cumulative intersection over union (IoU);
- new-growth IoU;
- advancing-front F1 after one-cell spatial tolerance;
- mean symmetric boundary displacement;
- 95th-percentile Hausdorff distance;
- symmetric-difference area and area bias; and
- ordinary, balanced, and active-domain Brier scores for ensembles.

## Aggregate results

Values are means over 24 held-out transitions. Intervals in the figures and
machine-readable analysis are 95% incident-cluster bootstrap intervals.

| Method | Cumulative IoU | Growth F1, all | Growth F1, active only | Boundary distance (m) | Symmetric difference (km2) |
|---|---:|---:|---:|---:|---:|
| Persistence | **0.8732** | 0.1250 | 0.0000 | **156.5** | **4.409** |
| Raw | 0.7274 | 0.1318 | 0.1506 | 492.9 | 14.183 |
| Raw + history | 0.7398 | 0.1417 | 0.1620 | 454.0 | 12.316 |
| Calibrated | 0.5930 | 0.1238 | 0.1415 | 950.9 | 29.412 |
| Calibrated + history | 0.6098 | 0.1274 | 0.1456 | 883.1 | 25.456 |
| Ensemble | 0.7335 | 0.1531 | 0.1749 | 424.2 | 10.753 |
| Ensemble + history | **0.7458** | **0.1554** | **0.1776** | **396.3** | **10.054** |

Bold within the physics rows marks the strongest physics result. Persistence is
shown separately because it predicts no new growth and exploits the high
overlap of consecutive daily cumulative perimeters.

The calibrated deterministic model is worse than the raw model. Scalar
calibration overfits the first transition's unresolved combination of weather,
suppression, fuels, moisture, and observation timing. Bear selects the upper
search boundary of 5.0; the other selected multipliers range from 0.987 to 2.0.
This is evidence against treating an incident-specific spread scalar as a
transferable physical parameter.

## Arrival-history ablation

Positive IoU and F1 deltas are improvements. Negative distance and area deltas
are improvements.

| Baseline to history variant | Delta IoU | Delta growth F1 | Delta boundary (m) | Delta symmetric difference (km2) |
|---|---:|---:|---:|---:|
| Raw | +0.0123 | +0.0099 | -38.9 | -1.867 |
| Calibrated | +0.0168 | +0.0036 | -67.8 | -3.956 |
| Ensemble | +0.0123 | +0.0023 | -27.9 | -0.698 |

The calibrated history improvement is the clearest: its cluster interval
excludes zero for cumulative IoU, boundary displacement, Hausdorff distance,
and symmetric-difference area. The raw and ensemble deltas have the same
direction in aggregate but wider incident-level uncertainty. Arrival history
is therefore retained as the default research initialization, while the
geometric-only initializer remains available as an ablation.

## Probabilistic results

Brier skill is \(1 - S_{\mathrm{ensemble}}/S_{\mathrm{persistence}}\), evaluated
on the 21 transitions with observed new growth.

| Ensemble | Whole-domain ordinary | Whole-domain balanced | Active-domain ordinary | Active-domain balanced |
|---|---:|---:|---:|---:|
| Calibrated | -102.8% | +18.4% | +14.1% | +22.3% |
| Calibrated + history | -93.8% | +18.9% | +14.3% | +21.4% |

The contrast is important. The ensemble discriminates portions of the active
front better than persistence, but assigns too much probability to growth
across the large easy-negative background. Operational probability products
would require posterior calibration, spatial reliability analysis, and
out-of-incident validation.

## Incident heterogeneity

The history ensemble's mean cumulative IoU ranges from 0.628 on Dry Lake and
Crockets Knob to 0.932 on Ridge Creek. Its mean growth F1 ranges from 0.036 on
Electra to 0.306 on Crockets Knob. Electra is the clearest false-growth failure:
persistence IoU is 0.970 while the history ensemble is 0.718 and has +11.1 km2
mean area bias. Ridge Creek is nearly static at this grid and cadence, so both
persistence and the ensemble score highly on cumulative extent while front F1
remains low.

This heterogeneity is not noise that should be averaged away. It identifies
the major confounding structure: unresolved suppression and local wind can
make the same nominal fuel/weather model overgrow one incident and undergrow
another.

## Current validity boundary

The v3 study establishes:

- causal historical initialization and forcing alignment;
- numerically complete daily integrations;
- measurable, repeatable held-out advancing-front signal;
- a beneficial aggregate effect from two-perimeter state reconstruction; and
- incident-cluster uncertainty for deterministic and paired comparisons.

It does not establish:

- operationally accurate incident spread;
- superiority to persistence on cumulative perimeter or boundary location;
- accuracy under plume-dominated, extreme-wind, crown-fire, or long-range
  spotting regimes;
- historical suppression attribution;
- regional or year-held-out generalization;
- calibrated posterior fire probabilities; or
- policy transfer from simulation to incident operations.

## Limits and closure experiments

### 2.1 Historical forcing and observations

**Wind.** NASA POWER point forcing cannot represent terrain-channelled flow,
frontal passage timing, convective outflow, vertical shear, or fire-induced
wind. Replace it with HRRR analysis fields downscaled to the incident grid,
then assimilate quality-controlled RAWS observations as innovations. Preserve
the uncorrected HRRR field, station observations, correction field, and
uncertainty separately. Evaluate wind direction and speed before evaluating
spread.

**Moisture.** The new dead-fuel model has appropriate state memory but is
spatially homogeneous because its forcing is a point series. Drive it with
gridded temperature, humidity, precipitation, radiation, and canopy exposure.
Add live-fuel moisture and herbaceous curing from dated remote-sensing or fuel
sampling products. Validate moisture at station/fuel-stick sites independently
of perimeter fit.

**Perimeters.** Daily NIROPS outlines alias sub-daily wind and suppression
effects. Add GOFER or FEDS hourly progressions and model observation footprints,
cloud, geolocation, acquisition interval, and analyst uncertainty. Use those
products as observations with likelihoods, not as exact truth rasters.

**Suppression confounding.** Acquire time-indexed fireline, aerial-drop, firing,
resource, and engagement records where available. Mask or jointly infer
suppression-affected front segments. A free-spread model cannot be expected to
match an actively suppressed incident without representing the actions.

Exit criterion: at least 30 incidents, incident/geography/year-held-out splits,
1/3/6/12/24-hour targets, and positive advancing-front skill over persistence
with cluster intervals excluding zero.

### 2.2 Coupled fire state and behavior

**State assimilation.** Extend the two-perimeter reconstruction into a
sequential localized ensemble filter over arrival time, spread correction,
wind correction, moisture, and spotting state. Register or morph fronts before
amplitude updates. Evaluate improvement on the next observation, not fit to the
assimilated observation.

**Atmospheric coupling.** Use WRF-SFIRE and QUIC-Fire ensembles as teacher and
stress-test systems. Learn bounded residual correction fields for terrain wind,
fire-induced flow, rate of spread, and spotting uncertainty. Reject or widen
uncertainty in unsupported plume-dominated conditions.

**Fuel dynamics.** The exact FBFM40 mass/depth correction removes a structural
error, but lookup behavior still uses fixed reference live moisture and curing.
Add dynamic herbaceous curing, live woody moisture, canopy foliar moisture,
crown transition calibration, and dated disturbance treatments.

**Spotting.** The benchmark disables spotting because there is no calibrated
incident-scale ember model. Implement transport, lofting, landing, ignition
delay, and receptive-fuel probability, then validate spot-fire distance and
timing distributions separately before enabling it in historical scores.

Exit criterion: error reductions against independent wind, moisture, spread,
and spotting observations, plus stable held-out skill across fuel and weather
regimes.

### 2.3 Statistical validation and training implications

**Sample size.** Six incidents are enough to expose implementation defects but
not to estimate broad generalization. Expand by geography, year, fuel family,
wind regime, topography, incident size, and observation source. Keep all
transitions from one incident in one split.

**Calibration.** Replace a single fitted multiplier with hierarchical priors
and sequential state estimation. Calibration parameters must have physical
meaning, posterior uncertainty, identifiability checks, and out-of-incident
transfer tests. A boundary-selected parameter is a failed calibration, not a
successful fit.

**Probability calibration.** Report reliability curves, expected calibration
error, log score, Brier decomposition, and CRPS by distance-to-front and
horizon. Calibrate on incidents disjoint from evaluation incidents. Include
coverage tests for arrival-time intervals.

**RL domain distribution.** Do not train policies against one fitted
deterministic fire. Sample from the empirically supported posterior over
weather corrections, moisture, spread, spotting, observation error, and
suppression response. Freeze an incident-, region-, and year-held-out
evaluation suite. Require policy rankings to remain stable in higher-fidelity
teacher replays.

Exit criterion: a preregistered, immutable evaluation manifest; independent
calibration/evaluation incidents; at least five policy training seeds; and
paired evaluation against doctrine, greedy, optimization, and no-action
baselines.

The broader compute-bound and non-compute-bound program is maintained in
[Limits and solution program](limits_and_solutions.md).

## Reproduction

Refresh only the weather/moisture portion of already prepared incident bundles:

```bash
aeolus-study refresh-weather \
  --manifest configs/historical_validation.yaml \
  --prepared-root /path/to/original/incidents \
  --out /path/to/refreshed/incidents
```

Run the corrected study:

```bash
aeolus-study run \
  --manifest configs/historical_validation.yaml \
  --prepared-root /path/to/refreshed/incidents \
  --out results/frontier_fire/historical_validation_v3
```

Build the derived statistics and figures:

```bash
python tools/analyze_historical_accuracy.py \
  --results results/frontier_fire/historical_validation_v3/historical_validation_results.json \
  --examples results/frontier_fire/historical_validation_v3/historical_validation_examples.npz \
  --prepared-root /path/to/refreshed/incidents \
  --out results/frontier_fire/historical_validation_v3/analysis
```

Primary artifacts:

- `results/frontier_fire/historical_validation_v3/historical_validation_results.json`
- `results/frontier_fire/historical_validation_v3/historical_validation_examples.npz`
- `results/frontier_fire/historical_validation_v3/analysis/historical_accuracy_analysis.json`
- `results/frontier_fire/historical_validation_v3/analysis/historical_accuracy_intervals.csv`

## References

1. Magstadt et al. (2026), *A high spatial resolution daily fire perimeter
   progression dataset for wildfires in the Western United States: 2020-2024*.
   <https://doi.org/10.17632/95rj5d379g.1>
2. Kochanski et al. (2023), *Analysis of methods for assimilating fire
   perimeters into a coupled fire-atmosphere model*.
   <https://doi.org/10.3389/ffgc.2023.1203578>
3. Mandel et al. (2011), *Coupled atmosphere-wildland fire modeling with
   WRF-Fire version 3.3*. <https://arxiv.org/abs/1208.1059>
4. OpenWFM, *Fuel moisture model*.
   <https://wiki.openwfm.org/wiki/Fuel_moisture_model>
5. Liu et al. (2024), *Systematically tracking the hourly progression of large
   wildfires using GOES satellite observations*.
   <https://doi.org/10.5194/essd-16-1395-2024>
6. NASA POWER, *Hourly API documentation*.
   <https://power.larc.nasa.gov/docs/services/api/temporal/hourly/>
7. NOAA, *High-Resolution Rapid Refresh (HRRR)*.
   <https://rapidrefresh.noaa.gov/hrrr/>
8. LANDFIRE, *Landscape fire and resource management planning tools*.
   <https://landfire.gov/>
