# Historical fidelity iteration v4

**Implementation date:** 2026-07-30  
**Scope:** non-compute-bound accuracy constraints in forcing, fuel state,
observations, suppression context, and benchmark design  
**Status:** implementation and audit complete; one-incident paired ablation
complete; expanded multi-incident evaluation not yet complete

## Purpose

The v3 study showed advancing-front signal but did not beat persistence on
cumulative extent or boundary location. Its principal uncontrolled errors were
incident wind, spatial moisture, fixed live fuels, daily observation cadence,
unmodeled perimeter uncertainty, unobserved suppression, and six-incident
sampling. This iteration changes the representation and evaluation paths needed
to test those errors. It does not treat a more detailed input as evidence of
better forecasts until a held-out comparison is complete.

## Implemented changes

### Incident atmospheric analysis

`aeolus.data.hrrr` reads the public NOAA HRRR analysis Zarr archive by native
tile. It maps scenario cell centers to the approximately 3-km HRRR grid and
extracts 10-m U/V wind, 2-m temperature/relative humidity, and surface
precipitation rate.

The importer caches each hourly source tile, records source-index bounds, and
requires a declared minimum analysis coverage. Missing hours are retained as
provenance and filled only by temporal interpolation between available
analyses. A 60-day NASA POWER series remains the spin-up background; HRRR
replaces coincident incident-period fields.

The Electra extraction requested 148 hours and obtained 147 direct analyses,
for 99.32% coverage. The missing analysis is
`2022-07-06T19:00:00+00:00`. Moisture integration uses 1,608 hourly spin-up
and incident samples on the 128×128 fire grid. The stored rollout forcing
retains 149 bracketing incident-window samples; the 1,442 discarded spin-up
samples remain recorded in provenance.

Over the incident period, the HRRR spatial-mean wind differs from the prior
NASA POWER point forcing by a mean absolute 0.727 m/s in speed and 47.53° in
direction. HRRR's mean within-domain 10th-to-90th-percentile wind-speed range
is 1.000 m/s, with a maximum of 2.535 m/s. These are differences between
forcing products, not wind-error estimates, because this iteration does not
yet include an independent station reference.

This is an incident-scale mesoscale analysis. Nearest-native-cell sampling does
not create terrain flow below the HRRR grid and does not assimilate local RAWS
observations. The archive and model characteristics are documented by the
[NOAA HRRR open-data registry](https://registry.opendata.aws/noaa-hrrr-pds/)
and the [HRRR program](https://rapidrefresh.noaa.gov/hrrr/).

### Spatial thermodynamic forcing and fuel moisture

Point or coarse atmospheric forcing is projected to the scenario terrain with
a declared temperature lapse rate. Relative humidity is recomputed at local
temperature while conserving background vapor pressure. Wind and precipitation
are not orographically modified.

Dead 1-, 10-, and 100-hour fuels are integrated separately at every grid cell
with the existing WRF-SFIRE-compatible equilibrium time-lag equations. In the
Electra forcing case, topographic conditioning produces a final-time spatial
standard deviation of 0.00248 kg/kg in 1-hour dead-fuel moisture.

Live fuel uses the NFDRS version 4 growing-season-index construction:

- minimum-temperature ramp from −2 to 5 °C;
- inverted maximum-VPD ramp from 900 to 4,100 Pa;
- photoperiod ramp from 36,000 to 39,600 seconds;
- 28-day precipitation ramp from 0 to 10 mm; and
- a 28-day trailing mean of the daily limiting-factor product.

The smoothed index maps to documented live-moisture ranges of 0.30–2.50 kg/kg
for herbaceous fuel and 0.60–2.00 kg/kg for woody fuel. Electra's derived
herbaceous series spans 0.30–0.774 kg/kg during the retained forcing period.
The equations and ranges follow the
[NFDRS v4 technical documentation](https://research.fs.usda.gov/download/treesearch/68223.pdf).

### Dynamic live-fuel behavior

The packaged Pyretechnics table previously fixed live herbaceous moisture at
0.75 and live woody moisture at 0.60 kg/kg. It now evaluates explicit axes for
dead 1-hour, live herbaceous, live woody, wind, and slope state. NumPy and
PyTorch use the same multilinear interpolation.

At dead moisture 0.07, wind 4 m/s, and slope 0.2:

| Fuel model | Live herb 0.30 | 0.75 | 1.20 | 2.50 kg/kg |
|---|---:|---:|---:|---:|
| FBFM1, static grass | 12.140 | 12.140 | 12.140 | 12.140 |
| FBFM102, dynamic grass | 8.123 | 4.040 | 0.060 | 0.032 |
| FBFM122, dynamic grass-shrub | 6.119 | 4.932 | 0.893 | 0.628 |

Values are spread rate in m/min from the packaged table. The dynamic response
is the Scott-Burgan herbaceous load transfer implemented by Pyretechnics and
documented in the
[NWCG live-fuel guidance](https://www.nwcg.gov/publications/pms437/fuel-moisture/live-fuel-moisture-content).

### Observation representation

Source perimeter features are preserved. The evaluation series now unions
features with an identical timestamp before rasterization is treated as a time
step. In the FEDS case, 38 source features become 21 nonempty unique frames;
17 duplicate-bin fragments are coalesced.

The observation module adds signed-distance geometry, Gaussian localization
sensitivity, probabilistic occupancy targets, soft scores, boundary-envelope
coverage, and interval-censored arrival loss.

Sigma values are not presented as NIROPS error estimates. They are sensitivity
parameters because the source release does not provide incident-specific
localization variances. Across the existing 24 transitions, predicted-boundary
envelope coverage rises from 0.398 under exact-cell matching to 0.756 at a
declared 350-m sigma. Soft IoU changes from 0.687 to 0.627 because the target
itself becomes a diffuse occupancy field; the two scores answer different
questions.

The construction is consistent with published observation models that
represent geolocation and detection likelihood rather than exact burning
pixels, including [Haley et al.](https://arxiv.org/abs/1808.03318).

### Cadence

The cadence audit contains 12,706 NIROPS source features from 737 incidents:

| Scope | Median interval | 90th percentile | Intervals over 36 h |
|---|---:|---:|---:|
| all NIROPS incidents | 24.43 h | 69.93 h | 23.89% |
| current six incidents | 24.00 h | 40.43 h | 12.86% |
| FEDS case after coalescing | 12.00 h | 36.00 h | 5.00% |

FEDS improves nominal cadence but remains a VIIRS-derived modeled perimeter
product at 375-m nominal resolution and 12-hour local-solar bins. It is a
separate observation model, not an independent high-resolution replacement
for NIROPS. Product semantics are described by
[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/descriptions/FEDS_VIIRS_SNPP.html).

### Unobserved suppression

The Fireline Engagement archive is joined to NIROPS by normalized IRWIN ID,
not incident name. The six existing study incidents have 4,172 archived line
features; 175 contain a usable `LineDateTime`.

At a 300-m context buffer, 33.38% of false-positive growth from the existing
history ensemble lies near any archived line. This is retrospective spatial
association. Restricting context to a line with a timestamp no later than the
forecast target yields 2.03%. The difference shows why the archive cannot be
used as a complete causal suppression trace.

The audit separately records Held, Burned Over, and Not Engaged outcomes. It
does not infer line completion time, aerial drops, firing operations, resource
availability, or treatment effectiveness. The source's intended use and
limitations are described by the
[USFS Fireline Effectiveness project](https://research.fs.usda.gov/rmrs/products/dataandtools/fireline-effectiveness-fle-dashboard).

### Expanded incident benchmark

The selector audits all 737 NIROPS incident series. Eligibility requires five
consecutive adjacent transitions between 6 and 36 hours, at least 100 maximum
reported acres, and at least 0.75 monotonic area fraction.

There are 386 eligible incidents. The frozen expanded manifest selects 36 by
round-robin strata over state and quartile of log maximum reported acres. It
spans Arizona, California, Colorado, Idaho, Montana, New Mexico, Nevada,
Oregon, and Utah, and all source years from 2020 through 2024.

This is a larger development benchmark, not yet a complete generalization
design. Fuel-family, ecoregion, wind-regime, observation-quality, and final
train/development/test balance still require explicit frozen splits.

## Paired Electra ablation

The complete v3 study output and the v4 Electra output have 28 common forecast
records: seven methods over four held-out perimeter transitions. The
calibration pair is unchanged. The fitted spread multiplier changes from 2.0
to 1.5. Candidate forcing combines HRRR analysis, terrain-conditioned
thermodynamics, spatial dead-fuel moisture, and dynamic live-fuel moisture, so
this comparison does not attribute effects to individual components.

| Method | Cumulative IoU v3 → v4 | Boundary distance v3 → v4 | Advancing-front F1 v3 → v4 |
|---|---:|---:|---:|
| raw physics | 0.818 → 0.743 | 326 → 474 m | 0.056 → 0.035 |
| history raw physics | 0.817 → 0.747 | 360 → 469 m | 0.055 → 0.032 |
| calibrated physics | 0.487 → 0.528 | 1,090 → 1,165 m | 0.067 → 0.080 |
| history calibrated physics | 0.511 → 0.534 | 1,048 → 1,116 m | 0.071 → 0.073 |
| calibrated ensemble | 0.711 → 0.847 | 560 → 244 m | 0.067 → 0.024 |
| history calibrated ensemble | 0.718 → 0.844 | 546 → 240 m | 0.073 → 0.024 |
| persistence | 0.970 → 0.970 | 28 → 28 m | 0.000 → 0.000 |

Cumulative IoU and boundary distance use all four transitions.
Advancing-front F1 is the one-cell-tolerant score over the two transitions
with observed growth; no-growth empty-set matches are excluded.

The combined forcing changes improve ensemble cumulative extent substantially:
the calibrated ensemble gains 0.135 IoU, reduces mean symmetric boundary
distance by 315 m, and reduces symmetric difference by 7.24 km². The history
ensemble is similar. Raw-physics cumulative scores deteriorate, and ensemble
advancing-front F1 falls sharply. Calibrated deterministic front F1 improves
slightly while its boundary distance remains poor.

For the two observed-growth transitions, ensemble active-domain Brier score
improves from 0.332 to 0.318 and is below the 0.354 persistence score. Across
all four transitions, including two with no observed growth, its Brier score
is 0.198 versus 0.177 for persistence. The posterior therefore gains
probabilistic skill in active-growth intervals while placing too much growth
probability in the two no-growth intervals.

This is a useful failure localization rather than an accuracy claim.
Persistence remains decisively stronger on cumulative extent. The forcing
changes improve ensemble boundary placement but do not solve ignition/front
localization, no-growth discrimination, or same-incident calibration
dependence. Four transitions from one incident cannot establish
generalization.

## Artifacts

- `configs/historical_validation_expanded.yaml`
- `configs/historical_validation_electra_fidelity.yaml`
- `results/frontier_fire/historical_benchmark_inventory.csv`
- `results/frontier_fire/historical_benchmark_inventory.json`
- `results/frontier_fire/historical_validation_v4_context/observation_cadence.json`
- `results/frontier_fire/historical_validation_v4_context/observation_uncertainty.json`
- `results/frontier_fire/historical_validation_v4_context/observation_uncertainty.csv`
- `results/frontier_fire/historical_validation_v4_context/suppression_confounding.json`
- `results/frontier_fire/historical_validation_v4_context/electra_forcing_fidelity.png`
- `results/frontier_fire/historical_validation_v4_context/electra_forcing_fidelity.json`
- `results/frontier_fire/historical_validation_v4_context/historical_fidelity_context.png`
- `results/frontier_fire/historical_validation_v4_electra/historical_validation_results.json`
- `results/frontier_fire/historical_validation_v4_electra/electra_fidelity_comparison.json`
- `results/frontier_fire/historical_validation_v4_electra/electra_fidelity_comparison.png`

## Reproduction

Install meteorological and geospatial dependencies:

```bash
python -m pip install -e '.[geo,met,dev]'
```

Build the expanded manifest:

```bash
python tools/select_nirops_benchmark.py \
  /path/to/NIROPS_2020_2024_R1_R6.shp \
  configs/historical_validation_expanded.yaml \
  results/frontier_fire/historical_benchmark_inventory.csv
```

Refresh prepared forcing and run a study:

```bash
aeolus-study refresh-weather \
  --manifest configs/historical_validation.yaml \
  --prepared-root /path/to/prepared/incidents \
  --out /path/to/refreshed/incidents

aeolus-study run \
  --manifest configs/historical_validation.yaml \
  --prepared-root /path/to/refreshed/incidents \
  --out results/frontier_fire/historical_validation_v4
```

The refresh caches HRRR tiles below each incident's `provenance/hrrr-cache`
directory. Preserve that cache for repeatability and to avoid unnecessary
archive traffic.

Compare a candidate study with the same incident in an earlier study:

```bash
python tools/compare_historical_fidelity.py \
  results/frontier_fire/historical_validation_v3/historical_validation_results.json \
  results/frontier_fire/historical_validation_v4_electra/historical_validation_results.json \
  CA-AEU-017769_Electra \
  results/frontier_fire/historical_validation_v4_electra/electra_fidelity_comparison.json \
  --figure results/frontier_fire/historical_validation_v4_electra/electra_fidelity_comparison.png
```

## Remaining validity gates

The main unresolved non-compute-bound items are:

1. quality-controlled RAWS ingestion and wind validation before fire fitting;
2. terrain-flow diagnostics or coupled correction fields below the HRRR grid;
3. observed live/dead fuel-moisture validation and dated fuel/disturbance maps;
4. cloud, missed-detection, acquisition-window, and analyst-interpretation
   components in the perimeter likelihood;
5. time-complete line, firing, drop, and resource records for a subset of
   incidents;
6. frozen geography/year/fuel/weather holdouts in the 36-incident benchmark;
7. comparison with an established external spread model under identical
   forcing and initialization; and
8. positive held-out advancing-front skill over persistence with
   incident-cluster uncertainty.

Until those gates are met, the correct claim remains a research simulator with
historical hindcast infrastructure, not an incident prediction system.
