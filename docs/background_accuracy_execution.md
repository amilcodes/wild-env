# Background historical-accuracy execution program

Date: 2026-08-04

This document controls the work that can proceed while the frozen
36-incident benchmark runs. Its purpose is to improve the next experiment
without changing the code, data contract, or parameter candidates used by the
active experiment.

## Active benchmark

The chronological partition contains 22 training incidents from 2020–2021,
seven development incidents from 2022, and seven untouched test incidents
from 2023–2024. The contract is
`configs/historical_validation_frozen_36.yaml`. The test incidents cannot be
used for parameter fitting, method choice, exception design, or qualitative
selection.

The local execution has five resumable stages:

1. build metric-CRS incident bundles;
2. reconstruct the closest time-admissible fuel landscape available from the
   current public service;
3. materialize HRRR forecast fields available at the declared issue time;
4. fit and score the geometric baselines; and
5. fit the declared global physics candidates on training incidents, select
   on development incidents, and evaluate once on test incidents.

Live state is written atomically to
`results/frontier_fire/historical_validation_frozen_36_local/local_run_status.json`.
An hourly task heartbeat reads that file, counts completed incident bundles,
checks the recent stage log, and verifies the execution fingerprint. It stays
quiet while progress is healthy and unchanged.

The active execution fingerprint is
`results/frontier_fire/historical_validation_frozen_36_local/execution_source_manifest_v2_hrrr_gap_repair.json`.
It hashes 89 code, configuration, native-kernel, lookup-table, and dependency
inputs and records the complete installed Python distribution set. Run:

```bash
python tools/freeze_execution_source.py --verify \
  results/frontier_fire/historical_validation_frozen_36_local/execution_source_manifest_v2_hrrr_gap_repair.json
```

Any reported change invalidates continuation under the same execution claim.
Independent new modules and documents are allowed when they are outside the
recorded import and command closure.

### Declared execution revision: HRRR incomplete-field repair

The first execution revision failed deterministically while preparing the
Mahogany incident (`NV-HTF-500736_Mahogany`). The requested 172-hour interval
had 148 complete HRRR analysis groups. All 24 groups on 2 July 2020 lacked the
surface precipitation-rate array, although their wind, temperature, and
relative-humidity arrays were available. Treating each incomplete group as an
entirely absent weather hour produced 0.860 analysis coverage and failed the
predeclared 0.950 gate.

The revision retains every available analysis field and fills only a field
absent from the archive. It uses the newest preceding six-hour HRRR forecast
cycle that covers the complete contiguous gap, tries older eligible cycles
deterministically if necessary, and rejects the incident if repaired weather
coverage still falls below the existing gate. Mahogany precipitation uses the
1 July 2020 18:00 UTC cycle at leads F06--F29. Its bundle records:

- raw analysis coverage: 148/172, or 0.860465;
- usable weather coverage after repair: 172/172;
- repaired variable and valid timestamps;
- forecast reference time and every lead hour; and
- zero unresolved weather hours.

The first fingerprint is retained as a superseded record. Of the files in
that manifest, only `src/aeolus/data/hrrr.py` changed. Revision 2 also includes
the independent GOFER and progression adapters added during the background
research stream; neither is imported by the frozen benchmark commands. The
incident contract, split, targets, candidate set, and fitting rules are
unchanged. Mahogany belongs to the training split, and the repair decision was
made from source availability before any development or test score existed.
Unit tests verify that valid analysis fields survive, only the absent field is
substituted, the selected cycle precedes the gap, and raw and final coverage
remain distinct in provenance.

## Research decisions

### 1. Hourly progression observations

Daily or irregular NIROPS snapshots weakly constrain the diurnal cycle and
reward cumulative-area persistence. The first independent addition is a
version-locked adapter for GOFER v0.2. GOFER supplies hourly cumulative
perimeters, concurrent active-fire lines at six detection-confidence
thresholds, retrospective active-fire lines, and progression summaries for 28
California fires from 2019–2021.

The adapter retains each source hour as an acquisition window ending at
`tUTC`. It leaves `available_at` empty and marks every frame retrospective.
That prevents a reconstructed product from entering an operational forecast
as if it had been issued in real time. It also retains GOFER's published
population-level spatial validation statistics without converting a mean edge
error into a per-frame Gaussian standard deviation.

GOFER-Combined has a reported final-perimeter mean IoU of 0.77 against FRAP.
Its mean perimeter-edge error is 0.75 ± 0.21 km across fires, and the reported
mean of the maximum edge error is 2.86 ± 1.14 km. Accuracy is substantially
lower early in an incident; the paper reports unstable IoU before roughly 100
hours and warns about early-perimeter inflation. GOFER is therefore an
hourly, uncertain observation stream. It is unsuitable as fine-scale truth for
spotting or local fireline shape.

The exact GOFER–NIROPS crosswalk is in
`configs/data/gofer_v02_frozen36_crosswalk.yaml`. Four frozen training
incidents share an ICS-209 identifier with GOFER. North Complex and Tamarack
are incident-scope matches. LNU Lightning Complex and River Complex are
aggregate products containing the smaller component named in the NIROPS
bundle, so their spatial targets are excluded. No development or test incident
is affected.

Implementation and evidence:

- `src/aeolus/data/gofer.py` normalizes the actual v0.2 shapefile schema;
- `tools/import_gofer_progression.py` imports directly from the published zip
  or an extracted product directory;
- `tests/test_gofer.py` checks acquisition windows, retrospective
  availability, source filtering, confidence thresholds, monotonicity, and
  fail-closed output behavior; and
- `results/frontier_fire/gofer_v02_kincade/` is an end-to-end source-archive
  smoke artifact with 134 consecutive hourly frames and source hashes.

`src/aeolus/data/progression.py` and
`tools/rasterize_progression_observations.py` align those vectors to an
IncidentBundle grid. The compressed cube retains cumulative perimeter masks,
raw and source-active concurrent lines, acquisition times, clipped area,
coverage fraction, and first-observed frame. Its manifest audits missing line
times, dormant periods, cumulative nesting, domain coverage, and every input
checksum.

River Complex provides a real negative control for the scope gate. All 1,353
hourly frames rasterize cleanly onto the Cronan grid with zero nesting
violations, yet the median grid coverage is only 6.0% of the reported complex
area and final coverage is 3.8%. This quantitatively confirms that the
complex-wide product cannot score the component bundle. The audit artifact is
`results/frontier_fire/gofer_v02_scope_audits/river-complex-on-cronan-grid.npz.manifest.json`.

The next GOFER experiment is declared before looking at physics results:

1. rasterize North Complex and Tamarack onto their frozen incident grids as
   soon as those prepared bundles complete;
2. quantify source-footprint truncation, empty early frames, cumulative-mask
   nesting, and active-line coverage;
3. estimate diurnal growth and active-line likelihood terms on training data
   only;
4. keep the current seven-incident development and seven-incident test sets
   unchanged; and
5. accept the addition only if it improves next-perimeter proper scores and
   advancing-front localization without degrading cumulative IoU beyond a
   predeclared tolerance.

### 2. National hourly growth reference

Fang et al. released hourly progression and matched environmental factors for
294 CONUS fires from 2017–2024. The archive contains one `NewAREA.csv`, RTMA
weather table, LCMS fuel-fraction table, and mean-slope table per event. It
does not contain spatial perimeters. Its useful role is a population reference
for hourly area-growth distributions, dormancy, extreme-growth frequency, and
weather-conditioned response.

The archive has been integrity-checked locally: 294 event directories, 1,176
CSV files, 18 states, all years from 2017 through 2024, and the published MD5
`92c66f372ca6b9f2a6fd95470bbbf730`. Scientific integration is held until the
paper's variable definitions and units are encoded in a source contract. This
avoids silently treating a source wind-speed column as SI units.

Planned acceptance tests:

- every event has a one-to-one hourly join across area, weather, fuel, and
  slope tables;
- time gaps, duplicate hours, missing values, and impossible values are
  enumerated rather than imputed silently;
- incident-level chronological partitions are used for any fitted comparison;
- simulated and observed distributions are compared by season, fuel family,
  and wind regime; and
- distributional agreement remains separate from spatial forecast skill.

### 3. Observation likelihood calibration

The code already represents acquisition duration, processing latency,
localization, detection probability, false-alarm probability, and obscuration.
The unresolved work is empirical calibration by source.

Sequence:

1. pair GOFER with the NIROPS snapshots used by its authors and with FEDS at
   the documented 12-hour comparison times;
2. stratify residuals by hours since ignition, satellite variant, terrain,
   coastal/water adjacency, cloud/smoke quality, and fire size;
3. fit localization and omission models on frozen training incidents;
4. evaluate reliability and proper scores on incidents outside that fit; and
5. propagate the calibrated likelihood through arrival-history assimilation
   and the next-perimeter forecast.

The acceptance criterion is prospective: assimilation must improve the next
observation, not only agreement with the observation being assimilated.

### 4. Model discrepancy and latent suppression

A global spread multiplier cannot distinguish wind error, fuel error,
moisture error, spotting, plume effects, and unobserved suppression. The next
model class is a localized, regularized correction field carried as part of
the ensemble state. Its correlation scale and temporal persistence will be
fit on training incidents and constrained by held-out station and perimeter
evidence.

Suppression observability remains an interval attribute. Intervals with no
usable action chronology receive a latent-intervention grade. Primary
free-spread physics results will exclude the highest-confounding intervals or
integrate over a declared latent treatment prior. Aerial and line-effect
claims require dated geometry and pre/post observations.

### 5. Throughput and numerical optimization

Optimization begins with profiles from the completed run. The primary
quantities are wall time per forecast hour, level-set substeps, active-front
cell count, weather-loading time, process memory, and cache reuse. Candidate
changes are evaluated in this order:

1. content-addressed or copy-on-write storage for immutable weather assets;
2. time-windowed/chunked weather reads and a compact binary perimeter format;
3. shared immutable meteorological grid indexes and bounds-keyed cache entries;
4. memory-mapped read-only incident arrays across local worker processes;
5. narrow-band level-set updates with equivalence checks against the full-grid
   WENO5/RK3 path;
6. batched candidate and ensemble evaluation; and
7. accelerator execution after local numerical equivalence is established.

The first live storage profile changes the expected priority. The prepared
River/Cronan bundle is 2.5 GB: a 1.22 GB dense weather file appears once as the
incident asset and once as the retained provenance artifact, while its raw
perimeter GeoJSON is 135 MB. The per-incident HRRR latitude/longitude index is
27 MB. The scientifically safe repair is immutable content addressing or
copy-on-write cloning with hash-preserving logical assets, followed by
time-chunked reads. Hard-link mutation risk is unacceptable. The active run
has sufficient disk space, so its storage layout remains unchanged.

Each optimization must reproduce perimeter masks, arrival times, and reported
metrics within a declared tolerance on a fixed case set. Wall-clock gains
without an equivalence result are not accepted.

## Iteration order and stop conditions

| Order | Work item | Compute class | Evidence required to continue |
|---:|---|---|---|
| 1 | Complete frozen 36 run | laptop/background | valid source fingerprint; complete artifacts; untouched test |
| 2 | Diagnose errors by incident, horizon, growth regime, fuel, and weather | light | clustered uncertainty; baseline-relative paired errors |
| 3 | Rasterize exact-scope GOFER training matches | light | overlap/truncation audit; hourly acquisition lineage |
| 4 | Calibrate source observation likelihoods | light–moderate | held-out reliability and next-frame improvement |
| 5 | Add localized discrepancy state | moderate | development gain under frozen objective and stability gates |
| 6 | Treat latent suppression | light–moderate | interval grades and sensitivity bounds |
| 7 | Optimize the numerical path | moderate, then cluster | numerical equivalence and profiler evidence |
| 8 | Freeze a new contract and evaluate once | cluster or long local run | preregistered candidates and new source hash |

The current run can disprove the present model class. A test failure leads to
error decomposition and a new contract. The same test observations will not
be reused for repeated method selection.

## Current external requirements

No user input is required for the active run, GOFER integration, or national
hourly-reference audit. Cluster access becomes necessary for large ensembles,
accelerator profiling, and full policy training. Complete historical
suppression chronology and authorized aircraft performance surfaces remain
external data constraints; neither blocks the present spread-accuracy work.

## Primary sources

1. Liu et al. (2024), GOFER method and validation:
   <https://doi.org/10.5194/essd-16-1395-2024>
2. GOFER v0.2 archive: <https://doi.org/10.5281/zenodo.14642378>
3. NASA FEDS product description:
   <https://firms.modaps.eosdis.nasa.gov/descriptions/FEDS_VIIRS_SNPP.html>
4. Fang et al. (2026), 294-fire hourly progression analysis:
   <https://doi.org/10.1016/j.jag.2026.105288>
5. Fang et al. hourly intermediate data:
   <https://doi.org/10.5281/zenodo.18309021>
6. Kochanski et al. (2023), coupled-state perimeter assimilation:
   <https://doi.org/10.3389/ffgc.2023.1203578>
7. Xu et al. (2026), WILDFIREIA leakage-controlled chronological benchmark:
   <https://arxiv.org/abs/2606.15529>
