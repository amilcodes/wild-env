# Non-compute-bound P1 remaining work

Date: 2026-07-30

This is the execution register for unresolved P1 validity gates. The order is
intentional. Historical training and policy comparison should wait until items
P1-01 through P1-05 have clean, frozen data products.

## Gate state

| ID | Gate | Current state | Compute class | Blocks |
|---|---|---|---|---|
| P1-01 | Time-admissible historical fuels | Partial pass: the six-case reconstruction passes the version gate; the frozen 36 rebuild is running with metric-CRS LANDFIRE requests; closer archived vintages and pixel-level disturbance lineage remain | Non-compute | Historical accuracy |
| P1-02 | Source-calibrated observation likelihood | Partial: interval/latency semantics and a version-locked GOFER v0.2 hourly adapter exist; source parameters remain uncalibrated | Non-compute | Assimilation and scoring |
| P1-03 | Frozen expanded benchmark | Execution: immutable chronological split is 22 train / 7 development / 7 test, contract audit passes, source fingerprint is frozen, and the complete local run is active | Non-compute, followed by hybrid evaluation | Generalization |
| P1-04 | Incident wind and moisture evidence | Partial: operational HRRR windows, downscaling, spin-up, and live-fuel model exist; the 36-case forcing materialization is queued and systematic station evaluation remains absent | Non-compute | Historical accuracy |
| P1-05 | Unobserved suppression treatment | Failed: action chronology is absent for the current historical sample | Non-compute | Causal spread and suppression claims |
| P1-06 | Empirical crown and spotting validity | Open: mechanisms exist and are excluded from the current validated regime | Hybrid | Extreme-fire scenarios |
| P1-07 | Vehicle-specific aviation performance | Partial: nine named current-use profiles and a mixed-fleet scenario are traceable; 0/9 have authorized flight-manual or engineering surfaces | Non-compute/data | Operational feasibility |
| P1-08 | Incident airspace and route safety | Partial: straight-leg volumes, terrain, and crosswind are enforced | Non-compute | Operational feasibility |
| P1-09 | Objective and operations optimizer | Failed: configured trade favors near-no-action behavior; short rollout does not separate from no action | Non-compute | Planner imitation, RL, and policy regret |
| P1-10 | Canonical/tensor constraint parity | Partial: new aircraft gates exist only in the canonical environment | Compute-bound implementation and non-compute tests | Cluster policy validity |

## P1-01 — reconstruct time-admissible historical fuels

**Current evidence**

The original six incidents are dated 2020–2023 and use LANDFIRE 2025.
Replacement bundles now use an exact-grid reconstruction from LANDFIRE 2016
Remap, whose disturbance inputs end in 2016 and capable-fuel condition is
effective for 2019. All six pass the version-level historical gate. Across the
six incident grids, 62–91% of FBFM40 cells and 0.6–13.1% of burnability cells
change relative to the 2025 inputs.

The selector separately records the preferred eligible vintage. LF2019L is
preferred for Bear and LF2020 for the 2022–2023 incidents. Those complete
fuel/vegetation vintages are absent from the current image service, leaving
five archive substitutions. The rebuild retains raw layers, exact transforms,
source endpoints, and SHA-256 checksums.

A 96-run fixed-parameter screening ablation over 24 held-out intervals finds
that the rebuilt fuels improve mean IoU by 0.065 in the raw branch and 0.060
under frozen reference spread coefficients. Mean boundary distance falls by
406 m and 507 m, respectively. This uses the legacy adaptive-Huygens solver;
the primary WENO5 ensemble rerun remains open.

**Next work**

1. Obtain LF2019L and LF2020 archived FBFM40 and canopy layers and replace the
   five documented fallback substitutions.
2. Query LANDFIRE disturbance/public-events layers for each incident footprint
   through the simulation date.
3. Reconstruct changed pixels from the last admissible base landscape plus
   only pre-incident disturbances and treatments.
4. Store product edition, source cutoff, disturbance event IDs/dates,
   treatment IDs/dates, native resolution, resampling, mapping, and checksums.
5. Produce mapped before/after fuel-code and canopy-difference rasters for
   expert review; the current manifest already retains numeric differences.
6. Retain the controlled paired hindcast ablation with the frozen corpus.

**Pass criterion**

- 100% of evaluated pixels have a product/data cutoff at or before incident
  initialization;
- all later disturbance events are excluded;
- all retained transformations are reproducible from archived source assets;
- a reviewer can trace every changed pixel to a dated event; and
- the historical result states sensitivity to plausible alternative fuel
  reconstructions.

**Dependency**

Public LANDFIRE historical products and public-events layers. No proprietary
input is inherently required.

## P1-02 — calibrate observation likelihoods by source

**Current evidence**

The evaluator now handles acquisition start/end, availability, spatial
localization, detection, false alarm, and obscuration. A version-locked GOFER
v0.2 adapter now preserves hourly acquisition windows, retrospective
availability, concurrent/retrospective active-line state, product lineage, and
the published population-level spatial validation terms. Its real-archive
smoke test contains 134 consecutive Kincade frames with valid geometry and no
cumulative-area reversal. The grid-aligned cube writer preserves mask/line
state and audits clipping; a River-Complex negative control measured only 6.0%
median coverage on the Cronan component grid and correctly remains excluded.
NIROPS and FEDS source terms remain incomplete, and
GOFER's aggregate edge error has not been calibrated into a conditional
per-frame likelihood.

**Next work**

1. Add source adapters for NIROPS, FEDS/VIIRS, and incident infrared line
   products used in evaluation; retain the completed GOFER adapter as the
   hourly reference implementation.
2. Preserve source acquisition start, acquisition end, processing/publication
   time, footprint/scan geometry, quality flags, and cloud/smoke state.
3. Estimate geolocation error and acquisition-time error from repeat passes,
   overlapping sensors, and analyst revisions.
4. Estimate detection and false-alarm curves by source, fire radiative power
   or intensity, view geometry, cloud/smoke, and land cover.
5. Use source-calibrated likelihoods in ensemble weights and reliability
   scoring.
6. Test sensitivity to alternative acquisition-time priors when scan timing is
   unavailable.

**Pass criterion**

- every scored frame declares sensing and availability time separately;
- every scored cell has a usable observability weight;
- likelihood calibration is evaluated on incidents outside the fitting set;
- reliability curves and proper scores improve over exact-timestamp and
  binary-clear assumptions; and
- the next-perimeter forecast improves after assimilation.

**Dependency**

Public product metadata can support much of the work. NIROPS processing logs or
quality layers would improve calibration if available.

## P1-03 — execute and audit the frozen expanded benchmark

**Current evidence**

The 36-incident selector covers nine states and 2020–2024, with four
reported-size strata per state. The immutable chronological contract now
assigns 22 incidents from 2020–2021 to training, seven incidents from 2022 to
development, and seven incidents from 2023–2024 to test. Incident overlap is
zero, test targets are prohibited from fitting, the base-manifest hash is
frozen, and an 87-file execution-source/environment fingerprint passes. The
resumable local run is active. Ecoregion, dominant fuel family, and weather
regime remain post-freeze audit attributes; they cannot be used to reshuffle
the active partition.

**Next work**

1. Complete metric-grid preparation, historical-fuel reconstruction, and
   operationally admissible HRRR forcing without inspecting model scores.
2. Derive EPA/USFS ecoregion from ignition and burned footprint.
3. Derive dominant and perimeter-adjacent FBFM40 families from the
   time-admissible landscape.
4. Derive weather regime from wind, humidity, fuel moisture, and percentile
   context using a declared pre-fit rule.
5. Audit the already-frozen temporal split for geographic and regime coverage;
   report deficiencies rather than moving incidents after the fact.
6. Run partition, source-date, metric-CRS, forcing-availability, and missingness
   audits before physics fitting.
7. Verify the execution fingerprint at every stage transition and freeze the
   completed source-bundle checksums.
8. Fit only declared training candidates, select only with development
   incidents, and score the test partition once.

**Pass criterion**

- no incident appears in more than one split;
- no tile or later timestamp from a test incident is used for tuning;
- all split/regime fields are complete for stratified reporting;
- train, development, and test coverage deficiencies are explicit;
- exclusions are logged with reason codes; and
- the test manifest hash predates final parameter selection.

**Dependency**

P1-01 and P1-04. The full run becomes compute-consuming after this
non-compute-bound preparation step.

## P1-04 — evaluate incident wind and spatial moisture

**Current evidence**

The repository has HRRR analysis ingestion, explicit missing-hour coverage,
terrain lapse-rate and conserved-vapor-pressure downscaling, dead-fuel
spin-up, and herbaceous/woody live-fuel moisture. It lacks a systematic
incident-held-out comparison to RAWS wind, fuel-stick observations, and
independent live-fuel-moisture products.

**Next work**

1. Acquire all RAWS stations within a declared distance/elevation envelope for
   each incident.
2. Apply time, range, persistence, step, and cross-variable quality control;
   retain every rejection flag.
3. Compare raw HRRR, terrain projection, and station-conditioned analysis in
   Cartesian wind components with station-held-out cross-validation.
4. Validate 1/10/100 h moisture against fuel-stick observations where
   available.
5. Validate live fuels against field records and Globe-LFMC or another
   independent spatial product.
6. Add precipitation interception, canopy shading/radiation, 1000 h dead fuel,
   dynamic curing uncertainty, and station representativeness error.
7. Carry the resulting correction covariance into the forecast ensemble.

**Pass criterion**

- held-out station wind-vector RMSE and bias improve over raw HRRR;
- probabilistic intervals achieve declared coverage;
- dead/live moisture errors are reported by fuel family and season;
- missing forcing hours and fallback behavior remain explicit; and
- fire-skill conclusions remain stable across reasonable correction variants.

**Dependency**

Public RAWS and gridded products. Incident-specific portable-station or fuel
sampling data would improve coverage.

## P1-05 — handle historical suppression as a latent intervention

**Current evidence**

The current six-incident perimeter series lacks sufficiently complete
dispatch, line, firing-operation, and aerial-drop chronology. Apparent spread
residuals can therefore combine fire physics with unobserved treatment.

**Next work**

1. Link IRWIN IDs across NIROPS, ICS-209, Fireline Engagement, aviation
   telemetry, and incident archives.
2. Normalize line geometry/status, drop polygons, release times, resource
   class, firing operations, and known natural barriers.
3. Assign interval-level suppression observability grades.
4. Censor high-confounding intervals for the primary spread-physics result.
5. Retain them in a sensitivity analysis with suppression represented as a
   latent treatment field and uncertainty propagated.
6. Restrict suppression-effect claims to records with dated placement and
   pre/post spread observations.

**Pass criterion**

- every validation interval has a suppression observability grade;
- the primary physics benchmark excludes or models unidentified intervention;
- action-conditioned effects use predeclared causal estimands; and
- conclusions are reported both with and without ambiguous intervals.

**Dependency**

Public archives may support partial reconstruction. Complete aviation and
ground-operation chronology may require agency data access.

## P1-06 — calibrate crown transition and spotting

**Current evidence**

The simulator has crown and spotting mechanisms. The validity audit marks any
episode using them as `mechanism_only_unvalidated_regime`.

**Next work**

1. Assemble independent cases with canopy structure, surface/crown fire type,
   spotting observations, wind profile, fuel moisture, and arrival history.
2. Calibrate crown initiation, passive/active transition, crown spread, and
   return-to-surface separately.
3. Calibrate ember production, lofting, transport, burnout, landing,
   ignition, and secondary growth as separate distributions.
4. Add unsupported-regime detection for plume-dominated cases.
5. Evaluate fast-model rank stability against WRF-SFIRE or QUIC-Fire teacher
   cases.

**Pass criterion**

- transition discrimination and arrival/spread errors are reported on held-out
  incidents;
- spot-distance and ignition distributions pass posterior predictive checks;
- uncertainty rises outside the calibrated domain; and
- plume-dominated cases are identified rather than silently processed as
  ordinary wind-driven fire.

**Dependency**

Data assembly is non-compute-bound. Teacher ensembles and full calibration are
hybrid/compute-bound.

## P1-07 — replace the synthetic aircraft surface

**Current evidence**

The mobility-surface schema, interpolation, extrapolation gate, and canonical
simulator integration are complete. A provenance-graded catalog covers six named
CAL FIRE crewed aircraft and three interagency-reference UAS. A nine-resource
scenario exercises tanker, helicopter, tactical/intelligence, and UAS roles.
Public nominal specifications and research assumptions are recorded separately.
The catalog reports zero field-performance-ready mobility profiles because no
authorized exact-current flight-manual or engineering surface is attached.
Public-source acquisition added a 19-document configuration registry and an
engineering delivery surface from controlled USFS tests of the CAL FIRE S-2T.
The S-2T line table now drives suppression geometry. Public Air Force C-130H
MAFFS and Army UH-1H data were retained as explicit proxies because they do not
match the CAL FIRE RDS and -703 Super Huey configurations. See
[`aviation_evidence_acquisition.md`](aviation_evidence_acquisition.md). Four
focused CAL FIRE request texts and the intake procedure are ready in
[`aviation_records_request.md`](aviation_records_request.md).

**Next work**

1. Submit the prepared exact-configuration records/data requests and confirm
   tail applicability for the selected S-2T,
   C-130H, S-70i, UH-1H, OV-10A, King Air 200, Skydio X10, ANAFI USA, and
   Alta X profiles.
2. Obtain authorization and transcribe reviewed flight-manual or engineering performance tables with
   source page/revision and configuration.
3. Add pressure altitude/altimeter setting, gross weight, center-of-gravity
   limits, climb/descent, hover/one-engine performance where applicable,
   takeoff/landing distance, and phase-specific fuel or energy use.
4. Cross-check the normalized surface against an independent six-degree-of-
   freedom model or manufacturer calculation.
5. Establish configuration-control and reviewer signoff for each surface.

**Pass criterion**

- no operational scenario references the synthetic surface;
- interpolation reproduces source tables within declared tolerance;
- outside-envelope combinations remain hard failures;
- reserve and diversion tests pass across the environmental envelope; and
- each table has vehicle configuration, revision, source, and reviewer.

**Dependency requiring user or partner input**

The selected vehicle list and public-source search are complete. Exact manuals
or engineering datasets may be controlled, licensed, or available only through
operator tools. CAL FIRE permits anonymous public-records requests, but still
requires an email address or other contact channel for correspondence and
release. Operator or manufacturer cooperation is needed for S-70i iFly data
and configuration-controlled supplements. The project will not fabricate
them.

## P1-08 — route, altitude, and incident-airspace safety

**Current evidence**

The canonical mask blocks a straight leg that intersects a time/altitude
volume, violates payload/density altitude, terrain ceiling, crosswind, service
geometry, endurance, or reserve. It does not find an alternate route.

**Next work**

1. Add route nodes and corridors over terrain and obstacle clearance surfaces.
2. Separate departure, climb, cruise, approach, drop, escape, and recovery
   phases.
3. Ingest TFR/NOTAM and incident-airspace records into versioned volume
   geometry with effective times.
4. Add ATGS/lead-plane authorization, drop-lane occupancy, approach/egress
   corridors, vertical/lateral separation, lost-link paths, and alternates.
5. Return structured constraint reason codes and feasible repaired actions from
   an external safety shield.
6. Record planned and executed routes plus every shield intervention.

**Pass criterion**

- route feasibility is checked over the complete trajectory and time interval;
- all hard constraints have deterministic tests and explicit reason codes;
- scenarios with feasible detours are routed rather than blocked;
- separation and lane conflicts are detected under concurrent assignments; and
- action-mask/shield decisions are reproducible from replay state.

**Dependency**

Public FAA/NIFC products cover part of the data. Incident airspace and
supervision authorization may require operational feeds.

## P1-09 — build a planning comparator with a consistent objective

**Current evidence**

Exact one-step joint assignment is implemented and performs best in the small
control study on terminal fire loss. The rollout comparator improves the
configured loss-plus-cost objective relative to exact assignment because the
sortie-cost weight overwhelms the measured loss avoided. It does not achieve a
positive paired interval relative to no action. The current control therefore
fails to identify a useful cost/loss trade before policy quality is considered.

**Next work**

1. Estimate or declare resource costs in physical units and asset/fire loss in
   a common decision-analysis scale; publish sensitivity curves and the
   implied marginal exchange rate.
2. Separate safety, reserve, capacity, authority, and exposure limits from the
   scalar reward and enforce them as constraints.
3. Specify one planner objective aligned with every reported primary outcome.
4. Implement rolling-horizon CP-SAT or mixed-integer optimization for resource,
   attack segment, service site, time, payload, stock, and route reservations.
5. Carry assignments already dispatched as commitment state.
6. Add a calibrated terminal value or a horizon reaching through travel,
   delivery, and response latency.
7. Add stochastic scenarios for wind, fire response, queue time, site closure,
   and resource failure.
8. Evaluate optimality gap on small exact cases and paired outcome regret on
   held-out simulator cases.
9. Use the accepted planner for behavior cloning only after it beats exact
   assignment and no action on the predeclared criteria.

**Pass criterion**

- the configured marginal cost/loss trade is documented and passes sensitivity
  review;
- the planner has a positive paired interval relative to no action and the
  strongest assignment baseline on primary outcomes;
- no receding-horizon procrastination in fixed diagnostic cases;
- feasibility is identical to the external safety layer;
- solver gap and wall time are reported;
- paired case-cluster interval favors the planner on held-out cases; and
- planner demonstrations carry objective, constraints, and solver status.

**Dependency**

No cluster is required for the initial deterministic solver. Larger stochastic
scenario sets are hybrid.

## P1-10 — preserve constraints in the cluster environment

**Current evidence**

The canonical simulator now applies the new aircraft and airspace controls.
`TensorOperationsEnv` still uses its prior constant-performance route and mask
semantics.

**Next work**

1. Normalize performance surfaces into fixed tensors.
2. Add batched density-altitude interpolation, vector groundspeed, route
   reserve, service geometry, and time-active volume masks.
3. Define one shared set of structured constraint reason codes.
4. Build property tests over randomized canonical/tensor states.
5. Block cluster training until parity passes for all hard constraints.

**Pass criterion**

- exact mask parity for randomized in-domain cases;
- matched violation reason codes;
- numerical leg-time agreement within declared tolerance;
- no host synchronization in the tensor transition; and
- parity artifacts retained with the training configuration.

**Dependency**

Accelerator implementation is compute-bound. Contracts, fixtures, and CPU
property tests are not.

## Work after the P1 gates

The next tier becomes defensible only after the preceding evidence is in place:

- sequential localized assimilation with front morphing and backward replay;
- fire-atmosphere correction fields trained on coupled-model teachers;
- suppression placement and response calibration from treatment chronology;
- a unified accelerator-resident fire/belief/operations transition;
- constrained MARL training across five or more seeds;
- held-out policy comparison over at least 100 paired seeds per scenario
  family; and
- operational software-in-the-loop, hardware-in-the-loop, hazard, security,
  environmental, and human-factors programs.

## External inputs that materially change the result

| Input | Why it matters | Can public data substitute? |
|---|---|---|
| Selected vehicle configurations and reviewed performance sources | Converts the aviation interface into vehicle evidence | Sometimes; manufacturer or agency material may be restricted |
| Dated drop, line, dispatch, and firing-operation records | Separates fire behavior from suppression intervention | Partially |
| NIROPS acquisition/processing quality metadata | Calibrates sensing and availability time | Partially |
| Incident portable-weather and fuel-moisture observations | Extends beyond permanent RAWS coverage | Partially |
| Target cluster and runtime image | Required for tensor parity throughput and training evidence | No, once compute work begins |

No other user decision is required for the public-data work in P1-01 through
P1-05. The scientific result should remain blocked at the relevant gate when a
required source is unavailable.
