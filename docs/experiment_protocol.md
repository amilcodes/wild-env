# Experiment and validation protocol

## Dataset partition

Assign entire incidents and geographic regions to train, development and test
before producing raster tiles. All timestamps from one incident remain in one
split. Parameter calibration, reward selection, checkpoint selection and
render selection use train/development data only.

## Simulator calibration

Use no-aerial hindcasts for spread calibration. Candidate parameters include
fuel-family multipliers, directional wind response, residual correlation,
spotting frequency and fuel-consumption rate. Optimize a multi-objective score
over perimeter IoU, area bias, symmetric difference and arrival time. Retain
multiple calibrated parameter sets when equifinality is substantial.

Suppression coefficients require a separate evidence set with treatment
location/time and holding outcome. Perimeter-only data is insufficient.

## Policy comparison

Required policies:

- no aerial intervention;
- nearest feasible task;
- doctrine-inspired anchor/flank heuristic;
- greedy marginal value;
- exact joint resource/task assignment;
- learned MAPPO checkpoint;
- stochastic MPC or another planning baseline before publication claims.

Use common random numbers across policies. The default minimum reporting set is
100 paired seeds per scenario and at least 30 held-out scenarios. Report paired
bootstrap 95% intervals, per-scenario distributions and failure cases rather
than only pooled means.

## Primary endpoints

- asset-weighted loss;
- burned area and containment/escape rate;
- exposure, flight minutes, reload cycles and treatment cost;
- masked/conflicting action rate;
- decision latency and simulator steps/s.

Historical spatial metrics are diagnostic endpoints. In counterfactual
suppression branches, similarity to the observed perimeter is not a direct
measure of policy quality because the historical suppression trajectory is
unknown.

## Reproducibility record

Archive:

- IncidentBundle and source retrieval timestamp;
- exact YAML/JSON experiment configuration;
- code revision and container digest;
- checkpoint plus SHA-256;
- seed manifest;
- ReplayBundle for all reported examples;
- raw episode table and aggregation script.

Renderings are selected after quantitative evaluation and retain their replay
metadata. A rendering is never the sole record of an episode.
