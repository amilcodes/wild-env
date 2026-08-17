# Historical fuel validity: P0 reconstruction

## Problem

The six prepared NIROPS incidents used LANDFIRE 2025 surface-fuel and canopy
layers. The incidents occurred from 2020 through 2023. A 2025 landscape can
contain disturbances, vegetation change, or fuel-model updates that occurred
after the fire being predicted. That is target leakage: the simulator is given
landscape information that would not describe the incident-time state.

This affects every historical skill result, even if the fire solver itself is
unchanged. The previous validity audit therefore failed all six incident
bundles.

## Selection rule

`aeolus.data.historical_fuels` defines version records using:

- version year;
- last included disturbance year;
- effective fuel-condition year;
- completion/publication year;
- current access status; and
- exact image-service endpoints.

A landscape is eligible when its disturbance record ends before the incident
year and its effective fuel condition is no later than the incident year. The
selector first identifies the preferred time-admissible vintage and then the
most recent qualifying vintage that can be reproduced from a current public
service.

LANDFIRE's official comparison and alert pages are the governing source:

- [LANDFIRE product-version comparison](https://landfire.gov/data/comparison-table)
- [LANDFIRE alerts and retired-product notices](https://www.landfire.gov/data/alerts)

LANDFIRE 2016 Remap is the current reproducible source for all five required
layers: FBFM40, canopy cover, canopy height, canopy base height, and canopy bulk
density. Its disturbance inputs end in 2016 and its capable-fuel condition is
effective for 2019.

For the 2021 Bear incident, LF2019L is a closer preferred state. For the two
2022 and two 2023 incidents, LF2020 is closer. Those complete fuel and
vegetation versions are not currently exposed by the image service. LANDFIRE
reports that LF2020 was retired and remains available by request or through the
USGS ScienceBase library. Five of six bundles therefore retain an explicit
archive-substitution item.

## Reconstruction

The rebuild tool:

1. reads the exact grid, CRS, bounds, transform, elevation, assets, weather, and
   observations from each existing incident;
2. exports the five historical LANDFIRE rasters onto that exact grid;
3. rejects any shape, CRS, or transform mismatch;
4. derives oven-dry surface load and burnability from historical FBFM40;
5. converts canopy layers from native LANDFIRE scaling;
6. writes a new six-band GeoTIFF and simulator bundle;
7. stores every raw raster and SHA-256 checksum; and
8. re-runs the product-year and disturbance-cutoff gate.

The old corpus is preserved at
`../outputs/historical-validation-v4-electra`. The replacement is at
`../outputs/historical-validation-v5-time-admissible`.

## Reconstruction result

All six rebuilt bundles pass the historical-admissibility gate.

| Incident | FBFM40 cells changed | Burnability changed | Preferred | Reproduced |
|---|---:|---:|---|---|
| Dry Lake | 76.15% | 3.02% | LF2016 Remap | LF2016 Remap |
| Electra | 87.25% | 8.18% | LF2020 | LF2016 Remap |
| Ridge Creek | 68.96% | 0.60% | LF2020 | LF2016 Remap |
| Davis | 62.13% | 8.60% | LF2020 | LF2016 Remap |
| Crocket's Knob | 90.53% | 2.61% | LF2020 | LF2016 Remap |
| Bear | 73.01% | 13.13% | LF2019L | LF2016 Remap |

The mean FBFM40 change is 76.34%; the mean burnability change is 6.02%. These
large differences show that the P0 was substantive. They do not by themselves
show that the older fuel state will produce more accurate spread forecasts.
That question requires the controlled held-out hindcast comparison.

The complete reconstruction manifest is
`../outputs/historical-validation-v5-time-admissible/fuel_rebuild_manifest.json`.

## Fixed-parameter screening result

A paired screening study ran 96 hindcasts:

- six incidents;
- four held-out intervals per incident;
- original and rebuilt fuel/canopy state; and
- raw and frozen-reference spread coefficients.

Elevation, weather, observations, initialization, code, parameter values, and
seeds are identical within every pair. Only the landscape fuel and canopy
state change. The study uses the legacy adaptive-Huygens solver to keep the
ablation locally tractable. It does not replace the WENO5 posterior-ensemble
benchmark.

| Fixed branch | Metric | 2025 fuels | Time-admissible fuels | Paired change |
|---|---|---:|---:|---:|
| Raw physics | Perimeter IoU | 0.325 | 0.390 | +0.065 |
| Raw physics | Mean boundary distance | 1,913 m | 1,507 m | -406 m |
| Raw physics | 1-cell growth F1 | 0.120 | 0.144 | +0.023 |
| Frozen reference coefficient | Perimeter IoU | 0.221 | 0.281 | +0.060 |
| Frozen reference coefficient | Mean boundary distance | 2,737 m | 2,230 m | -507 m |
| Frozen reference coefficient | 1-cell growth F1 | 0.072 | 0.095 | +0.023 |

The raw-physics paired bootstrap interval excludes zero for mean IoU,
boundary distance, and symmetric-difference area; its growth-F1 interval
slightly crosses zero. The frozen-reference intervals exclude zero for all
four recorded metrics. Between 58% and 71% of individual intervals improve,
depending on metric.

This is evidence that the invalid fuel vintage materially affected the
forecasts and that the rebuilt inputs move this screening model in a favorable
direction. Absolute accuracy is still weak, and the finding cannot be
transferred to the WENO5 ensemble until that study is rerun.

The primary paired WENO5 study is packaged as a two-element Slurm array:

```bash
AEOLUS_IMAGE=/path/to/aeolus.sif \
AEOLUS_HISTORICAL_DATA_ROOT=/archive/aeolus/historical \
sbatch deploy/slurm/historical_fuel_weno.sbatch
```

Array task 0 runs the original corpus and task 1 runs the replacement. Each
uses one 32-core node by default. `aeolus-study run --workers N` provides an
execution override without modifying the frozen scientific manifest.

Machine-readable paired results are in
`results/aviation_fuel_p0/fuel_ablation/paired_results.json`. The combined
aviation/fuel result and figure are in
`results/aviation_fuel_p0/aviation_fuel_p0_study.json` and
`results/aviation_fuel_p0/aviation_fuel_p0_study.png`.

## Claim boundary

Passing this gate means the represented disturbance cutoff predates the
incident. It does not prove:

- pixel-level historical truth;
- that no undocumented local fuel treatment occurred;
- that LF2016 is the closest available state for the five archive-substitution
  incidents;
- that the Scott/Burgan fuel category is correct at every cell; or
- that differences in hindcast error are caused only by fuel state.

The controlled comparison holds the study manifest, observations, weather,
code, and seeds fixed. Even then, historical perimeter growth includes
unobserved suppression and observational uncertainty. Results should be
interpreted as a model-input sensitivity and validity repair, not causal fuel
attribution.

## Remaining work

1. Request or retrieve LF2019L and LF2020 FBFM40/canopy archives.
2. Run the same exact-grid reconstruction and paired hindcast protocol.
3. Add disturbance-year rasters where available and audit incident pixels,
   rather than relying only on national version cutoffs.
4. Incorporate local treatment and incident GIS records.
5. Compare LANDFIRE states against contemporaneous high-resolution imagery and
   field fuel observations for a smaller, deeply characterized benchmark.
