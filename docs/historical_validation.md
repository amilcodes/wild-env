# Historical validation

## Scope

This study evaluates whether Aeolus-IA v0.3 reproduces observed wildfire
progression after initialization from an analyst-interpreted fire perimeter. It
also audits the public evidence needed to evaluate historical suppression. The
spread and suppression questions require different experimental designs:

- Spread forecasting is scoreable from time-indexed fire perimeters.
- Suppression-effect prediction requires time-indexed actions and resource
  states in addition to perimeters.

The study is a hindcast benchmark. It is not an evaluation of a learned MARL
policy. No historical action trace exists for the aerial and ground decisions
represented by the environment, so policy effects cannot yet be identified
from the public records used here.

## Data

The primary reference is the 2026 NIROPS progression dataset by Magstadt et al.
It contains 12,705 analyst-interpreted airborne-infrared observations for 737
western U.S. fires from 2020 through 2024. The data were standardized to
EPSG:4326, topologically checked, and restricted by the authors to incidents
with at least three collection days and one consecutive-day pair. The present
study uses six incidents:

| Incident | State | NIROPS observations | Raster cell |
|---|---:|---:|---:|
| Electra | CA | 8 | 143 m |
| Crockets Knob | OR | 16 | 134 m |
| Dry Lake | AZ | 12 | 134 m |
| Ridge Creek | ID | 17 | 142 m |
| Davis | NM | 9 | 135 m |
| Bear | UT | 14 | 188 m |

Terrain is sampled from USGS 3DEP. Fuels and canopy fields are sampled from
LANDFIRE 2025. Hourly wind speed, wind direction, temperature, relative
humidity, and precipitation come from the NASA POWER point API. POWER supplies
analysis-ready hourly meteorology at its native source resolution. It is a
coarse reanalysis forcing product, not incident-station or coupled fire-weather
data.

The suppression audit uses the USDA Forest Service Fireline Effectiveness
archive RDS-2025-0011. That archive assigns `Held`, `Burned Over`, or
`Not Engaged` outcomes to reported fireline geometry for incidents larger than
1,000 acres from 2017 through 2024.

## Experimental protocol

Each incident contributes one calibration interval followed by four later
validation intervals. No validation interval is used to select the calibrated
spread multiplier.

For each validation interval:

1. Rasterize the observed NIROPS perimeter on the incident grid.
2. Initialize the simulator burned state from that perimeter.
3. Integrate for the exact elapsed time to the next NIROPS observation.
4. Apply the historical hourly weather sequence.
5. Run without simulated suppression.
6. Compare the resulting perimeter and newly burned cells with the next
   observed perimeter.

Three forecasts are scored:

- **Persistence:** the initial perimeter does not grow.
- **Raw physics:** the configured v0.3 fire behavior model with spread
  adjustment 1.0.
- **Calibrated physics:** the same model with a single incident-specific
  surface/crown spread multiplier selected on the earlier calibration
  interval.

The scalar calibration score combines newly burned-area IoU with a penalty on
growth-area ratio. It intentionally avoids simultaneous fitting of fuel
moisture, wind, fuel model, suppression, and spread rate from a single
perimeter transition, because those factors are not identifiable from that
transition alone.

The primary metrics are:

- cumulative burned-extent intersection over union (IoU);
- newly burned-area IoU;
- newly burned-area precision, recall, and F1 after a one-cell localization
  tolerance;
- symmetric mean perimeter-boundary distance;
- 95th-percentile Hausdorff boundary distance; and
- symmetric-difference area.

Uncertainty intervals are obtained with 2,000 bootstrap samples clustered by
incident. Clustering matters because four intervals from one fire are not four
independent fires.

## Results

The study contains 24 held-out forecasts across six incidents. Twenty-one
intervals contain observed growth. Aggregate means are:

| Method | Cumulative IoU | Active-growth IoU | Active-growth tolerance F1 | Boundary distance | Symmetric difference |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.873 | 0.000 | 0.000 | 156 m | 4.4 km² |
| Raw physics | 0.534 | 0.053 | 0.145 | 1,223 m | 50.1 km² |
| Calibrated physics | 0.611 | 0.056 | 0.151 | 938 m | 50.3 km² |

The corresponding incident-cluster 95% intervals for cumulative IoU are
0.830-0.923 for persistence, 0.327-0.719 for raw physics, and 0.394-0.813 for
calibrated physics. For active-growth one-cell-tolerance F1 they are 0,
0.069-0.227, and 0.095-0.213.

These results support three conclusions.

First, persistence is the strongest cumulative-extent forecast. Daily
perimeters contain a large common interior, so cumulative overlap rewards
conservative forecasts. Persistence also has the lowest mean boundary
displacement. Any operational spread model should be required to beat this
baseline.

Second, the physics model has weak, nonzero information about the advancing
front. Persistence has zero overlap and zero tolerance F1 on active-growth
intervals, while calibrated physics reaches 0.056 growth IoU and 0.151
tolerance F1. That signal is too small for an accuracy claim, but it is
measurably different from no growth.

Third, one-interval calibration is unstable under changing fire regimes.
Selected multipliers range from 0.13 to 3.5. Calibration substantially
improves Electra, Dry Lake, and Ridge Creek cumulative scores. It degrades
Crockets Knob and Davis, and fails severely on Bear. Bear's calibrated mean
cumulative IoU is 0.243, its mean boundary displacement is 3.11 km, and its
mean symmetric-difference area is 235 km². The calibration interval selected
the upper candidate 3.5, which overpredicts later growth as observed growth
decelerates.

## Suppression-evidence audit

The Fireline Effectiveness archive contains 2,236,584 line features associated
with 1,161 IRWIN incident identifiers:

| Engagement outcome | Features |
|---|---:|
| Held | 928,014 |
| Burned Over | 492,341 |
| Not Engaged | 816,229 |

Only 6.61% of national features contain `LineDateTime`, the only candidate
construction-time field. `CreateDate` is a geodatabase record field and is not
treated as construction time.

Crockets Knob is present in both study datasets. Its Fireline Effectiveness
record contains 1,742 line features: 1,002 held, 220 burned over, and 520 not
engaged. Only 3 features, or 0.17%, contain `LineDateTime`.

The archive therefore supports spatial outcome analysis. It does not provide
the action chronology needed to replay the fight for most incidents. An
engagement label is also an outcome of the final-perimeter overlay, not a
randomized treatment effect. It cannot by itself establish how a line changed
the fire.

Aeolus-IA v0.3 represents aerial water or retardant placement and only an
implicit ground-hold behavior. It does not yet expose explicit ground-line
construction as a resource-constrained agent action. No public, event-level
aerial drop sequence matched to the six study incidents was identified.

Historical suppression evaluation therefore remains unscored. A valid causal
replay requires:

- time-stamped retardant and water-drop polygons with coverage level;
- dispatch, arrival, reload, turnaround, and availability states;
- time-stamped line-construction segments and production rates;
- firing-operation geometry and timing;
- resource assignments and handoffs;
- weather and perimeter observation timestamps; and
- a model representation that matches those actions.

Until those fields exist, learned-policy replays on historical incidents are
counterfactual scenario studies. They can compare policies under a fixed
simulator, but they cannot be described as historical suppression accuracy.

## Position relative to current practice

Current operational and research systems commonly validate spread by
overlaying forecast and observed perimeters. ELMFIRE documents historical and
real-time perimeter validation and uses ensembles to perturb uncertain inputs.
WRF-SFIRE couples a Rothermel-type level-set fire model to atmospheric
dynamics, and current research periodically assimilates observed perimeters to
correct the evolving state. GOFER provides hourly satellite-derived
progressions for 28 very large California fires; its published final-perimeter
mapping scores are IoU 0.77 for GOFER and 0.83 for VIIRS-derived FEDS, although
those are observation-reconstruction scores rather than free forward
forecasts.

Recent conditional generative work reports mean Dice 0.81 on five fires while
conditioning reconstruction on VIIRS, GOES, and terrain observations. That is
also a reconstruction problem. Direct comparison with the present free
one-step forecast would conflate observation assimilation with propagation
skill.

Relative to these systems, Aeolus-IA v0.3 now has a credible validation harness
and a spatial fire-behavior kernel suitable for high-throughput policy
experiments. Its empirical spread skill is behind the level required for
historical forecast claims. The highest-value next model work is:

1. perimeter state assimilation with uncertainty-aware ensembles;
2. incident-grade RAWS or gridded mesoscale weather, with spatial and temporal
   wind fields;
3. dead/live fuel-moisture initialization and updating;
4. calibrated spotting with explicit ember transport and ignition delay;
5. scale tests on smaller cells around active fronts;
6. explicit line construction and firing operations; and
7. probabilistic scoring for containment and resource outcomes.

MARL training should use parameter ensembles spanning the empirical
uncertainty set. A policy trained against one fitted deterministic fire is
likely to exploit simulator error.

## Reproduction

The frozen result files live under `results/historical_validation/`. To rebuild
source bundles and rerun the study:

```bash
aeolus-study prepare \
  --manifest configs/historical_validation.yaml \
  --source-shapefile /path/to/NIROPS_2020_2024_R1_R6.shp \
  --out outputs/historical-validation/incidents

aeolus-study run \
  --manifest configs/historical_validation.yaml \
  --prepared-root outputs/historical-validation/incidents \
  --out outputs/historical-validation/results
```

The NIROPS source archive used for this study had SHA-256
`b19fb16ce2792d9a9c01f1768d09962566b0f3cada8d1f23a9851ab3fce75615`.
The Fireline Effectiveness archive had SHA-256
`7698cccb39a07369b1dcc3f1bf83bfa12f8a6ee7afdd2c2f473599487b4bc64d`.

## Primary references

1. Magstadt et al. (2026), *A high spatial resolution daily fire perimeter
   progression dataset for wildfires in the Western United States: 2020-2024*.
   <https://doi.org/10.17632/95rj5d379g.1>
2. Arkowitz et al. (2025), *Quality assured spatial dataset of wildfire
   containment firelines and engagement outcomes 2017 to 2024*.
   <https://doi.org/10.1038/s41597-025-05208-0>
3. Liu et al. (2024), *Systematically tracking the hourly progression of large
   wildfires using GOES satellite observations*.
   <https://doi.org/10.5194/essd-16-1395-2024>
4. Kochanski et al. (2023), *Analysis of methods for assimilating fire
   perimeters into a coupled fire-atmosphere model*.
   <https://doi.org/10.3389/ffgc.2023.1203578>
5. Shaddy et al. (2025), *Generative Algorithms for Wildfire Progression
   Reconstruction from Multi-Modal Satellite Active Fire Measurements and
   Terrain Height*. <https://arxiv.org/abs/2506.10404>
6. NASA POWER, *Hourly API documentation*.
   <https://power.larc.nasa.gov/docs/services/api/temporal/hourly/>
7. ELMFIRE, *Validation documentation*. <https://elmfire.io/validation.html>
