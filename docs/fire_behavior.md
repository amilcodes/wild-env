# Fire-behavior model and validation contract

## Scope

Version 0.4 has two execution paths over one local fire-behavior model:

- the canonical NumPy incident simulator, used by the PettingZoo environment,
  historical workflows, interventions and replay;
- `TensorFireKernel`, which keeps batches of landscapes resident in PyTorch on
  CUDA, ROCm, MPS or CPU.

This is an operational-equation, raster-front model. It is substantially more
specific than the version 0.2 stochastic cellular kernel. It does not reproduce
two-way atmosphere/fire coupling, plume dynamics or combustion-resolving CFD.

## Surface behavior

The packaged table contains all 53 burnable Anderson 13 and Scott/Burgan 40
fuel models. It was generated with Pyretechnics 2025.5.15 over:

- 1-hour dead-fuel moisture: 0.03–0.40 kg/kg;
- 10 m wind speed: 0–30 m/s;
- slope: 0–1 rise/run.

The table stores heading/backing rate of spread, heading fireline intensity and
flame length. `tools/build_fire_lookup.py` is the reproducible generator and
the NPZ records the reference version, units and conditioning assumptions.
Runtime interpolation is bilinear in moisture and wind or moisture and slope.
Exact table nodes are tested against retained Pyretechnics fixtures.

Wind and slope factors are recovered relative to the no-wind/no-slope rate
`R0`:

```text
phi_w = max(R_wind / R0 - 1, 0)
phi_s = max(R_slope / R0 - 1, 0)
phi_vector = phi_w * downwind_unit + phi_s * upslope_unit
R_head = R0 * (1 + |phi_vector|)
```

Heading is the direction of `phi_vector`. Heading and backing values determine
ellipse eccentricity. Directional rate toward a neighbor uses:

```text
R(theta) = R_head * (1 - e) / (1 - e cos(theta))
```

This is the vector-combination and elliptical Huygens class used by ELMFIRE
and operational spread tools. Residual uncertainty, fuel-load conditioning and
treatments are explicit multiplicative fields, retained in replay.

## Crown fire

Crown initiation uses Van Wagner critical intensity:

```text
I_critical = [0.01 * CBH * (460 + 2600 * M_foliar)]^(3/2)  kW/m
```

Potential crown rate uses the Cruz relation:

```text
R_active = 11.02 * U10_km_h^0.9 * CBD^0.19 * exp(-17 * M1)  m/min
R_critical = 3 / CBD
```

Cells above initiation intensity are passive crown when potential rate is below
`R_critical`, and active crown otherwise. Canopy fuel consumption contributes
to fireline intensity. The replay stores surface/passive-crown/active-crown
type, rate of spread and flame length at every recorded minute.

## Fuel moisture and weather

CF-NetCDF weather supplies 10 m wind speed and direction, air temperature,
relative humidity and optional precipitation rate. Each variable can be a time
series or a time-varying raster on the fire grid. Wind direction is interpolated
as a circular quantity, so a transition from 359 to 1 degrees passes through
zero. Dead 1/10/100-hour moisture uses separate Van Wagner-Pickett drying and
wetting equilibria, sorption hysteresis, and exact exponential 1/10/100-hour
lag updates. Rain and water drops wet fuels; water also reduces current
intensity.

Historical preparation derives live herbaceous and woody moisture from the
NFDRS-v4 growing-season-index factors: minimum temperature, maximum vapor
pressure deficit, photoperiod, and trailing precipitation. The unsmoothed
product receives a 28-day trailing mean and maps to 0.30–2.50 kg/kg
herbaceous and 0.60–2.00 kg/kg woody moisture. A 60-day weather spin-up is used
so the rolling state is developed before the first perimeter.

The packaged Pyretechnics lookup has explicit dead 1-hour, live-herbaceous,
live-woody, wind, and slope axes. Pyretechnics performs Scott-Burgan dynamic
herbaceous load transfer below 1.20 kg/kg, with complete transfer at 0.30
kg/kg. NumPy and PyTorch interpolate the same five-dimensional table. At dead
moisture 0.07, 4 m/s wind, and slope 0.2, FBFM102 spread falls from 8.12 m/min
at 0.30 live-herbaceous moisture to 0.032 m/min at 2.50; static FBFM1 remains
12.14 m/min across the same live-moisture sweep.

This remains a fuel-conditioning model. GSI is meteorologically derived rather
than incident fuel sampling, remote-sensing retrieval, or species-specific
phenology. A research result must state whether moisture was observed,
initialized, derived, calibrated, or left at the scenario assumption.

## Front propagation

The canonical front is the zero contour of a signed-distance level-set field:
negative values are inside the represented fire and positive values are
outside. Propagation solves the anisotropic Hamilton-Jacobi equation using
Jiang-Shu WENO5 one-sided derivatives and third-order strong-stability-
preserving Runge-Kutta time integration. A first-order Godunov comparator is
retained. Adaptive substeps enforce the configured CFL bound.

The directional speed at the front is recovered from the local heading rate
and ellipse eccentricity. The Hamiltonian is evaluated in a narrow band and on
the connected exterior of the represented fire. The latter causal restriction
prevents a contour from nucleating across a nonburnable barrier. The signed
distance field is periodically reinitialized. The NumPy path uses an exact
Euclidean distance transform; the PyTorch path uses an on-device pseudo-time
reinitialization equation and does not copy state back to the host.

Arrival time is interpolated within the numerical step when the zero contour
crosses a cell center. The prior eight-neighbor accumulated-travel solver is
available as `adaptive_huygens` for ablation.

`aeolus-fire verify-front` runs two manufactured-solution checks. After a
30-minute constant-speed circular expansion, the 15 m WENO5 run has 1.65 m
equivalent-radius error. An anisotropic front rotated through eight headings
has 0.60% coefficient of variation in reached area and 1.82% in heading
extent. These checks establish numerical consistency for the tested idealized
conditions; they do not validate physical rate-of-spread parameters.

This front formulation is parallel to the numerical class used by WRF-Fire
and CFBM. The present model remains one-way forced and includes empirical crown
and spotting modules, while WRF-Fire/CFBM provide atmosphere coupling and
their own surface-fire assumptions.

## Spotting

Eligible high-intensity cells emit a Poisson number of embers. Downwind
distance is lognormal with wind- and intensity-dependent median; crosswind
offset is normal. Survival decays with distance and ignition depends on target
dead-fuel moisture and treatment. Parameters, maximum distance and per-minute
candidate cap are in `FireBehaviorConfig`.

The implementation follows the statistical structure exposed by ELMFIRE. Its
coefficients are research assumptions until calibrated against incident- or
fuel-type-specific ember observations.

## Validation commands

Resolve a local behavior case:

```bash
aeolus-fire point \
  --fuel-model 145 \
  --moisture 0.07 \
  --wind 6
```

Run accelerator throughput:

```bash
aeolus-fire benchmark \
  --device cuda \
  --batch 256 \
  --height 128 \
  --width 128 \
  --steps 100
```

Generate idealized cases and the validation atlas:

```bash
aeolus-fire validate \
  --device cuda \
  --size 128 \
  --minutes 90 \
  --output runs/fire-validation
```

Run manufactured-solution grid and rotation checks:

```bash
aeolus-fire verify-front \
  --output runs/front-verification
```

Calibrate one effective spread parameter on one historical interval and apply
it without refitting to the next interval:

```bash
aeolus-historical \
  --incident runs/incidents/example \
  --mode calibrate \
  --policy no_aerial \
  --start-index 0 \
  --target-index 1 \
  --validation-target-index 2 \
  --out runs/fire-calibration.json
```

Calibration reports cumulative-perimeter and incremental-growth metrics
separately. The historical study also fits a posterior parameter ensemble over
effective spread, wind exposure, wind direction, and dead-fuel moisture. The
ensemble emits burn probability and conditional arrival-time mean and
standard deviation, and is scored with Brier, balanced Brier, logarithmic, and
reliability metrics. These are joint predictive particles; their weights do
not identify which physical input was wrong.

## Reference systems

- Pyretechnics: <https://pypi.org/project/pyretechnics/>
- ELMFIRE technical reference: <https://elmfire.io/tech_ref.html>
- ELMFIRE crown verification: <https://elmfire.io/verification/verification_02.html>
- ELMFIRE spotting: <https://elmfire.io/user_guide/spotting.html>
- Community Fire Behavior Model:
  <https://gmd.copernicus.org/articles/19/3035/2026/>
- WRF-Fire user guide:
  <https://www2.mmm.ucar.edu/wrf/site/documentation/users_guide/fire.html>
- JaxWildfire: <https://arxiv.org/abs/2512.06102>
- PyTorchFire: <https://arxiv.org/abs/2502.18738>
