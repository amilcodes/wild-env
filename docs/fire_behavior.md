# Fire-behavior model and validation contract

## Scope

Version 0.3 has two execution paths over one local fire-behavior model:

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
relative humidity and optional precipitation rate. Dead 1/10/100-hour moisture
approaches Simard equilibrium moisture with 1/10/100-hour exponential time
lags. Rain and water drops wet fuels; water also reduces current intensity.
Live-herbaceous, live-woody and foliar moisture remain scenario fields.

This is a fuel-conditioning approximation. A research result must state
whether moisture was observed, initialized, calibrated or left at the
scenario assumption.

## Front propagation

Propagation accumulates directional travel fraction from the active perimeter
to eight neighboring targets. Adaptive substeps enforce a configured
cell-fraction CFL bound and permit a fast front to traverse multiple cells in a
minute. Flaming cells remain part of the propagating front until it has passed,
is held, or reaches the residence bound. This avoids the coarse-grid failure
where a slow fire self-extinguishes at a cell center before reaching the next
cell.

This method is an accelerator-friendly Huygens raster front. It is not the
WENO5/RK3 level-set solver used by WRF-Fire and the Community Fire Behavior
Model. Grid convergence, rotational invariance and arrival-time error remain
required evidence for any predictive use.

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
separately. An effective multiplier can absorb missing weather, conditioning
and suppression; it must not be interpreted as an identified physical
parameter.

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
