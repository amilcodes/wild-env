# Fire-state and behavior research increment

## Research question

The increment tests whether the fast training simulator can move from a
directionally biased raster-front approximation toward the numerical and
state-estimation structure used in current research fire models, while
remaining suitable for large policy ensembles.

The work was divided into four hypotheses:

1. A signed level set with high-order spatial and temporal integration will
   reduce grid-orientation and arrival-time artifacts.
2. Fire state should retain continuous perimeter geometry and arrival time,
   rather than only a phase label.
3. Historical perimeter calibration should produce a probability distribution,
   because spread, wind exposure and fuel moisture are not separately
   identifiable from one perimeter transition.
4. The reference and accelerator kernels must implement the same front
   semantics before throughput claims are useful.

## Reference model class

WRF-Fire and the 2026 Community Fire Behavior Model (CFBM) use a signed level
set, third-order Runge-Kutta integration, WENO5 derivatives near the front and
reinitialization of the signed-distance field. CFBM can be run standalone or
coupled through ESMF/NUOPC, and its UFS implementation exchanges winds,
surface state and fire fluxes with the atmosphere.

WRF-SFIRE perimeter-assimilation studies show that instantaneous perimeter
ignition can produce an inconsistent coupled atmospheric state. Replaying an
arrival-time history between observed perimeters provides a spin-up path for
the fire and atmosphere. QUIC-Fire occupies a different point in the fidelity
space: it couples a fast 3-D wind solver to a cellular combustion model and
represents three-dimensional fuel structure.

This increment adopts the level-set numerical structure and uncertain
perimeter state. It does not claim the coupled atmosphere, plume or 3-D fuel
physics of those systems.

Primary technical references:

- WRF-Fire user guide:
  <https://www2.mmm.ucar.edu/wrf/site/documentation/users_guide/fire.html>
- Jiménez y Muñoz et al. (2026), CFBM:
  <https://doi.org/10.5194/gmd-19-3035-2026>
- Kochanski et al. (2023), perimeter initialization and spin-up:
  <https://doi.org/10.3389/ffgc.2023.1203578>
- Linn et al. (2020), QUIC-Fire:
  <https://doi.org/10.1016/j.envsoft.2019.104616>

## Implemented fire state

Truth now contains:

- categorical phase and surface/passive-crown/active-crown type;
- signed level-set distance in metres;
- sub-minute cell arrival time;
- intensity, heading rate, flame length and remaining fuel;
- 1/10/100-hour dead-fuel moisture and canopy/fuel fields;
- spatial treatment, barrier and residual-spread fields.

Belief now contains:

- burn probability rather than only a hard reached/not-reached flag;
- intensity mean and standard deviation;
- arrival-time mean and standard deviation;
- source and delivery time of the most recent observation.

Perimeter assimilation converts an observed mask to a signed distance and a
localization-aware probability field. The observation is combined in log-odds
space with the prior belief. Delayed local sensor measurements update the same
probability and arrival-time fields. These fields are recorded in replay.

## Front equation

The zero contour of \(\phi\) is the fire perimeter, with negative \(\phi\)
inside. The solver advances

\[
\phi_t + R(\mathbf{n}) \lVert \nabla \phi \rVert = 0,
\]

where \(\mathbf{n}\) is the local front normal and the directional rate is

\[
R(\mathbf{n}) =
R_h\frac{1-e}{1-e(\mathbf{h}\cdot\mathbf{n})}.
\]

Here \(R_h\), \(e\), and heading unit vector \(\mathbf{h}\) come from the local
surface/crown behavior calculation. One-sided Jiang-Shu WENO5 derivatives feed
a Godunov upwind gradient norm. Time integration uses SSP-RK3. The CFL bound
sets adaptive substeps.

The Hamiltonian is active only in a configured narrow band and on the
connected one-cell exterior of the current fire. This support constraint is
required for barriers: evolving every signed-distance contour independently
can otherwise create a disconnected zero contour beyond a zero-speed cell.

NumPy reinitialization uses an exact Euclidean distance transform. PyTorch
solves the standard pseudo-time reinitialization equation on device. The
accelerator state remains resident across behavior lookup, moisture, level-set
propagation, reinitialization, crown transition and ember transport.

## Forcing and uncertain parameters

Weather forcing accepts either incident-wide time series `(time,)` or aligned
fields `(time, y, x)`. Wind direction is interpolated on the unit circle.
Fire-behavior configuration exposes auditable adjustments for wind speed,
wind-direction bias and dead-fuel moisture.

Historical calibration samples joint particles over:

- effective surface/crown spread adjustment;
- wind-speed exposure multiplier;
- wind-direction bias;
- dead-fuel moisture bias.

An observation-error scale controls a pseudo-likelihood combining symmetric
final-perimeter displacement and log cumulative-area ratio. Incremental growth
uses pseudocount-stabilized log area ratio and one-cell-tolerance F1, which
remains finite when a particle predicts no new cells. Adaptive likelihood
tempering retains at least 35% of the finite ensemble's effective sample size;
the untempered ESS and applied likelihood power are both reported. Posterior weights generate a burn
probability field and conditional arrival-time moments. Effective sample size
and posterior entropy diagnose collapse. Systematic resampling is available
for sequential filters.

The particles express predictive uncertainty. Their posterior does not
identify which input or missing process caused the discrepancy.

## Numerical verification

`aeolus-fire verify-front` produces a machine-readable report and figure.

| Check | Result |
|---|---:|
| Circular front, WENO5, 15 m cells, 30 min | 1.65 m equivalent-radius error |
| Circular front, Godunov, 15 m cells, 30 min | 2.05 m equivalent-radius error |
| Eight-angle anisotropic area coefficient of variation | 0.60% |
| Eight-angle heading-extent coefficient of variation | 1.82% |
| Accelerator barrier crossing in a spanning-barrier test | zero cells |

The full automated suite also checks exact linear WENO derivatives,
signed-distance sign, spatial-weather NetCDF round trips, circular wind
interpolation, probabilistic score behavior and ensemble normalization.

## Throughput

On the local Apple MPS device, a batch of 64 independent 128 × 128 landscapes
advanced for 20 fire minutes at 2.83 million cell-steps/s and 173 environment
minutes/s, using 77 MiB of resident state. This is a workstation measurement,
not a CUDA-cluster result. Historical member forecasts are dispatched through
a persistent spawn-safe process pool; cluster manifests can set
`parallel_workers` or `AEOLUS_EVAL_WORKERS`.

## Remaining model gap

The front discretization is now in the WRF-Fire/CFBM numerical class. Overall
physical fidelity is still below those coupled systems because the simulator
does not solve fire-modified winds, sensible/latent heat exchange, plume
dynamics or smoke transport. It is below QUIC-Fire for fine-scale 3-D fuel and
flow interaction. Additional gaps are:

- incident-grade gridded winds and fuel-moisture initialization;
- 1000-hour and live-fuel moisture evolution;
- spotting ignition delay and validation against ember observations;
- arrival-history spin-up between multiple observed perimeters;
- coupled-model distillation or correction fields for training ensembles;
- independent experimental validation of crown transition and treatment
  response; and
- held-out historical improvement over persistence on advancing-front metrics.

The frozen six-incident NIROPS study contains 24 held-out transitions, 21 with
observed growth. The posterior ensemble reaches cumulative IoU 0.862 and
167 m mean boundary displacement, close to persistence at 0.873 and 156 m.
Its forecast-independent active-domain balanced Brier score is 0.470 versus
0.500 for persistence, or +6.0% skill. Thresholded advancing-front F1 is only
0.096. The immediate research gate is therefore localization: improve
held-out active-growth scores while preserving numerical verification and
proper-score skill.
