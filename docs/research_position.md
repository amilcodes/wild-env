# Research position and simulator fidelity

## What the system is for

The current research question is whether decentralized policies can allocate a
heterogeneous initial-attack fleet under delayed fire information, mission
latency, task competition, payload/reload constraints and uncertain spread.
The relevant output is a policy comparison with a documented validity
envelope. A visually plausible fire animation is not evidence by itself.

## Reference classes

The simulator landscape has three useful fidelity classes:

| Class | Examples | Primary strength | Cost / limitation |
|---|---|---|---|
| fast cellular research simulators | SimFire, Cell2Fire, JaxWildfire | many episodes, control/RL integration, reproducibility | simplified fire behavior and suppression |
| operational/semi-empirical spread tools | Behave/FlamMap/FARSITE family | mature fuel and fire-behavior workflows | weaker fit for millions of closed-loop MARL steps |
| coupled physics models | WRF-Fire, FIRETEC, QUIC-Fire | atmosphere/fire or high-resolution plume/flow behavior | expensive setup and execution; unsuitable as the sole training kernel |

Aeolus v0.2 belongs in the first class. Its distinguishing content is the
decision problem: actor/critic information separation, heterogeneous logistics,
belief-driven candidate tasks, capacity conflicts, exact assignment baseline,
historical timestamp assimilation, and causal-mode separation. Its fire kernel
is less mature than established fire simulators and has not been calibrated.

Recent wildfire MARL papers commonly study grid surveillance, suppression
motion, task allocation or aerial coordination in synthetic environments. That
literature supports MARL as a decision method but does not establish that a
particular cellular fire model transfers to operations. This project therefore
treats fire-model validation and policy evaluation as separate workstreams.

## Historical evaluation modes

Three modes answer different questions:

1. **Open-loop hindcast** initializes from perimeter A and predicts perimeter B
   without injecting intermediate truth. With `no_aerial`, this measures spread
   model error. Intervention runs are hypothetical branches.
2. **Shadow replay** injects each historical perimeter into policy belief only
   after its timestamp. Actions are logged, while historical observations are
   never presented as causal outcomes of simulated drops.
3. **Paired counterfactual** starts multiple policies from the same observed
   perimeter and random seeds. Differences are attributable to modelled policy
   actions inside the simulator, not to the historical fire record.

FEDS/FIRMS timestamps are valuable observations but do not encode hidden
suppression actions, incident command decisions, detection limits, plume
occlusion or all relevant weather. They cannot identify an “optimal historical
response” on their own.

## Current validity envelope

Version 0.2 is suitable for:

- software and MARL method development;
- ablation of observation delay, resource mix and assignment semantics;
- repeatable paired comparisons inside the stated kernel;
- import/replay of public perimeter timestamps and terrain;
- profiling the policy/environment boundary before native-kernel work.

It is not yet suitable for:

- operational dispatch recommendations;
- claimed prediction of real perimeter location or containment probability;
- crown-fire or plume-dominated incidents;
- evaluating retardant effectiveness without calibrated treatment data;
- comparing agencies, crews or historical incident decisions.

## Evidence required for a defensible result

1. Calibrate no-suppression spread parameters on training incidents only.
2. Report held-out spatial IoU, symmetric difference, area and arrival-time
   error with uncertainty intervals.
3. Validate treatment response against independent experimental or operational
   fireline-effectiveness records.
4. Compare MAPPO against no-action, doctrine heuristic, exact assignment and
   stochastic model-predictive control.
5. Use paired seeds, geographic holdouts, multiple resource/weather regimes and
   robustness tests for observation failure.
6. Report simulator throughput, policy inference time, constraint violations,
   exposure and resource utilization alongside loss/containment.

## Primary references

- SimFire: <https://arxiv.org/abs/2311.15925>
- Cell2Fire: <https://arxiv.org/abs/1905.09317>
- JaxWildfire: <https://arxiv.org/abs/2512.06102>
- WRF-Fire user guide: <https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.4/users_guide_chap-fire.html>
- QUIC-Fire: <https://research.fs.usda.gov/treesearch/59686>
- NASA FEDS: <https://firms.modaps.eosdis.nasa.gov/descriptions/FEDS_VIIRS_SNPP.html>
- LANDFIRE landscape products: <https://landfire.gov/fuel/landscape>
- USFS Fireline Effectiveness dashboard: <https://research.fs.usda.gov/rmrs/products/dataandtools/fireline-effectiveness-fle-dashboard>
- OGC STAC: <https://www.ogc.org/standards/stac/>
- CF conventions: <https://cfconventions.org/cf-conventions/cf-conventions.html>
