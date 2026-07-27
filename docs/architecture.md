# Technical architecture

## Decision process

Aeolus implements a cooperative semi-Markov Dec-POMDP at the task-allocation
layer. Internal fire and mission dynamics advance at one-minute intervals;
agents act every `decision_interval_min` or at a configured interruption point.
An action is a task choice, not a low-level flight command.

The truth state contains fire phase/intensity, mutable fuels, treatment layers,
weather, resource mission state, ground-line state, assets, and the random
number-generator state. The policy observation contains only task features,
resource-local status, a delayed belief map summary, forecast summaries, and an
action mask. The centralized critic can receive an explicit global feature
vector during training.

## Task and matching semantics

Candidate tasks are generated from detected front segments: observe, direct
water support, retardant line construction, reinforce line, and hold. Each task
has a target, compatibility vector, airspace capacity, estimated travel time,
belief risk, ground dependency, and uncertainty. One resource-task assignment
is accepted per task by default. Conflicts are resolved in a fixed random-order
auction seeded from the episode seed, and both attempted and accepted actions
are logged.

## Fire and intervention kernel

The fast kernel uses fuel-model parameters, moisture, wind and slope to form a
Rothermel-style baseline surface rate of spread. Cell-to-cell ignition samples
from the resulting arrival hazard, then adds an explicitly separate correlated
residual field and short-range spotting process. It is a training kernel, not a
replacement for Behave, FlamMap/FARSITE, Cell2Fire, or coupled fire-atmosphere
models.

Water reduces intensity and temporarily changes the spread hazard. Retardant
applies an oriented coverage field with line continuity and ground-engagement
modifiers. A ground-line graph is abstracted as raster holding strength. All
intervention parameters are scenario-distributed and logged.

## Learning baseline

The included learner uses a shared task-pointer actor over resource-local
features and candidate-task features. Its centralized critic pools all resource
and task features. Action masks are applied before sampling. The learner is a
reproducible MAPPO baseline: clipped policy objective, GAE, entropy bonus,
value clipping, checkpointed optimizer/RNG/config state, and optional AMP/DDP.

This baseline does not establish a MARL result. Required comparisons include
heuristics, maximum-value bipartite assignment, stochastic MPC, and a
centralized graph-assignment policy.
