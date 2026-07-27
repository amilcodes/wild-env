# Aeolus-IA research stack

Aeolus-IA is a Python-first environment for tactical aerial resource allocation
during wildfire initial attack. It is deliberately separated into four layers:

1. **Truth simulator** — fire, treatments, ground engagement, resources,
   logistics, and mission execution.
2. **Observation/belief model** — delayed, noisy perimeter updates and task
   candidates. Execution policies never receive the truth state.
3. **Decision interfaces** — PettingZoo Parallel API, an optional RLlib adapter,
   and a fixed-shape batched interface for the included MAPPO learner.
4. **Experiment system** — explicit YAML configuration, common-random-number
   evaluation, seed manifests, structured episode records, Docker, and Slurm.

The default kernel is a fast, semi-empirical surface-spread approximation. Its
rate-of-spread parameters are interpretable, but it has not been calibrated or
validated as a fire-behavior model. Validation against Behave/FlamMap, SimFire,
Cell2Fire, and selected historical cases is a planned research activity, not an
accomplished claim.

## Local setup

```bash
cd aeolus_py
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
aeolus-train --config configs/smoke.yaml
aeolus-eval --config configs/smoke.yaml --checkpoint runs/smoke/checkpoint.pt
```

For CUDA, use the PyTorch installation command appropriate to the cluster's
driver/CUDA version, then install the project with `--no-deps` if necessary.
`requirements/base.lock` records the tested direct dependencies, while
`requirements/cuda-cu128.lock` is the CUDA 12.8 container constraint set.
For multi-node training, use `deploy/slurm/train.sbatch`; the learner uses
`torchrun` and DDP when launched with `WORLD_SIZE > 1`.

## Interfaces

- `aeolus.envs.parallel.AeolusParallelEnv`: PettingZoo `ParallelEnv`, suitable
  for standards testing, heuristic policies, and third-party MARL libraries.
- `aeolus.envs.rllib.AeolusRLlibEnv`: optional RLlib `MultiAgentEnv` wrapper.
- `aeolus.training.train`: parameter-shared recurrent-style task-pointer MAPPO
  baseline with centralized critic and masked discrete actions.

The action at each tactical decision is a task index for each resource. The
environment resolves an invalid or conflicting assignment deterministically,
records it, and returns the next decision event after the configured interval.
This makes the resource-task matching semantics inspectable rather than hiding
them inside a raster action head.

See `docs/architecture.md` and `docs/cluster.md` for the execution and cluster
contracts.
