# Contributing

## Local verification

Use Python 3.10 or 3.11 from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[geo,render,dev]'
ruff check src tests tools
pytest -q
```

The reference aviation scenario currently resolves its measured delivery table
relative to the repository root. Run the test suite from that directory.

The optional native smoke target is independent of the Python extension
boundary:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

The reviewed GitHub Actions definition is stored at
`docs/ci/github-actions-ci.yml`. Copy it to `.github/workflows/ci.yml` when the
repository credential used for that change has GitHub's `workflow` scope.

## Research changes

A change that alters physical behavior, suppression efficacy, observation
timing, reward, or a historical score should include:

- the governing assumption or equation;
- a focused deterministic test;
- the experiment configuration and random-seed contract;
- comparison with the prior mechanism or a simpler baseline;
- a statement of what the result does and does not establish;
- an update to the relevant limitation or execution-plan document.

Do not calibrate on frozen test incidents. Changes discovered while inspecting
test outcomes require a new versioned benchmark contract before evaluation.

## Artifacts

Follow [`docs/repository_artifact_policy.md`](docs/repository_artifact_policy.md).
Avoid committing credentials, raw incident downloads, checkpoints, replay
stores, local logs, or materialized geospatial cubes.
