# Run experiments on EC2

Use the already-authorized EC2 host from the RSGbot setup. Keep compute off the
laptop. The existing machine also serves live RSG jobs, so use a separate checkout
and Python environment, a single benchmark worker, `nice -n 15`, and one thread
per numerical library. Do not stop, restart, or reconfigure the live RSG jobs.
No new instance, GPU, public endpoint, or web server is needed.

Host, SSH key, and provider credentials are machine-local settings; do not commit
them. The following assumes `WILD_ENV_SSH_TARGET` and `WILD_ENV_SSH_KEY` have been
set locally to the existing authorized host and key path.

## Sync committed code

Use the same Git-bundle approach as RSGbot. No private key or local model login
is copied to EC2. If `git rev-parse --is-shallow-repository` prints `true`, run
`git fetch --unshallow origin` before creating a bundle for a fresh clone.
Use `git bundle verify` to check that it records complete history.

```bash
mkdir -p runs/transfers
git bundle create runs/transfers/wild-env.bundle codex/frontier-agent-benchmark
scp -i "$WILD_ENV_SSH_KEY" runs/transfers/wild-env.bundle "$WILD_ENV_SSH_TARGET:wild-env.bundle"
# First setup on the remote machine:
git clone -b codex/frontier-agent-benchmark ~/wild-env.bundle ~/wild-env-frontier
# Later updates, in the dedicated remote checkout:
git fetch ~/wild-env.bundle codex/frontier-agent-benchmark
git merge --ff-only FETCH_HEAD
```

Check `uptime`, `free -h`, `df -h`, and running jobs before starting. The simulator
requires Python 3.10–3.12; the host system Python may be newer. Install a separate
Python 3.11 under a project-owned tools directory, leaving the OS and RSG
Python environments intact. Install CPU-only PyTorch on this CPU host, then
`pip install -e '.[geo,met,render,dev]'` in this project's venv. No CUDA download
is necessary. Use the recorded numerical versions when reproducing old results.

## Execute and retrieve evidence

From the EC2 checkout, launch a detached remote job so loss of the SSH connection
does not kill it. Choose a new run directory each time:

```bash
mkdir -p runs
nohup setsid nice -n 15 bash deploy/ec2/run_agent_benchmark.sh \
  runs/ec2-baselines-v1 > runs/ec2-baselines-v1-launch.log 2>&1 < /dev/null &
```

The script caps OpenMP, BLAS, MKL, and NumExpr threads; runs lint and tests; executes
the fixed frontier suite; and replays every episode. It creates `exit_code`,
`tests.txt`, `lint.txt`, `progress.txt`, `console.json`, and a `benchmark/` directory
of JSON/JSONL evidence. No dashboard is generated. Inspect progress through SSH,
and retrieve artifacts with scp/rsync. `exit_code` is written on normal shell exit;
its absence means running/interrupted, not success.

Pass a remote agent manifest as the second argument to run models, and an alternate
suite as the third. Credentials remain in the remote environment or an existing
approved inference bridge. RSG's Claude/Codex bridges are separate services;
do not restart or change them to run simulator validation. A baseline run needs
no model credentials and consumes no model quota.
