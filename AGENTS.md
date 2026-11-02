# Working on wild-env

- Run compute-heavy benchmarks, experiments, training, and data collection on
  the user's existing EC2 compute host, not their laptop. Follow the established
  RSGbot workflow: SSH/git-bundle sync, an isolated project environment, persistent
  output directories, low process priority, and capped numerical-library threads.
- Do not interrupt or reconfigure existing RSG jobs. Do not resize or provision
  cloud instances as part of ordinary benchmark execution.
- Keep this checkout for light edits and inspection. The EC2 recipe is in
  `docs/ec2_agent_benchmarks.md`; credentials and host details stay outside Git.
- Results should be JSON/JSONL/CSV and concise experiment notes. Do not build
  dashboards, HTML reports, or presentation interfaces unless explicitly requested.
- Preserve the separation of public agent observations and evaluator-only state.
  Do not label fixture/baseline output as live frontier-model results.
