# Development record

## Purpose

This record separates the chronology supported by local evidence from the Git
history created during the public-repository migration. It exists so that code,
experiment results, and claims can be reviewed without inferring a development
timeline from reconstructed commits or pull requests.

## Repository import

The simulator was developed inside the `aeolus_py/` subtree of a larger local
repository. Four commits that already contained that subtree were extracted
with their original authorship and timestamps and joined to the existing
`amilcodes/wild-env` initial commit. Later work existed as a tested local
working-tree snapshot. It is committed during the migration in subsystem-sized
review units using the migration date.

The extracted commit sequence is:

| Local date | Recorded change |
| --- | --- |
| 2026-07-26 | Initial wildfire MARL environment and cluster contracts |
| 2026-07-27 | Historical incident, replay, evaluation, and native-kernel research stack |
| 2026-07-28 | Operational fire equations and accelerator batches |
| 2026-07-28 | NIROPS historical-validation study and results paper |

These are real Git commits. The later evidence windows below are reconstructed
from file modification times, frozen result manifests, study outputs, and the
research notes in this repository. They are not represented as historical pull
requests.

## Evidence-supported work sequence

| Evidence window | Work represented in the snapshot | Primary records |
| --- | --- | --- |
| 2026-07-29 | WENO/level-set front work, coupled suppression mechanics, two-perimeter initialization, localization, and initial native viewer | `docs/frontier_fire_research.md`, `docs/suppression_operations_research.md`, `docs/viewer.md` |
| 2026-07-30 | Historical accuracy analysis, observation uncertainty, fuel moisture, and fire-behavior documentation | `docs/historical_accuracy_report.md`, `docs/historical_fidelity_v4.md`, `docs/fire_behavior.md` |
| 2026-07-31 | Historical-fuel chronology repair, aviation evidence/catalog work, scenario configuration, and non-compute P1 study | `docs/historical_fuel_p0.md`, `docs/aviation_vehicle_closure.md`, `docs/noncompute_p1_research.md` |
| 2026-08-02 | Accelerator-resident tensor operations/incident environments, entity-attention MAPPO, cluster topology, and transfer gates | `docs/rl_compute_research.md`, `docs/tensor_incident_environment.md`, `docs/rl_training_execution_plan.md` |
| 2026-08-04 to 2026-08-05 | Incident-held-out evaluation, frozen 36-incident partition, GOFER progression import, HRRR forcing, and execution controls | `docs/heldout_historical_skill_v6.md`, `docs/background_accuracy_execution.md`, `docs/noncompute_p1_remaining_work.md` |
| 2026-08-12 | Canonical-to-tensor fidelity plan and local throughput/memory measurements | `docs/rl_surrogate_fidelity_execution_plan.md`, `results/tensor_incident/` |
| 2026-08-13 | Resumable frozen benchmark reached 29 of 36 prepared incidents before an external HRRR Zarr endpoint failure | `docs/background_accuracy_execution.md` and local run records retained outside Git |

## Interpretation

Commit boundaries in the migration are organized for technical review. They do
not claim that each subsystem was originally implemented in isolation or that a
pull request existed on the evidence date. Scientific claims should be traced
to frozen manifests and machine-readable results, not to commit count or GitHub
activity.
