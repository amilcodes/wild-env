# Development record

## Purpose

This record describes the chronology used to organize the private repository
history. It exists so that code, experiment results, and claims can be traced to
the underlying research records rather than inferred from commit count alone.

## Repository import

The simulator was developed inside the `aeolus_py/` subtree of a larger local
repository. The tested snapshot was reorganized into 42 subsystem-sized commits
dated from 2026-06-18 through 2026-08-15 to present the technical dependency
sequence in Git. These dates are an editorial reconstruction of the development
progression; they are not historical GitHub pull-request timestamps.

The remote state before reconstruction is retained on
`backup/pre-reconstruction-main-20260817` and the tested import is retained on
`backup/pre-reconstruction-import-20260817`.

The source repository contained these recorded milestones:

| Local date | Recorded change |
| --- | --- |
| 2026-07-26 | Initial wildfire MARL environment and cluster contracts |
| 2026-07-27 | Historical incident, replay, evaluation, and native-kernel research stack |
| 2026-07-28 | Operational fire equations and accelerator batches |
| 2026-07-28 | NIROPS historical-validation study and results paper |

The later evidence windows below come from file modification times, frozen
result manifests, study outputs, and the research notes in this repository.

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

Commit boundaries are organized for technical review and dependency order.
Scientific claims should be traced to frozen manifests and machine-readable
results rather than to commit count or GitHub activity.
