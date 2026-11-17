# Ladder results (`wildfire-ops-v3`)

| Directory | Split | Generator | What it is |
| --- | --- | --- | --- |
| `evaluation-v1/` | evaluation (seeds 100000+) | v3 | Frozen evaluation of the 8-model panel. The headline result. |
| `calibration-v1/` | development | v1 (front-loaded releases) | Difficulty calibration, tiers 3–6, 6 models. |
| `calibration-v2/` | development | v2 (spread windows) | Difficulty calibration, tiers 5–8, 5 models. |
| `packing-probe/` | development | prototype | 4 capacity-bound instances × 5 models; motivated tiers 6–8. |

Calibration runs used generator versions that were superseded before the evaluation
was frozen (the version is part of every instance's RNG seed). Their instances
can't be regenerated from the current code, so they're design evidence only and
never enter the evaluation. Every run directory keeps `calls.jsonl` with every
model response, usage, latency, skipped entry and score.
