"""Model-facing prompt and public observation for the plan benchmark.

The observation is the public instance only: no optimum, no hidden world, no
baseline output. Job listing order and numeric IDs are randomized by the generator.
"""

from __future__ import annotations

from typing import Any

from aeolus.agents.operations import Instance
from aeolus.ladder.plan import WORLDS

PROTOCOL = "wildfire-ops-v3"

SYSTEM_PROMPT = """You are the planning section chief for a synthetic wildfire incident.
Write ONE incident action plan that maximizes EXPECTED net protected value.
Values are abstract planning units, not dollars or lives.

TIME. Ticks are 10 minutes, numbered 0 .. horizon_ticks-1 for starts.

TEAMS AND JOBS.
- A team can do a job only if the team appears in that job's duration_by_resource.
- A job started at tick t by team r takes d = duration_by_resource[r] ticks (travel,
  work and return included). The team is busy on ticks t .. t+d-1, the job completes
  at tick t+d, and the team is free again at tick t+d.
- A job may START at tick t only if ALL hold: t >= release_tick; t + d <= deadline_tick
  and t + d <= horizon_ticks; every prerequisite job has COMPLETED by tick t; the job
  has never been started before (each job at most once); water_remaining >= water_required.
- Water: water_required is consumed when the job starts. The incident supply is never refilled.
- Starting a job costs start_cost (also in the world where it turns out to be worthless).
  A completed job earns value_by_forecast[actual world]. ACCESS jobs earn nothing
  themselves but are prerequisites for other jobs.

FORECAST. Exactly one wind corridor, alpha or beta, is true, with the public prior.
It becomes known at the scheduled bulletin tick, or earlier at the completion tick of
the first SURVEY job you complete. SURVEY jobs earn nothing themselves and cannot start
once the forecast is known. Before the reveal you must act without knowing the world.

PLAN FORMAT. A list of entries {resource, job, start, branch}.
- branch "both": executed whichever world is true.
- branch "alpha" / "beta": executed only in that world, and its start must be >= the
  reveal tick (the bulletin tick, or the completion tick of your first scheduled survey
  if that is earlier). Earlier world-specific entries are rejected.
- Use exact team ids and numeric job ids from the packet.

EXECUTION (deterministic, no retries). For the actual world, each team takes its
entries (branch "both" plus that world's branch) in order of start, ties by listing
order. An entry begins at the first tick >= its start at which the team is free. If
the job is not startable at that tick, the entry is skipped (no cost, no value) and
the team moves to its next entry. Within one tick, teams are processed in the order
of the resources list (this matters only if two teams want the same job or the last water).

SCORE. Sum over worlds of prior[world] x (values of jobs completed in that world minus
start costs of jobs started in that world). It is compared with a proven optimal plan.
A shorter, feasible plan beats a long plan with skipped entries only if it earns more.

Return only the JSON object {"rationale": <at most a few sentences>, "plan": [...]}.
"""


def observation(instance: Instance) -> dict[str, Any]:
    uncertain = any(job.values[0] != job.values[1] for job in instance.jobs)
    jobs = []
    for job in instance.jobs:
        jobs.append(
            {
                "id": job.id,
                "label": job.label,
                "kind": job.kind,
                "duration_by_resource": {
                    name: d for name, d in zip(instance.resources, job.durations, strict=True) if d > 0
                },
                "release_tick": job.release,
                "deadline_tick": job.deadline,
                "prerequisites": list(job.prerequisites),
                "water_required": job.water,
                "start_cost": job.cost,
                "value_by_forecast": dict(zip(WORLDS, job.values, strict=True)),
            }
        )
    return {
        "protocol": PROTOCOL,
        "tick_minutes": 10,
        "horizon_ticks": instance.horizon,
        "water_available": instance.water,
        "forecast": {
            "worlds": list(WORLDS),
            "prior": dict(zip(WORLDS, instance.prior, strict=True)),
            "bulletin_tick": instance.bulletin_tick if uncertain else None,
            "note": "values differ by world; plan for both"
            if uncertain
            else "every job has the same value in both worlds; use branch 'both'",
        },
        "resources": list(instance.resources),
        "jobs": jobs,
    }
