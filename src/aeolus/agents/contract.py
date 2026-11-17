"""Versioned, provider-independent agent protocol. No evaluator state is exposed."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "wildfire-desk-v1.1"
SYSTEM_PROMPT = """You command a simulated wildfire initial-attack desk.
Minimize final asset-weighted fire loss, then resource operating cost and exposure.
This is a research simulator, not operational guidance. You receive only delayed
fire belief, static mapped assets, current weather, resource telemetry, service
stocks, and feasible tasks. Unknown cells are not evidence of no fire. Task value
is a heuristic, not a prediction or ground-truth reward. No future weather or
hidden fire state is available. Reconnaissance can improve subsequent observations.

Call submit_dispatch exactly once per decision. Include the current decision_id
and one assignment per resource, using a task_id from that resource's legal_tasks.
Task IDs change every decision. Task 0 means hold/continue the current mission;
it does not cancel a flight. Respect each nonzero task's shared capacity. Invalid,
stale, or incomplete submissions cause the entire fleet to hold for one interval.
There are no free retries. A short operational rationale is optional (use an empty
string if unnecessary). Put any information to retain in memory (at most 2000
characters). Only memory and the last dispatch receipt persist between decisions.
Dispatch advances the simulation by decision_interval_min. The simulator handles
travel, payload delivery, return, queues, reload, and safety gates. A forecast or
proposal never changes the incident until dispatch is submitted. Terminal scores
are reserved for evaluation and will not be shown during the episode.

Task semantics: OBSERVE sends the sensor to collect a noisy local measurement;
delivery is delayed by observation_delay_min. WATER drops water on the target.
RETARDANT drops a line across the wind; REINFORCE and AERIAL_LINE use the listed
heading (AERIAL_LINE also supports water aircraft). LINE sends a ground crew to
construct a line over time. SERVICE sends an aircraft to the listed site to
refill compatible payload and/or refuel, subject to stocks, queues and opening
hours. With service_sites present, an aircraft becomes AVAILABLE at its mission
target after a drop or observation: choose SERVICE explicitly when it needs
refilling. With no service_sites, return and reload happen automatically.
Only AVAILABLE resources accept a new nonzero task; busy resources continue
with task 0. travel_min is a geometric estimate, excluding dispatch latency and
service time. legal_tasks already enforces current feasibility and reserve gates.
Observation ages mean time since delivery to belief, not time since capture;
measurements are older by the configured delivery delay (except initial belief).
Coordinates are grid column x and row y. Sector asset values are static totals;
heuristic_value ranks belief-derived candidate tasks and is not an outcome oracle.
"""

DISPATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision_id": {"type": "string"},
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"resource_id": {"type": "string"}, "task_id": {"type": "integer"}},
                "required": ["resource_id", "task_id"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string", "maxLength": 1000},
        "memory": {"type": "string", "maxLength": 2000},
    },
    "required": ["decision_id", "assignments", "rationale", "memory"],
    "additionalProperties": False,
}
TOOL_DESCRIPTION = (
    "Commit one joint resource dispatch and advance to the next decision. "
    "Use the current decision_id and legal task IDs; assign each resource once. "
    "Shared task capacities apply across resources. Invalid submissions advance with all resources holding. "
    "Return a brief operational explanation and bounded memory for the next decision."
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parse_json(text: str) -> Any:
    """Reject non-JSON constants and duplicate keys rather than silently repairing output."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError("nonfinite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_dispatch(value: Any, observation: dict[str, Any]) -> dict[str, int]:
    """Validate atomically against the exact public observation, including joint capacity."""
    if not isinstance(value, dict) or set(value) != set(DISPATCH_SCHEMA["required"]):
        raise ValueError("dispatch must contain decision_id, assignments, rationale, and memory only")
    if value["decision_id"] != observation["decision_id"]:
        raise ValueError("stale or incorrect decision_id")
    for key, limit in (("rationale", 1000), ("memory", 2000)):
        if not isinstance(value[key], str) or len(value[key]) > limit:
            raise ValueError(f"{key} must be a string of at most {limit} characters")
    assignments = value["assignments"]
    resources = {resource["id"]: resource for resource in observation["resources"]}
    tasks = {task["id"]: task for task in observation["tasks"]}
    if not isinstance(assignments, list) or len(assignments) != len(resources):
        raise ValueError("assign every resource exactly once")
    actions: dict[str, int] = {}
    counts: dict[int, int] = {}
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {"resource_id", "task_id"}:
            raise ValueError("each assignment needs resource_id and task_id only")
        resource_id, task_id = item["resource_id"], item["task_id"]
        if not isinstance(resource_id, str) or resource_id not in resources or resource_id in actions:
            raise ValueError("unknown or duplicate resource_id")
        if type(task_id) is not int or task_id not in resources[resource_id]["legal_tasks"]:
            raise ValueError("task_id is not legal for this resource")
        counts[task_id] = counts.get(task_id, 0) + 1
        if task_id and counts[task_id] > tasks[task_id]["capacity"]:
            raise ValueError("joint dispatch exceeds task capacity")
        actions[resource_id] = task_id
    return actions
