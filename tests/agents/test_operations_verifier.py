"""Hand-solved reached-state checks independent of the environment's transitions."""

import runpy

import pytest

pytest.importorskip("scipy.optimize")
verify = runpy.run_path("tools/validate_operations_states.py")["optimum"]


def public_packet():
    def job(task, durations, deadline, value, water=0, prerequisites=(), cost=1, **status):
        return {
            "id": task,
            "kind": "PROTECT",
            "duration_by_resource": dict(zip(("a", "b"), durations, strict=True)),
            "deadline_tick": deadline,
            "release_tick": 0,
            "value_by_forecast": {"alpha": value, "beta": value},
            "water_required": water,
            "start_cost": cost,
            "prerequisites": list(prerequisites),
            "started": False,
            "completed": False,
            **status,
        }

    return {
        "forecast": "alpha",
        "horizon_ticks": 5,
        "tick": 1,
        "water_remaining": 2,
        "resources": [
            {"id": "a", "active_task": 11, "available_at_tick": 2},
            {"id": "b", "active_task": 0, "available_at_tick": 1},
        ],
        "tasks": [
            {"id": 0},
            job(11, (2, 2), 2, 0, started=True),
            job(12, (1, 2), 3, 10, water=1, prerequisites=(11,)),
            job(13, (0, 2), 3, 9, water=2),
            job(14, (2, 2), 5, 14, water=1),
            job(15, (1, 0), 2, 50),
            job(16, (1, 1), 4, 7, prerequisites=(12,)),
            job(17, (1, 1), 1, 5, cost=2, started=True, completed=True),
        ],
    }


def test_public_mip_respects_busy_resources_sunk_costs_water_and_active_prerequisites():
    # Past/committed reward: -1 + (5 - 2) = 2.
    # A: finish 11 at 2, do 12 at [2,3), then 14 at [3,5).
    # B: do 16 at [3,4). New net reward = 9 + 13 + 6 = 28.
    # Job 15 has already become impossible; job 13 spends water needed for 12/14.
    obs = public_packet()
    assert verify(obs) == pytest.approx(30)
    obs["forecast"] = "unknown"  # Irrelevant uncertainty must not change this value.
    assert verify(obs) == pytest.approx(30)


def test_public_mip_refuses_to_use_hidden_forecast_for_conditional_values():
    obs = public_packet()
    obs["forecast"] = "unknown"
    obs["tasks"][2]["value_by_forecast"]["beta"] = 0
    with pytest.raises(ValueError, match="before revelation"):
        verify(obs)
