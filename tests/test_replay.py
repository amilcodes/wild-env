from __future__ import annotations

import pytest

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.policies import greedy_value
from aeolus.replay import (
    ReplayBundle,
    export_paraview,
    record_episode,
    render_frame_2d,
    render_frame_3d,
)
from aeolus.viewer.model import ReplayModel


def test_replay_round_trip_and_render(tmp_path) -> None:
    config = ScenarioConfig(
        scenario_id="test-replay",
        title="Replay contract test",
        location_name="Synthetic test grid",
        time_origin="2026-08-17T14:00:00-07:00",
        seed=41,
        width=24,
        height=24,
        horizon_min=12,
        decision_interval_min=3,
        max_tasks=16,
        wind_speed_m_s=0.6,
        wind_variability=0.0,
        spotting_rate=0.0,
        ground_arrival_min=4,
    )
    replay = record_episode(
        AeolusSimulator(config),
        greedy_value,
        tmp_path / "replay",
        seed=41,
        policy_name="greedy-value",
    )
    restored = ReplayBundle.open(replay.root)
    assert restored.frame_count >= 2
    assert len(restored.events()) > 0
    assert "truth/fire_type" in restored.states
    assert "truth/spread_rate_m_min" in restored.states
    assert "truth/moisture_dead_1h" in restored.states
    assert "static/fuel_model_number" in restored.states
    assert "environment/wind_speed_m_s" in restored.states
    assert "resources/endurance_remaining_min" in restored.states
    assert restored.metadata["schema_version"] == 2
    assert restored.metadata["scenario_identity"]["id"] == "test-replay"
    model = ReplayModel(restored)
    assert model.title == "Replay contract test"
    assert model.location_name == "Synthetic test grid"
    assert model.clock_label(60) == "2026-08-17T15:00-07:00"
    conditions = model.conditions(0, 12, 12)
    assert conditions["wind_speed_m_s"] == pytest.approx(0.6)
    assert model.resource(0, 0)["endurance_remaining_min"] > 0
    image_2d = render_frame_2d(restored, tmp_path / "frame-2d.png")
    image_3d = render_frame_3d(restored, tmp_path / "frame-3d.png")
    assert image_2d.stat().st_size > 10_000
    assert image_3d.stat().st_size > 10_000
    pytest.importorskip("pyvista")
    paraview = export_paraview(restored, tmp_path / "paraview", max_frames=2)
    assert paraview.exists()
    assert len(list((tmp_path / "paraview" / "frames").glob("*.vtm"))) == 2
