from __future__ import annotations

from aeolus.config import ScenarioConfig
from aeolus.core.simulator import AeolusSimulator
from aeolus.policies import greedy_value
from aeolus.replay import ReplayBundle, record_episode, render_frame_2d, render_frame_3d


def test_replay_round_trip_and_render(tmp_path) -> None:
    config = ScenarioConfig(
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
    image_2d = render_frame_2d(restored, tmp_path / "frame-2d.png")
    image_3d = render_frame_3d(restored, tmp_path / "frame-3d.png")
    assert image_2d.stat().st_size > 10_000
    assert image_3d.stat().st_size > 10_000
