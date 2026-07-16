from pathlib import Path

import pytest

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.tasks import TaskKind

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "aviation" / "us_wildfire_reference_operations.yaml"


def test_reference_operations_config_loads_traceable_mixed_fleet() -> None:
    experiment = load_config(CONFIG_PATH)
    resources = experiment.scenario.resources

    assert len(resources) == 9
    assert {resource.kind for resource in resources} == {
        "retardant",
        "water",
        "sensor",
    }
    assert all(resource.vehicle_profile_id for resource in resources)
    assert all(resource.performance_evidence_grade == "scenario_assumption" for resource in resources)
    assert {resource.autonomy_level for resource in resources} == {
        "crewed",
        "remotely_piloted",
        "supervised_autonomy",
    }
    assert all(isinstance(resource.operational_roles, tuple) for resource in resources)
    s2t = [resource for resource in resources if resource.vehicle_profile_id == "calfire-s2t-retardant"]
    assert len(s2t) == 2
    assert all(resource.delivery_surface_path for resource in s2t)
    assert all(resource.delivery_evidence_grade == "engineering_validated" for resource in s2t)


def test_reference_s2t_drop_uses_measured_line_table() -> None:
    experiment = load_config(CONFIG_PATH)
    simulator = AeolusSimulator(experiment.scenario)
    resource = next(
        item for item in simulator.state.resources if item.spec.vehicle_profile_id == "calfire-s2t-retardant"
    )
    x, y = experiment.scenario.width // 2, experiment.scenario.height // 2
    simulator.state.truth.fuel_model_number[y, x] = 143
    simulator.state.truth.intensity_kw_m[y, x] = 500.0
    resource.target_xy = (x, y)
    resource.task_kind = int(TaskKind.RETARDANT)
    resource.payload_fraction = 1.0

    simulator._execute_mission(resource)

    event = next(item for item in reversed(simulator.state.events) if item["kind"] == "retardant_drop")
    assert event["delivery_surface_id"] == "calfire-s2t-mtdc-2006-gum-v1"
    assert event["requested_coverage_gpc"] == 3.0
    assert event["drop_length_m"] == pytest.approx(595.0 * 0.3048)
    assert event["drop_width_m"] < 25.0
    assert event["peak_coverage_gpc"] < 3.0
    assert event["peak_effective_coverage_gpc"] == 3.0
