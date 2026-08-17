from __future__ import annotations

from pathlib import Path

import numpy as np

from aeolus.evaluation.baselines import (
    isotropic_spread_prediction,
    recent_equivalent_radius_rate_m_min,
    wind_ellipse_prediction,
)
from aeolus.evaluation.frozen_benchmark import (
    audit_frozen_contract,
    load_frozen_contract,
)


def test_isotropic_and_recent_growth_baselines_are_causal() -> None:
    previous = np.zeros((41, 41), dtype=np.bool_)
    previous[19:22, 19:22] = True
    current = isotropic_spread_prediction(
        previous,
        rate_m_min=1.0,
        duration_min=60.0,
        cell_size_m=30.0,
    )
    rate = recent_equivalent_radius_rate_m_min(
        previous,
        current,
        elapsed_min=60.0,
        cell_size_m=30.0,
    )
    forecast = isotropic_spread_prediction(
        current,
        rate_m_min=rate,
        duration_min=60.0,
        cell_size_m=30.0,
    )
    assert current.sum() > previous.sum()
    assert forecast.sum() > current.sum()
    assert np.all(forecast[previous])


def test_wind_ellipse_extends_farther_downwind() -> None:
    initial = np.zeros((61, 61), dtype=np.bool_)
    initial[30, 30] = True
    # Wind from west propagates toward the east.
    predicted = wind_ellipse_prediction(
        initial,
        head_rate_m_min=2.0,
        duration_min=300.0,
        cell_size_m=30.0,
        wind_from_direction_deg=270.0,
    )
    columns = np.flatnonzero(predicted[30])
    assert columns.max() - 30 > 30 - columns.min()


def test_frozen_contract_has_disjoint_chronological_incident_splits() -> None:
    root = Path(__file__).parents[1]
    pilot = load_frozen_contract(root / "configs" / "historical_validation_frozen_pilot.yaml")
    audit = audit_frozen_contract(pilot)
    assert audit["valid"]
    assert audit["counts"] == {"development": 2, "train": 2, "test": 2}
    assert not audit["test_targets_used_for_fitting"]

    expanded = load_frozen_contract(root / "configs" / "historical_validation_frozen_36.yaml")
    expanded_audit = audit_frozen_contract(expanded)
    assert expanded_audit["valid"]
    assert expanded_audit["counts"] == {
        "train": 22,
        "development": 7,
        "test": 7,
    }
