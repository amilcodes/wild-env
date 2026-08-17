from __future__ import annotations

import numpy as np
import pytest

from aeolus.evaluation.observation import (
    AcquisitionWindow,
    RasterObservationModel,
    acquisition_aware_observation_metrics,
    acquisition_window_burn_probability,
    expected_raster_observation_probability,
    interval_censored_arrival_score,
    perimeter_observation_envelope,
    soft_perimeter_probability,
    uncertainty_aware_perimeter_metrics,
)


def test_soft_perimeter_probability_respects_distance_and_zero_sigma() -> None:
    mask = np.zeros((9, 9), dtype=np.bool_)
    mask[2:7, 2:7] = True
    assert np.array_equal(
        soft_perimeter_probability(mask, sigma_m=0.0, cell_size_m=30.0),
        mask.astype(np.float32),
    )
    probability = soft_perimeter_probability(mask, sigma_m=45.0, cell_size_m=30.0)
    assert probability[4, 4] > probability[2, 2] > 0.5
    assert probability[2, 1] < 0.5
    assert probability[0, 0] < probability[2, 1]


def test_uncertainty_aware_metrics_reward_near_boundary_displacement() -> None:
    observed = np.zeros((31, 31), dtype=np.bool_)
    observed[8:23, 8:23] = True
    near = np.zeros_like(observed)
    near[8:23, 9:24] = True
    far = np.zeros_like(observed)
    far[8:23, 15:30] = True
    near_metrics = uncertainty_aware_perimeter_metrics(
        near.astype(np.float32),
        observed,
        sigma_m=60.0,
        cell_size_m=30.0,
    )
    far_metrics = uncertainty_aware_perimeter_metrics(
        far.astype(np.float32),
        observed,
        sigma_m=60.0,
        cell_size_m=30.0,
    )
    assert near_metrics["soft_iou"] > far_metrics["soft_iou"]
    assert near_metrics["soft_brier_score"] < far_metrics["soft_brier_score"]
    assert (
        near_metrics["predicted_boundary_envelope_coverage"]
        > far_metrics["predicted_boundary_envelope_coverage"]
    )


def test_observation_envelope_expands_with_sigma() -> None:
    mask = np.zeros((21, 21), dtype=np.bool_)
    mask[7:14, 7:14] = True
    narrow = perimeter_observation_envelope(
        mask,
        sigma_m=15.0,
        cell_size_m=30.0,
    )
    broad = perimeter_observation_envelope(
        mask,
        sigma_m=90.0,
        cell_size_m=30.0,
    )
    assert narrow.sum() < broad.sum()


def test_interval_censored_arrival_loss() -> None:
    start = np.zeros((2, 3), dtype=np.bool_)
    start[0, 0] = True
    target = start.copy()
    target[0, 1] = True
    arrival = np.full((2, 3), np.inf)
    arrival[0, 0] = 0.0
    arrival[0, 1] = 90.0
    arrival[1, 0] = 30.0
    score = interval_censored_arrival_score(
        arrival,
        start,
        target,
        interval_minutes=120.0,
        time_scale_minutes=60.0,
    )
    assert score["newly_observed_cell_count"] == 1.0
    assert score["right_censored_cell_count"] == 4.0
    assert score["interval_censored_arrival_mae_scaled"] == pytest.approx(90.0 / 5.0 / 60.0)


def test_acquisition_window_integrates_arrival_probability() -> None:
    arrival = np.asarray(
        [
            [-10.0, 0.0, 30.0],
            [60.0, 90.0, np.inf],
        ]
    )
    acquisition = AcquisitionWindow(
        start_minute=0.0,
        end_minute=60.0,
        available_minute=85.0,
    )
    probability = acquisition_window_burn_probability(arrival, acquisition)
    assert np.allclose(
        probability,
        [[1.0, 1.0, 0.5], [0.0, 0.0, 0.0]],
    )
    assert acquisition.processing_latency_minute == 25.0


def test_acquisition_aware_likelihood_handles_detection_and_obscuration() -> None:
    yy, xx = np.mgrid[:31, :31]
    arrival = (np.hypot(xx - 15.0, yy - 15.0) - 3.0) * 10.0
    observed = arrival <= 30.0
    model = RasterObservationModel(
        acquisition=AcquisitionWindow(30.0, 30.0, 45.0),
        localization_sigma_m=0.0,
        detection_probability=0.98,
        false_alarm_probability=0.01,
    )
    expected = expected_raster_observation_probability(
        arrival,
        model=model,
        cell_size_m=30.0,
    )
    assert expected[15, 15] == pytest.approx(0.98)
    assert expected[0, 0] == pytest.approx(0.01)
    identity = acquisition_aware_observation_metrics(
        arrival,
        observed,
        model=model,
        cell_size_m=30.0,
    )
    shifted = acquisition_aware_observation_metrics(
        arrival + 35.0,
        observed,
        model=model,
        cell_size_m=30.0,
    )
    obscured = acquisition_aware_observation_metrics(
        arrival,
        observed,
        model=model,
        cell_size_m=30.0,
        obscured_probability=0.25,
    )
    assert identity["brier_score"] < shifted["brier_score"]
    assert identity["mean_log_score"] < shifted["mean_log_score"]
    assert obscured["effective_observed_cells"] == pytest.approx(0.75 * observed.size)


def test_acquisition_localization_spreads_probability_without_invalid_values() -> None:
    arrival = np.full((31, 31), np.inf)
    arrival[15, 15] = 0.0
    model = RasterObservationModel(
        acquisition=AcquisitionWindow(0.0, 0.0, 0.0),
        localization_sigma_m=60.0,
    )
    probability = expected_raster_observation_probability(
        arrival,
        model=model,
        cell_size_m=30.0,
    )
    assert 0.0 < probability[15, 14] < probability[15, 15] < 1.0
    assert np.all((0.0 <= probability) & (probability <= 1.0))
