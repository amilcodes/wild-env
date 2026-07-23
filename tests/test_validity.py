from datetime import datetime, timezone

import numpy as np

from aeolus.core.state import FireType
from aeolus.data import ScenarioBundle, WeatherForcing
from aeolus.evaluation.validity import (
    assess_fast_kernel_validity,
    assess_forcing_availability,
    assess_historical_fuel_provenance,
)


def _forcing(kind: str) -> WeatherForcing:
    return WeatherForcing(
        minute=np.asarray([0.0, 60.0, 120.0]),
        wind_speed_m_s=np.ones(3, dtype=np.float32),
        wind_direction_deg=np.zeros(3, dtype=np.float32),
        air_temperature_c=np.full(3, 20.0, dtype=np.float32),
        relative_humidity_pct=np.full(3, 30.0, dtype=np.float32),
        metadata={
            "time_units": "minutes since 2022-07-05T00:00:00Z",
            "analysis_or_forecast": kind,
            "forecast_reference_time": "2022-07-05T00:00:00Z",
            "assumed_availability_lag_hours": 1.0,
        },
    )


def test_forcing_availability_separates_analysis_and_forecast_claims() -> None:
    start = datetime(2022, 7, 5, 1, tzinfo=timezone.utc)
    end = datetime(2022, 7, 5, 2, tzinfo=timezone.utc)
    analysis = assess_forcing_availability(
        _forcing("analysis"),
        forecast_start=start,
        forecast_end=end,
    )
    assert analysis["supports_retrospective_hindcast"]
    assert not analysis["supports_operational_forecast"]
    forecast = assess_forcing_availability(
        _forcing("forecast"),
        forecast_start=start,
        forecast_end=end,
    )
    assert forecast["supports_operational_forecast"]


def _scenario(source: dict[str, object]) -> ScenarioBundle:
    shape = (16, 16)
    return ScenarioBundle(
        elevation_m=np.zeros(shape, dtype=np.float32),
        fuel_load_kg_m2=np.ones(shape, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": "EPSG:32610",
            "cell_size_m": 30.0,
            "sources": [source],
            "transformations": [],
            "split": "test",
        },
    )


def test_historical_fuel_provenance_detects_future_and_ambiguous_products() -> None:
    future = assess_historical_fuel_provenance(
        _scenario({"name": "LANDFIRE 2025", "product_year": 2025}),
        incident_start="2022-07-05T00:00:00Z",
    )
    assert future.status == "potential_post_incident_information"
    assert future.post_incident_product_sources == ("LANDFIRE 2025",)

    disturbance_leak = assess_historical_fuel_provenance(
        _scenario(
            {
                "name": "LANDFIRE reconstructed",
                "product_year": 2016,
                "disturbance_through_year": 2022,
            }
        ),
        incident_start="2022-07-05T00:00:00Z",
    )
    assert disturbance_leak.status == "potential_post_incident_information"
    assert disturbance_leak.incident_or_later_disturbance_sources == ("LANDFIRE reconstructed",)

    ambiguous = assess_historical_fuel_provenance(
        _scenario({"name": "LANDFIRE 2022", "product_year": 2022}),
        incident_start="2022-07-05T00:00:00Z",
    )
    assert ambiguous.status == "same_year_cutoff_unresolved"

    admissible = assess_historical_fuel_provenance(
        _scenario(
            {
                "name": "LANDFIRE 2021",
                "product_year": 2021,
                "data_cutoff": "2021-12-31T00:00:00Z",
            }
        ),
        incident_start="2022-07-05T00:00:00Z",
    )
    assert admissible.status == "historically_admissible_by_product_date"


def test_fast_kernel_validity_distinguishes_lookup_and_mechanism_regimes() -> None:
    within = assess_fast_kernel_validity(
        wind_speed_m_s=6.0,
        slope_tan=0.2,
        moisture_dead_1h=0.08,
        moisture_live_herbaceous=0.9,
        moisture_live_woody=1.0,
        fire_type=np.full((4, 4), int(FireType.SURFACE)),
    )
    assert within["classification"] == "surface_lookup_domain"
    assert within["supports_current_accuracy_claim"]

    crown = assess_fast_kernel_validity(
        wind_speed_m_s=6.0,
        slope_tan=0.2,
        moisture_dead_1h=0.08,
        moisture_live_herbaceous=0.9,
        moisture_live_woody=1.0,
        fire_type=np.full((4, 4), int(FireType.ACTIVE_CROWN)),
    )
    assert crown["classification"] == "mechanism_only_unvalidated_regime"
    assert crown["active_crown_cells"] == 16

    clipped = assess_fast_kernel_validity(
        wind_speed_m_s=np.asarray([6.0, 35.0]),
        slope_tan=0.2,
        moisture_dead_1h=0.08,
        moisture_live_herbaceous=0.9,
        moisture_live_woody=1.0,
    )
    assert clipped["classification"] == "outside_lookup_domain"
    assert clipped["lookup_domain"]["wind_speed_m_s"]["outside_values"] == 1
