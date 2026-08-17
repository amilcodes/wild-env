from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from aeolus.data.bundle import ScenarioBundle
from aeolus.data.historical_fuels import (
    LANDFIRE_HISTORICAL_VERSIONS,
    reconstruct_historical_landscape,
    select_historical_landfire_version,
)
from aeolus.evaluation.validity import assess_historical_fuel_provenance


def _original_landscape(
    tmp_path: Path,
    *,
    crs: str = "EPSG:3857",
) -> tuple[ScenarioBundle, Path]:
    shape = (16, 16)
    elevation = np.arange(256, dtype=np.float32).reshape(shape)
    old_fuel = np.full(shape, 101, dtype=np.int16)
    bundle = ScenarioBundle(
        elevation_m=elevation,
        fuel_load_kg_m2=np.ones(shape, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        fuel_model_number=old_fuel,
        canopy_cover=np.full(shape, 0.5, dtype=np.float32),
        canopy_height_m=np.full(shape, 5.0, dtype=np.float32),
        canopy_base_height_m=np.full(shape, 2.0, dtype=np.float32),
        canopy_bulk_density_kg_m3=np.full(shape, 0.1, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": crs,
            "cell_size_m": 30.0,
            "sources": [
                {"name": "USGS 3DEP"},
                {"name": "LANDFIRE 2025", "product_year": 2025},
            ],
            "transformations": ["test fixture"],
            "split": "evaluation",
        },
    )
    source_path = tmp_path / "source_landscape.tif"
    profile = {
        "driver": "GTiff",
        "width": 16,
        "height": 16,
        "count": 6,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(0.0, 480.0, 30.0, 30.0),
    }
    descriptions = (
        "elevation_m",
        "fbfm40_code",
        "canopy_cover",
        "canopy_height",
        "canopy_base_height",
        "canopy_bulk_density",
    )
    with rasterio.open(source_path, "w", **profile) as output:
        for index, description in enumerate(descriptions, start=1):
            output.write(
                elevation if index == 1 else np.ones(shape, dtype=np.float32),
                index,
            )
            output.set_band_description(index, description)
    return bundle, source_path


def test_version_selector_excludes_incident_and_future_disturbance() -> None:
    assert (
        select_historical_landfire_version(
            "2020-06-07T04:00:00Z",
            require_streamable=True,
        ).version_id
        == "lf2016-remap"
    )
    assert (
        select_historical_landfire_version(
            "2023-08-05T04:01:00Z",
            require_streamable=False,
        ).version_id
        == "lf2020"
    )
    assert (
        select_historical_landfire_version(
            "2023-08-05T04:01:00Z",
            require_streamable=True,
        ).version_id
        == "lf2016-remap"
    )


@pytest.mark.parametrize("crs", ["EPSG:3857", "EPSG:32610"])
def test_reconstruction_replaces_future_fuels_and_records_cutoff(
    tmp_path: Path,
    crs: str,
) -> None:
    original, source_path = _original_landscape(tmp_path, crs=crs)
    version = LANDFIRE_HISTORICAL_VERSIONS[0]

    def fake_download(
        service_url,
        *,
        bounds,
        crs,
        size,
        destination,
        categorical,
    ):
        del service_url, bounds, categorical
        values = np.full(
            (size[1], size[0]),
            91 if "fuel_model" in str(destination) else 20,
            dtype=np.int16,
        )
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=size[0],
            height=size[1],
            count=1,
            dtype="int16",
            crs=crs,
            transform=from_origin(0.0, 480.0, 30.0, 30.0),
            nodata=-9999,
        ) as output:
            output.write(values, 1)
        return Path(destination)

    rebuilt, stats = reconstruct_historical_landscape(
        original,
        source_path,
        version=version,
        provenance_directory=tmp_path / "raw",
        output_landscape_path=tmp_path / "historical.tif",
        downloader=fake_download,
    )
    assessment = assess_historical_fuel_provenance(
        rebuilt,
        incident_start="2020-06-07T04:00:00Z",
    )

    assert np.all(rebuilt.fuel_model_number == 91)
    assert rebuilt.barrier.all()
    assert stats["fuel_model_changed_fraction"] == pytest.approx(1.0)
    assert stats["burnability_changed_fraction"] == pytest.approx(1.0)
    assert assessment.status == "historically_admissible_by_product_date"
    assert rebuilt.metadata["sources"][-1]["disturbance_through_year"] == 2016
