from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from aeolus.data.bundle import ScenarioBundle
from aeolus.data.incident import write_incident_bundle
from aeolus.data.progression import rasterize_progression_observation_bundle


def _polygon(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _incident(root: Path) -> Path:
    shape = (24, 24)
    transform = from_origin(0.0, 24.0, 1.0, 1.0)
    landscape = ScenarioBundle(
        elevation_m=np.zeros(shape, dtype=np.float32),
        fuel_load_kg_m2=np.ones(shape, dtype=np.float32),
        barrier=np.zeros(shape, dtype=np.bool_),
        asset_value=np.zeros(shape, dtype=np.float32),
        metadata={
            "schema_version": 2,
            "crs": "EPSG:4326",
            "cell_size_m": 1.0,
            "sources": [{"name": "synthetic"}],
            "transformations": ["unit-test"],
            "split": "train",
            "transform": list(tuple(transform)[:6]),
        },
    )
    write_incident_bundle(
        root,
        incident_id="test-progression",
        bbox=(0.0, 0.0, 24.0, 24.0),
        start_datetime="2020-08-01T00:00:00Z",
        end_datetime="2020-08-01T02:00:00Z",
        scenario_bundle=landscape,
        perimeter_collection={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": _polygon(10.0, 10.0, 12.0, 12.0),
                    "properties": {"observed_at": "2020-08-01T00:00:00Z"},
                },
                {
                    "type": "Feature",
                    "geometry": _polygon(9.0, 9.0, 13.0, 13.0),
                    "properties": {"observed_at": "2020-08-01T01:00:00Z"},
                },
            ],
        },
    )
    return root


def _observations(root: Path) -> Path:
    root.mkdir()
    frames = []
    lines = []
    for hour, polygon, area, active in (
        (1, _polygon(10.0, 10.0, 12.0, 12.0), 4e-6, 1.0),
        (2, _polygon(9.0, 9.0, 13.0, 13.0), 16e-6, 0.0),
    ):
        properties = {
            "acquisition_start": f"2020-08-01T0{hour - 1}:00:00Z",
            "acquisition_end": f"2020-08-01T0{hour}:00:00Z",
            "reported_cumulative_area_km2": area,
        }
        frames.append(
            {
                "type": "Feature",
                "id": f"perimeter-{hour}",
                "geometry": polygon,
                "properties": properties,
            }
        )
        lines.append(
            {
                "type": "Feature",
                "id": f"line-{hour}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[10.0, 10.0], [12.0, 12.0]],
                },
                "properties": {**properties, "source_active_state": active},
            }
        )
    (root / "perimeters.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": frames}),
        encoding="utf-8",
    )
    (root / "concurrent_active_lines.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": lines}),
        encoding="utf-8",
    )
    (root / "observation_manifest.json").write_text(
        json.dumps({"observation_semantics": {"source_crs": "EPSG:4326"}}),
        encoding="utf-8",
    )
    return root


def test_progression_cube_preserves_masks_times_and_dormancy(tmp_path: Path) -> None:
    incident = _incident(tmp_path / "incident")
    observations = _observations(tmp_path / "observations")
    output = tmp_path / "cube.npz"
    manifest = rasterize_progression_observation_bundle(incident, observations, output)
    with np.load(output, allow_pickle=False) as cube:
        assert cube["perimeter_mask"].shape == (2, 24, 24)
        assert cube["perimeter_mask"][0].sum() == 4
        assert cube["perimeter_mask"][1].sum() == 16
        assert cube["elapsed_minute"].tolist() == [0.0, 60.0]
        assert cube["active_line_mask_raw"][1].any()
        assert not cube["active_line_mask"][1].any()
        assert cube["source_active_state"].tolist() == [1.0, 0.0]
        assert np.allclose(cube["spatial_coverage_fraction"], 1.0)
    assert manifest["audit"]["cumulative_nesting_violation_frames"] == 0
    assert manifest["audit"]["dormant_line_frames"] == 1
    assert manifest["audit"]["missing_concurrent_active_line_count"] == 0
    assert output.with_suffix(".npz.manifest.json").is_file()

