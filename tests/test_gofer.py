from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import shapefile

from aeolus.data.gofer import gofer_fire_catalog, write_gofer_observation_bundle


def _base_fields(writer: shapefile.Writer) -> None:
    writer.field("fname", "C", size=80)
    writer.field("fyear", "N", size=9)
    writer.field("tUTC", "C", size=80)
    writer.field("tLocal", "C", size=80)
    writer.field("tLocalGMT", "C", size=80)
    writer.field("timestep", "N", size=24, decimal=15)


def _write_product(root: Path) -> Path:
    product = root / "GOFER"
    combined = product / "GOFER_Combined"
    combined.mkdir(parents=True)
    (product / "fireData.csv").write_text(
        "fname,fyear,acres_official,GOESIg_UTC,local_tz,local_tzGMT\n"
        "Example,2020,12345,2020-08-01 00,America/Los_Angeles,Etc/GMT+8\n"
        "Other,2021,50000,2021-09-01 00,America/Los_Angeles,Etc/GMT+8\n",
        encoding="utf-8",
    )
    polygon = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
    perimeter_path = combined / "GOFERC_fireProg"
    with shapefile.Writer(str(perimeter_path), shapeType=shapefile.POLYGON) as writer:
        _base_fields(writer)
        writer.field("farea", "N", size=24, decimal=15)
        writer.field("fareaPer", "N", size=24, decimal=15)
        writer.field("fperim", "N", size=24, decimal=15)
        for hour, area in ((1, 1.0), (2, 2.0)):
            writer.poly([polygon])
            writer.record(
                "Example",
                2020,
                f"2020-08-01 0{hour}:00:00",
                f"2020-07-31 1{hour + 7}:00:00",
                f"2020-07-31 1{hour + 6}:00:00",
                hour,
                area,
                area * 10.0,
                4.0,
            )
        writer.poly([polygon])
        writer.record(
            "Other",
            2021,
            "2021-09-01 01:00:00",
            "2021-08-31 18:00:00",
            "2021-08-31 17:00:00",
            1,
            1.0,
            1.0,
            4.0,
        )

    line = [[0.0, 0.0], [1.0, 1.0]]
    concurrent_path = combined / "GOFERC_cfireLine"
    with shapefile.Writer(str(concurrent_path), shapeType=shapefile.POLYLINE) as writer:
        _base_fields(writer)
        writer.field("cflinelen", "N", size=24, decimal=15)
        writer.field("fconf", "N", size=24, decimal=15)
        writer.field("fstate", "N", size=24, decimal=15)
        for hour in (1, 2):
            for confidence in (0.05, 0.1):
                writer.line([line])
                writer.record(
                    "Example",
                    2020,
                    f"2020-08-01 0{hour}:00:00",
                    f"2020-07-31 1{hour + 7}:00:00",
                    f"2020-07-31 1{hour + 6}:00:00",
                    hour,
                    1.4,
                    confidence,
                    int(hour == 1),
                )

    retrospective_path = combined / "GOFERC_rfireLine"
    with shapefile.Writer(str(retrospective_path), shapeType=shapefile.POLYLINE) as writer:
        _base_fields(writer)
        writer.field("rflinelen", "N", size=24, decimal=15)
        writer.field("fstate", "N", size=9)
        for hour in (1, 2):
            writer.line([line])
            writer.record(
                "Example",
                2020,
                f"2020-08-01 0{hour}:00:00",
                f"2020-07-31 1{hour + 7}:00:00",
                f"2020-07-31 1{hour + 6}:00:00",
                hour,
                1.4,
                1,
            )

    for stem in ("GOFERC_fireProg", "GOFERC_cfireLine", "GOFERC_rfireLine"):
        (combined / f"{stem}.prj").write_text("GEOGCS[\"GCS_WGS_1984\"]", encoding="utf-8")
    with (combined / "GOFERC_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("fname", "fyear", "tUTC", "timestep"))
        writer.writeheader()
        writer.writerow(
            {"fname": "Example", "fyear": 2020, "tUTC": "2020-08-01 01:00:00", "timestep": 1}
        )
        writer.writerow(
            {"fname": "Example", "fyear": 2020, "tUTC": "2020-08-01 02:00:00", "timestep": 2}
        )
        writer.writerow(
            {"fname": "Other", "fyear": 2021, "tUTC": "2021-09-01 01:00:00", "timestep": 1}
        )
    return product


def test_gofer_import_preserves_hourly_and_retrospective_semantics(tmp_path: Path) -> None:
    source = _write_product(tmp_path / "source")
    output = tmp_path / "output"
    manifest = write_gofer_observation_bundle(
        source,
        output,
        fire_name="example",
        fire_year=2020,
    )

    perimeters = json.loads((output / "perimeters.geojson").read_text(encoding="utf-8"))
    concurrent = json.loads((output / "concurrent_active_lines.geojson").read_text(encoding="utf-8"))
    assert len(perimeters["features"]) == 2
    assert len(concurrent["features"]) == 2
    first = perimeters["features"][0]["properties"]
    assert first["acquisition_start"] == "2020-08-01T00:00:00Z"
    assert first["acquisition_end"] == "2020-08-01T01:00:00Z"
    assert first["available_at"] is None
    assert first["operationally_available"] is False
    assert concurrent["features"][0]["properties"]["fire_detection_confidence_threshold"] == 0.05
    assert manifest["audit"]["perimeters"]["reported_cumulative_area_decrease_count"] == 0
    assert manifest["audit"]["perimeters"]["non_hourly_gap_count"] == 0
    assert manifest["audit"]["perimeter_geometry"]["invalid_geometry_count"] == 0
    assert manifest["audit"]["concurrent_active_line_geometry"]["geometry_types"] == ["LineString"]
    assert manifest["audit"]["concurrent_active_line_dormant_records"] == 1
    assert manifest["observation_semantics"]["spatial_validation_scope"].startswith(
        "population statistics"
    )
    assert len(manifest["source"]["component_sha256"]) == 14


def test_gofer_catalog_and_validation_fail_closed(tmp_path: Path) -> None:
    source = _write_product(tmp_path / "source")
    catalog = gofer_fire_catalog(source)
    assert [(record["fire_name"], record["fire_year"]) for record in catalog] == [
        ("Example", 2020),
        ("Other", 2021),
    ]
    with pytest.raises(ValueError, match="published threshold"):
        write_gofer_observation_bundle(
            source,
            tmp_path / "bad-confidence",
            fire_name="Example",
            fire_year=2020,
            concurrent_confidence=0.2,
        )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        write_gofer_observation_bundle(
            source,
            occupied,
            fire_name="Example",
            fire_year=2020,
        )
