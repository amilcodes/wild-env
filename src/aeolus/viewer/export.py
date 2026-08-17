"""Interoperability exports for independent scientific inspection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import numpy as np

from aeolus.viewer.model import ReplayModel

if TYPE_CHECKING:
    from aeolus.replay.recorder import ReplayBundle


def export_paraview(
    replay: ReplayBundle,
    destination: str | Path,
    *,
    max_frames: int | None = None,
) -> Path:
    """Write a VTK multiblock time series and PVD index for ParaView."""

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[viewer] for ParaView export") from exc

    model = ReplayModel(replay)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    frame_root = output / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    if max_frames is None or max_frames >= model.frame_count:
        selected = np.arange(model.frame_count, dtype=int)
    else:
        selected = np.unique(np.linspace(0, model.frame_count - 1, max_frames, dtype=int))
    height, width = model.shape
    y, x = np.mgrid[0:height, 0:width]
    x_m = x.astype(np.float64) * model.cell_size_m
    y_m = y.astype(np.float64) * model.cell_size_m
    elevation = model.static("static/elevation_m").astype(np.float64)
    dynamic_fields = (
        "truth/phase",
        "truth/intensity_kw_m",
        "truth/fire_type",
        "truth/spread_rate_m_min",
        "truth/flame_length_m",
        "truth/fuel_remaining",
        "truth/moisture_dead_1h",
        "belief/intensity_mean",
        "belief/intensity_std",
        "belief/burn_probability",
        "belief/arrival_time_mean",
        "belief/arrival_time_std",
        "treatment/water_coverage_gpc",
        "treatment/retardant_coverage_gpc",
        "treatment/constructed_line",
        "treatment/line_status",
        "environment/wind_speed_m_s",
        "environment/wind_direction_deg",
        "environment/air_temperature_c",
        "environment/relative_humidity_pct",
    )
    root = Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = SubElement(root, "Collection")
    for output_index, frame in enumerate(selected):
        terrain = pv.StructuredGrid(x_m, y_m, elevation)
        for name in dynamic_fields:
            if not model.has(name):
                continue
            key = name.replace("/", "_")
            terrain.point_data[key] = model.field(name, int(frame)).reshape(
                -1,
                order="F",
            )
        terrain.point_data["fuel_model_number"] = model.static("static/fuel_model_number").reshape(
            -1, order="F"
        )
        terrain.point_data["asset_value"] = model.static("static/asset_value").reshape(
            -1,
            order="F",
        )
        resources = model.resources(int(frame))
        resource_points = np.asarray(
            [
                (
                    resource["x"] * model.cell_size_m,
                    resource["y"] * model.cell_size_m,
                    elevation[
                        int(np.clip(round(resource["y"]), 0, height - 1)),
                        int(np.clip(round(resource["x"]), 0, width - 1)),
                    ],
                )
                for resource in resources
            ],
            dtype=np.float64,
        )
        resource_data = pv.PolyData(resource_points)
        resource_data["resource_index"] = np.arange(len(resources), dtype=np.int32)
        resource_data["kind"] = np.asarray(
            [{"retardant": 0, "water": 1, "sensor": 2, "crew": 3}[item["kind"]] for item in resources],
            dtype=np.int8,
        )
        resource_data["status"] = np.asarray(
            [item["status"] for item in resources],
            dtype=np.int8,
        )
        resource_data["payload_fraction"] = np.asarray(
            [item["payload_fraction"] for item in resources],
            dtype=np.float32,
        )
        resource_data["endurance_remaining_min"] = np.asarray(
            [item["endurance_remaining_min"] for item in resources],
            dtype=np.float32,
        )
        resource_data["task_kind"] = np.asarray(
            [item["task_kind"] for item in resources],
            dtype=np.int8,
        )
        site_points = np.asarray(
            [
                (
                    site["x"] * model.cell_size_m,
                    site["y"] * model.cell_size_m,
                    elevation[int(site["y"]), int(site["x"])],
                )
                for site in model.service_sites
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        sites = pv.PolyData(site_points)
        if len(model.service_sites):
            sites["site_index"] = np.arange(len(model.service_sites), dtype=np.int32)
            if model.has("service_sites/remaining_volume_l"):
                sites["remaining_volume_l"] = model.field(
                    "service_sites/remaining_volume_l",
                    int(frame),
                )
        blocks = pv.MultiBlock(
            {
                "terrain_and_fields": terrain,
                "resources": resource_data,
                "service_sites": sites,
            }
        )
        frame_name = f"frame-{output_index:05d}.vtm"
        blocks.save(frame_root / frame_name)
        SubElement(
            collection,
            "DataSet",
            timestep=str(int(model.minutes[frame])),
            group="",
            part="0",
            file=f"frames/{frame_name}",
        )
    indent(root)
    index_path = output / "aeolus-replay.pvd"
    ElementTree(root).write(index_path, encoding="utf-8", xml_declaration=True)
    (output / "README.txt").write_text(
        (
            "Open aeolus-replay.pvd in ParaView. Time is simulation minute. "
            "Horizontal coordinates are local metres from the replay grid origin; "
            "terrain elevation is metres. Resource points are ground-projected "
            "because the simulator does not resolve aircraft altitude.\n"
        ),
        encoding="utf-8",
    )
    return index_path
