"""Chunked replay records independent of the training process."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle, load_bundle

Policy = Callable[[AeolusSimulator], dict[str, int]]

REPLAY_SCHEMA_VERSION = 2
SUPPORTED_REPLAY_SCHEMA_VERSIONS = {1, 2}

WEATHER_FIELDS = (
    "wind_speed_m_s",
    "wind_direction_deg",
    "air_temperature_c",
    "relative_humidity_pct",
    "precipitation_rate_mm_h",
)


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayRecorder:
    """In-memory observer that writes a chunked Zarr episode on completion."""

    def __init__(self) -> None:
        self.minutes: list[int] = []
        self.phase: list[np.ndarray] = []
        self.intensity: list[np.ndarray] = []
        self.fire_type: list[np.ndarray] = []
        self.spread_rate: list[np.ndarray] = []
        self.flame_length: list[np.ndarray] = []
        self.fuel_remaining: list[np.ndarray] = []
        self.moisture_dead_1h: list[np.ndarray] = []
        self.level_set: list[np.ndarray] = []
        self.belief_mean: list[np.ndarray] = []
        self.belief_std: list[np.ndarray] = []
        self.burn_probability: list[np.ndarray] = []
        self.arrival_time_mean: list[np.ndarray] = []
        self.arrival_time_std: list[np.ndarray] = []
        self.known_burned: list[np.ndarray] = []
        self.water: list[np.ndarray] = []
        self.retardant: list[np.ndarray] = []
        self.ground_hold: list[np.ndarray] = []
        self.water_coverage_gpc: list[np.ndarray] = []
        self.retardant_coverage_gpc: list[np.ndarray] = []
        self.retardant_effective_coverage_gpc: list[np.ndarray] = []
        self.constructed_line: list[np.ndarray] = []
        self.line_status: list[np.ndarray] = []
        self.resource_x: list[np.ndarray] = []
        self.resource_y: list[np.ndarray] = []
        self.resource_status: list[np.ndarray] = []
        self.resource_payload: list[np.ndarray] = []
        self.resource_endurance_remaining: list[np.ndarray] = []
        self.resource_eta: list[np.ndarray] = []
        self.resource_task_kind: list[np.ndarray] = []
        self.resource_task_heading: list[np.ndarray] = []
        self.resource_target_x: list[np.ndarray] = []
        self.resource_target_y: list[np.ndarray] = []
        self.resource_current_site: list[np.ndarray] = []
        self.resource_service_site: list[np.ndarray] = []
        self.site_remaining_volume: list[np.ndarray] = []
        self.weather: dict[str, list[np.ndarray]] = {name: [] for name in WEATHER_FIELDS}
        self._last_simulator: AeolusSimulator | None = None

    def __call__(self, simulator: AeolusSimulator) -> None:
        self._last_simulator = simulator
        state = simulator.state
        replace = bool(self.minutes and self.minutes[-1] == state.minute)
        index = -1 if replace else None

        def append(values: list[Any], value: Any) -> None:
            if index is None:
                values.append(value)
            else:
                values[index] = value

        append(self.minutes, int(state.minute))
        append(self.phase, state.truth.phase.astype(np.uint8, copy=True))
        append(self.intensity, state.truth.intensity_kw_m.astype(np.float32, copy=True))
        append(self.fire_type, state.truth.fire_type.astype(np.uint8, copy=True))
        append(
            self.spread_rate,
            state.truth.spread_rate_m_min.astype(np.float32, copy=True),
        )
        append(
            self.flame_length,
            state.truth.flame_length_m.astype(np.float32, copy=True),
        )
        append(
            self.fuel_remaining,
            state.truth.fuel_remaining.astype(np.float32, copy=True),
        )
        append(
            self.moisture_dead_1h,
            state.truth.moisture_dead_1h.astype(np.float32, copy=True),
        )
        append(
            self.level_set,
            state.truth.level_set_m.astype(np.float32, copy=True),
        )
        append(self.belief_mean, state.belief.intensity_mean.astype(np.float32, copy=True))
        append(self.belief_std, state.belief.intensity_std.astype(np.float32, copy=True))
        append(
            self.burn_probability,
            state.belief.burn_probability.astype(np.float32, copy=True),
        )
        append(
            self.arrival_time_mean,
            state.belief.arrival_time_mean.astype(np.float32, copy=True),
        )
        append(
            self.arrival_time_std,
            state.belief.arrival_time_std.astype(np.float32, copy=True),
        )
        append(self.known_burned, state.belief.known_burned.astype(np.float32, copy=True))
        append(self.water, state.truth.water.astype(np.float32, copy=True))
        append(self.retardant, state.truth.retardant.astype(np.float32, copy=True))
        append(self.ground_hold, state.truth.ground_hold.astype(np.float32, copy=True))
        append(
            self.water_coverage_gpc,
            state.truth.water_coverage_gpc.astype(np.float32, copy=True),
        )
        append(
            self.retardant_coverage_gpc,
            state.truth.retardant_coverage_gpc.astype(np.float32, copy=True),
        )
        append(
            self.retardant_effective_coverage_gpc,
            state.truth.retardant_effective_coverage_gpc.astype(np.float32, copy=True),
        )
        append(
            self.constructed_line,
            state.truth.constructed_line.astype(np.float32, copy=True),
        )
        append(
            self.line_status,
            state.truth.line_status.astype(np.uint8, copy=True),
        )
        append(
            self.resource_x,
            np.asarray([resource.x for resource in state.resources], dtype=np.float32),
        )
        append(
            self.resource_y,
            np.asarray([resource.y for resource in state.resources], dtype=np.float32),
        )
        append(
            self.resource_status,
            np.asarray([resource.status for resource in state.resources], dtype=np.uint8),
        )
        append(
            self.resource_payload,
            np.asarray([resource.payload_fraction for resource in state.resources], dtype=np.float32),
        )
        append(
            self.resource_endurance_remaining,
            np.asarray(
                [resource.endurance_remaining_min for resource in state.resources],
                dtype=np.float32,
            ),
        )
        append(
            self.resource_eta,
            np.asarray([resource.eta_min for resource in state.resources], dtype=np.int32),
        )
        append(
            self.resource_task_kind,
            np.asarray([resource.task_kind for resource in state.resources], dtype=np.int8),
        )
        append(
            self.resource_task_heading,
            np.asarray(
                [resource.task_heading_deg for resource in state.resources],
                dtype=np.float32,
            ),
        )
        append(
            self.resource_target_x,
            np.asarray(
                [
                    np.nan if resource.target_xy is None else resource.target_xy[0]
                    for resource in state.resources
                ],
                dtype=np.float32,
            ),
        )
        append(
            self.resource_target_y,
            np.asarray(
                [
                    np.nan if resource.target_xy is None else resource.target_xy[1]
                    for resource in state.resources
                ],
                dtype=np.float32,
            ),
        )
        site_index = {site.site_id: index for index, site in enumerate(state.service_sites)}
        append(
            self.resource_current_site,
            np.asarray(
                [site_index.get(resource.current_site_id, -1) for resource in state.resources],
                dtype=np.int16,
            ),
        )
        append(
            self.resource_service_site,
            np.asarray(
                [site_index.get(resource.service_site_id, -1) for resource in state.resources],
                dtype=np.int16,
            ),
        )
        append(
            self.site_remaining_volume,
            np.asarray(
                [site.remaining_volume_l for site in state.service_sites],
                dtype=np.float64,
            ),
        )
        weather = simulator.current_weather()
        for name in WEATHER_FIELDS:
            raw = np.asarray(weather[name], dtype=np.float32)
            field = (
                np.full(state.truth.phase.shape, float(raw), dtype=np.float32)
                if raw.ndim == 0
                else raw.astype(np.float32, copy=True)
            )
            if field.shape != state.truth.phase.shape:
                raise ValueError(f"weather field {name} shape {field.shape} does not match replay grid")
            append(self.weather[name], field)

    def save(
        self,
        destination: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        policy_name: str = "unknown",
    ) -> ReplayBundle:
        try:
            import pandas as pd
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] to write replay bundles") from exc
        if not self.minutes or self._last_simulator is None:
            raise RuntimeError("cannot save an empty replay")

        simulator = self._last_simulator
        root_path = Path(destination)
        root_path.mkdir(parents=True, exist_ok=True)
        store_path = root_path / "states.zarr"
        group = zarr.open_group(str(store_path), mode="w")
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

        def write(name: str, value: np.ndarray, chunks: tuple[int, ...]) -> None:
            group.create_dataset(
                name,
                data=value,
                chunks=chunks,
                compressor=compressor,
                overwrite=True,
            )

        shape = self.phase[0].shape
        raster_chunks = (1, min(128, shape[0]), min(128, shape[1]))
        write("time/minute", np.asarray(self.minutes, dtype=np.int32), (min(256, len(self.minutes)),))
        for name, values in (
            ("truth/phase", self.phase),
            ("truth/intensity_kw_m", self.intensity),
            ("truth/fire_type", self.fire_type),
            ("truth/spread_rate_m_min", self.spread_rate),
            ("truth/flame_length_m", self.flame_length),
            ("truth/fuel_remaining", self.fuel_remaining),
            ("truth/moisture_dead_1h", self.moisture_dead_1h),
            ("truth/level_set_m", self.level_set),
            ("belief/intensity_mean", self.belief_mean),
            ("belief/intensity_std", self.belief_std),
            ("belief/burn_probability", self.burn_probability),
            ("belief/arrival_time_mean", self.arrival_time_mean),
            ("belief/arrival_time_std", self.arrival_time_std),
            ("belief/known_burned", self.known_burned),
            ("treatment/water", self.water),
            ("treatment/retardant", self.retardant),
            ("treatment/ground_hold", self.ground_hold),
            ("treatment/water_coverage_gpc", self.water_coverage_gpc),
            (
                "treatment/retardant_coverage_gpc",
                self.retardant_coverage_gpc,
            ),
            (
                "treatment/retardant_effective_coverage_gpc",
                self.retardant_effective_coverage_gpc,
            ),
            ("treatment/constructed_line", self.constructed_line),
            ("treatment/line_status", self.line_status),
        ):
            write(name, np.stack(values), raster_chunks)
        resource_chunks = (min(256, len(self.minutes)), len(simulator.state.resources))
        write("resources/x", np.stack(self.resource_x), resource_chunks)
        write("resources/y", np.stack(self.resource_y), resource_chunks)
        write("resources/status", np.stack(self.resource_status), resource_chunks)
        write("resources/payload_fraction", np.stack(self.resource_payload), resource_chunks)
        write(
            "resources/endurance_remaining_min",
            np.stack(self.resource_endurance_remaining),
            resource_chunks,
        )
        write("resources/eta_min", np.stack(self.resource_eta), resource_chunks)
        write("resources/task_kind", np.stack(self.resource_task_kind), resource_chunks)
        write(
            "resources/task_heading_deg",
            np.stack(self.resource_task_heading),
            resource_chunks,
        )
        write("resources/target_x", np.stack(self.resource_target_x), resource_chunks)
        write("resources/target_y", np.stack(self.resource_target_y), resource_chunks)
        write(
            "resources/current_site_index",
            np.stack(self.resource_current_site),
            resource_chunks,
        )
        write(
            "resources/service_site_index",
            np.stack(self.resource_service_site),
            resource_chunks,
        )
        if simulator.state.service_sites:
            site_chunks = (
                min(256, len(self.minutes)),
                len(simulator.state.service_sites),
            )
            write(
                "service_sites/remaining_volume_l",
                np.stack(self.site_remaining_volume),
                site_chunks,
            )
        for name, values in self.weather.items():
            write(f"environment/{name}", np.stack(values), raster_chunks)
        write("static/elevation_m", simulator.state.truth.elevation_m, shape)
        write("static/fuel_load_kg_m2", simulator.state.truth.fuel_load, shape)
        write(
            "static/fuel_model_number",
            simulator.state.truth.fuel_model_number,
            shape,
        )
        write(
            "static/canopy_cover",
            simulator.state.truth.canopy_cover,
            shape,
        )
        write(
            "static/canopy_height_m",
            simulator.state.truth.canopy_height_m,
            shape,
        )
        write(
            "static/canopy_base_height_m",
            simulator.state.truth.canopy_base_height_m,
            shape,
        )
        write(
            "static/canopy_bulk_density_kg_m3",
            simulator.state.truth.canopy_bulk_density_kg_m3,
            shape,
        )
        write("static/barrier", simulator.state.truth.barrier.astype(np.uint8), shape)
        write("static/asset_value", simulator.state.truth.asset_value, shape)

        events = []
        for event in simulator.state.events:
            payload = {key: value for key, value in event.items() if key not in {"minute", "kind"}}
            events.append(
                {
                    "minute": int(event["minute"]),
                    "kind": str(event["kind"]),
                    "payload_json": json.dumps(payload, sort_keys=True),
                }
            )
        pd.DataFrame(events, columns=["minute", "kind", "payload_json"]).to_parquet(
            root_path / "events.parquet", index=False
        )
        spatial_reference: dict[str, Any] = {
            "crs": None,
            "transform": None,
            "bounds": None,
            "cell_size_m": simulator.config.cell_size_m,
            "shape": list(shape),
        }
        incident: dict[str, Any] | None = None
        if simulator.config.landscape_bundle:
            landscape_path = Path(simulator.config.landscape_bundle)
            if landscape_path.is_dir() and (landscape_path / "item.json").exists():
                incident_bundle = IncidentBundle.load(landscape_path)
                landscape = incident_bundle.scenario_bundle()
                properties = incident_bundle.item.get("properties", {})
                incident = {
                    "id": incident_bundle.incident_id,
                    "title": properties.get("title", incident_bundle.incident_id),
                    "bbox_wgs84": list(incident_bundle.bbox),
                    "start_datetime": properties.get("start_datetime"),
                    "end_datetime": properties.get("end_datetime"),
                }
            else:
                landscape = load_bundle(landscape_path)
            spatial_reference.update(
                {key: landscape.metadata.get(key) for key in ("crs", "transform", "bounds", "cell_size_m")}
            )
        metadata = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "policy_name": policy_name,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "resource_ids": simulator.agent_ids,
            "resource_kinds": [resource.spec.kind for resource in simulator.state.resources],
            "service_site_ids": [site.site_id for site in simulator.state.service_sites],
            "spatial_reference": spatial_reference,
            "incident": incident,
            "scenario_identity": {
                "id": simulator.config.scenario_id,
                "title": simulator.config.title,
                "location_name": simulator.config.location_name,
                "time_origin": simulator.config.time_origin,
            },
            "scenario": asdict(simulator.config),
            "episode": simulator.episode_record(),
            "frame_count": len(self.minutes),
        }
        (root_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return ReplayBundle.open(root_path)


class ReplayBundle:
    def __init__(self, root: Path, metadata: dict[str, Any], states: Any):
        self.root = root
        self.metadata = metadata
        self.states = states

    @classmethod
    def open(cls, root: str | Path) -> ReplayBundle:
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] to read replay bundles") from exc
        path = Path(root)
        with (path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("schema_version") not in SUPPORTED_REPLAY_SCHEMA_VERSIONS:
            raise ValueError("unsupported replay schema")
        states = zarr.open_group(str(path / "states.zarr"), mode="r")
        if len(states["time/minute"]) != int(metadata["frame_count"]):
            raise ValueError("replay metadata frame count does not match state arrays")
        return cls(path, metadata, states)

    @property
    def frame_count(self) -> int:
        return int(self.metadata["frame_count"])

    def events(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] to read replay events") from exc
        return pd.read_parquet(self.root / "events.parquet")


def record_episode(
    simulator: AeolusSimulator,
    policy: Policy,
    destination: str | Path,
    *,
    seed: int | None = None,
    checkpoint_path: str | Path | None = None,
    policy_name: str = "unknown",
    initialize: Callable[[AeolusSimulator], None] | None = None,
) -> ReplayBundle:
    recorder = ReplayRecorder()
    simulator.reset(seed)
    simulator.add_state_observer(recorder)
    try:
        if initialize is not None:
            initialize(simulator)
        while not simulator.state.terminated and not simulator.state.truncated:
            simulator.decision_step(policy(simulator))
    finally:
        simulator.remove_state_observer(recorder)
    return recorder.save(
        destination,
        checkpoint_path=checkpoint_path,
        policy_name=policy_name,
    )
