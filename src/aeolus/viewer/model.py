"""Read-only model used by native and headless replay views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from aeolus.core.state import ResourceStatus
from aeolus.core.tasks import TaskKind

if TYPE_CHECKING:
    from aeolus.replay.recorder import ReplayBundle

STATUS_NAMES = {
    int(ResourceStatus.AVAILABLE): "ready",
    int(ResourceStatus.OUTBOUND): "outbound",
    int(ResourceStatus.RETURNING): "returning",
    int(ResourceStatus.RELOADING): "servicing",
    int(ResourceStatus.WITHDRAWN): "withdrawn",
    int(ResourceStatus.WORKING): "working",
    int(ResourceStatus.QUEUED): "queued",
}
TASK_NAMES = {int(item): item.name.lower() for item in TaskKind}


@dataclass(frozen=True)
class ReplayEvent:
    minute: int
    kind: str
    payload: dict[str, Any]


class ReplayModel:
    """Convenience access to one immutable replay bundle."""

    def __init__(self, replay: ReplayBundle):
        self.replay = replay
        self.states = replay.states
        self.metadata = replay.metadata
        self.minutes = np.asarray(self.states["time/minute"], dtype=np.int32)
        self.resource_ids = tuple(self.metadata["resource_ids"])
        self.resource_kinds = tuple(self.metadata["resource_kinds"])
        self.resource_specs = tuple(self.metadata["scenario"].get("resources", []))
        self.service_sites = tuple(self.metadata["scenario"].get("service_sites", []))
        self.service_site_ids = tuple(
            self.metadata.get(
                "service_site_ids",
                [site["site_id"] for site in self.service_sites],
            )
        )
        self.spatial_reference = self.metadata.get(
            "spatial_reference",
            {
                "crs": None,
                "transform": None,
                "bounds": None,
                "cell_size_m": self.metadata["scenario"]["cell_size_m"],
                "shape": list(self.states["static/elevation_m"].shape),
            },
        )
        self._events = self._read_events()

    @property
    def frame_count(self) -> int:
        return self.replay.frame_count

    @property
    def title(self) -> str:
        incident = self.metadata.get("incident")
        if incident:
            return str(incident.get("title") or incident.get("id"))
        identity = self.metadata.get("scenario_identity", {})
        return str(
            identity.get("title") or self.metadata["scenario"].get("title") or "Synthetic research domain"
        )

    @property
    def location_name(self) -> str | None:
        incident = self.metadata.get("incident")
        if incident:
            return str(incident.get("title") or incident.get("id"))
        identity = self.metadata.get("scenario_identity", {})
        value = identity.get("location_name") or self.metadata["scenario"].get("location_name")
        return None if value is None else str(value)

    @property
    def time_origin(self) -> str | None:
        incident = self.metadata.get("incident")
        if incident and incident.get("start_datetime"):
            return str(incident["start_datetime"])
        identity = self.metadata.get("scenario_identity", {})
        value = identity.get("time_origin") or self.metadata["scenario"].get("time_origin")
        return None if value is None else str(value)

    @property
    def cell_size_m(self) -> float:
        return float(self.metadata["scenario"]["cell_size_m"])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.states["static/elevation_m"].shape)

    @property
    def events(self) -> tuple[ReplayEvent, ...]:
        return self._events

    def _read_events(self) -> tuple[ReplayEvent, ...]:
        frame = self.replay.events()
        return tuple(
            ReplayEvent(
                minute=int(row.minute),
                kind=str(row.kind),
                payload=json.loads(row.payload_json),
            )
            for row in frame.itertuples(index=False)
        )

    def has(self, name: str) -> bool:
        return name in self.states

    def field(self, name: str, frame: int) -> np.ndarray:
        if not 0 <= frame < self.frame_count:
            raise IndexError(frame)
        return np.asarray(self.states[name][frame])

    def static(self, name: str) -> np.ndarray:
        return np.asarray(self.states[name])

    def frame_for_minute(self, minute: float) -> int:
        position = int(np.searchsorted(self.minutes, minute))
        if position <= 0:
            return 0
        if position >= self.frame_count:
            return self.frame_count - 1
        before = self.minutes[position - 1]
        after = self.minutes[position]
        return position - 1 if minute - before <= after - minute else position

    def clock_label(self, minute: int) -> str:
        """Return incident local/offset time when an origin exists, otherwise elapsed time."""

        if self.time_origin is None:
            return f"T+{minute:03d} min"
        origin = datetime.fromisoformat(self.time_origin.replace("Z", "+00:00"))
        return (origin + timedelta(minutes=minute)).isoformat(timespec="minutes")

    def resource_index(self, resource_id: str) -> int:
        try:
            return self.resource_ids.index(resource_id)
        except ValueError as exc:
            raise KeyError(resource_id) from exc

    def resource(self, frame: int, resource: int | str) -> dict[str, Any]:
        index = self.resource_index(resource) if isinstance(resource, str) else int(resource)

        def value(name: str, default: float | int) -> float | int:
            return np.asarray(self.states[name][frame, index]).item() if self.has(name) else default

        status = int(value("resources/status", 0))
        task_kind = int(value("resources/task_kind", 0))
        current_site = int(value("resources/current_site_index", -1))
        service_site = int(value("resources/service_site_index", -1))
        spec = self.resource_specs[index] if index < len(self.resource_specs) else {}
        return {
            "id": self.resource_ids[index],
            "kind": self.resource_kinds[index],
            "x": float(value("resources/x", 0.0)),
            "y": float(value("resources/y", 0.0)),
            "status": status,
            "status_name": STATUS_NAMES.get(status, f"status-{status}"),
            "payload_fraction": float(value("resources/payload_fraction", 0.0)),
            "payload_capacity_l": float(spec.get("payload_l", 0.0)),
            "endurance_remaining_min": float(value("resources/endurance_remaining_min", np.nan)),
            "eta_min": int(value("resources/eta_min", 0)),
            "task_kind": task_kind,
            "task_name": TASK_NAMES.get(task_kind, f"task-{task_kind}"),
            "task_heading_deg": float(value("resources/task_heading_deg", 0.0)),
            "target_x": float(value("resources/target_x", np.nan)),
            "target_y": float(value("resources/target_y", np.nan)),
            "current_site": (
                self.service_site_ids[current_site]
                if 0 <= current_site < len(self.service_site_ids)
                else None
            ),
            "service_site": (
                self.service_site_ids[service_site]
                if 0 <= service_site < len(self.service_site_ids)
                else None
            ),
        }

    def resources(self, frame: int) -> list[dict[str, Any]]:
        return [self.resource(frame, index) for index in range(len(self.resource_ids))]

    def conditions(self, frame: int, x: float, y: float) -> dict[str, float]:
        height, width = self.shape
        ix = int(np.clip(round(x), 0, width - 1))
        iy = int(np.clip(round(y), 0, height - 1))
        scenario = self.metadata["scenario"]

        def sample(name: str, fallback: float) -> float:
            path = f"environment/{name}"
            return float(np.asarray(self.states[path][frame, iy, ix])) if self.has(path) else float(fallback)

        return {
            "wind_speed_m_s": sample(
                "wind_speed_m_s",
                scenario.get("wind_speed_m_s", 0.0),
            ),
            "wind_direction_deg": sample(
                "wind_direction_deg",
                scenario.get("wind_direction_deg", 0.0),
            ),
            "air_temperature_c": sample(
                "air_temperature_c",
                scenario.get("air_temperature_c", 0.0),
            ),
            "relative_humidity_pct": sample(
                "relative_humidity_pct",
                scenario.get("relative_humidity_pct", 0.0),
            ),
            "precipitation_rate_mm_h": sample(
                "precipitation_rate_mm_h",
                scenario.get("precipitation_rate_mm_h", 0.0),
            ),
            "dead_fuel_moisture": float(np.asarray(self.states["truth/moisture_dead_1h"][frame, iy, ix])),
        }

    def grid_to_world(self, x: float, y: float) -> tuple[float, float] | None:
        transform = self.spatial_reference.get("transform")
        if not transform or len(transform) != 6:
            return None
        a, b, c, d, e, f = (float(item) for item in transform)
        return (
            a * (x + 0.5) + b * (y + 0.5) + c,
            d * (x + 0.5) + e * (y + 0.5) + f,
        )

    def events_near(self, minute: int, tolerance: int = 0) -> tuple[ReplayEvent, ...]:
        return tuple(event for event in self.events if abs(event.minute - minute) <= tolerance)
