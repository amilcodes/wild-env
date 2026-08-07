"""Validated YAML configuration for desktop and headless replay views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

VIEWER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WindowConfig:
    width: int = 1600
    height: int = 1000
    start_view: str = "operational_2d"
    show_vehicle_panel: bool = True
    show_event_panel: bool = True
    show_layer_panel: bool = True

    def __post_init__(self) -> None:
        if self.width < 1024 or self.height < 700:
            raise ValueError("viewer window must be at least 1024 by 700 pixels")
        if self.start_view not in {"operational_2d", "terrain_3d"}:
            raise ValueError("start_view must be operational_2d or terrain_3d")


@dataclass(frozen=True)
class PlaybackConfig:
    rate: float = 4.0
    refresh_hz: int = 20
    loop: bool = False
    trail_minutes: int = 45
    event_autoselect: bool = True

    def __post_init__(self) -> None:
        if not 0.1 <= self.rate <= 128.0:
            raise ValueError("playback rate must be within [0.1, 128]")
        if not 5 <= self.refresh_hz <= 60:
            raise ValueError("refresh_hz must be within [5, 60]")
        if self.trail_minutes < 0:
            raise ValueError("trail_minutes cannot be negative")


@dataclass(frozen=True)
class CameraConfig:
    mode: str = "incident"
    elevation_deg: float = 42.0
    azimuth_deg: float = -132.0
    vertical_exaggeration: float = 1.6
    follow_resource: str | None = None
    follow_radius_cells: float = 28.0

    def __post_init__(self) -> None:
        if self.mode not in {"incident", "north_up", "follow"}:
            raise ValueError("camera mode must be incident, north_up, or follow")
        if not 0.2 <= self.vertical_exaggeration <= 8.0:
            raise ValueError("vertical_exaggeration must be within [0.2, 8]")
        if self.follow_radius_cells <= 1.0:
            raise ValueError("follow_radius_cells must exceed one cell")


@dataclass(frozen=True)
class LayerConfig:
    imagery: bool = False
    hillshade: bool = True
    contours: bool = True
    fuels: bool = False
    active_fire: bool = True
    burned_area: bool = True
    fire_type: bool = True
    belief_perimeter: bool = True
    belief_uncertainty: bool = False
    water: bool = True
    retardant: bool = True
    constructed_line: bool = True
    assets: bool = True
    service_sites: bool = True
    vehicle_tracks: bool = True
    vehicle_labels: bool = True
    targets: bool = True
    wind: bool = True
    coordinate_grid: bool = True


@dataclass(frozen=True)
class ImageryConfig:
    path: str | None = None
    bands: tuple[int, int, int] = (1, 2, 3)
    gamma: float = 1.0
    opacity: float = 0.82
    attribution: str | None = None

    def __post_init__(self) -> None:
        if len(self.bands) != 3 or min(self.bands) < 1:
            raise ValueError("imagery bands must contain three one-based indices")
        if self.gamma <= 0.0:
            raise ValueError("imagery gamma must be positive")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("imagery opacity must be within [0, 1]")


@dataclass(frozen=True)
class ExportConfig:
    width: int = 1920
    height: int = 1080
    dpi: int = 160
    fps: int = 20
    codec: str = "libx264"

    def __post_init__(self) -> None:
        if self.width < 640 or self.height < 480:
            raise ValueError("export dimensions are too small")
        if not 72 <= self.dpi <= 600:
            raise ValueError("export DPI must be within [72, 600]")
        if not 1 <= self.fps <= 120:
            raise ValueError("export FPS must be within [1, 120]")


@dataclass(frozen=True)
class ViewerConfig:
    schema_version: int = VIEWER_SCHEMA_VERSION
    preset: str = "operational"
    window: WindowConfig = field(default_factory=WindowConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    layers: LayerConfig = field(default_factory=LayerConfig)
    imagery: ImageryConfig = field(default_factory=ImageryConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        if self.schema_version != VIEWER_SCHEMA_VERSION:
            raise ValueError("unsupported viewer configuration schema")
        if self.preset not in {
            "operational",
            "fire_behavior",
            "belief",
            "suppression",
            "logistics",
            "imagery",
        }:
            raise ValueError(f"unknown viewer preset: {self.preset}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_PRESET_LAYERS: dict[str, dict[str, bool]] = {
    "operational": {},
    "fire_behavior": {
        "water": False,
        "retardant": False,
        "belief_perimeter": False,
        "vehicle_labels": False,
        "fire_type": True,
        "contours": True,
    },
    "belief": {
        "active_fire": False,
        "water": False,
        "retardant": False,
        "belief_perimeter": True,
        "belief_uncertainty": True,
        "fire_type": False,
    },
    "suppression": {
        "belief_perimeter": False,
        "water": True,
        "retardant": True,
        "constructed_line": True,
        "vehicle_tracks": True,
    },
    "logistics": {
        "belief_perimeter": False,
        "fire_type": False,
        "water": False,
        "retardant": False,
        "service_sites": True,
        "vehicle_tracks": True,
        "vehicle_labels": True,
        "targets": True,
    },
    "imagery": {
        "imagery": True,
        "hillshade": True,
        "contours": False,
        "belief_perimeter": False,
        "fire_type": False,
    },
}


def _nested(cls, value: Any):
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} configuration must be a mapping")
    payload = dict(value)
    if cls is ImageryConfig and "bands" in payload:
        payload["bands"] = tuple(payload["bands"])
    return cls(**payload)


def load_viewer_config(path: str | Path | None = None) -> ViewerConfig:
    """Load a viewer manifest and resolve imagery relative to that manifest."""

    raw: dict[str, Any] = {}
    source: Path | None = None
    if path is not None:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError("viewer configuration must be a mapping")
        raw = loaded
    preset = str(raw.get("preset", "operational"))
    layer_values = dict(_PRESET_LAYERS.get(preset, {}))
    explicit_layers = raw.get("layers", {})
    if not isinstance(explicit_layers, dict):
        raise TypeError("layers configuration must be a mapping")
    layer_values.update(explicit_layers)
    imagery = _nested(ImageryConfig, raw.get("imagery"))
    if imagery.path is not None and source is not None:
        imagery_path = Path(imagery.path)
        if not imagery_path.is_absolute():
            imagery = replace(
                imagery,
                path=str((source.parent / imagery_path).resolve()),
            )
    return ViewerConfig(
        schema_version=int(raw.get("schema_version", VIEWER_SCHEMA_VERSION)),
        preset=preset,
        window=_nested(WindowConfig, raw.get("window")),
        playback=_nested(PlaybackConfig, raw.get("playback")),
        camera=_nested(CameraConfig, raw.get("camera")),
        layers=LayerConfig(**layer_values),
        imagery=imagery,
        export=_nested(ExportConfig, raw.get("export")),
    )
