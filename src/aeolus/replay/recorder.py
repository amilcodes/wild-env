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

Policy = Callable[[AeolusSimulator], dict[str, int]]

REPLAY_SCHEMA_VERSION = 1


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
        self.belief_mean: list[np.ndarray] = []
        self.belief_std: list[np.ndarray] = []
        self.known_burned: list[np.ndarray] = []
        self.water: list[np.ndarray] = []
        self.retardant: list[np.ndarray] = []
        self.ground_hold: list[np.ndarray] = []
        self.resource_x: list[np.ndarray] = []
        self.resource_y: list[np.ndarray] = []
        self.resource_status: list[np.ndarray] = []
        self.resource_payload: list[np.ndarray] = []
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
        append(self.belief_mean, state.belief.intensity_mean.astype(np.float32, copy=True))
        append(self.belief_std, state.belief.intensity_std.astype(np.float32, copy=True))
        append(self.known_burned, state.belief.known_burned.astype(np.float32, copy=True))
        append(self.water, state.truth.water.astype(np.float32, copy=True))
        append(self.retardant, state.truth.retardant.astype(np.float32, copy=True))
        append(self.ground_hold, state.truth.ground_hold.astype(np.float32, copy=True))
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
            ("belief/intensity_mean", self.belief_mean),
            ("belief/intensity_std", self.belief_std),
            ("belief/known_burned", self.known_burned),
            ("treatment/water", self.water),
            ("treatment/retardant", self.retardant),
            ("treatment/ground_hold", self.ground_hold),
        ):
            write(name, np.stack(values), raster_chunks)
        resource_chunks = (min(256, len(self.minutes)), len(simulator.state.resources))
        write("resources/x", np.stack(self.resource_x), resource_chunks)
        write("resources/y", np.stack(self.resource_y), resource_chunks)
        write("resources/status", np.stack(self.resource_status), resource_chunks)
        write("resources/payload_fraction", np.stack(self.resource_payload), resource_chunks)
        write("static/elevation_m", simulator.state.truth.elevation_m, shape)
        write("static/fuel_load_kg_m2", simulator.state.truth.fuel_load, shape)
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
        metadata = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "policy_name": policy_name,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "resource_ids": simulator.agent_ids,
            "resource_kinds": [resource.spec.kind for resource in simulator.state.resources],
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
        if metadata.get("schema_version") != REPLAY_SCHEMA_VERSION:
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
