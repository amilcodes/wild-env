"""Record and render a policy episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import PerimeterSeries
from aeolus.replay import (
    export_paraview,
    record_episode,
    render_frame_2d,
    render_frame_3d,
    render_video,
)
from aeolus.viewer import load_viewer_config
from aeolus.workflows import resolve_policy, scenario_from_incident


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and render a deterministic Aeolus replay")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--incident")
    parser.add_argument("--policy", default="joint_assignment")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--horizon-min", type=int, default=180)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--out", required=True, help="replay bundle directory")
    parser.add_argument("--frame-2d")
    parser.add_argument("--frame-3d")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index for still exports; negative indices count from the end",
    )
    parser.add_argument("--video")
    parser.add_argument("--paraview", help="Output directory for a VTK/ParaView time series")
    parser.add_argument("--viewer-config", help="Viewer/export YAML manifest")
    parser.add_argument(
        "--view",
        choices=("operational_2d", "terrain_3d"),
        default="operational_2d",
        help="Video view",
    )
    parser.add_argument("--selected-resource", help="Vehicle to select or follow in exports")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-video-frames", type=int, default=120)
    args = parser.parse_args()
    viewer_config = load_viewer_config(args.viewer_config)

    initializer = None
    if args.incident:
        incident = IncidentBundle.load(args.incident)
        config = scenario_from_incident(
            incident,
            seed=args.seed,
            horizon_min=args.horizon_min,
        )
        frame = PerimeterSeries.from_incident(incident).frames[args.start_index]

        def initialize(simulator: AeolusSimulator) -> None:
            simulator.initialize_from_observed_perimeter(
                frame.mask,
                source=f"incident:{incident.incident_id}:{frame.timestamp.isoformat()}",
            )

        initializer = initialize
    else:
        experiment = load_config(args.config)
        config = experiment.scenario

    policy, checkpoint_path = resolve_policy(args.policy, checkpoint=args.checkpoint)
    replay = record_episode(
        AeolusSimulator(config),
        policy,
        args.out,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        policy_name=args.policy,
        initialize=initializer,
    )
    episode = replay.metadata["episode"]
    outputs: dict[str, object] = {
        "replay": str(Path(args.out).resolve()),
        "frames": replay.frame_count,
        "episode": {
            key: episode[key]
            for key in (
                "minute",
                "escaped",
                "contained",
                "truncated",
                "weighted_loss",
                "burned_fraction",
                "active_fraction",
                "blocked_actions",
                "resource",
            )
        },
    }
    if args.frame_2d:
        outputs["frame_2d"] = str(
            render_frame_2d(
                replay,
                args.frame_2d,
                frame=args.frame,
                viewer_config=viewer_config,
                selected_resource=args.selected_resource,
            ).resolve()
        )
    if args.frame_3d:
        outputs["frame_3d"] = str(
            render_frame_3d(
                replay,
                args.frame_3d,
                frame=args.frame,
                viewer_config=viewer_config,
                selected_resource=args.selected_resource,
            ).resolve()
        )
    if args.video:
        outputs["video"] = str(
            render_video(
                replay,
                args.video,
                fps=args.fps,
                max_frames=args.max_video_frames,
                viewer_config=viewer_config,
                view=args.view,
                selected_resource=args.selected_resource,
            ).resolve()
        )
    if args.paraview:
        outputs["paraview"] = str(
            export_paraview(
                replay,
                args.paraview,
                max_frames=args.max_video_frames,
            ).resolve()
        )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
