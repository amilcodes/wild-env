"""Open a replay in the native desktop viewer."""

from __future__ import annotations

import argparse

from aeolus.replay import ReplayBundle
from aeolus.viewer.app import run_viewer
from aeolus.viewer.config import load_viewer_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Aeolus replay in the native Qt/VTK viewer")
    parser.add_argument("--replay", required=True, help="ReplayBundle directory")
    parser.add_argument("--config", help="Viewer YAML manifest")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame index")
    args = parser.parse_args()
    replay = ReplayBundle.open(args.replay)
    config = load_viewer_config(args.config)
    raise SystemExit(run_viewer(replay, config, start_frame=args.frame))


if __name__ == "__main__":
    main()
