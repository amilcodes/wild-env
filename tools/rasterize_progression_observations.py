#!/usr/bin/env python3
"""Align a progression observation bundle to an IncidentBundle grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.data.progression import rasterize_progression_observation_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = rasterize_progression_observation_bundle(
        args.incident,
        args.observations,
        args.out,
    )
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
