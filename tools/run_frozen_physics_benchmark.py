#!/usr/bin/env python3
"""Run canonical physics under the frozen incident-holdout contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.evaluation.frozen_physics import run_frozen_physics_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--operational-forcing-root", type=Path)
    args = parser.parse_args()
    result = run_frozen_physics_benchmark(
        args.contract,
        args.prepared_root,
        parallel_workers=args.workers,
        cache_directory=args.cache or args.output.parent / "physics_cache",
        operational_forcing_root=args.operational_forcing_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"fit": result["fit"], "summaries": result["summaries"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
