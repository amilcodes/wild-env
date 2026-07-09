"""Prepare and run the historical validation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.evaluation.study import (
    prepare_study,
    refresh_prepared_weather,
    refresh_study_summaries,
    run_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIROPS historical validation study")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--source-shapefile", required=True)
    prepare.add_argument("--out", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--prepared-root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument(
        "--workers",
        type=int,
        help="Override manifest parallel_workers for this execution",
    )
    refresh = subparsers.add_parser("refresh-summaries")
    refresh.add_argument("--results", required=True)
    refresh_weather = subparsers.add_parser("refresh-weather")
    refresh_weather.add_argument("--manifest", required=True)
    refresh_weather.add_argument("--prepared-root", required=True)
    refresh_weather.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_study(
            args.manifest,
            args.source_shapefile,
            args.out,
        )
        summary = result
    elif args.command == "refresh-weather":
        result = refresh_prepared_weather(
            args.manifest,
            args.prepared_root,
            args.out,
        )
        summary = result
    elif args.command == "run":
        result = run_study(
            args.manifest,
            args.prepared_root,
            args.out,
            workers=args.workers,
        )
        summary = {
            "study": result["study"],
            "forecast_count": len(result["forecasts"]),
            "summaries": result["summaries"],
            "probabilistic_summaries": result["probabilistic_summaries"],
            "artifacts": {
                "results": str((Path(args.out) / "historical_validation_results.json").resolve()),
                "examples": str((Path(args.out) / "historical_validation_examples.npz").resolve()),
            },
        }
    else:
        result = refresh_study_summaries(args.results)
        summary = {
            "results": str(Path(args.results).resolve()),
            "probabilistic_summaries": result["probabilistic_summaries"],
            "probabilistic_active_growth_summaries": result["probabilistic_active_growth_summaries"],
            "probabilistic_skill_against_persistence": result["probabilistic_skill_against_persistence"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
