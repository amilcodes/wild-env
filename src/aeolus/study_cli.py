"""Prepare and run the historical validation study."""

from __future__ import annotations

import argparse
import json

from aeolus.evaluation.study import prepare_study, run_study


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
    args = parser.parse_args()
    result = (
        prepare_study(args.manifest, args.source_shapefile, args.out)
        if args.command == "prepare"
        else run_study(args.manifest, args.prepared_root, args.out)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
