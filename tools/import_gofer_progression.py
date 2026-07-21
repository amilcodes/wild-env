#!/usr/bin/env python3
"""Import one fire from the published GOFER v0.2 archive."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from aeolus.data.gofer import gofer_fire_catalog, write_gofer_observation_bundle


def _extract_archive(archive: Path, destination: Path, variant: str) -> Path:
    names = {
        "combined": ("GOFER_Combined", "GOFERC"),
        "east": ("GOFER_East", "GOFERE"),
        "west": ("GOFER_West", "GOFERW"),
    }
    directory, prefix = names[variant]
    members = {
        "GOFER/fireData.csv",
        f"GOFER/{directory}/{prefix}_fireProg.shp",
        f"GOFER/{directory}/{prefix}_fireProg.shx",
        f"GOFER/{directory}/{prefix}_fireProg.dbf",
        f"GOFER/{directory}/{prefix}_fireProg.prj",
        f"GOFER/{directory}/{prefix}_cfireLine.shp",
        f"GOFER/{directory}/{prefix}_cfireLine.shx",
        f"GOFER/{directory}/{prefix}_cfireLine.dbf",
        f"GOFER/{directory}/{prefix}_cfireLine.prj",
        f"GOFER/{directory}/{prefix}_rfireLine.shp",
        f"GOFER/{directory}/{prefix}_rfireLine.shx",
        f"GOFER/{directory}/{prefix}_rfireLine.dbf",
        f"GOFER/{directory}/{prefix}_rfireLine.prj",
        f"GOFER/{directory}/{prefix}_summary.csv",
    }
    with zipfile.ZipFile(archive) as source:
        available = set(source.namelist())
        missing = sorted(members - available)
        if missing:
            raise FileNotFoundError(f"GOFER archive members are missing: {missing}")
        for member in sorted(members):
            target = (destination / member).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe archive member: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as compressed, target.open("wb") as extracted:
                while chunk := compressed.read(1024 * 1024):
                    extracted.write(chunk)
    return destination / "GOFER"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="GOFER.zip or extracted GOFER directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fire", help="exact GOFER fire name")
    parser.add_argument("--year", type=int)
    parser.add_argument("--variant", choices=("combined", "east", "west"), default="combined")
    parser.add_argument("--concurrent-confidence", type=float, default=0.05)
    parser.add_argument("--list", action="store_true", help="list fires and exit")
    args = parser.parse_args()

    if not args.list and (args.fire is None or args.year is None):
        parser.error("--fire and --year are required unless --list is used")

    if args.source.is_file():
        with tempfile.TemporaryDirectory(prefix="aeolus-gofer-") as temporary:
            root = _extract_archive(args.source, Path(temporary), args.variant)
            if args.list:
                print(json.dumps(gofer_fire_catalog(root), indent=2))
                return
            manifest = write_gofer_observation_bundle(
                root,
                args.out,
                fire_name=args.fire,
                fire_year=args.year,
                variant=args.variant,
                concurrent_confidence=args.concurrent_confidence,
                source_archive=args.source,
            )
    else:
        if args.list:
            print(json.dumps(gofer_fire_catalog(args.source), indent=2))
            return
        manifest = write_gofer_observation_bundle(
            args.source,
            args.out,
            fire_name=args.fire,
            fire_year=args.year,
            variant=args.variant,
            concurrent_confidence=args.concurrent_confidence,
        )
    print(json.dumps({"fire": manifest["fire"], "audit": manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
