"""Execute and audit the finite operations benchmark on the authorized compute host."""

import argparse
import json
from pathlib import Path

from aeolus.agents.operations_benchmark import analyze, replay, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    execute = subs.add_parser("run")
    execute.add_argument("--suite", required=True)
    execute.add_argument("--agents")
    execute.add_argument("--out", required=True, type=Path)
    execute.add_argument("--workers", type=int, default=4)
    execute.add_argument("--models-only", action="store_true")
    inspect = subs.add_parser("replay")
    inspect.add_argument("trace", type=Path)
    audit = subs.add_parser("analyze")
    audit.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.suite, args.agents, args.out, args.workers, not args.models_only)
        print(json.dumps(result["leaderboard"], indent=2))
        if not all(row["eligible"] for row in result["records"]):
            raise SystemExit(2)
    elif args.command == "replay":
        print(json.dumps(replay(args.trace), indent=2))
    else:
        print(json.dumps(analyze(args.out), indent=2))


if __name__ == "__main__":
    main()
