"""Command line: ``aeolus-ladder run|summarize|show|references``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.ladder.generator import TIERS, generate
from aeolus.ladder.prompt import SYSTEM_PROMPT, observation
from aeolus.ladder.runner import load_models, load_suite, references, run, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aeolus-ladder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="solve references, run baselines and models, write summary")
    p.add_argument("--suite", required=True)
    p.add_argument("--models", help="models YAML; omit to run references and baselines only")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*", help="restrict to these model names")
    p.add_argument("--max-attempts", type=int, default=2, help="attempts per call on infrastructure errors")
    s = sub.add_parser("summarize", help="rebuild summary.json / report.txt from calls.jsonl")
    s.add_argument("out", type=Path)
    s.add_argument("--suite", required=True)
    s.add_argument("--models")
    v = sub.add_parser("show", help="print the exact model input for one instance")
    v.add_argument("--tier", type=int, required=True, choices=sorted(TIERS))
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--system", action="store_true", help="also print the system prompt")
    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run(args.suite, args.models, args.out, args.workers, args.max_attempts, args.only)
        print((args.out / "report.txt").read_text())
        return 0 if summary else 1
    if args.command == "summarize":
        _, instances = load_suite(args.suite)
        refs = references(instances, args.out)
        specs = load_models(args.models) if args.models else []
        summarize(args.out, instances, refs, specs)
        print((args.out / "report.txt").read_text())
        return 0
    if args.system:
        print(SYSTEM_PROMPT)
    print(json.dumps(observation(generate(args.tier, args.seed)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
