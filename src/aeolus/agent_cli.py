"""Run, inspect, and replay frontier-agent incident-desk benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from aeolus.agents.adapters import BASELINES, AgentSpec
from aeolus.agents.benchmark import load_suite, replay_trace, run_benchmark
from aeolus.agents.contract import DISPATCH_SCHEMA, PROTOCOL_VERSION, SYSTEM_PROMPT, canonical_json
from aeolus.agents.desk import IncidentDesk
from aeolus.agents.report import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire incident desk: benchmark LLMs and agents")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute matched-seed episodes and write an audit report")
    run.add_argument("--suite", required=True)
    run.add_argument("--agents", help="YAML file with an agents list; default runs hold, nearest, greedy")
    run.add_argument(
        "--out", type=Path, required=True, help="new output directory (existing runs are never overwritten)"
    )
    inspect = commands.add_parser("inspect", help="print the first agent-visible packet without an API call")
    inspect.add_argument("--suite", required=True)
    inspect.add_argument("--case", help="case ID; defaults to first case")
    inspect.add_argument("--seed", type=int, help="defaults to first configured seed")
    replay = commands.add_parser("replay", help="verify a transcript by re-executing it without a model")
    replay.add_argument("trace", type=Path)
    report = commands.add_parser("report", help="regenerate a self-contained HTML report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--out", type=Path)
    commands.add_parser("schema", help="print the versioned prompt and submit_dispatch schema")
    args = parser.parse_args()
    try:
        if args.command == "run":
            if args.agents:
                raw = yaml.safe_load(Path(args.agents).read_text())
                if not isinstance(raw, dict) or set(raw) != {"agents"} or not isinstance(raw["agents"], list):
                    raise ValueError("agent manifest must contain an agents list")
                specs = [AgentSpec(**item) for item in raw["agents"]]
            else:
                specs = [AgentSpec(name=name, model=name) for name in BASELINES]
            result = run_benchmark(
                args.suite, specs, args.out, progress=lambda msg: print(msg, file=sys.stderr)
            )
            report_path = render_report(args.out)
            print(
                json.dumps(
                    {
                        "complete": result["complete"],
                        "episodes": len(result["records"]),
                        "report": str(report_path.resolve()),
                        "leaderboard": result["leaderboard"],
                    },
                    indent=2,
                )
            )
            if any(not row["eligible"] for row in result["leaderboard"]):
                raise SystemExit(2)
        elif args.command == "inspect":
            _, cases = load_suite(args.suite)
            case = (
                next((case for case in cases if case.case_id == args.case), None) if args.case else cases[0]
            )
            if case is None:
                raise ValueError("unknown case ID")
            seed = case.seeds[0] if args.seed is None else args.seed
            print(canonical_json(IncidentDesk(case.scenario, seed).observation()))
        elif args.command == "replay":
            print(json.dumps(replay_trace(args.trace), indent=2))
        elif args.command == "report":
            print(render_report(args.run_dir, args.out).resolve())
        else:
            print(
                json.dumps(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "system": SYSTEM_PROMPT,
                        "submit_dispatch": DISPATCH_SCHEMA,
                    },
                    indent=2,
                )
            )
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
