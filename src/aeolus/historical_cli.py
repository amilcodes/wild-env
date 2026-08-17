"""Historical hindcast, shadow replay, and counterfactual evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.core.simulator import AeolusSimulator
from aeolus.data import IncidentBundle
from aeolus.evaluation.historical import (
    PerimeterSeries,
    calibrate_spread_adjustment,
    compare_counterfactual_policies,
    run_hindcast,
    run_shadow_replay,
)
from aeolus.workflows import resolve_policy, scenario_from_incident


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate against historical wildfire timestamps")
    parser.add_argument("--incident", required=True)
    parser.add_argument(
        "--mode",
        choices=("hindcast", "shadow", "counterfactual", "calibrate"),
        required=True,
    )
    parser.add_argument("--policy", default="no_aerial")
    parser.add_argument("--policies", default="no_aerial,joint_assignment")
    parser.add_argument("--checkpoint")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=1)
    parser.add_argument("--validation-target-index", type=int)
    parser.add_argument(
        "--spread-candidates",
        default="",
        help="optional comma-separated positive spread multipliers",
    )
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    incident = IncidentBundle.load(args.incident)
    series = PerimeterSeries.from_incident(incident)
    delta = round(
        (
            series.frames[args.target_index].timestamp
            - series.frames[args.start_index].timestamp
        ).total_seconds()
        / 60.0
    )
    config = scenario_from_incident(
        incident,
        seed=args.seed,
        horizon_min=max(delta, 1),
    )
    if args.mode == "calibrate":
        policy = resolve_policy(args.policy, checkpoint=args.checkpoint)[0]
        candidates = (
            [float(value) for value in args.spread_candidates.split(",") if value]
            if args.spread_candidates
            else None
        )
        result = calibrate_spread_adjustment(
            config,
            series,
            policy,
            start_index=args.start_index,
            target_index=args.target_index,
            validation_target_index=args.validation_target_index,
            candidates=candidates,
        )
    elif args.mode == "counterfactual":
        policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
        policies = {
            name: resolve_policy(name, checkpoint=args.checkpoint)[0]
            for name in policy_names
        }
        seeds = [args.seed + index * 7919 for index in range(args.seeds)]
        result = compare_counterfactual_policies(
            config,
            series,
            policies,
            seeds,
            start_index=args.start_index,
            target_index=args.target_index,
        )
    else:
        policy = resolve_policy(args.policy, checkpoint=args.checkpoint)[0]
        simulator = AeolusSimulator(config)
        operation = run_hindcast if args.mode == "hindcast" else run_shadow_replay
        result = operation(
            simulator,
            series,
            policy,
            start_index=args.start_index,
            **(
                {"target_index": args.target_index}
                if args.mode == "hindcast"
                else {"end_index": args.target_index}
            ),
        )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printable = result.get(
        "summary",
        {key: value for key, value in result.items() if key != "episode"},
    )
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
