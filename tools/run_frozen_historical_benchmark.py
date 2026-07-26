#!/usr/bin/env python3
"""Run the frozen incident-holdout geometric benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeolus.evaluation.frozen_benchmark import run_frozen_baseline_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run_frozen_baseline_benchmark(args.contract, args.prepared_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"claim_gate": result["claim_gate"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
