"""A runnable command-protocol example, not an LLM or a reported frontier result."""

import json
import sys

from aeolus.agents.adapters import baseline_dispatch

request = json.load(sys.stdin)
print(json.dumps(baseline_dispatch(request["observation"], "greedy")))
