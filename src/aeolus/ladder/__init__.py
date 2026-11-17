"""Wildfire operations ladder (protocol ``wildfire-ops-v3``).

A difficulty-graded planning benchmark and RL environment built on the v2
operations dynamics (:mod:`aeolus.agents.operations`). Every instance has an
exact, information-matched optimum computed by a two-world mixed-integer
program, so scores are a fraction of a proven optimum at every size.
"""

from aeolus.ladder.generator import TIERS, generate
from aeolus.ladder.plan import PlanOutcome, evaluate_plan, execute_plan
from aeolus.ladder.solver import Reference, solve

__all__ = ["TIERS", "PlanOutcome", "Reference", "evaluate_plan", "execute_plan", "generate", "solve"]
