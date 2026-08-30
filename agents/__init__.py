"""Specialised agent roles for the RankAgent research loop.

    ProductManagerAgent  what to work on   (coverage, stopping, rotation)
    ResearchAgent        what to try       (hypotheses with mechanisms)
    EngineerAgent        how to run it     (validated, non-duplicate commands)
    QAAgent              did it work       (pre-flight, verdicts, self-healing)

They coordinate through ``AgentTeam`` over a shared ``ResearchContext`` blackboard.
See agents/team.py for the per-iteration flow and the cost controls.
"""
from agents.base import LLMClient
from agents.context import DIMENSIONS, ResearchContext, TrialRecord, command_signature
from agents.engineer import EngineerAgent, TrialSpec, validate_args
from agents.product_manager import Directive, ProductManagerAgent
from agents.qa import QAAgent
from agents.researcher import Hypothesis, ResearchAgent
from agents.team import AgentTeam, IterationPlan

__all__ = [
    "LLMClient", "ResearchContext", "TrialRecord", "DIMENSIONS", "command_signature",
    "ProductManagerAgent", "Directive", "ResearchAgent", "Hypothesis",
    "EngineerAgent", "TrialSpec", "validate_args", "QAAgent",
    "AgentTeam", "IterationPlan",
]
