"""AgentTeam — coordinates the four roles for one iteration.

    Product Manager  ──▶ sets the direction (every N iterations, not every one)
    ML Researcher    ──▶ proposes k hypotheses inside that direction
    Engineer         ──▶ picks the first that validates and is not a repeat
    QA               ──▶ pre-flights it, then judges the result / repairs failures

All four read and write one ``ResearchContext``. Nothing is retold in prose between
roles, so the chain cannot drift the way a pipeline of paraphrases does.

COST
----
Naively this is four LLM calls an iteration, which would move the run from the
cheapest Feasibility tier into the most expensive one for no gain in score. So:

  * the Product Manager runs once every ``pm_refresh`` iterations (or when progress
    stalls), because a roadmap that changes every trial is not a roadmap;
  * the Researcher returns several hypotheses per call, so the Engineer has
    alternatives without a second round trip;
  * the Engineer only calls out when validation fails;
  * QA only calls out when a trial actually breaks.

In the steady state that is roughly one call per iteration, not four.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base import LLMClient
from agents.context import ResearchContext, TrialRecord, command_signature
from agents.engineer import EngineerAgent, TrialSpec
from agents.product_manager import Directive, ProductManagerAgent
from agents.qa import QAAgent
from agents.researcher import Hypothesis, ResearchAgent
from orchestrator.schemas import TokenUsage


@dataclass
class IterationPlan:
    """What the team decided to do this iteration, and how it got there."""
    spec: Optional[TrialSpec]
    directive: Optional[Directive]
    hypotheses: List[Hypothesis] = field(default_factory=list)
    preflight_problems: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    pm_refreshed: bool = False

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def as_log(self) -> Dict[str, Any]:
        return {
            "directive": self.directive.model_dump() if self.directive else None,
            "hypotheses_considered": [h.model_dump() for h in self.hypotheses],
            "chosen": self.spec.model_dump() if self.spec else None,
            "rejected_by_engineer": self.spec.rejected if self.spec else [],
            "preflight_problems": self.preflight_problems,
            "pm_refreshed": self.pm_refreshed,
            "trace": self.trace,
        }


class AgentTeam:
    def __init__(self, meter: TokenUsage, data_dir: Optional[str] = None,
                 pm_refresh: int = 3, proposals: int = 3, max_retries: int = 3,
                 verbose: bool = True):
        self.llm = LLMClient(meter)
        self.data_dir = data_dir
        self.verbose = verbose
        self.pm = ProductManagerAgent(self.llm, refresh_every=pm_refresh, verbose=verbose)
        self.researcher = ResearchAgent(self.llm, proposals=proposals, verbose=verbose)
        self.engineer = EngineerAgent(self.llm, verbose=verbose)
        self.qa = QAAgent(self.llm, max_retries=max_retries, verbose=verbose)

    @property
    def llm_available(self) -> bool:
        return self.llm.available

    @property
    def cost_by_agent(self) -> Dict[str, Dict[str, int]]:
        return self.llm.per_agent

    def _say(self, text: str):
        if self.verbose:
            print(text)

    # -- one iteration's planning ----------------------------------------

    def plan(self, ctx: ResearchContext) -> IterationPlan:
        """Run PM -> Researcher -> Engineer -> QA pre-flight for one iteration."""
        trace: List[str] = []

        # 1. Product Manager — direction.
        refreshed = self.pm.should_refresh(ctx)
        if refreshed:
            directive = self.pm.run(ctx)
            ctx.directive = directive
            self._say(f"    [PM]         phase '{directive.phase}' — focus "
                      f"{', '.join(directive.focus_dimensions)}")
            self._say(f"                 {directive.reasoning}")
            trace.append(f"PM set phase '{directive.phase}'")
        else:
            directive = ctx.directive
            trace.append(f"PM directive '{directive.phase}' still current")

        # 2. Researcher — hypotheses.
        hypotheses = self.researcher.run(ctx)
        self._say(f"    [RESEARCH]   {len(hypotheses)} hypothesis candidate(s) in "
                  f"{', '.join(sorted({h.dimension for h in hypotheses}))}")
        trace.append(f"Researcher proposed {len(hypotheses)}")

        # 3. Engineer — a runnable, non-duplicate, validated command.
        spec = self.engineer.build(ctx, hypotheses, data_dir=self.data_dir)
        if spec is None:
            # Every candidate was invalid or already run. Widen the directive once
            # and ask again rather than wasting the iteration.
            trace.append("Engineer rejected every candidate; widening the directive")
            self._say("    [ENGINEER]   all candidates invalid or duplicates; widening search")
            ctx.directive = self.pm.fallback(ctx)
            hypotheses = self.researcher.fallback(ctx)
            spec = self.engineer.build(ctx, hypotheses, data_dir=self.data_dir)

        if spec is None:
            trace.append("no runnable experiment found")
            return IterationPlan(spec=None, directive=directive, hypotheses=hypotheses,
                                 trace=trace, pm_refreshed=refreshed)

        for note in spec.rejected:
            self._say(f"    [ENGINEER]   rejected {note}")
            trace.append(f"rejected {note}")
        self._say(f"    [ENGINEER]   {spec.args}")

        # 4. QA — pre-flight before spending a training run.
        verdict = self.qa.preflight(ctx, spec)
        if not verdict.ok:
            for p in verdict.problems:
                self._say(f"    [QA]         BLOCKED: {p}")
            trace.extend(f"QA blocked: {p}" for p in verdict.problems)
            return IterationPlan(spec=None, directive=directive, hypotheses=hypotheses,
                                 preflight_problems=verdict.problems, trace=trace,
                                 pm_refreshed=refreshed)

        self._say("    [QA]         pre-flight passed")
        trace.append("QA pre-flight passed")
        return IterationPlan(spec=spec, directive=directive, hypotheses=hypotheses,
                             trace=trace, pm_refreshed=refreshed)

    # -- after execution --------------------------------------------------

    def review(self, primary: Optional[float]):
        """QA's verdict on a completed trial."""
        verdict = self.qa.judge(primary)
        if verdict.note:
            self._say(f"    [QA]         {verdict.note}")
        return verdict

    def recover(self, command: str, traceback_text: str, run: Callable[[str], Any]):
        return self.qa.recover(command, traceback_text, run)

    def record(self, ctx: ResearchContext, iteration: int, spec: TrialSpec,
               primary: Optional[float], status: str, error_kind: Optional[str] = None):
        ctx.record(TrialRecord(
            iteration=iteration, dimension=spec.dimension, hypothesis=spec.hypothesis,
            command=spec.command, signature=command_signature(spec.args),
            primary=primary, status=status,
            source="llm" if self.llm_available else "fallback",
            error_kind=error_kind))
