"""Product Manager agent — decides *where* to look, not *what* to build.

WHAT THIS ROLE IS FOR
---------------------
It would be easy to dismiss a PM agent as org-chart cosplay. The run logs say
otherwise. In the LLM-driven run of 2026-08-29 the model:

  * proposed MMoE at iteration 1, tuned it at iteration 3, and at iteration 4
    re-ran an identical configuration while claiming it had changed a parameter;
  * never once varied the loss function across four iterations, despite listwise
    being the strongest single lever measured on this benchmark.

Both are failures of *portfolio management*, not of hypothesis quality. A single
agent optimising locally will exploit the first thing that works and never notice
an untouched axis. This role owns exactly that: coverage, and knowing when to stop
exploiting.

It also matters for the halt condition. Convergence fires when the best-so-far
curve gains under 0.002 over 3 iterations, so a run that front-loads its best idea
halts early with most of the budget unspent. Steering exploration keeps genuine
improvements arriving.

Cost control: this agent runs once every ``refresh_every`` iterations, not every
iteration, because a roadmap that changes every trial is not a roadmap.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from agents.base import Agent, validated
from agents.context import DIMENSIONS, KNOWN_DEAD_ENDS, ResearchContext
from agents.playbook import DIMENSION_PRIORITY


class Directive(BaseModel):
    """The standing instruction the Researcher must work within."""
    phase: str = Field(..., description="Short name for the current research phase")
    focus_dimensions: List[str] = Field(..., min_length=1)
    avoid_dimensions: List[str] = Field(default_factory=list)
    reasoning: str = ""
    valid_for: int = 3

    @field_validator("focus_dimensions", "avoid_dimensions")
    @classmethod
    def _known_dimensions(cls, v: List[str]) -> List[str]:
        cleaned = [d.strip().lower() for d in v if d and d.strip().lower() in DIMENSIONS]
        return cleaned

    @field_validator("focus_dimensions")
    @classmethod
    def _non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("focus_dimensions contained no recognised dimension")
        return v


SYSTEM = """You are the Product Manager on an autonomous ML research team working on
the KuaiRand-Pure within-user ranking benchmark. You do not design models and you do
not write code. You decide which parts of the search space the team should work on
next, and you are accountable for coverage.

Your standing concerns, in order:
1. COVERAGE. A dimension that has never been tried is worth more than a fourth
   variation of one that has. Untried axes are listed to you explicitly.
2. EVIDENCE. Do not direct effort at approaches already measured as dead ends.
3. STOPPING. The run halts when the best score gains less than 0.002 over 3
   consecutive iterations. If recent work has stalled, change dimension rather than
   letting the run converge on a local optimum with budget left.
4. SIGNIFICANCE. Seed noise is 0.0008. Differences under 0.0024 on a single seed
   are not real findings and must not drive a change of direction.

Reply with a single JSON object and nothing else."""


class ProductManagerAgent(Agent):
    name = "product_manager"
    system_prompt = SYSTEM
    max_tokens = 700

    def __init__(self, llm=None, refresh_every: int = 3, verbose: bool = True):
        super().__init__(llm, verbose)
        self.refresh_every = refresh_every

    def should_refresh(self, ctx: ResearchContext) -> bool:
        """Re-plan on the first iteration, on expiry, or when progress stalls."""
        if ctx.directive is None:
            return True
        if ctx.iteration % self.refresh_every == 1:
            return True
        return self._stalled(ctx)

    @staticmethod
    def _stalled(ctx: ResearchContext, window: int = 3) -> bool:
        recent = [t for t in ctx.trials if t.succeeded][-window:]
        if len(recent) < window or ctx.best_score is None:
            return False
        return all(t.primary < ctx.best_score for t in recent)

    # -- LLM path ---------------------------------------------------------

    def _build_prompt(self, ctx: ResearchContext, **kwargs) -> str:
        return f"""Iteration {ctx.iteration} of {ctx.max_iterations}. \
{ctx.iterations_remaining} remain, {ctx.budget_remaining_s / 60:.0f} minutes of budget left.

Official baseline validation primary: {ctx.baseline:.4f}
Best so far: {('%.4f' % ctx.best_score) if ctx.best_score is not None else 'nothing yet'} \
(delta {ctx.best_delta:+.4f})
{ctx.significance_note()}

{ctx.coverage_report()}

Experiments so far:
{ctx.history_table()}

Measured dead ends — do not direct effort here:
{chr(10).join('- ' + d for d in KNOWN_DEAD_ENDS)}

Available dimensions: {', '.join(DIMENSIONS)}

Set the research direction for the next few iterations.

{{
  "phase": "short name for this phase",
  "focus_dimensions": ["one or two dimensions from the list above"],
  "avoid_dimensions": ["dimensions to stay off for now"],
  "reasoning": "why this is the right place to spend the next iterations",
  "valid_for": 3
}}"""

    def _parse(self, payload: Any, ctx: ResearchContext, **kwargs) -> Directive:
        return validated(Directive, payload)

    # -- deterministic path -----------------------------------------------

    def fallback(self, ctx: ResearchContext, **kwargs) -> Directive:
        """Coverage-first planning without an LLM.

        Untried dimensions win, taken in measured-priority order. Once everything
        has been touched, exploit whichever dimension produced the best result,
        unless progress has stalled — then deliberately move elsewhere.
        """
        unexplored = [d for d in DIMENSION_PRIORITY if d in ctx.unexplored_dimensions()]
        if unexplored:
            focus = unexplored[:2]
            return Directive(
                phase=f"explore {focus[0]}",
                focus_dimensions=focus,
                reasoning=(f"{', '.join(focus)} has no experiments yet; an untried axis "
                           f"is worth more than another variation of a tried one."),
                valid_for=self.refresh_every)

        best_dim = self._best_dimension(ctx)
        if self._stalled(ctx) and best_dim:
            alternatives = [d for d in DIMENSION_PRIORITY if d != best_dim]
            counts = ctx.dimension_counts()
            alternatives.sort(key=lambda d: counts.get(d, 0))
            return Directive(
                phase=f"rotate away from {best_dim}",
                focus_dimensions=alternatives[:2],
                avoid_dimensions=[best_dim],
                reasoning=("the last three experiments all failed to beat the best score; "
                           "continuing on the same axis risks converging with budget unspent."),
                valid_for=self.refresh_every)

        focus = [best_dim] if best_dim else [DIMENSION_PRIORITY[0]]
        return Directive(
            phase=f"exploit {focus[0]}",
            focus_dimensions=focus,
            reasoning=f"{focus[0]} produced the current best result; refine it.",
            valid_for=self.refresh_every)

    @staticmethod
    def _best_dimension(ctx: ResearchContext) -> Optional[str]:
        best = None
        best_score = float("-inf")
        for t in ctx.trials:
            if t.primary is not None and t.primary > best_score:
                best, best_score = t.dimension, t.primary
        return best
