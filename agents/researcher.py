"""ML Research agent — proposes *what* to try, in the direction the PM set.

It produces hypotheses, not commands. Keeping the two apart is deliberate: the
Engineer agent is responsible for turning a hypothesis into something runnable and
for guaranteeing the two match. When one agent did both, the run log ended up
claiming DCN-v2 while executing ``--model mmoe``.

The agent proposes ``k`` hypotheses per call rather than one. That costs the same
single round trip but gives the Engineer alternatives to fall back on when the
first choice turns out to be a duplicate of something already run — which is
exactly how an earlier run wasted an iteration.
"""
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field, field_validator

from agents.base import Agent, validated
from agents.context import DIMENSIONS, KNOWN_DEAD_ENDS, ResearchContext
from agents.playbook import entries_for


class Hypothesis(BaseModel):
    dimension: str
    hypothesis: str = Field(..., min_length=10)
    mechanism: str = Field("", description="why this should move the metric")
    args: str = Field(..., description="pipeline.train arguments that test it")

    @field_validator("dimension")
    @classmethod
    def _known(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in DIMENSIONS:
            raise ValueError(f"unknown dimension {v!r}")
        return v


class HypothesisSet(BaseModel):
    hypotheses: List[Hypothesis] = Field(..., min_length=1)


SYSTEM = """You are the ML Research scientist on an autonomous team working on the
KuaiRand-Pure within-user ranking benchmark (label: long_view, metrics: GAUC and
nDCG@5, primary = their mean).

You propose experiments. You do not decide the research direction — the Product
Manager does that and you must stay inside the focus dimensions you are given. You
do not write or validate commands — the Engineer does that.

Benchmark facts you must reason with:
- Official baseline validation primary: 0.6016. Oracle ceiling: 0.8484, NOT 1.0,
  because 27% of users have no positive label at all.
- Seed noise sigma = 0.0008. A single-seed gain under 0.0024 is not evidence.
- Ranking happens WITHIN a user's own impressions. Any feature constant across a
  user's impressions cannot change their order.
- Training users average 43 impressions; evaluation users average 5.6.

Trainer arguments available (`python -m pipeline.train`):
  --model       fm | fm_torch | deepfm | din | mmoe | lgb
  --loss        pointwise | listwise | bpr      (ignored by fm and lgb)
  --embed_dim   16 | 32 | 64
  --experts     4 | 6 | 8           --expert_dim 64 | 96 | 128
  --aux_weight  0.1 | 0.3 | 0.5
  --lr          0.0003 | 0.0005 | 0.001 | 0.003
  --epochs      8 | 12 | 15 | 25    --batch_size 4096 | 8192 | 16384
  --trees       200 | 400 | 600     --num_leaves 31 | 63 | 127
  --objective   lambdarank | binary --max_seq_len 5 | 10 | 20
  --cwm         --seed <int>

Every hypothesis must state a MECHANISM: the reason this change should move a
within-user ranking metric. "It is a stronger model" is not a mechanism.

Reply with a single JSON object and nothing else."""


class ResearchAgent(Agent):
    name = "researcher"
    system_prompt = SYSTEM
    max_tokens = 1600

    def __init__(self, llm=None, proposals: int = 3, verbose: bool = True):
        super().__init__(llm, verbose)
        self.proposals = proposals

    def _build_prompt(self, ctx: ResearchContext, **kwargs) -> str:
        d = ctx.directive
        focus = ", ".join(d.focus_dimensions) if d else ", ".join(DIMENSIONS)
        avoid = ", ".join(d.avoid_dimensions) if d and d.avoid_dimensions else "nothing"
        tried = sorted(ctx.tried_signatures())
        tried_block = "\n".join(f"- {s}" for s in tried) or "- (nothing yet)"

        return f"""Product Manager directive — phase "{d.phase if d else 'open'}":
  focus on: {focus}
  stay off: {avoid}
  because: {d.reasoning if d else 'no directive set'}

Best so far: {('%.4f' % ctx.best_score) if ctx.best_score is not None else 'nothing yet'} \
(delta {ctx.best_delta:+.4f} vs baseline {ctx.baseline:.4f})
{ctx.significance_note()}

Experiments so far:
{ctx.history_table()}

Configurations ALREADY RUN — proposing any of these again wastes an iteration:
{tried_block}

Measured dead ends:
{chr(10).join('- ' + x for x in KNOWN_DEAD_ENDS)}

Propose {self.proposals} distinct experiments inside the focus dimensions, ordered
best first. Each must differ from every configuration already run.

{{
  "hypotheses": [
    {{
      "dimension": "one of the focus dimensions",
      "hypothesis": "what you are testing",
      "mechanism": "why this should move a within-user ranking metric",
      "args": "--model ... --loss ..."
    }}
  ]
}}"""

    def _parse(self, payload: Any, ctx: ResearchContext, **kwargs) -> List[Hypothesis]:
        result = validated(HypothesisSet, payload)
        allowed = set(ctx.directive.focus_dimensions) if ctx.directive else set(DIMENSIONS)
        # Honour the directive: drop anything outside it rather than silently
        # letting the researcher wander off the PM's plan.
        kept = [h for h in result.hypotheses if h.dimension in allowed]
        return kept or result.hypotheses

    def fallback(self, ctx: ResearchContext, **kwargs) -> List[Hypothesis]:
        """Playbook entries in the directive's dimensions that have not been run."""
        dims = ctx.directive.focus_dimensions if ctx.directive else list(DIMENSIONS)
        out: List[Hypothesis] = []
        for entry in entries_for(dims):
            out.append(Hypothesis(dimension=entry["dimension"],
                                  hypothesis=entry["hypothesis"],
                                  mechanism=entry["mechanism"],
                                  args=entry["args"]))
            if len(out) >= self.proposals:
                break
        if not out:
            # Directive dimensions exhausted — widen rather than return nothing.
            for entry in entries_for(list(DIMENSIONS)):
                out.append(Hypothesis(dimension=entry["dimension"],
                                      hypothesis=entry["hypothesis"],
                                      mechanism=entry["mechanism"],
                                      args=entry["args"]))
                if len(out) >= self.proposals:
                    break
        return out
