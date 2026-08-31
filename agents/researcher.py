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

    #: Which mutable file implementing this would touch, and a prose sketch of
    #: the change. Empty means "no code change needed" — a config-only trial,
    #: which is still a legitimate experiment (a seed replicate, or running an
    #: architecture that already exists).
    target_file: str = ""
    edit_sketch: str = ""

    #: Who actually authored this: "llm" or "playbook". Deliberately has no
    #: default — see the note on TrialSpec.source in agents/engineer.py.
    source: str

    @field_validator("dimension")
    @classmethod
    def _known(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in DIMENSIONS:
            raise ValueError(f"unknown dimension {v!r}")
        return v


class HypothesisSet(BaseModel):
    hypotheses: List[Hypothesis] = Field(..., min_length=1)


def _stamp_source(payload: Any, source: str) -> Any:
    """Set provenance on parsed hypotheses.

    The model is never asked to declare where a hypothesis came from — it would
    have no way to know, and a field the model fills is a field the model can
    get wrong. Provenance is stamped by whichever code path produced it, which
    is the only place that actually knows.
    """
    if isinstance(payload, dict):
        for h in payload.get("hypotheses", []) or []:
            if isinstance(h, dict):
                h["source"] = source
    return payload


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
  --loss        pointwise | listwise | bpr
                NOTE: `fm` (numpy) and `lgb` (LightGBM) have their own trainers
                and IGNORE --loss completely. A hypothesis about an objective
                must run on fm_torch, deepfm, din or mmoe, or the new loss is
                never called and you measure the stock model instead.
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

WRITING CODE IS THE POINT
The flags above only reach methods that already exist. Prefer hypotheses that
require NEW CODE, and say where it goes:
  target_file  pipeline/models.py    losses and architectures
               pipeline/features.py  encoders, causal statistics, sequences
               pipeline/train.py     optimiser, schedule, batching, early stop
  edit_sketch  what to implement, concretely enough for an engineer to write it.

Established methods worth implementing here, with the reason each suits a
*within-user* ranking objective:
- ApproxNDCG (Qin et al. 2010) / NeuralNDCG (Pobrotyn 2021) / LambdaLoss (Wang
  et al. 2018): optimise a smooth surrogate of nDCG@5 directly, instead of a
  likelihood that only correlates with it.
- ListMLE (Xia et al. 2008): full-permutation likelihood over a user's list.
- Focal loss (Lin et al. 2017): long_view positives are sparse; down-weight the
  easy negatives that dominate the gradient.
- DCN-v2 (Wang et al. 2021): explicit bounded-degree feature crosses.
- PLE (Tang et al. 2020): separates shared from task-specific experts where
  MMoE's shared trunk lets tasks fight.
- User x item cross statistics in CausalStats: user-constant features cannot
  reorder a user's list, but interactions between the user and the candidate can.

Leave target_file and edit_sketch empty ONLY when the experiment genuinely needs
no new code (a seed replicate, or a configuration of something already built).

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

Code already applied to the pipeline you are extending:
  {ctx.lineage}

{{
  "hypotheses": [
    {{
      "dimension": "one of the focus dimensions",
      "hypothesis": "what you are testing",
      "mechanism": "why this should move a within-user ranking metric",
      "target_file": "pipeline/models.py, pipeline/features.py, pipeline/train.py, or \\"\\"",
      "edit_sketch": "the code change to make, concretely; empty if none is needed",
      "args": "--model ... --loss ..."
    }}
  ]
}}"""

    def _parse(self, payload: Any, ctx: ResearchContext, **kwargs) -> List[Hypothesis]:
        payload = _stamp_source(payload, "llm")
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
                                  args=entry["args"], source="playbook"))
            if len(out) >= self.proposals:
                break
        if not out:
            # Directive dimensions exhausted — widen rather than return nothing.
            for entry in entries_for(list(DIMENSIONS)):
                out.append(Hypothesis(dimension=entry["dimension"],
                                      hypothesis=entry["hypothesis"],
                                      mechanism=entry["mechanism"],
                                      args=entry["args"], source="playbook"))
                if len(out) >= self.proposals:
                    break
        return out
