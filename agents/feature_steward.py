"""Feature Steward — the fifth role, owning the feature space.

The other four roles reason about *code*. This one reasons about *features*, and
it is deliberately not a code generator. Its action space is the
``FeatureRecipe``: a Pydantic-validated, content-hashed configuration of which
causal statistics to compute and how hard to smooth them.

WHY A BOUNDED ACTION SPACE IS THE RIGHT ONE HERE
------------------------------------------------
Feature engineering is where leakage happens. ``long_view`` is close to a
deterministic function of ``play_time_ms / duration_ms``, so a single plausible
line — ``watch_ratio = play_time / duration`` — scores brilliantly on validation
and is meaningless on the hidden test split, where watch time is withheld. Free
code generation over ``features.py`` is exactly the wrong tool for the one part
of the pipeline where a mistake is invisible and fatal.

So the Steward proposes recipes, which cannot leak by construction: every
feature they can select is already declared in a manifest and already proven, by
mutation test, not to depend on its own row's outcome. When genuinely new
feature *code* is wanted the Engineer still writes it — and then the Steward's
audits gate it, which is the same guarantee arriving by a different route.

A recipe is a third kind of experiment, distinct from both a flag toggle and a
code patch: its ranges are enforced, its identity is a hash of its behaviour,
and re-running the same id reproduces the same features exactly. The run log
records it as ``proposal_source: "recipe"`` for that reason.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from agents.base import Agent, validated
from agents.context import ResearchContext
from pipeline.feature_agent import FeatureEngineeringAgent
from pipeline.feature_recipes import FeatureRecipe, deterministic_recipes


class RecipeProposal(BaseModel):
    """One recipe the Steward wants to try, and why."""
    name: str = "proposed"
    base_profile: str = "affinity"
    include_features: Optional[List[str]] = None
    exclude_features: List[str] = Field(default_factory=list)
    item_smoothing: float = 15.0
    cross_smoothing: float = 8.0
    global_prior_strength: float = 50.0
    completion_ratio_clip: float = 5.0
    recency_cap_days: int = 30
    rationale: str = ""

    def to_recipe(self) -> FeatureRecipe:
        data = self.model_dump(exclude={"rationale"})
        return FeatureRecipe(**data)


SYSTEM = """You are the Feature Steward on an autonomous ML research team working on
the KuaiRand-Pure within-user ranking benchmark (label long_view, metrics GAUC
and nDCG@5, primary = their mean; official baseline 0.6016).

You do not write code. You configure the causal feature pipeline by proposing a
recipe, and every recipe is validated and audited before it runs.

WHAT MATTERS ON THIS BENCHMARK
Ranking happens WITHIN one user's own impression list. A feature that takes the
same value for every impression of a user cannot change that user's ordering and
therefore cannot help, however predictive it looks in aggregate. Features that
vary with the CANDIDATE — item history, author affinity, user x item crosses,
recency of this user's contact with this author — are the ones that can move the
metric.

The organizers measured that static side features and larger embeddings do not
help. Do not propose those.

SMOOTHING IS THE REAL LEVER
Every rate is a smoothed historical estimate. Low smoothing makes rare items
noisy; high smoothing collapses them toward the prior and destroys the signal
that separates candidates. item_smoothing and cross_smoothing trade those off,
and cross statistics are sparser so they usually want less smoothing than item
statistics, not more.

Reply with a single JSON object and nothing else."""


class FeatureStewardAgent(Agent):
    name = "feature_steward"
    system_prompt = SYSTEM
    max_tokens = 900

    def __init__(self, llm=None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.auditor = FeatureEngineeringAgent()

    # -- audits ------------------------------------------------------------

    def audit_recipe(self, recipe: FeatureRecipe) -> tuple:
        """``(ok, message)``. Catches a bad recipe before it costs a run."""
        report = self.auditor.recipe_audit(recipe)
        if report.get("status") == "PASS":
            return True, f"{len(report['selected_features'])} features selected"
        unknown = report.get("unknown_features") or []
        if unknown:
            return False, (f"unknown feature name(s) {unknown}. Choose only from "
                           f"the declared feature set.")
        return False, "the recipe selected no features at all"

    def audit_manifest(self) -> tuple:
        """Does the manifest still describe what the code produces?"""
        report = self.auditor.static_audit()
        if report.get("status") == "PASS":
            return True, f"{report['feature_count']} features declared and produced"
        return False, (f"manifest drift — missing: {report.get('missing_manifests')}, "
                       f"stale: {report.get('stale_manifests')}, "
                       f"unsafe: {report.get('unsafe_current_row_features')}")

    # -- proposing ---------------------------------------------------------

    def save_recipe(self, recipe: FeatureRecipe, workspace, iteration: int) -> str:
        """Write an audited recipe into the node's own workspace."""
        ok, why = self.audit_recipe(recipe)
        if not ok:
            raise ValueError(f"refusing to write a recipe that fails audit: {why}")
        root = getattr(workspace, "root", ".")
        path = os.path.join(root, "recipes",
                            f"iter_{iteration:02d}_{recipe.recipe_id}.json")
        recipe.save(path)
        return path

    def _build_prompt(self, ctx: ResearchContext, **kwargs) -> str:
        from pipeline.features import CausalStats
        tried = kwargs.get("tried_ids") or []
        return f"""Iteration {ctx.iteration} of {ctx.max_iterations}.

Best so far: {('%.4f' % ctx.best_score) if ctx.best_score is not None else 'nothing yet'} \
against a baseline of {ctx.baseline:.4f}.
{ctx.significance_note()}

Available features, by family:
  core     ({len(CausalStats.CORE_FEATURES)}): {', '.join(CausalStats.CORE_FEATURES)}
  affinity ({len(CausalStats.AFFINITY_FEATURES)}): {', '.join(CausalStats.AFFINITY_FEATURES)}
  temporal ({len(CausalStats.TEMPORAL_FEATURES)}): {', '.join(CausalStats.TEMPORAL_FEATURES)}

base_profile selects a family: 'core', 'affinity' (core+affinity), or 'full' (all).
Then refine with exclude_features, and tune the smoothing constants.

Recipes already tried this run (do not repeat): {', '.join(tried) or 'none'}

Experiments so far:
{ctx.history_table()}

Propose ONE recipe.

{{
  "name": "short_slug",
  "base_profile": "core | affinity | full",
  "exclude_features": ["names to drop, or empty"],
  "item_smoothing": 1.0 to 100.0,
  "cross_smoothing": 1.0 to 50.0,
  "global_prior_strength": 5.0 to 500.0,
  "completion_ratio_clip": 1.0 to 10.0,
  "recency_cap_days": 7 to 90,
  "rationale": "why this should change a WITHIN-USER ordering"
}}"""

    def _parse(self, payload: Any, ctx: ResearchContext, **kwargs) -> FeatureRecipe:
        proposal = validated(RecipeProposal, payload)
        recipe = proposal.to_recipe()
        ok, why = self.audit_recipe(recipe)
        if not ok:
            raise ValueError(why)
        self.last_rationale = proposal.rationale
        return recipe

    def fallback(self, ctx: ResearchContext, **kwargs) -> FeatureRecipe:
        """A hand-authored recipe that has not been run yet.

        These are the deterministic bounded mutations from the source branch:
        they span the profile axis and the smoothing axis, so a no-LLM run still
        covers the feature dimension properly rather than repeating one config.
        """
        tried = set(kwargs.get("tried_ids") or [])
        for recipe in deterministic_recipes():
            if recipe.recipe_id not in tried:
                self.last_rationale = (
                    f"deterministic bounded mutation: profile={recipe.base_profile}, "
                    f"item smoothing={recipe.item_smoothing}, "
                    f"cross smoothing={recipe.cross_smoothing}"
                    + (f", excluding {recipe.exclude_features}"
                       if recipe.exclude_features else ""))
                return recipe
        # Exhausted: vary smoothing so the next trial is at least a new point.
        base = deterministic_recipes()[0]
        widened = base.model_copy(update={
            "name": f"widened_{ctx.iteration}",
            "item_smoothing": min(100.0, base.item_smoothing * (1 + 0.25 * ctx.iteration)),
        })
        self.last_rationale = "recipe playbook exhausted; widening item smoothing"
        return widened
