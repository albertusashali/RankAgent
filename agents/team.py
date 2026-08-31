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

import os
import sys

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base import LLMClient
from agents.context import (ResearchContext, TrialRecord, command_signature,
                            parse_flags)
from agents.engineer import EngineerAgent, TrialSpec
from agents.feature_steward import FeatureStewardAgent
from agents.product_manager import Directive, ProductManagerAgent
from agents.qa import QAAgent
from agents.researcher import Hypothesis, ResearchAgent
from orchestrator.schemas import TokenUsage

PY = sys.executable


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


#: Models with their own trainers, which never consult ``--loss``. Testing a new
#: objective on one of these silently trains the stock model instead.
LOSS_IGNORING_MODELS = {"fm", "lgb"}


def _set_flag(args: str, flag: str, value: str) -> str:
    """Set ``--flag value`` in an argument string, in either spelling."""
    import re
    if re.search(rf"--{flag}[=\s]", args):
        return re.sub(rf"--{flag}(=|\s+)\S+", f"--{flag} {value}", args, count=1)
    return f"{args} --{flag} {value}".strip()


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
        self.steward = FeatureStewardAgent(self.llm, verbose=verbose)
        #: recipe_id -> iteration, so a recipe is never run twice. Recipes are
        #: content-hashed, so this is exact rather than a string comparison.
        self.tried_recipes: Dict[str, int] = {}

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

    def plan(self, ctx: ResearchContext, workspace=None) -> IterationPlan:
        """Run PM -> Researcher -> Engineer -> QA pre-flight for one iteration.

        When ``workspace`` is given, the Engineer may additionally *write code*
        into it. A hypothesis that names a target file becomes a source patch;
        one that does not stays a configuration-only trial. Both are legitimate
        experiments, and keeping the config path alive matters: it is the
        fallback when code generation fails, and it is what guarantees the run
        still has a submittable result on its worst day.
        """
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

        # 2a. Feature Steward — owns the feature dimension.
        #
        # When the PM points at `features`, a recipe is the better instrument
        # than generated code: feature engineering is where leakage happens, and
        # a recipe cannot leak by construction because every feature it can
        # select is already manifest-declared and mutation-tested. Generated
        # feature code is still available — the Researcher proposes it and the
        # Steward's audits gate it — but it is not the first thing tried.
        if directive and "features" in (directive.focus_dimensions or []):
            spec = self._propose_recipe(ctx, workspace, trace)
            if spec is not None:
                verdict = self.qa.preflight(ctx, spec, workspace=workspace)
                if verdict.ok:
                    self._say("    [QA]         pre-flight passed")
                    trace.append("QA pre-flight passed")
                    return IterationPlan(spec=spec, directive=directive,
                                         hypotheses=[], trace=trace,
                                         pm_refreshed=refreshed)
                for p in verdict.problems:
                    self._say(f"    [QA]         BLOCKED: {p}")
                trace.extend(f"QA blocked the recipe: {p}" for p in verdict.problems)

        # 2b. Researcher — hypotheses.
        hypotheses = self.researcher.run(ctx)
        self._say(f"    [RESEARCH]   {len(hypotheses)} hypothesis candidate(s) in "
                  f"{', '.join(sorted({h.dimension for h in hypotheses}))}")
        trace.append(f"Researcher proposed {len(hypotheses)}")

        # 3. Engineer — write the code first, then a runnable command.
        #
        # Ordering matters and was wrong at first. Argument validation used to
        # run before code generation, so a hypothesis proposing `--loss
        # neuralndcg` alongside a patch that registers `neuralndcg` was rejected
        # as an unknown loss — the agent's own work was refused because it had
        # not been done yet. Any hypothesis that asks for new code gets the code
        # written first; its arguments are then validated against the patched
        # source, where the new name exists.
        spec = None
        runnable = hypotheses
        if workspace is not None:
            spec, unwritable = self._implement_first(ctx, workspace, hypotheses, trace)
            # A hypothesis whose code could not be written is not testable by
            # running the old code under a new flag. Letting it through produced
            # exactly the drift this project exists to prevent: a failed DCN-v2
            # patch fell back to `--model deepfm`, and the run log recorded
            # "Implement DCN-v2" beside a result that was plain DeepFM.
            runnable = [h for h in hypotheses if h not in unwritable]

        if spec is None:
            if not runnable:
                trace.append("every hypothesis required code that could not be "
                             "written; none is testable by configuration alone")
                self._say("    [ENGINEER]   no hypothesis is testable without "
                          "the code that failed to generate")
            spec = self.engineer.build(ctx, runnable, data_dir=self.data_dir)
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
        verdict = self.qa.preflight(ctx, spec, workspace=workspace)
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

    def _propose_recipe(self, ctx: ResearchContext, workspace,
                        trace: List[str]) -> Optional[TrialSpec]:
        """Ask the Steward for a feature recipe and bind it to a command."""
        if workspace is None:
            return None

        ok, why = self.steward.audit_manifest()
        if not ok:
            self._say(f"    [STEWARD]    manifest audit failed: {why}")
            trace.append(f"feature manifest audit failed: {why}")
            return None

        try:
            recipe = self.steward.run(ctx, tried_ids=list(self.tried_recipes))
        except Exception as exc:
            trace.append(f"feature steward produced nothing: {exc}")
            return None

        if recipe.recipe_id in self.tried_recipes:
            self._say(f"    [STEWARD]    recipe {recipe.recipe_id} already run at "
                      f"iteration {self.tried_recipes[recipe.recipe_id]}")
            trace.append(f"recipe {recipe.recipe_id} is a duplicate")
            return None

        try:
            path = self.steward.save_recipe(recipe, workspace, ctx.iteration)
        except ValueError as exc:
            self._say(f"    [STEWARD]    {exc}")
            trace.append(str(exc))
            return None

        self.tried_recipes[recipe.recipe_id] = ctx.iteration
        rationale = getattr(self.steward, "last_rationale", "")
        self._say(f"    [STEWARD]    recipe {recipe.recipe_id} '{recipe.name}' "
                  f"(profile={recipe.base_profile}, item_sm={recipe.item_smoothing}, "
                  f"cross_sm={recipe.cross_smoothing})")
        trace.append(f"feature recipe {recipe.recipe_id}: {rationale}")

        # The recipe only reaches a model that consumes dense features.
        args = (f"--model lgb --feature_recipe {os.path.relpath(path, workspace.root)} "
                f"--objective lambdarank")
        return TrialSpec(
            dimension="features",
            hypothesis=f"Feature recipe '{recipe.name}': {rationale}",
            mechanism=rationale,
            args=args,
            command=f"{PY} -m pipeline.train {args}"
                    + (f" --data_dir {self.data_dir}" if self.data_dir else ""),
            checkpoint="lgb",
            # A recipe is neither a flag toggle nor a code patch: its ranges are
            # enforced, its identity is a hash of its behaviour, and re-running
            # the id reproduces the features exactly. Logging it as either of
            # the other two would misdescribe it.
            source="recipe",
            recipe_id=recipe.recipe_id,
            recipe=recipe.model_dump(),
        )

    def _implement_first(self, ctx: ResearchContext, workspace,
                         hypotheses, trace: List[str]):
        """Write code for the first hypothesis that asks for it.

        Returns ``(spec, unwritable)`` — a ``TrialSpec`` carrying the patch, or
        ``None`` to fall through to a configuration-only trial, plus the
        hypotheses whose code could not be written and which must therefore not
        be run as configuration-only experiments.

        Falling through is not a failure mode to be avoided: the config path is
        what keeps the run producing results on the day code generation does not
        land. It just must not misattribute what was tested.
        """
        unwritable: List[Any] = []
        from agents.codegen import registered_losses, registered_models
        from agents.engineer import validate_args

        for h in hypotheses:
            if not (h.target_file or h.edit_sketch):
                continue
            if ctx.is_duplicate(h.args):
                continue

            losses_before = registered_losses(workspace.read("pipeline/models.py"))

            impl, problems = self.engineer.implement(ctx, workspace, h)
            if impl is None:
                for p in problems[:1]:
                    self._say(f"    [ENGINEER]   patch failed: "
                              f"{p.splitlines()[0][:110]}")
                trace.append(f"code generation failed for '{h.hypothesis[:50]}' "
                             f"after {self.engineer.code_rounds} round(s)")
                unwritable.append(h)
                continue

            known_losses = registered_losses(workspace.read("pipeline/models.py"))
            args = h.args

            # A patch that registers a new objective the command never selects
            # is untested code. It happened on the first real code-generating
            # run: the Engineer implemented ApproxNDCG and the trial then ran
            # `--loss listwise`, so the log showed a diff beside a result that
            # owed nothing to it. Point the command at the new objective, and
            # say so — silently running the old one is the worse outcome.
            new_losses = known_losses - losses_before
            if new_losses:
                picked = sorted(new_losses)[0]
                selected = parse_flags(args).get("loss")
                if selected not in new_losses:
                    args = _set_flag(args, "loss", picked)
                    self._say(f"    [ENGINEER]   the patch registered "
                              f"'{picked}' but the command selected "
                              f"'{selected}'; running the new objective")
                    trace.append(f"redirected --loss to the newly implemented "
                                 f"'{picked}' (was '{selected}')")

                # `fm` (numpy) and `lgb` (LightGBM) have their own trainers and
                # ignore --loss entirely, so pairing a new objective with either
                # trains the stock baseline and reports it as the new method's
                # result. A real run did exactly that: it implemented focal loss,
                # ran `--model=fm --loss=focal`, and scored 0.6015 — the baseline
                # to four decimals, because the loss was never called.
                model = parse_flags(args).get("model")
                if model in LOSS_IGNORING_MODELS:
                    args = _set_flag(args, "model", "fm_torch")
                    self._say(f"    [ENGINEER]   '{model}' has its own trainer "
                              f"and ignores --loss; running '{picked}' on "
                              f"fm_torch so the new objective is actually used")
                    trace.append(f"redirected --model from '{model}' to fm_torch: "
                                 f"'{model}' ignores --loss, so the new objective "
                                 f"would not have been exercised")

            # Validate against the source as it now stands, so a name this very
            # patch registered is recognised.
            ok, why = validate_args(
                args,
                known_models=registered_models(workspace.read("pipeline/train.py"),
                                               workspace.read("pipeline/models.py")),
                known_losses=known_losses)
            if not ok:
                self._say(f"    [ENGINEER]   patch applied but its arguments do "
                          f"not run: {why}")
                trace.append(f"patch applied but arguments invalid: {why}")
                unwritable.append(h)
                continue

            spec = self.engineer._spec(h, args, [], self.data_dir)
            spec.implementation = impl
            self._say(f"    [ENGINEER]   patched {', '.join(impl.target_files)} "
                      f"(+{impl.lines_added}/-{impl.lines_removed}, "
                      f"{impl.rounds} round(s))")
            trace.append(f"patched {', '.join(impl.target_files)} "
                         f"+{impl.lines_added}/-{impl.lines_removed}")
            return spec, unwritable
        return None, unwritable

    # -- after execution --------------------------------------------------

    def review(self, primary: Optional[float]):
        """QA's verdict on a completed trial."""
        verdict = self.qa.judge(primary)
        if verdict.note:
            self._say(f"    [QA]         {verdict.note}")
        return verdict

    def repair_code(self, ctx: ResearchContext, workspace, spec: TrialSpec,
                    traceback_text: str):
        """Ask the Engineer to fix code it just wrote, given the traceback.

        This is deliberately separate from ``recover``. The self-healing
        debugger repairs *command lines* — halve the batch size, drop an unknown
        flag — and those heuristics are actively wrong for a bug in generated
        source: handed a shape mismatch inside a new loss function it would
        "fix" it by resetting ``--embed_dim 16``, which looks like a repair,
        changes nothing, and costs an iteration.
        """
        h = Hypothesis(
            dimension=spec.dimension, hypothesis=spec.hypothesis,
            mechanism=spec.mechanism, args=spec.args, source=spec.source,
            target_file=(spec.implementation.target_files[0]
                         if spec.implementation and spec.implementation.target_files
                         else ""),
            edit_sketch="Fix the failure in the code you just wrote. Change as "
                        "little as possible and keep the mechanism intact.")
        # Three rounds, not one. With a single round a malformed reply — the
        # wrong output format, an anchor that did not match — ended the repair
        # immediately, with no chance to act on the feedback the parser had just
        # produced. Four of five iterations in one run died that way.
        impl, problems = self.engineer.implement(
            ctx, workspace, h, max_rounds=3, traceback_text=traceback_text)
        if impl is None:
            for p in problems[:1]:
                self._say(f"    [ENGINEER]   repair failed: {p.splitlines()[0][:110]}")
            return None
        self._say(f"    [ENGINEER]   repaired {', '.join(impl.target_files)} "
                  f"(+{impl.lines_added}/-{impl.lines_removed})")
        return impl

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
