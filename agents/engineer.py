"""Software Engineer agent — turns a hypothesis into something that actually runs.

This role exists to make one guarantee: **the experiment that runs is the
experiment that was proposed.** An earlier single-agent loop generated prose and a
command in one breath and then let a fallback silently swap the command, producing
a run log that claimed DCN-v2 while executing ``--model mmoe``. Splitting the roles
does not fix that by itself — a chain of agents has more places to drift, not
fewer. What fixes it is validation at the boundary:

  * every candidate command is parsed against the **real** ``pipeline.train``
    argparse spec before it is allowed to run, so an invented flag or an
    out-of-range choice is caught here rather than becoming a failed trial;
  * every candidate is checked against the run's history, so a configuration that
    has already been executed is rejected instead of burning an iteration — the
    exact waste observed at iteration 4 of the 2026-08-29 run;
  * the returned spec carries the hypothesis it came from, so the logger records a
    matched pair rather than two independently-authored strings.

If the leading hypothesis fails validation, the Engineer works down the
Researcher's ranked alternatives before giving up, which is why the Researcher
proposes several at a time.
"""
from __future__ import annotations

import contextlib
import io
import shlex
import sys
from typing import Any, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from agents.base import Agent, validated
from agents.context import ResearchContext, command_signature, parse_flags
from agents.researcher import Hypothesis

PY = sys.executable


class Implementation(BaseModel):
    """A code change actually written to a workspace, with its measured diff."""
    target_files: List[str]
    diff: str
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    rounds: int = 1
    summary: str = ""

    @property
    def is_empty(self) -> bool:
        return self.files_changed == 0


class TrialSpec(BaseModel):
    """A runnable experiment, bound to the hypothesis that motivated it."""
    dimension: str
    hypothesis: str
    mechanism: str = ""
    args: str
    command: str
    checkpoint: Optional[str] = None
    #: Who authored the hypothesis this spec implements: "llm" or "playbook".
    #: NO DEFAULT, deliberately. It used to default to "llm", and because the
    #: only construction site never passed it, every iteration in every archived
    #: run logged `proposal_source: "llm"` — including runs with no API key at
    #: all, where the hypothesis came verbatim from a hard-coded playbook. These
    #: logs are a deliverable, so the field has to be impossible to leave unset.
    source: str
    implementation: Optional[Implementation] = None
    rejected: List[str] = Field(default_factory=list)


class RepairedArgs(BaseModel):
    args: str
    explanation: str = ""


SYSTEM = """You are the Software Engineer on an autonomous ML research team. You are
given a research hypothesis and a set of `python -m pipeline.train` arguments that
failed validation against the real command-line spec. Your job is to correct the
arguments so they run, while still testing the hypothesis as stated.

Rules:
- Change as little as possible. Do not redesign the experiment.
- Use only flags and values that exist in the spec you are shown.
- If the hypothesis cannot be expressed with the available flags, say so in the
  explanation and return the closest valid approximation.

Reply with a single JSON object and nothing else."""


def validate_args(args: str,
                  known_models: Optional[Iterable[str]] = None,
                  known_losses: Optional[Iterable[str]] = None) -> Tuple[bool, str]:
    """Parse ``args`` against pipeline.train's actual parser, then check names.

    Returns ``(ok, message)``. Importing the trainer is cheap: it pulls in numpy
    but neither torch nor LightGBM, which are imported lazily inside the trainers.

    ``--model`` and ``--loss`` deliberately no longer carry argparse ``choices=``,
    because a hard-coded choice list is a duplicate of the code's own registry and
    would make a model or objective the agent *writes* unreachable from the CLI.
    The whitelist therefore moves here, where it can be parameterised: pass
    ``known_models`` / ``known_losses`` discovered from the node's own source to
    accept newly registered names, or leave them ``None`` to accept only what the
    harness ships. Either way an invented name is still caught before a trial
    runs — the check moved, it did not disappear.
    """
    try:
        from pipeline.train import ARCHS, LOSSES_DOC, build_parser
    except Exception as exc:                       # pragma: no cover - import guard
        return False, f"could not load the trainer's argument spec: {exc}"

    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return False, f"arguments are not shell-parseable: {exc}"

    parser = build_parser()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            parsed, extra = parser.parse_known_args(tokens)
    except SystemExit:
        message = err.getvalue().strip().splitlines()
        return False, (message[-1] if message else "arguments rejected by the parser")
    if extra:
        return False, f"unrecognised arguments: {' '.join(extra)}"
    if not getattr(parsed, "model", None):
        return False, "no --model supplied"

    models = set(known_models) if known_models is not None else set(ARCHS)
    if parsed.model not in models:
        return False, (f"unknown model {parsed.model!r}; registered: {sorted(models)}. "
                       f"Implement and register it in the code before selecting it.")

    losses = set(known_losses) if known_losses is not None else set(LOSSES_DOC)
    if getattr(parsed, "loss", None) and parsed.loss not in losses:
        return False, (f"unknown loss {parsed.loss!r}; registered: {sorted(losses)}. "
                       f"Add it to pipeline/models.py LOSSES before selecting it.")
    return True, "ok"


def spec_summary() -> str:
    """The trainer's own help text — the single source of truth for flags."""
    try:
        from pipeline.train import build_parser
        return build_parser().format_help()
    except Exception:
        return "(argument spec unavailable)"


def checkpoint_name(args: str) -> Optional[str]:
    """Mirror the trainer's checkpoint naming so the submission step can find it."""
    flags = parse_flags(args)
    model = flags.get("model")
    if model is None:
        return None
    if model in ("fm", "mmoe", "lgb"):
        return model
    return f"{model}_{flags.get('loss', 'listwise')}"


#: Where a change in each research dimension most naturally lands, used when the
#: Researcher does not name a target file.
_DEFAULT_TARGET = {
    "loss": "pipeline/models.py",
    "architecture": "pipeline/models.py",
    "sequence": "pipeline/features.py",
    "features": "pipeline/features.py",
    "multi_task": "pipeline/train.py",
    "optimisation": "pipeline/train.py",
    "capacity": "pipeline/models.py",
}


class EngineerAgent(Agent):
    name = "engineer"
    system_prompt = SYSTEM
    max_tokens = 500
    #: Patches are larger than repaired flag strings and truncation here means a
    #: wasted round trip, so the code path gets its own, roomier ceiling. It has
    #: to fit a whole-file rewrite of models.py (~400 lines), which is the
    #: fallback when the model cannot produce a reliable anchor.
    code_max_tokens = 12000
    #: Rounds the last `implement` call used, for the run log.
    code_rounds = 0

    def build(self, ctx: ResearchContext, hypotheses: List[Hypothesis],
              data_dir: Optional[str] = None) -> Optional[TrialSpec]:
        """First hypothesis that validates and is not a repeat.

        Returns ``None`` only when every alternative is exhausted, which the team
        treats as "ask the Researcher again" rather than as an error.
        """
        rejected: List[str] = []
        duplicates: List[Tuple[Hypothesis, str]] = []

        # Pass 1 — a genuinely new configuration. Exploring an untried config
        # always beats re-running an old one, so every alternative is considered
        # before falling back to replication below.
        for h in hypotheses:
            args, ok = self._validated_args(ctx, h, rejected)
            if not ok:
                continue
            if ctx.is_duplicate(args):
                duplicates.append((h, args))
                continue
            return self._spec(h, args, rejected, data_dir)

        # Pass 2 — everything the Researcher offered has already been run. Re-run
        # the best of them under a fresh seed: that measures the noise floor,
        # which this benchmark needs given the whole improvement so far is roughly
        # 3 sigma. Re-running under the *same* seed would be pure waste.
        for h, args in duplicates:
            varied = self._differentiate(ctx, args)
            if varied is None:
                rejected.append(f"`{command_signature(args)}` already run")
                continue
            rejected.append(f"`{command_signature(args)}` already run — repeating "
                            f"under a new seed to measure variance")
            return self._spec(h, varied, rejected, data_dir)

        return None

    # -- code generation --------------------------------------------------

    def implement(self, ctx: ResearchContext, ws, h: Hypothesis,
                  max_rounds: int = 3,
                  traceback_text: str = "") -> Tuple[Optional[Implementation], List[str]]:
        """Write ``h`` into ``ws`` as a source patch.

        Returns ``(implementation, problems)``. On failure the implementation is
        ``None`` and ``problems`` explains why in terms the model can act on —
        the caller either retries with a different hypothesis or prunes.

        Nothing is written to the workspace until a full round of edits has
        applied cleanly and every touched file still parses, so a failed attempt
        leaves the node's code exactly as its parent left it.
        """
        from agents.codegen import SYSTEM as CODE_SYSTEM, build_prompt
        from agents.patch import apply_all, parse_edit_blocks
        from sandbox.workspace import MUTABLE

        if self.llm is None or not self.llm.available:
            return None, ["no LLM available to write code"]

        targets = [h.target_file] if h.target_file in MUTABLE else []
        if not targets:
            # The Researcher did not name a file, or named one it may not touch.
            # Infer from the dimension rather than refusing outright.
            targets = [_DEFAULT_TARGET.get(h.dimension, "pipeline/models.py")]

        before = ws.snapshot()
        problems: List[str] = []
        self.code_rounds = 0

        for attempt in range(1, max_rounds + 1):
            self.code_rounds = attempt
            try:
                reply = self.llm.complete(
                    self.name, CODE_SYSTEM,
                    build_prompt(h, targets, before, lineage=ctx.lineage,
                                 problems=problems or None,
                                 traceback_text=traceback_text),
                    max_tokens=self.code_max_tokens, json_mode=False)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return None, [f"the model call failed: {self.last_error}"]

            edits, errors = parse_edit_blocks(reply, MUTABLE)

            if edits:
                after, applied_problems = apply_all(before, edits)
                if applied_problems:
                    problems = errors + applied_problems
                    continue
            else:
                # No anchors. Very often the model has returned the whole
                # corrected file instead, which is a perfectly good answer we
                # were previously discarding as malformed — it was the largest
                # single cause of failed repairs.
                from agents.patch import parse_whole_file
                rewritten, whole_problems = parse_whole_file(
                    reply, targets, {p: before[p] for p in targets if p in before})
                if whole_problems or not rewritten:
                    problems = errors + whole_problems
                    continue
                after = {**before, **rewritten}
                self._say(f"    [ENGINEER]   accepted a whole-file rewrite of "
                          f"{', '.join(rewritten)}")

            if all(before.get(rel) == text for rel, text in after.items()):
                problems = ["your edits produced no change to any file."]
                continue

            for rel, text in after.items():
                if before.get(rel) != text:
                    ws.write(rel, text)

            # Diff against what this node INHERITED, not against the state the
            # model was shown. They differ on a repair: the model sees the code
            # it just broke, but the run log must show one change against the
            # parent, not the repair in isolation.
            from sandbox.workspace import unified_diff
            base = ws.base or before
            diff, (files, added, removed) = unified_diff(base, after)

            return Implementation(
                target_files=sorted(p for p in after if base.get(p) != after[p]),
                diff=diff, files_changed=files, lines_added=added,
                lines_removed=removed, rounds=attempt,
                summary=f"{h.hypothesis[:80]}"), []

        return None, problems

    def _validated_args(self, ctx: ResearchContext, h: Hypothesis,
                        rejected: List[str]) -> Tuple[str, bool]:
        """Validate a hypothesis's arguments, repairing once if the LLM can."""
        args = h.args.strip()
        ok, why = validate_args(args)
        if ok:
            return args, True

        repaired = self._repair(ctx, h, why)
        if not repaired:
            rejected.append(f"`{args}` invalid ({why})")
            return args, False

        ok2, why2 = validate_args(repaired)
        if not ok2:
            rejected.append(f"`{args}` invalid ({why}); repair also invalid ({why2})")
            return args, False

        rejected.append(f"repaired `{args}` ({why})")
        return repaired, True

    @staticmethod
    def _spec(h: Hypothesis, args: str, rejected: List[str],
              data_dir: Optional[str]) -> TrialSpec:
        command = f"{PY} -m pipeline.train {args}"
        if data_dir and "--data_dir" not in args:
            command = f"{command} --data_dir {data_dir}"
        return TrialSpec(dimension=h.dimension, hypothesis=h.hypothesis,
                         mechanism=h.mechanism, args=args, command=command,
                         checkpoint=checkpoint_name(args), source=h.source,
                         rejected=list(rejected))

    @staticmethod
    def _differentiate(ctx: ResearchContext, args: str) -> Optional[str]:
        """Vary the seed to turn a repeat into a genuine replicate.

        Re-running a configuration under a new seed is a legitimate experiment —
        it measures the noise floor, which this benchmark badly needs given that
        the whole improvement so far is roughly 3 sigma. Re-running it under the
        *same* seed is pure waste.
        """
        if "--seed" in args:
            return None
        for seed in range(1, 6):
            candidate = f"{args} --seed {seed}"
            if not ctx.is_duplicate(candidate):
                return candidate
        return None

    def _repair(self, ctx: ResearchContext, h: Hypothesis, why: str) -> Optional[str]:
        """Ask the LLM to fix invalid arguments; None if unavailable or unusable."""
        if self.llm is None or not self.llm.available:
            return None
        try:
            text = self.llm.complete(
                self.name, self.system_prompt,
                f"""Hypothesis: {h.hypothesis}
Mechanism: {h.mechanism}

Proposed arguments (INVALID): {h.args}
Validator said: {why}

The real argument spec:
{spec_summary()}

{{"args": "--model ... corrected arguments only, no `python -m pipeline.train` prefix",
  "explanation": "what you changed"}}""",
                max_tokens=self.max_tokens)
            from agents.base import extract_json
            return validated(RepairedArgs, extract_json(text)).args.strip()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    # The base-class hooks are unused: this agent's entry point is `build`,
    # which orchestrates validation rather than making a single LLM call.
    def _build_prompt(self, ctx, **kwargs) -> str:                # pragma: no cover
        raise NotImplementedError("EngineerAgent.build is the entry point")

    def _parse(self, payload: Any, ctx, **kwargs):                # pragma: no cover
        raise NotImplementedError("EngineerAgent.build is the entry point")

    def fallback(self, ctx, **kwargs):                            # pragma: no cover
        raise NotImplementedError("EngineerAgent.build is the entry point")
