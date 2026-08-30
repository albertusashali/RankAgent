"""QA & Debugger agent — the only role that was already mostly built.

``sandbox.debugger.SelfHealingDebugger`` already classifies a failed trial, applies
heuristic repairs, re-runs, and records every attempt. Rebuilding that as a fresh
"QA agent" would have been duplication for the sake of an org chart. This class
wraps it instead, and adds the two things a reviewer role should own that the
debugger did not:

  * **Pre-flight checks.** Cheap assertions run *before* a trial is executed, so an
    obviously doomed experiment costs nothing instead of costing a training run.
  * **A verdict on results.** A trial that "succeeds" can still be junk — a score
    below the random-scoring floor, or a suspiciously large jump, means something
    is wrong with the harness rather than right with the model. Judging that is a
    QA responsibility, not the Researcher's.

The LLM repair path is handed to the wrapped debugger, so an unfixable failure
prunes the branch and the loop continues.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from pydantic import BaseModel

from agents.base import Agent
from agents.context import ResearchContext, parse_flags
from agents.engineer import TrialSpec, spec_summary, validate_args
from sandbox.debugger import SelfHealingDebugger

#: Reference rungs from the starter kit, on validation. A model that lands below
#: item-popularity is not a weak result, it is a broken pipeline.
RANDOM_FLOOR = 0.4834
POPULARITY_RUNG = 0.5807
ORACLE_CEILING = 0.8484


class PreflightVerdict(BaseModel):
    ok: bool
    problems: List[str] = []


class ResultVerdict(BaseModel):
    trustworthy: bool
    note: str = ""


class QAAgent(Agent):
    name = "qa"
    max_tokens = 500

    def __init__(self, llm=None, max_retries: int = 3, verbose: bool = True):
        super().__init__(llm, verbose)
        self.debugger = SelfHealingDebugger(max_retries=max_retries,
                                            llm_repair=self._llm_repair)

    # -- before the trial -------------------------------------------------

    def preflight(self, ctx: ResearchContext, spec: TrialSpec) -> PreflightVerdict:
        """Cheap checks that cost nothing next to a training run."""
        problems: List[str] = []

        ok, why = validate_args(spec.args)
        if not ok:
            problems.append(f"arguments do not parse: {why}")

        if ctx.is_duplicate(spec.args):
            problems.append("this configuration has already been run in this session")

        if spec.checkpoint is None:
            problems.append("no --model, so no checkpoint would be produced")

        # The hypothesis and the command must plausibly agree. This is a shallow
        # check, but it catches the drift mode that produced a log claiming DCN-v2
        # while running MMoE.
        if not self._mentions_the_model(spec):
            problems.append("hypothesis does not mention the model the command runs")

        return PreflightVerdict(ok=not problems, problems=problems)

    #: Words that plausibly describe each model.
    MODEL_ALIASES = {
        "fm": ("factorization", "factorisation", " fm", "baseline"),
        "fm_torch": ("factorization", "factorisation", " fm", "listwise", "pairwise",
                     "bpr", "pointwise", "loss"),
        "deepfm": ("deepfm", "deep fm", "mlp", "cross"),
        "din": ("din", "attention", "sequence", "sequential", "history", "interest"),
        "mmoe": ("mmoe", "mixture", "expert", "multi-task", "multi task", "auxiliary"),
        "lgb": ("lightgbm", "gbdt", "tree", "lambdarank", "boost"),
    }

    #: Words that plausibly describe each objective.
    LOSS_ALIASES = {
        "listwise": ("listwise", "list-wise", "softmax", "list wise"),
        "bpr": ("bpr", "pairwise", "pair-wise", "bayesian personalised",
                "bayesian personalized", "margin"),
        "pointwise": ("pointwise", "point-wise", "bce", "cross-entropy", "binary"),
    }

    @classmethod
    def _mentions_the_model(cls, spec: TrialSpec) -> bool:
        """Does the prose describe what this trial actually varies?

        The check has to be dimension-aware. A hypothesis in the ``loss``
        dimension is about the objective, and the architecture it happens to run
        on is incidental — demanding it name the model rejects a perfectly
        coherent experiment. A real run lost an iteration to exactly that: the
        Researcher proposed a BPR experiment on DeepFM, and a model-only check
        blocked it for not saying "DeepFM".

        So: for loss-dimension trials the objective must be named; otherwise the
        model must be. Either satisfies the check when the dimension is unclear.
        """
        text = f"{spec.hypothesis} {spec.mechanism}".lower()
        flags = parse_flags(spec.args)
        model, loss = flags.get("model"), flags.get("loss")

        names_model = model is None or any(a in text for a in cls.MODEL_ALIASES.get(model, ()))
        names_loss = loss is None or any(a in text for a in cls.LOSS_ALIASES.get(loss, ()))

        if spec.dimension == "loss":
            return names_loss
        if spec.dimension in ("architecture", "sequence", "multi_task"):
            return names_model
        # Dimensions such as capacity or optimisation vary a knob rather than the
        # model or the objective, so naming either is enough to show coherence.
        return names_model or names_loss

    # -- after the trial --------------------------------------------------

    def judge(self, primary: Optional[float]) -> ResultVerdict:
        """Is this number believable?"""
        if primary is None:
            return ResultVerdict(trustworthy=False, note="no metrics produced")
        if primary < RANDOM_FLOOR:
            return ResultVerdict(
                trustworthy=False,
                note=(f"primary {primary:.4f} is below the random-scoring floor "
                      f"({RANDOM_FLOOR}); the pipeline is broken, not the model"))
        if primary > ORACLE_CEILING:
            return ResultVerdict(
                trustworthy=False,
                note=(f"primary {primary:.4f} exceeds the oracle ceiling "
                      f"({ORACLE_CEILING}); this is only possible with label leakage"))
        if primary < POPULARITY_RUNG:
            return ResultVerdict(
                trustworthy=True,
                note=(f"primary {primary:.4f} is below the item-popularity rung "
                      f"({POPULARITY_RUNG}) — a real but poor result"))
        return ResultVerdict(trustworthy=True)

    # -- failure handling -------------------------------------------------

    def recover(self, command: str, traceback_text: str,
                run: Callable[[str], Any]):
        """Delegate to the self-healing debugger and return its outcome."""
        return self.debugger.attempt_repair(command, traceback_text, run)

    def _llm_repair(self, command: str, traceback_text: str, kind: str) -> Optional[str]:
        if self.llm is None or not self.llm.available:
            return None
        try:
            text = self.llm.complete(
                self.name,
                "You repair failed machine-learning trial commands. Reply with only the "
                "corrected command line, or the single word UNFIXABLE.",
                f"""The command failed.

Command:
  {command}

Failure kind: {kind}

Traceback (tail):
{(traceback_text or '')[-2500:]}

Valid argument spec:
{spec_summary()}

Change as little as possible. Reply with only the corrected command line.""",
                max_tokens=self.max_tokens)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        for line in (text or "").splitlines():
            if "pipeline.train" in line:
                fixed = line.strip().strip("`").strip()
                if fixed.startswith("python "):
                    import sys
                    fixed = f"{sys.executable} " + fixed[len("python "):]
                return fixed
        return None

    # Entry points are preflight/judge/recover, not the single-call base flow.
    def _build_prompt(self, ctx, **kwargs) -> str:                # pragma: no cover
        raise NotImplementedError("QAAgent uses preflight/judge/recover")

    def _parse(self, payload: Any, ctx, **kwargs):                # pragma: no cover
        raise NotImplementedError("QAAgent uses preflight/judge/recover")

    def fallback(self, ctx, **kwargs):                            # pragma: no cover
        raise NotImplementedError("QAAgent uses preflight/judge/recover")
