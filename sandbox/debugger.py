"""Self-healing debugger: classify a failed trial, repair it, re-run it.

The previous version defined only a prompt builder and a heuristic string-replace
helper. The orchestrator called ``attempt_repair()``, which did not exist, on an
instance constructed with the wrong argument — so the first failed trial raised
``AttributeError`` and killed the run. Robustness is graded on recovery from
failure, so that path is now the most carefully handled one in the codebase.

Strategy, cheapest first:

  1. **Classify** the traceback into a known failure mode.
  2. **Heuristic repair** — an argument-level fix that needs no LLM call
     (halve the batch on OOM, drop an unsupported flag, fall back to CPU).
  3. **LLM repair** — only if a repair callback was supplied and the heuristics
     do not apply.

Every attempt is recorded, so the run-log can show what broke and how it was
handled. A repair that fails is not an error: it returns a RepairOutcome saying
the branch should be pruned, and the loop continues with the next hypothesis.
"""
import re
import shlex
from dataclasses import dataclass, field
from typing import Callable, List, Optional

OOM = "OUT_OF_MEMORY"
UNKNOWN_ARG = "UNKNOWN_ARGUMENT"
SHAPE = "SHAPE_MISMATCH"
IMPORT = "MISSING_DEPENDENCY"
TIMEOUT = "TIMEOUT"
DATA = "MISSING_DATA"
OTHER = "UNCLASSIFIED"


@dataclass
class RepairAttempt:
    kind: str
    strategy: str
    detail: str
    command: Optional[str] = None


@dataclass
class RepairOutcome:
    repaired: bool
    command: Optional[str] = None
    kind: str = OTHER
    attempts: List[RepairAttempt] = field(default_factory=list)

    def as_log(self) -> dict:
        return {
            "recovered": self.repaired,
            "failure_kind": self.kind,
            "attempts": [a.__dict__ for a in self.attempts],
        }


def classify(traceback_text: str) -> str:
    """Map a traceback onto a failure mode we know how to act on."""
    t = (traceback_text or "").lower()
    if "out of memory" in t or "cuda oom" in t or "cannot allocate" in t:
        return OOM
    if "unrecognized arguments" in t or "invalid choice" in t or "no such option" in t:
        return UNKNOWN_ARG
    if ("size mismatch" in t or "shapes cannot be multiplied" in t
            or "dimension out of range" in t or "must match the size" in t):
        return SHAPE
    if "modulenotfounderror" in t or "importerror" in t:
        return IMPORT
    if "timed out" in t or "timeout" in t:
        return TIMEOUT
    if "filenotfounderror" in t and ("kuairand" in t or ".csv" in t):
        return DATA
    return OTHER


def _replace_int_flag(cmd: str, flag: str, transform: Callable[[int], int]) -> Optional[str]:
    parts = shlex.split(cmd)
    for i, tok in enumerate(parts):
        if tok == flag and i + 1 < len(parts):
            try:
                parts[i + 1] = str(transform(int(parts[i + 1])))
            except ValueError:
                return None
            return " ".join(shlex.quote(p) if " " in p else p for p in parts)
    return None


def _drop_unknown_flag(cmd: str, traceback_text: str) -> Optional[str]:
    m = re.search(r"unrecognized arguments:\s*(\S+)", traceback_text or "", re.I)
    if not m:
        m = re.search(r"argument\s+(--\S+?):", traceback_text or "", re.I)
    if not m:
        return None
    bad = m.group(1).split("=")[0]
    parts = shlex.split(cmd)
    out, skip = [], False
    for i, tok in enumerate(parts):
        if skip:
            skip = False
            continue
        if tok == bad or tok.startswith(bad + "="):
            # Drop the flag, plus its value if the next token is not another flag.
            if tok == bad and i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                skip = True
            continue
        out.append(tok)
    repaired = " ".join(out)
    return repaired if repaired != cmd else None


class SelfHealingDebugger:
    """Attempts bounded, auditable repairs on a failed trial command.

    ``llm_repair`` is an optional callable ``(command, traceback, kind) -> str|None``
    returning a corrected command. It is only consulted when the heuristics do not
    apply, which keeps token spend down.
    """

    def __init__(self, max_retries: int = 3,
                 llm_repair: Optional[Callable[[str, str, str], Optional[str]]] = None):
        self.max_retries = max_retries
        self.llm_repair = llm_repair

    def heuristic_repair(self, command: str, traceback_text: str,
                         kind: str, attempt: int) -> Optional[RepairAttempt]:
        if kind == OOM:
            for flag, name in (("--batch_size", "batch size"), ("--embed_dim", "embedding dim")):
                fixed = _replace_int_flag(command, flag, lambda v: max(256, v // 2))
                if fixed and fixed != command:
                    return RepairAttempt(kind, "halve_" + flag.lstrip("-"),
                                         f"halved {name} after an allocation failure", fixed)
            if "--batch_size" not in command:
                return RepairAttempt(kind, "add_small_batch",
                                     "no batch flag present; pinning a small batch",
                                     f"{command} --batch_size 2048")
        if kind == UNKNOWN_ARG:
            fixed = _drop_unknown_flag(command, traceback_text)
            if fixed:
                return RepairAttempt(kind, "drop_unsupported_flag",
                                     "removed a flag the trainer does not accept", fixed)
        if kind == SHAPE:
            # Shape errors in these models almost always come from an embedding
            # size the checkpoint or head does not expect; retreat to the default.
            fixed = _replace_int_flag(command, "--embed_dim", lambda _v: 16)
            if fixed and fixed != command:
                return RepairAttempt(kind, "reset_embed_dim",
                                     "reset embedding dim to the known-good default", fixed)
        if kind == TIMEOUT:
            fixed = _replace_int_flag(command, "--epochs", lambda v: max(3, v // 2))
            if fixed and fixed != command:
                return RepairAttempt(kind, "halve_epochs",
                                     "halved the epoch budget after a timeout", fixed)
        return None

    def attempt_repair(self, command: str, traceback_text: str,
                       run: Callable[[str], object]) -> RepairOutcome:
        """Try up to ``max_retries`` repairs, re-running after each.

        ``run(command)`` must return an object with a ``.status`` attribute, as
        ``ExecutionRunner.run_command`` does. Returns a ``RepairOutcome`` whose
        ``attempts`` list is written verbatim into the iteration log.
        """
        kind = classify(traceback_text)
        outcome = RepairOutcome(repaired=False, kind=kind)
        current, current_tb = command, traceback_text

        for attempt in range(1, self.max_retries + 1):
            candidate = self.heuristic_repair(current, current_tb, kind, attempt)

            if candidate is None and self.llm_repair is not None:
                try:
                    proposed = self.llm_repair(current, current_tb, kind)
                except Exception as exc:                       # never let a repair crash the run
                    outcome.attempts.append(
                        RepairAttempt(kind, "llm_repair", f"repair call failed: {exc}"))
                    break
                if proposed and proposed != current:
                    candidate = RepairAttempt(kind, "llm_repair",
                                              "LLM proposed a corrected command", proposed)

            if candidate is None or not candidate.command:
                outcome.attempts.append(
                    RepairAttempt(kind, "give_up",
                                  "no applicable repair for this failure mode"))
                break

            outcome.attempts.append(candidate)
            try:
                result = run(candidate.command)
            except Exception as exc:
                current_tb = str(exc)
                current = candidate.command
                continue

            if getattr(result, "status", None) == "SUCCESS":
                outcome.repaired = True
                outcome.command = candidate.command
                outcome.result = result           # type: ignore[attr-defined]
                return outcome

            current = candidate.command
            current_tb = getattr(result, "error_traceback", "") or ""
            kind = classify(current_tb)

        return outcome
