"""The blackboard: shared state every agent reads and writes.

WHY A BLACKBOARD RATHER THAN A PIPELINE
---------------------------------------
The obvious way to wire specialised agents is a chain — PM feeds Research, which
feeds SWE, which feeds QA. That design has a failure mode this project has already
been bitten by: each handoff is a lossy paraphrase, so the PM says "focus on
cold-start", Research invents a cold-start hypothesis, SWE emits
``--model din --embed_dim 64``, and the link to cold-start is fiction. An earlier
run logged exactly that shape of drift, claiming DCN-v2 while running ``--model mmoe``.

Here every agent reads the same structured record and writes structured objects
back to it. Nothing is retold in prose between roles, so there is no telephone
game, and each agent can be tested against a hand-built context in isolation.

The context also owns the facts that make search *disciplined*, which is where the
measured failures were:

  * ``tried_signatures`` — an LLM run wasted an iteration re-running a config it
    had already run, having hallucinated that it changed a parameter.
  * ``dimension_counts`` — that same run never once varied the loss function,
    despite loss alignment being the strongest single lever measured (+0.0013),
    because nothing tracked which axes were unexplored.
"""
from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The axes of the search space. Coverage across these is what the Product
#: Manager agent is responsible for; a run that only ever tunes one axis is the
#: failure mode we are guarding against.
DIMENSIONS = [
    "loss",             # pointwise / listwise / bpr — metric alignment
    "architecture",     # fm / deepfm / din / mmoe / lgb
    "multi_task",       # auxiliary signals and their weight
    "sequence",         # user behaviour history
    "features",         # field set, causal statistics
    "capacity",         # embedding width, experts, trees
    "optimisation",     # learning rate, epochs, batch size
]

#: Directions the organizers have already measured as dead ends. Feeding these to
#: the agents keeps them from re-deriving known-null results.
KNOWN_DEAD_ENDS = [
    "Adding static side features (all 13 CWM domains) scored 0.5940 vs 0.5950 for "
    "the base 5 fields — no gain, slightly worse.",
    "Raising embedding capacity: k = 8 / 16 / 32 gave 0.5895 / 0.5902 / 0.5887. "
    "Capacity is not the bottleneck.",
    "Pure user-side features cannot change within-user ordering; only features that "
    "vary across a user's own impressions can.",
]


@dataclass
class TrialRecord:
    """One executed experiment and what came of it."""
    iteration: int
    dimension: str
    hypothesis: str
    command: str
    signature: str
    primary: Optional[float]
    status: str
    source: str = "llm"
    error_kind: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.primary is not None


def parse_flags(command: str) -> Dict[str, str]:
    """Extract ``--flag value`` and ``--flag=value`` pairs from a command line.

    Both spellings must be handled: argparse accepts either, so an agent that
    writes ``--model=deepfm`` produces a perfectly valid trial. A parser that only
    understood the space-separated form once blocked two iterations at QA
    pre-flight for "no --model" on commands that plainly had one.
    """
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    flags: Dict[str, str] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t.lstrip("-")
            if "=" in key:
                k, v = key.split("=", 1)
                flags[k] = v
                i += 1
                continue
            if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                flags[key] = toks[i + 1]
                i += 2
                continue
            flags[key] = "true"
        i += 1
    return flags


def command_signature(command: str) -> str:
    """A canonical form of a trial command, for duplicate detection.

    Normalises away the interpreter path, ``--data_dir``, and argument order, so
    two commands that run the same experiment compare equal regardless of how
    they were spelled.
    """
    flags = parse_flags(command)
    flags.pop("data_dir", None)
    return " ".join(f"{k}={flags[k]}" for k in sorted(flags))


@dataclass
class ResearchContext:
    """Everything an agent needs to decide what to do next."""
    baseline: float
    max_iterations: int
    wall_clock_budget_s: float
    started_at: float = field(default_factory=time.time)

    iteration: int = 0
    best_score: Optional[float] = None
    best_iteration: Optional[int] = None
    trials: List[TrialRecord] = field(default_factory=list)
    directive: Optional[Any] = None          # set by the Product Manager agent
    seed_noise: float = 0.0008               # published std over 5 seeds

    # -- budget -----------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    @property
    def budget_remaining_s(self) -> float:
        return max(0.0, self.wall_clock_budget_s - self.elapsed_s)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.max_iterations - self.iteration)

    # -- history ----------------------------------------------------------

    @property
    def best_delta(self) -> float:
        return 0.0 if self.best_score is None else self.best_score - self.baseline

    def tried_signatures(self) -> set:
        return {t.signature for t in self.trials}

    def is_duplicate(self, command: str) -> bool:
        return command_signature(command) in self.tried_signatures()

    def dimension_counts(self) -> Dict[str, int]:
        counts = {d: 0 for d in DIMENSIONS}
        for t in self.trials:
            if t.dimension in counts:
                counts[t.dimension] += 1
        return counts

    def unexplored_dimensions(self) -> List[str]:
        return [d for d, n in self.dimension_counts().items() if n == 0]

    def record(self, trial: TrialRecord):
        self.trials.append(trial)
        if trial.primary is not None and (self.best_score is None
                                          or trial.primary > self.best_score):
            self.best_score = trial.primary
            self.best_iteration = trial.iteration

    # -- rendering for prompts -------------------------------------------

    def history_table(self, limit: int = 12) -> str:
        if not self.trials:
            return "(no experiments run yet)"
        lines = ["| iter | dimension | primary | delta | status | command |",
                 "|---|---|---|---|---|---|"]
        for t in self.trials[-limit:]:
            score = f"{t.primary:.4f}" if t.primary is not None else "FAILED"
            delta = f"{t.primary - self.baseline:+.4f}" if t.primary is not None else "—"
            lines.append(f"| {t.iteration} | {t.dimension} | {score} | {delta} | "
                         f"{t.status} | `{t.signature}` |")
        return "\n".join(lines)

    def coverage_report(self) -> str:
        counts = self.dimension_counts()
        parts = [f"{d}={counts[d]}" for d in DIMENSIONS]
        unexplored = self.unexplored_dimensions()
        out = "Experiments per dimension: " + ", ".join(parts)
        if unexplored:
            out += f"\nNEVER TRIED: {', '.join(unexplored)}"
        return out

    def significance_note(self) -> str:
        """How large a delta has to be before it means anything."""
        return (f"Seed noise is sigma={self.seed_noise:.4f}, so a single-seed gain "
                f"below {3 * self.seed_noise:+.4f} is not yet evidence of a real effect.")
