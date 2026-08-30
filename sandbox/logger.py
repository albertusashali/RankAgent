"""Run-log writer.

Produces the two artefacts the challenge asks for as deliverable 3:

  ``logs/run_summary.json``  machine-readable per-iteration record plus the
                             resource totals Feasibility is scored on
  ``logs/run_log.md``        the human-readable journal

Each iteration records its hypothesis, the code diff, the resulting metrics, and
any error/recovery event. Two things are deliberate:

  * The markdown log is **truncated at the start of a run**, not appended to
    forever. The previous version appended, so repeated runs interleaved and the
    file showed "Iteration 0" four times with different numbers.
  * ``capture_diff`` shells out to ``git diff`` so the recorded diff is what
    actually changed on disk, rather than a restatement of the command.
"""
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

from orchestrator.schemas import IterationLogEntry, RunSummary


class RunLogger:
    def __init__(self, log_dir: str = "logs", run_id: Optional[str] = None):
        self.log_dir = log_dir
        self.run_id = run_id or f"rankagent-{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        self.json_path = os.path.join(log_dir, "run_summary.json")
        self.md_path = os.path.join(log_dir, "run_log.md")

        # Every run also lands under its own id. The canonical paths above are
        # overwritten by each new run, so without this an experiment you intend
        # to submit is one `make agent` away from being destroyed.
        self.archive_dir = os.path.join(log_dir, "runs")
        os.makedirs(self.archive_dir, exist_ok=True)
        self.archive_json = os.path.join(self.archive_dir, f"{self.run_id}.json")
        self.archive_md = os.path.join(self.archive_dir, f"{self.run_id}.md")
        self.entries: List[Dict] = []
        #: iteration id -> checkpoint name, so the submission step can find the winner.
        self._checkpoints: Dict[int, str] = {}
        self._init_md()

    def _init_md(self):
        header = (f"# RankAgent run log\n\n"
                  f"- **Run ID**: `{self.run_id}`\n"
                  f"- **Started**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"- **Benchmark**: KuaiRand-Pure — validation selection only, "
                  f"hidden test sealed until submission\n\n---\n\n")
        for path in (self.md_path, self.archive_md):
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)

    # -- diffs -------------------------------------------------------------

    #: Paths the agent's own bookkeeping writes to. Excluded from the captured
    #: diff because including them makes the field self-referential: the log
    #: records a diff of the log, which contains the previous diff of the log.
    #: In practice that consumed the entire size budget with churn and left every
    #: iteration's `code_diff` byte-identical and uninformative.
    DIFF_EXCLUDES = ("logs", "submissions", "checkpoints", "data")

    def capture_diff(self, max_chars: int = 8000) -> str:
        """The working-tree diff at this moment, as the agent left it.

        Only source changes — the agent's own artefacts are excluded, so this
        field answers "what code did this iteration change?" and nothing else.
        """
        cmd = ["git", "diff", "--unified=3", "--"]
        cmd += [f":(exclude){p}" for p in self.DIFF_EXCLUDES]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=20)
            diff = out.stdout or ""
        except Exception as exc:
            return f"(diff unavailable: {exc})"
        if not diff.strip():
            return "(no working-tree changes; trial varied configuration only)"
        return diff[:max_chars] + ("\n… truncated …" if len(diff) > max_chars else "")

    # -- checkpoints -------------------------------------------------------

    def note_checkpoint(self, iteration_id: int, name: str):
        self._checkpoints[iteration_id] = name

    def checkpoint_for(self, iteration_id: int) -> Optional[str]:
        """Which checkpoint an iteration produced.

        Falls back to reading the command, because the trainer names checkpoints
        deterministically from ``--model`` and ``--loss``.
        """
        if iteration_id in self._checkpoints:
            return self._checkpoints[iteration_id]
        for e in self.entries:
            if e["iteration_id"] == iteration_id:
                return self._infer_checkpoint(e.get("command", ""))
        return None

    @staticmethod
    def _infer_checkpoint(command: str) -> Optional[str]:
        """Deterministic checkpoint name for a trial command.

        Handles both `--model x` and `--model=x`; argparse accepts either, so a
        parser that only understood one form would fail to locate the winning
        checkpoint at submission time.
        """
        from agents.context import parse_flags
        flags = parse_flags(command)
        model = flags.get("model")
        if model is None:
            return None
        if model in ("fm", "mmoe", "lgb"):
            return model
        return f"{model}_{flags.get('loss', 'listwise')}"

    # -- writing -----------------------------------------------------------

    def _append_md(self, text: str):
        """Append to both the canonical log and this run's archived copy."""
        for path in (self.md_path, self.archive_md):
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)

    def _write_json(self, blob: dict):
        for path in (self.json_path, self.archive_json):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)

    @staticmethod
    def _render_iteration(entry: IterationLogEntry) -> str:
        out = [f"### Iteration {entry.iteration_id} — {entry.stage} (`{entry.status}`)\n\n",
               f"**Hypothesis.** {entry.hypothesis}\n\n"]
        if entry.rationale:
            out.append(f"**Rationale.** {entry.rationale}\n\n")
        out.append(f"- Proposal source: `{entry.proposal_source}`\n")
        out.append(f"- Target file: `{entry.target_file}`\n")
        out.append(f"- Command: `{entry.command}`\n")
        if entry.metrics:
            m = entry.metrics
            out.append(f"- **Validation**: GAUC {m.get('gauc', 0):.4f} | "
                       f"nDCG@5 {m.get('ndcg_5', 0):.4f} | "
                       f"primary {m.get('primary_score', 0):.4f} "
                       f"(delta {m.get('delta_from_baseline', 0):+.4f})\n")
        else:
            out.append("- **Validation**: none — the trial produced no metrics\n")
        cum = entry.cumulative_prompt_tokens + entry.cumulative_completion_tokens
        out.append(f"- Wall clock: {entry.wall_clock_seconds:.1f}s\n")
        out.append(f"- Tokens this iteration: {entry.iteration_tokens:,d} "
                   f"({entry.prompt_tokens:,d} in + {entry.completion_tokens:,d} out, "
                   f"{entry.llm_calls} call{'s' if entry.llm_calls != 1 else ''}) "
                   f"| cumulative: {cum:,d}\n")
        if entry.error_recovery:
            r = entry.error_recovery
            out.append(f"\n**Error / recovery.** Failure classified as "
                       f"`{r.get('failure_kind')}`; recovered: {r.get('recovered')}.\n")
            for a in r.get("attempts", []):
                out.append(f"  - `{a.get('strategy')}` — {a.get('detail')}\n")
        if entry.code_diff and not entry.code_diff.startswith("(no working-tree"):
            out.append(f"\n<details><summary>Code diff</summary>\n\n"
                       f"```diff\n{entry.code_diff}\n```\n\n</details>\n")
        out.append("\n---\n\n")
        return "".join(out)

    def log_iteration(self, entry: IterationLogEntry):
        self.entries.append(entry.model_dump())
        ckpt = self._infer_checkpoint(entry.command)
        if ckpt:
            self._checkpoints[entry.iteration_id] = ckpt

        self._write_json({"run_id": self.run_id,
                          "total_iterations": len(self.entries),
                          "iterations": self.entries})
        self._append_md(self._render_iteration(entry))

    def write_summary(self, summary: RunSummary):
        payload = summary.model_dump()
        payload["iterations"] = self.entries
        payload["total_tokens"] = summary.total_tokens
        self._write_json(payload)

        best = (f"{summary.best_valid_primary:.4f}"
                if summary.best_valid_primary is not None else "n/a")
        delta = (f"{summary.best_delta:+.4f}"
                 if summary.best_delta is not None else "n/a")
        rows = [
            ("Halt reason", summary.halt_reason),
            ("Best validation primary", f"{best} (iteration {summary.best_iteration})"),
            ("Delta over official baseline", delta),
            ("Iterations used", f"{summary.iterations_used} / {summary.iteration_cap}"),
            ("Agent wall clock", f"{summary.wall_clock_seconds/60:.1f} min"),
            ("LLM tokens (in + out)", f"{summary.total_tokens:,d} in {summary.llm_calls} calls"),
            ("Error recoveries", summary.error_recoveries),
            ("Failed iterations", summary.failed_iterations),
            ("Manual interventions", summary.manual_interventions),
            ("Submission", f"`{summary.submission_path or 'not produced'}`"),
        ]
        md = ["## Run summary\n\n", "| | |\n|---|---|\n"]
        md += [f"| {k} | {v} |\n" for k, v in rows]
        md.append(f"\n> Archived copy of this run: `{self.archive_json}`\n")
        self._append_md("".join(md))
