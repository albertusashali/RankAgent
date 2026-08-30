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
        self.entries: List[Dict] = []
        #: iteration id -> checkpoint name, so the submission step can find the winner.
        self._checkpoints: Dict[int, str] = {}
        self.hypotheses: List[str] = []
        self._init_md()

    def _init_md(self):
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(f"# RankAgent run log\n\n"
                    f"- **Run ID**: `{self.run_id}`\n"
                    f"- **Started**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"- **Benchmark**: KuaiRand-Pure — validation selection only, "
                    f"hidden test sealed until submission\n\n---\n\n")

    # -- diffs -------------------------------------------------------------

    def capture_diff(self, max_chars: int = 8000) -> str:
        """The working-tree diff at this moment, as the agent left it."""
        try:
            out = subprocess.run(["git", "diff", "--unified=3"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=20)
            diff = out.stdout or ""
        except Exception as exc:
            return f"(diff unavailable: {exc})"
        if not diff.strip():
            return "(no working-tree changes; trial varied configuration only)"
        return diff[:max_chars] + ("\n… truncated …" if len(diff) > max_chars else "")


    # -- hypotheses --------------------------------------------------------
    
    def log_hypothesis(self, hypothesis: str):
        try:
            self.hypotheses.append(hypothesis)
        except Exception as exc:
            print(f"Failed to log hypothesis: {exc}")


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
        toks = command.split()
        model = loss = None
        for i, t in enumerate(toks):
            if t == "--model" and i + 1 < len(toks):
                model = toks[i + 1]
            if t == "--loss" and i + 1 < len(toks):
                loss = toks[i + 1]
        if model is None:
            return None
        if model in ("fm", "mmoe", "lgb"):
            return model
        return f"{model}_{loss or 'listwise'}"

    # -- writing -----------------------------------------------------------

    def log_iteration(self, entry: IterationLogEntry):
        record = entry.model_dump()
        self.entries.append(record)
        ckpt = self._infer_checkpoint(entry.command)
        if ckpt:
            self._checkpoints[entry.iteration_id] = ckpt

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "total_iterations": len(self.entries),
                       "iterations": self.entries}, f, indent=2)

        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"### Iteration {entry.iteration_id} — {entry.stage} "
                    f"(`{entry.status}`)\n\n")
            f.write(f"**Hypothesis.** {entry.hypothesis}\n\n")
            if entry.rationale:
                f.write(f"**Rationale.** {entry.rationale}\n\n")
            f.write(f"- Proposal source: `{entry.proposal_source}`\n")
            f.write(f"- Target file: `{entry.target_file}`\n")
            f.write(f"- Command: `{entry.command}`\n")
            if entry.metrics:
                m = entry.metrics
                f.write(f"- **Validation**: GAUC {m.get('gauc', 0):.4f} | "
                        f"nDCG@5 {m.get('ndcg_5', 0):.4f} | "
                        f"primary {m.get('primary_score', 0):.4f} "
                        f"(delta {m.get('delta_from_baseline', 0):+.4f})\n")
            else:
                f.write("- **Validation**: none — the trial produced no metrics\n")
            f.write(f"- Wall clock: {entry.wall_clock_seconds:.1f}s | "
                    f"cumulative tokens: {entry.prompt_tokens + entry.completion_tokens:,d}\n")
            if entry.error_recovery:
                r = entry.error_recovery
                f.write(f"\n**Error / recovery.** Failure classified as "
                        f"`{r.get('failure_kind')}`; "
                        f"recovered: {r.get('recovered')}.\n")
                for a in r.get("attempts", []):
                    f.write(f"  - `{a.get('strategy')}` — {a.get('detail')}\n")
            if entry.code_diff and not entry.code_diff.startswith("(no working-tree"):
                f.write(f"\n<details><summary>Code diff</summary>\n\n"
                        f"```diff\n{entry.code_diff}\n```\n\n</details>\n")
            f.write("\n---\n\n")

    def write_summary(self, summary: RunSummary):
        payload = summary.model_dump()
        payload["iterations"] = self.entries
        payload["total_tokens"] = summary.total_tokens
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("## Run summary\n\n")
            f.write("| | |\n|---|---|\n")
            f.write(f"| Halt reason | {summary.halt_reason} |\n")
            best = (f"{summary.best_valid_primary:.4f}"
                    if summary.best_valid_primary is not None else "n/a")
            delta = (f"{summary.best_delta:+.4f}"
                     if summary.best_delta is not None else "n/a")
            f.write(f"| Best validation primary | {best} (iteration {summary.best_iteration}) |\n")
            f.write(f"| Delta over official baseline | {delta} |\n")
            f.write(f"| Iterations used | {summary.iterations_used} / {summary.iteration_cap} |\n")
            f.write(f"| Agent wall clock | {summary.wall_clock_seconds/60:.1f} min |\n")
            f.write(f"| LLM tokens (in + out) | {summary.total_tokens:,d} "
                    f"in {summary.llm_calls} calls |\n")
            f.write(f"| Error recoveries | {summary.error_recoveries} |\n")
            f.write(f"| Failed iterations | {summary.failed_iterations} |\n")
            f.write(f"| Manual interventions | {summary.manual_interventions} |\n")
            f.write(f"| Submission | `{summary.submission_path or 'not produced'}` |\n")
