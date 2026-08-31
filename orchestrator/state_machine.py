"""The autonomous research loop.

    INIT -> BASELINE -> [ HYPOTHESIZE -> RUN -> EVAL -> REFLECT ] * -> SUBMIT -> HALT

Guarantees this loop makes, each of which was broken before:

  * **It finishes.** Every iteration is wrapped so that no failure — a crash, a
    timeout, an unparseable stdout, a bad LLM response — can end the run. Failures
    go to the self-healing debugger and are recorded either way.
  * **The log is truthful.** A hypothesis and the command that tested it are
    generated together and logged together, so the run-log cannot claim one thing
    and have run another. Token counts come from the API response.
  * **It never sees the hidden test set.** Trials load train + valid only; test
    labels are sealed in the loader. The submission is built once, at the end,
    from whichever checkpoint won on validation.
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agents.context import ResearchContext
from agents.team import AgentTeam
from agents.qa import RANDOM_FLOOR
from orchestrator.interventions import InterventionLedger
from orchestrator.schemas import (ExecutionResult, HypothesisProposal,
                                  IterationLogEntry, RunSummary, TokenUsage)
from orchestrator.tree_manager import BASELINE_VAL_PRIMARY, TreeManager
from sandbox.debugger import SelfHealingDebugger
from sandbox.logger import RunLogger
from sandbox.runner import ExecutionRunner

PY = sys.executable
CONFIG_DIR = "configs"


def load_dotenv_if_present(path: str = ".env"):
    """Minimal .env reader — avoids a dependency for three lines of parsing."""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'").strip('"')
                    if k and not os.environ.get(k):
                        os.environ[k] = v
    except OSError:
        pass


def load_yaml(name: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# The research space
# ---------------------------------------------------------------------------
#
# Ordered by expected value, informed by the organizers' own ablations: they
# measured that adding static side features and raising embedding capacity do
# nothing, so neither appears here as a primary lever. What they flagged as
# untested — aligning the loss with the ranking metric, and modelling the user's
# behaviour sequence — leads.

STRATEGY_BANK: List[Dict[str, str]] = [
    {"stage": "Loss Function",
     "hypothesis": "Replace pointwise BCE with a within-user listwise softmax. The metrics "
                   "(GAUC, nDCG@5) rank inside a user's impression list, so a per-impression "
                   "likelihood optimises the wrong quantity; a listwise objective is invariant "
                   "to per-user score offsets exactly as the metrics are.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model fm_torch --loss listwise --epochs 15"},

    {"stage": "Multi-Task Learning",
     "hypothesis": "Train long_view jointly with click, like and forward through an MMoE. The "
                   "auxiliary signals are logged on every impression and should regularise the "
                   "shared embedding without diluting the scored head, which keeps its own gate.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model mmoe --loss listwise --experts 4 --epochs 12"},

    {"stage": "Tree-based Ranker",
     "hypothesis": "Fit LightGBM with the lambdarank objective truncated at 5 over causal "
                   "engagement statistics. A GBDT exploits dense count features that an "
                   "embedding model handles poorly, giving an ensemble member with "
                   "decorrelated errors.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank --trees 400"},

    {"stage": "Sequential Modelling",
     "hypothesis": "Add target attention over the user's last 10 impressions (DIN). Nothing in "
                   "the baseline uses behaviour order, and attention conditioned on the "
                   "candidate video should separate durable taste from incidental exposure.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model din --loss listwise --max_seq_len 10 --epochs 10"},

    {"stage": "Loss Function",
     "hypothesis": "Compare a pairwise BPR objective against the listwise one. BPR optimises "
                   "the pairwise ordering AUC counts directly, so if GAUC matters more than "
                   "top-heavy nDCG it should win.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model fm_torch --loss bpr --epochs 15"},

    {"stage": "Architecture",
     "hypothesis": "Add an MLP branch over the field embeddings (DeepFM) under the listwise "
                   "loss, testing whether implicit higher-order crosses add anything over the "
                   "second-order FM term at this data scale.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model deepfm --loss listwise --epochs 12"},

    {"stage": "Regularisation",
     "hypothesis": "Widen the MMoE expert pool while holding embedding width fixed. Capacity in "
                   "the embedding table is known not to help; capacity in the task-routing "
                   "layer is a different axis and has not been tested.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model mmoe --loss listwise --experts 8 --expert_dim 96 --epochs 12"},

    {"stage": "Multi-Task Learning",
     "hypothesis": "Lower the auxiliary task weight. If the rare signals are dominating the "
                   "shared trunk rather than regularising it, a smaller weight should recover "
                   "the scored head's accuracy.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model mmoe --loss listwise --aux_weight 0.1 --epochs 12"},

    {"stage": "Hyperparameter Tuning",
     "hypothesis": "Halve the learning rate on the listwise FM and extend the epoch budget; "
                   "the listwise objective converged in 8 epochs, which suggests the step size "
                   "is overshooting a shallow optimum.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model fm_torch --loss listwise --lr 0.0005 --epochs 25"},

    {"stage": "Tree-based Ranker",
     "hypothesis": "Deepen the GBDT with more leaves and a lower learning rate. The causal "
                   "features are low-dimensional, so extra depth is affordable and may capture "
                   "interactions the embedding models get for free.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model lgb --num_leaves 127 --lr 0.03 --trees 600"},
]


class RankAgentOrchestrator:
    #: How far the reproduced baseline may sit from the published 0.6016 before
    #: the run refuses to continue. Seed noise is 0.0008, so 0.005 is roughly
    #: 6 sigma — comfortably wide for legitimate variation and far too narrow to
    #: let a genuinely broken harness through.
    BASELINE_TOLERANCE = 0.005

    #: Consecutive failures after which the run halts rather than continuing.
    #: A capable agent can legitimately fail several times in a row on a hard
    #: problem, so this is deliberately not tight; it exists to stop a run that
    #: is broken rather than merely struggling from burning six hours.
    MAX_CONSECUTIVE_FAILURES = 8

    #: How many times the Engineer may fix its own generated code before the
    #: node is pruned. Each attempt costs one LLM call plus one ~6s smoke run.
    MAX_CODE_REPAIRS = 2

    def __init__(self, data_dir: Optional[str] = None,
                 max_iterations: Optional[int] = None,
                 max_wall_clock: Optional[int] = None, run_id: Optional[str] = None,
                 run_baseline: bool = True):
        load_dotenv_if_present()
        agent_cfg = load_yaml("agent_config.yaml")
        bench_cfg = load_yaml("benchmark_kuairand.yaml")
        conv = (bench_cfg.get("convergence") or {})

        self.data_dir = data_dir
        self.run_baseline_flag = run_baseline
        #: Whether this run actually re-ran the baseline, or merely asserted it.
        self.baseline_reproduced = False
        # Explicit arguments win over the YAML defaults; YAML wins over the
        # hard-coded fallback. The previous order let the config silently
        # override a budget the operator had asked for on the command line.
        self.max_iterations = int(
            max_iterations if max_iterations is not None
            else conv.get("max_iterations", 50) or 50)
        self.max_wall_clock = int(
            max_wall_clock if max_wall_clock is not None
            else conv.get("max_wall_clock_seconds", 21600) or 21600)
        self.epsilon = float(conv.get("epsilon", 0.002))
        self.patience = int(conv.get("patience_iterations", 3))

        sandbox_cfg = (agent_cfg.get("sandbox") or {})
        dbg_cfg = (agent_cfg.get("debugger") or {})

        self.tree = TreeManager(epsilon=self.epsilon, n_convergence=self.patience,
                                max_iterations=self.max_iterations)
        self.runner = ExecutionRunner(
            timeout_seconds=int(sandbox_cfg.get("timeout_seconds_per_iteration", 1800)))
        self.debugger = SelfHealingDebugger(
            max_retries=int(dbg_cfg.get("max_self_healing_attempts", 3)),
            llm_repair=self._llm_repair)
        self.logger = RunLogger(run_id=run_id)

        self.tokens = TokenUsage()
        self.team = AgentTeam(self.tokens, data_dir=data_dir,
                              pm_refresh=int(agent_cfg.get("agent", {}).get("pm_refresh", 3)),
                              proposals=int(agent_cfg.get("agent", {}).get("proposals_per_call", 3)),
                              max_retries=int(dbg_cfg.get("max_self_healing_attempts", 3)))
        self.ctx = ResearchContext(baseline=BASELINE_VAL_PRIMARY,
                                   max_iterations=self.max_iterations,
                                   wall_clock_budget_s=float(self.max_wall_clock))
        self.manual_interventions = 0
        self.error_recoveries = 0
        self.failed_iterations = 0
        self._used_commands: set = set()
        #: node_id -> Workspace, for every node that produced a trusted score. A
        #: child materialises from its parent's entry here, which is the
        #: mechanism by which successive edits compose; and the submission is
        #: exported from the winner's, where its checkpoint actually lives.
        self.workspaces: Dict[int, Any] = {}
        self.ledger = InterventionLedger()
        self.baseline_measured: Optional[float] = None
        self.baseline_drift: Optional[float] = None
        self.submission_decision: Dict[str, Any] = {}
        self._client = None

    # -- data_dir threading ------------------------------------------------

    def _with_data_dir(self, cmd: str) -> str:
        """Thread --data_dir into a trial command. Previously accepted and dropped."""
        if self.data_dir and "--data_dir" not in cmd:
            return f"{cmd} --data_dir {self.data_dir}"
        return cmd

    # -- phase 0 -----------------------------------------------------------

    def run_baseline(self) -> bool:
        print("=" * 74)
        print(">>> PHASE 0 — reproduce the official Factorization Machine baseline")
        print("=" * 74)
        # Node 0 gets its own workspace so the baseline is reproduced through
        # exactly the same path every later trial takes. Verifying the harness
        # in a configuration the run never uses again would prove little.
        ws = self._materialise(0, None)
        self.workspaces[0] = ws
        cmd = self._with_data_dir(f"{PY} -m pipeline.train --model fm")
        res = self.runner.run_command(cmd, env_vars=ws.env() if ws else None,
                                      cwd=ws.root if ws else None)
        if res.status == "SUCCESS" and res.metrics:
            m = res.metrics
            drift = m.primary_score - BASELINE_VAL_PRIMARY
            self.baseline_measured = m.primary_score
            self.baseline_drift = drift
            print(f"[BASELINE] valid GAUC {m.gauc:.4f} | nDCG@5 {m.ndcg_5:.4f} | "
                  f"primary {m.primary_score:.4f} (published {BASELINE_VAL_PRIMARY:.4f}, "
                  f"drift {drift:+.4f})")
            if abs(drift) > self.BASELINE_TOLERANCE:
                # Previously this printed a warning and continued. Continuing
                # means every delta the run reports is measured against a
                # reference that does not match the published one, which makes
                # the whole run unreportable — so it is now a halt.
                print(f"[HALT] baseline drift {drift:+.4f} exceeds the "
                      f"{self.BASELINE_TOLERANCE} tolerance. Every later delta "
                      f"would be measured against an unverified reference.")
                return False
            self.baseline_reproduced = True
            self.ctx.baseline = m.primary_score
            self.tree.record_baseline(m.primary_score, node_id=0)
            # Keep the tree's reference in step with what was actually measured.
            # It previously stayed at the published constant, so agent prompts
            # and the run log reported deltas against two different numbers.
            self.tree.baseline = m.primary_score
            self.logger.log_iteration(IterationLogEntry(
                iteration_id=0, node_id=0, parent_node_id=None,
                stage="Baseline Reproduction",
                hypothesis="Reproduce the organizer's FM baseline end to end",
                rationale="Every later delta is measured against this run.",
                target_file="pipeline/train.py", command=cmd, proposal_source="fallback",
                status="ACCEPTED", metrics=m.model_dump(),
                delta_over_baseline=m.primary_score - BASELINE_VAL_PRIMARY,
                wall_clock_seconds=res.wall_clock_seconds))
            return True

        print(f"[BASELINE FAILED] {(res.error_traceback or '')[-800:]}")
        return False

    # -- hypothesis generation --------------------------------------------

    def _fallback_proposal(self, iteration_id: int) -> HypothesisProposal:
        """Deterministic research plan, used when no LLM is configured.

        Skips strategies already tried so the loop keeps making progress instead
        of cycling, which is what the modulo-indexed bank used to do.
        """
        for strat in STRATEGY_BANK:
            if strat["command"] not in self._used_commands:
                return HypothesisProposal(source="fallback", rationale="deterministic plan",
                                          **strat)
        strat = STRATEGY_BANK[(iteration_id - 1) % len(STRATEGY_BANK)]
        return HypothesisProposal(source="fallback",
                                  rationale="plan exhausted; repeating with a new seed",
                                  **{**strat, "command": f"{strat['command']} --seed {iteration_id}"})

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key and len(key) > 10 and not key.startswith("your"):
            try:
                import anthropic
                self._client = ("anthropic", anthropic.Anthropic(api_key=key))
                return self._client
            except Exception:
                pass
        key = os.environ.get("OPENAI_API_KEY")
        if key and len(key) > 10 and not key.startswith("your"):
            try:
                import openai
                self._client = ("openai", openai.OpenAI(api_key=key))
                return self._client
            except Exception:
                pass
        return None

    def propose(self, iteration_id: int) -> HypothesisProposal:
        """Ask the LLM for the next hypothesis; fall back to the plan on any problem.

        The hypothesis and its command are parsed from the *same* response and
        returned as one object. If anything is missing or malformed the whole
        proposal is replaced — never half of it.
        """
        from prompts.templates import HYPOTHESIS_PROMPT, SYSTEM_PROMPT

        client = self._client_or_none()
        if client is None:
            return self._fallback_proposal(iteration_id)

        prompt = HYPOTHESIS_PROMPT.format(
            iteration_id=iteration_id,
            best_score=self.tree.best_primary_score,
            delta=self.tree.best_delta,
            history_summary=self.tree.get_history_summary(),
        )
        try:
            kind, api = client
            if kind == "anthropic":
                resp = api.messages.create(
                    model=os.environ.get("RANKAGENT_MODEL", "claude-sonnet-5"),
                    max_tokens=1200, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in resp.content if b.type == "text")
                self.tokens.add(resp.usage.input_tokens, resp.usage.output_tokens)
            else:
                resp = api.chat.completions.create(
                    model=os.environ.get("RANKAGENT_MODEL", "gpt-4o"),
                    response_format={"type": "json_object"}, temperature=0.3,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}])
                text = resp.choices[0].message.content
                self.tokens.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)

            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
            cmd = str(data["execution_command"]).strip()
            if "pipeline.train" not in cmd:
                raise ValueError("command does not invoke pipeline.train")
            if cmd.startswith("python "):
                cmd = f"{PY} " + cmd[len("python "):]
            return HypothesisProposal(
                stage=str(data.get("stage", "Unspecified")),
                hypothesis=str(data["hypothesis"]),
                rationale=str(data.get("rationale", "")),
                target_file=str(data.get("target_file", "pipeline/train.py")),
                command=cmd, source="llm")
        except Exception as exc:
            print(f"  [WARN] LLM proposal unusable ({exc}); using the deterministic plan.")
            return self._fallback_proposal(iteration_id)

    def _llm_repair(self, command: str, traceback_text: str, kind: str) -> Optional[str]:
        """Repair callback handed to the debugger. Returns a corrected command."""
        from prompts.templates import DEBUGGER_PROMPT

        client = self._client_or_none()
        if client is None:
            return None
        try:
            api_kind, api = client
            prompt = DEBUGGER_PROMPT.format(command=command, failure_kind=kind,
                                            error_traceback=(traceback_text or "")[-3000:])
            if api_kind == "anthropic":
                resp = api.messages.create(
                    model=os.environ.get("RANKAGENT_MODEL", "claude-sonnet-5"),
                    max_tokens=400, messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in resp.content if b.type == "text")
                self.tokens.add(resp.usage.input_tokens, resp.usage.output_tokens)
            else:
                resp = api.chat.completions.create(
                    model=os.environ.get("RANKAGENT_MODEL", "gpt-4o"), temperature=0.0,
                    messages=[{"role": "user", "content": prompt}])
                text = resp.choices[0].message.content
                self.tokens.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            for line in text.splitlines():
                if "pipeline.train" in line:
                    fixed = line.strip().strip("`").strip()
                    if fixed.startswith("python "):
                        fixed = f"{PY} " + fixed[len("python "):]
                    return fixed
        except Exception:
            return None
        return None

    # -- one iteration -----------------------------------------------------

    # -- workspaces --------------------------------------------------------

    def _materialise(self, iteration_id: int, parent_id: Optional[int]):
        """This node's code state, copied from its parent's."""
        from sandbox.workspace import materialise
        try:
            parent_ws = self.workspaces.get(parent_id) if parent_id is not None else None
            return materialise(iteration_id, parent=parent_ws)
        except Exception as exc:
            # Losing the workspace must not lose the iteration: fall back to a
            # configuration-only trial against the canonical repo.
            print(f"  [WARN] could not materialise a workspace ({exc}); "
                  f"running configuration-only against the repository")
            return None

    def _record_failure(self, iteration_id, parent_id, spec, plan, command, res,
                        tokens_before, note: str) -> bool:
        """Log a trial that never produced metrics, and keep the run going."""
        self.failed_iterations += 1
        impl = spec.implementation
        from sandbox.debugger import classify
        kind = classify(res.error_traceback or "")
        self.team.record(self.ctx, iteration_id, spec, None, "FAILED", error_kind=kind)
        trace = plan.as_log()
        trace["failure"] = {"note": note, "status": res.status, "kind": kind}
        self.logger.log_iteration(IterationLogEntry(
            iteration_id=iteration_id, node_id=iteration_id, parent_node_id=parent_id,
            stage=spec.dimension, hypothesis=spec.hypothesis,
            rationale=spec.mechanism,
            target_file=", ".join(impl.target_files) if impl else "(configuration only)",
            command=command, proposal_source=spec.source,
            code_diff=impl.diff if impl else "",
            status="FAILED", metrics=None, agent_trace=trace,
            error_recovery={"recovered": False, "failure_kind": kind,
                            "attempts": [{"strategy": note,
                                          "detail": (res.error_traceback or "")[-600:]}]},
            prompt_tokens=self.tokens.prompt_tokens - tokens_before[0],
            completion_tokens=self.tokens.completion_tokens - tokens_before[1],
            llm_calls=self.tokens.calls - tokens_before[2],
            cumulative_prompt_tokens=self.tokens.prompt_tokens,
            cumulative_completion_tokens=self.tokens.completion_tokens,
            wall_clock_seconds=res.wall_clock_seconds))
        return self.tree.add_node(iteration_id, parent_id, spec.hypothesis,
                                  "-", None)

    def run_iteration(self, iteration_id: int) -> bool:
        """Execute one hypothesis. Returns whether the run should halt."""
        print(f"\n{'=' * 74}\n>>> ITERATION {iteration_id}/{self.max_iterations}\n{'=' * 74}")

        # Snapshot the meters so this iteration's entry records what THIS
        # iteration spent. Reading the running totals here instead logged the
        # cumulative figure in a per-iteration field, which made a 1.5k-token
        # iteration look like a 5.7k-token one. Differencing also captures the
        # repair calls the debugger makes later in this same iteration.
        tokens_before = (self.tokens.prompt_tokens, self.tokens.completion_tokens,
                         self.tokens.calls)

        self.ctx.iteration = iteration_id

        # Materialise this node's code from its PARENT, so an edit here sits on
        # top of every edit already accepted on that branch. This is what makes
        # improvements compose instead of each trial restarting from baseline.
        parent_id = self.tree.select_parent()
        workspace = self._materialise(iteration_id, parent_id)

        plan = self.team.plan(self.ctx, workspace=workspace)

        if not plan.ok:
            # The team could not produce a runnable, non-duplicate experiment.
            # That is a real outcome worth logging, not a crash — and it must not
            # count toward convergence, or a stuck planner would look converged.
            print("  [SKIPPED] no runnable experiment this iteration")
            self.failed_iterations += 1
            self.logger.log_iteration(IterationLogEntry(
                iteration_id=iteration_id, node_id=iteration_id,
                parent_node_id=self.tree.select_parent(),
                stage="Planning", hypothesis="No runnable experiment could be produced.",
                rationale="; ".join(plan.trace), target_file="-",
                command="(none)", proposal_source="fallback",
                status="FAILED", metrics=None, agent_trace=plan.as_log(),
                prompt_tokens=self.tokens.prompt_tokens - tokens_before[0],
                completion_tokens=self.tokens.completion_tokens - tokens_before[1],
                llm_calls=self.tokens.calls - tokens_before[2],
                cumulative_prompt_tokens=self.tokens.prompt_tokens,
                cumulative_completion_tokens=self.tokens.completion_tokens))
            return self.tree.add_node(iteration_id, self.tree.select_parent(),
                                      "no runnable experiment", "-", None)

        spec = plan.spec
        command = spec.command
        self._used_commands.add(spec.args)

        print(f"  Dimension  : {spec.dimension}")
        print(f"  Hypothesis : {spec.hypothesis}")
        if spec.implementation is not None:
            impl = spec.implementation
            print(f"  Code change: {', '.join(impl.target_files)} "
                  f"(+{impl.lines_added}/-{impl.lines_removed})")
        print(f"  Command    : {command}")

        env = workspace.env() if workspace else None
        cwd = workspace.root if workspace else None

        # Smoke gate: one epoch on a small user subsample, ~5s. Generated code
        # fails often, and without this every syntax slip or shape bug costs a
        # full training run. Only worth it when code was actually written —
        # a config-only trial is running code that already passed this once.
        if workspace is not None and spec.implementation is not None:
            smoke = self.runner.run_command(f"{command} --smoke", env_vars=env,
                                            cwd=cwd, timeout_seconds=300)

            # A generated patch that crashes is a bug report, not a dead end.
            # Hand the traceback back to the Engineer and let it fix its own
            # code — the whole exchange costs seconds because the smoke run is
            # subsampled, where discovering the same bug in a full run would
            # cost minutes.
            def smoke_verdict(r):
                """Did the smoke run produce something worth a full run?

                Exit code alone is not enough. A loss with the sign flipped, or
                one that normalises over the wrong axis, trains happily to
                nonsense — and a run that scores below the random floor is a
                broken implementation, not a weak idea. That is checkable on the
                subsample: the random-scoring floor does not depend on how long
                you trained. One run reached the full trainer with a ListMLE
                that scored 0.3774, well under the 0.4834 floor, because smoke
                only asked whether the process exited zero.
                """
                if r.failed:
                    return r.status, r.error_traceback or ""
                if r.metrics and r.metrics.primary_score < RANDOM_FLOOR:
                    return "BELOW_RANDOM", (
                        f"The smoke run completed but scored "
                        f"{r.metrics.primary_score:.4f}, below the "
                        f"{RANDOM_FLOOR} floor a random scorer achieves. The "
                        f"implementation is wrong, not merely weak — check the "
                        f"sign of the loss, the axis it reduces over, and that "
                        f"it returns a scalar to be MINIMISED.")
                return None, ""

            repairs = 0
            bad, why = smoke_verdict(smoke)
            while bad and repairs < self.MAX_CODE_REPAIRS:
                repairs += 1
                print(f"  [SMOKE FAILED:{bad}] repair attempt "
                      f"{repairs}/{self.MAX_CODE_REPAIRS}")
                fixed = self.team.repair_code(self.ctx, workspace, spec, why)
                if fixed is None:
                    break
                spec.implementation = fixed
                self.error_recoveries += 1
                smoke = self.runner.run_command(f"{command} --smoke", env_vars=env,
                                                cwd=cwd, timeout_seconds=300)
                bad, why = smoke_verdict(smoke)

            if bad:
                print(f"  [PRUNED] {bad} after {repairs} repair attempt(s); "
                      f"not spending a full run on it")
                return self._record_failure(
                    iteration_id, parent_id, spec, plan, command, smoke,
                    tokens_before, note=f"{bad} after {repairs} repair attempts")
            print(f"  [SMOKE] passed in {smoke.wall_clock_seconds:.0f}s"
                  + (f" after {repairs} repair(s)" if repairs else ""))

        res = self.runner.run_command(command, env_vars=env, cwd=cwd)
        recovery: Optional[dict] = None
        status = "REJECTED"

        if res.failed:
            print(f"  [FAILURE:{res.status}] handing to QA")
            outcome = self.team.recover(command, res.error_traceback or "",
                                        self.runner.run_command)
            recovery = outcome.as_log()
            if outcome.repaired:
                self.error_recoveries += 1
                res = getattr(outcome, "result")
                command = outcome.command or command
                status = "ERROR_RECOVERED"
                print(f"  [RECOVERED] repaired via {outcome.attempts[-1].strategy}")
            else:
                self.failed_iterations += 1
                print(f"  [PRUNED] {outcome.kind}: no repair applied; moving on")

        metrics = res.metrics if res.status == "SUCCESS" else None
        if metrics:
            print(f"  [RESULT] valid GAUC {metrics.gauc:.4f} | nDCG@5 {metrics.ndcg_5:.4f} | "
                  f"primary {metrics.primary_score:.4f} (delta {metrics.delta_from_baseline:+.4f})")

        # QA judges whether the number is believable before it is allowed to
        # become the new best. A score below the random floor or above the oracle
        # ceiling means the harness broke, not that the model improved.
        verdict = self.team.review(metrics.primary_score if metrics else None)
        trusted = metrics if (metrics and verdict.trustworthy) else None
        if metrics and not verdict.trustworthy:
            status = "REJECTED"

        impl = spec.implementation
        target_file = ", ".join(impl.target_files) if impl else "(configuration only)"

        parent = parent_id
        converged = self.tree.add_node(iteration_id, parent, spec.hypothesis,
                                       target_file, trusted)
        if trusted is not None and status != "ERROR_RECOVERED":
            status = self.tree.nodes[iteration_id]["status"]
        elif metrics is None:
            status = "FAILED"

        # Remember every node's workspace, not just the ones that improved. The
        # submission is exported from the WINNING node's workspace, and its
        # checkpoints live only there; resolving the winner by name against the
        # repository's shared checkpoints/ directory is what previously let a
        # stale model from an earlier run be exported under this run's score.
        if workspace is not None and trusted is not None:
            self.workspaces[iteration_id] = workspace

        # A code change that improved on its parent becomes part of the lineage
        # every later node builds on and every later prompt is told about.
        if impl is not None and status == "ACCEPTED":
            self.ctx.edits_applied.append(impl.summary or spec.hypothesis[:60])
            self.ctx.lineage = "baseline + " + " + ".join(self.ctx.edits_applied)

        self.team.record(self.ctx, iteration_id, spec,
                         trusted.primary_score if trusted else None, status,
                         error_kind=(recovery or {}).get("failure_kind"))

        trace = plan.as_log()
        trace["qa_verdict"] = verdict.model_dump()
        self.logger.log_iteration(IterationLogEntry(
            iteration_id=iteration_id, node_id=iteration_id, parent_node_id=parent,
            stage=spec.dimension, hypothesis=spec.hypothesis,
            rationale=spec.mechanism, target_file=target_file,
            command=command, proposal_source=spec.source,
            # The diff is computed from the bytes on disk before and after the
            # patch, against this node's PARENT — not from what the model said
            # it did, and not from `git diff` over the repo working tree, which
            # is why every archived run logged unrelated README churn.
            code_diff=impl.diff if impl else "",
            status=status, metrics=metrics.model_dump() if metrics else None,
            delta_over_baseline=metrics.delta_from_baseline if metrics else None,
            error_recovery=recovery, agent_trace=trace,
            prompt_tokens=self.tokens.prompt_tokens - tokens_before[0],
            completion_tokens=self.tokens.completion_tokens - tokens_before[1],
            llm_calls=self.tokens.calls - tokens_before[2],
            cumulative_prompt_tokens=self.tokens.prompt_tokens,
            cumulative_completion_tokens=self.tokens.completion_tokens,
            wall_clock_seconds=res.wall_clock_seconds,
            manual_interventions=0))
        print(f"  [STATUS] {status}")
        return converged

    # -- terminal state ----------------------------------------------------

    def designate_submission(self) -> Dict[str, Any]:
        """Choose the final submission, and record *why*.

        This used to be a bare argmax with nothing logged. Two things were wrong
        with that. First, the margin was never weighed against seed noise, so a
        +0.0001 winner was designated as confidently as a +0.01 one even though
        sigma is 0.0008. Second, when nothing beat the baseline the run silently
        exported the organizers' own FM as "the agent's final submission" — a
        result the log should state outright rather than obscure.
        """
        scored = [(nid, n) for nid, n in self.tree.nodes.items()
                  if n.get("primary") is not None]
        scored.sort(key=lambda t: -t[1]["primary"])
        sigma = self.ctx.seed_noise

        decision: Dict[str, Any] = {"sigma": sigma}
        if not scored:
            decision.update(chosen_iteration=None, rationale=(
                "No experiment produced a trustworthy validation score, so there "
                "is nothing to submit."))
            return decision

        best_id, best = scored[0]
        decision["chosen_iteration"] = best_id
        decision["valid_primary"] = best["primary"]
        decision["delta_vs_baseline"] = best["primary"] - self.tree.baseline
        decision["beat_baseline"] = best_id != 0

        if len(scored) > 1:
            runner_id, runner = scored[1]
            margin = best["primary"] - runner["primary"]
            decision.update(runner_up_iteration=runner_id,
                            runner_up_primary=runner["primary"],
                            margin=margin, margin_in_sigma=margin / sigma)
            significant = margin > 3 * sigma
            decision["margin_is_significant"] = significant
            decision["rationale"] = (
                f"Iteration {best_id} scored {best['primary']:.4f} on validation, "
                f"{margin:+.4f} ({margin / sigma:.1f} sigma) ahead of iteration "
                f"{runner_id} at {runner['primary']:.4f}. "
                + ("That margin exceeds 3 sigma, so the ordering is trustworthy."
                   if significant else
                   f"That margin is within {3 * sigma:.4f} (3 sigma of seed noise), "
                   f"so the two are not separated by this evidence; the higher "
                   f"validation score is used as the tie-break."))
        else:
            decision["rationale"] = (
                f"Iteration {best_id} is the only trustworthy result, at "
                f"{best['primary']:.4f}.")

        if best_id == 0:
            decision["rationale"] += (
                " NOTE: this is the reproduced official baseline. No experiment "
                "in this run improved on it, and the submission is therefore the "
                "baseline model rather than an agent-discovered one.")
        return decision

    def build_submission(self) -> Optional[str]:
        """Export the validation-best checkpoint. The only step that reads test rows."""
        decision = self.submission_decision = self.designate_submission()
        best_id = decision.get("chosen_iteration")
        if best_id is None:
            print("[SUBMIT] no successful iteration; nothing to submit")
            return None
        name = self.logger.checkpoint_for(best_id)
        if not name:
            print("[SUBMIT] could not identify the winning checkpoint; skipping export")
            return None

        out = os.path.join("submissions", "kuairand_pure_final.csv")
        print(f"\n[SUBMIT] designated iteration {best_id}, checkpoint {name!r} "
              f"(valid primary {decision['valid_primary']:.4f})")
        print(f"         {decision['rationale']}")

        # Export from the WINNING NODE'S workspace. Its weights live in that
        # node's own checkpoints/ directory, and checkpoint names encode only
        # model and loss — so resolving the name against the shared repository
        # directory can pick up a same-named model from a different run, or from
        # a later losing trial, and ship it under this score.
        ws = self.workspaces.get(best_id)
        env = ws.env() if ws else None
        cwd = ws.root if ws else None
        if ws is None:
            print("         [WARN] the winning node's workspace is unavailable; "
                  "falling back to the repository checkpoint directory")

        cmd = f"{PY} -m pipeline.submit --generate --checkpoint {name} --file {out}"
        if cwd:
            # submit.py writes relative to cwd, so keep the artefact in the repo.
            cmd = (f"{PY} -m pipeline.submit --generate --checkpoint {name} "
                   f"--file {os.path.abspath(out)}")
        res = self.runner.run_command(self._with_data_dir(cmd), env_vars=env, cwd=cwd)
        if res.status == "SUCCESS" or "VALIDATION PASS" in (res.stdout_summary or ""):
            return out
        print(f"[SUBMIT FAILED] {(res.error_traceback or '')[-600:]}")
        return None

    # -- driver ------------------------------------------------------------

    def start_loop(self):
        t0 = time.time()
        print("=" * 74)
        print("RankAgent — autonomous RecSys research agent")
        print(f"budget: {self.max_iterations} iterations | {self.max_wall_clock/3600:.1f}h "
              f"| convergence eps={self.epsilon} N={self.patience}")
        print("=" * 74)

        if self.run_baseline_flag:
            if not self.run_baseline():
                print("[HALT] baseline did not reproduce; refusing to iterate on an unverified harness")
                self._write_summary(t0, "baseline reproduction failed", None)
                return
        else:
            # Development shortcut: skip the ~90s reproduction run and seed the
            # tree with the published constant. Every delta is then measured
            # against a number that was typed in rather than verified, so the log
            # must say so — a reader cannot otherwise tell the two runs apart.
            print(f"[BASELINE] ASSERTED from the published value "
                  f"({BASELINE_VAL_PRIMARY:.4f}) — not reproduced in this run.")
            print("[BASELINE] Deltas below are relative to an unverified reference. "
                  "Drop --skip-baseline for a submittable run.")
            self.baseline_reproduced = False
            self.tree.record_baseline(BASELINE_VAL_PRIMARY, node_id=0)
            self.logger.log_iteration(IterationLogEntry(
                iteration_id=0, node_id=0, parent_node_id=None,
                stage="Baseline (asserted, not reproduced)",
                hypothesis="Seed the search with the organizer's published validation "
                           "primary of 0.6016 without re-running it.",
                rationale="--skip-baseline was set; this run did not verify the harness.",
                target_file="pipeline/train.py", command="(not executed)",
                proposal_source="fallback", status="ACCEPTED",
                metrics=None, delta_over_baseline=0.0, wall_clock_seconds=0.0))

        halt_reason = f"reached the {self.max_iterations}-iteration cap"
        for it in range(1, self.max_iterations + 1):
            elapsed = time.time() - t0
            if elapsed >= self.max_wall_clock:
                halt_reason = f"wall-clock ceiling of {self.max_wall_clock}s reached"
                print(f"\n[HALT] {halt_reason}")
                break
            try:
                if self.run_iteration(it):
                    halt_reason = self.tree.halt_reason or "converged"
                    print(f"\n[CONVERGED] {halt_reason}")
                    break
            except KeyboardInterrupt:
                halt_reason = "interrupted by operator"
                self.ledger.record_interrupt(f"iteration {it}", iteration=it)
                print("\n[HALT] interrupted")
                break
            except Exception as exc:
                # An orchestrator-level bug must not end the run either.
                self.failed_iterations += 1
                print(f"  [ORCHESTRATOR ERROR] {type(exc).__name__}: {exc} — continuing")

            # Stall and divergence guards. Convergence only looks at the
            # best-so-far curve, which is monotone and appended to only on
            # success — so a run where everything crashes can never converge,
            # and a run whose scores steadily degrade reads as converged. Both
            # would otherwise burn the full budget or halt for the wrong reason.
            stall = self._stall_reason()
            if stall:
                halt_reason = stall
                print(f"\n[HALT] {stall}")
                break

        # The submission and the summary are the deliverables. They previously
        # sat outside every guard, so an exception in either — or a second
        # Ctrl-C during the export, which spawns its own subprocess — meant the
        # run produced no RunSummary at all.
        submission = None
        try:
            submission = self.build_submission()
        except KeyboardInterrupt:
            halt_reason = "interrupted by operator during submission export"
            self.ledger.record_interrupt("submission export")
        except Exception as exc:
            print(f"[SUBMIT FAILED] {type(exc).__name__}: {exc}")
        finally:
            self._write_summary(t0, halt_reason, submission)

    def _stall_reason(self) -> Optional[str]:
        """Halt conditions the convergence rule structurally cannot detect."""
        consecutive = self.ctx.consecutive_failures()
        if consecutive >= self.MAX_CONSECUTIVE_FAILURES:
            return (f"stalled: {consecutive} consecutive experiments failed to "
                    f"produce a result. Continuing would spend the remaining "
                    f"budget re-deriving the same failure.")
        used = len([n for n in self.tree.nodes if n != 0])
        if used >= self.max_iterations // 2 and not self.tree.has_result():
            return (f"stalled: {used} iterations used and no experiment has yet "
                    f"produced a trustworthy score.")
        return None

    def _autonomy_report(self) -> Dict[str, Any]:
        """How much of this run the model actually authored.

        A run with no API key and a run driven entirely by the model used to
        produce structurally identical logs — same fields, same
        ``proposal_source: "llm"`` on every iteration — differing only in a token
        total a reader would have to notice was zero. Anyone reading the log to
        judge autonomy deserves to be told directly.
        """
        entries = self.logger.entries
        trials = [e for e in entries if e.get("iteration_id", 0) > 0]
        llm_authored = [e for e in trials if e.get("proposal_source") == "llm"]
        with_code = [e for e in trials if (e.get("code_diff") or "").strip()]
        # Search all entries, not just trials: when nothing beats the baseline
        # the winner IS iteration 0, and reporting its provenance as null hides
        # the most important fact about the run.
        best = next((e for e in entries
                     if e.get("iteration_id") == self.tree.best_node_id), None)
        return {
            "llm_available": self.team.llm_available,
            "iterations_total": len(trials),
            "iterations_llm_authored": len(llm_authored),
            "iterations_from_playbook": len(trials) - len(llm_authored),
            "nodes_with_generated_code": len(with_code),
            "generated_lines_added": sum(
                sum(1 for l in (e.get("code_diff") or "").splitlines()
                    if l.startswith("+") and not l.startswith("+++"))
                for e in with_code),
            "best_node_source": (best or {}).get("proposal_source"),
            "best_node_had_code_change": bool((best or {}).get("code_diff", "").strip()),
            "note": ("This run had no usable API key: every hypothesis came from "
                     "the hard-coded playbook and no code was generated. It "
                     "demonstrates the harness, not autonomy."
                     if not self.team.llm_available else
                     f"{len(llm_authored)}/{len(trials)} iterations were authored "
                     f"by the model; {len(with_code)} applied a code patch."),
        }

    def _write_summary(self, t0: float, halt_reason: str, submission: Optional[str]):
        elapsed = time.time() - t0
        summary = RunSummary(
            run_id=self.logger.run_id,
            best_valid_primary=self.tree.best_primary_score if self.tree.has_result() else None,
            best_delta=self.tree.best_delta if self.tree.has_result() else None,
            best_iteration=self.tree.best_node_id,
            iterations_used=len([n for n in self.tree.nodes if n != 0]),
            iteration_cap=self.max_iterations,
            halt_reason=halt_reason,
            wall_clock_seconds=elapsed,
            baseline_measured=self.baseline_measured,
            baseline_drift=self.baseline_drift,
            submission_decision=self.submission_decision,
            interventions=self.ledger.as_list(),
            autonomy=self._autonomy_report(),
            total_prompt_tokens=self.tokens.prompt_tokens,
            total_completion_tokens=self.tokens.completion_tokens,
            llm_calls=self.tokens.calls,
            manual_interventions=len(self.ledger),
            error_recoveries=self.error_recoveries,
            failed_iterations=self.failed_iterations,
            submission_path=submission,
            baseline_reproduced=self.baseline_reproduced,
            cost_by_agent=self.team.cost_by_agent,
        )
        self.logger.write_summary(summary)
        print("\n" + "=" * 74)
        print(f"[FINISHED] {halt_reason}")
        print(f"  baseline           : {'reproduced this run' if self.baseline_reproduced else 'ASSERTED (not reproduced)'}")
        if self.tree.has_result():
            print(f"  best valid primary : {self.tree.best_primary_score:.4f} "
                  f"(delta {self.tree.best_delta:+.4f} vs {BASELINE_VAL_PRIMARY})")
        print(f"  iterations         : {summary.iterations_used}/{self.max_iterations}")
        print(f"  wall clock         : {elapsed/60:.1f} min")
        print(f"  LLM tokens         : {summary.total_tokens:,d} in {self.tokens.calls} calls")
        for role, c in sorted(self.team.cost_by_agent.items()):
            print(f"    {role:<16s}: {c['prompt'] + c['completion']:>7,d} tokens "
                  f"in {c['calls']} call{'s' if c['calls'] != 1 else ''}")
        print(f"  recoveries/failures: {self.error_recoveries}/{self.failed_iterations}")
        print(f"  code patches       : {summary.autonomy.get('nodes_with_generated_code', 0)}"
              f" of {summary.autonomy.get('iterations_total', 0)} iterations "
              f"(+{summary.autonomy.get('generated_lines_added', 0)} lines)")
        print(f"  llm-authored       : "
              f"{summary.autonomy.get('iterations_llm_authored', 0)}"
              f"/{summary.autonomy.get('iterations_total', 0)} iterations")
        print(f"  manual intervention: {self.ledger.summary_line()}")
        if summary.submission_decision.get("rationale"):
            print(f"  designated         : {summary.submission_decision['rationale']}")
        print(f"  submission         : {submission or 'not produced'}")
        print("=" * 74)


AutonomousOrchestrator = RankAgentOrchestrator


if __name__ == '__main__':
    RankAgentOrchestrator(max_iterations=5).start_loop()
