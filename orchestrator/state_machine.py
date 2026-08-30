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

from orchestrator.schemas import (ExecutionResult, HypothesisProposal,
                                  IterationLogEntry, RunSummary, TokenUsage)
from orchestrator.tree_manager import BASELINE_VAL_PRIMARY, TreeManager
from sandbox.debugger import SelfHealingDebugger
from sandbox.logger import RunLogger
from sandbox.runner import ExecutionRunner

PY = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def load_dotenv_if_present(path: Optional[str] = None):
    """Minimal .env reader — avoids a dependency for three lines of parsing."""
    path = path or os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8-sig") as fh:
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
    {"stage": "Feature Engineering",
     "hypothesis": "Fuse strictly causal affinity, completion-history, recency and momentum "
                   "features into DeepFM so feature recipes affect a strong identity model "
                   "instead of being isolated in a weaker tree-only ablation.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
                "--feature_profile full --select_features --epochs 12"},

    {"stage": "Noise Reduction",
     "hypothesis": "Reduce optimization variance without changing labels or deleting "
                   "interactions: train a second seed of the validated causal DeepFM and "
                   "let validation-only rank blending average uncorrelated errors.",
     "target_file": "pipeline/train.py",
     "command": f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
                "--feature_profile full --select_features --seed 1 --epochs 12"},
    {"stage": "Loss Function",
     "hypothesis": "Optimize both halves of the primary metric with a hybrid objective: "
                   "BPR supplies GAUC-oriented pair ordering while delta-nDCG weights errors "
                   "near rank 5. Preserve every power-user impression instead of capping groups.",
     "target_file": "pipeline/models.py",
     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
                "--auc_weight 0.5 --max_group_rows 0 --epochs 15"},

    {"stage": "Causal Debiasing",
     "hypothesis": "Use clipped self-normalized density ratios learned from KuaiRand's "
                   "random-exposure log to reduce logging-policy bias while bounding variance.",
     "target_file": "pipeline/propensity.py",
     "command": f"{PY} -m pipeline.train --model fm_torch --loss hybrid "
                "--auc_weight 0.5 --propensity snips --propensity_clip 10 --epochs 15"},

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

    {"stage": "Feature Engineering",
     "hypothesis": "Add strictly historical watch-completion, repeat-affinity and recency "
                   "features. Candidate-row play time is forbidden because it leaks long_view; "
                   "expanding-window aggregates preserve its useful history safely.",
     "target_file": "pipeline/features.py",
     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
                "--feature_profile full --select_features --trees 400"},

    {"stage": "Feature Engineering",
     "hypothesis": "Ablate interaction crosses against core popularity and history features "
                   "to measure whether user-video, user-author, duration and tab affinities "
                   "add within-user ranking signal.",
     "target_file": "pipeline/features.py",
     "command": f"{PY} -m pipeline.train --model lgb --objective lambdarank "
                "--feature_profile core --select_features --trees 400"},

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

# Put reliable controls before speculative objectives. The convergence rule can
# stop after four trials, so ordering is part of the research design: previously
# two weak experiments consumed half the useful horizon before the proven
# listwise control ran, and the feature ranker was never reached.
def _strategy_priority(strategy: Dict[str, str]) -> int:
    cmd = strategy["command"]
    if "--model fm_torch --loss listwise" in cmd and "--lr" not in cmd:
        return 0
    if "--model deepfm_dense" in cmd:
        return 1 if "--seed 1" not in cmd else 2
    if "--feature_profile full" in cmd:
        return 5
    if "--model mmoe --loss listwise --experts 4" in cmd:
        return 2
    if "--model deepfm" in cmd:
        return 3
    if "--model din" in cmd:
        return 4
    if "--loss hybrid" in cmd and "--propensity" not in cmd:
        return 7
    if "--propensity" in cmd:
        return 99
    return 10


STRATEGY_BANK.sort(key=_strategy_priority)


class RankAgentOrchestrator:
    def __init__(self, data_dir: Optional[str] = None,
                 max_iterations: Optional[int] = None,
                 max_wall_clock: Optional[int] = None, run_id: Optional[str] = None,
                 run_baseline: bool = False):
        load_dotenv_if_present()
        agent_cfg = load_yaml("agent_config.yaml")
        bench_cfg = load_yaml("benchmark_kuairand.yaml")
        conv = (bench_cfg.get("convergence") or {})

        self.data_dir = data_dir
        self.run_baseline_flag = run_baseline
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
            timeout_seconds=int(sandbox_cfg.get("timeout_seconds_per_iteration", 1800)),
            cwd=PROJECT_ROOT)
        self.debugger = SelfHealingDebugger(
            max_retries=int(dbg_cfg.get("max_self_healing_attempts", 3)),
            llm_repair=self._llm_repair)
        self.logger = RunLogger(log_dir=os.path.join(PROJECT_ROOT, "logs"), run_id=run_id)

        self.tokens = TokenUsage()
        self.manual_interventions = 0
        self.error_recoveries = 0
        self.failed_iterations = 0
        self._used_commands: set = set()
        self._used_recipe_ids: set = set()
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
        cmd = self._with_data_dir(f"{PY} -m pipeline.train --model fm")
        res = self.runner.run_command(cmd)
        if res.status == "SUCCESS" and res.metrics:
            m = res.metrics
            drift = abs(m.primary_score - BASELINE_VAL_PRIMARY)
            print(f"[BASELINE] valid GAUC {m.gauc:.4f} | nDCG@5 {m.ndcg_5:.4f} | "
                  f"primary {m.primary_score:.4f} (published {BASELINE_VAL_PRIMARY:.4f}, "
                  f"drift {drift:+.4f})")
            if drift > 0.005:
                print(f"[WARN] baseline drift exceeds 0.005 — investigate before trusting deltas")
            self.tree.record_baseline(m.primary_score, node_id=0)
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
                if strat["stage"] == "Feature Engineering":
                    from pipeline.feature_recipes import deterministic_recipes
                    for recipe in deterministic_recipes():
                        if recipe.recipe_id not in self._used_recipe_ids:
                            path = self._save_recipe(recipe, iteration_id)
                            command = (f"{PY} -m pipeline.train --model deepfm_dense "
                                       f"--loss pointwise --feature_recipe {path} --epochs 12")
                            return HypothesisProposal(
                                source="fallback", stage="Feature Engineering",
                                hypothesis=(f"Test recipe {recipe.name}: profile={recipe.base_profile}, "
                                            f"item smoothing={recipe.item_smoothing}, cross smoothing="
                                            f"{recipe.cross_smoothing}, exclusions={recipe.exclude_features}."),
                                rationale="deterministic bounded recipe mutation",
                                target_file="pipeline/features.py", command=command,
                                feature_recipe=recipe.model_dump(), recipe_id=recipe.recipe_id)
                return HypothesisProposal(source="fallback", rationale="deterministic plan",
                                          **strat)
        strat = STRATEGY_BANK[(iteration_id - 1) % len(STRATEGY_BANK)]
        return HypothesisProposal(source="fallback",
                                  rationale="plan exhausted; repeating with a new seed",
                                  **{**strat, "command": f"{strat['command']} --seed {iteration_id}"})

    def _save_recipe(self, recipe, iteration_id: int) -> str:
        from pipeline.feature_agent import FeatureEngineeringAgent
        audit = FeatureEngineeringAgent().recipe_audit(recipe)
        if audit['status'] != 'PASS':
            raise ValueError(f"feature recipe failed audit: {audit}")
        path = os.path.join("experiments", "recipes",
                            f"iteration_{iteration_id:02d}_{recipe.recipe_id}.json")
        recipe.save(path)
        self._used_recipe_ids.add(recipe.recipe_id)
        return path

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
            history_summary=self.tree.get_history_summary() + self._feature_feedback_summary(),
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
            stage = str(data.get("stage", "Unspecified"))
            cmd = str(data["execution_command"]).strip()
            if "pipeline.train" not in cmd:
                raise ValueError("command does not invoke pipeline.train")
            if cmd.startswith("python "):
                cmd = f"{PY} " + cmd[len("python "):]
            recipe_payload = data.get("feature_recipe")
            recipe_id = None
            if stage == "Feature Engineering" and recipe_payload is None:
                raise ValueError("Feature Engineering proposal omitted feature_recipe")
            if stage == "Feature Engineering":
                from pipeline.feature_recipes import FeatureRecipe
                recipe = FeatureRecipe.model_validate(recipe_payload)
                if recipe.recipe_id in self._used_recipe_ids:
                    raise ValueError(f"feature recipe {recipe.recipe_id} was already tested")
                path = self._save_recipe(recipe, iteration_id)
                # Compile every AI recipe into the fused identity + causal-feature
                # ranker so the mutation can improve the strongest model family.
                cmd = (f"{PY} -m pipeline.train --model deepfm_dense --loss pointwise "
                       f"--feature_recipe {path} --epochs 12")
                recipe_payload, recipe_id = recipe.model_dump(), recipe.recipe_id
            return HypothesisProposal(
                stage=stage,
                hypothesis=str(data["hypothesis"]),
                rationale=str(data.get("rationale", "")),
                target_file=str(data.get("target_file", "pipeline/train.py")),
                command=cmd, source="llm",
                feature_recipe=recipe_payload if stage == "Feature Engineering" else None,
                recipe_id=recipe_id)
        except Exception as exc:
            print(f"  [WARN] LLM proposal unusable ({exc}); using the deterministic plan.")
            return self._fallback_proposal(iteration_id)

    def _feature_feedback_summary(self) -> str:
        """Feed top feature gains and tested recipe IDs into the next AI turn."""
        blocks = []
        for checkpoint in dict.fromkeys(self.logger._checkpoints.values()):
            path = os.path.join("checkpoints", f"{checkpoint}.importance.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                top = payload.get("features", [])[:5]
                rendered = ", ".join(
                    f"{f['name']}={100*f['gain_fraction']:.1f}%" for f in top)
                blocks.append(f"\n- Feature feedback {payload.get('recipe_id')}: {rendered}")
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return ("\n\nFeature-recipe feedback:" + "".join(blocks)) if blocks else ""

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

    def run_iteration(self, iteration_id: int) -> bool:
        """Execute one hypothesis. Returns whether the run should halt."""
        print(f"\n{'=' * 74}\n>>> ITERATION {iteration_id}/{self.max_iterations}\n{'=' * 74}")
        proposal = self.propose(iteration_id)
        command = self._with_data_dir(proposal.command)
        self._used_commands.add(proposal.command)
        if proposal.recipe_id:
            self._used_recipe_ids.add(proposal.recipe_id)

        print(f"  Stage      : {proposal.stage}  [{proposal.source}]")
        print(f"  Hypothesis : {proposal.hypothesis}")
        print(f"  Command    : {command}")

        res = self.runner.run_command(command)
        recovery: Optional[dict] = None
        status = "REJECTED"

        if res.failed:
            print(f"  [FAILURE:{res.status}] invoking the self-healing debugger")
            outcome = self.debugger.attempt_repair(
                command, res.error_traceback or "", self.runner.run_command)
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

        parent = self.tree.select_parent()
        converged = self.tree.add_node(iteration_id, parent, proposal.hypothesis,
                                       proposal.target_file, metrics)
        if metrics is not None and status != "ERROR_RECOVERED":
            status = self.tree.nodes[iteration_id]["status"]
        elif metrics is None:
            status = "FAILED"

        self.logger.log_iteration(IterationLogEntry(
            iteration_id=iteration_id, node_id=iteration_id, parent_node_id=parent,
            stage=proposal.stage, hypothesis=proposal.hypothesis,
            rationale=proposal.rationale, target_file=proposal.target_file,
            command=command, proposal_source=proposal.source,
            feature_recipe=proposal.feature_recipe, recipe_id=proposal.recipe_id,
            code_diff=self.logger.capture_diff(),
            status=status, metrics=metrics.model_dump() if metrics else None,
            delta_over_baseline=metrics.delta_from_baseline if metrics else None,
            error_recovery=recovery,
            prompt_tokens=self.tokens.prompt_tokens, completion_tokens=self.tokens.completion_tokens,
            wall_clock_seconds=res.wall_clock_seconds,
            manual_interventions=0))
        print(f"  [STATUS] {status}")
        return converged

    # -- terminal state ----------------------------------------------------

    def build_submission(self) -> Optional[str]:
        """Select a safe validation blend, diagnose it, then export hidden-test scores."""
        best_id = self.tree.best_node_id
        if best_id is None:
            print("[SUBMIT] no successful iteration; nothing to submit")
            return None
        name = self.logger.checkpoint_for(best_id)
        if not name:
            print("[SUBMIT] could not identify the winning checkpoint; skipping export")
            return None
        selected = [name]
        weight = None

        # Every trainer caches validation predictions in original row order.
        # Search a two-model rank blend, but accept it only if official public
        # validation improves. Weak feature models therefore cannot degrade the
        # final submission; they may only contribute a small decorrelated signal.
        best_cache = os.path.join("checkpoints", f"{name}.valid.npy")
        if os.path.exists(best_cache):
            try:
                from pipeline.data import load_kuairand
                from pipeline.submit import blend_weight_on_valid
                splits = load_kuairand(self.data_dir)
                blend_primary = self.tree.best_primary_score
                for candidate in dict.fromkeys(self.logger._checkpoints.values()):
                    if candidate == name:
                        continue
                    if not os.path.exists(os.path.join("checkpoints", f"{candidate}.valid.npy")):
                        continue
                    w, primary = blend_weight_on_valid([name, candidate], splits, grid=41)
                    if primary > blend_primary + 1e-5:
                        selected, weight, blend_primary = [name, candidate], w, primary
                if len(selected) == 2:
                    print(f"[ENSEMBLE] accepted {selected} at weight {weight:.3f}: "
                          f"valid primary {blend_primary:.4f}")
                    self.tree.best_primary_score = blend_primary
            except Exception as exc:
                print(f"[ENSEMBLE] skipped safely: {type(exc).__name__}: {exc}")

        # Produce test predictions for selected components in isolated processes.
        for checkpoint in selected:
            export = f"{PY} -m pipeline.submit --export --checkpoint {checkpoint} --split test"
            result = self.runner.run_command(self._with_data_dir(export), allow_no_metrics=True)
            if result.status != "SUCCESS":
                print(f"[SUBMIT FAILED] could not export {checkpoint}: "
                      f"{(result.error_traceback or '')[-500:]}")
                return None

        # Diagnostics are automatic and validation-only. For a blend, each
        # component remains inspectable and the blend score is printed above.
        diag = f"{PY} -m pipeline.diagnostics --checkpoint {name}"
        self.runner.run_command(self._with_data_dir(diag), allow_no_metrics=True)

        out = os.path.join("submissions", "kuairand_pure_final.csv")
        print(f"\n[SUBMIT] exporting validation-selected {selected} "
              f"(valid primary {self.tree.best_primary_score:.4f})")
        ckpts = " ".join(selected)
        weight_arg = f" --weight {weight}" if weight is not None else ""
        cmd = (f"{PY} -m pipeline.submit --generate --checkpoint {ckpts}{weight_arg} --file {out}")
        res = self.runner.run_command(self._with_data_dir(cmd))
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

        # Full real-data outcome-mutation audit, launched in isolation so the
        # orchestrator remains small and `python main.py` is the only entrypoint.
        audit_path = os.path.join("logs", "feature_audit.json")
        audit_cmd = f"{PY} -m pipeline.feature_agent --dynamic --output {audit_path}"
        audit_result = self.runner.run_command(self._with_data_dir(audit_cmd),
                                               allow_no_metrics=True)
        try:
            with open(audit_path, encoding="utf-8") as fh:
                feature_audit = json.load(fh)
        except (OSError, json.JSONDecodeError):
            feature_audit = {"status": "FAIL"}
        print(f"[FEATURE AUDIT] {feature_audit['status']} -> {audit_path}")
        if audit_result.status != "SUCCESS" or feature_audit['status'] != 'PASS':
            print("[HALT] feature manifest/leakage governance failed")
            self._write_summary(t0, "feature audit failed", None)
            return

        client = self._client_or_none()
        if client is None:
            print("[AI] no API key detected; using deterministic adaptive feature recipes")
        else:
            print(f"[AI] {client[0]} connected; validation and feature-importance feedback enabled")

        if self.run_baseline_flag:
            if not self.run_baseline():
                print("[HALT] baseline did not reproduce; refusing to iterate on an unverified harness")
                self._write_summary(t0, "baseline reproduction failed", None)
                return
        else:
            print(f"[BASELINE] Initialized with published benchmark valid primary: {BASELINE_VAL_PRIMARY:.4f}")
            self.tree.record_baseline(BASELINE_VAL_PRIMARY, node_id=0)

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
                self.manual_interventions += 1
                print("\n[HALT] interrupted")
                break
            except Exception as exc:
                # An orchestrator-level bug must not end the run either.
                self.failed_iterations += 1
                print(f"  [ORCHESTRATOR ERROR] {type(exc).__name__}: {exc} — continuing")

        submission = self.build_submission()
        self._write_summary(t0, halt_reason, submission)

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
            total_prompt_tokens=self.tokens.prompt_tokens,
            total_completion_tokens=self.tokens.completion_tokens,
            llm_calls=self.tokens.calls,
            manual_interventions=self.manual_interventions,
            error_recoveries=self.error_recoveries,
            failed_iterations=self.failed_iterations,
            submission_path=submission,
        )
        self.logger.write_summary(summary)
        print("\n" + "=" * 74)
        print(f"[FINISHED] {halt_reason}")
        if self.tree.has_result():
            print(f"  best valid primary : {self.tree.best_primary_score:.4f} "
                  f"(delta {self.tree.best_delta:+.4f} vs {BASELINE_VAL_PRIMARY})")
        print(f"  iterations         : {summary.iterations_used}/{self.max_iterations}")
        print(f"  wall clock         : {elapsed/60:.1f} min")
        print(f"  LLM tokens         : {summary.total_tokens:,d} in {self.tokens.calls} calls")
        print(f"  recoveries/failures: {self.error_recoveries}/{self.failed_iterations}")
        print(f"  manual intervention: {self.manual_interventions}")
        print(f"  submission         : {submission or 'not produced'}")
        print("=" * 74)


AutonomousOrchestrator = RankAgentOrchestrator


if __name__ == '__main__':
    RankAgentOrchestrator(max_iterations=5).start_loop()
