"""
Autonomous State Machine and FSM Loop for RankAgent.
Implements multi-turn iteration cycles:
INIT -> HYPOTHESIZE -> GENERATE -> RUN -> EVAL -> REFLECT/PRUNE -> HALT
Includes automatic .env API key loading and multi-architecture trial dispatching.
"""
import os
import sys
import time
import json
from typing import Optional, Dict, Any, Tuple

from orchestrator.tree_manager import TreeManager
from orchestrator.schemas import IterationLogEntry, ExecutionResult
from sandbox.runner import ExecutionRunner
from sandbox.debugger import SelfHealingDebugger
from sandbox.logger import RunLogger
from prompts.templates import SYSTEM_PROMPT, HYPOTHESIS_PROMPT
from pipeline.submit import generate_submission, check_submission

def load_dotenv_if_present():
    """Lightweight .env loader without external dependencies."""
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

class RankAgentOrchestrator:
    def __init__(self, data_dir: Optional[str] = None, max_iterations: int = 10, max_wall_clock: int = 21600):
        load_dotenv_if_present()
        self.data_dir = data_dir
        self.max_iterations = max_iterations
        self.max_wall_clock = max_wall_clock
        
        self.tree_manager = TreeManager()
        self.runner = ExecutionRunner(timeout_seconds=900)
        self.debugger = SelfHealingDebugger(self.runner)
        self.logger = RunLogger()
        
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    def run_baseline_sanity_check(self) -> bool:
        """Executes Phase 0 official Factorization Machine baseline reproduction."""
        print("==========================================================================")
        print(">>> PHASE 0: Official Factorization Machine Baseline Sanity Check <<<")
        print("==========================================================================")
        cmd = f"{sys.executable} -m pipeline.train --model fm --epochs 15"
        res = self.runner.run_command(cmd)
        
        if res.status == "SUCCESS" and res.metrics:
            m = res.metrics
            print(f"[SANITY CHECK PASSED] Val GAUC: {m.gauc:.4f} | nDCG@5: {m.ndcg_5:.4f} | Primary: {m.primary_score:.4f}")
            self.tree_manager.best_primary_score = m.primary_score
            return True
        else:
            print(f"[SANITY CHECK FAILED] Error: {res.stderr}")
            return False

    def query_llm_hypothesis(self, iteration_id: int) -> Tuple[Dict[str, Any], int, int]:
        """Queries LLM or falls back to curated RecSys hypothesis roadmap."""
        strategies = [
            {"stage": "Multi-Task Learning", "hypothesis": "Train Multi-Task MMoE on long_view + click + like with embed_dim=32.", "target_file": "pipeline/models.py", "cmd": f"{sys.executable} -m pipeline.train --model mmoe --embed_dim 32 --experts 6 --epochs 10"},
            {"stage": "Ensemble Blending", "hypothesis": "Blend Multi-Task MMoE predictions with LightGBM GBDT via Rank Normalization.", "target_file": "pipeline/train.py", "cmd": f"{sys.executable} -m pipeline.train --model ensemble --weight_ensemble 0.65"},
            {"stage": "Hyperparameter Tuning", "hypothesis": "Tune MMoE learning rate to 0.0005 with 8 experts for higher capacity.", "target_file": "pipeline/models.py", "cmd": f"{sys.executable} -m pipeline.train --model mmoe --embed_dim 32 --experts 8 --lr 0.0005 --epochs 12"},
            {"stage": "Multi-Domain Factorization", "hypothesis": "Train DeepFM with 13 CWM metadata domains and dense projection.", "target_file": "pipeline/models.py", "cmd": f"{sys.executable} -m pipeline.train --model deepfm --cwm --epochs 10"},
            {"stage": "Sequential Attention", "hypothesis": "Train Deep Interest Network (DIN) with Target-Attention pooling over user past watch history.", "target_file": "pipeline/models.py", "cmd": f"{sys.executable} -m pipeline.train --model din --embed_dim 32 --epochs 8"}
        ]
        
        if not self.openai_api_key or self.openai_api_key.startswith("your-") or len(self.openai_api_key) < 10:
            strat = strategies[(iteration_id - 1) % len(strategies)]
            return {
                "stage": strat["stage"],
                "hypothesis": strat["hypothesis"],
                "target_file": strat["target_file"],
                "cmd": strat["cmd"]
            }, 0, 0

        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            prompt = HYPOTHESIS_PROMPT.format(
                iteration_id=iteration_id,
                best_score=self.tree_manager.best_primary_score,
                delta=self.tree_manager.best_primary_score - 0.6016,
                history_summary=self.tree_manager.get_history_summary(),
                target_file="pipeline/models.py"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            
            data = json.loads(response.choices[0].message.content)
            
            # Map execution command
            if "execution_command" in data and "pipeline.train" in data["execution_command"]:
                raw_cmd = data["execution_command"]
                # Ensure correct python executable prefix
                if raw_cmd.startswith("python "):
                    raw_cmd = f"{sys.executable} " + raw_cmd[7:]
                data["cmd"] = raw_cmd
            else:
                strat = strategies[(iteration_id - 1) % len(strategies)]
                data["cmd"] = strat["cmd"]
                
            return data, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"  [WARN] LLM API Call Exception: {e}. Using deterministic Strategy Bank.")
            strat = strategies[(iteration_id - 1) % len(strategies)]
            return {
                "stage": strat["stage"],
                "hypothesis": strat["hypothesis"],
                "target_file": strat["target_file"],
                "cmd": strat["cmd"]
            }, 0, 0

    def start_loop(self):
        """Starts the autonomous exploration state machine."""
        start_wall_clock = time.time()
        
        # 1. Phase 0 Sanity Check
        self.run_baseline_sanity_check()
        
        # 2. Iteration Loop (1 to max_iterations)
        for iter_id in range(1, self.max_iterations + 1):
            elapsed = time.time() - start_wall_clock
            if elapsed >= self.max_wall_clock:
                print(f"[HALT] Reached 6-hour wall-clock limit ({elapsed:.1f}s). Terminating run.")
                break

            print(f"\n==========================================================================")
            print(f">>> [ITERATION {iter_id}/{self.max_iterations}] Proposing next hypothesis... <<<")
            print(f"==========================================================================")
            proposal, p_tokens, c_tokens = self.query_llm_hypothesis(iter_id)
            print(f"  Stage: {proposal['stage']}")
            print(f"  Hypothesis: {proposal['hypothesis']}")
            print(f"  Target File: {proposal['target_file']}")
            if p_tokens > 0:
                print(f"  [LLM TOKENS] Prompt: {p_tokens} | Completion: {c_tokens} (Total: {p_tokens + c_tokens})")
            
            # Execute Trial in Sandbox
            print(f"  Running trial in sandbox: {proposal.get('cmd')}")
            exec_res = self.runner.run_command(proposal.get("cmd", f"{sys.executable} -m pipeline.train --model fm"))
            
            if exec_res.status == "SUCCESS" and exec_res.metrics:
                m = exec_res.metrics
                print(f"  [RESULT] Val GAUC: {m.gauc:.4f} | nDCG@5: {m.ndcg_5:.4f} | Primary: {m.primary_score:.4f} (Delta Baseline: {m.delta_from_baseline:+.4f})")
                
                is_converged = self.tree_manager.add_node(
                    node_id=iter_id,
                    parent_id=self.tree_manager.best_node_id,
                    hypothesis=proposal['hypothesis'],
                    target_file=proposal['target_file'],
                    metrics=m
                )
                
                status_verdict = "ACCEPTED" if m.primary_score > self.tree_manager.best_primary_score - 1e-5 else "REJECTED"
                if m.primary_score > self.tree_manager.best_primary_score:
                    status_verdict = "ACCEPTED"
                print(f"  [STATUS] {status_verdict}")
                
                self.logger.log_iteration(IterationLogEntry(
                    iteration_id=iter_id,
                    parent_node_id=self.tree_manager.best_node_id,
                    node_id=iter_id,
                    stage=proposal['stage'],
                    hypothesis=proposal['hypothesis'],
                    target_file=proposal['target_file'],
                    code_diff=f"Executed: {proposal.get('cmd')}",
                    status=status_verdict,
                    metrics=m.model_dump(),
                    delta_over_baseline=m.delta_from_baseline,
                    wall_clock_seconds=exec_res.wall_clock_seconds,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    manual_interventions=0
                ))
                
                if is_converged:
                    print(f"\n[CONVERGENCE REACHED] Primary score delta ≤ 0.002 across 3 consecutive iterations.")
                    print(f"Final Global Best Score: {self.tree_manager.best_primary_score:.4f}")
                    break
            else:
                print(f"  [RUNTIME ERROR] Traceback detected. Triggering Self-Healing Debugger...")
                fixed = self.debugger.attempt_repair(proposal.get("cmd", ""), exec_res.stderr)
                if fixed:
                    print("  [DEBUGGER] Trial repaired successfully.")
                else:
                    print("  [DEBUGGER] Repair limit exceeded. Pruning branch.")

        total_time = time.time() - start_wall_clock
        print(f"\n==========================================================================")
        print(f"[EXECUTION FINISHED] Completed in {total_time:.1f}s.")
        print(f"Global Best Validation Primary Score: {self.tree_manager.best_primary_score:.4f}")
        print(f"Telemetry Logs Saved: logs/run_summary.json & logs/run_log.md")
        print(f"==========================================================================")

AutonomousOrchestrator = RankAgentOrchestrator

if __name__ == '__main__':
    orchestrator = RankAgentOrchestrator(max_iterations=5)
    orchestrator.start_loop()
