"""
Finite State Machine and Master Orchestrator for RankAgent.
"""
import os
import sys
import json
import time
import argparse
from typing import Optional

from orchestrator.schemas import IterationLogEntry, MetricResult, ExecutionResult
from orchestrator.tree_manager import TreeManager
from sandbox.runner import ExecutionRunner
from sandbox.debugger import SelfHealingDebugger
from sandbox.logger import RunLogger
from prompts.templates import SYSTEM_PROMPT, HYPOTHESIS_PROMPT

def load_dotenv_if_present():
    """Loads key-value pairs from .env if present."""
    for path in [".env", ".env.local"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and not os.environ.get(k):
                            os.environ[k] = v

class RankAgentOrchestrator:
    def __init__(self, data_dir: Optional[str] = None, max_iterations: int = 50, max_wall_clock: int = 21600):
        load_dotenv_if_present()
        self.data_dir = data_dir
        self.max_iterations = max_iterations
        self.max_wall_clock = max_wall_clock
        self.tree_manager = TreeManager(max_iterations=max_iterations)
        self.runner = ExecutionRunner(timeout_seconds=900)
        self.debugger = SelfHealingDebugger(max_retries=3)
        self.logger = RunLogger(log_dir="logs", run_id=f"rankagent-{int(time.time())}")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    def run_baseline_sanity_check(self) -> ExecutionResult:
        """Phase 0: Stand up and verify official baseline."""
        print("\n========================================================")
        print(" [PHASE 0] Reproducing Official Baseline (NumPy FM)... ")
        print("========================================================")
        cmd = f"{sys.executable} -m pipeline.train --model fm"
        if self.data_dir:
            cmd += f" --data_dir {self.data_dir}"
            
        result = self.runner.run_command(cmd)
        if result.status == "SUCCESS" and result.metrics:
            print(f"==> Baseline Reproduced Successfully! Val Primary: {result.metrics.primary_score:.4f} (GAUC: {result.metrics.gauc:.4f}, nDCG@5: {result.metrics.ndcg_5:.4f})")
            self.tree_manager.best_primary_score = result.metrics.primary_score
            self.logger.log_iteration(IterationLogEntry(
                iteration_id=0,
                parent_node_id=None,
                node_id=0,
                stage="Baseline Reproduction",
                hypothesis="Stand up official Factorization Machine baseline on KuaiRand-Pure.",
                target_file="pipeline/models.py",
                code_diff="Baseline reference code.",
                status="ACCEPTED",
                metrics=result.metrics.model_dump(),
                delta_over_baseline=0.0,
                wall_clock_seconds=result.wall_clock_seconds
            ))
        else:
            print(f"[WARNING] Baseline check encountered issue: {result.error_traceback}")
        return result

    def query_llm_hypothesis(self, iteration_id: int) -> dict:
        """Queries LLM for next hypothesis or falls back to domain strategy bank."""
        if not self.openai_api_key:
            # Domain Strategy Bank Fallback if API key not yet set in environment
            strategies = [
                {"stage": "Feature Engineering", "hypothesis": "Expand 5 fields to CWM 13 user/video domains.", "target_file": "pipeline/train.py", "cmd": f"{sys.executable} -m pipeline.train --model fm --cwm"},
                {"stage": "Architecture", "hypothesis": "Train DeepFM with 2nd order feature factor embeddings & Deep MLP.", "target_file": "pipeline/models.py", "cmd": f"{sys.executable} -m pipeline.train --model deepfm --cwm"},
                {"stage": "Hyperparameter Tuning", "hypothesis": "Tune learning rate to 0.0005 with weight decay on DeepFM.", "target_file": "pipeline/train.py", "cmd": f"{sys.executable} -m pipeline.train --model deepfm --cwm --lr 0.0005 --epochs 30"}
            ]
            strat = strategies[(iteration_id - 1) % len(strategies)]
            return {
                "stage": strat["stage"],
                "hypothesis": strat["hypothesis"],
                "target_file": strat["target_file"],
                "cmd": strat["cmd"]
            }

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
        data = json.loads(response.choices[0].message.content)
        data["cmd"] = f"{sys.executable} -m pipeline.train --model deepfm --cwm"
        return data

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

            print(f"\n>>> [ITERATION {iter_id}/{self.max_iterations}] Proposing next hypothesis... <<<")
            proposal = self.query_llm_hypothesis(iter_id)
            print(f"  Stage: {proposal['stage']}")
            print(f"  Hypothesis: {proposal['hypothesis']}")
            print(f"  Target File: {proposal['target_file']}")
            
            # Execute Trial in Sandbox
            print("  Running trial in sandbox...")
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
                
                self.logger.log_iteration(IterationLogEntry(
                    iteration_id=iter_id,
                    parent_node_id=self.tree_manager.best_node_id,
                    node_id=iter_id,
                    stage=proposal['stage'],
                    hypothesis=proposal['hypothesis'],
                    target_file=proposal['target_file'],
                    code_diff=f"Executed: {proposal.get('cmd')}",
                    status="ACCEPTED" if m.primary_score > 0.6016 else "REJECTED",
                    metrics=m.model_dump(),
                    delta_over_baseline=m.delta_from_baseline,
                    wall_clock_seconds=exec_res.wall_clock_seconds
                ))
                
                if is_converged:
                    print(f"\n[CONVERGENCE REACHED] Primary score plateaued over consecutive trials. Gracefully halting.")
                    break
            else:
                print(f"  [ERROR] Trial encountered error: {exec_res.error_traceback[:200]}")
                self.tree_manager.add_node(iter_id, self.tree_manager.best_node_id, proposal['hypothesis'], proposal['target_file'], None)

        print("\n========================================================")
        print(" [COMPLETE] RankAgent Run Finished!")
        print(f" Best Validation Primary Score: {self.tree_manager.best_primary_score:.4f}")
        print(f" Total Log Entries Saved to: logs/run_summary.json & logs/run_log.md")
        print("========================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, help='KuaiRand data directory')
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--max_wall_clock', type=int, default=21600)
    args = parser.parse_args()

    orchestrator = RankAgentOrchestrator(
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        max_wall_clock=args.max_wall_clock
    )
    orchestrator.start_loop()
