"""
Telemetry and Run Logger for serialized JSON schemas and Markdown logs.
"""
import os
import json
import time
from typing import Dict, Any, List
from orchestrator.schemas import IterationLogEntry, MetricResult

class RunLogger:
    def __init__(self, log_dir: str = "logs", run_id: str = "rankagent-run"):
        self.log_dir = log_dir
        self.run_id = run_id
        os.makedirs(log_dir, exist_ok=True)
        self.json_path = os.path.join(log_dir, "run_summary.json")
        self.md_path = os.path.join(log_dir, "run_log.md")
        self.entries: List[Dict[str, Any]] = []
        self._init_md_log()

    def _init_md_log(self):
        if not os.path.exists(self.md_path):
            with open(self.md_path, "w", encoding="utf-8") as f:
                f.write(f"# RankAgent Experiment Run Log\n- **Run ID**: `{self.run_id}`\n\n---\n\n")

    def log_iteration(self, entry: IterationLogEntry):
        self.entries.append(entry.model_dump())
        
        # 1. Update JSON log
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "total_iterations": len(self.entries),
                "iterations": self.entries
            }, f, indent=2)

        # 2. Append to Markdown journal
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"### Iteration {entry.iteration_id}: {entry.stage}\n")
            f.write(f"* **Status**: `{entry.status}`\n")
            f.write(f"* **Target File**: `{entry.target_file}`\n")
            f.write(f"* **Hypothesis**: {entry.hypothesis}\n")
            if entry.metrics:
                f.write(f"* **Metrics**: GAUC: {entry.metrics.get('gauc', 0):.4f} | nDCG@5: {entry.metrics.get('ndcg_5', 0):.4f} | Primary: {entry.metrics.get('primary_score', 0):.4f}\n")
            f.write(f"* **Telemetry**: {entry.wall_clock_seconds:.1f}s | Tokens: {entry.prompt_tokens + entry.completion_tokens}\n\n---\n\n")

