"""
Tree Search and Exploration Manager with robust convergence tracking and backtracking.
"""
from typing import Dict, List, Optional, Any
from orchestrator.schemas import MetricResult

BASELINE_VAL_PRIMARY = 0.6016

class TreeManager:
    def __init__(self, epsilon: float = 0.002, n_convergence: int = 3, max_iterations: int = 50):
        self.epsilon = epsilon
        self.n_convergence = n_convergence
        self.max_iterations = max_iterations
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.best_primary_score = BASELINE_VAL_PRIMARY
        self.best_node_id: Optional[int] = None
        self.stagnant_counter = 0

    def add_node(self, node_id: int, parent_id: Optional[int], hypothesis: str, target_file: str, metrics: Optional[MetricResult]) -> bool:
        """
        Adds node to exploration tree and checks convergence against global best frontier.
        Returns: is_converged (bool)
        """
        if metrics is None:
            self.nodes[node_id] = {
                "parent_id": parent_id,
                "hypothesis": hypothesis,
                "target_file": target_file,
                "metrics": None,
                "status": "FAILED"
            }
            return False

        improvement = metrics.primary_score - self.best_primary_score
        
        if improvement > self.epsilon:
            self.best_primary_score = metrics.primary_score
            self.best_node_id = node_id
            self.stagnant_counter = 0
            is_converged = False
        else:
            self.stagnant_counter += 1
            is_converged = (self.stagnant_counter >= self.n_convergence)

        self.nodes[node_id] = {
            "parent_id": parent_id,
            "hypothesis": hypothesis,
            "target_file": target_file,
            "metrics": metrics,
            "is_best": (self.best_node_id == node_id),
            "status": "ACCEPTED" if improvement > 0 else "REJECTED"
        }
        
        if node_id >= self.max_iterations:
            is_converged = True

        return is_converged

    def get_history_summary(self, max_items: int = 5) -> str:
        """Summarizes recent trials for LLM context injection."""
        summary = []
        recent_ids = sorted(self.nodes.keys())[-max_items:]
        for nid in recent_ids:
            n = self.nodes[nid]
            m = n.get("metrics")
            if m:
                summary.append(f"- Iter {nid}: {n['hypothesis']} -> Primary: {m.primary_score:.4f} (Delta {m.delta_from_baseline:+.4f}) [{n['status']}]")
            else:
                summary.append(f"- Iter {nid}: {n['hypothesis']} -> [FAILED / ERROR]")
        return "\n".join(summary) if summary else "No previous iterations."
