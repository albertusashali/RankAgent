"""
Pydantic schemas and data contracts for RankAgent modules.
"""
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field

class MetricResult(BaseModel):
    gauc: float = Field(..., description="Group AUC weighted by positive count")
    ndcg_5: float = Field(..., description="Normalized Discounted Cumulative Gain @ 5")
    primary_score: float = Field(..., description="mean(GAUC, nDCG@5)")
    delta_from_baseline: float = Field(0.0, description="Improvement over official FM baseline")
    is_converged: bool = Field(False, description="Whether convergence criterion is satisfied")
    raw_stdout: Optional[str] = None

class ExecutionResult(BaseModel):
    status: Literal["SUCCESS", "RUNTIME_ERROR", "TIMEOUT", "SYNTAX_ERROR", "CONVERGENCE_HALT"]
    metrics: Optional[MetricResult] = None
    error_traceback: Optional[str] = None
    stdout_summary: str = ""
    wall_clock_seconds: float = 0.0
    command_executed: str = ""

class IterationLogEntry(BaseModel):
    iteration_id: int
    parent_node_id: Optional[int] = None
    node_id: int
    stage: str
    hypothesis: str
    target_file: str
    code_diff: str
    status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
    metrics: Optional[Dict[str, Any]] = None
    delta_over_baseline: Optional[float] = None
    error_recovery: Optional[Dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_seconds: float = 0.0
    manual_interventions: int = 0

class HypothesisProposal(BaseModel):
    stage: Literal["Feature Engineering", "Architecture", "Multi-Task", "Loss Function", "Ensembling", "Hyperparameter Tuning"]
    hypothesis: str
    target_file: str
    rationale: str
    code_changes: str
