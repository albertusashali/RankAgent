"""Data contracts shared by the orchestrator, sandbox and logger."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Validation metrics for one trial. Hidden-test metrics never appear here."""
    gauc: float = Field(..., description="Group AUC, weighted by positive count")
    ndcg_5: float = Field(..., description="nDCG@5 over each user's impressions")
    primary_score: float = Field(..., description="mean(GAUC, nDCG@5)")
    delta_from_baseline: float = Field(0.0, description="primary - official baseline (0.6016)")
    raw_stdout: Optional[str] = None


class ExecutionResult(BaseModel):
    status: Literal["SUCCESS", "RUNTIME_ERROR", "TIMEOUT", "SYNTAX_ERROR", "NO_METRICS"]
    metrics: Optional[MetricResult] = None
    error_traceback: Optional[str] = None
    stdout_summary: str = ""
    wall_clock_seconds: float = 0.0
    command_executed: str = ""

    @property
    def failed(self) -> bool:
        return self.status != "SUCCESS"


class TokenUsage(BaseModel):
    """Real LLM accounting. Feasibility is graded on this, so it must be truthful."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int):
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.calls += 1


class HypothesisProposal(BaseModel):
    """A hypothesis and the command that tests it, bound together.

    They travel as one object precisely because the previous implementation let a
    fallback overwrite the command while keeping the LLM's prose, producing a
    run-log where iteration 1 claimed "DCN-v2" and ran ``--model mmoe``.
    ``source`` records where the pair actually came from.
    """
    stage: str
    hypothesis: str
    rationale: str = ""
    target_file: str = "pipeline/train.py"
    command: str
    source: Literal["llm", "fallback", "repair"] = "llm"


class IterationLogEntry(BaseModel):
    iteration_id: int
    parent_node_id: Optional[int] = None
    node_id: int
    stage: str
    hypothesis: str
    rationale: str = ""
    target_file: str
    command: str
    proposal_source: str = "llm"
    code_diff: str = ""
    status: Literal["ACCEPTED", "REJECTED", "ERROR_RECOVERED", "FAILED"]
    metrics: Optional[Dict[str, Any]] = None
    delta_over_baseline: Optional[float] = None
    error_recovery: Optional[Dict[str, Any]] = None
    #: How the four agent roles arrived at this iteration's experiment: the PM's
    #: directive, the hypotheses considered, what the Engineer rejected and why,
    #: and QA's pre-flight. This is the audit trail for the multi-agent design.
    agent_trace: Optional[Dict[str, Any]] = None
    #: Spent by THIS iteration (hypothesis call plus any repair calls it made).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    #: Running totals as of the end of this iteration, so the log reads both ways
    #: without a reader having to sum the column.
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    wall_clock_seconds: float = 0.0
    manual_interventions: int = 0

    @property
    def iteration_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunSummary(BaseModel):
    """Everything the Feasibility and Autonomy criteria are scored on."""
    run_id: str
    benchmark: str = "KuaiRand-Pure"
    baseline_valid_primary: float = 0.6016
    #: True only if this run re-ran and verified the baseline. False means the
    #: reference was asserted from the published value, so every delta below is
    #: relative to an unverified number.
    baseline_reproduced: bool = False
    best_valid_primary: Optional[float] = None
    best_delta: Optional[float] = None
    best_iteration: Optional[int] = None
    iterations_used: int = 0
    iteration_cap: int = 50
    halt_reason: str = ""
    wall_clock_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    llm_calls: int = 0
    manual_interventions: int = 0
    error_recoveries: int = 0
    failed_iterations: int = 0
    submission_path: Optional[str] = None
    #: What the baseline reproduction actually measured, and how far it drifted
    #: from the published reference. Previously only a bool was recorded, so a
    #: run whose baseline came out at 0.55 looked identical to one that hit
    #: 0.6015 exactly.
    baseline_measured: Optional[float] = None
    baseline_drift: Optional[float] = None
    #: Which result was designated final and why — margin over the runner-up in
    #: units of seed noise, and an explicit statement when nothing beat the
    #: baseline. The choice used to be an unrecorded argmax.
    submission_decision: Dict[str, Any] = Field(default_factory=dict)
    #: Every human touchpoint in the run. Autonomy is scored on this, so it
    #: records what was actually supplied rather than what flatters the run.
    interventions: List[Dict[str, Any]] = Field(default_factory=list)
    #: How much of the run the model actually authored, per stage.
    autonomy: Dict[str, Any] = Field(default_factory=dict)
    #: Token spend attributed per agent role, so the cost of the multi-agent
    #: design is visible rather than buried in a single total.
    cost_by_agent: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    iterations: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
