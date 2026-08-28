"""Extract validation metrics from a trial's stdout.

The trainer prints exactly one machine-readable line per run::

    [EVAL] GAUC: 0.6671 | nDCG@5: 0.5358 | Primary: 0.6015

Only validation metrics are ever emitted, so there is no way for a hidden-test
number to reach the orchestrator through this channel.
"""
import re
from typing import Optional

from orchestrator.schemas import MetricResult
from orchestrator.tree_manager import BASELINE_VAL_PRIMARY

EVAL_RE = re.compile(
    r"\[EVAL\]\s*GAUC:\s*([0-9.]+)\s*\|\s*nDCG@5:\s*([0-9.]+)"
    r"(?:\s*\|\s*Primary:\s*([0-9.]+))?")


def parse_execution_output(stdout: str) -> Optional[MetricResult]:
    """Return the LAST [EVAL] line's metrics, or None if the trial printed none."""
    matches = list(EVAL_RE.finditer(stdout or ""))
    if not matches:
        return None
    g, n, p = matches[-1].groups()
    gauc, ndcg = float(g), float(n)
    primary = float(p) if p else (gauc + ndcg) / 2.0
    return MetricResult(
        gauc=gauc, ndcg_5=ndcg, primary_score=primary,
        delta_from_baseline=primary - BASELINE_VAL_PRIMARY,
        raw_stdout=(stdout or "")[-1500:])
