"""
Regex parser for extracting GAUC, nDCG@5, and Primary score from execution stdout.
"""
import re
from typing import Optional
from orchestrator.schemas import MetricResult

BASELINE_PRIMARY = 0.6016  # Official baseline val primary score

def parse_execution_output(stdout: str) -> Optional[MetricResult]:
    """
    Extracts metrics from standard evaluate.py output:
    [EVAL] GAUC: 0.6674 | nDCG@5: 0.5357 | Primary: 0.6016
    """
    pattern = r"\[EVAL\]\s+GAUC:\s+([\d\.]+)\s+\|\s+nDCG@5:\s+([\d\.]+)(?:\s+\|\s+Primary:\s+([\d\.]+))?"
    matches = list(re.finditer(pattern, stdout))
    
    if not matches:
        return None
        
    last_match = matches[-1]
    gauc = float(last_match.group(1))
    ndcg = float(last_match.group(2))
    
    if last_match.group(3):
        primary = float(last_match.group(3))
    else:
        primary = (gauc + ndcg) / 2.0
        
    delta = primary - BASELINE_PRIMARY
    
    return MetricResult(
        gauc=gauc,
        ndcg_5=ndcg,
        primary_score=primary,
        delta_from_baseline=delta,
        raw_stdout=stdout[-500:]
    )

