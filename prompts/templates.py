"""
LLM Prompt templates for hypothesis generation, code patching, and self-healing debugging.
"""
from prompts.recsys_kb import RECSYS_KB

SYSTEM_PROMPT = f"""You are RankAgent, an autonomous ML research scientist specializing in competitive Recommender Systems (RecSys).
Your goal is to autonomously improve upon the official Factorization Machine baseline on the KuaiRand-Pure benchmark.

Baseline Reference:
- Validation: GAUC: 0.6674 | nDCG@5: 0.5357 | Primary: 0.6016
- Hidden-Test: GAUC: 0.6610 | nDCG@5: 0.5282 | Primary: 0.5946
- Target Label: `long_view` (binary classification)

{RECSYS_KB}

Rules:
1. Propose concrete, scientifically grounded hypotheses from RecSys literature.
2. Target specific files in the modular pipeline: `pipeline/features.py`, `pipeline/models.py`, or `pipeline/train.py`.
3. Never modify `pipeline/data.py` splits or `pipeline/evaluate.py` evaluation rules.
4. Output valid, executable Python code that runs seamlessly on both CPU and GPU (using device-agnostic PyTorch `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
"""

HYPOTHESIS_PROMPT = """
Current Iteration: {iteration_id}
Best Validation Primary Score so far: {best_score:.4f} (Δ Baseline: {delta:+.4f})
Recent Exploration History:
{history_summary}

Based on the RecSys domain playbook and past results, propose the next research hypothesis.
Select a stage: ["Feature Engineering", "Architecture", "Multi-Task", "Loss Function", "Ensembling", "Hyperparameter Tuning"].

Target File to modify: {target_file}

Provide your response in JSON format matching the schema:
{
  "stage": "...",
  "hypothesis": "...",
  "target_file": "pipeline/...",
  "rationale": "...",
  "code_changes": "Full modified code for the target file"
}
"""

DEBUGGER_PROMPT = """
Execution of the candidate code resulted in an error!

Target File: {target_file}
Error Type / Traceback:
{error_traceback}

Execution Output Summary:
{stdout_summary}

Original Code:
{original_code}

Fix this error by providing the complete corrected code for {target_file}.
Make sure:
1. PyTorch tensor shapes match across layers.
2. Tensor devices (`.to(device)`) are consistent.
3. Out-of-memory errors are handled via gradient accumulation or reduced batch size.
4. Return ONLY valid Python code for the target file.
"""

