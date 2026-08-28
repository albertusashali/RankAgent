"""
Self-Healing Debugger for automatic traceback parsing and LLM code repair.
"""
import os
from typing import Optional, Tuple
from prompts.templates import DEBUGGER_PROMPT

class SelfHealingDebugger:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def generate_repair_prompt(self, target_file: str, original_code: str, error_traceback: str, stdout_summary: str) -> str:
        return DEBUGGER_PROMPT.format(
            target_file=target_file,
            original_code=original_code,
            error_traceback=error_traceback,
            stdout_summary=stdout_summary
        )

    def heuristic_quick_fix(self, target_file: str, code: str, error_traceback: str) -> Optional[str]:
        """Applies immediate heuristic fixes for known common errors."""
        # 1. CUDA out of memory -> adjust batch size
        if "out of memory" in error_traceback.lower() or "cuda oom" in error_traceback.lower():
            if "bs = 4096" in code:
                return code.replace("bs = 4096", "bs = 2048")
            if "bs: int = 4096" in code:
                return code.replace("bs: int = 4096", "bs: int = 2048")
            if "batch_size=bs" in code and "bs = 8192" in code:
                return code.replace("bs = 8192", "bs = 4096")
                
        return None

