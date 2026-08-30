"""Subprocess sandbox for one trial.

Isolation matters for more than tidiness here: PyTorch and LightGBM cannot share
a process (conflicting OpenMP runtimes), so running each trial in its own
interpreter is what lets the agent explore both families in one run. It also
means a segfault costs one iteration rather than the whole session.
"""
import os
import subprocess
import sys
import time
from typing import Dict, Optional

from orchestrator.schemas import ExecutionResult
from sandbox.parser import parse_execution_output


class ExecutionRunner:
    def __init__(self, timeout_seconds: int = 1800, python_executable: Optional[str] = None,
                 cwd: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.cwd = cwd or os.getcwd()

    def run_command(self, cmd: str, env_vars: Optional[Dict[str, str]] = None,
                    allow_no_metrics: bool = False) -> ExecutionResult:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(self.cwd)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        # Belt and braces: the hidden-test seal is enforced in the loader, and the
        # variable that would lift it is explicitly cleared for every trial.
        env.pop('RANKAGENT_UNSEAL_TEST', None)
        if env_vars:
            env.update(env_vars)

        t0 = time.time()
        proc = None
        try:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    encoding='utf-8', errors='replace',
                                    env=env, cwd=self.cwd)
            stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
            elapsed = time.time() - t0

            if proc.returncode != 0:
                detail = stderr or stdout or ""
                if proc.returncode < 0:
                    detail = (f"process died on signal {-proc.returncode} "
                              f"(exit {proc.returncode})\n{detail}")
                return ExecutionResult(status="RUNTIME_ERROR", error_traceback=detail[-6000:],
                                       stdout_summary=(stdout or "")[-2000:],
                                       wall_clock_seconds=elapsed, command_executed=cmd)

            metrics = parse_execution_output(stdout)
            if metrics is None:
                if allow_no_metrics:
                    return ExecutionResult(status="SUCCESS", stdout_summary=(stdout or "")[-2000:],
                                           wall_clock_seconds=elapsed, command_executed=cmd)
                return ExecutionResult(
                    status="NO_METRICS",
                    error_traceback="Trial exited 0 but printed no [EVAL] line; "
                                    "the trainer may have skipped evaluation.",
                    stdout_summary=(stdout or "")[-2000:],
                    wall_clock_seconds=elapsed, command_executed=cmd)

            return ExecutionResult(status="SUCCESS", metrics=metrics,
                                   stdout_summary=(stdout or "")[-2000:],
                                   wall_clock_seconds=elapsed, command_executed=cmd)

        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
                proc.communicate()
            return ExecutionResult(status="TIMEOUT",
                                   error_traceback=f"Exceeded the {self.timeout_seconds}s trial timeout.",
                                   wall_clock_seconds=time.time() - t0, command_executed=cmd)
        except Exception as exc:
            return ExecutionResult(status="RUNTIME_ERROR",
                                   error_traceback=f"{type(exc).__name__}: {exc}",
                                   wall_clock_seconds=time.time() - t0, command_executed=cmd)
