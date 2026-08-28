"""
Execution Sandbox and Process Manager with timeout guards and metric parsing.
"""
import os
import sys
import time
import subprocess
from typing import Optional, Dict
from orchestrator.schemas import ExecutionResult
from sandbox.parser import parse_execution_output

class ExecutionRunner:
    def __init__(self, timeout_seconds: int = 900, python_executable: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable

    def run_command(self, cmd: str, env_vars: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """
        Executes a command inside the sandbox with strict timeout and stdout/stderr capture.
        """
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath('.')
        if env_vars:
            env.update(env_vars)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            elapsed_time = time.time() - start_time

            if process.returncode != 0:
                return ExecutionResult(
                    status="RUNTIME_ERROR",
                    error_traceback=stderr or stdout,
                    stdout_summary=stdout[-1000:] if stdout else "",
                    wall_clock_seconds=elapsed_time,
                    command_executed=cmd
                )

            metrics = parse_execution_output(stdout)
            if metrics is None:
                return ExecutionResult(
                    status="RUNTIME_ERROR",
                    error_traceback="Process succeeded but failed to parse [EVAL] metrics from stdout.",
                    stdout_summary=stdout[-1000:],
                    wall_clock_seconds=elapsed_time,
                    command_executed=cmd
                )

            return ExecutionResult(
                status="SUCCESS",
                metrics=metrics,
                stdout_summary=stdout[-1000:],
                wall_clock_seconds=elapsed_time,
                command_executed=cmd
            )

        except subprocess.TimeoutExpired:
            process.kill()
            elapsed_time = time.time() - start_time
            return ExecutionResult(
                status="TIMEOUT",
                error_traceback=f"Execution exceeded timeout of {self.timeout_seconds} seconds.",
                stdout_summary="",
                wall_clock_seconds=elapsed_time,
                command_executed=cmd
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            return ExecutionResult(
                status="RUNTIME_ERROR",
                error_traceback=str(e),
                stdout_summary="",
                wall_clock_seconds=elapsed_time,
                command_executed=cmd
            )

