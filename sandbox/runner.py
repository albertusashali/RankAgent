"""Subprocess sandbox for one trial.

Isolation matters for more than tidiness here: PyTorch and LightGBM cannot share
a process (conflicting OpenMP runtimes), so running each trial in its own
interpreter is what lets the agent explore both families in one run. It also
means a segfault costs one iteration rather than the whole session.
"""
import os
import signal
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

    @staticmethod
    def _is_secret(name: str) -> bool:
        """Credentials must not reach a trial subprocess.

        This was harmless while every trial ran code we wrote. Once the agent
        generates the code that runs here, inheriting the whole environment —
        which ``load_dotenv_if_present`` has just populated with the API keys
        from ``.env`` — would be a live exfiltration path out of a process
        launched with ``shell=True``.
        """
        upper = name.upper()
        return (upper.endswith(('_KEY', '_TOKEN', '_SECRET', '_PASSWORD'))
                or 'API_KEY' in upper or upper in ('OPENAI_API_KEY',
                                                   'ANTHROPIC_API_KEY',
                                                   'GEMINI_API_KEY'))

    def run_command(self, cmd: str, env_vars: Optional[Dict[str, str]] = None,
                    cwd: Optional[str] = None,
                    timeout_seconds: Optional[int] = None) -> ExecutionResult:
        cwd = cwd or self.cwd
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds

        env = {k: v for k, v in os.environ.items() if not self._is_secret(k)}
        env['PYTHONPATH'] = os.path.abspath(cwd)
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
            # start_new_session puts the shell and everything it spawns in one
            # process group. Without it, killing a timed-out trial kills only the
            # shell and orphans the trainer, which keeps its memory and its file
            # handles — including a checkpoint it may be halfway through writing.
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    encoding='utf-8', errors='replace',
                                    env=env, cwd=cwd, start_new_session=True)
            stdout, stderr = proc.communicate(timeout=timeout)
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
            self._kill_group(proc)
            return ExecutionResult(status="TIMEOUT",
                                   error_traceback=f"Exceeded the {timeout}s trial timeout.",
                                   wall_clock_seconds=time.time() - t0, command_executed=cmd)
        except Exception as exc:
            self._kill_group(proc)
            return ExecutionResult(status="RUNTIME_ERROR",
                                   error_traceback=f"{type(exc).__name__}: {exc}",
                                   wall_clock_seconds=time.time() - t0, command_executed=cmd)
        except BaseException:
            # KeyboardInterrupt and SystemExit are not Exception subclasses, so
            # they reach here. Reap the trial before re-raising, or Ctrl-C leaves
            # the trainer running in the background holding its memory.
            # Ordering matters: this clause must come AFTER `except Exception`,
            # or it would swallow ordinary errors and break the guarantee that
            # run_command returns a result rather than raising.
            self._kill_group(proc)
            raise

    @staticmethod
    def _kill_group(proc) -> None:
        """SIGKILL the trial's whole process group, then reap it."""
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                return
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
