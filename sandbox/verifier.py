"""Static gates on generated code, run before a single epoch is trained.

Three things are being defended, in descending order of how badly they would
hurt if they got through.

**Leakage.** The agent optimises against validation, so any code that reaches
the label — directly, or through a post-impression outcome column — produces a
number that looks like a breakthrough and is worthless on the hidden test set.
The most dangerous case is not malice but a plausible one-liner:
``watch_ratio = play_time_ms / duration_ms`` is the single most natural feature
to write given a row dict, and ``long_view`` is close to a deterministic
function of it. ``pipeline/data.py`` now withholds those columns on the test
split, so such a feature would be garbage at submission time; this gate stops it
being written in the first place, and says why.

**Causality.** ``CausalStats`` is fed by expanding window so a row's own label
never reaches its own features. Generated code can break that invariant while
still running perfectly — and the resulting validation score goes *up*, so
nothing downstream would flag it. A grep cannot catch it; the property test in
``tests/test_harness.py`` can, so it is re-run against the workspace's code.

**Blast radius.** No network, no subprocesses, no environment access.

Every failure message is written for the model that will read it: what it did,
why that is wrong here, and what to do instead.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

#: Names that reach the hidden test set.
DENY_NAMES: Set[str] = {"load_test_labels", "load_unbiased_valid"}

#: Strings that lift the seal.
DENY_STRINGS: Set[str] = {"RANKAGENT_UNSEAL_TEST"}

#: Row keys that only exist *after* the impression happened. They are withheld
#: on the hidden test split, so a feature derived from one is an artefact.
OUTCOME_KEYS: Set[str] = {
    "play_time_ms", "profile_stay_time", "comment_stay_time",
    "is_hate", "is_profile_enter",
}

#: Third-party packages the trial environment actually has.
ALLOWED_THIRD_PARTY: Set[str] = {"numpy", "torch", "lightgbm", "scipy", "pipeline"}

#: Everything else in the standard library is fine, so the allowlist is
#: "stdlib minus the deny list" rather than a hand-maintained enumeration. An
#: enumeration would be wrong the first time a patch legitimately reaches for
#: `argparse` or `itertools`, and a false positive here blocks working code.
_STDLIB: Set[str] = set(getattr(sys, "stdlib_module_names", ())) or {
    "math", "random", "time", "json", "os", "sys", "collections", "itertools",
    "functools", "typing", "dataclasses", "abc", "copy", "warnings", "re",
    "heapq", "bisect", "argparse", "csv", "io", "contextlib", "textwrap",
}

DENY_IMPORTS: Set[str] = {
    "subprocess", "socket", "requests", "urllib", "http", "ftplib",
    "shutil", "ctypes", "pickle", "multiprocessing", "importlib",
}

ALLOWED_IMPORTS: Set[str] = (ALLOWED_THIRD_PARTY | _STDLIB) - DENY_IMPORTS

#: Builtins that execute arbitrary text. Only flagged when called bare — a
#: method call of the same name is unrelated, and `model.eval()` (switch a torch
#: module to inference mode) appears throughout the existing trainers.
DENY_BUILTINS: Set[str] = {"eval", "exec", "compile", "__import__"}

#: Attribute calls that shell out, matched on the full dotted path.
DENY_ATTR_CALLS: Set[str] = {"os.system", "os.popen", "os.execv", "os.fork"}


@dataclass
class Finding:
    gate: str
    path: str
    line: int
    message: str
    fatal: bool = False

    def __str__(self) -> str:
        return f"[{self.gate}] {self.path}:{self.line} {self.message}"


@dataclass
class VerifyReport:
    findings: List[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def fatal(self) -> bool:
        return any(f.fatal for f in self.findings)

    def messages(self) -> List[str]:
        return [str(f) for f in self.findings]


def _excerpt(src: str, line: int, span: int = 2) -> str:
    lines = src.splitlines()
    lo, hi = max(0, line - 1 - span), min(len(lines), line + span)
    return "\n".join(f"  {i + 1:>5}| {lines[i]}" for i in range(lo, hi))


def verify_source(path: str, src: str,
                  baseline_src: Optional[str] = None) -> List[Finding]:
    """Static gates for one generated module. Cheap: pure AST, no import."""
    out: List[Finding] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [Finding("syntax", path, exc.lineno or 0,
                        f"the file does not parse: {exc.msg}. "
                        f"Lines around it:\n{_excerpt(src, exc.lineno or 1)}",
                        fatal=False)]

    for node in ast.walk(tree):
        # -- imports ------------------------------------------------------
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in DENY_IMPORTS:
                    out.append(Finding(
                        "imports", path, node.lineno,
                        f"`import {a.name}` is not permitted. There is no network "
                        f"access and no shelling out from a trial. Available: "
                        f"{', '.join(sorted(ALLOWED_IMPORTS))}."))
                elif root not in ALLOWED_IMPORTS:
                    out.append(Finding(
                        "imports", path, node.lineno,
                        f"`import {a.name}` is not available in the trial "
                        f"environment. Implement it with "
                        f"{', '.join(sorted(ALLOWED_IMPORTS & {'numpy', 'torch', 'scipy'}))} "
                        f"or the standard library."))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in DENY_IMPORTS:
                out.append(Finding(
                    "imports", path, node.lineno,
                    f"`from {node.module} import ...` is not permitted."))

        # -- names that reach the hidden test set --------------------------
        elif isinstance(node, ast.Name) and node.id in DENY_NAMES:
            out.append(Finding(
                "leak", path, node.lineno,
                f"`{node.id}` reads the hidden test labels. Model selection in "
                f"this project is on validation only, and the run is scored once "
                f"on a test set the agent never sees. Remove it.", fatal=True))

        elif isinstance(node, ast.Attribute) and node.attr in DENY_NAMES:
            out.append(Finding(
                "leak", path, node.lineno,
                f"`.{node.attr}` reads the hidden test labels. Remove it.",
                fatal=True))

        # -- dangerous calls ----------------------------------------------
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in DENY_BUILTINS:
                out.append(Finding(
                    "imports", path, node.lineno,
                    f"`{fn.id}(...)` executes arbitrary text and is not permitted "
                    f"in generated code."))
            elif isinstance(fn, ast.Attribute):
                dotted = f"{getattr(fn.value, 'id', '?')}.{fn.attr}"
                if dotted in DENY_ATTR_CALLS:
                    out.append(Finding(
                        "imports", path, node.lineno,
                        f"`{dotted}(...)` shells out and is not permitted."))

        # -- string literals ----------------------------------------------
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in DENY_STRINGS:
                out.append(Finding(
                    "leak", path, node.lineno,
                    f"`{node.value}` is the flag that unseals the hidden test "
                    f"labels. It must never appear in pipeline code.", fatal=True))

        # -- outcome columns and the test split ----------------------------
        elif isinstance(node, ast.Subscript):
            key = _const_str(node.slice)
            if key in OUTCOME_KEYS:
                out.append(Finding(
                    "leak", path, node.lineno,
                    f"this reads `{key}` from a row. That is a POST-impression "
                    f"outcome: it is withheld (-1) on the hidden test split, so "
                    f"any feature derived from it looks excellent on validation "
                    f"and is meaningless at submission time. Use pre-impression "
                    f"context instead — duration_ms, tab, is_rand, the ids, or an "
                    f"aggregate over *other* rows via CausalStats.\n"
                    f"{_excerpt(src, node.lineno)}", fatal=True))
            elif key == "test":
                out.append(Finding(
                    "leak", path, node.lineno,
                    f"this indexes the 'test' split by name. Trials load train "
                    f"and valid only; the test split is materialised once, at "
                    f"submission time, by code you cannot edit.\n"
                    f"{_excerpt(src, node.lineno)}", fatal=True))

        # -- .get('play_time_ms', ...) -------------------------------------
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get" \
                and node.args:
            key = _const_str(node.args[0])
            if key in OUTCOME_KEYS and not _is_causalstats_observe(tree, node):
                out.append(Finding(
                    "leak", path, node.lineno,
                    f"this reads `{key}` via .get(). It is a post-impression "
                    f"outcome, withheld on the hidden test split. See the note "
                    f"about pre-impression context.\n{_excerpt(src, node.lineno)}",
                    fatal=True))

    return out


def _const_str(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Index):                       # py<3.9 compatibility
        return _const_str(node.value)
    return None


def _is_causalstats_observe(tree: ast.AST, target: ast.AST) -> bool:
    """Allow the one legitimate reading of play_time_ms.

    ``CausalStats.observe`` folds watch time into a *duration-bucket* aggregate,
    and it only ever sees rows whose label is present — test rows are skipped by
    the ``if y < 0: continue`` guard at the top of the loop. That is an aggregate
    over other impressions, not a feature of the row being scored, so it does not
    leak. Flagging it would force the agent to delete working baseline code.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "observe":
            for sub in ast.walk(node):
                if sub is target:
                    return True
    return False


def verify_workspace(ws, changed: Optional[Sequence[str]] = None) -> VerifyReport:
    """Run every static gate over a workspace's mutable files.

    Immutable files are restored first, so the report also carries anything the
    agent tried to change that it may not.
    """
    report = VerifyReport()

    for violation in ws.restore_immutable():
        report.findings.append(Finding(
            "immutable", violation.path, 0,
            f"this file is immutable and your edit to it was discarded. "
            f"pipeline/evaluate.py decides every number the run is judged on and "
            f"pipeline/data.py withholds the hidden test labels, so neither is "
            f"editable. Express the change in pipeline/models.py, "
            f"pipeline/features.py or pipeline/train.py instead.", fatal=False))

    from sandbox.workspace import MUTABLE
    for rel in (changed if changed is not None else MUTABLE):
        try:
            src = ws.read(rel)
        except OSError:
            continue
        report.findings.extend(verify_source(rel, src))

    return report
