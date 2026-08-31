"""Per-node code workspaces — the agent's action space is a directory, not a flag.

Each tree node owns a copy of the pipeline source. A child is copied from its
**parent**, so an edit applied at node 9 sits on top of the edits at nodes 4 and
7: changes compose instead of every trial restarting from the baseline. That
compounding is the whole reason the tree exists.

TWO SURFACES
------------
``MUTABLE`` is what the agent may rewrite. ``IMMUTABLE`` is hard-restored from
the canonical repo and SHA-256-verified *before every run*, so a generated edit
to the scorer or the data loader is reverted and reported rather than executed.
This is a filesystem boundary, not a prompt instruction: the agent cannot reach
``pipeline/evaluate.py`` (which decides every number the run is judged on) or
``pipeline/data.py`` (which withholds the hidden-test labels), however it is
asked to.

Rollback is therefore free — a bad node is simply never made a parent — and the
canonical repo is never mutated, so ``git status`` stays clean during a run.
"""
from __future__ import annotations

import difflib
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Files the agent may rewrite. Everything else under ``pipeline/`` is restored.
MUTABLE: Tuple[str, ...] = (
    "pipeline/models.py",      # architectures and loss functions
    "pipeline/features.py",    # encoders, causal statistics, sequence extraction
    "pipeline/train.py",       # optimiser, schedule, batching, early stopping
)

#: Copied into every workspace and verified. ``pipeline/evaluate.py`` resolves the
#: official scorer relative to its own ``__file__``, so the starter kit has to
#: travel with it — otherwise it silently falls back to the embedded
#: reimplementation and the run is scored by something other than the organizers'
#: code. ``assert_official_scorer`` below turns that silence into an error.
IMMUTABLE: Tuple[str, ...] = (
    "pipeline/__init__.py",
    "pipeline/data.py",             # the hidden-test seal
    "pipeline/evaluate.py",         # the official scorer
    "pipeline/models_np.py",        # the reproducible baseline
    "pipeline/submit.py",
    # The feature auditor and the recipe schema. Immutable for the same reason
    # the scorer is: a checker the agent can edit is not a check. If generated
    # code could weaken FORBIDDEN_CURRENT_ROW_SOURCES or relax the recipe's
    # validated ranges, the leak gate would pass whatever it was asked to pass.
    "pipeline/feature_agent.py",
    "pipeline/feature_recipes.py",
    "pipeline/diagnostics.py",
    "kuairand-starter-kit/evaluate.py",
)

WORKSPACES_DIR = "workspaces"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_manifest(repo_root: str) -> Dict[str, str]:
    """SHA-256 of every immutable file, as it exists in the canonical repo."""
    out: Dict[str, str] = {}
    for rel in IMMUTABLE:
        src = os.path.join(repo_root, rel)
        if os.path.exists(src):
            out[rel] = _sha256(src)
    return out


@dataclass
class Violation:
    """An immutable file the agent modified. Reverted, then reported."""
    path: str
    expected: str
    found: str

    def __str__(self) -> str:
        return (f"{self.path} is immutable and your edit to it was discarded "
                f"(sha {self.found[:12]} != {self.expected[:12]})")


@dataclass
class Workspace:
    node_id: int
    root: str                       # absolute path to workspaces/node_007
    parent_root: Optional[str]
    repo_root: str
    violations: List[Violation] = field(default_factory=list)
    #: Contents of the mutable files as inherited from the parent, captured at
    #: materialisation. Every diff this node reports is computed against this,
    #: so a patch that is later repaired still shows as one change against the
    #: parent rather than the repair alone.
    base: Dict[str, str] = field(default_factory=dict)

    # -- paths ------------------------------------------------------------

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.root, "checkpoints")

    def path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def read(self, rel: str) -> str:
        with open(self.path(rel), encoding="utf-8") as fh:
            return fh.read()

    def write(self, rel: str, text: str) -> None:
        if rel not in MUTABLE:
            raise PermissionError(
                f"{rel} is not in the mutable surface {list(MUTABLE)}")
        with open(self.path(rel), "w", encoding="utf-8") as fh:
            fh.write(text)

    def snapshot(self) -> Dict[str, str]:
        """Current contents of every mutable file, for diffing."""
        return {rel: self.read(rel) for rel in MUTABLE
                if os.path.exists(self.path(rel))}

    def fingerprint(self) -> Dict[str, str]:
        """SHA-256 of each mutable file — the node's code identity.

        Two nodes with the same fingerprint and the same arguments are the same
        experiment. Once code is mutable, comparing command strings is no longer
        enough to detect a duplicate.
        """
        return {rel: _sha256(self.path(rel)) for rel in MUTABLE
                if os.path.exists(self.path(rel))}

    # -- the guarantee ----------------------------------------------------

    def restore_immutable(self) -> List[Violation]:
        """Overwrite every immutable file from the canonical repo.

        Called before every run. Returns whatever the agent had changed, so the
        caller can log the attempt; the files themselves are already back to
        canonical by the time this returns.
        """
        found: List[Violation] = []
        for rel in IMMUTABLE:
            src = os.path.join(self.repo_root, rel)
            if not os.path.exists(src):
                continue
            dst = self.path(rel)
            want = _sha256(src)
            if os.path.exists(dst):
                have = _sha256(dst)
                if have != want:
                    found.append(Violation(rel, want, have))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        self.violations = found
        return found

    def assert_official_scorer(self) -> None:
        """Fail loudly if the workspace would score with the embedded fallback.

        ``pipeline/evaluate.py`` falls back to its own reimplementation when the
        starter kit is missing. The two are meant to agree, but "silently" and
        "scoring" is a combination worth refusing.
        """
        kit = self.path("kuairand-starter-kit/evaluate.py")
        if not os.path.exists(kit):
            raise RuntimeError(
                f"{kit} is missing: pipeline/evaluate.py would silently fall back "
                f"to its embedded scorer instead of the organizers' implementation.")

    # -- environment for a trial -----------------------------------------

    def env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Environment overrides that point a trial at this node."""
        out = {
            "PYTHONPATH": self.root,
            # Per-node checkpoints. Names encode only model and loss, so without
            # this two sibling nodes training the same architecture overwrite
            # each other's weights and the submission exports the wrong model.
            "RANKAGENT_CHECKPOINTS": self.checkpoints,
        }
        if extra:
            out.update(extra)
        return out


def materialise(node_id: int, parent: Optional[Workspace] = None,
                repo_root: Optional[str] = None,
                workspaces_dir: str = WORKSPACES_DIR) -> Workspace:
    """Create node ``node_id``'s workspace, seeded from ``parent`` or the repo.

    Seeding from the parent is what makes edits compose. Checkpoints are never
    inherited — they belong to the run that produced them.
    """
    repo_root = os.path.abspath(repo_root or os.getcwd())
    root = os.path.abspath(os.path.join(workspaces_dir, f"node_{node_id:03d}"))
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    source = parent.root if parent is not None else repo_root
    for rel in MUTABLE + IMMUTABLE:
        src = os.path.join(source, rel)
        if not os.path.exists(src):
            # Immutable files always come from the canonical repo, so a parent
            # that somehow lacks one is not a problem.
            src = os.path.join(repo_root, rel)
            if not os.path.exists(src):
                continue
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    os.makedirs(os.path.join(root, "checkpoints"), exist_ok=True)
    ws = Workspace(node_id=node_id, root=root,
                   parent_root=parent.root if parent else None,
                   repo_root=repo_root)
    ws.restore_immutable()
    ws.assert_official_scorer()
    ws.base = ws.snapshot()
    return ws


def unified_diff(before: Dict[str, str], after: Dict[str, str],
                 context: int = 3) -> Tuple[str, Tuple[int, int, int]]:
    """Diff two snapshots. Returns ``(text, (files, added, removed))``.

    The run log's ``code_diff`` is computed here, from the bytes actually on
    disk before and after the patch — never from what the model claimed it did.
    The previous implementation shelled out to ``git diff`` over the repository
    working tree, which is why archived runs logged unrelated README churn.
    """
    chunks: List[str] = []
    files = added = removed = 0
    for path in sorted(set(before) | set(after)):
        a = before.get(path, "").splitlines(keepends=True)
        b = after.get(path, "").splitlines(keepends=True)
        if a == b:
            continue
        files += 1
        for line in difflib.unified_diff(a, b, fromfile=f"a/{path}",
                                         tofile=f"b/{path}", n=context):
            chunks.append(line if line.endswith("\n") else line + "\n")
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    return "".join(chunks), (files, added, removed)
