"""SEARCH/REPLACE patches — the Engineer's output format.

WHY NOT UNIFIED DIFFS
---------------------
A unified diff requires every context line to match byte-for-byte, including
line numbers and hunk offsets. Models are unreliable at producing them, and the
failure mode is silent-ish: the patch "almost" applies. Debugging patch
application instead of the agent is not how to spend a hackathon.

Anchored SEARCH/REPLACE blocks (the format aider popularised, and therefore the
one frontier models have seen most) move the burden to *quoting existing code*,
which models do well. The anchor either matches or it does not, and when it does
not we can say exactly where we looked.

The parser never raises: it returns ``(edits, errors)`` so a malformed reply
becomes a message back to the model rather than a crashed iteration. Application
is transactional — nothing reaches disk until every edit has matched and every
touched file still parses — so a half-applied patch never exists.
"""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HEAD = "<<<<<<< SEARCH"
MID = "======="
TAIL = ">>>>>>> REPLACE"

_FENCE = re.compile(r"^\s*```")


@dataclass
class Edit:
    """One anchored replacement in one file."""
    path: str
    search: str
    replace: str

    @property
    def is_creation(self) -> bool:
        return self.search.strip() == ""


@dataclass
class MatchOutcome:
    kind: str                       # "ok" | "AMBIGUOUS" | "NO_MATCH"
    strategy: str = ""
    lines: List[int] = field(default_factory=list)   # 1-based, for AMBIGUOUS
    near: List[Tuple[int, str]] = field(default_factory=list)  # for NO_MATCH

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def _norm_ws(line: str) -> str:
    return line.rstrip()


def _dedent_key(lines: Sequence[str]) -> Tuple[str, ...]:
    """Lines with their common leading indentation removed."""
    real = [l for l in lines if l.strip()]
    if not real:
        return tuple(l.strip() for l in lines)
    pad = min(len(l) - len(l.lstrip()) for l in real)
    return tuple(l[pad:].rstrip() if l.strip() else "" for l in lines)


def _indent_of(lines: Sequence[str]) -> str:
    for l in lines:
        if l.strip():
            return l[:len(l) - len(l.lstrip())]
    return ""


def _reindent(lines: Sequence[str], claimed: str, actual: str) -> List[str]:
    """Rebase ``lines`` from the model's indent level onto the file's.

    Only the *base* indent is rebased; indentation relative to the block's first
    line is preserved, because in Python that relative structure is semantics —
    a line silently moved into or out of a loop body is a different program, not
    a formatting fix. This is why the caller only reaches here for a *uniform*
    shift: a block whose internal relative indentation disagrees with the file
    is rejected as NO_MATCH rather than guessed at.
    """
    out: List[str] = []
    for l in lines:
        if not l.strip():
            out.append(l)
        elif claimed and l.startswith(claimed):
            out.append(actual + l[len(claimed):])
        else:
            out.append(actual + l.lstrip())
    return out


def nearest_windows(src: str, needle: str, k: int = 3) -> List[Tuple[int, str]]:
    """The ``k`` regions of ``src`` most similar to ``needle``.

    This is what makes a retry land. Telling the model "your anchor did not
    match" is nearly useless; showing it the three closest passages with real
    line numbers lets it re-anchor on text it can see.
    """
    src_lines = src.splitlines()
    n = max(1, len(needle.splitlines()))
    scored: List[Tuple[float, int]] = []
    for i in range(max(1, len(src_lines) - n + 1)):
        window = "\n".join(src_lines[i:i + n])
        ratio = difflib.SequenceMatcher(None, window, needle).quick_ratio()
        scored.append((ratio, i))
    scored.sort(key=lambda t: -t[0])

    out: List[Tuple[int, str]] = []
    taken: List[int] = []
    for ratio, i in scored:
        if len(out) >= k:
            break
        if any(abs(i - j) < n for j in taken):     # don't return overlaps
            continue
        taken.append(i)
        out.append((i + 1, "\n".join(src_lines[i:i + n])))
    return out


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_edit_blocks(text: str,
                      allowed: Iterable[str]) -> Tuple[List[Edit], List[str]]:
    """Extract SEARCH/REPLACE blocks. Never raises.

    The path is the last non-empty, non-fence line before ``<<<<<<< SEARCH``.
    """
    allowed = set(allowed)
    lines = (text or "").splitlines()
    edits: List[Edit] = []
    errors: List[str] = []

    i = 0
    last_path: Optional[str] = None
    while i < len(lines):
        line = lines[i]
        if line.strip() == HEAD:
            if last_path is None:
                errors.append(
                    f"line {i + 1}: a SEARCH block has no file path above it. "
                    f"Put the path on its own line immediately before "
                    f"`{HEAD}`.")
                path = "<unknown>"
            else:
                path = last_path

            search: List[str] = []
            replace: List[str] = []
            bucket = search
            i += 1
            closed = False
            while i < len(lines):
                cur = lines[i]
                s = cur.strip()
                if s == MID and bucket is search:
                    bucket = replace
                elif s == TAIL:
                    closed = True
                    break
                elif s == HEAD:
                    errors.append(f"line {i + 1}: a new `{HEAD}` opened before "
                                  f"the previous block was closed with `{TAIL}`.")
                    break
                else:
                    bucket.append(cur)
                i += 1

            if not closed:
                errors.append(f"the block for {path} is not terminated with `{TAIL}`.")
            elif path not in allowed and path != "<unknown>":
                errors.append(
                    f"{path} is not editable. You may only edit: "
                    f"{', '.join(sorted(allowed))}.")
            elif path != "<unknown>":
                edits.append(Edit(path=path,
                                  search="\n".join(search),
                                  replace="\n".join(replace)))
            last_path = None
        else:
            s = line.strip()
            if s and not _FENCE.match(line):
                last_path = s
        i += 1

    return edits, errors


def parse_whole_file(text: str, allowed: Iterable[str],
                     current: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Fall back to a complete-file reply when no anchors were given.

    Asking a model to "fix this file" very often gets the whole corrected file
    back in a fenced block rather than SEARCH/REPLACE edits — that was the single
    largest cause of failed repairs, and treating it as a malformed reply threw
    away work that was probably correct.

    Two guards, because accepting a whole file is riskier than accepting an
    anchored edit: the result must parse, and it must not be drastically shorter
    than what it replaces, which is what a truncated reply looks like.
    """
    allowed = set(allowed)
    problems: List[str] = []
    out: Dict[str, str] = {}

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if not blocks:
        return {}, ["your reply contained neither SEARCH/REPLACE blocks nor a "
                    "fenced code block."]

    # Which file is this? An explicit mention wins; otherwise, if exactly one
    # file was on the table, it is that one.
    named = [p for p in allowed if p in (text or "")]
    candidates = named or ([p for p in allowed if p in current] if len(current) == 1 else [])
    if len(candidates) != 1:
        return {}, [f"your reply had no SEARCH/REPLACE blocks, and the file it "
                    f"applies to is ambiguous. Name one of "
                    f"{', '.join(sorted(allowed))} on its own line, or use "
                    f"SEARCH/REPLACE blocks."]

    path = candidates[0]
    body = max(blocks, key=len)
    old = current.get(path, "")

    try:
        ast.parse(body)
    except SyntaxError as exc:
        return {}, [f"the complete file you returned for {path} does not parse: "
                    f"{exc.msg} at line {exc.lineno}."]

    if old and len(body.splitlines()) < 0.6 * len(old.splitlines()):
        return {}, [f"the file you returned for {path} is "
                    f"{len(body.splitlines())} lines, far shorter than the "
                    f"{len(old.splitlines())} it replaces — it looks truncated. "
                    f"Reply with SEARCH/REPLACE blocks for just the part that "
                    f"changes."]

    out[path] = body if body.endswith("\n") else body + "\n"
    return out, problems


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------

def apply_edit(src: str, edit: Edit) -> Tuple[str, MatchOutcome]:
    """Apply one edit, trying progressively more forgiving matches.

    Uniqueness is required. An anchor matching twice is ambiguous, and guessing
    which occurrence was meant is exactly the kind of silent wrong-thing this
    whole design exists to avoid.
    """
    if edit.is_creation:
        return edit.replace, MatchOutcome("ok", "create")

    # 1. exact substring — the common case.
    hits = []
    start = src.find(edit.search)
    while start != -1:
        hits.append(start)
        start = src.find(edit.search, start + 1)
    if len(hits) == 1:
        at = hits[0]
        return src[:at] + edit.replace + src[at + len(edit.search):], \
            MatchOutcome("ok", "exact")
    if len(hits) > 1:
        return src, MatchOutcome("AMBIGUOUS", "exact",
                                 lines=[src[:h].count("\n") + 1 for h in hits])

    src_lines = src.splitlines()
    needle = edit.search.splitlines()
    repl = edit.replace.splitlines()
    n = len(needle)

    # 2. line-wise, ignoring trailing whitespace.
    want = [_norm_ws(l) for l in needle]
    found = [i for i in range(len(src_lines) - n + 1)
             if [_norm_ws(l) for l in src_lines[i:i + n]] == want]
    if len(found) == 1:
        i = found[0]
        out = src_lines[:i] + repl + src_lines[i + n:]
        return "\n".join(out) + ("\n" if src.endswith("\n") else ""), \
            MatchOutcome("ok", "trailing-whitespace")
    if len(found) > 1:
        return src, MatchOutcome("AMBIGUOUS", "trailing-whitespace",
                                 lines=[i + 1 for i in found])

    # 3. uniform re-indentation — the single most common model error is a block
    #    copied at the wrong indent level. Re-indent the replacement by the same
    #    delta so the result stays syntactically consistent.
    key = _dedent_key(needle)
    found = [i for i in range(len(src_lines) - n + 1)
             if _dedent_key(src_lines[i:i + n]) == key]
    if len(found) == 1:
        i = found[0]
        out = src_lines[:i] + _reindent(repl, _indent_of(needle),
                                        _indent_of(src_lines[i:i + n])) \
            + src_lines[i + n:]
        return "\n".join(out) + ("\n" if src.endswith("\n") else ""), \
            MatchOutcome("ok", "reindent")
    if len(found) > 1:
        return src, MatchOutcome("AMBIGUOUS", "reindent",
                                 lines=[i + 1 for i in found])

    return src, MatchOutcome("NO_MATCH", near=nearest_windows(src, edit.search))


def apply_all(files: Dict[str, str],
              edits: Sequence[Edit]) -> Tuple[Dict[str, str], List[str]]:
    """Apply every edit in order. Returns ``(new_files, problems)``.

    Edits apply against the progressively-updated buffer, so a later edit can
    depend on an earlier one in the same patch. If ANY edit fails, or any touched
    file stops parsing, ``problems`` is non-empty and the caller must discard the
    result — nothing should be written to disk.
    """
    out = dict(files)
    problems: List[str] = []

    for idx, e in enumerate(edits, 1):
        if e.path not in out and not e.is_creation:
            problems.append(f"edit {idx}: {e.path} does not exist in the workspace.")
            continue
        src = out.get(e.path, "")
        new, outcome = apply_edit(src, e)
        if outcome.ok:
            out[e.path] = new
            continue
        if outcome.kind == "AMBIGUOUS":
            # Show what surrounds each match. Line numbers alone are not enough
            # to re-anchor on — a model given only "matches at 31, 33, 142"
            # reissues the same block verbatim, which is what a real run did
            # twice in a row before this context was included.
            src_lines = src.splitlines()
            shown = []
            for ln in outcome.lines[:4]:
                lo, hi = max(0, ln - 3), min(len(src_lines), ln + 2)
                shown.append(f"  --- around line {ln} ---\n" + "\n".join(
                    f"  {i + 1:>5}| {src_lines[i]}" for i in range(lo, hi)))
            problems.append(
                f"edit {idx} for {e.path}: the SEARCH block matches "
                f"{len(outcome.lines)} times, so it is ambiguous. Here is what "
                f"surrounds each match:\n" + "\n\n".join(shown) +
                f"\n\nPick ONE of those locations and extend your SEARCH block "
                f"upward to include a line unique to it — the enclosing `def`, "
                f"or a distinctive comment.")
        else:
            near = "\n\n".join(
                f"  --- lines {ln}-{ln + len(chunk.splitlines()) - 1} ---\n{chunk}"
                for ln, chunk in outcome.near)
            problems.append(
                f"edit {idx} for {e.path}: the SEARCH block did not match. The "
                f"closest regions in the current file are:\n{near}\n"
                f"Re-issue the edit with an anchor copied exactly from one of "
                f"those excerpts.")

    if problems:
        return files, problems

    for path, text in out.items():
        if not path.endswith(".py") or files.get(path) == text:
            continue
        try:
            ast.parse(text)
        except SyntaxError as exc:
            lines = text.splitlines()
            lo = max(0, (exc.lineno or 1) - 4)
            hi = min(len(lines), (exc.lineno or 1) + 3)
            excerpt = "\n".join(f"{i + 1:>5}| {lines[i]}" for i in range(lo, hi))
            problems.append(
                f"{path} no longer parses after your edit: {exc.msg} at line "
                f"{exc.lineno}, column {exc.offset}. Lines {lo + 1}-{hi}:\n"
                f"{excerpt}\nFix the syntax without restructuring the change.")

    if problems:
        return files, problems
    return out, []
