"""Prompt construction and static source inspection for the code-writing Engineer.

Two jobs here, both about keeping the Engineer cheap and grounded:

* **Digests instead of dumps.** The mutable surface is ~1050 lines. Sending all
  of it every call would cost ~13k tokens per iteration for context the model
  mostly does not need. Instead it sees an ``ast``-derived outline of everything
  and the full text of only the file it is editing.
* **Reading the registry out of source.** After the Engineer adds a loss, the
  orchestrator has to know that ``--loss approx_ndcg`` is now valid — but it
  cannot import the workspace's ``models.py`` to find out, because that pulls
  torch into the orchestrator process and torch and LightGBM segfault when both
  load together. So the registry is read *statically*, from the dict literal.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# static inspection
# ---------------------------------------------------------------------------

def source_digest(path: str, src: str) -> str:
    """A one-line-per-symbol outline of a module."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"{path} (DOES NOT PARSE: {exc.msg} at line {exc.lineno})"

    out: List[str] = [f"{path} ({len(src.splitlines())} lines)"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"  L{node.lineno:<4} def {node.name}({_sig(node)})")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_unparse(b) for b in node.bases)
            out.append(f"  L{node.lineno:<4} class {node.name}({bases})")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"  L{sub.lineno:<4}   .{sub.name}({_sig(sub)})")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.append(f"  L{node.lineno:<4} {t.id} = ...")
    return "\n".join(out)


def _sig(node) -> str:
    parts = [a.arg for a in node.args.args]
    if node.args.vararg:
        parts.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        parts.append("**" + node.args.kwarg.arg)
    return ", ".join(parts)


def _unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _dict_keys(src: str, name: str) -> Set[str]:
    """String keys of a module-level dict literal, without importing it."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if name not in [t.id for t in node.targets if isinstance(t, ast.Name)]:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _list_items(src: str, name: str) -> Set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if name not in [t.id for t in node.targets if isinstance(t, ast.Name)]:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        return {e.value for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def registered_losses(models_src: str) -> Set[str]:
    """Keys of the ``LOSSES`` dict literal, read without importing torch."""
    return _dict_keys(models_src, "LOSSES")


def registered_models(train_src: str, models_src: str = "") -> Set[str]:
    """Every selectable ``--model`` name.

    Two sources, because the architectures live in two places for a reason that
    is not going away: ``MODELS`` in models.py holds the torch architectures,
    while ``fm`` (numpy) and ``lgb`` (LightGBM) have their own trainers and are
    listed in ``ARCHS``. Both are read statically — importing models.py here
    would pull torch into the orchestrator process, where it cannot coexist with
    LightGBM.
    """
    return _list_items(train_src, "ARCHS") | _dict_keys(models_src, "MODELS")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are the Software Engineer on an autonomous ML research team working on
the KuaiRand-Pure within-user ranking benchmark. You implement a research
hypothesis by EDITING SOURCE CODE.

THE BENCHMARK
- Label `long_view`; metrics GAUC and nDCG@5; primary = their mean.
- Official baseline validation primary 0.6016. Oracle ceiling 0.8484, not 1.0.
- Seed noise sigma = 0.0008, so a gain under 0.0024 is not evidence.
- Ranking is WITHIN one user's own impression list. A feature that is constant
  across a user's impressions cannot change their order, and cannot help.

WHAT YOU MAY EDIT
  pipeline/models.py     architectures and loss functions
  pipeline/features.py   encoders, causal statistics, sequence extraction
  pipeline/train.py      optimiser, schedule, batching, early stopping

WHAT YOU MAY NOT EDIT (restored and hash-checked before every run)
  pipeline/data.py       the hidden-test seal
  pipeline/evaluate.py   the official scorer
  pipeline/submit.py, pipeline/models_np.py

CONTRACT — code that violates any of these is rejected before it runs
1. Never read the hidden test labels. Do not reference `load_test_labels`,
   `load_unbiased_valid`, `RANKAGENT_UNSEAL_TEST`, or `splits['test']`.
2. Never derive a feature from a post-impression outcome: `play_time_ms`,
   `profile_stay_time`, `comment_stay_time`, or any of the label/aux fields of
   the row being scored. These are withheld on the hidden test split, so a
   feature built from them scores brilliantly in development and is garbage at
   submission time. `duration_ms`, `tab`, `is_rand` and the ids are safe —
   they are known before the impression happens.
3. Target statistics must stay causal: a row's own label must never reach its
   own features. `CausalStats` is fed by expanding window over dates; keep it.
4. Do NOT add `import torch` or `import lightgbm` at module scope in
   pipeline/train.py. They vendor conflicting OpenMP runtimes and segfault when
   both load in one process. Import inside the function that needs them.
5. `build_parser()` must keep working, every trainer must still print exactly
   one `[EVAL]` line, and the existing LOSSES keys (pointwise, listwise, bpr)
   and model classes must keep working.
6. Imports: numpy, torch, lightgbm, scipy and the standard library. There is no
   network access.

REGISTERING NEW WORK
Both registries live in pipeline/models.py, and adding an entry is all that is
needed — the argument parser has no separate list to update and there is no
dispatch chain to edit.

A NEW LOSS: a function decorated with `@ranking_loss(requires_groups=...)`, plus
one entry in the `LOSSES` dict. It becomes `--loss <key>` immediately.

    def my_loss(logits, labels, group, n_groups) -> Tensor

SHAPES — most generated losses fail here, so read carefully:
  logits    (N,) float, RAW scores for N impressions. Not probabilities.
  labels    (N,) float, 0.0 or 1.0.
  group     (N,) long. group[i] is which user-list row i belongs to, a value in
            [0, n_groups). Rows of the SAME user share a value. The batch is a
            flat concatenation of several users' lists — it is NOT (users, items),
            and there is no padding.
  n_groups  int, the number of distinct lists in this batch.
  returns   a 0-dim scalar to be MINIMISED.

Because the batch is flat and ragged, `logits.view(n_groups, -1)` WILL crash —
users have different list lengths. Two correct tools are already in the file:

  _segment_logsumexp(logits, group, n_groups) -> (n_groups,)
      numerically stable per-group log-sum-exp.

  group_padded(values, group, n_groups, pad_value) -> (padded, mask)
      separates the flat batch into a (n_groups, max_len) matrix and a bool
      mask of which slots are real. THIS is how you sort or rank within a
      user — which every nDCG-family loss needs. Pad with 0.0 for labels and
      float('-inf') for scores you are about to sort or softmax, so the pads
      fall to the end and contribute nothing. Reduce with
      `(x * mask).sum(dim=1)`, and divide by `mask.sum(dim=1).clamp(min=1)`.

Do NOT hand-roll the regrouping. `labels[group == torch.arange(n_groups).
unsqueeze(1)]` is the mistake that keeps being made: it builds an (n_groups, N)
mask and indexes a 1-D tensor with it, raising IndexError. Call `group_padded`.
`torch.argsort` over the flat tensor sorts across users and is never what you
want.

Guard the degenerate cases the way the existing losses do: a batch with no
positives must return `logits.sum() * 0.0` rather than dividing by zero, which
keeps the graph connected and the gradient finite.

Read `listwise_softmax` and `bpr_pairwise` in the file below before writing —
they are correct, and they show both patterns.

A NEW ARCHITECTURE follows below.

A NEW ARCHITECTURE: an `nn.Module` class, plus a builder decorated with
`@architecture(needs_history=...)`, plus one entry in the `MODELS` dict. It
becomes `--model <key>` immediately. The builder signature is:
    build_x(rows, n_fields, embed_dim, pad_id, **kw) -> nn.Module
`rows` is the size of the shared embedding table and `n_fields` the number of
categorical fields. The module's `forward(x_cat)` takes a (B, n_fields) long
tensor of ids and returns (B,) raw logits — no sigmoid. With
`needs_history=True` the table gains the reserved padding row and forward takes
`(x_cat, x_hist)`, where `x_hist` is (B, max_seq_len) of past video ids padded
with `pad_id`.

OUTPUT FORMAT
Reply with one or more SEARCH/REPLACE blocks and nothing else except a short
sentence of explanation. Each block is:

pipeline/models.py
<<<<<<< SEARCH
(text copied EXACTLY from the current file, enough to be unique)
=======
(the replacement text)
>>>>>>> REPLACE

The SEARCH text must match the current file character for character. Keep each
block small and anchored on something distinctive.

If you cannot produce a reliable anchor — typically when fixing a file you have
already edited — reply instead with the COMPLETE corrected file in a single
```python fenced block, naming the file on its own line before it. Do not
abbreviate or elide any part of it; a truncated file is rejected."""


def build_prompt(hypothesis, target_files: Sequence[str],
                 sources: Dict[str, str],
                 lineage: str = "",
                 problems: Optional[Sequence[str]] = None,
                 traceback_text: str = "") -> str:
    """The volatile half of the Engineer's prompt."""
    parts: List[str] = []

    if lineage:
        parts.append(f"CURRENT CODE STATE\n  {lineage}\n")

    parts.append(
        "HYPOTHESIS TO IMPLEMENT\n"
        f"  dimension : {hypothesis.dimension}\n"
        f"  claim     : {hypothesis.hypothesis}\n"
        f"  mechanism : {hypothesis.mechanism or '(none given)'}\n"
        f"  sketch    : {getattr(hypothesis, 'edit_sketch', '') or '(none given)'}\n"
        f"  will run  : python -m pipeline.train {hypothesis.args}\n")

    others = [p for p in sources if p not in target_files]
    if others:
        parts.append("THE REST OF THE MUTABLE SURFACE (outline only)\n" +
                     "\n\n".join(source_digest(p, sources[p]) for p in sorted(others)))

    for p in target_files:
        parts.append(f"FULL CURRENT CONTENTS OF {p}\n"
                     f"```python\n{sources[p]}\n```")

    if traceback_text:
        parts.append("THE PREVIOUS ATTEMPT FAILED WHEN RUN:\n"
                     f"```\n{traceback_text[-2500:]}\n```")

    if problems:
        parts.append("YOUR PREVIOUS REPLY WAS REJECTED:\n" +
                     "\n\n".join(f"  - {p}" for p in problems) +
                     "\n\nRe-issue corrected SEARCH/REPLACE blocks.")

    parts.append("Implement the hypothesis now. Reply with SEARCH/REPLACE blocks only.")
    return "\n\n".join(parts)
