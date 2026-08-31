"""The code-generation path, exercised with no API key and no training.

Every patch here is hand-written, so these run in milliseconds and are
deterministic. They cover the three things that have to hold for a generated
patch to be worth anything:

  * it applies, or the failure explains itself well enough to be retried;
  * a registered loss or architecture is immediately *reachable* from the CLI —
    the whole point of removing the duplicated `choices=` lists;
  * the immutable surface stays immutable, and a leak is caught before any
    training happens.
"""
import os
import shutil

import pytest

from agents.codegen import registered_losses, registered_models, source_digest
from agents.patch import Edit, apply_all, parse_edit_blocks
from sandbox.workspace import MUTABLE, materialise, unified_diff

WS_DIR = "workspaces_test"


@pytest.fixture
def ws():
    if os.path.exists(WS_DIR):
        shutil.rmtree(WS_DIR)
    w = materialise(900, workspaces_dir=WS_DIR)
    yield w
    shutil.rmtree(WS_DIR, ignore_errors=True)


# --- golden patches: what the Engineer is expected to produce ---------------

ADD_LOSS = """pipeline/models.py
<<<<<<< SEARCH
LOSSES = {
    'pointwise': pointwise_bce,
=======
@ranking_loss(requires_groups=True)
def approx_ndcg(logits, labels, group, n_groups, tau=1.0):
    \"\"\"Sigmoid-smoothed nDCG surrogate (Qin et al. 2010).\"\"\"
    return -(logits * labels).sum() / (labels.sum() + 1e-9)


LOSSES = {
    'approx_ndcg': approx_ndcg,
    'pointwise': pointwise_bce,
>>>>>>> REPLACE"""

ADD_MODEL = """pipeline/models.py
<<<<<<< SEARCH
MODELS = {
    'fm_torch': build_fm_torch,
=======
@architecture(needs_history=False)
def build_wide(rows, n_fields, embed_dim, pad_id, **kw):
    return TorchFM(rows, n_fields, embed_dim * 2)


MODELS = {
    'wide': build_wide,
    'fm_torch': build_fm_torch,
>>>>>>> REPLACE"""


def test_a_generated_loss_becomes_selectable(ws):
    """The registry unblock: adding to LOSSES is enough to reach --loss."""
    before = ws.snapshot()
    assert "approx_ndcg" not in registered_losses(before["pipeline/models.py"])

    edits, errors = parse_edit_blocks(ADD_LOSS, MUTABLE)
    assert not errors and len(edits) == 1
    after, problems = apply_all(before, edits)
    assert not problems, problems

    assert "approx_ndcg" in registered_losses(after["pipeline/models.py"])

    from agents.engineer import validate_args
    ok, why = validate_args("--model fm_torch --loss approx_ndcg",
                            known_losses=registered_losses(after["pipeline/models.py"]))
    assert ok, why
    # ...and it is still rejected against the *unpatched* source, so the check
    # is doing work rather than waving everything through.
    ok, _ = validate_args("--model fm_torch --loss approx_ndcg",
                          known_losses=registered_losses(before["pipeline/models.py"]))
    assert not ok


def test_a_generated_architecture_becomes_selectable(ws):
    before = ws.snapshot()
    edits, errors = parse_edit_blocks(ADD_MODEL, MUTABLE)
    assert not errors
    after, problems = apply_all(before, edits)
    assert not problems, problems

    names = registered_models(after["pipeline/train.py"], after["pipeline/models.py"])
    assert "wide" in names
    from agents.engineer import validate_args
    ok, why = validate_args("--model wide --loss listwise", known_models=names)
    assert ok, why


def test_the_diff_is_measured_not_claimed(ws):
    before = ws.snapshot()
    edits, _ = parse_edit_blocks(ADD_LOSS, MUTABLE)
    after, problems = apply_all(before, edits)
    assert not problems

    for rel, text in after.items():
        if before.get(rel) != text:
            ws.write(rel, text)

    diff, (files, added, removed) = unified_diff(ws.base, ws.snapshot())
    assert files == 1 and added > 0
    assert "approx_ndcg" in diff
    assert diff.startswith("--- a/pipeline/models.py")


def test_immutable_files_are_restored_and_the_attempt_reported(ws):
    with open(ws.path("pipeline/evaluate.py"), "a", encoding="utf-8") as fh:
        fh.write("\ndef evaluate(*a, **k):\n    return {'primary': 0.99}\n")

    violations = ws.restore_immutable()
    assert [v.path for v in violations] == ["pipeline/evaluate.py"]
    assert "return {'primary': 0.99}" not in ws.read("pipeline/evaluate.py")


def test_a_leaking_patch_is_blocked_before_it_runs(ws):
    """Verified end to end through the workspace, not just the scanner."""
    from sandbox.verifier import verify_workspace

    src = ws.read("pipeline/features.py")
    ws.write("pipeline/features.py",
             src + "\n\ndef watch_ratio(row):\n"
                   "    return row['play_time_ms'] / max(row['duration_ms'], 1.0)\n")

    report = verify_workspace(ws)
    assert not report.ok and report.fatal
    assert any("play_time_ms" in m for m in report.messages())


def test_a_clean_workspace_verifies(ws):
    from sandbox.verifier import verify_workspace
    report = verify_workspace(ws)
    assert report.ok, report.messages()


def test_patch_failures_explain_themselves():
    """A retry can only land if the message says where to re-anchor."""
    files = {"pipeline/models.py": "def a():\n    return 1\n\ndef b():\n    return 1\n"}

    _, problems = apply_all(files, [Edit("pipeline/models.py", "    return 1", "    return 2")])
    assert problems and "ambiguous" in problems[0]
    assert "around line" in problems[0], "must show context, not just line numbers"

    _, problems = apply_all(files, [Edit("pipeline/models.py", "def nowhere():", "x")])
    assert problems and "did not match" in problems[0]
    assert "closest regions" in problems[0]


def test_source_digest_is_a_cheap_outline():
    """The digest keeps prompts affordable across a long run."""
    src = open("pipeline/train.py", encoding="utf-8").read()
    digest = source_digest("pipeline/train.py", src)
    assert "def build_parser" in digest and "def main" in digest
    assert len(digest) < len(src) / 5, "the digest must be far smaller than the file"


def test_workspaces_compose_from_their_parent():
    """A child inherits its parent's edits — this is why changes stack."""
    if os.path.exists(WS_DIR):
        shutil.rmtree(WS_DIR)
    try:
        parent = materialise(1, workspaces_dir=WS_DIR)
        edits, _ = parse_edit_blocks(ADD_LOSS, MUTABLE)
        after, problems = apply_all(parent.snapshot(), edits)
        assert not problems
        parent.write("pipeline/models.py", after["pipeline/models.py"])

        child = materialise(2, parent=parent, workspaces_dir=WS_DIR)
        assert "approx_ndcg" in registered_losses(child.read("pipeline/models.py"))

        sibling = materialise(3, workspaces_dir=WS_DIR)
        assert "approx_ndcg" not in registered_losses(sibling.read("pipeline/models.py")), (
            "a node materialised from the repo must not inherit another "
            "branch's edits")
    finally:
        shutil.rmtree(WS_DIR, ignore_errors=True)


def test_a_new_loss_is_never_paired_with_a_trainer_that_ignores_it():
    """`fm` and `lgb` have their own trainers and never consult --loss.

    Pairing a newly implemented objective with either trains the stock model and
    reports the result as the new method's. A real run did this: it implemented
    focal loss, ran `--model=fm --loss=focal`, and scored 0.6015 — the baseline
    to four decimals, because the loss function was never called.
    """
    from agents.team import LOSS_IGNORING_MODELS, _set_flag

    assert LOSS_IGNORING_MODELS == {"fm", "lgb"}

    for bad in ("--model=fm --loss=focal", "--model fm --loss focal"):
        fixed = _set_flag(bad, "model", "fm_torch")
        assert "fm_torch" in fixed and "--model=fm " not in fixed
        assert "focal" in fixed, "the objective under test must survive"

    # Setting a flag that is absent appends it rather than corrupting the string.
    assert _set_flag("--model fm_torch", "loss", "focal") == \
        "--model fm_torch --loss focal"
    # Both spellings are handled, and only the first occurrence is replaced.
    assert _set_flag("--loss=bpr --epochs 5", "loss", "focal") == \
        "--loss focal --epochs 5"


def test_a_whole_file_reply_is_accepted_when_anchors_are_not_given():
    """The largest single cause of failed repairs was discarding these.

    Asking a model to "fix this file" often returns the whole corrected file
    rather than SEARCH/REPLACE edits. Treating that as malformed threw away work
    that was usually correct — four of five iterations in one run died this way.
    """
    from agents.patch import parse_whole_file

    current = {"pipeline/models.py": "\n".join(f"line{i}" for i in range(100)) + "\n"}
    body = "\n".join(f"line{i}" for i in range(100)) + "\nx = 1\n"

    out, problems = parse_whole_file(
        f"Here is the corrected pipeline/models.py:\n\n```python\n{body}```",
        ["pipeline/models.py"], current)
    assert not problems and out["pipeline/models.py"].endswith("x = 1\n")


def test_a_whole_file_reply_is_rejected_when_it_looks_wrong():
    """Accepting a whole file is riskier than an anchored edit, so it is guarded."""
    from agents.patch import parse_whole_file

    current = {"pipeline/models.py": "\n".join(f"line{i}" for i in range(100)) + "\n"}

    _, problems = parse_whole_file(
        "pipeline/models.py\n```python\nline0\nline1\n```",
        ["pipeline/models.py"], current)
    assert problems and "truncated" in problems[0]

    _, problems = parse_whole_file(
        "pipeline/models.py\n```python\ndef broken(:\n```",
        ["pipeline/models.py"], current)
    assert problems and "does not parse" in problems[0]

    _, problems = parse_whole_file("no code at all", ["pipeline/models.py"], current)
    assert problems


def test_smoke_rejects_a_result_below_the_random_floor():
    """Exit code zero is not enough to justify a full training run.

    A loss with a flipped sign trains happily to nonsense. One run reached the
    full trainer with a ListMLE scoring 0.3774 — under the 0.4834 a random
    scorer gets — because the gate only asked whether the process exited zero.
    """
    from agents.qa import RANDOM_FLOOR, QAAgent

    assert QAAgent().judge(0.3774).trustworthy is False
    assert 0.3774 < RANDOM_FLOOR < 0.6015


def test_group_padded_regroups_a_ragged_batch():
    """The helper that makes nDCG-family losses writable.

    A batch is a flat concatenation of users' lists, so sorting within a user
    needs the users separated first. Generated code kept doing that by hand as
    `labels[group == arange(n_groups).unsqueeze(1)]`, which indexes a 1-D tensor
    with a 2-D mask and raises IndexError. The harness provides it once instead.
    """
    torch = pytest.importorskip("torch")
    from pipeline.models import group_padded

    group = torch.tensor([0, 1, 2, 1, 0, 1, 1])
    values = torch.tensor([1., 10., 100., 11., 2., 12., 13.])

    padded, mask = group_padded(values, group, 3, pad_value=0.0)
    assert padded.shape == (3, 4)
    assert padded[0][:2].tolist() == [1., 2.]
    assert padded[1].tolist() == [10., 11., 12., 13.]
    assert padded[2][:1].tolist() == [100.]
    assert mask.sum().item() == 7

    # -inf pads sort to the end and vanish under softmax, so ranking losses can
    # use the padded matrix directly.
    scores, _ = group_padded(values, group, 3, pad_value=float("-inf"))
    ordered, _ = torch.sort(scores, dim=1, descending=True)
    assert ordered[2][0].item() == 100.0 and ordered[2][1].item() == float("-inf")
    assert bool(torch.softmax(scores, dim=1).isfinite().all())

    # Degenerate shapes must not crash: one group, and one row.
    solo, solo_mask = group_padded(torch.tensor([5.]), torch.tensor([0]), 1)
    assert solo.tolist() == [[5.0]] and solo_mask.tolist() == [[True]]
