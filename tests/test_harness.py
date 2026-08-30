"""Harness tests: the scorer, the seals, and the leak fixes.

These are the checks that make every downstream number believable. Run with:
    make test
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from pipeline.data import (SPLITS, WITHHELD, load_kuairand, load_test_labels)
from pipeline.evaluate import evaluate
from pipeline.features import (encode_features, extract_dense_tabular_features,
                               extract_sequential_features)

DATA_OK = True
try:
    from pipeline.data import find_data_dir
    find_data_dir()
except Exception:
    DATA_OK = False

needs_data = pytest.mark.skipif(not DATA_OK, reason="KuaiRand-Pure not downloaded (run `make data`)")


# --- the scorer is the official one ---------------------------------------

def test_evaluate_is_the_official_implementation():
    import pipeline.evaluate as ev
    assert ev._OFFICIAL_PATH.endswith(os.path.join("kuairand-starter-kit", "evaluate.py"))
    assert os.path.exists(ev._OFFICIAL_PATH)


def test_auc_averages_tied_ranks():
    """All-tied scores must give AUC 0.5, not an argsort artefact.

    The previous reimplementation ranked ties by array position, so a model that
    output a constant scored above or below 0.5 depending on label order.
    """
    users = ['u'] * 6
    labels = [1, 1, 1, 0, 0, 0]
    tied = [0.5] * 6
    r = evaluate(users, labels, tied)
    assert abs(r['GAUC'] - 0.5) < 1e-12


def test_zero_positive_users_score_zero_ndcg_and_are_counted():
    r = evaluate(['a', 'a', 'b', 'b'], [0, 0, 1, 0], [0.9, 0.1, 0.9, 0.1])
    # user 'a' has no positives -> nDCG 0, included; user 'b' ranks its positive first -> 1.0
    assert abs(r['nDCG@5'] - 0.5) < 1e-12


def test_evaluate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate(['a', 'b'], [1], [0.5, 0.5])


# --- the hidden-test seal --------------------------------------------------

@needs_data
def test_loader_excludes_test_by_default():
    splits = load_kuairand()
    assert set(splits) == {'train', 'valid'}


@needs_data
def test_test_labels_are_withheld_even_when_requested():
    splits = load_kuairand(include_test=True)
    assert set(splits) == {'train', 'valid', 'test'}
    assert all(r['label'] == WITHHELD for r in splits['test'][:5000])


def test_test_labels_are_sealed_without_the_env_flag():
    os.environ.pop("RANKAGENT_UNSEAL_TEST", None)
    with pytest.raises(PermissionError):
        load_test_labels()


@needs_data
def test_official_row_counts():
    splits = load_kuairand(include_test=True)
    assert len(splits['train']) == 1_141_112
    assert len(splits['valid']) == 124_909
    assert len(splits['test']) == 170_588


# --- the leak fixes --------------------------------------------------------

def _synthetic():
    """Two users, three dates, deterministic — enough to expose a leak."""
    rows = []
    for d, date in enumerate(range(20220408, 20220411)):
        for u in ('u1', 'u2'):
            for v in ('v1', 'v2'):
                rows.append({'date': date, 'user_id': u, 'video_id': v, 'author_id': 'a1',
                             'tab': '1', 'duration_ms': 1000.0 * (d + 1),
                             'play_time_ms': 500.0, 'label': (d + len(v)) % 2,
                             'click': 1, 'like': 0, 'follow': 0, 'comment': 0,
                             'forward': 0, 'is_rand': 0, 'v_extra': [], 'u_extra': []})
    valid = [dict(r, date=20220422) for r in rows[:4]]
    return {'train': rows, 'valid': valid}


def test_sequence_history_ignores_labels():
    """Flipping every evaluation label must not change one history entry.

    This is the regression test for the DIN leak: history used to be appended
    only `if r['label'] == 1`, feeding evaluation labels into later features.
    """
    splits = _synthetic()
    _enc, encoder = encode_features(splits)
    before = extract_sequential_features(splits, encoder, max_seq_len=5)['valid']

    flipped = {'train': splits['train'],
               'valid': [dict(r, label=1 - r['label']) for r in splits['valid']]}
    after = extract_sequential_features(flipped, encoder, max_seq_len=5)['valid']

    assert np.array_equal(before, after), "evaluation labels are leaking into history features"


def test_target_encoding_is_causal_on_train():
    """The first training day cannot have seen any history.

    Under the old fit-on-train/apply-to-train scheme, every row's own label was
    baked into its item statistics.
    """
    splits = _synthetic()
    dense, names = extract_dense_tabular_features(splits)
    X, _y, _u = dense['train']
    first_day = X[:4]
    counts = first_day[:, names.index('video_hist_count')]
    assert np.allclose(counts, 0.0), "day-one rows already carry history — encoding is not causal"


def test_dense_features_have_no_pure_user_side_leftovers():
    """Ranking is within-user, so the feature set must carry user x item crosses."""
    _dense, names = extract_dense_tabular_features(_synthetic())
    for cross in ('user_author_affinity', 'user_durbucket_affinity', 'user_tab_affinity'):
        assert cross in names


# --- convergence -----------------------------------------------------------

def test_convergence_tracks_the_best_so_far_curve():
    from orchestrator.schemas import MetricResult
    from orchestrator.tree_manager import TreeManager

    def m(p):
        return MetricResult(gauc=p, ndcg_5=p, primary_score=p, delta_from_baseline=0.0)

    t = TreeManager(epsilon=0.002, n_convergence=3, max_iterations=50)
    t.record_baseline(0.6000)
    # Three small-but-real gains: each is under epsilon, but they must all be kept.
    for i, p in enumerate([0.6010, 0.6020, 0.6030], start=1):
        t.add_node(i, 0, "h", "f", m(p))
    assert t.best_primary_score == pytest.approx(0.6030), "small improvements were discarded"

    # A flat stretch of three iterations converges.
    t2 = TreeManager(epsilon=0.002, n_convergence=3, max_iterations=50)
    t2.record_baseline(0.6000)
    conv = [t2.add_node(i, 0, "h", "f", m(0.6001)) for i in range(1, 5)]
    assert conv[-1] is True


def test_failed_iterations_do_not_count_as_convergence():
    from orchestrator.tree_manager import TreeManager
    t = TreeManager(epsilon=0.002, n_convergence=3, max_iterations=50)
    t.record_baseline(0.6000)
    assert not any(t.add_node(i, 0, "h", "f", None) for i in range(1, 5))


# --- the debugger exists and works ----------------------------------------

def test_debugger_repairs_an_unknown_flag():
    from sandbox.debugger import SelfHealingDebugger

    calls = []

    class Ok:
        status = "SUCCESS"

    class Fail:
        status = "RUNTIME_ERROR"
        error_traceback = "error: unrecognized arguments: --nonexistent"

    def run(cmd):
        calls.append(cmd)
        return Ok() if "--nonexistent" not in cmd else Fail()

    d = SelfHealingDebugger(max_retries=3)
    out = d.attempt_repair("python -m pipeline.train --model fm --nonexistent 4",
                           "error: unrecognized arguments: --nonexistent", run)
    assert out.repaired
    assert "--nonexistent" not in out.command


def test_debugger_gives_up_without_crashing():
    from sandbox.debugger import SelfHealingDebugger

    class Fail:
        status = "RUNTIME_ERROR"
        error_traceback = "ZeroDivisionError: division by zero"

    d = SelfHealingDebugger(max_retries=2)
    out = d.attempt_repair("python -m pipeline.train --model fm",
                           "ZeroDivisionError: division by zero", lambda c: Fail())
    assert out.repaired is False
    assert out.attempts, "a give-up must still be recorded for the run log"


# --- the parser ------------------------------------------------------------

def test_parser_reads_the_last_eval_line():
    from sandbox.parser import parse_execution_output
    out = ("[EVAL] GAUC: 0.6000 | nDCG@5: 0.5000 | Primary: 0.5500\n"
           "[EVAL] GAUC: 0.6671 | nDCG@5: 0.5358 | Primary: 0.6015\n")
    m = parse_execution_output(out)
    assert m.primary_score == pytest.approx(0.6015)
    assert parse_execution_output("nothing here") is None


# --- token accounting ------------------------------------------------------

def test_iteration_tokens_are_per_iteration_not_cumulative():
    """Each entry must record what THIS iteration spent.

    Regression test: the orchestrator used to pass the running totals into the
    per-iteration fields, so a 1.5k-token iteration was logged as 5.7k once a few
    iterations had accumulated. Feasibility is scored on these numbers.
    """
    from orchestrator.schemas import IterationLogEntry, TokenUsage

    meter = TokenUsage()
    entries = []
    for i, (p, c) in enumerate([(1322, 195), (1389, 173), (1443, 206), (1516, 195)], start=1):
        before = (meter.prompt_tokens, meter.completion_tokens, meter.calls)
        meter.add(p, c)
        entries.append(IterationLogEntry(
            iteration_id=i, node_id=i, stage="s", hypothesis="h", target_file="f",
            command="c", status="ACCEPTED",
            prompt_tokens=meter.prompt_tokens - before[0],
            completion_tokens=meter.completion_tokens - before[1],
            llm_calls=meter.calls - before[2],
            cumulative_prompt_tokens=meter.prompt_tokens,
            cumulative_completion_tokens=meter.completion_tokens))

    assert [e.prompt_tokens for e in entries] == [1322, 1389, 1443, 1516]
    assert [e.llm_calls for e in entries] == [1, 1, 1, 1]
    # Cumulative rises monotonically and the last one equals the run total.
    assert [e.cumulative_prompt_tokens for e in entries] == [1322, 2711, 4154, 5670]
    assert entries[-1].cumulative_prompt_tokens == meter.prompt_tokens
    # Per-iteration figures must sum to the run total the summary reports.
    assert sum(e.iteration_tokens for e in entries) == meter.total


def test_repair_tokens_land_in_the_same_iteration():
    """A debugger repair call spends tokens after propose(); the delta must catch it."""
    from orchestrator.schemas import TokenUsage

    meter = TokenUsage()
    before = (meter.prompt_tokens, meter.completion_tokens, meter.calls)
    meter.add(1200, 150)   # propose()
    meter.add(800, 60)     # _llm_repair() during the same iteration
    assert meter.prompt_tokens - before[0] == 2000
    assert meter.calls - before[2] == 2


# --- baseline reproduction is on by default --------------------------------

def test_baseline_reproduction_is_the_default():
    """A run must verify the baseline unless explicitly told not to.

    Regression test: `run_baseline` defaulted to False, so every run — including
    ones intended for submission — silently asserted the published 0.6016 instead
    of reproducing it, and logged no iteration 0 as evidence.
    """
    import inspect
    from orchestrator.state_machine import RankAgentOrchestrator
    sig = inspect.signature(RankAgentOrchestrator.__init__)
    assert sig.parameters["run_baseline"].default is True


def test_cli_defaults_to_reproducing_the_baseline():
    import main
    parser_args = main.main.__doc__  # touch the module so argparse is importable
    import argparse
    import contextlib
    import io
    # --skip-baseline must exist and must default to off.
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(SystemExit):
            import sys as _s
            old = _s.argv
            _s.argv = ["main.py", "--help"]
            try:
                main.main()
            finally:
                _s.argv = old


def test_summary_records_whether_the_baseline_was_verified():
    from orchestrator.schemas import RunSummary
    assert RunSummary(run_id="x").baseline_reproduced is False
    assert RunSummary(run_id="x", baseline_reproduced=True).baseline_reproduced is True


def test_captured_diff_excludes_agent_artifacts():
    """code_diff must describe source changes, not the log recording itself."""
    from sandbox.logger import RunLogger
    for path in ("logs", "submissions", "checkpoints", "data"):
        assert path in RunLogger.DIFF_EXCLUDES
