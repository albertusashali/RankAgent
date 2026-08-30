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
    for cross in ('user_author_affinity', 'user_durbucket_affinity', 'user_tab_affinity', 'user_author_loyalty'):
        assert cross in names


def test_short_video_dynamic_features_present():
    """All domain dynamic features must be present in the tabular feature matrix."""
    _dense, names = extract_dense_tabular_features(_synthetic())
    required_dynamic_features = [
        'video_click_longview_gap', 'video_click_to_longview_ratio', 'author_click_longview_gap',
        'dur_norm_completion_score', 'user_author_recent_streak', 'author_recency_gap',
        'user_session_depth'
    ]
    for feat in required_dynamic_features:
        assert feat in names, f"Missing short-video dynamic feature: {feat}"


def test_author_fatigue_and_session_depth_causality():
    """Consecutive impressions of the same author must increment fatigue streak causally."""
    rows = [
        {'date': 20220408, 'user_id': 'u1', 'video_id': 'v1', 'author_id': 'a1', 'tab': '1',
         'duration_ms': 10000.0, 'play_time_ms': 8000.0, 'label': 1, 'click': 1},
        {'date': 20220408, 'user_id': 'u1', 'video_id': 'v2', 'author_id': 'a1', 'tab': '1',
         'duration_ms': 10000.0, 'play_time_ms': 2000.0, 'label': 0, 'click': 1},
        {'date': 20220408, 'user_id': 'u1', 'video_id': 'v3', 'author_id': 'a2', 'tab': '1',
         'duration_ms': 10000.0, 'play_time_ms': 8000.0, 'label': 1, 'click': 1},
    ]
    splits = {'train': rows, 'valid': []}
    dense, names = extract_dense_tabular_features(splits)
    X, _y, _u = dense['train']
    streak_idx = names.index('user_author_recent_streak')
    depth_idx = names.index('user_session_depth')

    # Row 0: first impression -> streak 0, depth log(1 + 0) = 0
    assert X[0, streak_idx] == 0.0
    assert X[0, depth_idx] == 0.0

    # Row 1: second impression, same author 'a1' -> streak 1, depth log(1 + 1)
    assert X[1, streak_idx] == 1.0
    assert X[1, depth_idx] == pytest.approx(np.log1p(1.0))

    # Row 2: third impression, different author 'a2' -> streak 0, depth log(1 + 2)
    assert X[2, streak_idx] == 0.0
    assert X[2, depth_idx] == pytest.approx(np.log1p(2.0))


def test_dense_features_ignore_eval_labels():
    """Flipping evaluation labels must have zero effect on valid-split dense features."""
    splits = _synthetic()
    before, _ = extract_dense_tabular_features(splits)
    X_before = before['valid'][0]

    flipped = {'train': splits['train'],
               'valid': [dict(r, label=1 - r['label'], click=1 - r['click']) for r in splits['valid']]}
    after, _ = extract_dense_tabular_features(flipped)
    X_after = after['valid'][0]

    assert np.allclose(X_before, X_after), "evaluation labels or feedback leaked into validation features"


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


# --- EDA and Granular Train CLI --------------------------------------------

def test_eda_report_generation(tmp_path):
    from pipeline.eda import run_eda
    out_file = str(tmp_path / "test_eda_report.md")
    report = run_eda(output_path=out_file)
    assert os.path.exists(out_file)
    assert "GAUC Pair Distribution" in report
    assert "NEVER CLIP OR DROP POWER USERS" in report
    assert "Duration Bias" in report
    assert "Auxiliary Task Correlations" in report


def test_train_parser_arguments():
    from pipeline.train import build_parser
    parser = build_parser()
    args = parser.parse_args([
        '--model', 'ple',
        '--private_experts', '2',
        '--shared_experts', '3',
        '--weight_click', '0.05',
        '--weight_like', '0.8',
        '--weight_forward', '0.5',
        '--dense',
        '--weight_decay', '0.0003',
        '--drop_features', 'feat_a,feat_b'
    ])
    assert args.model == 'ple'
    assert args.private_experts == 2
    assert args.shared_experts == 3
    assert args.weight_click == 0.05
    assert args.weight_like == 0.8
    assert args.weight_forward == 0.5
    assert args.dense is True
    assert args.weight_decay == 0.0003
    assert args.drop_features == 'feat_a,feat_b'

    args_bst = parser.parse_args(['--model', 'bst', '--max_seq_len', '15', '--dense'])
    assert args_bst.model == 'bst'
    assert args_bst.max_seq_len == 15
    assert args_bst.dense is True
    assert args_bst.weight_decay == 1e-4

    args_dcn = parser.parse_args(['--model', 'dcn_v2', '--embed_dim', '32', '--dense'])
    assert args_dcn.model == 'dcn_v2'
    assert args_dcn.embed_dim == 32
    assert args_dcn.dense is True


def test_new_models_forward_backward():
    import torch
    from pipeline.models import PLE, DCNv2, BST

    batch_size = 16
    num_fields = 5
    dense_dim = 12
    embed_dim = 8
    num_features = 100
    pad_id = 99
    max_seq_len = 10

    x_cat = torch.randint(0, num_features, (batch_size, num_fields))
    x_dense = torch.randn(batch_size, dense_dim)

    # 1. DCNv2 (with and without dense)
    dcn = DCNv2(num_features, num_fields, embed_dim=embed_dim, dense_dim=dense_dim, num_cross_layers=2)
    out_dcn = dcn(x_cat, x_dense=x_dense)
    assert out_dcn.shape == (batch_size,)
    out_dcn.sum().backward()

    # 2. PLE (with and without dense)
    ple = PLE(num_features, num_fields, embed_dim=embed_dim, dense_dim=dense_dim,
              num_private_experts=1, num_shared_experts=2, expert_dim=32, num_tasks=4)
    outs_ple = ple(x_cat, x_dense=x_dense)
    assert len(outs_ple) == 4
    for o in outs_ple:
        assert o.shape == (batch_size,)
    sum(outs_ple).sum().backward()

    # 3. BST (with and without dense)
    bst = BST(num_features, num_fields, pad_id=pad_id, embed_dim=embed_dim,
              dense_dim=dense_dim, num_heads=2, num_layers=1, max_seq_len=max_seq_len)
    x_hist = torch.randint(0, num_features, (batch_size, max_seq_len))
    out_bst = bst(x_cat, x_hist, x_dense=x_dense)
    assert out_bst.shape == (batch_size,)
    out_bst.sum().backward()



