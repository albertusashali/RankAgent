"""Build and validate the KuaiRand-Pure submission.

This is the *only* module that touches the hidden-test split, and it touches it
in the one legitimate way: it scores test rows whose labels the loader has
withheld. It never reads a test label, and it never chooses anything — the
checkpoint it exports has already been selected on validation.

Submission format (starter kit, pinned)::

    row_id,user_id,video_id,score

``row_id`` is a 0-based, strictly increasing index into ``data.load()[split]``.
It is required because ``(user_id, video_id)`` is *not* unique in the evaluation
split — 3.06% of test rows are repeated pairs, up to 12 times — so the redundant
id columns exist only to prove the file is aligned, not to key it.
"""
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from pipeline.data import load_kuairand
from pipeline.evaluate import evaluate
from pipeline.features import (encode_features, extract_dense_tabular_features,
                               extract_sequential_features)

HEADER = ['row_id', 'user_id', 'video_id', 'score']
CHECKPOINTS_DIR = "checkpoints"
SUBMISSIONS_DIR = "submissions"


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------

def write_submission(rows: List[dict], scores: Sequence[float], path: str) -> str:
    if len(rows) != len(scores):
        raise ValueError(f"length mismatch: {len(rows)} rows vs {len(scores)} scores")
    arr = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{bad} score(s) are NaN or Inf; the official checker rejects these")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (r, s) in enumerate(zip(rows, arr)):
            w.writerow([i, r['user_id'], r['video_id'], f"{float(s):.6g}"])
    print(f"[SUBMIT] wrote {len(rows):,d} rows to {path}")
    return path


def check_submission(path: str, rows: List[dict]) -> List[float]:
    """Full alignment check, matching the starter kit's ``submit.py --check``.

    The previous version validated the header, row count, ``row_id`` continuity
    and NaN/Inf, but never checked ``user_id``/``video_id`` against the evaluation
    split — so a correctly-shaped file built from mis-ordered predictions would
    pass here and score as noise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"submission not found: {path}")
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f"header must be {','.join(HEADER)}, got {head}")
        scores: List[float] = []
        n = 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"line {ln}: {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            if n >= len(rows):
                raise ValueError(f"line {ln}: more rows than the evaluation split ({len(rows)})")
            if int(rid) != n:
                raise ValueError(f"line {ln}: row_id={rid}, expected {n} (0-based, contiguous)")
            if uid != rows[n]['user_id'] or vid != rows[n]['video_id']:
                raise ValueError(
                    f"line {ln}: misaligned — file has ({uid},{vid}), split row {n} is "
                    f"({rows[n]['user_id']},{rows[n]['video_id']})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"line {ln}: score {sc!r} is not a number")
            if v != v or v in (float('inf'), float('-inf')):
                raise ValueError(f"line {ln}: score is NaN/Inf, which is rejected")
            scores.append(v)
            n += 1
    if n != len(rows):
        raise ValueError(f"submission has {n} rows, evaluation split has {len(rows)}")
    print(f"[VALIDATION PASS] {path}: {n:,d} rows, header, row_id and alignment all correct")
    return scores


# ---------------------------------------------------------------------------
# Scoring a split from a checkpoint
# ---------------------------------------------------------------------------

def load_meta(name: str) -> dict:
    path = os.path.join(CHECKPOINTS_DIR, f"{name}.meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no metadata for checkpoint {name!r} at {path}. Train it first — the "
            "submission step rebuilds the model from its recorded config rather "
            "than assuming default hyperparameters.")
    with open(path) as fh:
        return json.load(fh)


def predict_split(name: str, splits: Dict[str, List[dict]], split: str) -> np.ndarray:
    """Score ``split`` with the checkpoint ``name``, rebuilt from its own metadata.

    Reconstructing from ``<name>.meta.json`` is what keeps this honest: the old
    implementation hard-coded ``MMoE(embed_dim=16, num_experts=4)`` and failed to
    load any run that used different hyperparameters.
    """
    meta = load_meta(name)
    arch = meta["model"]

    if arch == 'lgb':
        import lightgbm as lgb
        dense, _names = extract_dense_tabular_features(splits)
        X = dense[split][0]
        booster = lgb.Booster(model_file=os.path.join(CHECKPOINTS_DIR, "lgb.txt"))
        return np.asarray(booster.predict(X), dtype=np.float64)

    enc, encoder = encode_features(splits, use_cwm_fields=meta.get("use_cwm", False))
    X = enc[split][0]

    if arch == 'fm':
        from pipeline.models_np import NumpyFM
        data = np.load(os.path.join(CHECKPOINTS_DIR, "fm.npz"))
        m = NumpyFM(encoder.total_dim, k=meta.get("k", 16))
        m.V, m.W, m.b = data['V'], data['W'], data['b']
        return np.asarray(m.predict(X), dtype=np.float64)

    import torch
    from pipeline.models import DEVICE, MMoE, resolve_model
    from pipeline.train import _predict_torch

    dim = meta.get("embed_dim", 16)
    n_fields = len(encoder.field_names)
    hist = None

    if arch == 'mmoe':
        # MMoE has its own trainer (multiple heads, auxiliary tasks), so it is
        # not in the single-head architecture registry.
        from pipeline.train import AUX_TASKS
        model = MMoE(encoder.total_dim, n_fields, embed_dim=dim,
                     num_experts=meta.get("num_experts", 4),
                     expert_dim=meta.get("expert_dim", 64),
                     num_tasks=1 + len(AUX_TASKS))
    else:
        # Rebuild through the SAME builder that training used. Inference used to
        # carry its own copy of the dispatch chain, so an architecture the agent
        # registered could train and score on validation and then fail at
        # submission time — the one moment there is no chance to recover.
        builder = resolve_model(arch)
        needs_hist = getattr(builder, 'needs_history', False)
        if needs_hist:
            seqs = extract_sequential_features(
                splits, encoder, max_seq_len=meta.get("max_seq_len", 10))
            hist = seqs[split]
        rows = encoder.embedding_rows if needs_hist else encoder.total_dim
        model = builder(rows, n_fields, dim, encoder.pad_id)

    ckpt = os.path.join(CHECKPOINTS_DIR, f"{name}.pt")
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return _predict_torch(model.to(DEVICE), X, hist)


def prediction_cache_path(name: str, split: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"{name}.{split}.npy")


def export_predictions(name: str, splits: Dict[str, List[dict]], split: str) -> np.ndarray:
    """Score a split and cache it to disk.

    Caching is not an optimisation, it is what makes ensembling possible at all:
    LightGBM and PyTorch cannot be loaded into the same process (conflicting
    OpenMP runtimes, see pipeline/models_np.py), so a blend of a GBDT and a neural
    model must score each in its own process and combine the saved arrays.
    """
    preds = predict_split(name, splits, split)
    np.save(prediction_cache_path(name, split), preds)
    print(f"[EXPORT] cached {len(preds):,d} {split} predictions for {name}")
    return preds


def load_cached(name: str, split: str) -> np.ndarray:
    path = prediction_cache_path(name, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cached {split} predictions for {name!r}. Run:\n"
            f"  python -m pipeline.submit --export --checkpoint {name}")
    return np.load(path)


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] by rank — a monotone transform, so within-user order
    is preserved while different models are put on a comparable scale."""
    order = np.argsort(np.argsort(x, kind='stable'), kind='stable')
    return order / max(len(x) - 1, 1)


def blend_weight_on_valid(names: Sequence[str], splits: Dict[str, List[dict]],
                          grid: int = 21) -> tuple:
    """Choose a two-model blend weight on validation only."""
    if len(names) != 2:
        raise ValueError("blending currently supports exactly two checkpoints")
    yva = [r['label'] for r in splits['valid']]
    uva = [r['user_id'] for r in splits['valid']]
    a = _rank_normalise(load_cached(names[0], 'valid'))
    b = _rank_normalise(load_cached(names[1], 'valid'))
    best_w, best_p = 1.0, -1.0
    for w in np.linspace(0.0, 1.0, grid):
        p = evaluate(uva, yva, w * a + (1 - w) * b)['primary']
        if p > best_p:
            best_w, best_p = float(w), p
    print(f"[BLEND] best weight {best_w:.2f} for {names[0]} vs {names[1]} "
          f"-> valid primary {best_p:.4f}")
    return best_w, best_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_submission(names: Sequence[str], out_path: str, data_dir: Optional[str] = None,
                     weight: Optional[float] = None) -> str:
    """Export the validation-selected checkpoint(s) as a test submission."""
    splits = load_kuairand(data_dir, include_test=True)
    test_rows = splits['test']

    if len(names) == 1:
        scores = predict_split(names[0], splits, 'test')
    else:
        if weight is None:
            weight, _ = blend_weight_on_valid(names, splits)
        a = _rank_normalise(load_cached(names[0], 'test'))
        b = _rank_normalise(load_cached(names[1], 'test'))
        scores = weight * a + (1 - weight) * b

    path = write_submission(test_rows, scores, out_path)
    check_submission(path, test_rows)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Build or validate a KuaiRand-Pure submission")
    p.add_argument('--data_dir', default=None)
    p.add_argument('--file', default=os.path.join(SUBMISSIONS_DIR, "kuairand_pure_final.csv"))
    p.add_argument('--checkpoint', nargs='+', default=None,
                   help="checkpoint name(s) under checkpoints/, e.g. fm_torch_listwise")
    p.add_argument('--weight', type=float, default=None,
                   help="blend weight for the first checkpoint; tuned on valid if omitted")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--generate', action='store_true')
    g.add_argument('--check', action='store_true')
    g.add_argument('--export', action='store_true',
                   help="cache valid+test predictions for one checkpoint (run once per model)")
    a = p.parse_args(argv)

    if a.export:
        if not a.checkpoint:
            p.error("--export needs --checkpoint")
        splits = load_kuairand(a.data_dir, include_test=True)
        for name in a.checkpoint:
            for split in ('valid', 'test'):
                export_predictions(name, splits, split)
    elif a.generate:
        if not a.checkpoint:
            p.error("--generate needs --checkpoint")
        build_submission(a.checkpoint, a.file, a.data_dir, a.weight)
    else:
        splits = load_kuairand(a.data_dir, include_test=True)
        check_submission(a.file, splits['test'])


if __name__ == '__main__':
    main()
