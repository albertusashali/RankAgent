"""Training harness for KuaiRand-Pure.

Every trainer here obeys two rules that the challenge imposes:

  1. **Selection is on validation only.** Nothing in this file computes, prints or
     returns a hidden-test metric. Test rows arrive from the loader with their
     labels withheld, so a test score is not merely discouraged — it is
     unavailable. Test *predictions* are produced only by ``predict_test``,
     which the submission step calls after a checkpoint has been chosen.
  2. **The baseline is reproducible.** ``--model fm`` runs the numpy FM with the
     organizer's hyperparameters and must land on validation primary 0.6016.

Every run writes ``checkpoints/<model>.meta.json`` next to its weights, recording
the exact constructor arguments. The submission step rebuilds the model from that
file rather than guessing, which is what previously broke when a run used a
non-default embedding size.
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.data import load_kuairand
from pipeline.evaluate import evaluate, format_eval_line
from pipeline.features import (encode_features, extract_dense_tabular_features,
                               extract_sequential_features)
from pipeline.models_np import NumpyFM

# NOTE ON IMPORTS. Neither torch nor LightGBM is imported at module scope. They
# vendor conflicting OpenMP runtimes and segfault when both are loaded into one
# process, in either order, so each trainer imports only what it needs at call
# time. A process that trains the numpy baseline loads neither. Keep it that way:
# a module-level `import torch` here would break `--model lgb` again.

CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

#: Auxiliary MMoE tasks, in tower order after task 0 (long_view).
AUX_TASKS = ['click', 'like', 'forward']


class TrainResult(dict):
    """Validation metrics plus enough metadata to rebuild the model later."""

    @property
    def primary(self) -> float:
        return self['valid']['primary']


def _save_meta(name: str, meta: dict):
    with open(os.path.join(CHECKPOINTS_DIR, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def _finish(name: str, valid_metrics: dict, meta: dict) -> TrainResult:
    _save_meta(name, meta)
    print(format_eval_line(valid_metrics))
    return TrainResult(valid=valid_metrics, checkpoint=name, meta=meta)


# ---------------------------------------------------------------------------
# Grouped batching for the ranking losses
# ---------------------------------------------------------------------------

def _user_groups(users: List[str]) -> List[np.ndarray]:
    idx: Dict[str, List[int]] = {}
    for i, u in enumerate(users):
        idx.setdefault(u, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in idx.values()]


def _grouped_batches(groups: List[np.ndarray], rng: np.random.Generator,
                     target_rows: int, max_group: int = 64):
    """Yield ``(row_indices, group_ids)`` where each group is one user's impressions.

    Pointwise training can shuffle rows freely, but a within-user loss needs a
    user's impressions to arrive together. Oversized users are subsampled so one
    heavy user cannot dominate a batch.
    """
    order = rng.permutation(len(groups))
    rows: List[np.ndarray] = []
    gids: List[np.ndarray] = []
    n_rows = 0
    g = 0
    for gi in order:
        members = groups[gi]
        if len(members) > max_group:
            members = rng.choice(members, size=max_group, replace=False)
        if len(members) < 2:
            continue                       # a singleton list has no ordering to learn
        rows.append(members)
        gids.append(np.full(len(members), g, dtype=np.int64))
        n_rows += len(members)
        g += 1
        if n_rows >= target_rows:
            yield np.concatenate(rows), np.concatenate(gids), g
            rows, gids, n_rows, g = [], [], 0, 0
    if rows:
        yield np.concatenate(rows), np.concatenate(gids), g


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_torch(model, X: np.ndarray, hist: Optional[np.ndarray] = None,
                   bs: int = 65536) -> np.ndarray:
    import torch
    from pipeline.models import DEVICE
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(DEVICE)
            if hist is not None:
                hb = torch.from_numpy(hist[i:i + bs]).long().to(DEVICE)
                out.append(model(xb, hb).cpu().numpy())
            else:
                res = model(xb)
                out.append((res[0] if isinstance(res, tuple) else res).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Official baseline — numpy FM
# ---------------------------------------------------------------------------

def train_numpy_fm(splits: Dict, k: int = 16, lr: float = 0.001, epochs: int = 40,
                   bs: int = 8192, patience: int = 4, seed: int = 0,
                   use_cwm: bool = False, verbose: bool = True) -> TrainResult:
    enc, encoder = encode_features(splits, use_cwm_fields=use_cwm)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    m = NumpyFM(encoder.total_dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                  for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    m.V, m.W, m.b = best_state
    np.savez_compressed(os.path.join(CHECKPOINTS_DIR, "fm.npz"), V=m.V, W=m.W, b=m.b)
    return _finish("fm", evaluate(uva, yva, m.predict(Xva)),
                   {"model": "fm", "k": k, "lr": lr, "seed": seed, "use_cwm": use_cwm})


# ---------------------------------------------------------------------------
# 2. Torch single-task models (FM / DeepFM / DIN) with swappable loss
# ---------------------------------------------------------------------------

def train_torch(splits: Dict, arch: str = 'fm_torch', loss: str = 'listwise',
                embed_dim: int = 16, lr: float = 0.001, epochs: int = 20,
                bs: int = 8192, patience: int = 3, seed: int = 0,
                use_cwm: bool = False, weight_decay: float = 1e-6,
                max_seq_len: int = 10, verbose: bool = True) -> TrainResult:
    """Train an embedding model under a pointwise, pairwise or listwise objective."""
    import torch
    import torch.nn as nn
    from pipeline.models import DEVICE, LOSSES, DIN, DeepFM, TorchFM

    torch.manual_seed(seed)
    np.random.seed(seed)

    enc, encoder = encode_features(splits, use_cwm_fields=use_cwm)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    n_fields = len(encoder.field_names)

    needs_hist = (arch == 'din')
    if needs_hist:
        seqs = extract_sequential_features(splits, encoder, max_seq_len=max_seq_len)
        Htr, Hva = seqs['train'], seqs['valid']
    else:
        Htr = Hva = None

    # DIN indexes the reserved padding row, so its table is one row longer.
    rows = encoder.embedding_rows if needs_hist else encoder.total_dim
    if arch == 'fm_torch':
        model = TorchFM(rows, n_fields, embed_dim)
    elif arch == 'deepfm':
        model = DeepFM(rows, n_fields, embed_dim)
    elif arch == 'din':
        model = DIN(rows, n_fields, pad_id=encoder.pad_id, embed_dim=embed_dim)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model = model.to(DEVICE)

    loss_fn = LOSSES[loss]
    grouped = loss in ('listwise', 'bpr')
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if grouped else None

    best, best_state, bad = -1.0, None, 0
    if verbose:
        print(f"==> {arch} | loss={loss} | dim={embed_dim} | fields={n_fields} | {DEVICE}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        if grouped:
            batches = _grouped_batches(groups, rng, target_rows=bs)
        else:
            perm = rng.permutation(len(ytr))
            batches = ((perm[i:i + bs], None, 0) for i in range(0, len(perm), bs))

        for rows_idx, gids, n_groups in batches:
            xb = torch.from_numpy(Xtr[rows_idx]).long().to(DEVICE)
            yb = torch.from_numpy(ytr[rows_idx]).float().to(DEVICE)
            gb = torch.from_numpy(gids).long().to(DEVICE) if gids is not None else None
            logits = model(xb, torch.from_numpy(Htr[rows_idx]).long().to(DEVICE)) \
                if needs_hist else model(xb)

            opt.zero_grad()
            l = loss_fn(logits, yb, gb, n_groups)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(l.item())

        va = evaluate(uva, yva, _predict_torch(model, Xva, Hva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    name = f"{arch}_{loss}"
    torch.save(best_state, os.path.join(CHECKPOINTS_DIR, f"{name}.pt"))
    return _finish(name, evaluate(uva, yva, _predict_torch(model.to(DEVICE), Xva, Hva)),
                   {"model": arch, "loss": loss, "embed_dim": embed_dim, "lr": lr,
                    "seed": seed, "use_cwm": use_cwm, "max_seq_len": max_seq_len})


# ---------------------------------------------------------------------------
# 3. Multi-task MMoE
# ---------------------------------------------------------------------------

def train_mmoe(splits: Dict, embed_dim: int = 16, num_experts: int = 4,
               expert_dim: int = 64, lr: float = 0.001, epochs: int = 20,
               bs: int = 8192, patience: int = 3, seed: int = 0,
               loss: str = 'listwise', aux_weight: float = 0.3,
               weight_click: Optional[float] = None, weight_like: Optional[float] = None,
               weight_forward: Optional[float] = None,
               use_cwm: bool = False, verbose: bool = True) -> TrainResult:
    """Joint long_view + auxiliary feedback training.

    The scored head (task 0) uses the chosen ranking loss; auxiliary heads stay
    pointwise, since their job is to regularise the shared embedding rather than
    to be ranked.
    """
    import torch
    import torch.nn as nn
    from pipeline.models import DEVICE, LOSSES, MMoE

    torch.manual_seed(seed)
    np.random.seed(seed)

    enc, encoder = encode_features(splits, use_cwm_fields=use_cwm)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    aux_tr = {t: np.asarray([r[t] for r in splits['train']], dtype=np.float32)
              for t in AUX_TASKS}

    task_weights = {
        'click': weight_click if weight_click is not None else aux_weight,
        'like': weight_like if weight_like is not None else aux_weight,
        'forward': weight_forward if weight_forward is not None else aux_weight,
    }

    model = MMoE(encoder.total_dim, len(encoder.field_names), embed_dim=embed_dim,
                 num_experts=num_experts, expert_dim=expert_dim,
                 num_tasks=1 + len(AUX_TASKS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    main_loss = LOSSES[loss]
    grouped = loss in ('listwise', 'bpr')
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if grouped else None

    best, best_state, bad = -1.0, None, 0
    if verbose:
        w_str = f"weights(clk={task_weights['click']}, lk={task_weights['like']}, fwd={task_weights['forward']})"
        print(f"==> mmoe | loss={loss} | experts={num_experts} | dim={embed_dim} | {w_str} | {DEVICE}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        if grouped:
            batches = _grouped_batches(groups, rng, target_rows=bs)
        else:
            perm = rng.permutation(len(ytr))
            batches = ((perm[i:i + bs], None, 0) for i in range(0, len(perm), bs))

        for rows_idx, gids, n_groups in batches:
            xb = torch.from_numpy(Xtr[rows_idx]).long().to(DEVICE)
            yb = torch.from_numpy(ytr[rows_idx]).float().to(DEVICE)
            gb = torch.from_numpy(gids).long().to(DEVICE) if gids is not None else None

            outs = model(xb)
            opt.zero_grad()
            total = main_loss(outs[0], yb, gb, n_groups)
            for j, task in enumerate(AUX_TASKS, start=1):
                tb = torch.from_numpy(aux_tr[task][rows_idx]).float().to(DEVICE)
                w = task_weights.get(task, aux_weight)
                total = total + w * bce(outs[j], tb)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(total.item())

        va = evaluate(uva, yva, _predict_torch(model, Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(CHECKPOINTS_DIR, "mmoe.pt"))
    return _finish("mmoe", evaluate(uva, yva, _predict_torch(model.to(DEVICE), Xva)),
                   {"model": "mmoe", "loss": loss, "embed_dim": embed_dim,
                    "num_experts": num_experts, "expert_dim": expert_dim,
                    "aux_weight": aux_weight, "task_weights": task_weights,
                    "lr": lr, "seed": seed, "use_cwm": use_cwm})


# ---------------------------------------------------------------------------
# 4. LightGBM LambdaMART
# ---------------------------------------------------------------------------

def train_lightgbm(splits: Dict, num_trees: int = 400, lr: float = 0.05,
                   num_leaves: int = 63, seed: int = 0,
                   drop_features: Optional[str] = None,
                   objective: str = 'lambdarank', verbose: bool = True) -> TrainResult:
    """GBDT on the causal dense features.

    Defaults to ``lambdarank`` with user groups and truncation at 5, so the tree
    ensemble optimises the same top-5 ordering nDCG@5 measures. The previous
    ``binary`` objective scored at item-popularity level.
    """
    import lightgbm as lgb

    dense, names = extract_dense_tabular_features(splits)
    if drop_features:
        drop_set = set(f.strip() for f in drop_features.split(','))
        keep_idx = [i for i, n in enumerate(names) if n not in drop_set]
        names = [names[i] for i in keep_idx]
        Xtr, ytr, utr = dense['train'][0][:, keep_idx], dense['train'][1], dense['train'][2]
        Xva, yva, uva = dense['valid'][0][:, keep_idx], dense['valid'][1], dense['valid'][2]
    else:
        Xtr, ytr, utr = dense['train']
        Xva, yva, uva = dense['valid']

    # lambdarank needs contiguous groups, so sort each split by user.
    def regroup(X, y, users):
        order = np.argsort(np.asarray(users), kind='stable')
        u_sorted = np.asarray(users)[order]
        _, counts = np.unique(u_sorted, return_counts=True)
        return X[order], y[order], list(u_sorted), counts, order

    params = {
        'objective': objective,
        'learning_rate': lr,
        'num_leaves': num_leaves,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': seed,
        'num_threads': 0,
    }

    if objective == 'lambdarank':
        Xtr_s, ytr_s, _, gtr, _ = regroup(Xtr, ytr, utr)
        Xva_s, yva_s, uva_s, gva, va_order = regroup(Xva, yva, uva)
        params.update({'metric': 'ndcg', 'ndcg_eval_at': [5],
                       'lambdarank_truncation_level': 5, 'label_gain': [0, 1]})
        dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=gtr, feature_name=names)
        dvalid = lgb.Dataset(Xva_s, label=yva_s, group=gva, feature_name=names,
                             reference=dtrain)
    else:
        Xva_s, yva_s, uva_s = Xva, yva, uva
        params.update({'metric': 'auc'})
        dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=names)
        dvalid = lgb.Dataset(Xva, label=yva, feature_name=names, reference=dtrain)

    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=num_trees,
                        valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    if verbose:
        print(f"==> lightgbm[{objective}] {time.time()-t0:.1f}s, "
              f"best iteration {booster.best_iteration}")

    importances = booster.feature_importance(importance_type='gain')
    top_idx = np.argsort(importances)[::-1][:5]
    top_feats = [f"{names[i]}:{importances[i]:.1f}" for i in top_idx if importances[i] > 0]
    if top_feats and verbose:
        print(f"  [TOP FEATURES] {', '.join(top_feats)}")

    booster.save_model(os.path.join(CHECKPOINTS_DIR, "lgb.txt"))
    va_preds = booster.predict(Xva_s, num_iteration=booster.best_iteration)
    return _finish("lgb", evaluate(uva_s, yva_s, va_preds),
                   {"model": "lgb", "objective": objective, "num_trees": num_trees,
                    "lr": lr, "num_leaves": num_leaves, "seed": seed,
                    "drop_features": drop_features})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ARCHS = ['fm', 'fm_torch', 'deepfm', 'din', 'mmoe', 'lgb']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a KuaiRand-Pure ranker (valid-only selection)")
    p.add_argument('--data_dir', default=None)
    p.add_argument('--model', default='fm', choices=ARCHS)
    p.add_argument('--loss', default='listwise', choices=['pointwise', 'listwise', 'bpr'])
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--experts', type=int, default=4)
    p.add_argument('--expert_dim', type=int, default=64)
    p.add_argument('--aux_weight', type=float, default=0.3)
    p.add_argument('--weight_click', type=float, default=None)
    p.add_argument('--weight_like', type=float, default=None)
    p.add_argument('--weight_forward', type=float, default=None)
    p.add_argument('--drop_features', type=str, default=None, help='comma-separated list of feature names to drop')
    p.add_argument('--lr', type=float, default=None, help='defaults per model')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8192)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--trees', type=int, default=400)
    p.add_argument('--num_leaves', type=int, default=63)
    p.add_argument('--objective', default='lambdarank', choices=['lambdarank', 'binary'])
    p.add_argument('--max_seq_len', type=int, default=10)
    p.add_argument('--cwm', action='store_true', help='add music_id/video_type/upload_type fields')
    p.add_argument('--seed', type=int, default=0)
    return p


def main(argv=None) -> TrainResult:
    args = build_parser().parse_args(argv)
    print("==> loading KuaiRand-Pure (train + valid; hidden test is sealed)")
    splits = load_kuairand(args.data_dir, include_extra_features=args.cwm)
    print({k: len(v) for k, v in splits.items()})

    if args.model == 'fm':
        return train_numpy_fm(splits, lr=args.lr or 0.001, epochs=max(args.epochs, 40),
                              bs=args.batch_size, patience=max(args.patience, 4),
                              seed=args.seed, use_cwm=args.cwm)
    if args.model == 'mmoe':
        return train_mmoe(splits, embed_dim=args.embed_dim, num_experts=args.experts,
                          expert_dim=args.expert_dim, lr=args.lr or 0.001,
                          epochs=args.epochs, bs=args.batch_size, patience=args.patience,
                          seed=args.seed, loss=args.loss, aux_weight=args.aux_weight,
                          weight_click=args.weight_click, weight_like=args.weight_like,
                          weight_forward=args.weight_forward, use_cwm=args.cwm)
    if args.model == 'lgb':
        return train_lightgbm(splits, num_trees=args.trees, lr=args.lr or 0.05,
                              num_leaves=args.num_leaves, seed=args.seed,
                              drop_features=args.drop_features,
                              objective=args.objective)
    return train_torch(splits, arch=args.model, loss=args.loss, embed_dim=args.embed_dim,
                       lr=args.lr or 0.001, epochs=args.epochs, bs=args.batch_size,
                       patience=args.patience, seed=args.seed, use_cwm=args.cwm,
                       max_seq_len=args.max_seq_len)


if __name__ == '__main__':
    main()
