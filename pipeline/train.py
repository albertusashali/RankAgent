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
                               extract_sequential_features, select_rank_features)
from pipeline.feature_recipes import FeatureRecipe, load_recipe
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


def _cache_valid(name: str, predictions: np.ndarray):
    """Cache validation predictions in original split order for safe ensembling."""
    np.save(os.path.join(CHECKPOINTS_DIR, f"{name}.valid.npy"),
            np.asarray(predictions, dtype=np.float64))


# ---------------------------------------------------------------------------
# Grouped batching for the ranking losses
# ---------------------------------------------------------------------------

def _user_groups(users: List[str]) -> List[np.ndarray]:
    idx: Dict[str, List[int]] = {}
    for i, u in enumerate(users):
        idx.setdefault(u, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in idx.values()]


def _grouped_batches(groups: List[np.ndarray], rng: np.random.Generator,
                     target_rows: int, max_group: Optional[int] = None):
    """Yield ``(row_indices, group_ids)`` where each group is one user's impressions.

    Pointwise training can shuffle rows freely, but a within-user loss needs a
    user's impressions to arrive together. By default no user is truncated:
    GAUC weights users by positive count, so silently capping power users at 64
    rows misaligns training with evaluation. An explicit cap remains available
    as a compute ablation and samples without replacement each epoch.
    """
    order = rng.permutation(len(groups))
    rows: List[np.ndarray] = []
    gids: List[np.ndarray] = []
    n_rows = 0
    g = 0
    for gi in order:
        members = groups[gi]
        if max_group and len(members) > max_group:
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
                   dense: Optional[np.ndarray] = None, bs: int = 65536) -> np.ndarray:
    import torch
    from pipeline.models import DEVICE
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(DEVICE)
            if dense is not None:
                db = torch.from_numpy(dense[i:i + bs]).float().to(DEVICE)
                out.append(model(xb, db).cpu().numpy())
            elif hist is not None:
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
    valid_preds = m.predict(Xva)
    _cache_valid("fm", valid_preds)
    return _finish("fm", evaluate(uva, yva, valid_preds),
                   {"model": "fm", "k": k, "lr": lr, "seed": seed, "use_cwm": use_cwm})


# ---------------------------------------------------------------------------
# 2. Torch single-task models (FM / DeepFM / DIN) with swappable loss
# ---------------------------------------------------------------------------

def train_torch(splits: Dict, arch: str = 'fm_torch', loss: str = 'listwise',
                embed_dim: int = 16, lr: float = 0.001, epochs: int = 20,
                bs: int = 8192, patience: int = 3, seed: int = 0,
                use_cwm: bool = False, weight_decay: float = 1e-6,
                max_seq_len: int = 10, max_group_rows: int = 0,
                auc_weight: float = 0.5, ndcg_cutoff: int = 5,
                propensity_mode: str = 'none', propensity_clip: float = 10.0,
                random_rows: Optional[List[dict]] = None,
                feature_recipe: FeatureRecipe = None,
                label_smoothing: float = 0.0,
                recency_half_life: float = 0.0,
                verbose: bool = True) -> TrainResult:
    """Train an embedding model under a pointwise, pairwise or listwise objective."""
    import torch
    import torch.nn as nn
    from pipeline.models import DEVICE, LOSSES, DIN, DeepFM, DenseDeepFM, TorchFM

    torch.manual_seed(seed)
    np.random.seed(seed)

    enc, encoder = encode_features(splits, use_cwm_fields=use_cwm)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    n_fields = len(encoder.field_names)

    needs_dense = (arch == 'deepfm_dense')
    dense_names, dense_mean, dense_std = [], None, None
    if needs_dense:
        recipe = feature_recipe or FeatureRecipe(base_profile='full', use_rank_selection=True)
        dense_parts, dense_names = extract_dense_tabular_features(splits, recipe=recipe)
        if recipe.use_rank_selection:
            dense_parts, dense_names = select_rank_features(
                dense_parts, dense_names, recipe.min_within_user_variance)
        Dtr, Dva = dense_parts['train'][0], dense_parts['valid'][0]
        dense_mean = Dtr.mean(axis=0).astype(np.float32)
        dense_std = Dtr.std(axis=0).astype(np.float32)
        dense_std[dense_std < 1e-6] = 1.0
        Dtr = np.nan_to_num((Dtr - dense_mean) / dense_std, posinf=0., neginf=0.)
        Dva = np.nan_to_num((Dva - dense_mean) / dense_std, posinf=0., neginf=0.)
    else:
        recipe, Dtr, Dva = None, None, None

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
    elif arch == 'deepfm_dense':
        model = DenseDeepFM(rows, n_fields, len(dense_names), embed_dim)
    elif arch == 'din':
        model = DIN(rows, n_fields, pad_id=encoder.pad_id, embed_dim=embed_dim)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model = model.to(DEVICE)

    base_loss = LOSSES[loss]
    loss_fn = ((lambda logits, labels, group, n_groups, sample_weight=None:
                base_loss(logits, labels, group, n_groups,
                          auc_weight=auc_weight, cutoff=ndcg_cutoff,
                          sample_weight=sample_weight))
               if loss == 'hybrid' else base_loss)
    grouped = loss in ('listwise', 'bpr', 'hybrid')
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if grouped else None
    if propensity_mode != 'none':
        from pipeline.propensity import estimate_random_exposure_weights
        train_weights = estimate_random_exposure_weights(
            splits['train'], random_rows or [], mode=propensity_mode,
            clip=propensity_clip, seed=seed)
        print(f"==> propensity[{propensity_mode}] mean={train_weights.mean():.3f} "
              f"p99={np.quantile(train_weights, .99):.3f} max={train_weights.max():.3f}")
    else:
        train_weights = np.ones(len(ytr), dtype=np.float32)
    if recency_half_life > 0:
        dates = np.asarray([r['date'] for r in splits['train']], dtype=np.float32)
        temporal = np.exp2(-(dates.max() - dates) / recency_half_life).astype(np.float32)
        temporal /= temporal.mean()
        train_weights *= temporal
        print(f"==> temporal denoising half_life={recency_half_life:g}d "
              f"oldest_weight={temporal.min():.3f} newest_weight={temporal.max():.3f}")

    best, best_state, bad = -1.0, None, 0
    if verbose:
        print(f"==> {arch} | loss={loss} | dim={embed_dim} | fields={n_fields} | {DEVICE}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        if grouped:
            batches = _grouped_batches(groups, rng, target_rows=bs,
                                       max_group=max_group_rows or None)
        else:
            perm = rng.permutation(len(ytr))
            batches = ((perm[i:i + bs], None, 0) for i in range(0, len(perm), bs))

        for rows_idx, gids, n_groups in batches:
            xb = torch.from_numpy(Xtr[rows_idx]).long().to(DEVICE)
            yb = torch.from_numpy(ytr[rows_idx]).float().to(DEVICE)
            gb = torch.from_numpy(gids).long().to(DEVICE) if gids is not None else None
            wb = torch.from_numpy(train_weights[rows_idx]).float().to(DEVICE)
            if needs_dense:
                logits = model(xb, torch.from_numpy(Dtr[rows_idx]).float().to(DEVICE))
            elif needs_hist:
                logits = model(xb, torch.from_numpy(Htr[rows_idx]).long().to(DEVICE))
            else:
                logits = model(xb)

            opt.zero_grad()
            loss_targets = (yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
                            if loss == 'pointwise' else yb)
            l = loss_fn(logits, loss_targets, gb, n_groups, wb)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(l.item())

        va = evaluate(uva, yva, _predict_torch(model, Xva, Hva, Dva))
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
    if recipe is not None:
        name += f"_recipe_{recipe.recipe_id}"
    if loss == 'hybrid':
        name += f"_aw{int(round(auc_weight * 100))}"
    if propensity_mode != 'none':
        name += f"_{propensity_mode}{int(round(propensity_clip))}"
    if max_group_rows:
        name += f"_cap{max_group_rows}"
    if label_smoothing:
        name += f"_ls{int(round(label_smoothing * 100))}"
    if recency_half_life:
        name += f"_hl{int(round(recency_half_life))}"
    if seed:
        name += f"_s{seed}"
    torch.save(best_state, os.path.join(CHECKPOINTS_DIR, f"{name}.pt"))
    valid_preds = _predict_torch(model.to(DEVICE), Xva, Hva, Dva)
    _cache_valid(name, valid_preds)
    return _finish(name, evaluate(uva, yva, valid_preds),
                   {"model": arch, "loss": loss, "embed_dim": embed_dim, "lr": lr,
                    "seed": seed, "use_cwm": use_cwm, "max_seq_len": max_seq_len,
                    "max_group_rows": max_group_rows, "auc_weight": auc_weight,
                    "ndcg_cutoff": ndcg_cutoff, "propensity_mode": propensity_mode,
                    "propensity_clip": propensity_clip,
                    "label_smoothing": label_smoothing,
                    "recency_half_life": recency_half_life,
                    "feature_recipe": recipe.model_dump() if recipe else None,
                    "dense_feature_names": dense_names,
                    "dense_mean": dense_mean.tolist() if dense_mean is not None else None,
                    "dense_std": dense_std.tolist() if dense_std is not None else None})


# ---------------------------------------------------------------------------
# 3. Multi-task MMoE
# ---------------------------------------------------------------------------

def train_mmoe(splits: Dict, embed_dim: int = 16, num_experts: int = 4,
               expert_dim: int = 64, lr: float = 0.001, epochs: int = 20,
               bs: int = 8192, patience: int = 3, seed: int = 0,
               loss: str = 'listwise', aux_weight: float = 0.3,
               use_cwm: bool = False, max_group_rows: int = 0,
               auc_weight: float = 0.5, ndcg_cutoff: int = 5,
               propensity_mode: str = 'none', propensity_clip: float = 10.0,
               random_rows: Optional[List[dict]] = None,
               verbose: bool = True) -> TrainResult:
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

    model = MMoE(encoder.total_dim, len(encoder.field_names), embed_dim=embed_dim,
                 num_experts=num_experts, expert_dim=expert_dim,
                 num_tasks=1 + len(AUX_TASKS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    base_loss = LOSSES[loss]
    main_loss = ((lambda logits, labels, group, n_groups, sample_weight=None:
                  base_loss(logits, labels, group, n_groups,
                            auc_weight=auc_weight, cutoff=ndcg_cutoff,
                            sample_weight=sample_weight))
                 if loss == 'hybrid' else base_loss)
    grouped = loss in ('listwise', 'bpr', 'hybrid')
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if grouped else None
    if propensity_mode != 'none':
        from pipeline.propensity import estimate_random_exposure_weights
        train_weights = estimate_random_exposure_weights(
            splits['train'], random_rows or [], mode=propensity_mode,
            clip=propensity_clip, seed=seed)
    else:
        train_weights = np.ones(len(ytr), dtype=np.float32)

    best, best_state, bad = -1.0, None, 0
    if verbose:
        print(f"==> mmoe | loss={loss} | experts={num_experts} | dim={embed_dim} | {DEVICE}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        if grouped:
            batches = _grouped_batches(groups, rng, target_rows=bs,
                                       max_group=max_group_rows or None)
        else:
            perm = rng.permutation(len(ytr))
            batches = ((perm[i:i + bs], None, 0) for i in range(0, len(perm), bs))

        for rows_idx, gids, n_groups in batches:
            xb = torch.from_numpy(Xtr[rows_idx]).long().to(DEVICE)
            yb = torch.from_numpy(ytr[rows_idx]).float().to(DEVICE)
            gb = torch.from_numpy(gids).long().to(DEVICE) if gids is not None else None
            wb = torch.from_numpy(train_weights[rows_idx]).float().to(DEVICE)

            outs = model(xb)
            opt.zero_grad()
            total = main_loss(outs[0], yb, gb, n_groups, wb)
            for j, task in enumerate(AUX_TASKS, start=1):
                tb = torch.from_numpy(aux_tr[task][rows_idx]).float().to(DEVICE)
                total = total + aux_weight * bce(outs[j], tb)
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
    name = f"mmoe_{loss}"
    if loss == 'hybrid':
        name += f"_aw{int(round(auc_weight * 100))}"
    if propensity_mode != 'none':
        name += f"_{propensity_mode}{int(round(propensity_clip))}"
    if max_group_rows:
        name += f"_cap{max_group_rows}"
    torch.save(best_state, os.path.join(CHECKPOINTS_DIR, f"{name}.pt"))
    valid_preds = _predict_torch(model.to(DEVICE), Xva)
    _cache_valid(name, valid_preds)
    return _finish(name, evaluate(uva, yva, valid_preds),
                   {"model": "mmoe", "loss": loss, "embed_dim": embed_dim,
                    "num_experts": num_experts, "expert_dim": expert_dim,
                    "aux_weight": aux_weight, "lr": lr, "seed": seed, "use_cwm": use_cwm,
                    "max_group_rows": max_group_rows, "auc_weight": auc_weight,
                    "ndcg_cutoff": ndcg_cutoff, "propensity_mode": propensity_mode,
                    "propensity_clip": propensity_clip})


# ---------------------------------------------------------------------------
# 4. LightGBM LambdaMART
# ---------------------------------------------------------------------------

def train_lightgbm(splits: Dict, num_trees: int = 400, lr: float = 0.05,
                   num_leaves: int = 63, seed: int = 0,
                   objective: str = 'lambdarank', feature_profile: str = 'affinity',
                   select_features: bool = False, feature_recipe: FeatureRecipe = None,
                   verbose: bool = True) -> TrainResult:
    """GBDT on the causal dense features.

    Defaults to ``lambdarank`` with user groups and truncation at 5, so the tree
    ensemble optimises the same top-5 ordering nDCG@5 measures. The previous
    ``binary`` objective scored at item-popularity level.
    """
    import lightgbm as lgb

    recipe = feature_recipe or FeatureRecipe(base_profile=feature_profile,
                                              use_rank_selection=select_features)
    dense, names = extract_dense_tabular_features(splits, recipe=recipe)
    apply_selection = recipe.use_rank_selection or select_features
    if apply_selection:
        before = list(names)
        dense, names = select_rank_features(
            dense, names, min_within_user_variance=recipe.min_within_user_variance)
        print(f"==> rank-aware feature selection kept {len(names)}/{len(before)}; "
              f"dropped {[n for n in before if n not in names]}")
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

    suffix = (f"recipe_{recipe.recipe_id}" if feature_recipe is not None else
              f"{feature_profile}{'_selected' if apply_selection else ''}")
    name, artifact = f"lgb_{suffix}", f"lgb_{suffix}.txt"
    booster.save_model(os.path.join(CHECKPOINTS_DIR, artifact))
    importance = sorted(zip(names, booster.feature_importance(importance_type='gain')),
                        key=lambda pair: pair[1], reverse=True)
    total_gain = max(float(sum(v for _, v in importance)), 1e-12)
    with open(os.path.join(CHECKPOINTS_DIR, f"{name}.importance.json"), 'w',
              encoding='utf-8') as fh:
        json.dump({'recipe_id': recipe.recipe_id,
                   'features': [{'name': n, 'gain_fraction': float(v / total_gain)}
                                for n, v in importance]}, fh, indent=2)
    va_preds = booster.predict(Xva_s, num_iteration=booster.best_iteration)
    if objective == 'lambdarank':
        aligned_preds = np.empty_like(va_preds)
        aligned_preds[va_order] = va_preds
    else:
        aligned_preds = va_preds
    _cache_valid(name, aligned_preds)
    return _finish(name, evaluate(uva_s, yva_s, va_preds),
                   {"model": "lgb", "objective": objective, "num_trees": num_trees,
                    "lr": lr, "num_leaves": num_leaves, "seed": seed,
                    "feature_profile": recipe.base_profile,
                    "select_features": apply_selection,
                    "feature_recipe": recipe.model_dump(),
                    "artifact": artifact})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ARCHS = ['fm', 'fm_torch', 'deepfm', 'deepfm_dense', 'din', 'mmoe', 'lgb']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a KuaiRand-Pure ranker (valid-only selection)")
    p.add_argument('--data_dir', default=None)
    p.add_argument('--model', default='fm', choices=ARCHS)
    p.add_argument('--loss', default='listwise', choices=['pointwise', 'listwise', 'bpr', 'hybrid'])
    p.add_argument('--auc_weight', type=float, default=0.5)
    p.add_argument('--ndcg_cutoff', type=int, default=5)
    p.add_argument('--max_group_rows', type=int, default=0,
                   help='0 keeps every impression; positive values are a compute ablation')
    p.add_argument('--propensity', default='none', choices=['none', 'ips', 'snips'])
    p.add_argument('--propensity_clip', type=float, default=10.0)
    p.add_argument('--label_smoothing', type=float, default=0.0,
                   help='pointwise target smoothing in [0, 0.2] to reduce label noise')
    p.add_argument('--recency_half_life', type=float, default=0.0,
                   help='training-day half-life; 0 disables temporal denoising')
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--experts', type=int, default=4)
    p.add_argument('--expert_dim', type=int, default=64)
    p.add_argument('--aux_weight', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=None, help='defaults per model')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8192)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--trees', type=int, default=400)
    p.add_argument('--num_leaves', type=int, default=63)
    p.add_argument('--objective', default='lambdarank', choices=['lambdarank', 'binary'])
    p.add_argument('--feature_profile', default='affinity', choices=['core', 'affinity', 'full'])
    p.add_argument('--select_features', action='store_true')
    p.add_argument('--feature_recipe', default=None,
                   help='validated FeatureRecipe JSON; overrides --feature_profile')
    p.add_argument('--max_seq_len', type=int, default=10)
    p.add_argument('--cwm', action='store_true', help='add music_id/video_type/upload_type fields')
    p.add_argument('--seed', type=int, default=0)
    return p


def main(argv=None) -> TrainResult:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.label_smoothing <= 0.2:
        raise ValueError('--label_smoothing must be between 0 and 0.2')
    if args.recency_half_life < 0:
        raise ValueError('--recency_half_life must be non-negative')
    print("==> loading KuaiRand-Pure (train + valid; hidden test is sealed)")
    splits = load_kuairand(args.data_dir, include_extra_features=args.cwm)
    print({k: len(v) for k, v in splits.items()})
    random_rows = None
    if args.propensity != 'none':
        from pipeline.data import load_unbiased_valid
        print("==> loading randomized-exposure log for propensity estimation")
        random_rows = load_unbiased_valid(args.data_dir)

    if args.model == 'fm':
        return train_numpy_fm(splits, lr=args.lr or 0.001, epochs=max(args.epochs, 40),
                              bs=args.batch_size, patience=max(args.patience, 4),
                              seed=args.seed, use_cwm=args.cwm)
    if args.model == 'mmoe':
        return train_mmoe(splits, embed_dim=args.embed_dim, num_experts=args.experts,
                          expert_dim=args.expert_dim, lr=args.lr or 0.001,
                          epochs=args.epochs, bs=args.batch_size, patience=args.patience,
                          seed=args.seed, loss=args.loss, aux_weight=args.aux_weight,
                          use_cwm=args.cwm, max_group_rows=args.max_group_rows,
                          auc_weight=args.auc_weight, ndcg_cutoff=args.ndcg_cutoff,
                          propensity_mode=args.propensity, propensity_clip=args.propensity_clip,
                          random_rows=random_rows)
    if args.model == 'lgb':
        recipe = load_recipe(args.feature_recipe, args.feature_profile,
                             args.select_features) if args.feature_recipe else None
        return train_lightgbm(splits, num_trees=args.trees, lr=args.lr or 0.05,
                              num_leaves=args.num_leaves, seed=args.seed,
                              objective=args.objective, feature_profile=args.feature_profile,
                              select_features=args.select_features, feature_recipe=recipe)
    recipe = (load_recipe(args.feature_recipe, args.feature_profile, args.select_features)
              if args.feature_recipe else
              (FeatureRecipe(base_profile=args.feature_profile,
                             use_rank_selection=args.select_features)
               if args.model == 'deepfm_dense' else None))
    return train_torch(splits, arch=args.model, loss=args.loss, embed_dim=args.embed_dim,
                       lr=args.lr or 0.001, epochs=args.epochs, bs=args.batch_size,
                       patience=args.patience, seed=args.seed, use_cwm=args.cwm,
                       max_seq_len=args.max_seq_len, max_group_rows=args.max_group_rows,
                       auc_weight=args.auc_weight, ndcg_cutoff=args.ndcg_cutoff,
                       propensity_mode=args.propensity, propensity_clip=args.propensity_clip,
                       random_rows=random_rows, feature_recipe=recipe,
                       label_smoothing=args.label_smoothing,
                       recency_half_life=args.recency_half_life)


if __name__ == '__main__':
    main()
