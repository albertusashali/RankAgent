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

# Where weights and .meta.json land. Overridable per node so that two workspaces
# training the same architecture cannot clobber each other's checkpoints — the
# names encode only model and loss, so without this a later losing run silently
# replaces an earlier winning one and the submission exports the wrong weights.
CHECKPOINTS_DIR = os.environ.get("RANKAGENT_CHECKPOINTS", "checkpoints")
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

def _predict_torch(model, X: np.ndarray, side: Optional[np.ndarray] = None,
                   bs: int = 65536, side_is_long: bool = True) -> np.ndarray:
    """Score a split. ``side`` is the model's second forward argument, if any —
    integer history ids (``side_is_long``) or float dense features."""
    import torch
    from pipeline.models import DEVICE
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).long().to(DEVICE)
            if side is not None:
                sb = torch.from_numpy(side[i:i + bs])
                sb = (sb.long() if side_is_long else sb.float()).to(DEVICE)
                out.append(model(xb, sb).cpu().numpy())
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
                max_seq_len: int = 10, verbose: bool = True,
                feature_recipe=None) -> TrainResult:
    """Train an embedding model under a pointwise, pairwise or listwise objective."""
    import torch
    import torch.nn as nn
    from pipeline.models import DEVICE, resolve_loss, resolve_model

    torch.manual_seed(seed)
    np.random.seed(seed)

    enc, encoder = encode_features(splits, use_cwm_fields=use_cwm)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    n_fields = len(encoder.field_names)

    builder = resolve_model(arch)
    # Which side input a model consumes is declared on the builder, so
    # registering an architecture stays a single-site edit.
    needs_hist = getattr(builder, 'needs_history', False)
    needs_dense = getattr(builder, 'needs_dense', False)

    Htr = Hva = Dtr = Dva = None
    num_dense = 0
    if needs_hist:
        seqs = extract_sequential_features(splits, encoder, max_seq_len=max_seq_len)
        Htr, Hva = seqs['train'], seqs['valid']
    if needs_dense:
        dense, dense_names = extract_dense_tabular_features(splits, recipe=feature_recipe)
        Dtr, Dva = dense['train'][0], dense['valid'][0]
        num_dense = Dtr.shape[1]
        # encode_features and extract_dense_tabular_features both iterate the
        # split in order, so row i means the same impression in both. Assert it
        # rather than trust it: a mismatch would silently pair each row's
        # embeddings with another row's history statistics.
        if len(Dtr) != len(Xtr) or len(Dva) != len(Xva):
            raise ValueError(
                f"dense/categorical row mismatch: train {len(Dtr)} vs {len(Xtr)}, "
                f"valid {len(Dva)} vs {len(Xva)}")
        if verbose:
            print(f"==> {num_dense} causal dense features: "
                  f"{', '.join(dense_names[:6])}{' ...' if len(dense_names) > 6 else ''}")

    # A history-consuming model indexes the reserved padding row, so its table is
    # one row longer.
    rows = encoder.embedding_rows if needs_hist else encoder.total_dim
    model = builder(rows, n_fields, embed_dim, encoder.pad_id,
                    num_dense=num_dense).to(DEVICE)

    # Whichever side input this model takes, it arrives as the second forward
    # argument. History is integer ids; dense features are floats.
    side_tr, side_va = (Htr, Hva) if needs_hist else (Dtr, Dva)
    side_is_long = needs_hist

    loss_fn = resolve_loss(loss)
    # Whether the objective is computed within a user's impression list is a
    # property of the loss, declared by @ranking_loss. Reading it here rather
    # than testing `loss in ('listwise','bpr')` means a newly registered
    # listwise-family loss gets grouped batches instead of shuffled ones.
    grouped = getattr(loss_fn, 'requires_groups', loss in ('listwise', 'bpr'))
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
            if side_tr is not None:
                sb = torch.from_numpy(side_tr[rows_idx])
                sb = (sb.long() if side_is_long else sb.float()).to(DEVICE)
                logits = model(xb, sb)
            else:
                logits = model(xb)

            opt.zero_grad()
            l = loss_fn(logits, yb, gb, n_groups)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(l.item())

        va = evaluate(uva, yva, _predict_torch(model, Xva, side_va, side_is_long=side_is_long))
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
    return _finish(name, evaluate(uva, yva, _predict_torch(model.to(DEVICE), Xva, side_va,
                                                    side_is_long=side_is_long)),
                   {"model": arch, "loss": loss, "embed_dim": embed_dim, "lr": lr,
                    "seed": seed, "use_cwm": use_cwm, "max_seq_len": max_seq_len,
                    # A dense model must be rebuilt with the SAME feature set at
                    # inference, or its first layer is the wrong width.
                    "num_dense": num_dense,
                    "feature_recipe": (feature_recipe.model_dump()
                                       if feature_recipe is not None else None)})


# ---------------------------------------------------------------------------
# 3. Multi-task MMoE
# ---------------------------------------------------------------------------

def train_mmoe(splits: Dict, embed_dim: int = 16, num_experts: int = 4,
               expert_dim: int = 64, lr: float = 0.001, epochs: int = 20,
               bs: int = 8192, patience: int = 3, seed: int = 0,
               loss: str = 'listwise', aux_weight: float = 0.3,
               use_cwm: bool = False, verbose: bool = True) -> TrainResult:
    """Joint long_view + auxiliary feedback training.

    The scored head (task 0) uses the chosen ranking loss; auxiliary heads stay
    pointwise, since their job is to regularise the shared embedding rather than
    to be ranked.
    """
    import torch
    import torch.nn as nn
    from pipeline.models import DEVICE, MMoE, resolve_loss

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
    main_loss = resolve_loss(loss)
    grouped = getattr(main_loss, 'requires_groups', loss in ('listwise', 'bpr'))
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if grouped else None

    best, best_state, bad = -1.0, None, 0
    if verbose:
        print(f"==> mmoe | loss={loss} | experts={num_experts} | dim={embed_dim} | {DEVICE}")

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
    torch.save(best_state, os.path.join(CHECKPOINTS_DIR, "mmoe.pt"))
    return _finish("mmoe", evaluate(uva, yva, _predict_torch(model.to(DEVICE), Xva)),
                   {"model": "mmoe", "loss": loss, "embed_dim": embed_dim,
                    "num_experts": num_experts, "expert_dim": expert_dim,
                    "aux_weight": aux_weight, "lr": lr, "seed": seed, "use_cwm": use_cwm})


# ---------------------------------------------------------------------------
# 4. LightGBM LambdaMART
# ---------------------------------------------------------------------------

def train_lightgbm(splits: Dict, num_trees: int = 400, lr: float = 0.05,
                   num_leaves: int = 63, seed: int = 0,
                   objective: str = 'lambdarank', verbose: bool = True,
                   feature_recipe=None) -> TrainResult:
    """GBDT on the causal dense features.

    Defaults to ``lambdarank`` with user groups and truncation at 5, so the tree
    ensemble optimises the same top-5 ordering nDCG@5 measures. The previous
    ``binary`` objective scored at item-popularity level.
    """
    import lightgbm as lgb

    dense, names = extract_dense_tabular_features(splits, recipe=feature_recipe)
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

    booster.save_model(os.path.join(CHECKPOINTS_DIR, "lgb.txt"))
    va_preds = booster.predict(Xva_s, num_iteration=booster.best_iteration)
    return _finish("lgb", evaluate(uva_s, yva_s, va_preds),
                   {"model": "lgb", "objective": objective, "num_trees": num_trees,
                    "lr": lr, "num_leaves": num_leaves, "seed": seed,
                    # The recipe travels with the checkpoint. Inference has to
                    # build the SAME feature matrix that training used; without
                    # this, a model trained on a 15-feature recipe would be
                    # scored against the default set and silently mispredict.
                    "feature_recipe": (feature_recipe.model_dump()
                                       if feature_recipe is not None else None),
                    "feature_recipe_id": (feature_recipe.recipe_id
                                          if feature_recipe is not None else None),
                    "feature_names": names})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: Architectures shipped with the harness. This is documentation, NOT a
#: whitelist: ``--model`` deliberately has no ``choices=``, so a model the agent
#: registers in a generated patch is selectable without also editing the parser.
ARCHS = ['fm', 'fm_torch', 'deepfm', 'din', 'mmoe', 'lgb']

#: Objectives shipped with the harness. Same rule — the authoritative list is
#: ``pipeline.models.LOSSES``, which cannot be imported here because it pulls
#: torch to module scope and re-breaks ``--model lgb`` (see the import note at
#: the top of this file). Names are therefore validated inside the trainer,
#: where torch is already loaded, via ``models.resolve_loss``.
LOSSES_DOC = ['pointwise', 'listwise', 'bpr']


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a KuaiRand-Pure ranker (valid-only selection)")
    p.add_argument('--data_dir', default=None)
    p.add_argument('--model', default='fm',
                   help=f"architecture; shipped: {', '.join(ARCHS)} "
                        f"(agent-registered models are also accepted)")
    p.add_argument('--loss', default='listwise',
                   help=f"ranking objective; shipped: {', '.join(LOSSES_DOC)} "
                        f"(any key in pipeline.models.LOSSES is accepted)")
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
    p.add_argument('--max_seq_len', type=int, default=10)
    p.add_argument('--cwm', action='store_true', help='add music_id/video_type/upload_type fields')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--feature_recipe', default=None,
                   help='path to a FeatureRecipe JSON. A validated, content-hashed '
                        'feature configuration — the Feature Steward\'s action '
                        'space, which needs no code generation.')
    p.add_argument('--feature_profile', default=None,
                   choices=['core', 'affinity', 'full'],
                   help='shorthand for a recipe with this base profile')
    p.add_argument('--smoke', action='store_true',
                   help='correctness check: one epoch on a small user subsample. '
                        'The reported score is NOT comparable to a full run.')
    p.add_argument('--smoke_users', type=int, default=3000,
                   help='users retained in --smoke mode (default 3000)')
    return p


def resolve_recipe(args):
    """Build the FeatureRecipe this run should use, or None for the default.

    A recipe is the Feature Steward's action space: a validated, content-hashed
    feature configuration that needs no code generation. It is audited before it
    is written, so an unknown feature name or an out-of-range smoothing constant
    is caught at compile time rather than becoming a wasted training run.
    """
    path = getattr(args, 'feature_recipe', None)
    profile = getattr(args, 'feature_profile', None)
    if path is None and profile is None:
        return None
    from pipeline.feature_recipes import load_recipe
    return load_recipe(path, base_profile=profile or 'affinity')


def subsample_for_smoke(splits: Dict[str, List[dict]], max_users: int = 3000,
                        seed: int = 0) -> Dict[str, List[dict]]:
    """Keep every row belonging to a deterministic subset of users.

    Sampling by *user* rather than by row is not a detail: GAUC and nDCG@5 are
    both computed within a user's impression list, so tearing users apart would
    leave one-row groups that the metrics discard, and the smoke run would report
    a score with almost nothing behind it. Keeping whole users intact means the
    smoke score is a small, noisy estimate of the real one rather than a
    different quantity — good enough to catch NaNs and shape errors, which is all
    it is for.
    """
    users = sorted({r['user_id'] for r in splits.get('valid', [])})
    if len(users) > max_users:
        rng = np.random.default_rng(seed)
        keep = set(np.asarray(users, dtype=object)[
            rng.permutation(len(users))[:max_users]].tolist())
    else:
        keep = set(users)
    return {name: [r for r in rows if r['user_id'] in keep]
            for name, rows in splits.items()}


def main(argv=None) -> TrainResult:
    args = build_parser().parse_args(argv)
    print("==> loading KuaiRand-Pure (train + valid; hidden test is sealed)")
    splits = load_kuairand(args.data_dir, include_extra_features=args.cwm)
    print({k: len(v) for k, v in splits.items()})

    if args.smoke:
        splits = subsample_for_smoke(splits, max_users=args.smoke_users, seed=args.seed)
        print(f"==> SMOKE MODE — {args.smoke_users} users, 1 epoch. This verifies the "
              f"code runs; the score is not comparable to a full run.")
        print({k: len(v) for k, v in splits.items()})

    # One epoch, no early stopping, and a token forest in smoke mode. The numpy
    # FM raises its own floors on epochs/patience to reproduce the official
    # baseline, so it needs the override too or --smoke would still run 40 epochs.
    epochs = 1 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience
    fm_epochs = 1 if args.smoke else max(args.epochs, 40)
    fm_patience = 1 if args.smoke else max(args.patience, 4)
    trees = 20 if args.smoke else args.trees

    if args.model == 'fm':
        return train_numpy_fm(splits, lr=args.lr or 0.001, epochs=fm_epochs,
                              bs=args.batch_size, patience=fm_patience,
                              seed=args.seed, use_cwm=args.cwm)
    if args.model == 'mmoe':
        return train_mmoe(splits, embed_dim=args.embed_dim, num_experts=args.experts,
                          expert_dim=args.expert_dim, lr=args.lr or 0.001,
                          epochs=epochs, bs=args.batch_size, patience=patience,
                          seed=args.seed, loss=args.loss, aux_weight=args.aux_weight,
                          use_cwm=args.cwm)
    if args.model == 'lgb':
        return train_lightgbm(splits, num_trees=trees, lr=args.lr or 0.05,
                              num_leaves=args.num_leaves, seed=args.seed,
                              objective=args.objective,
                              feature_recipe=resolve_recipe(args))
    return train_torch(splits, arch=args.model, loss=args.loss, embed_dim=args.embed_dim,
                       lr=args.lr or 0.001, epochs=epochs, bs=args.batch_size,
                       patience=patience, seed=args.seed, use_cwm=args.cwm,
                       max_seq_len=args.max_seq_len,
                       feature_recipe=resolve_recipe(args))


if __name__ == '__main__':
    main()
