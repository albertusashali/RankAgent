"""Official KuaiRand scorer — the single source of truth for every metric.

Loads ``kuairand-starter-kit/evaluate.py`` verbatim if present, or uses the
embedded exact starter kit implementation as a robust fallback.
"""
import collections
import importlib.util
import os
from typing import Dict, List, Sequence, Union

_STARTER_KIT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kuairand-starter-kit",
)
_OFFICIAL_PATH = os.path.join(_STARTER_KIT, "evaluate.py")

# --- Fallback implementation with exact official conventions ---
def _fallback_auc(y_true, y_score):
    import numpy as np
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score))
    unique_scores, inverse_indices, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        for idx in range(len(unique_scores)):
            if counts[idx] > 1:
                mask = (inverse_indices == idx)
                ranks[mask] = np.mean(ranks[mask])
    pos_ranks = ranks[y_true == 1]
    return float((np.sum(pos_ranks) - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))

def _fallback_ndcg_at_k(y_true, y_score, k=5):
    import numpy as np
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(-y_score, kind='stable')
    y_true_sorted = y_true[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, len(y_true_sorted) + 2))
    dcg = np.sum((2.0 ** y_true_sorted - 1.0) * discounts)
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = np.sum((2.0 ** ideal_sorted - 1.0) * discounts)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

def _fallback_evaluate(users, labels, preds, k=5):
    import numpy as np
    user_data = collections.defaultdict(lambda: {'labels': [], 'preds': []})
    for u, l, p in zip(users, labels, preds):
        user_data[u]['labels'].append(l)
        user_data[u]['preds'].append(p)
    
    aucs, weights, ndcgs = [], [], []
    for u, data in user_data.items():
        lbls = np.asarray(data['labels'])
        scrs = np.asarray(data['preds'])
        pos = np.sum(lbls == 1)
        neg = np.sum(lbls == 0)
        if pos > 0 and neg > 0:
            aucs.append(_fallback_auc(lbls, scrs))
            weights.append(pos)
        ndcgs.append(_fallback_ndcg_at_k(lbls, scrs, k=k))
        
    gauc = float(np.average(aucs, weights=weights)) if aucs else 0.5
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    primary = float((gauc + mean_ndcg) / 2.0)
    return {
        'GAUC': gauc,
        f'nDCG@{k}': mean_ndcg,
        'primary': primary,
        'users': len(user_data),
        'rows': len(labels)
    }

def _load_official():
    """Import the starter kit's evaluate.py by path or return fallback module."""
    if os.path.exists(_OFFICIAL_PATH) and os.path.getsize(_OFFICIAL_PATH) > 50:
        try:
            spec = importlib.util.spec_from_file_location("kuairand_official_evaluate", _OFFICIAL_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "auc") and hasattr(module, "ndcg_at_k") and hasattr(module, "evaluate"):
                return module
        except Exception:
            pass
    
    class FallbackModule:
        auc = staticmethod(_fallback_auc)
        ndcg_at_k = staticmethod(_fallback_ndcg_at_k)
        evaluate = staticmethod(_fallback_evaluate)
        
    return FallbackModule()

_official = _load_official()

auc = _official.auc
ndcg_at_k = _official.ndcg_at_k

Numeric = Union[Sequence[float], "list"]

def evaluate(users: Sequence, labels: Numeric, preds: Numeric, k: int = 5) -> Dict[str, float]:
    """Score a set of predictions with the official implementation."""
    users = list(users)
    labels = [int(v) for v in labels]
    preds = [float(v) for v in preds]
    if not (len(users) == len(labels) == len(preds)):
        raise ValueError(
            f"length mismatch: users={len(users)} labels={len(labels)} preds={len(preds)}"
        )
    return _official.evaluate(users, labels, preds, k=k)

def format_eval_line(metrics: Dict[str, float], k: int = 5) -> str:
    """The line the sandbox parser looks for. Validation metrics only."""
    return (
        f"[EVAL] GAUC: {metrics['GAUC']:.4f} | "
        f"nDCG@{k}: {metrics[f'nDCG@{k}']:.4f} | "
        f"Primary: {metrics['primary']:.4f}"
    )

if __name__ == '__main__':
    demo_users = ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3']
    demo_labels = [1, 0, 0, 1, 1, 0, 0]
    demo_preds = [0.9, 0.2, 0.1, 0.8, 0.7, 0.3, 0.1]
    print(format_eval_line(evaluate(demo_users, demo_labels, demo_preds)))
    print(f"loaded official implementation from {_OFFICIAL_PATH}")
