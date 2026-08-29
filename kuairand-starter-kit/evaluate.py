"""Official KuaiRand Evaluator from Starter Kit."""
import collections
import numpy as np

def auc(y_true, y_score):
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

def ndcg_at_k(y_true, y_score, k=5):
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

def evaluate(users, labels, preds, k=5):
    users = list(users)
    labels = list(labels)
    preds = list(preds)
    if not (len(users) == len(labels) == len(preds)):
        raise ValueError(f"length mismatch: users={len(users)} labels={len(labels)} preds={len(preds)}")
        
    user_data = collections.defaultdict(lambda: {'labels': [], 'preds': []})
    for u, l, p in zip(users, labels, preds):
        user_data[u]['labels'].append(l)
        user_data[u]['preds'].append(p)
    
    aucs = []
    weights = []
    ndcgs = []
    for u, data in user_data.items():
        lbls = np.asarray(data['labels'])
        scrs = np.asarray(data['preds'])
        pos = np.sum(lbls == 1)
        neg = np.sum(lbls == 0)
        if pos > 0 and neg > 0:
            a = auc(lbls, scrs)
            aucs.append(a)
            weights.append(pos)
        
        ndcg = ndcg_at_k(lbls, scrs, k=k)
        ndcgs.append(ndcg)
        
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

