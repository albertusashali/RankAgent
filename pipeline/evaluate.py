"""
Official KuaiRand Evaluator: GAUC & nDCG@5 calculation.
Exact match to Starter Kit evaluate.py.
"""
import numpy as np
import collections
from typing import Dict, List, Union

def _auc_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """Computes AUC for a single user's impression list."""
    pos_mask = (labels == 1)
    n_pos = np.sum(pos_mask)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    order = np.argsort(preds)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(preds)) + 1
    
    pos_ranks = rank[pos_mask]
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def _ndcg_at_k(labels: np.ndarray, preds: np.ndarray, k: int = 5) -> float:
    """Computes nDCG@k using (2^rel - 1) / log2(rank + 1)."""
    n_pos = np.sum(labels == 1)
    if n_pos == 0:
        return 0.0  # Convention: users with zero positives score 0
        
    top_indices = np.argsort(-preds)[:k]
    top_labels = labels[top_indices]
    
    # DCG
    discounts = np.log2(np.arange(len(top_labels)) + 2.0)
    gains = (2.0 ** top_labels) - 1.0
    dcg = np.sum(gains / discounts)
    
    # IDCG (Ideal DCG)
    ideal_labels = np.sort(labels)[::-1][:k]
    ideal_gains = (2.0 ** ideal_labels) - 1.0
    idcg = np.sum(ideal_gains / discounts[:len(ideal_labels)])
    
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)

def evaluate(users: List[str], labels: Union[np.ndarray, List[float]], preds: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Computes Group AUC (GAUC) and nDCG@5 across all users.
    Output format matches: [EVAL] GAUC: X | nDCG@5: Y | Primary: Z
    """
    users = np.asarray(users)
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)
    
    user_groups = collections.defaultdict(list)
    for i in range(len(users)):
        user_groups[users[i]].append(i)
        
    gauc_weights = []
    gauc_values = []
    ndcg_values = []
    
    for u, idxs in user_groups.items():
        u_idx = np.asarray(idxs)
        u_labels = labels[u_idx]
        u_preds = preds[u_idx]
        
        n_pos = int(np.sum(u_labels == 1))
        n_total = len(u_labels)
        
        # nDCG@5 is calculated for ALL users
        ndcg_values.append(_ndcg_at_k(u_labels, u_preds, k=5))
        
        # GAUC is only calculated for users with 0 < pos < total
        if 0 < n_pos < n_total:
            auc = _auc_score(u_labels, u_preds)
            gauc_values.append(auc)
            gauc_weights.append(n_pos)
            
    total_weight = sum(gauc_weights)
    if total_weight > 0:
        gauc = float(np.sum(np.array(gauc_values) * np.array(gauc_weights)) / total_weight)
    else:
        gauc = 0.5
        
    ndcg5 = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    primary = (gauc + ndcg5) / 2.0
    
    return {
        'GAUC': gauc,
        'nDCG@5': ndcg5,
        'primary': primary
    }

if __name__ == '__main__':
    # Standalone sanity test
    dummy_users = ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3']
    dummy_labels = [1, 0, 0, 1, 1, 0, 0]
    dummy_preds = [0.9, 0.2, 0.1, 0.8, 0.7, 0.3, 0.1]
    res = evaluate(dummy_users, dummy_labels, dummy_preds)
    print(f"[EVAL] GAUC: {res['GAUC']:.4f} | nDCG@5: {res['nDCG@5']:.4f} | Primary: {res['primary']:.4f}")

