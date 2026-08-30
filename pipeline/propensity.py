"""Randomized-exposure density-ratio estimation for training ablations."""
from typing import List, Sequence

import numpy as np


def _tokens(row: dict) -> list:
    duration = max(float(row.get('duration_ms', 0.0)), 0.0)
    bucket = int(np.clip(np.log2(duration + 1.0), 0, 20))
    return [f"video={row['video_id']}", f"author={row['author_id']}",
            f"tab={row.get('tab', '1')}", f"duration_log2={bucket}"]


def estimate_random_exposure_weights(standard_rows: Sequence[dict], random_rows: Sequence[dict],
                                     mode: str = 'snips', clip: float = 10.0,
                                     max_fit_rows: int = 300000,
                                     seed: int = 0) -> np.ndarray:
    """Estimate p(random|x)/p(standard|x) using only pre-exposure covariates.

    This is a density-ratio ablation, not a claim that treatment is fully
    unconfounded. Clipping controls variance; SNIPS additionally normalizes the
    weights to mean one. The official validation set remains the sole selector.
    """
    if mode == 'none':
        return np.ones(len(standard_rows), dtype=np.float32)
    if mode not in ('ips', 'snips'):
        raise ValueError("propensity mode must be none, ips, or snips")
    if not random_rows:
        raise ValueError('randomized-exposure log is empty')
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import SGDClassifier

    rng = np.random.default_rng(seed)
    n_std = min(len(standard_rows), max_fit_rows // 2)
    n_rnd = min(len(random_rows), max_fit_rows // 2)
    si = rng.choice(len(standard_rows), n_std, replace=False)
    ri = rng.choice(len(random_rows), n_rnd, replace=False)
    hasher = FeatureHasher(n_features=2 ** 18, input_type='string', alternate_sign=False)
    fit_rows = [_tokens(standard_rows[i]) for i in si] + [_tokens(random_rows[i]) for i in ri]
    X = hasher.transform(fit_rows)
    y = np.r_[np.zeros(n_std, dtype=np.int8), np.ones(n_rnd, dtype=np.int8)]
    clf = SGDClassifier(loss='log_loss', alpha=1e-6, max_iter=30,
                        class_weight='balanced', random_state=seed, n_jobs=-1)
    clf.fit(X, y)

    out = np.empty(len(standard_rows), dtype=np.float32)
    batch = 100000
    for start in range(0, len(standard_rows), batch):
        rows = standard_rows[start:start + batch]
        p = clf.predict_proba(hasher.transform([_tokens(r) for r in rows]))[:, 1]
        odds = p / np.maximum(1.0 - p, 1e-6)
        out[start:start + len(rows)] = np.clip(odds, 1.0 / clip, clip)
    if mode == 'snips':
        out /= max(float(out.mean()), 1e-12)
    return out
