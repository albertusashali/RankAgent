"""
Feature Engineering and Categorical Encoding module for KuaiRand.
Includes:
- Categorical ID Encodings
- Hypothesis 1.1: Historical User/Item Engagement Aggregations
- Hypothesis 1.2: Smooth Bayesian Target Encoding (m-estimate)
- Hypothesis 1.4: Sequential User Watch Histories (DIN Sequences)
"""
import collections
import numpy as np
from typing import Dict, List, Tuple, Any

BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def compute_dur_buckets(durations: List[float], n_buckets: int = 10) -> np.ndarray:
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n_buckets + 1)[1:-1])

def compute_historical_statistics(tr_rows: List[Dict[str, Any]], m_smoothing: float = 15.0) -> Dict[str, Any]:
    global_long_views = [r['label'] for r in tr_rows]
    global_mean = float(np.mean(global_long_views)) if global_long_views else 0.35

    user_stats = collections.defaultdict(lambda: {'count': 0, 'long_view': 0, 'click': 0, 'like': 0, 'dur_sum': 0.0})
    video_stats = collections.defaultdict(lambda: {'count': 0, 'long_view': 0, 'click': 0})
    author_stats = collections.defaultdict(lambda: {'count': 0, 'long_view': 0})

    for r in tr_rows:
        u = r['user_id']
        v = r['video_id']
        a = r['author_id']
        y = r['label']
        c = r.get('click', 0)
        l = r.get('like', 0)
        dur = r['duration_ms']

        user_stats[u]['count'] += 1
        user_stats[u]['long_view'] += y
        user_stats[u]['click'] += c
        user_stats[u]['like'] += l
        user_stats[u]['dur_sum'] += dur

        video_stats[v]['count'] += 1
        video_stats[v]['long_view'] += y
        video_stats[v]['click'] += c

        author_stats[a]['count'] += 1
        author_stats[a]['long_view'] += y

    user_smooth_rates = {u: (st['long_view'] + m_smoothing * global_mean) / (st['count'] + m_smoothing) for u, st in user_stats.items()}
    user_avg_durs = {u: st['dur_sum'] / max(1, st['count']) for u, st in user_stats.items()}
    video_smooth_rates = {v: (st['long_view'] + m_smoothing * global_mean) / (st['count'] + m_smoothing) for v, st in video_stats.items()}
    author_smooth_rates = {a: (st['long_view'] + m_smoothing * global_mean) / (st['count'] + m_smoothing) for a, st in author_stats.items()}

    return {
        'global_mean': global_mean,
        'user_stats': user_stats,
        'user_smooth_rates': user_smooth_rates,
        'user_avg_durs': user_avg_durs,
        'video_stats': video_stats,
        'video_smooth_rates': video_smooth_rates,
        'author_stats': author_stats,
        'author_smooth_rates': author_smooth_rates
    }

def extract_dense_tabular_features(splits: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]], List[str]]:
    tr = splits['train']
    hist = compute_historical_statistics(tr, m_smoothing=15.0)
    dur_edges = compute_dur_buckets([x['duration_ms'] for x in tr])

    feature_names = [
        'user_hist_count', 'user_hist_long_view_rate', 'user_hist_click_rate', 'user_hist_like_rate',
        'video_hist_count', 'video_hist_long_view_rate', 'video_hist_click_rate',
        'author_hist_count', 'author_hist_long_view_rate',
        'log_duration', 'dur_bucket', 'dur_to_user_avg_ratio'
    ]

    g_mean = hist['global_mean']
    u_rates = hist['user_smooth_rates']
    u_stats = hist['user_stats']
    u_durs = hist['user_avg_durs']
    v_rates = hist['video_smooth_rates']
    v_stats = hist['video_stats']
    a_rates = hist['author_smooth_rates']
    a_stats = hist['author_stats']

    enc = {}
    for name, split_rows in splits.items():
        n_samples = len(split_rows)
        X = np.empty((n_samples, len(feature_names)), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)
        users = []

        for idx, row in enumerate(split_rows):
            u = row['user_id']
            v = row['video_id']
            a = row['author_id']
            dur = row['duration_ms']

            u_st = u_stats.get(u, {'count': 0, 'click': 0, 'like': 0})
            u_cnt = np.log1p(u_st['count'])
            u_rate = u_rates.get(u, g_mean)
            u_click_rate = (u_st['click'] + 5.0 * 0.15) / (u_st['count'] + 5.0)
            u_like_rate = (u_st['like'] + 5.0 * 0.05) / (u_st['count'] + 5.0)
            u_avg_dur = u_durs.get(u, dur)

            v_st = v_stats.get(v, {'count': 0, 'click': 0})
            v_cnt = np.log1p(v_st['count'])
            v_rate = v_rates.get(v, g_mean)
            v_click_rate = (v_st['click'] + 5.0 * 0.15) / (v_st['count'] + 5.0)

            a_st = a_stats.get(a, {'count': 0})
            a_cnt = np.log1p(a_st['count'])
            a_rate = a_rates.get(a, g_mean)

            log_dur = np.log1p(dur)
            dur_b = float(np.searchsorted(dur_edges, dur))
            dur_ratio = float(dur / (u_avg_dur + 1.0))

            X[idx, 0] = u_cnt
            X[idx, 1] = u_rate
            X[idx, 2] = u_click_rate
            X[idx, 3] = u_like_rate
            X[idx, 4] = v_cnt
            X[idx, 5] = v_rate
            X[idx, 6] = v_click_rate
            X[idx, 7] = a_cnt
            X[idx, 8] = a_rate
            X[idx, 9] = log_dur
            X[idx, 10] = dur_b
            X[idx, 11] = dur_ratio

            y[idx] = row['label']
            users.append(u)

        enc[name] = (X, y, users)

    return enc, feature_names

def extract_sequential_features(splits: Dict[str, List[Dict[str, Any]]], max_seq_len: int = 5) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Extracts sequential past watched video histories for Deep Interest Network (DIN).
    Index 0 is reserved for padding.
    """
    tr = splits['train']
    vid_vocab = {}
    for r in tr:
        v = r['video_id']
        if v not in vid_vocab:
            vid_vocab[v] = len(vid_vocab) + 1  # 1-indexed

    num_vids = len(vid_vocab) + 2

    # Track user recent watch histories chronologically
    user_hist = collections.defaultdict(list)
    seqs = {}

    for split_name in ['train', 'valid', 'test']:
        rows = splits[split_name]
        hist_matrix = np.zeros((len(rows), max_seq_len), dtype=np.int64)

        for i, r in enumerate(rows):
            u = r['user_id']
            v = r['video_id']
            v_idx = vid_vocab.get(v, 1)

            # Get recent history
            h = user_hist[u]
            if len(h) > 0:
                recent = h[-max_seq_len:]
                hist_matrix[i, -len(recent):] = recent

            # Update history if user watched long
            if r['label'] == 1 or split_name == 'train':
                user_hist[u].append(v_idx)

        seqs[split_name] = hist_matrix

    return seqs, num_vids

def encode_features(splits: Dict[str, List[Dict[str, Any]]], use_cwm_fields: bool = False) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]], int, List[str]]:
    tr = splits['train']
    dur_edges = compute_dur_buckets([x['duration_ms'] for x in tr])

    field_names = list(BASE_FIELDS)
    if use_cwm_fields:
        field_names += ['music_id', 'video_type', 'upload_type']

    def extract_raw(row: Dict[str, Any]) -> List[str]:
        dur_idx = str(int(np.searchsorted(dur_edges, row['duration_ms'])))
        vals = [row['user_id'], row['video_id'], row['author_id'], str(row['tab']), dur_idx]
        if use_cwm_fields:
            vals += [str(v) for v in row.get('v_extra', ['UNK', 'UNK', 'UNK'])]
        return vals

    vocabs = [dict() for _ in field_names]
    for row in tr:
        for i, val in enumerate(extract_raw(row)):
            if val not in vocabs[i]:
                vocabs[i][val] = len(vocabs[i])

    unk_indices = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, split_rows in splits.items():
        n_samples = len(split_rows)
        X = np.empty((n_samples, len(field_names)), dtype=np.int32)
        y = np.empty(n_samples, dtype=np.float32)
        users = []
        
        for idx, row in enumerate(split_rows):
            raw_vals = extract_raw(row)
            for i, val in enumerate(raw_vals):
                X[idx, i] = vocabs[i].get(val, unk_indices[i]) + offsets[i]
            y[idx] = row['label']
            users.append(row['user_id'])
            
        enc[name] = (X, y, users)

    total_dim = int(sum(field_dims))
    return enc, total_dim, field_names
