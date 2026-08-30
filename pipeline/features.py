"""Feature engineering for KuaiRand-Pure.

Three encoders, all leak-free by construction:

  ``encode_features``            categorical IDs -> shared offset embedding space
  ``extract_dense_tabular_features``  causal (expanding-window) target statistics
  ``extract_sequential_features``     causal user watch history, in the same id
                                      space as ``encode_features``

WHY THE ENCODINGS ARE BUILT THE WAY THEY ARE
--------------------------------------------
1. *Causal target encoding.* Fitting statistics on train and then applying them
   to the same train rows lets each row see its own label. The model learns to
   trust a signal that is sharp at fit time and noisy at evaluation time, which
   both leaks and creates a train/serve mismatch. Here train rows for date *d*
   are encoded from dates *< d* only; valid/test are encoded from all of train.
   That is exactly the information a deployed model would have.

2. *User-side features are nearly inert.* Ranking happens within a user, so any
   feature constant across a user's impressions cannot change their order except
   by conditioning an interaction. The old feature set spent four of twelve slots
   on pure user aggregates. They are replaced by user x item affinity crosses
   (user x author, user x duration bucket, user x tab), which do vary within a
   user and carry real signal.
"""
import collections
from typing import Any, Dict, List, Tuple

import numpy as np

BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
CWM_VIDEO_FIELDS = ['music_id', 'video_type', 'upload_type']

#: Index of ``video_id`` within BASE_FIELDS — used to share embeddings with history.
VIDEO_FIELD_IDX = 1


def compute_dur_buckets(durations, n_buckets: int = 10) -> np.ndarray:
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n_buckets + 1)[1:-1])


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

class CategoricalEncoder:
    """Maps categorical values to a single shared offset id space.

    Every field gets a contiguous block of ids; unseen values fall into that
    field's UNK slot. Keeping one space means a model can share one embedding
    table across fields, which is what lets DIN compare a candidate video
    against history videos in the same space.
    """

    def __init__(self, field_names: List[str], vocabs, unk, field_dims, offsets, dur_edges):
        self.field_names = field_names
        self.vocabs = vocabs
        self.unk = unk
        self.field_dims = field_dims
        self.offsets = offsets
        self.dur_edges = dur_edges
        self.total_dim = int(sum(field_dims))
        #: One extra id past the end, reserved as the sequence padding slot.
        self.pad_id = self.total_dim
        self.embedding_rows = self.total_dim + 1

    def raw(self, row: Dict[str, Any]) -> List[str]:
        dur_idx = str(int(np.searchsorted(self.dur_edges, row['duration_ms'])))
        vals = [row['user_id'], row['video_id'], row['author_id'], str(row['tab']), dur_idx]
        if len(self.field_names) > len(BASE_FIELDS):
            vals += [str(v) for v in (row.get('v_extra') or ['UNK'] * len(CWM_VIDEO_FIELDS))]
        return vals

    def encode_row(self, row: Dict[str, Any]) -> List[int]:
        return [self.vocabs[i].get(v, self.unk[i]) + self.offsets[i]
                for i, v in enumerate(self.raw(row))]

    def video_id_to_slot(self, video_id: str) -> int:
        """Map a raw video_id into the same id the ``video_id`` field would get."""
        i = VIDEO_FIELD_IDX
        return self.vocabs[i].get(video_id, self.unk[i]) + self.offsets[i]


def encode_features(splits: Dict[str, List[dict]],
                    use_cwm_fields: bool = False
                    ) -> Tuple[Dict[str, tuple], CategoricalEncoder]:
    """Fit the categorical vocabulary on train, apply to every split.

    Returns ``({split: (X, y, users)}, encoder)``. ``y`` is ``-1`` for test rows,
    whose labels are withheld by the loader.
    """
    tr = splits['train']
    dur_edges = compute_dur_buckets([x['duration_ms'] for x in tr])

    field_names = list(BASE_FIELDS) + (CWM_VIDEO_FIELDS if use_cwm_fields else [])
    probe = CategoricalEncoder(field_names, [dict() for _ in field_names],
                               [], [], [], dur_edges)

    vocabs = [dict() for _ in field_names]
    for row in tr:
        for i, val in enumerate(probe.raw(row)):
            if val not in vocabs[i]:
                vocabs[i][val] = len(vocabs[i])

    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    enc = CategoricalEncoder(field_names, vocabs, unk, field_dims, offsets, dur_edges)

    out = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), len(field_names)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for idx, row in enumerate(rows):
            X[idx] = enc.encode_row(row)
            y[idx] = row['label']
            users.append(row['user_id'])
        out[name] = (X, y, users)
    return out, enc


# ---------------------------------------------------------------------------
# Causal target statistics
# ---------------------------------------------------------------------------

class _Counter:
    """Accumulates (positives, impressions) and reports an m-estimate rate."""

    __slots__ = ('pos', 'imp')

    def __init__(self):
        self.pos = 0.0
        self.imp = 0.0

    def add(self, y: float):
        self.pos += y
        self.imp += 1.0

    def rate(self, prior: float, m: float) -> float:
        return (self.pos + m * prior) / (self.imp + m)


class CausalStats:
    """Expanding-window engagement statistics and Short-Video Dynamics.

    ``observe(rows)`` folds a day's worth of impressions in; ``featurise(row)``
    reads the state as of everything observed so far. Encoding a split therefore
    means: for each date in ascending order, featurise that date's rows, then
    observe them.
    """

    M_ITEM = 15.0
    M_CROSS = 8.0

    FEATURE_NAMES = [
        # Baseline causal engagement
        'video_hist_count', 'video_hist_long_view_rate', 'video_hist_click_rate',
        'author_hist_count', 'author_hist_long_view_rate',
        'user_hist_count', 'user_hist_long_view_rate',

        # User x Item crosses (affinity back-off) & Creator Loyalty
        'user_author_affinity', 'user_durbucket_affinity', 'user_tab_affinity',
        'user_author_loyalty',

        # Duration dynamics & structural debiasing
        'log_duration', 'dur_bucket', 'dur_to_user_avg_ratio',
        'completion_ratio_prior', 'dur_norm_completion_score',

        # Clickbait duality (instant click vs true satisfaction)
        'video_click_longview_gap', 'video_click_to_longview_ratio',
        'author_click_longview_gap',

        # Topic fatigue & intra-day session dynamics
        'user_author_recent_streak', 'author_recency_gap',
        'user_session_depth',
    ]

    def __init__(self, dur_edges: np.ndarray, global_prior: float = 0.35):
        self.dur_edges = dur_edges
        self.prior = global_prior
        self.video = collections.defaultdict(_Counter)
        self.video_click = collections.defaultdict(_Counter)
        self.author = collections.defaultdict(_Counter)
        self.author_click = collections.defaultdict(_Counter)
        self.user = collections.defaultdict(_Counter)
        self.user_author = collections.defaultdict(_Counter)
        self.user_dur = collections.defaultdict(_Counter)
        self.user_tab = collections.defaultdict(_Counter)
        self.user_dur_sum = collections.defaultdict(float)
        self.user_dur_n = collections.defaultdict(float)
        # Mean completion ratio per duration bucket — a duration-bias correction
        # that needs no label at serve time.
        self.dur_completion = collections.defaultdict(lambda: [0.0, 0.0])
        # User exposure tracking (causal, label-free stream)
        self.user_recent_authors = collections.defaultdict(lambda: collections.deque(maxlen=10))
        self.user_daily_count = collections.defaultdict(int)

    def _bucket(self, duration_ms: float) -> int:
        return int(np.searchsorted(self.dur_edges, duration_ms))

    def observe(self, rows: List[dict]):
        for r in rows:
            y = r['label']
            if y < 0:          # withheld test label — never folded in
                continue
            u, v, a = r['user_id'], r['video_id'], r['author_id']
            b = self._bucket(r['duration_ms'])
            self.video[v].add(y)
            self.video_click[v].add(max(r.get('click', 0), 0))
            self.author[a].add(y)
            self.author_click[a].add(max(r.get('click', 0), 0))
            self.user[u].add(y)
            self.user_author[(u, a)].add(y)
            self.user_dur[(u, b)].add(y)
            self.user_tab[(u, r['tab'])].add(y)
            self.user_dur_sum[u] += r['duration_ms']
            self.user_dur_n[u] += 1.0
            dur = max(r['duration_ms'], 1.0)
            slot = self.dur_completion[b]
            slot[0] += min(r.get('play_time_ms', 0.0) / dur, 5.0)
            slot[1] += 1.0

    def featurise(self, row: dict) -> List[float]:
        u, v, a = row['user_id'], row['video_id'], row['author_id']
        dur = row['duration_ms']
        d = row.get('date', 0)
        b = self._bucket(dur)

        vc, ac, uc = self.video[v], self.author[a], self.user[u]
        v_click = self.video_click[v]
        a_click = self.author_click[a]

        v_long_rate = vc.rate(self.prior, self.M_ITEM)
        v_click_rate = v_click.rate(0.15, self.M_ITEM)
        a_long_rate = ac.rate(self.prior, self.M_ITEM)
        a_click_rate = a_click.rate(0.15, self.M_ITEM)
        u_rate = uc.rate(self.prior, self.M_ITEM)

        avg_dur = (self.user_dur_sum[u] / self.user_dur_n[u]) if self.user_dur_n[u] else dur
        comp = self.dur_completion[b]
        comp_prior = (comp[0] / comp[1]) if comp[1] else self.prior

        # User x item crosses (smooth Bayesian back-off to user prior)
        user_auth_aff = self.user_author[(u, a)].rate(u_rate, self.M_CROSS)
        user_dur_aff = self.user_dur[(u, b)].rate(u_rate, self.M_CROSS)
        user_tab_aff = self.user_tab[(u, row['tab'])].rate(u_rate, self.M_CROSS)

        # Creator loyalty: user's completion rate on this creator vs baseline
        user_auth_loyalty = user_auth_aff / (u_rate + 0.05)

        # Duration-normalized completion expectation
        dur_norm_comp = v_long_rate / (comp_prior + 1e-4)

        # Clickbait duality metrics (gap and ratio)
        clickbait_gap = v_click_rate - v_long_rate
        clickbait_ratio = v_click_rate / (v_long_rate + 0.05)
        author_clickbait_gap = a_click_rate - a_long_rate

        # Topic fatigue: recent consecutive author streak and position recency gap
        recent_authors = list(self.user_recent_authors[u])
        author_streak = 0.0
        for past_a in reversed(recent_authors):
            if past_a == a:
                author_streak += 1.0
            else:
                break

        author_recency = 10.0
        for step, past_a in enumerate(reversed(recent_authors), start=1):
            if past_a == a:
                author_recency = float(step)
                break

        # Intra-day session depth
        daily_key = (d, u)
        session_depth = np.log1p(self.user_daily_count[daily_key])

        # Update streaming exposure buffers causally (no label consulted)
        self.user_recent_authors[u].append(a)
        self.user_daily_count[daily_key] += 1

        return [
            np.log1p(vc.imp),
            v_long_rate,
            v_click_rate,
            np.log1p(ac.imp),
            a_long_rate,
            np.log1p(uc.imp),
            u_rate,
            user_auth_aff,
            user_dur_aff,
            user_tab_aff,
            user_auth_loyalty,
            np.log1p(dur),
            float(b),
            float(dur / (avg_dur + 1.0)),
            comp_prior,
            dur_norm_comp,
            clickbait_gap,
            clickbait_ratio,
            author_clickbait_gap,
            author_streak,
            author_recency,
            session_depth,
        ]


def extract_dense_tabular_features(splits: Dict[str, List[dict]]
                                   ) -> Tuple[Dict[str, tuple], List[str]]:
    """Dense engagement features, encoded causally.

    Train rows on date *d* see statistics from dates *< d*. Valid and test rows
    see statistics from all of train — the same information a model deployed at
    the end of the training window would have.
    """
    tr = splits['train']
    dur_edges = compute_dur_buckets([x['duration_ms'] for x in tr])
    prior = float(np.mean([r['label'] for r in tr])) if tr else 0.35
    stats = CausalStats(dur_edges, global_prior=prior)
    names = CausalStats.FEATURE_NAMES

    def pack(rows: List[dict], feats: List[List[float]]):
        X = np.asarray(feats, dtype=np.float32).reshape(len(rows), len(names))
        y = np.asarray([r['label'] for r in rows], dtype=np.float32)
        return X, y, [r['user_id'] for r in rows]

    out: Dict[str, tuple] = {}

    # --- train: expanding window, strictly causal ---
    # Featurise a date's rows from everything strictly before it, then fold that
    # date in. Working over positions keeps the original row order intact.
    by_date: Dict[int, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(tr):
        by_date[r['date']].append(i)

    feats: List[List[float]] = [None] * len(tr)
    for date in sorted(by_date):
        idxs = by_date[date]
        for i in idxs:
            feats[i] = stats.featurise(tr[i])
        stats.observe([tr[i] for i in idxs])
    out['train'] = pack(tr, feats)

    # --- valid / test: full-train statistics, frozen ---
    for name in ('valid', 'test'):
        if name in splits:
            out[name] = pack(splits[name], [stats.featurise(r) for r in splits[name]])

    return out, names


# ---------------------------------------------------------------------------
# Sequential history
# ---------------------------------------------------------------------------

def extract_sequential_features(splits: Dict[str, List[dict]],
                                encoder: CategoricalEncoder,
                                max_seq_len: int = 10) -> Dict[str, np.ndarray]:
    """Per-row history of the user's most recent previously-seen videos.

    Two properties matter, and the previous implementation had neither:

    * **Causal.** Every impression is appended to the user's history after it has
      been featurised, regardless of label. The old code appended a valid/test row
      only ``if r['label'] == 1``, which fed evaluation labels into the features of
      later rows for the same user — a direct leak.
    * **Shared id space.** History is emitted in ``encoder``'s id space, so DIN can
      use one embedding table for candidate and history. The old code used a
      separate table, leaving target attention comparing unrelated vectors.

    Padding uses ``encoder.pad_id`` and is left-aligned so the most recent item is
    always last.
    """
    history = collections.defaultdict(collections.deque)
    out: Dict[str, np.ndarray] = {}

    # Chronological order across splits, so valid sees train history and test sees both.
    for name in ('train', 'valid', 'test'):
        if name not in splits:
            continue
        rows = splits[name]
        mat = np.full((len(rows), max_seq_len), encoder.pad_id, dtype=np.int64)
        for i, r in enumerate(rows):
            h = history[r['user_id']]
            if h:
                recent = list(h)[-max_seq_len:]
                mat[i, max_seq_len - len(recent):] = recent
            # Append AFTER featurising — no label consulted.
            h.append(encoder.video_id_to_slot(r['video_id']))
            if len(h) > max_seq_len * 4:
                h.popleft()
        out[name] = mat
    return out
