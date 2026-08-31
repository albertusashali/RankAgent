"""KuaiRand-Pure loader with strict date-based splits and a hard hidden-test guard.

Row order is significant: the submission format indexes rows by position, so we
read ``log_standard_4_08_to_4_21_pure.csv`` first, then
``log_standard_4_22_to_5_08_pure.csv``, filter by date, and preserve file order —
byte-identical to the starter kit's ``data.load()``.

THE HIDDEN-TEST GUARD
---------------------
Challenge rule: "The agent develops using only the training split and the public
validation feedback — it never has access to the hidden test set."

That rule is enforced here in code rather than by convention, because KuaiRand's
test labels happen to sit in the same CSVs as everything else:

  * ``load_kuairand()`` returns **train and valid only**.
  * Test rows require ``include_test=True`` and come back with ``label = -1``.
    They carry features for scoring, never targets.
  * Real test labels are reachable only through ``load_test_labels()``, which
    refuses to run unless ``RANKAGENT_UNSEAL_TEST=1`` is set in the environment.
    Nothing in the agent loop sets it; it exists for the post-hoc scoring script
    a human runs after the submission is frozen.

So an agent iteration that tries to select on test performance fails loudly
instead of silently leaking.
"""
import csv
import os
from typing import Dict, List, Optional

LABEL = 'long_view'
SPLITS = {
    'train': (20220408, 20220421),
    'valid': (20220422, 20220428),
    'test':  (20220429, 20220508),
}

#: Splits the agent may develop against.
OPEN_SPLITS = ('train', 'valid')

#: Sentinel written into the ``label`` field of every test row.
WITHHELD = -1

USER_FE = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range',
           'friend_user_num_range', 'user_active_degree']
VID_FE = ['author_id', 'music_id', 'video_type', 'upload_type']

LOG_FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')

_CANDIDATE_DIRS = [
    "./data/KuaiRand-Pure/data",
    "./kuairand-starter-kit/KuaiRand-Pure/data",
    "./KuaiRand-Pure/data",
    "./data/KuaiRand-Pure/KuaiRand-Pure/data",
    "../data/KuaiRand-Pure/data",
    "./data",
]


def find_data_dir(data_dir: Optional[str] = None) -> str:
    """Resolve the KuaiRand-Pure data directory, or explain where to get it."""
    candidates = [data_dir] if data_dir else []
    candidates += _CANDIDATE_DIRS
    for d in candidates:
        if d and os.path.exists(os.path.join(d, LOG_FILES[0])):
            return d
    raise FileNotFoundError(
        "KuaiRand-Pure data not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\n\nFetch it with `make data`, or:\n"
          "  mkdir -p data && cd data\n"
          "  curl -L -O https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz\n"
          "  tar xzf KuaiRand-Pure.tar.gz"
    )


def _read_side_features(actual_dir: str, include_extra_features: bool):
    vid2info: Dict[str, List[str]] = {}
    vid_path = os.path.join(actual_dir, 'video_features_basic_pure.csv')
    if os.path.exists(vid_path):
        with open(vid_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                vid2info[r['video_id']] = [r.get(k, 'UNK') for k in VID_FE]

    u2info: Dict[str, List[str]] = {}
    user_path = os.path.join(actual_dir, 'user_features_pure.csv')
    if include_extra_features and os.path.exists(user_path):
        with open(user_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                u2info[r['user_id']] = [r.get(k, 'UNK') for k in USER_FE]
    return vid2info, u2info


def load_kuairand(data_dir: Optional[str] = None,
                  include_extra_features: bool = False,
                  include_test: bool = False) -> Dict[str, List[dict]]:
    """Load interaction logs, split by date.

    Returns ``{'train': [...], 'valid': [...]}`` and, when ``include_test`` is
    set, a ``'test'`` list whose rows carry features but no outcomes.

    Every row is a dict with the 5 base fields, the auxiliary feedback signals
    (``click``/``like``/``follow``/``comment``/``forward``), ``play_time_ms``
    for watch-time modelling, ``duration_ms``, and optional side features.

    **What "withheld" covers.** On the hidden test split, every *post-impression
    outcome* is set to ``WITHHELD`` (-1): the six binary signals and
    ``play_time_ms``. Pre-impression context — ``duration_ms``, ``tab``,
    ``is_rand``, the ids and the side features — is exposed on every split,
    because it is knowable at ranking time. The distinction matters:
    ``long_view`` is close to a deterministic function of
    ``play_time_ms / duration_ms``, so exposing watch time on the test split
    would hand out the label.
    """
    actual_dir = find_data_dir(data_dir)
    vid2info, u2info = _read_side_features(actual_dir, include_extra_features)

    wanted = dict(SPLITS) if include_test else {k: SPLITS[k] for k in OPEN_SPLITS}
    lo_all = min(lo for lo, _ in wanted.values())
    hi_all = max(hi for _, hi in wanted.values())

    unk_v = ['UNK'] * len(VID_FE)
    unk_u = ['UNK'] * len(USER_FE)
    out: Dict[str, List[dict]] = {name: [] for name in wanted}

    for fname in LOG_FILES:
        full_path = os.path.join(actual_dir, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing required log file: {full_path}")
        with open(full_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                if not (lo_all <= date <= hi_all):
                    continue
                split = None
                for name, (lo, hi) in wanted.items():
                    if lo <= date <= hi:
                        split = name
                        break
                if split is None:
                    continue

                video_id = r['video_id']
                vinfo = vid2info.get(video_id, unk_v)
                is_test = (split == 'test')

                out[split].append({
                    'date': date,
                    'user_id': r['user_id'],
                    'video_id': video_id,
                    'author_id': vinfo[0],
                    'tab': r.get('tab', '1'),
                    # duration_ms is a property of the video, known before the
                    # impression, so it is safe to expose on every split.
                    'duration_ms': float(r.get('duration_ms', 0.0)),
                    # --- outcomes: withheld on the hidden test split ---
                    # play_time_ms is a POST-impression outcome, not a feature.
                    # long_view is ~98% determined by play_time/duration, so
                    # leaving it unredacted here would let any feature derived
                    # from it score near-perfectly on the hidden test while being
                    # pure label leakage. It was previously passed through
                    # ungated; nothing read it per-row, but that made it a
                    # landmine rather than a safe omission.
                    'play_time_ms': float(WITHHELD) if is_test
                                    else float(r.get('play_time_ms', 0.0)),
                    'label':   WITHHELD if is_test else (1 if r.get(LABEL, '0') != '0' else 0),
                    'click':   WITHHELD if is_test else (1 if r.get('is_click', '0') != '0' else 0),
                    'like':    WITHHELD if is_test else (1 if r.get('is_like', '0') != '0' else 0),
                    'follow':  WITHHELD if is_test else (1 if r.get('is_follow', '0') != '0' else 0),
                    'comment': WITHHELD if is_test else (1 if r.get('is_comment', '0') != '0' else 0),
                    'forward': WITHHELD if is_test else (1 if r.get('is_forward', '0') != '0' else 0),
                    'is_rand': 1 if r.get('is_rand', '0') != '0' else 0,
                    'v_extra': vinfo[1:],
                    'u_extra': u2info.get(r['user_id'], unk_u) if include_extra_features else [],
                })
    return out


def load_test_labels(data_dir: Optional[str] = None) -> List[int]:
    """Hidden-test labels, in evaluation-split row order. SEALED BY DEFAULT.

    Raises unless ``RANKAGENT_UNSEAL_TEST=1``. This exists so a human can score a
    frozen submission after the fact; the agent loop must never call it, and the
    environment variable is never set anywhere inside the loop.
    """
    if os.environ.get("RANKAGENT_UNSEAL_TEST") != "1":
        raise PermissionError(
            "Hidden-test labels are sealed. The agent develops on train + valid only.\n"
            "If you are scoring a frozen submission after the fact, run with "
            "RANKAGENT_UNSEAL_TEST=1 (see scripts/score_frozen_submission.py)."
        )
    actual_dir = find_data_dir(data_dir)
    lo, hi = SPLITS['test']
    labels: List[int] = []
    for fname in LOG_FILES:
        with open(os.path.join(actual_dir, fname), encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if lo <= int(r['date']) <= hi:
                    labels.append(1 if r.get(LABEL, '0') != '0' else 0)
    return labels


def load_unbiased_valid(data_dir: Optional[str] = None) -> List[dict]:
    """Randomised-exposure log (1.18M rows) as an unbiased second validation set.

    These impressions were served by a random policy rather than the production
    ranker, so they are free of the logging-policy bias in the standard logs.
    Useful as a sanity check that a gain is real and not an artefact of fitting
    the logged policy. Overlaps the valid/test date range, so use it for
    diagnosis only — never as the selection metric.
    """
    actual_dir = find_data_dir(data_dir)
    vid2info, _ = _read_side_features(actual_dir, False)
    unk_v = ['UNK'] * len(VID_FE)
    path = os.path.join(actual_dir, 'log_random_4_22_to_5_08_pure.csv')
    rows: List[dict] = []
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            vinfo = vid2info.get(r['video_id'], unk_v)
            rows.append({
                'date': int(r['date']),
                'user_id': r['user_id'],
                'video_id': r['video_id'],
                'author_id': vinfo[0],
                'tab': r.get('tab', '1'),
                'duration_ms': float(r.get('duration_ms', 0.0)),
                'play_time_ms': float(r.get('play_time_ms', 0.0)),
                'label': 1 if r.get(LABEL, '0') != '0' else 0,
                'v_extra': vinfo[1:],
                'u_extra': [],
            })
    return rows
