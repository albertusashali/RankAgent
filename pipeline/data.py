"""
KuaiRand-Pure Data Loader with strict date-based splitting.
"""
import os
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional

LABEL = 'long_view'
SPLITS = {
    'train': (20220408, 20220421),
    'valid': (20220422, 20220428),
    'test':  (20220429, 20220508)
}

# 13 CWM Feature Fields
USER_FE = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range',
           'friend_user_num_range', 'user_active_degree']
VID_FE = ['author_id', 'music_id', 'video_type', 'upload_type']

def find_data_dir(candidate_dirs: Optional[List[str]] = None) -> str:
    """Auto-detect data directory location."""
    candidates = candidate_dirs or [
        "./data/KuaiRand-Pure/KuaiRand-Pure/data",
        "./data/KuaiRand-Pure/data",
        "./data",
        "../data/KuaiRand-Pure/KuaiRand-Pure/data"
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "log_standard_4_08_to_4_21_pure.csv")):
            return d
    return "./data"

def load_kuairand(data_dir: Optional[str] = None, include_extra_features: bool = False) -> Dict[str, List]:
    """
    Loads KuaiRand interaction logs and splits them strictly by date.
    Returns: dict mapping split name ('train', 'valid', 'test') to list of rows.
    """
    actual_dir = data_dir if (data_dir and os.path.exists(data_dir)) else find_data_dir()
    
    # 1. Load video features if available
    vid2info = {}
    vid_path = os.path.join(actual_dir, 'video_features_basic_pure.csv')
    if os.path.exists(vid_path):
        with open(vid_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                vid2info[r['video_id']] = [r.get(k, 'UNK') for k in VID_FE]

    # 2. Load user features if available
    u2info = {}
    user_path = os.path.join(actual_dir, 'user_features_pure.csv')
    if os.path.exists(user_path) and include_extra_features:
        with open(user_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                u2info[r['user_id']] = [r.get(k, 'UNK') for k in USER_FE]

    # 3. Load interaction logs
    rows = []
    log_files = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')
    for f in log_files:
        full_path = os.path.join(actual_dir, f)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing required log file: {full_path}")
        with open(full_path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                user_id = r['user_id']
                video_id = r['video_id']
                tab = r.get('tab', '1')
                duration_ms = float(r.get('duration_ms', 0.0))
                label = 1 if r.get(LABEL, '0') != '0' else 0
                
                # Basic author_id
                vinfo = vid2info.get(video_id, ['UNK'] * len(VID_FE))
                author_id = vinfo[0] if vinfo else 'UNK'
                
                # Multi-task auxiliary labels (if present)
                click = 1 if r.get('is_click', r.get('click', '0')) != '0' else 0
                like = 1 if r.get('is_like', r.get('like', '0')) != '0' else 0
                
                row_data = {
                    'date': date,
                    'user_id': user_id,
                    'video_id': video_id,
                    'author_id': author_id,
                    'tab': tab,
                    'duration_ms': duration_ms,
                    'label': label,
                    'click': click,
                    'like': like,
                    'v_extra': vinfo[1:] if len(vinfo) > 1 else [],
                    'u_extra': u2info.get(user_id, ['UNK'] * len(USER_FE)) if include_extra_features else []
                }
                rows.append(row_data)

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x['date'] <= hi]
        
    return out

