"""Validation-only segment diagnostics; never used to access hidden labels."""
import argparse
import collections
import json
import os

import numpy as np

from pipeline.data import load_kuairand
from pipeline.evaluate import evaluate
from pipeline.submit import predict_split


def segment_report(splits, predictions) -> dict:
    train, valid = splits['train'], splits['valid']
    user_n = collections.Counter(r['user_id'] for r in train)
    video_n = collections.Counter(r['video_id'] for r in train)
    durations = np.asarray([r['duration_ms'] for r in train], dtype=np.float64)
    dur_mid = float(np.median(durations))
    user_values = np.asarray(list(user_n.values()))
    video_values = np.asarray(list(video_n.values()))
    user_mid = float(np.median(user_values))
    video_mid = float(np.median(video_values))

    masks = {
        'all': np.ones(len(valid), dtype=bool),
        'cold_user': np.asarray([user_n[r['user_id']] == 0 for r in valid]),
        'warm_user': np.asarray([user_n[r['user_id']] > user_mid for r in valid]),
        'tail_video': np.asarray([video_n[r['video_id']] <= video_mid for r in valid]),
        'head_video': np.asarray([video_n[r['video_id']] > video_mid for r in valid]),
        'short_video': np.asarray([r['duration_ms'] <= dur_mid for r in valid]),
        'long_video': np.asarray([r['duration_ms'] > dur_mid for r in valid]),
    }
    users = np.asarray([r['user_id'] for r in valid])
    labels = np.asarray([r['label'] for r in valid])
    preds = np.asarray(predictions)
    report = {}
    for name, mask in masks.items():
        if mask.any():
            report[name] = {**evaluate(users[mask], labels[mask], preds[mask]),
                            'fraction': float(mask.mean())}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description='Validation-only model segment diagnostics')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', default=None)
    p.add_argument('--output', default=None)
    args = p.parse_args(argv)
    splits = load_kuairand(args.data_dir)
    report = segment_report(splits, predict_split(args.checkpoint, splits, 'valid'))
    out = args.output or os.path.join('logs', f'{args.checkpoint}_segments.json')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f"[SEGMENTS] validation-only report -> {out}")


if __name__ == '__main__':
    main()
