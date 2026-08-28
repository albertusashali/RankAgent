"""
KuaiRand Submission Generator and Strict Schema Checker.
Matches Starter Kit submit.py.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Optional

def generate_submission(test_rows: list, predictions: np.ndarray, output_path: str = "submission.csv"):
    """
    Generates submission CSV matching starter-kit constraints:
    header: row_id,user_id,video_id,score
    """
    if len(test_rows) != len(predictions):
        raise ValueError(f"Length mismatch: {len(test_rows)} rows vs {len(predictions)} predictions.")
        
    df = pd.DataFrame({
        'row_id': np.arange(len(test_rows)),
        'user_id': [r['user_id'] for r in test_rows],
        'video_id': [r['video_id'] for r in test_rows],
        'score': predictions
    })
    
    if df['score'].isnull().any() or np.isinf(df['score']).any():
        raise ValueError("Predictions contain NaN or Inf values!")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUBMIT] Wrote {len(df):,} rows to {output_path} successfully.")

def check_submission(submission_path: str, expected_rows: int = 170588) -> bool:
    """
    Validates submission against starter-kit requirements:
    1. Header must be row_id,user_id,video_id,score
    2. Exactly 170,588 rows for KuaiRand-Pure test set
    3. row_id must be strictly 0, 1, 2, ..., N-1
    4. No NaNs or Infs
    """
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file does not exist: {submission_path}")
        
    df = pd.read_csv(submission_path)
    expected_header = ['row_id', 'user_id', 'video_id', 'score']
    if list(df.columns) != expected_header:
        raise ValueError(f"Invalid header: {list(df.columns)}. Expected: {expected_header}")
        
    if len(df) != expected_rows:
        raise ValueError(f"Row count mismatch: got {len(df)}, expected {expected_rows}")
        
    if not np.array_equal(df['row_id'].values, np.arange(len(df))):
        raise ValueError("row_id must be strictly continuous 0-based integers (0, 1, 2, ..., N-1)")
        
    if df['score'].isnull().any():
        raise ValueError("Submission contains NaN values in 'score' column!")
        
    if np.isinf(df['score']).any():
        raise ValueError("Submission contains Inf values in 'score' column!")
        
    print(f"[VALIDATION PASS] {submission_path} conforms to all Starter Kit requirements! ({len(df):,} valid rows)")
    return True

def generate_from_best_checkpoint(output_path: str = "submission.csv"):
    """Loads best checkpoint and exports verified submission.csv."""
    from pipeline.data import load_kuairand
    from pipeline.features import encode_features
    from pipeline.models import MMoE, NumpyFM, DEVICE
    
    splits = load_kuairand()
    te_rows = splits['test']
    
    mmoe_ckpt = "checkpoints/best_mmoe.pt"
    fm_ckpt = "checkpoints/best_fm.npz"
    
    if os.path.exists(mmoe_ckpt):
        print(f"==> Generating submission from {mmoe_ckpt}...")
        enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=False)
        Xte_c = enc_cat['test'][0]
        
        weights = torch.load(mmoe_ckpt, map_location=DEVICE, weights_only=True)
        model = MMoE(total_dim, num_fields=len(field_names), embed_dim=16, num_experts=4, expert_dim=64, num_tasks=3).to(DEVICE)
        model.load_state_dict(weights)
        model.eval()
        
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xte_c), 65536):
                batch_c = torch.from_numpy(Xte_c[i:i+65536]).long().to(DEVICE)
                out_long, _, _ = model(batch_c)
                preds.append(out_long.cpu().numpy())
            te_preds = np.concatenate(preds)
    elif os.path.exists(fm_ckpt):
        print(f"==> Generating submission from {fm_ckpt}...")
        enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=False)
        Xte_c = enc_cat['test'][0]
        data = np.load(fm_ckpt)
        m = NumpyFM(total_dim, k=16)
        m.V, m.W, m.b = data['V'], data['W'], data['b']
        te_preds = m.predict(Xte_c)
    else:
        raise FileNotFoundError("No checkpoint found in checkpoints/. Run pipeline.train first!")
        
    generate_submission(te_rows, te_preds, output_path)
    check_submission(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate submission from best checkpoint')
    parser.add_argument('--check', action='store_true', help='Validate submission CSV')
    parser.add_argument('--file', default='submission.csv', help='Submission file path')
    args = parser.parse_args()
    
    if args.generate:
        generate_from_best_checkpoint(args.file)
    elif args.check:
        check_submission(args.file)
