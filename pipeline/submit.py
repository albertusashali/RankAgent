"""
KuaiRand Submission Generator and Strict Schema Checker.
Matches Starter Kit submit.py.
"""
import os
import argparse
import numpy as np
import pandas as pd
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate submission CSV')
    parser.add_argument('--file', default='submission.csv', help='Submission file path')
    args = parser.parse_args()
    
    if args.check:
        check_submission(args.file)

