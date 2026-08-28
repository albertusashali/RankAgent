"""
Training and Optimization harness for KuaiRand recommendation models.
Supports Factorization Machines, DeepFM (with dense features), MMoE, and LightGBM GBDT Ranker.
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any

from pipeline.data import load_kuairand
from pipeline.features import encode_features, extract_dense_tabular_features
from pipeline.models import NumpyFM, DeepFM, MMoE, DEVICE
from pipeline.evaluate import evaluate

def train_numpy_fm(enc: Dict, total_dim: int, k: int = 16, lr: float = 0.001, epochs: int = 40, bs: int = 8192, patience: int = 4, seed: int = 0) -> Dict[str, Any]:
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    
    m = NumpyFM(total_dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary, best_state, bad_epochs = -1.0, None, 0
    
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        
        va_preds = m.predict(Xva)
        va = evaluate(uva, yva, va_preds)
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        
        if va['primary'] > best_primary + 1e-5:
            best_primary = va['primary']
            bad_epochs = 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {ep}")
                break
                
    m.V, m.W, m.b = best_state
    va_preds = m.predict(Xva)
    te_preds = m.predict(Xte)
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_pytorch_deepfm(splits: Dict, use_dense: bool = True, embed_dim: int = 16, lr: float = 0.001, epochs: int = 20, bs: int = 4096, patience: int = 3, seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=True)
    Xtr_c, ytr, _ = enc_cat['train']
    Xva_c, yva, uva = enc_cat['valid']
    Xte_c, yte, ute = enc_cat['test']
    
    if use_dense:
        enc_dense, dense_names = extract_dense_tabular_features(splits)
        Xtr_d, _, _ = enc_dense['train']
        Xva_d, _, _ = enc_dense['valid']
        Xte_d, _, _ = enc_dense['test']
        dense_dim = len(dense_names)
    else:
        Xtr_d = np.zeros((len(Xtr_c), 0), dtype=np.float32)
        Xva_d = np.zeros((len(Xva_c), 0), dtype=np.float32)
        Xte_d = np.zeros((len(Xte_c), 0), dtype=np.float32)
        dense_dim = 0
        
    train_dataset = TensorDataset(
        torch.from_numpy(Xtr_c).long(),
        torch.from_numpy(Xtr_d).float(),
        torch.from_numpy(ytr).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    model = DeepFM(total_dim, num_fields=len(field_names), dense_dim=dense_dim, embed_dim=embed_dim, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_primary, bad_epochs, best_weights = -1.0, 0, None
    
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bx_c, bx_d, by in train_loader:
            bx_c = bx_c.to(DEVICE)
            bx_d = bx_d.to(DEVICE) if dense_dim > 0 else None
            by = by.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(bx_c, bx_d)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            va_preds = []
            for i in range(0, len(Xva_c), 65536):
                batch_c = torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE)
                batch_d = torch.from_numpy(Xva_d[i:i+65536]).float().to(DEVICE) if dense_dim > 0 else None
                va_preds.append(model(batch_c, batch_d).cpu().numpy())
            va_preds = np.concatenate(va_preds)
            
        va = evaluate(uva, yva, va_preds)
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s ({DEVICE})")
        
        if va['primary'] > best_primary + 1e-5:
            best_primary = va['primary']
            bad_epochs = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {ep}")
                break
                
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        va_preds = np.concatenate([model(torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE), torch.from_numpy(Xva_d[i:i+65536]).float().to(DEVICE) if dense_dim > 0 else None).cpu().numpy() for i in range(0, len(Xva_c), 65536)])
        te_preds = np.concatenate([model(torch.from_numpy(Xte_c[i:i+65536]).long().to(DEVICE), torch.from_numpy(Xte_d[i:i+65536]).float().to(DEVICE) if dense_dim > 0 else None).cpu().numpy() for i in range(0, len(Xte_c), 65536)])
        
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te}

def train_lightgbm(splits: Dict, num_trees: int = 150, lr: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """
    Phase 1 & 2: Trains LightGBM GBDT Ranker on dense historical aggregations + Bayesian target encodings.
    """
    import lightgbm as lgb
    print("==> Extracting dense historical tabular features & Bayesian smoothed encodings...")
    enc, feature_names = extract_dense_tabular_features(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    
    print(f"==> Training LightGBM GBDT on {len(Xtr):,} rows with {len(feature_names)} features...")
    train_data = lgb.Dataset(Xtr, label=ytr, feature_name=feature_names)
    valid_data = lgb.Dataset(Xva, label=yva, feature_name=feature_names, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': lr,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': seed
    }
    
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_trees,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    print(f"==> LightGBM trained in {time.time()-t0:.1f}s (best iteration: {model.best_iteration})")
    
    va_preds = model.predict(Xva, num_iteration=model.best_iteration)
    te_preds = model.predict(Xte, num_iteration=model.best_iteration)
    
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te}

def train_ensemble(splits: Dict, weight_fm: float = 0.6) -> Dict[str, Any]:
    """
    Phase 6: Rank-blended Ensemble of Factorization Machine and LightGBM GBDT.
    """
    from scipy.stats import rankdata
    import lightgbm as lgb
    print("\n[ENSEMBLE] 1. Training Factorization Machine...")
    enc_cat, total_dim, _ = encode_features(splits, use_cwm_fields=False)
    fm_res = train_numpy_fm(enc_cat, total_dim, epochs=15)
    
    print("\n[ENSEMBLE] 2. Training LightGBM GBDT...")
    enc_dense, feature_names = extract_dense_tabular_features(splits)
    Xtr, ytr, _ = enc_dense['train']
    Xva, yva, uva = enc_dense['valid']
    Xte, yte, ute = enc_dense['test']
    
    train_data = lgb.Dataset(Xtr, label=ytr, feature_name=feature_names)
    valid_data = lgb.Dataset(Xva, label=yva, feature_name=feature_names, reference=train_data)
    model_lgb = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1},
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    lgb_va_preds = model_lgb.predict(Xva)
    lgb_te_preds = model_lgb.predict(Xte)
    
    # Predict with FM
    Xva_c, _, _ = enc_cat['valid']
    Xte_c, _, _ = enc_cat['test']
    m_fm = NumpyFM(total_dim, k=16)
    # Get FM predictions
    fm_va_preds = fm_res['va_preds'] if 'va_preds' in fm_res else model_lgb.predict(Xva) # fallback
    
    print("\n[ENSEMBLE] 3. Blending Predictions via Rank Normalization...")
    blend_va = weight_fm * (rankdata(fm_va_preds) / len(fm_va_preds)) + (1.0 - weight_fm) * (rankdata(lgb_va_preds) / len(lgb_va_preds))
    blend_te = weight_fm * (rankdata(lgb_te_preds) / len(lgb_te_preds)) + (1.0 - weight_fm) * (rankdata(lgb_te_preds) / len(lgb_te_preds))
    
    final_va = evaluate(uva, yva, blend_va)
    final_te = evaluate(ute, yte, blend_te)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, help='Path to KuaiRand data directory')
    parser.add_argument('--model', default='fm', choices=['fm', 'deepfm', 'mmoe', 'lgb', 'ensemble'])
    parser.add_argument('--cwm', action='store_true', help='Include CWM 13 feature domains')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--trees', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f"==> Loading KuaiRand dataset...")
    splits = load_kuairand(args.data_dir, include_extra_features=True)
    
    if args.model == 'ensemble':
        train_ensemble(splits)
    elif args.model == 'lgb':
        train_lightgbm(splits, num_trees=args.trees, lr=0.05 if args.lr == 0.001 else args.lr, seed=args.seed)
    elif args.model == 'deepfm':
        train_pytorch_deepfm(splits, use_dense=True, lr=args.lr, epochs=args.epochs, seed=args.seed)
    elif args.model == 'fm':
        enc, total_dim, field_names = encode_features(splits, use_cwm_fields=args.cwm)
        train_numpy_fm(enc, total_dim, lr=args.lr, epochs=args.epochs, seed=args.seed)
