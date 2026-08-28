"""
Training and Optimization harness for KuaiRand recommendation models.
Supports Factorization Machines, DeepFM, MMoE, DIN (Deep Interest Network), and LightGBM GBDT Ranker.
Includes automatic model checkpoint persistence in checkpoints/.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any

from pipeline.data import load_kuairand
from pipeline.features import encode_features, extract_dense_tabular_features, extract_sequential_features
from pipeline.models import NumpyFM, DeepFM, MMoE, DIN, DEVICE
from pipeline.evaluate import evaluate
from pipeline.submit import generate_submission

CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

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
    
    np.savez_compressed(os.path.join(CHECKPOINTS_DIR, "best_fm.npz"), V=m.V, W=m.W, b=m.b)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_pytorch_deepfm(splits: Dict, use_dense: bool = True, embed_dim: int = 16, lr: float = 0.001, epochs: int = 20, bs: int = 4096, patience: int = 3, seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=True)
    Xtr_c, ytr, _ = enc_cat['train']
    Xva_c, yva, uva = enc_cat['valid']
    Xte_c, yte, ute = enc_cat['test']
    
    train_dataset = TensorDataset(
        torch.from_numpy(Xtr_c).long(),
        torch.from_numpy(ytr).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    model = DeepFM(total_dim, num_fields=len(field_names), embed_dim=embed_dim, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_primary, bad_epochs, best_weights = -1.0, 0, None
    
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bx_c, by in train_loader:
            bx_c = bx_c.to(DEVICE)
            by = by.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(bx_c)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            va_preds = []
            for i in range(0, len(Xva_c), 65536):
                batch_c = torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE)
                va_preds.append(model(batch_c).cpu().numpy())
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
    torch.save(best_weights, os.path.join(CHECKPOINTS_DIR, "best_deepfm.pt"))
    
    model.eval()
    with torch.no_grad():
        va_preds = np.concatenate([model(torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE)).cpu().numpy() for i in range(0, len(Xva_c), 65536)])
        te_preds = np.concatenate([model(torch.from_numpy(Xte_c[i:i+65536]).long().to(DEVICE)).cpu().numpy() for i in range(0, len(Xte_c), 65536)])
        
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_pytorch_din(splits: Dict, embed_dim: int = 16, lr: float = 0.001, epochs: int = 15, bs: int = 4096, patience: int = 3, seed: int = 0) -> Dict[str, Any]:
    """
    Phase 4: Deep Interest Network (DIN) with Sequential Target-Attention Pooling.
    """
    torch.manual_seed(seed)
    enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=False)
    seqs, num_vids = extract_sequential_features(splits, max_seq_len=5)
    
    Xtr_c, ytr, _ = enc_cat['train']
    Xtr_h = seqs['train']
    
    Xva_c, yva, uva = enc_cat['valid']
    Xva_h = seqs['valid']
    
    Xte_c, yte, ute = enc_cat['test']
    Xte_h = seqs['test']
    
    train_dataset = TensorDataset(
        torch.from_numpy(Xtr_c).long(),
        torch.from_numpy(Xtr_h).long(),
        torch.from_numpy(ytr).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    model = DIN(total_dim, num_fields=len(field_names), num_vids=num_vids, embed_dim=embed_dim, dropout=0.15).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_primary, bad_epochs, best_weights = -1.0, 0, None
    print(f"==> Training DIN (Deep Interest Network) with Sequential Attention on device: {DEVICE}...")
    
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bx_c, bx_h, by in train_loader:
            bx_c, bx_h, by = bx_c.to(DEVICE), bx_h.to(DEVICE), by.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(bx_c, bx_h)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            va_preds = []
            for i in range(0, len(Xva_c), 65536):
                batch_c = torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE)
                batch_h = torch.from_numpy(Xva_h[i:i+65536]).long().to(DEVICE)
                va_preds.append(model(batch_c, batch_h).cpu().numpy())
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
    torch.save(best_weights, os.path.join(CHECKPOINTS_DIR, "best_din.pt"))
    
    model.eval()
    with torch.no_grad():
        va_preds = np.concatenate([model(torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE), torch.from_numpy(Xva_h[i:i+65536]).long().to(DEVICE)).cpu().numpy() for i in range(0, len(Xva_c), 65536)])
        te_preds = np.concatenate([model(torch.from_numpy(Xte_c[i:i+65536]).long().to(DEVICE), torch.from_numpy(Xte_h[i:i+65536]).long().to(DEVICE)).cpu().numpy() for i in range(0, len(Xte_c), 65536)])
        
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_pytorch_mmoe(splits: Dict, embed_dim: int = 16, num_experts: int = 4, expert_dim: int = 64, lr: float = 0.001, epochs: int = 20, bs: int = 4096, patience: int = 3, seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    enc_cat, total_dim, field_names = encode_features(splits, use_cwm_fields=False)
    
    tr_rows = splits['train']
    Xtr_c = enc_cat['train'][0]
    
    ytr_long = np.array([r['label'] for r in tr_rows], dtype=np.float32)
    ytr_click = np.array([r.get('click', 0) for r in tr_rows], dtype=np.float32)
    ytr_like = np.array([r.get('like', 0) for r in tr_rows], dtype=np.float32)
    
    Xva_c, yva_long, uva = enc_cat['valid']
    Xte_c, yte_long, ute = enc_cat['test']
    
    train_dataset = TensorDataset(
        torch.from_numpy(Xtr_c).long(),
        torch.from_numpy(ytr_long).float(),
        torch.from_numpy(ytr_click).float(),
        torch.from_numpy(ytr_like).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    model = MMoE(total_dim, num_fields=len(field_names), embed_dim=embed_dim, num_experts=num_experts, expert_dim=expert_dim, num_tasks=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce_loss = nn.BCELoss()
    
    best_primary, bad_epochs, best_weights = -1.0, 0, None
    print(f"==> Training Multi-Task MMoE with {len(field_names)} fields on device: {DEVICE}...")
    
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bx_c, by_long, by_click, by_like in train_loader:
            bx_c = bx_c.to(DEVICE)
            by_long, by_click, by_like = by_long.to(DEVICE), by_click.to(DEVICE), by_like.to(DEVICE)
            
            optimizer.zero_grad()
            out_long, out_click, out_like = model(bx_c)
            
            l_long = bce_loss(out_long, by_long)
            l_click = bce_loss(out_click, by_click)
            l_like = bce_loss(out_like, by_like)
            
            total_loss = l_long + 0.3 * l_click + 0.3 * l_like
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())
            
        model.eval()
        with torch.no_grad():
            va_preds = []
            for i in range(0, len(Xva_c), 65536):
                batch_c = torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE)
                out_long, _, _ = model(batch_c)
                va_preds.append(out_long.cpu().numpy())
            va_preds = np.concatenate(va_preds)
            
        va = evaluate(uva, yva_long, va_preds)
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
    torch.save(best_weights, os.path.join(CHECKPOINTS_DIR, "best_mmoe.pt"))
    
    model.eval()
    with torch.no_grad():
        va_preds = np.concatenate([model(torch.from_numpy(Xva_c[i:i+65536]).long().to(DEVICE))[0].cpu().numpy() for i in range(0, len(Xva_c), 65536)])
        te_preds = np.concatenate([model(torch.from_numpy(Xte_c[i:i+65536]).long().to(DEVICE))[0].cpu().numpy() for i in range(0, len(Xte_c), 65536)])
        
    final_va = evaluate(uva, yva_long, va_preds)
    final_te = evaluate(ute, yte_long, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_lightgbm(splits: Dict, num_trees: int = 150, lr: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    import lightgbm as lgb
    print("==> Extracting dense historical tabular features & dynamic user-author affinities...")
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
    model.save_model(os.path.join(CHECKPOINTS_DIR, "best_lgb.txt"))
    
    va_preds = model.predict(Xva, num_iteration=model.best_iteration)
    te_preds = model.predict(Xte, num_iteration=model.best_iteration)
    
    final_va = evaluate(uva, yva, va_preds)
    final_te = evaluate(ute, yte, te_preds)
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te, 'va_preds': va_preds, 'te_preds': te_preds}

def train_ensemble(splits: Dict, weight_neural: float = 0.65) -> Dict[str, Any]:
    """
    Phase 6: Rank-blended Ensemble of Multi-Task MMoE and LightGBM GBDT.
    """
    from scipy.stats import rankdata
    import lightgbm as lgb
    print("\n[ENSEMBLE] 1. Training Multi-Task MMoE...")
    mmoe_res = train_pytorch_mmoe(splits, embed_dim=32, num_experts=6, epochs=10)
    
    print("\n[ENSEMBLE] 2. Training LightGBM GBDT...")
    lgb_res = train_lightgbm(splits, num_trees=150, lr=0.05)
    
    uva = splits['valid']
    ute = splits['test']
    yva = [r['label'] for r in uva]
    yte = [r['label'] for r in ute]
    users_va = [r['user_id'] for r in uva]
    users_te = [r['user_id'] for r in ute]
    
    mmoe_va_preds = mmoe_res['va_preds']
    mmoe_te_preds = mmoe_res['te_preds']
    lgb_va_preds = lgb_res['va_preds']
    lgb_te_preds = lgb_res['te_preds']
    
    print(f"\n[ENSEMBLE] 3. Blending Predictions via Rank Normalization (Weight Neural={weight_neural:.2f})...")
    blend_va = weight_neural * (rankdata(mmoe_va_preds) / len(mmoe_va_preds)) + (1.0 - weight_neural) * (rankdata(lgb_va_preds) / len(lgb_va_preds))
    blend_te = weight_neural * (rankdata(mmoe_te_preds) / len(mmoe_te_preds)) + (1.0 - weight_neural) * (rankdata(lgb_te_preds) / len(lgb_te_preds))
    
    final_va = evaluate(users_va, yva, blend_va)
    final_te = evaluate(users_te, yte, blend_te)
    
    generate_submission(ute, blend_te, "submission.csv")
    print(f"[EVAL] GAUC: {final_va['GAUC']:.4f} | nDCG@5: {final_va['nDCG@5']:.4f} | Primary: {final_va['primary']:.4f}")
    return {'valid': final_va, 'test': final_te}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, help='Path to KuaiRand data directory')
    parser.add_argument('--model', default='fm', choices=['fm', 'deepfm', 'mmoe', 'din', 'lgb', 'ensemble'])
    parser.add_argument('--cwm', action='store_true', help='Include CWM 13 feature domains')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--trees', type=int, default=150)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--experts', type=int, default=4)
    parser.add_argument('--weight_ensemble', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f"==> Loading KuaiRand dataset...")
    splits = load_kuairand(args.data_dir, include_extra_features=True)
    
    if args.model == 'din':
        train_pytorch_din(splits, embed_dim=args.embed_dim, lr=args.lr, epochs=args.epochs, seed=args.seed)
    elif args.model == 'mmoe':
        train_pytorch_mmoe(splits, embed_dim=args.embed_dim, num_experts=args.experts, lr=args.lr, epochs=args.epochs, seed=args.seed)
    elif args.model == 'ensemble':
        train_ensemble(splits, weight_neural=args.weight_ensemble)
    elif args.model == 'lgb':
        train_lightgbm(splits, num_trees=args.trees, lr=0.05 if args.lr == 0.001 else args.lr, seed=args.seed)
    elif args.model == 'deepfm':
        train_pytorch_deepfm(splits, use_dense=True, embed_dim=args.embed_dim, lr=args.lr, epochs=args.epochs, seed=args.seed)
    elif args.model == 'fm':
        enc, total_dim, field_names = encode_features(splits, use_cwm_fields=args.cwm)
        train_numpy_fm(enc, total_dim, lr=args.lr, epochs=args.epochs, seed=args.seed)
