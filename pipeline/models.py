"""
Recommendation Model Architectures for KuaiRand.
Includes:
- Official NumPy Factorization Machine (FM) Baseline
- PyTorch DeepFM (1st + 2nd order + Deep MLP + Dense Historical Features)
- PyTorch MMoE (Multi-gate Mixture-of-Experts)
Supports dynamic CPU / NVIDIA CUDA GPU execution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))

# =========================================================================
# 1. Official NumPy Factorization Machine Baseline
# =========================================================================
class NumpyFM:
    def __init__(self, num_features: int, k: int = 16, lr: float = 0.001, seed: int = 0):
        self.k = k
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.V = (rng.standard_normal((num_features, k), dtype=np.float32) * 0.01).astype(np.float32)
        self.W = np.zeros(num_features, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lin = self.W[X].sum(axis=1) + self.b
        Vx = self.V[X]
        sum_v = Vx.sum(axis=1)
        sum_v2 = (Vx ** 2).sum(axis=1)
        inter = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_v2.sum(axis=1))
        return lin + inter, sum_v, Vx

    def step(self, X: np.ndarray, y: np.ndarray, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> float:
        self.t += 1
        z, sum_v, Vx = self.logits(X)
        p = sigmoid(z)
        g = (p - y) / len(y)
        
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        
        gVx = g[:, None, None] * (sum_v[:, None, :] - Vx)
        gV = np.zeros_like(self.V)
        np.add.at(gV, X, gVx)
        
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            m_hat = M / (1 - b1 ** self.t)
            v_hat = Vv / (1 - b2 ** self.t)
            P -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            
        self.b -= self.lr * g.sum()
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        return float(loss)

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        preds = []
        for i in range(0, len(X), bs):
            z, _, _ = self.logits(X[i:i + bs])
            preds.append(sigmoid(z))
        return np.concatenate(preds)


# =========================================================================
# 2. PyTorch DeepFM (Linear + 2nd-order FM + Dense Historical Stats + Deep MLP)
# =========================================================================
class DeepFM(nn.Module):
    def __init__(self, num_features: int, num_fields: int, dense_dim: int = 0, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        
        # 1st order linear weights & bias
        self.linear_embeddings = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 2nd order factor embeddings
        self.factor_embeddings = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.factor_embeddings.weight, std=0.01)
        nn.init.zeros_(self.linear_embeddings.weight)
        
        # Dense feature linear projection if present
        if dense_dim > 0:
            self.dense_proj = nn.Sequential(
                nn.Linear(dense_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
            input_dim = num_fields * embed_dim + 32
        else:
            self.dense_proj = None
            input_dim = num_fields * embed_dim
        
        # Deep MLP
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Linear part
        linear_part = self.linear_embeddings(x_cat).sum(dim=1) + self.bias
        
        # 2. FM 2nd order interaction
        vx = self.factor_embeddings(x_cat)  # (batch, num_fields, embed_dim)
        sum_vx = vx.sum(dim=1)
        sum_vx_sq = (vx ** 2).sum(dim=1)
        fm_part = 0.5 * (sum_vx ** 2 - sum_vx_sq).sum(dim=1, keepdim=True)
        
        # 3. Deep MLP part
        deep_emb = vx.view(vx.size(0), -1)
        if self.dense_proj is not None and x_dense is not None:
            dense_feat = self.dense_proj(x_dense)
            deep_input = torch.cat([deep_emb, dense_feat], dim=1)
        else:
            deep_input = deep_emb
            
        deep_part = self.mlp(deep_input)
        logits = linear_part + fm_part + deep_part
        return torch.sigmoid(logits).squeeze(1)


# =========================================================================
# 3. PyTorch MMoE (Multi-gate Mixture-of-Experts for Multi-Task Learning)
# =========================================================================
class MMoE(nn.Module):
    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16, num_experts: int = 4, expert_dim: int = 64, num_tasks: int = 3):
        super().__init__()
        self.factor_embeddings = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.factor_embeddings.weight, std=0.01)
        
        input_dim = num_fields * embed_dim
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim)
            ) for _ in range(num_experts)
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        vx = self.factor_embeddings(x).view(x.size(0), -1)
        expert_outputs = torch.stack([exp(vx) for exp in self.experts], dim=1)
        
        task_outputs = []
        for gate, tower in zip(self.gates, self.towers):
            gate_weights = gate(vx).unsqueeze(1)
            task_rep = torch.bmm(gate_weights, expert_outputs).squeeze(1)
            task_outputs.append(tower(task_rep).squeeze(1))
            
        return tuple(task_outputs)
