"""Torch model architectures for KuaiRand-Pure within-user ranking.

IMPORTING THIS MODULE PULLS IN TORCH. Do not import it from a code path that also
uses LightGBM: the two vendor conflicting OpenMP runtimes and segfault when both
are loaded (see pipeline/models_np.py). The numpy baseline lives in
``pipeline.models_np`` precisely so it can be used without torch.

  ``TorchFM``   same architecture under autograd, so the *loss* can be swapped
  ``DeepFM``    linear + 2nd-order FM + MLP
  ``MMoE``      multi-gate mixture-of-experts over auxiliary feedback signals
  ``PLE``       progressive layered extraction with customized gate control
  ``DIN``       target attention over the user's watch history
  ``BST``       behavior sequence transformer with self-attention over watch history
  ``DCNv2``     deep & cross network v2 with explicit polynomial feature crosses

All torch models emit **raw logits**. Squashing to a probability inside the model
forces a pointwise objective and is numerically worse; ranking only cares about
order, and the loss functions below take logits.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-exported so callers that want the baseline can reach it from here, though
# pipeline.train imports it from pipeline.models_np directly to stay torch-free.
from pipeline.models_np import NumpyFM, sigmoid  # noqa: F401

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================================
# Ranking losses
# =========================================================================

def _segment_logsumexp(scores: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Numerically stable log-sum-exp of ``scores`` within each group id."""
    maxes = torch.full((n_groups,), float('-inf'), device=scores.device, dtype=scores.dtype)
    maxes = maxes.scatter_reduce(0, group, scores, reduce='amax', include_self=True)
    shifted = torch.exp(scores - maxes[group])
    sums = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
    sums = sums.index_add(0, group, shifted)
    return maxes + torch.log(sums + 1e-12)


def pointwise_bce(logits: torch.Tensor, labels: torch.Tensor,
                  group: Optional[torch.Tensor] = None,
                  n_groups: int = 0) -> torch.Tensor:
    """Per-impression binary cross-entropy. The original objective."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def listwise_softmax(logits: torch.Tensor, labels: torch.Tensor,
                     group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Within-user softmax cross-entropy (ListNet-style).

    The metrics are *within-user* ranking metrics, so the natural objective is a
    likelihood over each user's own impression list rather than over the whole
    corpus. Maximising ``P(positive | this user's impressions)`` optimises exactly
    the quantity GAUC and nDCG@5 measure, and — unlike pointwise BCE — it is
    invariant to any per-user score offset, which the metrics also ignore.
    """
    lse = _segment_logsumexp(logits, group, n_groups)
    log_prob = logits - lse[group]
    pos = labels > 0.5
    if not bool(pos.any()):
        return logits.sum() * 0.0
    return -(log_prob[pos]).mean()


def bpr_pairwise(logits: torch.Tensor, labels: torch.Tensor,
                 group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Bayesian Personalised Ranking over within-user positive/negative pairs.

    For each group, pair every positive against a randomly chosen negative from
    the same user and push their score margin apart. Directly optimises the
    pairwise ordering that AUC counts.
    """
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    if not (bool(pos_mask.any()) and bool(neg_mask.any())):
        return logits.sum() * 0.0

    # Sample one negative per group; groups without a negative are skipped.
    neg_idx = torch.full((n_groups,), -1, device=logits.device, dtype=torch.long)
    neg_positions = torch.nonzero(neg_mask, as_tuple=True)[0]
    perm = neg_positions[torch.randperm(neg_positions.numel(), device=logits.device)]
    neg_idx[group[perm]] = perm

    pos_positions = torch.nonzero(pos_mask, as_tuple=True)[0]
    partner = neg_idx[group[pos_positions]]
    keep = partner >= 0
    if not bool(keep.any()):
        return logits.sum() * 0.0
    margin = logits[pos_positions[keep]] - logits[partner[keep]]
    return -F.logsigmoid(margin).mean()


LOSSES = {
    'pointwise': pointwise_bce,
    'listwise': listwise_softmax,
    'bpr': bpr_pairwise,
}


# =========================================================================
# Torch models
# =========================================================================

class TorchFM(nn.Module):
    """Factorization Machine under autograd — architecturally the baseline.

    Exists so the loss function can be swapped while holding the model fixed.
    Any gain over ``NumpyFM`` is attributable to the objective, not the capacity.
    """

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        vx = self.factors(x_cat)
        inter = 0.5 * ((vx.sum(1) ** 2) - (vx ** 2).sum(1)).sum(1)
        return self.linear(x_cat).sum((1, 2)) + self.bias + inter


class DeepFM(nn.Module):
    """Linear + 2nd-order FM + deep MLP over concatenated field embeddings and dense features."""

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 dense_dim: int = 0, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        self.dense_dim = dense_dim
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        dim = num_fields * embed_dim + dense_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        vx = self.factors(x_cat)
        inter = 0.5 * ((vx.sum(1) ** 2) - (vx ** 2).sum(1)).sum(1)
        flat_vx = vx.flatten(1)
        if self.dense_dim > 0 and x_dense is not None:
            flat_vx = torch.cat([flat_vx, self.dense_bn(x_dense)], dim=1)
        deep = self.mlp(flat_vx).squeeze(1)
        return self.linear(x_cat).sum((1, 2)) + self.bias + inter + deep


class DCNv2(nn.Module):
    """Deep & Cross Network v2 (DCN-v2) with continuous dense causal feature fusion."""

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 dense_dim: int = 0, num_cross_layers: int = 3,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.15):
        super().__init__()
        self.dense_dim = dense_dim
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        dim = num_fields * embed_dim + dense_dim
        self.dim = dim
        self.cross_layers = nn.ModuleList([
            nn.Linear(dim, dim, bias=True) for _ in range(num_cross_layers)
        ])

        mlp_layers: List[nn.Module] = []
        in_dim = dim
        for h in hidden_dims:
            mlp_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)

        self.combination = nn.Linear(dim + in_dim, 1)

    def forward(self, x_cat: torch.Tensor, x_dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_0 = self.factors(x_cat).flatten(1)  # (B, cat_dim)
        if self.dense_dim > 0 and x_dense is not None:
            x_0 = torch.cat([x_0, self.dense_bn(x_dense)], dim=1)
        x_l = x_0
        for layer in self.cross_layers:
            # x_{l+1} = x_0 * W_l(x_l) + x_l
            x_l = x_0 * layer(x_l) + x_l

        x_deep = self.mlp(x_0)
        deep_cross = torch.cat([x_l, x_deep], dim=1)
        out = self.combination(deep_cross).squeeze(1)
        linear_term = self.linear(x_cat).sum((1, 2)) + self.bias
        return out + linear_term


class MMoE(nn.Module):
    """Multi-gate mixture-of-experts with continuous dense feature fusion."""

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 dense_dim: int = 0, num_experts: int = 4, expert_dim: int = 64,
                 num_tasks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.dense_dim = dense_dim
        self.factors = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.factors.weight, std=0.01)

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        dim = num_fields * embed_dim + dense_dim

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_dim), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(expert_dim, expert_dim))
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([nn.Linear(dim, num_experts) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, x_cat: torch.Tensor, x_dense: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        rep = self.factors(x_cat).flatten(1)
        if self.dense_dim > 0 and x_dense is not None:
            rep = torch.cat([rep, self.dense_bn(x_dense)], dim=1)
        expert_out = torch.stack([e(rep) for e in self.experts], dim=1)   # (B, E, D)
        outs = []
        for i in range(self.num_tasks):
            w = torch.softmax(self.gates[i](rep), dim=-1).unsqueeze(1)     # (B, 1, E)
            outs.append(self.towers[i](torch.bmm(w, expert_out).squeeze(1)).squeeze(1))
        return tuple(outs)


class PLE(nn.Module):
    """Progressive Layered Extraction (PLE / CGC) with continuous dense causal feature fusion."""

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 dense_dim: int = 0, num_private_experts: int = 1, num_shared_experts: int = 2,
                 expert_dim: int = 64, num_tasks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.dense_dim = dense_dim
        self.factors = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.factors.weight, std=0.01)

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        dim = num_fields * embed_dim + dense_dim

        self.private_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(dim, expert_dim), nn.ReLU(),
                              nn.Dropout(dropout), nn.Linear(expert_dim, expert_dim))
                for _ in range(num_private_experts)
            ])
            for _ in range(num_tasks)
        ])

        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_dim), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(expert_dim, expert_dim))
            for _ in range(num_shared_experts)
        ])

        total_task_experts = num_private_experts + num_shared_experts
        self.gates = nn.ModuleList([
            nn.Linear(dim, total_task_experts) for _ in range(num_tasks)
        ])

        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, x_cat: torch.Tensor, x_dense: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        rep = self.factors(x_cat).flatten(1)  # (B, dim)
        if self.dense_dim > 0 and x_dense is not None:
            rep = torch.cat([rep, self.dense_bn(x_dense)], dim=1)
        shared_out = [e(rep) for e in self.shared_experts]  # list of (B, expert_dim)

        outs = []
        for i in range(self.num_tasks):
            priv_out = [e(rep) for e in self.private_experts[i]]
            task_experts = torch.stack(priv_out + shared_out, dim=1)  # (B, E_priv + E_shared, expert_dim)
            gate_weights = torch.softmax(self.gates[i](rep), dim=-1).unsqueeze(1)  # (B, 1, E_priv + E_shared)
            gated_rep = torch.bmm(gate_weights, task_experts).squeeze(1)  # (B, expert_dim)
            outs.append(self.towers[i](gated_rep).squeeze(1))  # (B,)
        return tuple(outs)


class DIN(nn.Module):
    """Deep Interest Network with continuous dense causal feature fusion."""

    def __init__(self, num_features: int, num_fields: int, pad_id: int,
                 embed_dim: int = 16, dense_dim: int = 0, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.15, video_field_idx: int = 1):
        super().__init__()
        self.pad_id = pad_id
        self.video_field_idx = video_field_idx
        self.dense_dim = dense_dim
        # num_features already includes the reserved pad row.
        self.factors = nn.Embedding(num_features, embed_dim, padding_idx=pad_id)
        nn.init.normal_(self.factors.weight, std=0.01)
        with torch.no_grad():
            self.factors.weight[pad_id].zero_()

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        self.attention = nn.Sequential(
            nn.Linear(4 * embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        dim = num_fields * embed_dim + embed_dim + dense_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_hist: torch.Tensor,
                x_dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.factors(x_cat).flatten(1)
        if self.dense_dim > 0 and x_dense is not None:
            base = torch.cat([base, self.dense_bn(x_dense)], dim=1)
        cand = self.factors(x_cat[:, self.video_field_idx])          # (B, D)
        hist = self.factors(x_hist)                                   # (B, L, D)

        cand_exp = cand.unsqueeze(1).expand_as(hist)
        att_in = torch.cat([cand_exp, hist, cand_exp - hist, cand_exp * hist], dim=-1)
        weights = self.attention(att_in)                              # (B, L, 1)

        mask = (x_hist != self.pad_id).unsqueeze(-1)
        weights = weights.masked_fill(~mask, float('-inf'))
        empty = ~mask.any(dim=1, keepdim=True)
        weights = torch.where(empty.expand_as(weights), torch.zeros_like(weights), weights)
        weights = torch.softmax(weights, dim=1)
        weights = torch.where(empty.expand_as(weights), torch.zeros_like(weights), weights)

        interest = (weights * hist).sum(dim=1)                        # (B, D)
        return self.mlp(torch.cat([base, interest], dim=1)).squeeze(1)


class BST(nn.Module):
    """Behavior Sequence Transformer with continuous dense causal feature fusion."""

    def __init__(self, num_features: int, num_fields: int, pad_id: int,
                 embed_dim: int = 16, dense_dim: int = 0, num_heads: int = 2,
                 num_layers: int = 1, max_seq_len: int = 10,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1,
                 video_field_idx: int = 1):
        super().__init__()
        self.pad_id = pad_id
        self.video_field_idx = video_field_idx
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.max_seq_len = max_seq_len

        self.factors = nn.Embedding(num_features, embed_dim, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len + 1, embed_dim)
        nn.init.normal_(self.factors.weight, std=0.01)
        nn.init.normal_(self.pos_embedding.weight, std=0.01)
        with torch.no_grad():
            self.factors.weight[pad_id].zero_()

        if dense_dim > 0:
            self.dense_bn = nn.BatchNorm1d(dense_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        dim = num_fields * embed_dim + embed_dim + dense_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_hist: torch.Tensor,
                x_dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.factors(x_cat).flatten(1)  # (B, num_fields * D)
        if self.dense_dim > 0 and x_dense is not None:
            base = torch.cat([base, self.dense_bn(x_dense)], dim=1)
        cand = self.factors(x_cat[:, self.video_field_idx]).unsqueeze(1)  # (B, 1, D)
        hist = self.factors(x_hist)  # (B, L, D)

        seq = torch.cat([hist, cand], dim=1)  # (B, L+1, D)
        B, seq_len, _ = seq.shape

        # Positional encoding
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_embedding(positions)

        # Padding mask: True where key is padding
        cand_mask = torch.zeros((B, 1), dtype=torch.bool, device=seq.device)
        hist_mask = (x_hist == self.pad_id)
        pad_mask = torch.cat([hist_mask, cand_mask], dim=1)  # (B, L+1)

        trans_out = self.transformer(seq, src_key_padding_mask=pad_mask)
        # Extract candidate token output
        cand_trans = trans_out[:, -1, :]  # (B, D)

        return self.mlp(torch.cat([base, cand_trans], dim=1)).squeeze(1)

