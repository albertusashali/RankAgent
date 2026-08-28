"""Torch model architectures for KuaiRand-Pure within-user ranking.

IMPORTING THIS MODULE PULLS IN TORCH. Do not import it from a code path that also
uses LightGBM: the two vendor conflicting OpenMP runtimes and segfault when both
are loaded (see pipeline/models_np.py). The numpy baseline lives in
``pipeline.models_np`` precisely so it can be used without torch.

  ``TorchFM``   same architecture under autograd, so the *loss* can be swapped
  ``DeepFM``    linear + 2nd-order FM + MLP
  ``MMoE``      multi-gate mixture-of-experts over auxiliary feedback signals
  ``DIN``       target attention over the user's watch history

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
    """Linear + 2nd-order FM + deep MLP over the concatenated field embeddings."""

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

        dim = num_fields * embed_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        vx = self.factors(x_cat)
        inter = 0.5 * ((vx.sum(1) ** 2) - (vx ** 2).sum(1)).sum(1)
        deep = self.mlp(vx.flatten(1)).squeeze(1)
        return self.linear(x_cat).sum((1, 2)) + self.bias + inter + deep


class MMoE(nn.Module):
    """Multi-gate mixture-of-experts over the auxiliary feedback signals.

    KuaiRand logs long_view, click, like, follow, comment and forward on every
    impression. Training the rare signals jointly regularises the shared
    embedding without diluting the scored head, which keeps its own gate and tower.
    Task 0 is always ``long_view`` — the only head used for ranking.
    """

    def __init__(self, num_features: int, num_fields: int, embed_dim: int = 16,
                 num_experts: int = 4, expert_dim: int = 64, num_tasks: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.factors = nn.Embedding(num_features, embed_dim)
        nn.init.normal_(self.factors.weight, std=0.01)
        dim = num_fields * embed_dim

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

    def forward(self, x_cat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        rep = self.factors(x_cat).flatten(1)
        expert_out = torch.stack([e(rep) for e in self.experts], dim=1)   # (B, E, D)
        outs = []
        for i in range(self.num_tasks):
            w = torch.softmax(self.gates[i](rep), dim=-1).unsqueeze(1)     # (B, 1, E)
            outs.append(self.towers[i](torch.bmm(w, expert_out).squeeze(1)).squeeze(1))
        return tuple(outs)


class DIN(nn.Module):
    """Deep Interest Network — target attention over the user's watch history.

    Candidate and history embeddings come from **one** table, indexed in the
    shared id space built by ``CategoricalEncoder``. The previous version drew the
    candidate from the categorical table and the history from a separate one, so
    ``cand - hist`` and ``cand * hist`` compared vectors in unrelated spaces and
    the attention unit could not learn anything meaningful.
    """

    def __init__(self, num_features: int, num_fields: int, pad_id: int,
                 embed_dim: int = 16, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.15, video_field_idx: int = 1):
        super().__init__()
        self.pad_id = pad_id
        self.video_field_idx = video_field_idx
        # num_features already includes the reserved pad row.
        self.factors = nn.Embedding(num_features, embed_dim, padding_idx=pad_id)
        nn.init.normal_(self.factors.weight, std=0.01)
        with torch.no_grad():
            self.factors.weight[pad_id].zero_()

        self.attention = nn.Sequential(
            nn.Linear(4 * embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        dim = num_fields * embed_dim + embed_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        base = self.factors(x_cat).flatten(1)
        cand = self.factors(x_cat[:, self.video_field_idx])          # (B, D)
        hist = self.factors(x_hist)                                   # (B, L, D)

        cand_exp = cand.unsqueeze(1).expand_as(hist)
        att_in = torch.cat([cand_exp, hist, cand_exp - hist, cand_exp * hist], dim=-1)
        weights = self.attention(att_in)                              # (B, L, 1)

        mask = (x_hist != self.pad_id).unsqueeze(-1)
        weights = weights.masked_fill(~mask, float('-inf'))
        # A user with no history yet would give an all -inf row; softmax would be
        # NaN, so fall back to a zero interest vector for those rows.
        empty = ~mask.any(dim=1, keepdim=True)
        weights = torch.where(empty.expand_as(weights), torch.zeros_like(weights), weights)
        weights = torch.softmax(weights, dim=1)
        weights = torch.where(empty.expand_as(weights), torch.zeros_like(weights), weights)

        interest = (weights * hist).sum(dim=1)                        # (B, D)
        return self.mlp(torch.cat([base, interest], dim=1)).squeeze(1)
