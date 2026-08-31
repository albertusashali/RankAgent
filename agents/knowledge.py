"""Established methods the Researcher may draw on, with their citations.

The hackathon expects the agent to draw on published work rather than tweak
knobs, and the run log is where that has to be visible. But a citation the model
*types* is a citation the model can invent, and a fabricated reference in a
submitted log is worse than none.

So the model never writes one. It chooses a ``method_id`` from this table — a
closed set, validated on parse — and the orchestrator substitutes the citation.
The only two possible outcomes are a real reference from here, or ``novel``,
which is logged as unreferenced.

Each entry says why the method suits *within-user* ranking specifically, because
that is the reasoning the Researcher has to do and the thing a generic summary
of the paper would not supply.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple


class Method(NamedTuple):
    name: str
    citation: str
    dimension: str
    why_here: str


KB: Dict[str, Method] = {
    "approx_ndcg": Method(
        "ApproxNDCG", "Qin, Liu & Li (2010), Information Retrieval 13(4)", "loss",
        "Replaces the non-differentiable rank in nDCG with a sigmoid of score "
        "differences, so nDCG@5 itself becomes the training objective instead of "
        "something that merely correlates with it."),
    "neural_ndcg": Method(
        "NeuralNDCG", "Pobrotyn & Bialobrzeski (2021), arXiv:2102.07831", "loss",
        "Differentiable nDCG via NeuralSort; a smoother surrogate than "
        "ApproxNDCG and better behaved on short lists, which matters when "
        "evaluation users average 5.6 impressions."),
    "lambdaloss": Method(
        "LambdaLoss", "Wang, Li, Golbandi, Bendersky & Najork (2018), CIKM", "loss",
        "Casts LambdaRank as a probabilistic loss with an nDCG-aligned weighting, "
        "giving the metric-driven gradients of LambdaMART inside a neural model."),
    "listmle": Method(
        "ListMLE", "Xia, Liu, Wang, Zhang & Li (2008), ICML", "loss",
        "Full-permutation likelihood over a user's list. Like the existing "
        "listwise softmax it is invariant to per-user score offsets, which the "
        "metrics also ignore."),
    "focal": Method(
        "Focal Loss", "Lin, Goyal, Girshick, He & Dollar (2017), ICCV", "loss",
        "long_view positives are sparse, so easy negatives dominate the gradient; "
        "the modulating factor down-weights them."),
    "dcn_v2": Method(
        "DCN-v2", "Wang, Shivanna, Cheng, Jain, Lin, Hong & Chi (2021), WWW", "architecture",
        "Explicit bounded-degree feature crosses. The baseline FM captures only "
        "second-order interactions, and user x item crosses are what can reorder "
        "a single user's list."),
    "autoint": Method(
        "AutoInt", "Song, Shi, Xiao, Duan, Xu, Zhang & Tang (2019), CIKM", "architecture",
        "Multi-head self-attention over field embeddings learns which feature "
        "interactions matter rather than enumerating them."),
    "fibinet": Method(
        "FiBiNET", "Huang, Zhang & Zhang (2019), RecSys", "architecture",
        "SENet-style feature importance plus bilinear interaction, cheap to add "
        "on top of an existing embedding table."),
    "ple": Method(
        "PLE", "Tang, Liu, Zhao, Gao, Zhang & others (2020), RecSys", "multi_task",
        "Separates shared from task-specific experts, addressing the seesaw "
        "effect where MMoE's fully shared trunk lets auxiliary tasks fight the "
        "scored one."),
    "esmm": Method(
        "ESMM", "Ma, Zhao, Hao, Mao, Zhu, Gao & others (2018), SIGIR", "multi_task",
        "Models the impression -> click -> conversion chain over the entire "
        "space, correcting the sample-selection bias of training only on clicks."),
    "din": Method(
        "DIN", "Zhou, Zhu, Song, Fan, Zhu, Ma & others (2018), KDD", "sequence",
        "Target attention over the user's history, so the same history is "
        "weighted differently per candidate — a per-impression signal rather "
        "than a user-constant one."),
    "pda": Method(
        "PDA popularity debiasing", "Zhang, Feng, He, Wei, Song, Ling & Zhang (2021), SIGIR",
        "features",
        "Deconfounds item popularity from relevance; useful because the logging "
        "policy already favoured popular items."),
}


def entries_for(dimensions: List[str], limit: int = 6) -> List[str]:
    """Render the shortlist for a directive's dimensions."""
    wanted = [k for k, m in KB.items() if m.dimension in dimensions]
    rest = [k for k in KB if k not in wanted]
    out = []
    for key in (wanted + rest)[:limit]:
        m = KB[key]
        out.append(f'- id "{key}" — {m.name} ({m.citation})\n    {m.why_here}')
    return out


def citation_for(method_id: str) -> str:
    """The citation to record, or an explicit marker for unreferenced work."""
    m = KB.get(method_id)
    return m.citation if m else "(no published reference — proposed by the agent)"
