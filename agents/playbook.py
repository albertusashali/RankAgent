"""The deterministic research playbook.

Every agent has an LLM path and a fallback path. These entries are the fallback —
and they are also injected into the LLM prompts as worked examples of what a
grounded hypothesis looks like, which is why each carries a *mechanism* rather
than just a name.

Ordering reflects measured evidence, not intuition:

  * The organizers flagged loss/metric alignment as the most promising untested
    direction, and a controlled ablation on this repo confirmed it — holding the
    FM architecture fixed, listwise scored 0.6024 against pointwise 0.6011.
  * They also measured that static side features and larger embeddings do nothing,
    so ``capacity`` sits last and ``features`` is framed around causal statistics
    and crosses rather than adding raw fields.
"""
from typing import Dict, List

PLAYBOOK: List[Dict[str, str]] = [
    {
        "dimension": "loss",
        "stage": "Loss Function",
        "hypothesis": "Replace pointwise BCE with a within-user listwise softmax.",
        "mechanism": "GAUC and nDCG@5 rank inside one user's impression list, so a "
                     "per-impression likelihood optimises the wrong quantity. A listwise "
                     "objective is invariant to per-user score offsets, exactly as the "
                     "metrics are.",
        "args": "--model fm_torch --loss listwise --epochs 15",
    },
    {
        "dimension": "multi_task",
        "stage": "Multi-Task Learning",
        "hypothesis": "Train long_view jointly with click, like and forward via MMoE.",
        "mechanism": "The auxiliary signals are logged on every impression, so they can "
                     "regularise the shared embedding without diluting the scored head, "
                     "which keeps its own gate and tower.",
        "args": "--model mmoe --loss listwise --experts 4 --epochs 12",
    },
    {
        "dimension": "architecture",
        "stage": "Tree-based Ranker",
        "hypothesis": "Fit LightGBM with lambdarank truncated at 5 over causal statistics.",
        "mechanism": "A GBDT exploits dense count features that an embedding model handles "
                     "poorly, producing an ensemble member with decorrelated errors even if "
                     "it is weaker standalone.",
        "args": "--model lgb --objective lambdarank --trees 400",
    },
    {
        "dimension": "sequence",
        "stage": "Sequential Modelling",
        "hypothesis": "Add target attention over the user's last 10 impressions (DIN).",
        "mechanism": "Nothing in the baseline uses behaviour order. Attention conditioned on "
                     "the candidate video should separate durable taste from incidental "
                     "exposure.",
        "args": "--model din --loss listwise --max_seq_len 10 --epochs 10",
    },
    {
        "dimension": "loss",
        "stage": "Loss Function",
        "hypothesis": "Compare pairwise BPR against the listwise objective.",
        "mechanism": "BPR optimises the pairwise ordering AUC counts directly. If GAUC "
                     "dominates the primary score more than top-heavy nDCG does, it should win.",
        "args": "--model fm_torch --loss bpr --epochs 15",
    },
    {
        "dimension": "architecture",
        "stage": "Architecture",
        "hypothesis": "Add an MLP branch over the field embeddings (DeepFM), listwise.",
        "mechanism": "Tests whether implicit higher-order crosses add anything over the "
                     "second-order FM term at 1.14M rows.",
        "args": "--model deepfm --loss listwise --epochs 12",
    },
    {
        "dimension": "multi_task",
        "stage": "Multi-Task Learning",
        "hypothesis": "Lower the auxiliary task weight from 0.3 to 0.1.",
        "mechanism": "If the rare signals are dominating the shared trunk rather than "
                     "regularising it, a smaller weight should recover the scored head.",
        "args": "--model mmoe --loss listwise --aux_weight 0.1 --epochs 12",
    },
    {
        "dimension": "capacity",
        "stage": "Capacity",
        "hypothesis": "Widen the MMoE expert pool while holding embedding width fixed.",
        "mechanism": "Embedding capacity is measured not to help, but task-routing capacity "
                     "is a different axis and has not been tested.",
        "args": "--model mmoe --loss listwise --experts 8 --expert_dim 96 --epochs 12",
    },
    {
        "dimension": "optimisation",
        "stage": "Optimisation",
        "hypothesis": "Halve the learning rate on the listwise FM and extend the budget.",
        "mechanism": "The listwise objective converged in 8 epochs, suggesting the step size "
                     "overshoots a shallow optimum.",
        "args": "--model fm_torch --loss listwise --lr 0.0005 --epochs 25",
    },
    {
        "dimension": "sequence",
        "stage": "Sequential Modelling",
        "hypothesis": "Shorten the attention window to the last 5 impressions.",
        "mechanism": "Evaluation users average only 5.6 logged impressions, so a 10-step "
                     "history may be mostly padding at serve time.",
        "args": "--model din --loss listwise --max_seq_len 5 --epochs 10",
    },
    {
        "dimension": "features",
        "stage": "Feature Engineering",
        "hypothesis": "Add the CWM video-side fields on top of the causal statistics.",
        "mechanism": "Static fields alone are a measured dead end, but they have not been "
                     "tested in combination with a listwise objective.",
        "args": "--model fm_torch --loss listwise --cwm --epochs 15",
    },
    {
        "dimension": "capacity",
        "stage": "Tree-based Ranker",
        "hypothesis": "Deepen the GBDT with more leaves and a lower learning rate.",
        "mechanism": "The causal feature set is low-dimensional, so extra depth is affordable "
                     "and may capture interactions embedding models get for free.",
        "args": "--model lgb --num_leaves 127 --lr 0.03 --trees 600",
    },
]

#: Preference order when choosing an unexplored dimension, most promising first.
DIMENSION_PRIORITY = [
    "loss", "sequence", "multi_task", "features",
    "architecture", "optimisation", "capacity",
]


def entries_for(dimensions: List[str]) -> List[Dict[str, str]]:
    """Playbook entries whose dimension is in ``dimensions``, priority-ordered."""
    wanted = set(dimensions)
    hits = [e for e in PLAYBOOK if e["dimension"] in wanted]
    order = {d: i for i, d in enumerate(DIMENSION_PRIORITY)}
    return sorted(hits, key=lambda e: order.get(e["dimension"], 99))
