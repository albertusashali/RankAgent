"""Official KuaiRand scorer — the single source of truth for every metric.

This module deliberately contains NO metric implementation of its own. It loads
``kuairand-starter-kit/evaluate.py`` verbatim and re-exports it, so the score we
select on is byte-identical to the score we are ranked on.

The previous version of this file reimplemented GAUC and nDCG@5 and diverged
from the official code in three ways that all bite on tied scores:

  * the official ``auc`` averages ranks across ties (Mann-Whitney with tie
    correction); the reimplementation used a plain ``argsort``,
  * the official nDCG sorts stably, so tied scores keep log order; ``np.argsort``
    is not stable by default,
  * predictions were cast to float32, manufacturing ties that did not exist.

Pinned conventions, all defined by the official file:

  task        : within-user ranking over logged impressions
  label       : ``long_view`` (native column, 0/1)
  metrics     : GAUC, nDCG@5; primary = mean of the two
  zero-positive users : nDCG counts as 0.0 and is included in the average;
                        GAUC counts only users with 0 < positives < impressions,
                        weighted by positive count
  nDCG gain   : 2^rel - 1
"""
import importlib.util
import os
from typing import Dict, List, Sequence, Union

_STARTER_KIT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kuairand-starter-kit",
)
_OFFICIAL_PATH = os.path.join(_STARTER_KIT, "evaluate.py")


def _load_official():
    """Import the starter kit's evaluate.py by path (its directory is not a package)."""
    if not os.path.exists(_OFFICIAL_PATH):
        raise FileNotFoundError(
            f"Official evaluator not found at {_OFFICIAL_PATH}. "
            "The starter kit must sit alongside pipeline/ — do not vendor a copy."
        )
    spec = importlib.util.spec_from_file_location("kuairand_official_evaluate", _OFFICIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_official = _load_official()

# Re-exported verbatim so callers can reach the primitives if they need them.
auc = _official.auc
ndcg_at_k = _official.ndcg_at_k

Numeric = Union[Sequence[float], "list"]


def evaluate(users: Sequence, labels: Numeric, preds: Numeric, k: int = 5) -> Dict[str, float]:
    """Score a set of predictions with the official implementation.

    Returns ``{'GAUC', 'nDCG@k', 'primary', 'users', 'rows'}``.

    ``preds`` is used for relative order only. Values are passed through as
    Python floats — no float32 narrowing, which would create artificial ties.
    """
    users = list(users)
    labels = [int(v) for v in labels]
    preds = [float(v) for v in preds]
    if not (len(users) == len(labels) == len(preds)):
        raise ValueError(
            f"length mismatch: users={len(users)} labels={len(labels)} preds={len(preds)}"
        )
    return _official.evaluate(users, labels, preds, k=k)


def format_eval_line(metrics: Dict[str, float], k: int = 5) -> str:
    """The line the sandbox parser looks for. Validation metrics only."""
    return (
        f"[EVAL] GAUC: {metrics['GAUC']:.4f} | "
        f"nDCG@{k}: {metrics[f'nDCG@{k}']:.4f} | "
        f"Primary: {metrics['primary']:.4f}"
    )


if __name__ == '__main__':
    # Harness self-check, per the starter kit: a scorer that cannot reproduce these
    # numbers is broken and must be fixed before any model result is believed.
    demo_users = ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3']
    demo_labels = [1, 0, 0, 1, 1, 0, 0]
    demo_preds = [0.9, 0.2, 0.1, 0.8, 0.7, 0.3, 0.1]
    print(format_eval_line(evaluate(demo_users, demo_labels, demo_preds)))
    print(f"loaded official implementation from {_OFFICIAL_PATH}")
