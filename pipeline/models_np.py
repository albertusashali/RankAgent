"""The official baseline model, in numpy only.

Deliberately kept in its own module, free of any torch or LightGBM import.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
PyTorch and LightGBM each vendor their own OpenMP runtime. On macOS, loading
both into one process segfaults inside whichever thread pool starts second —
in either import order. The pipeline therefore never imports both: each trainer
pulls in only the framework it needs, at call time. This module holds the piece
that needs neither, so reproducing the baseline depends on nothing but numpy —
exactly like the starter kit it mirrors.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


class NumpyFM:
    """Factorization Machine with hand-rolled Adam, matching the starter kit.

    Kept byte-for-byte equivalent to ``kuairand-starter-kit/baseline.py`` so that
    ``--model fm`` reproduces the organizer's published 0.6016 validation score.
    Do not "improve" this class; build new models alongside it.
    """

    def __init__(self, num_features: int, k: int = 16, lr: float = 0.001,
                 l2: float = 1e-6, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (num_features, k)).astype(np.float32)
        self.W = np.zeros(num_features, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])
