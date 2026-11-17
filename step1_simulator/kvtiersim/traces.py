"""Page-criticality trace sources.

A trace source yields, for every decode step t, the criticality score of every
KV page of one (sequence, layer) pair, and the number of pages that exist.
The selected set S_t = sink + recent window + top-K by score (Quest-style).

Two sources:
  * SyntheticTrace : parameterised AR(1) model (default; sweepable)
  * NpzTrace       : real scores dumped by tools/extract_quest_traces.py
"""
import numpy as np


class SyntheticTrace:
    def __init__(self, n_pages0, page_tokens, tp, rng=None):
        self.tp = tp
        self.page_tokens = page_tokens
        self.rng = rng or np.random.default_rng(tp.seed)
        cap = n_pages0 + tp.steps // page_tokens + 2
        self.cap = cap
        self.a = self.rng.standard_normal(cap)
        self.z = self.rng.standard_normal(cap)
        self.n = n_pages0
        self.t = 0
        self._k = np.sqrt(max(0.0, 1.0 - tp.rho ** 2))

    def step(self):
        """Advance one decode step; returns (scores[:n], n)."""
        tp = self.tp
        if self.t > 0 and self.t % self.page_tokens == 0:
            self.n += 1                      # a new page is completed
        self.z[: self.n] = tp.rho * self.z[: self.n] + self._k * \
            self.rng.standard_normal(self.n)
        self.t += 1
        s = tp.sigma_a * self.a[: self.n] + tp.sigma_z * self.z[: self.n]
        return s, self.n


class NpzTrace:
    """Replays real per-step page scores: npz with array 'scores' of shape
    [steps, max_pages] (NaN for pages not yet existing)."""

    def __init__(self, path, layer=0, head=0):
        d = np.load(path)
        sc = d["scores"]
        if sc.ndim == 4:      # [steps, layers, heads, pages]
            sc = sc[:, layer, head, :]
        self.sc = sc
        self.t = 0

    def step(self):
        row = self.sc[self.t]
        n = int(np.sum(~np.isnan(row)))
        self.t += 1
        return np.nan_to_num(row[:n], nan=-1e9), n


def select(scores, n, sp):
    """Quest-style selection: pinned sink+recent pages plus top-K by score."""
    K = sp.budget_pages()
    pinned = np.zeros(n, dtype=bool)
    pinned[: sp.sink_pages] = True
    pinned[max(0, n - sp.recent_pages):] = True
    k_free = max(0, K - int(pinned.sum()))
    sel = pinned.copy()
    if k_free > 0 and n > pinned.sum():
        s = np.where(pinned, -np.inf, scores)
        k_free = min(k_free, n - int(pinned.sum()))
        idx = np.argpartition(-s, k_free - 1)[:k_free]
        sel[idx] = True
    return sel
