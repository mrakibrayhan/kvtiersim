"""Trace-driven page-level simulation -> per-step traffic statistics.

Because all sequences in a batch are statistically identical, the page-level
behaviour of a policy depends only on (context length, HBM quota fraction q,
trace parameters). We simulate one representative (sequence, layer) stream
and return average pages/step, which the performance model scales by
batch size x layers.
"""
from dataclasses import dataclass
import numpy as np
from .traces import SyntheticTrace, select
from .policies import POLICIES


@dataclass
class PageStats:
    hit: float        # selected pages served by HBM
    miss: float       # selected pages served by far tier
    far_extra: float  # background far reads (prefetch)
    hbm_wr: float     # promotions written into HBM
    sel: float        # selected pages per step
    churn: float      # 1 - |S_t & S_{t-1}| / |S_t|

    @property
    def hit_rate(self):
        return self.hit / max(self.sel, 1e-9)


def simulate(policy, ctx_tokens, q, sp, tp, trace=None, **kw):
    n0 = ctx_tokens // sp.page_tokens
    rng = np.random.default_rng(tp.seed + 7)
    tr = trace or SyntheticTrace(n0, sp.page_tokens, tp,
                                 rng=np.random.default_rng(tp.seed))
    cap = n0 + tp.steps // sp.page_tokens + 4
    Q = max(sp.sink_pages + sp.recent_pages, int(round(q * n0)))
    pol = POLICIES[policy](Q, cap, sp, rng, **kw)
    pol.init(n0)
    n_prev = n0
    prev_sel = None
    acc = np.zeros(6)
    cnt = 0
    for t in range(tp.steps):
        scores, n = tr.step()
        if n > n_prev:
            for i in range(n_prev, n):
                pol.new_page(i, n)
            n_prev = n
        sel = select(scores, n, sp)
        h, m, fx, w = pol.access(sel, scores, n)
        if prev_sel is not None and t >= tp.warmup:
            ov = np.sum(sel[: prev_sel.size] & prev_sel)
            acc += (h, m, fx, w, sel.sum(), 1 - ov / sel.sum())
            cnt += 1
        prev_sel = sel
    a = acc / max(cnt, 1)
    return PageStats(*a)


def hitrate_curve(policy, ctx_tokens, qs, sp, tp, **kw):
    return [simulate(policy, ctx_tokens, q, sp, tp, **kw) for q in qs]
