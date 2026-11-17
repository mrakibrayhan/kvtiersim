"""Decode-step performance model (bandwidth/compute roofline per step).

Step time T(B) = max(T_compute, T_HBM, T_far): HBM and far-tier transfers and
compute are assumed to overlap (optimistic; `overlap=False` sums the memory
terms instead, as a sensitivity check).
"""
import numpy as np
from .config import meta_bytes_per_page_layer
from .engine import PageStats

Q_HEADS = 32  # Llama-3.1-8B query heads


class StatsTable:
    """Interpolates PageStats over the HBM-quota fraction q."""

    def __init__(self, qs, stats):
        self.qs = np.asarray(qs)
        self.f = {k: np.array([getattr(s, k) for s in stats])
                  for k in ("hit", "miss", "far_extra", "hbm_wr", "sel", "churn")}

    def __call__(self, q):
        q = float(np.clip(q, self.qs[0], self.qs[-1]))
        return PageStats(**{k: float(np.interp(q, self.qs, v))
                            for k, v in self.f.items()})


def capacity_q(B, ctx, plat, model, sp, sparse, tiered):
    """HBM quota fraction q for batch B, or None if infeasible."""
    n = ctx // sp.page_tokens
    page_b = sp.page_tokens * model.kv_bytes_per_token_layer()
    budget = plat.hbm_cap * plat.hbm_usable - model.weight_bytes - plat.activation_bytes
    meta = B * model.layers * n * meta_bytes_per_page_layer(model) if sparse else 0.0
    kv_avail = budget - meta
    if kv_avail <= 0:
        return None
    q = kv_avail / (B * model.layers * n * page_b)
    pinned = (sp.sink_pages + sp.recent_pages) / n
    if not tiered:
        return 1.0 if q >= 1.0 else None
    if q >= 1.0:
        return 1.0
    # inclusive far-tier copy of all KV must fit
    if B * ctx * model.kv_bytes_per_token() > plat.far_cap:
        return None
    if q < pinned:          # cannot even hold sink + recent window
        return None
    return q


def step_time(B, ctx, q, plat, model, sp, sparse, stats=None, overlap=True):
    n = ctx // sp.page_tokens
    L = model.layers
    page_b = sp.page_tokens * model.kv_bytes_per_token_layer()
    if sparse:
        s = stats(q) if q < 1.0 else None
        if s is None:
            K = min(sp.budget_pages(), n)
            hit, miss, fx, wr, sel = K, 0.0, 0.0, 0.0, K
        else:
            hit, miss, fx, wr, sel = s.hit, s.miss, s.far_extra, s.hbm_wr, s.sel
        meta = n * meta_bytes_per_page_layer(model)
    else:
        hit, miss, fx, wr, sel = q * n, (1 - q) * n, 0.0, 0.0, n
        meta = 0.0
    hbm = model.weight_bytes + B * L * ((hit + wr) * page_b + meta)
    far = B * L * (miss + fx) * page_b
    if q < 1.0:  # write-once backing copy of each newly completed page
        far += B * L * page_b / sp.page_tokens
    att_tokens = sel * sp.page_tokens
    flops = B * (2 * model.params + 4 * att_tokens * Q_HEADS * model.head_dim * L)
    tc = flops / (plat.peak_flops * plat.flop_eff)
    th = hbm / (plat.hbm_bw * plat.bw_eff)
    tf = far / (plat.far_bw * plat.bw_eff)
    t = max(tc, th, tf) if overlap else max(tc, th + tf)
    return t, dict(tc=tc, th=th, tf=tf)


def best_throughput(ctx, plat, model, sp, sparse, tiered, stats=None,
                    slo=None, B_max=1024, overlap=True):
    """Max tokens/s over batch size, subject to capacity and optional TPOT SLO."""
    best = (0.0, 0, None, None)
    for B in range(1, B_max + 1):
        q = capacity_q(B, ctx, plat, model, sp, sparse, tiered)
        if q is None:
            if not tiered:
                break
            continue
        t, parts = step_time(B, ctx, q, plat, model, sp, sparse, stats, overlap)
        if slo is not None and t > slo:
            continue
        thr = B / t
        if thr > best[0]:
            best = (thr, B, q, parts)
    return best
