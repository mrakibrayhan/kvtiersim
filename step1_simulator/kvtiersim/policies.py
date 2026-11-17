"""KV-page placement policies for a two-tier (HBM + far) memory.

All policies manage one (sequence, layer) page set with an HBM quota of Q
pages. KV pages are write-once, so the far tier keeps an inclusive backing
copy of every completed page: demotion (eviction from HBM) is free, and
promotion costs one far read + one HBM write.

Per step a policy reports (in pages):
  hit      : selected pages served from HBM
  miss     : selected pages read directly from the far tier
  far_extra: extra far reads (background promotions/prefetch)
  hbm_wr   : HBM writes (promotions)
"""
import numpy as np
from .traces import select


class Base:
    name = "base"

    def __init__(self, Q, cap, sp, rng):
        self.Q, self.cap, self.sp, self.rng = Q, cap, sp, rng
        self.res = np.zeros(cap, dtype=bool)

    def init(self, n):
        raise NotImplementedError

    def new_page(self, i, n):
        raise NotImplementedError

    def access(self, sel, scores, n):
        raise NotImplementedError


class Interleave(Base):
    """Static capacity-proportional placement (address-hash interleaving)."""
    name = "Interleave"

    def init(self, n):
        idx = self.rng.choice(n, size=min(self.Q, n), replace=False)
        self.res[idx] = True
        self.q = self.Q / n

    def new_page(self, i, n):
        if self.res.sum() < self.Q and self.rng.random() < self.q:
            self.res[i] = True

    def access(self, sel, scores, n):
        h = int(np.sum(sel & self.res[:n]))
        return h, int(sel.sum()) - h, 0, 0


class Recency(Base):
    """Sink + most-recent pages in HBM (StreamingLLM-style static placement)."""
    name = "Recency"

    def init(self, n):
        self._place(n)

    def _place(self, n):
        self.res[:] = False
        s = self.sp.sink_pages
        self.res[:min(s, n)] = True
        k = max(0, self.Q - s)
        self.res[max(s, n - k):n] = True

    def new_page(self, i, n):
        self._place(n)   # oldest non-sink page demoted for free

    def access(self, sel, scores, n):
        h = int(np.sum(sel & self.res[:n]))
        return h, int(sel.sum()) - h, 0, 0


class LRU(Base):
    """Demand paging: every miss is promoted, LRU victim evicted."""
    name = "LRU"

    def init(self, n):
        self.ts = np.full(self.cap, -1.0)
        idx = np.arange(max(0, n - self.Q), n)   # most recent pages start in HBM
        self.res[idx] = True
        self.ts[idx] = 0.0
        self.t = 0

    def _evict(self, m, n, protect):
        cand = np.flatnonzero(self.res[:n] & ~protect)
        if m <= 0 or cand.size == 0:
            return
        m = min(m, cand.size)
        v = cand[np.argpartition(self.ts[cand], m - 1)[:m]]
        self.res[v] = False

    def new_page(self, i, n):
        prot = np.zeros(n, dtype=bool)
        self._evict(int(self.res[:n].sum()) + 1 - self.Q, n, prot)
        self.res[i] = True
        self.ts[i] = self.t

    def access(self, sel, scores, n):
        self.t += 1
        r = self.res[:n]
        hit = sel & r
        miss = sel & ~r
        h, m = int(hit.sum()), int(miss.sum())
        # promote misses (bounded by quota), evict LRU among non-selected
        need = int(r.sum()) + m - self.Q
        self._evict(need, n, sel)
        free = self.Q - int(self.res[:n].sum())
        mi = np.flatnonzero(miss)[:max(0, free)]
        self.res[mi] = True
        self.ts[np.flatnonzero(sel)] = self.t
        return h, m, 0, len(mi)


class CGP(Base):
    """Criticality-Guided Placement (proposed).

    Re-uses the page-criticality scores that the sparse-attention kernel
    already computes for *every* page each step (no extra metadata traffic):
      1. e_i <- beta*e_i + (1-beta)*score_i         (EMA of criticality)
      2. pin sink + recent window in HBM
      3. promote-on-miss only if the page ranks inside the HBM quota by e
         (the far read is already paid, so this costs only an HBM write),
         evicting the resident non-selected page with the lowest e
      4. optional background prefetch of up to M top-ranked non-resident
         pages per step, with hysteresis delta (default M=0: in a
         bandwidth-bound decode step prefetch moves traffic, it does not
         remove it -- see ablation)
    """
    name = "CGP"

    def __init__(self, Q, cap, sp, rng, beta=0.3, M=0, delta=0.0, gate=True):
        super().__init__(Q, cap, sp, rng)
        self.beta, self.M, self.delta, self.gate = beta, M, delta, gate
        self.e = np.zeros(cap)

    def _pinned(self, n):
        p = np.zeros(n, dtype=bool)
        p[: self.sp.sink_pages] = True
        p[max(0, n - self.sp.recent_pages):] = True
        return p

    def init(self, n):
        idx = np.arange(max(0, n - self.Q), n)
        self.res[idx] = True
        self.res[: self.sp.sink_pages] = True
        extra = int(self.res[:n].sum()) - self.Q
        if extra > 0:
            self.res[np.flatnonzero(self.res[:n])[self.sp.sink_pages:
                                                  self.sp.sink_pages + extra]] = False

    def _victims(self, k, n, pinned):
        cand = np.flatnonzero(self.res[:n] & ~pinned)
        if k <= 0 or cand.size == 0:
            return cand[:0]
        k = min(k, cand.size)
        return cand[np.argpartition(self.e[cand], k - 1)[:k]]

    def new_page(self, i, n):
        pinned = self._pinned(n)
        over = int(self.res[:n].sum()) + 1 - self.Q
        v = self._victims(over, n, pinned)
        self.res[v] = False
        self.res[i] = True
        self.e[i] = 0.0

    def access(self, sel, scores, n):
        b = self.beta
        self.e[:n] = b * self.e[:n] + (1 - b) * scores
        r = self.res[:n]
        hit = sel & r
        miss = sel & ~r
        h, m = int(hit.sum()), int(miss.sum())
        pinned = self._pinned(n)
        hbm_wr = 0
        far_extra = 0
        # threshold = e of the Q-th best page (desired HBM set boundary)
        Q = min(self.Q, n)
        thr = np.partition(self.e[:n], n - Q)[n - Q] if Q > 0 else np.inf
        # (3) promote-on-miss for pages inside the desired set
        cand = np.flatnonzero(miss & (self.e[:n] >= thr)) if self.gate else np.flatnonzero(miss)
        # (4) background prefetch of best non-resident pages
        nonres = np.flatnonzero(~self.res[:n] & ~miss)
        if nonres.size and self.M > 0:
            k = min(self.M, nonres.size)
            top = nonres[np.argpartition(-self.e[nonres], k - 1)[:k]]
            top = top[self.e[top] >= thr]
        else:
            top = nonres[:0]
        for group, is_bg in ((cand, False), (top, True)):
            if group.size == 0:
                continue
            group = group[np.argsort(-self.e[group])]
            free = self.Q - int(self.res[:n].sum())
            vict = self._victims(max(0, group.size - free), n, pinned | sel)
            vict = vict[np.argsort(self.e[vict])]
            nv = 0
            for g in group:
                if self.res[:n].sum() < self.Q:
                    pass
                elif nv < vict.size and self.e[g] > self.e[vict[nv]] + self.delta:
                    self.res[vict[nv]] = False
                    nv += 1
                else:
                    break
                self.res[g] = True
                hbm_wr += 1
                if is_bg:
                    far_extra += 1
        return h, m, far_extra, hbm_wr


class Oracle(Base):
    """Upper bound: HBM always holds the selected set (free migration)."""
    name = "Oracle"

    def init(self, n):
        pass

    def new_page(self, i, n):
        pass

    def access(self, sel, scores, n):
        k = int(sel.sum())
        h = min(k, self.Q)
        return h, k - h, 0, 0


POLICIES = {c.name: c for c in (Interleave, Recency, LRU, CGP, Oracle)}
