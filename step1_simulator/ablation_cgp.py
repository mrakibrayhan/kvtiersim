"""E7: CGP ablation on held-out seeds (200-204; CGP defaults were chosen on seed 1).
usage: python ablation_cgp.py [--res ../step2_results] [--seeds 5]"""
import argparse, csv, os, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtiersim.config import DEFAULT_SPARSITY as SP, DEFAULT_TRACE as TP
from kvtiersim.engine import simulate

VARIANTS = [
    ("LRU (baseline)", "LRU", {}),
    ("CGP default (beta=0.3, gate on, M=0)", "CGP", {}),
    ("CGP beta=0 (current score only)", "CGP", dict(beta=0.0)),
    ("CGP beta=0.6", "CGP", dict(beta=0.6)),
    ("CGP beta=0.9", "CGP", dict(beta=0.9)),
    ("CGP gate off (promote every miss)", "CGP", dict(gate=False)),
    ("CGP + prefetch M=2", "CGP", dict(M=2)),
    ("CGP hysteresis delta=0.25", "CGP", dict(delta=0.25)),
]
ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join("..", "step2_results"))
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
rows = []
for ctx in (32768, 131072):
    for rho in (0.5, 0.9):
        q = 0.04
        base = None
        for label, pol, kw in VARIANTS:
            far = []
            for s in range(a.seeds):
                tp = replace(TP, rho=rho, steps=512, warmup=128, seed=200 + s)
                st = simulate(pol, ctx, q, SP, tp, **kw)
                far.append(st.miss + st.far_extra)
            far = np.array(far)
            if base is None:
                base = far
            cut = 100 * (1 - far / base)
            rows.append(dict(ctx=ctx, rho=rho, q=q, variant=label,
                             far_pages_mean=round(far.mean(), 2),
                             cut_vs_lru_mean_pct=round(cut.mean(), 2),
                             cut_vs_lru_std_pct=round(cut.std(), 2), seeds=a.seeds))
            print(rows[-1])
with open(os.path.join(a.res, "E7_cgp_ablation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
