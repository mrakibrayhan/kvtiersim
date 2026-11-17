"""Multi-seed check of CGP vs LRU far-traffic reduction (Fig. 2b data).
usage: python seeds_cgp_vs_lru.py [--res ../step2_results] [--seeds 5]"""
import argparse, csv, os, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtiersim.config import DEFAULT_SPARSITY as SP, DEFAULT_TRACE as TP
from kvtiersim.engine import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=os.path.join("..", "step2_results"))
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
rows = []
for ctx in (32768, 131072):
    for rho in (0.5, 0.7, 0.9, 0.95, 0.99):
        for q in (0.02, 0.04, 0.08):
            red = []
            for s in range(a.seeds):
                tp = replace(TP, rho=rho, steps=512, warmup=128, seed=100 + s)
                l = simulate("LRU", ctx, q, SP, tp); c = simulate("CGP", ctx, q, SP, tp)
                red.append(100 * (1 - (c.miss + c.far_extra) / (l.miss + l.far_extra)))
            rows.append(dict(ctx=ctx, rho=rho, q=q, mean_cut_pct=round(np.mean(red), 2),
                             std_cut_pct=round(np.std(red), 2), seeds=a.seeds))
            print(rows[-1])
with open(os.path.join(a.res, "E6_cgp_vs_lru_seeds.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
