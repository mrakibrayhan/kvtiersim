"""Fit the synthetic trace parameters (sigma_a, sigma_z, rho) to a real trace
and report real selection churn, then compare policies on the real trace.

usage: python tools/fit_trace.py traces.npz [--layers 0,8,16,24] [--qs 0.02,0.04,0.08]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from kvtiersim.config import DEFAULT_SPARSITY as SP, DEFAULT_TRACE as TP
from kvtiersim.traces import NpzTrace
from kvtiersim.engine import simulate


def fit(sc):
    """sc: [steps, pages] (NaN = page not yet present). Static part = per-page
    mean over time; dynamic part = residual; rho = lag-1 autocorrelation."""
    n = int(np.sum(~np.isnan(sc[0])))
    x = sc[:, :n]
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    a = np.nanmean(x, axis=0)
    z = x - a
    rho = np.nansum(z[1:] * z[:-1]) / np.nansum(z[:-1] ** 2)
    return float(np.nanstd(a)), float(np.nanstd(z)), float(rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layers", default="0,8,16,24")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--qs", default="0.02,0.04,0.08")
    a = ap.parse_args()
    d = np.load(a.npz)
    steps = d["scores"].shape[0]
    ctx0 = int(d["ctx0"])
    ctx0 -= ctx0 % SP.page_tokens
    for l in map(int, a.layers.split(",")):
        if l >= d["scores"].shape[1]:
            continue
        sa, sz, rho = fit(d["scores"][:, l, a.head, :])
        print(f"layer {l}: sigma_a={sa:.2f} sigma_z={sz:.2f} rho={rho:.3f}")
        tp = replace(TP, steps=steps, warmup=min(32, steps // 4))
        for q in map(float, a.qs.split(",")):
            res = {}
            for pol in ("Interleave", "Recency", "LRU", "CGP", "Oracle"):
                tr = NpzTrace(a.npz, layer=l, head=a.head)
                s = simulate(pol, ctx0, q, SP, tp, trace=tr)
                res[pol] = s
            far = {k: round(v.miss + v.far_extra, 1) for k, v in res.items()}
            print(f"   q={q:.2f} churn={res['LRU'].churn:.3f} far pages/step: {far}")


if __name__ == "__main__":
    main()
