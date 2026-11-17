"""Runs every experiment in the paper and writes CSVs + figures.

usage:  python run_all.py [--quick] [--out ../step2_results]
"""
import argparse, csv, json, os, sys, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtiersim.config import (DEFAULT_MODEL as M, DEFAULT_SPARSITY as SP,
                              DEFAULT_TRACE as TP, PCIE_CXL, C2C_LPDDR, GB)
from kvtiersim.engine import simulate
from kvtiersim.perfmodel import StatsTable, best_throughput

POLS = ["Interleave", "Recency", "LRU", "CGP", "Oracle"]
CTXS = [8192, 16384, 32768, 65536, 131072]
QS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
RHOS = [0.5, 0.9, 0.99]
SLO = 0.100   # ASSUMPTION: 100 ms time-per-output-token interactive SLO

_cache = {}


def table(pol, ctx, rho, **kw):
    key = (pol, ctx, rho, tuple(sorted(kw.items())))
    if key not in _cache:
        tp = replace(TP, rho=rho)
        _cache[key] = StatsTable(QS, [simulate(pol, ctx, q, SP, tp, **kw)
                                      for q in QS])
    return _cache[key]


def wcsv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("..", "step2_results"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global CTXS, TP
    if a.quick:
        CTXS = [8192, 32768, 131072]
        TP = replace(TP, steps=256, warmup=64)
    else:
        TP = replace(TP, steps=512, warmup=128)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    plats = [PCIE_CXL, C2C_LPDDR]

    # ---------- E1: full vs sparse attention, HBM-only vs tiered ----------
    rows = []
    for P in plats:
        for ctx in CTXS:
            for sparse in (False, True):
                for tiered, pol in ((False, "HBM-only"), (True, "CGP"), (True, "Oracle")):
                    st = table(pol, ctx, 0.9) if (sparse and tiered) else None
                    thr, B, q, parts = best_throughput(ctx, P, M, SP, sparse, tiered,
                                                       st, slo=SLO)
                    rows.append(dict(platform=P.name, ctx=ctx,
                                     attention="sparse" if sparse else "full",
                                     config=pol, tok_s=round(thr, 1), batch=B,
                                     q=None if q is None else round(q, 4)))
    wcsv(os.path.join(a.out, "E1_full_vs_sparse.csv"), rows)
    print(f"E1 done {time.time()-t0:.0f}s")

    # ---------- E2: far-tier traffic vs HBM quota (policy curves) ----------
    rows = []
    for ctx in (32768, 131072):
        for rho in RHOS:
            for pol in POLS:
                st = table(pol, ctx, rho)
                for i, q in enumerate(QS):
                    rows.append(dict(ctx=ctx, rho=rho, policy=pol, q=q,
                                     hit_rate=round(st.f["hit"][i] / st.f["sel"][i], 4),
                                     far_pages=round(st.f["miss"][i] + st.f["far_extra"][i], 3),
                                     hbm_wr=round(st.f["hbm_wr"][i], 3),
                                     churn=round(st.f["churn"][i], 4)))
    wcsv(os.path.join(a.out, "E2_policy_curves.csv"), rows)
    print(f"E2 done {time.time()-t0:.0f}s")

    # ---------- E3: end-to-end throughput per policy ----------
    rows = []
    for P in plats:
        for ctx in CTXS:
            base = best_throughput(ctx, P, M, SP, True, False, None, slo=SLO)[0]
            for rho in RHOS:
                for pol in POLS:
                    thr, B, q, parts = best_throughput(ctx, P, M, SP, True, True,
                                                       table(pol, ctx, rho), slo=SLO)
                    rows.append(dict(platform=P.name, ctx=ctx, rho=rho, policy=pol,
                                     tok_s=round(thr, 1), batch=B,
                                     speedup_vs_hbm_only=round(thr / base, 3) if base else None))
    wcsv(os.path.join(a.out, "E3_policy_throughput.csv"), rows)
    print(f"E3 done {time.time()-t0:.0f}s")

    # ---------- E4: far-tier bandwidth sweep (break-even) ----------
    rows = []
    for ctx in (32768, 131072):
        for bw in (16, 32, 64, 128, 256, 450, 900):
            P = replace(PCIE_CXL, far_bw=bw * GB, name=f"far {bw} GB/s")
            base = best_throughput(ctx, P, M, SP, True, False, None, slo=SLO)[0]
            fbase = best_throughput(ctx, P, M, SP, False, False, None, slo=SLO)[0]
            for pol in ("Interleave", "LRU", "CGP", "Oracle"):
                thr = best_throughput(ctx, P, M, SP, True, True, table(pol, ctx, 0.9), slo=SLO)[0]
                rows.append(dict(ctx=ctx, far_bw_GBs=bw, policy=pol, attention="sparse",
                                 speedup_vs_hbm_only=round(thr / base, 3)))
            fthr = best_throughput(ctx, P, M, SP, False, True, None, slo=SLO)[0]
            rows.append(dict(ctx=ctx, far_bw_GBs=bw, policy="(full attention)",
                             attention="full", speedup_vs_hbm_only=round(fthr / fbase, 3)))
    wcsv(os.path.join(a.out, "E4_farbw_sweep.csv"), rows)
    print(f"E4 done {time.time()-t0:.0f}s")

    # ---------- E5: sensitivity (overlap model, SLO, prefetch ablation) ----------
    rows = []
    P = PCIE_CXL
    for ctx in (32768, 131072):
        base = best_throughput(ctx, P, M, SP, True, False, None, slo=SLO)[0]
        for label, kw, pol_kw in (
                ("default", dict(slo=SLO), {}),
                ("no-overlap", dict(slo=SLO, overlap=False), {}),
                ("SLO 50 ms", dict(slo=0.05), {}),
                ("no SLO", dict(slo=None), {}),
                ("CGP+prefetch M=2", dict(slo=SLO), dict(M=2))):
            b = best_throughput(ctx, P, M, SP, True, False, None,
                                **{k: v for k, v in kw.items() if k != "overlap"})[0]
            for pol in ("LRU", "CGP"):
                pk = pol_kw if pol == "CGP" else {}
                thr = best_throughput(ctx, P, M, SP, True, True,
                                      table(pol, ctx, 0.9, **pk), **kw)[0]
                rows.append(dict(ctx=ctx, setting=label, policy=pol,
                                 tok_s=round(thr, 1), speedup_vs_hbm_only=round(thr / b, 3)))
    wcsv(os.path.join(a.out, "E5_sensitivity.csv"), rows)
    print(f"E5 done {time.time()-t0:.0f}s")

    meta = dict(model=M.name, kv_bytes_per_token=M.kv_bytes_per_token(),
                page_tokens=SP.page_tokens, budget_tokens=SP.budget_tokens,
                trace=dict(sigma_a=TP.sigma_a, sigma_z=TP.sigma_z, steps=TP.steps,
                           warmup=TP.warmup, seed=TP.seed),
                slo_s=SLO, platforms=[vars(p) for p in plats],
                runtime_s=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "w"), indent=2)
    print("all done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
