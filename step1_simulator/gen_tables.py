"""Generates the LaTeX tables of the paper directly from the result CSVs,
so every number in a table matches step2_results exactly.

usage: python gen_tables.py [--res ../step2_results] [--out ../paper/tables]
"""
import argparse, csv, os


def rd(res, name):
    return list(csv.DictReader(open(os.path.join(res, name))))


def tab_allconfigs(res):
    """All policies x platforms x contexts (rho = 0.9): tok/s, batch, speedup."""
    e3 = rd(res, "E3_policy_throughput.csv")
    e1 = rd(res, "E1_full_vs_sparse.csv")
    plats = ["HBM3 + CXL (x16)", "HBM3 + LPDDR5X (C2C)"]
    ctxs = ["32768", "131072"]
    L = []
    L.append(r"\begin{tabular}{@{}lcccc@{}}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{2}{c}{Platform A (CXL)} & \multicolumn{2}{c}{Platform B (C2C)} \\")
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    L.append(r"Configuration & 32K & 128K & 32K & 128K \\")
    L.append(r"\midrule")

    def cell(thr, B, sp=None):
        s = f"{float(thr):.0f} ({B})"
        return s + (f", {float(sp):.2f}$\\times$" if sp is not None else "")

    for att, cfg, lab in (("full", "HBM-only", "Full attn., HBM-only"),
                          ("full", "CGP", "Full attn., tiered"),
                          ("sparse", "HBM-only", "Sparse, HBM-only")):
        row = [lab]
        for p in plats:
            for c in ctxs:
                x = next(r for r in e1 if r["platform"] == p and r["ctx"] == c
                         and r["attention"] == att and r["config"] == cfg)
                row.append(f"{float(x['tok_s']):.0f} ({x['batch']})")
        L.append(" & ".join(row) + r" \\")
    L.append(r"\midrule")
    for pol in ("Interleave", "Recency", "LRU", "CGP", "Oracle"):
        lab = {"CGP": "Sparse, CGP", "Oracle": "Sparse, max-hit oracle"}.get(pol, f"Sparse, {pol}")
        row = [lab]
        for p in plats:
            for c in ctxs:
                x = next(r for r in e3 if r["platform"] == p and r["ctx"] == c
                         and r["rho"] == "0.9" and r["policy"] == pol)
                row.append(f"{float(x['tok_s']):.0f} ({x['batch']}), "
                           f"{float(x['speedup_vs_hbm_only']):.2f}$\\times$")
        L.append(" & ".join(row) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def tab_ablation(res):
    """E5 (model sensitivity, throughput speedup) + E7 (CGP ablation, traffic)."""
    e5 = rd(res, "E5_sensitivity.csv")
    e7 = rd(res, "E7_cgp_ablation.csv")
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
         r"\multicolumn{3}{@{}l}{(a) Speedup over HBM-only, Platform A, $\rho=0.9$ (LRU / CGP)} \\",
         r"\midrule", r"Setting & 32K & 128K \\", r"\midrule"]
    for key, lab in (("default", "Default (overlapped tiers)"),
                     ("no-overlap", "Serialised HBM and far tier"),
                     ("SLO 50 ms", "TPOT SLO 50 ms"),
                     ("no SLO", "No TPOT SLO")):
        cells = []
        for c in ("32768", "131072"):
            v = {r["policy"]: r["speedup_vs_hbm_only"] for r in e5
                 if r["ctx"] == c and r["setting"] == key}
            cells.append(f"{float(v['LRU']):.2f} / {float(v['CGP']):.2f}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{3}{@{}l}{(b) Far-tier traffic cut vs.\ LRU (\%), quota 4\%, 5 held-out seeds} \\",
          r"\midrule", r"CGP variant & 32K, $\rho$=0.5 / 0.9 & 128K, $\rho$=0.5 / 0.9 \\", r"\midrule"]
    names = [("CGP default (beta=0.3, gate on, M=0)", r"Default ($\beta$=0.3, gate on, $M$=0)"),
             ("CGP beta=0 (current score only)", r"$\beta$=0 (no EMA)"),
             ("CGP beta=0.6", r"$\beta$=0.6"),
             ("CGP beta=0.9", r"$\beta$=0.9"),
             ("CGP gate off (promote every miss)", "Gate off"),
             ("CGP hysteresis delta=0.25", r"$\delta$=0.25"),
             ("CGP + prefetch M=2", r"Prefetch $M$=2")]
    for key, lab in names:
        cells = []
        for c in ("32768", "131072"):
            v = [next(r for r in e7 if r["ctx"] == c and r["rho"] == rho and r["variant"] == key)
                 for rho in ("0.5", "0.9")]
            cells.append(" / ".join(f"{float(x['cut_vs_lru_mean_pct']):.1f}".replace("-", "$-$") for x in v))
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join("..", "step2_results"))
    ap.add_argument("--out", default=os.path.join("..", "step2_results", "tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "tab_allconfigs.tex"), "w").write(tab_allconfigs(a.res) + "\n")
    open(os.path.join(a.out, "tab_ablation.tex"), "w").write(tab_ablation(a.res) + "\n")
    print("tables written to", a.out)
