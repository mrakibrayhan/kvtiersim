"""Builds the paper figures from the CSVs written by run_all.py.

usage: python make_figures.py [--res ../step2_results]
Figures are IEEE single-column width (3.5 in), PDF (vector) + PNG preview.
"""
import argparse, csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed order) + markers/linestyles so identity
# never depends on color alone (print / grayscale / CVD).
STY = {
    "Interleave": dict(color="#2a78d6", marker="s", ls="-"),
    "Recency":    dict(color="#eb6834", marker="^", ls="-"),
    "LRU":        dict(color="#1baf7a", marker="o", ls="-"),
    "CGP":        dict(color="#eda100", marker="D", ls="-"),
    "Oracle":     dict(color="#e87ba4", marker="v", ls="--"),
}
LABEL = {"Oracle": "Max-hit oracle", "CGP": "CGP (proposed)"}
INK, MUTED, GRID = "#2b2b28", "#6b6a63", "#e6e5e0"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "savefig.dpi": 300,
})


def rd(res, name):
    return list(csv.DictReader(open(os.path.join(res, name))))


def save(fig, res, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(res, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def ctx_lab(c):
    return f"{int(c)//1024}K"


def fig1(res):
    """Throughput vs context: full vs sparse attention, HBM-only vs tiered."""
    r = rd(res, "E1_full_vs_sparse.csv")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
    series = [("full", "HBM-only", "Full attn, HBM-only", "#6b6a63", "o", ":"),
              ("full", "CGP", "Full attn, tiered", "#6b6a63", "s", "-"),
              ("sparse", "HBM-only", "Sparse attn, HBM-only", "#2a78d6", "o", ":"),
              ("sparse", "CGP", "Sparse attn, tiered (CGP)", "#2a78d6", "s", "-")]
    for ax, plat in zip(axs, ["HBM3 + CXL (x16)", "HBM3 + LPDDR5X (C2C)"]):
        for att, cfg, lab, c, m, ls in series:
            pts = sorted((int(x["ctx"]), float(x["tok_s"])) for x in r
                         if x["platform"] == plat and x["attention"] == att and x["config"] == cfg)
            ax.plot([p[0] for p in pts], [p[1] / 1e3 for p in pts], color=c,
                    marker=m, ls=ls, label=lab)
        ax.set_xscale("log", base=2)
        xs = sorted({int(x["ctx"]) for x in r})
        ax.set_xticks(xs, [ctx_lab(x) for x in xs])
        ax.minorticks_off()
        ax.set_title(plat, color=INK)
        ax.set_xlabel("Context length (tokens)")
    axs[0].set_ylabel("Decode throughput (k tok/s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.53, 1.17))
    save(fig, res, "fig1_full_vs_sparse")


def fig2(res):
    """(a) far-tier traffic vs HBM quota; (b) CGP vs LRU traffic vs persistence."""
    r = rd(res, "E2_policy_curves.csv")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    ax = axs[0]
    for pol, st in STY.items():
        pts = sorted((float(x["q"]), float(x["far_pages"])) for x in r
                     if x["policy"] == pol and x["ctx"] == "32768" and x["rho"] == "0.9")
        pts = [p for p in pts if p[0] <= 0.3]
        ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts],
                label=LABEL.get(pol, pol), **st)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 30], ["0.5", "1", "2", "5", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("HBM quota (% of KV pages)")
    ax.set_ylabel("Far-tier reads (pages/step)")
    ax.set_title("(a) 32K ctx, $\\rho$=0.9", color=INK)

    ax = axs[1]
    e6 = os.path.join(res, "E6_cgp_vs_lru_seeds.csv")
    rows = rd(res, "E6_cgp_vs_lru_seeds.csv") if os.path.exists(e6) else []
    for (ctx, q), c, m in (((32768, 0.04), "#2a78d6", "o"), ((131072, 0.04), "#eb6834", "s"),
                           ((131072, 0.08), "#1baf7a", "^")):
        pts = sorted((float(x["rho"]), float(x["mean_cut_pct"]), float(x["std_cut_pct"]))
                     for x in rows if int(x["ctx"]) == ctx and float(x["q"]) == q)
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    color=c, marker=m, capsize=1.5, elinewidth=0.6,
                    label=f"{ctx_lab(ctx)}, quota {q*100:.0f}%")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Criticality persistence $\\rho$")
    ax.set_ylabel("Far traffic cut vs LRU (%)")
    ax.set_title("(b) CGP vs LRU (5 seeds)", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout(w_pad=0.8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=5.8,
               handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.5, 1.09))
    save(fig, res, "fig2_policy_traffic")


def fig3(res):
    """Speedup over HBM-only vs far-tier bandwidth."""
    r = rd(res, "E4_farbw_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
    for ax, ctx in zip(axs, ("32768", "131072")):
        for pol in ("Interleave", "LRU", "CGP", "Oracle"):
            pts = sorted((float(x["far_bw_GBs"]), float(x["speedup_vs_hbm_only"]))
                         for x in r if x["ctx"] == ctx and x["policy"] == pol)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=LABEL.get(pol, pol), **STY[pol])
        pts = sorted((float(x["far_bw_GBs"]), float(x["speedup_vs_hbm_only"]))
                     for x in r if x["ctx"] == ctx and x["attention"] == "full")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, marker="x",
                ls=":", label="Full attn (any policy)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 900], ["16", "64", "256", "900"])
        ax.minorticks_off()
        ax.axhline(1, color=MUTED, lw=0.6)
        ax.set_title(f"{ctx_lab(ctx)} context", color=INK)
        ax.set_xlabel("Far-tier bandwidth (GB/s)")
    axs[0].set_ylabel("Speedup vs HBM-only")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 1.17))
    save(fig, res, "fig3_farbw_sweep")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join("..", "step2_results"))
    a = ap.parse_args()
    fig1(a.res); fig2(a.res); fig3(a.res)
    print("figures written to", a.res)
