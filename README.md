# KVTierSim: KV-Cache Tiering for Long-Context LLM Decoding

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24301736.svg)](https://doi.org/10.5281/zenodo.24301736)

Trace-driven simulation study of HBM + far-tier (CXL / LPDDR5X) memory
hierarchies for the KV cache of long-context LLM decoding, with and without
query-aware sparse attention. Includes **CGP (Criticality-Guided Placement)**,
a KV-page placement policy that reuses the page scores already computed by
the sparse-attention kernel.

Author: Md. Rakib Rayhan (North South University)

---

## Step-by-step guide (do these in order)

### Step 0: Set up (about 1 min)
```bash
python3 -m pip install -r step1_simulator/requirements.txt   # numpy, matplotlib
```

### Step 1: Run all experiments E1–E5 (about 65 s)
```bash
cd step1_simulator
python3 run_all.py            # writes ../step2_results/E1..E5 *.csv + run_meta.json
```

### Step 2: Multi-seed check of CGP vs LRU, E6 (about 60 s)
```bash
python3 seeds_cgp_vs_lru.py   # writes ../step2_results/E6_cgp_vs_lru_seeds.csv
```

### Step 3: CGP ablation on held-out seeds, E7 (about 3 min)
```bash
python3 ablation_cgp.py       # writes ../step2_results/E7_cgp_ablation.csv
```

### Step 4: Make the figures and LaTeX tables (about 5 s)
```bash
python3 make_figures.py       # fig2..fig4 as .pdf and .png in ../step2_results
python3 gen_tables.py         # LaTeX tables in ../step2_results/tables
```
(Fig. 1 of the paper is a TikZ drawing of the modelled system and this flow.)

### Step 5 (optional): Replay real model traces (GPU, about 1–3 h)
```bash
pip install torch transformers
python3 tools/extract_traces.py --model meta-llama/Llama-3.2-1B \
        --prompt-file long_doc.txt --ctx 8192 --steps 256 --out real.npz
python3 tools/fit_trace.py real.npz --layers 0,4,8,12
```
`fit_trace.py` fits the synthetic trace parameters (σa, σz, ρ) to the real
trace, reports selection churn, and compares all placement policies on it.

---

## Repository layout

| Path | Contents |
|---|---|
| `step1_simulator/kvtiersim/config.py` | Model (Llama-3.1-8B, GQA), platforms (HBM3+CXL x16, HBM3+LPDDR5X C2C), trace parameters |
| `step1_simulator/kvtiersim/traces.py` | Synthetic AR(1) page-criticality traces; real-trace replayer |
| `step1_simulator/kvtiersim/policies.py` | Interleave, Recency, LRU, CGP, max-hit Oracle |
| `step1_simulator/kvtiersim/engine.py` | Page-level simulation → per-step page statistics |
| `step1_simulator/kvtiersim/perfmodel.py` | Step-time roofline + batch-size search under HBM/far capacity |
| `step1_simulator/tools/` | Real-trace extraction and fitting |
| `step2_results/` | All CSV results, run metadata, and figures |

## Experiments

| ID | File | What it measures |
|---|---|---|
| E1 | `E1_full_vs_sparse.csv` | Best decode throughput vs context, full vs sparse attention, HBM-only vs tiered |
| E2 | `E2_policy_curves.csv` | Far-tier traffic and hit rate vs HBM quota, per policy and trace persistence ρ |
| E3 | `E3_policy_throughput.csv` | End-to-end throughput per policy |
| E4 | `E4_farbw_sweep.csv` | Speedup vs far-tier bandwidth (16–900 GB/s) |
| E5 | `E5_sensitivity.csv` | Serialised memory tiers, TPOT SLO, prefetch ablation |
| E6 | `E6_cgp_vs_lru_seeds.csv` | CGP vs LRU far-traffic reduction, 5 seeds |
| E7 | `E7_cgp_ablation.csv` | CGP component ablation (EMA weight, gate, hysteresis, prefetch), 5 held-out seeds |

## Model assumptions (see `config.py`)
- Page-level traces are synthetic (Eq. in `traces.py`) unless replayed from `tools/`.
- Step time is a bandwidth/compute roofline; HBM, far tier and compute are
  assumed to overlap (a serialised variant is in E5). CXL latency and queueing
  are not modelled.
- Efficiency factors (50% FLOP, 80% bandwidth, 90% usable HBM) are assumptions.

## Citation
See `CITATION.cff` (GitHub: "Cite this repository").

## License
MIT
