"""Extract REAL per-step KV-page criticality traces from a HuggingFace model.

Run on a GPU machine (Colab T4/A100 or the lab PC), NOT needed for the
synthetic results. Output can be replayed by KVTierSim (NpzTrace) and fitted
to the synthetic model with tools/fit_trace.py.

Criticality of a page = total attention probability mass that the current
query puts on the page, summed over the query heads of each KV-head group
(the quantity Quest's min/max upper bound approximates).

usage (example):
  python tools/extract_traces.py --model meta-llama/Llama-3.2-1B \
      --prompt-file long_doc.txt --ctx 8192 --steps 256 --out traces_1b.npz

Notes
  * uses eager attention (output_attentions=True); memory grows with ctx,
    so start with ctx 4K-16K.
  * gated models (Llama) need `huggingface-cli login`.
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def page_mass(attn, page, n_pages, kv_heads):
    # attn: [1, q_heads, 1, ctx] -> [kv_heads, n_pages]
    a = attn[0, :, -1, :].float()
    qh, ctx = a.shape
    pad = n_pages * page - ctx
    if pad > 0:
        a = torch.nn.functional.pad(a, (0, pad))
    a = a.view(kv_heads, qh // kv_heads, n_pages, page).sum(dim=(1, 3))
    return a.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--out", default="traces.npz")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        attn_implementation="eager").to(dev).eval()
    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    L = cfg.num_hidden_layers

    text = open(a.prompt_file, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[:, : a.ctx].to(dev)
    ctx0 = ids.shape[1]
    max_pages = (ctx0 + a.steps) // a.page + 1
    scores = np.full((a.steps, L, kv_heads, max_pages), np.nan, dtype=np.float32)

    with torch.no_grad():
        out = model(ids, use_cache=True)          # prefill
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for t in range(a.steps):
            out = model(nxt, past_key_values=past, use_cache=True,
                        output_attentions=True)
            past = out.past_key_values
            ctx = ctx0 + t + 1
            n_pages = (ctx + a.page - 1) // a.page
            # only completed pages are tiered; the partial page stays in HBM
            n_full = ctx // a.page
            for l, att in enumerate(out.attentions):
                m = page_mass(att, a.page, n_pages, kv_heads)[:, :n_full]
                scores[t, l, :, :n_full] = np.log(m + 1e-9)
            nxt = out.logits[:, -1:].argmax(-1)
            if t % 32 == 0:
                print(f"step {t}/{a.steps}")
    np.savez_compressed(a.out, scores=scores, ctx0=ctx0, page=a.page, model=a.model)
    print("saved", a.out, scores.shape)


if __name__ == "__main__":
    main()
