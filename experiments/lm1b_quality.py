"""LM1B: quality checks beyond GPT-2-large perplexity, for the variants lm1b_diagnostic.py saved.

- Human anchor: held-out LM1B validation chunks (128 BERT ids, the same shape as the samples), with the same
  metrics as the samples (FLM gen. PPL, entropy, corpus distinct-n).
- MAUVE (Pillutla et al., 2021) between each variant and human chunks. Features are GPT-2-large last-layer
  states at the final token, as the mauve-text package computes them; per seed (256 vs 256) and pooled.
- Gen. PPL under a second reference LM from a different family (--ref2), so a gain is not specific to GPT-2.
  Text is decoded without special tokens and scored up to 256 tokens of that model; ratios get a paired
  bootstrap over samples.

python experiments/lm1b_quality.py --size medium --variants "baseline,vcycle|m=6,no-restrict|m=6"
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import LM1B, corpus_distinct
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="medium")
ap.add_argument("--variants", default="baseline,vcycle|m=6,no-restrict|m=6")
ap.add_argument("--ref", default="openai-community/gpt2-large")
ap.add_argument("--ref2", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--n_boot", type=int, default=1000)
ap.add_argument("--no_mauve", action="store_true", help="skip MAUVE (lm1b_mauve.py measures it with uncertainty)")
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ck = os.path.join(ARTIFACTS, "lm1b", a.size)
ncm = load(ck, device=dev)
lm = LM1B(ncm, LevelDecoder(ncm), ck, ref=a.ref)
gen_tok = ncm.pre_tokenizer.hf_tokenizer
variants = [v for v in a.variants.split(",") if v]

# Samples per seed and variant, from the diagnostic caches.
runs = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s*.pt"))):
    seed = int(re.search(r"_s(\d+)\.pt$", path).group(1))
    c = torch.load(path, weights_only=False)
    got = {v: c["ids"] if v == "baseline" else c.get("variant_ids", {}).get(v) for v in variants}
    if all(x is not None for x in got.values()):
        runs[seed] = got
    else:
        print(f"seed {seed}: missing {[v for v, x in got.items() if x is None]}, skipped", flush=True)
seeds = sorted(runs)
n = len(runs[seeds[0]]["baseline"])
print(f"LM1B-{a.size}: seeds {seeds}, {n} samples each, variants {variants}", flush=True)

# Human chunks: disjoint random validation chunks, one block of n per seed.
val = np.memmap(os.path.join(DATA, "lm1b", "valid_bert_30522.bin"), dtype=np.uint16, mode="r")
L = 128
pick = np.random.default_rng(0).permutation(len(val) // L)[:n * len(seeds)]
human = torch.from_numpy(np.stack([np.asarray(val[i * L:(i + 1) * L], dtype=np.int64) for i in pick]))
human_by_seed = {s: human[j * n:(j + 1) * n] for j, s in enumerate(seeds)}

out = {"size": a.size, "seeds": seeds, "n_per_seed": n, "variants": variants, "ref2": a.ref2, "device": dev}
cache = {}

# 1. Human anchor under the paper's protocol.
nl, nt = lm.nll(human)
q = lm.quality(human)
hum = lm.corpus(nl, nt, q)
hum["corpus_distinct2"], hum["corpus_distinct3"] = corpus_distinct(human, lm.pad, 2, lm.cut_eos), corpus_distinct(human, lm.pad, 3, lm.cut_eos)
hum["corpus_distinct2_per_seed_block"] = [corpus_distinct(human_by_seed[s], lm.pad, 2, lm.cut_eos) for s in seeds]
out["human"] = hum
print("human:", hum, flush=True)


def texts(ids):
    return [t.strip() for t in gen_tok.batch_decode(ids.tolist(), skip_special_tokens=True)]


# 2. MAUVE with GPT-2-large features (the model lm.nll already loaded).
def features(ids, bs=16):
    rtok, ref = lm._ref["rtok"], lm._ref["lm"]
    ref.to(dev)
    feats = []
    for i in range(0, len(ids), bs):
        enc = rtok(texts(ids[i:i + bs]), return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True,
                   max_length=256)
        h = ref(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1]
        last = enc["attention_mask"].sum(1).to(dev) - 1
        feats.append(h[torch.arange(len(last), device=dev), last].float().cpu())
    ref.cpu()
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


if lm.offload:
    ncm.net.cpu()
if not a.no_mauve:
    import mauve  # noqa: E402

    with Timer() as t:
        hf = {s: features(human_by_seed[s]) for s in seeds}
        out["mauve"] = {}
        for v in variants:
            vf = {s: features(runs[s][v]) for s in seeds}
            per = [float(mauve.compute_mauve(p_features=hf[s], q_features=vf[s], verbose=False, seed=25).mauve) for s in seeds]
            pooled = float(mauve.compute_mauve(p_features=np.concatenate([hf[s] for s in seeds]),
                                               q_features=np.concatenate([vf[s] for s in seeds]), verbose=False, seed=25).mauve)
            out["mauve"][v] = {"per_seed": per, "pooled": pooled}
            print(f"MAUVE {v:18s} per seed {[round(x, 3) for x in per]} pooled {pooled:.3f}", flush=True)
    out["seconds_mauve"] = t.s
    save_json(out, f"lm1b_quality_{a.size}.json")

# 3. Gen. PPL under a second reference LM.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok2 = AutoTokenizer.from_pretrained(a.ref2)
tok2.pad_token = tok2.pad_token or tok2.eos_token
tok2.padding_side = "right"
lm2 = AutoModelForCausalLM.from_pretrained(a.ref2, dtype=torch.bfloat16).to(dev).eval()


def nll2(ids, bs=4):  # Qwen's 152k vocabulary: 16 x 256 fp32 logits alone would be 2.5 GB
    s_, n_ = [], []
    tx = texts(ids)
    for i in range(0, len(tx), bs):
        enc = tok2(tx[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        logits = lm2(**enc).logits.float()
        loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), enc["input_ids"][:, 1:], reduction="none")
        mask = enc["attention_mask"][:, 1:].float()
        s_.append((loss * mask).sum(1).cpu())
        n_.append(mask.sum(1).cpu())
    return torch.cat(s_).numpy().astype(float), torch.cat(n_).numpy().astype(float)


ppl = lambda x: float(np.exp(x[0].sum() / x[1].sum()))
h2 = nll2(human)
out["ref2_ppl"] = {"human": ppl(h2)}
cache["human"] = h2
pooled2 = {}
for v in variants:
    per = [nll2(runs[s][v]) for s in seeds]
    pooled2[v] = (np.concatenate([p[0] for p in per]), np.concatenate([p[1] for p in per]))
    cache[v] = pooled2[v]
    out["ref2_ppl"][v] = {"per_seed": [ppl(p) for p in per], "pooled": ppl(pooled2[v])}
    print(f"{a.ref2} PPL {v:18s} per seed {[round(ppl(p), 1) for p in per]} pooled {ppl(pooled2[v]):.1f}", flush=True)
N = len(pooled2["baseline"][0])
idx = np.random.default_rng(0).integers(0, N, size=(a.n_boot, N))
bp = lambda v, ix: np.exp(pooled2[v][0][ix].sum(-1) / pooled2[v][1][ix].sum(-1))
out["ref2_ratios"] = {}
for v in variants:
    for b in ["baseline"] + ([v.replace("vcycle", "no-restrict")] if v.startswith("vcycle") else []):
        if v == b or b not in pooled2:
            continue
        r = bp(v, idx) / bp(b, idx)
        out["ref2_ratios"][f"{v} / {b}"] = [float(bp(v, np.arange(N)) / bp(b, np.arange(N))),
                                           float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
        print(f"ratio {v} / {b}: {out['ref2_ratios'][f'{v} / {b}']}", flush=True)
save_json(out, f"lm1b_quality_{a.size}.json")
torch.save(cache, os.path.join(RESULTS, "cache", f"lm1b_quality_{a.size}.pt"))
