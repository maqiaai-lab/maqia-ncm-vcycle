"""LM1B: MAUVE with uncertainty (paired half-sample subsampling), for the variants lm1b_diagnostic.py saved.

Features: GPT-2-large last-layer state at the final token (as mauve-text computes them) of each sample's text,
decoded without special tokens; human reference = held-out validation chunks (same draw as lm1b_quality.py).
Uncertainty: each round draws half of the pooled samples (one index set shared by every variant, since all
variants restart from the same single-pass samples) and half of the human chunks, without replacement, and
recomputes MAUVE for every variant with a fresh k-means seed. Resampling *with* replacement is not used: the
duplicates distort MAUVE's quantization and bias every estimate down (an earlier run's intervals excluded their
own point estimates). Absolute MAUVE depends on the set size, so half-size values sit below the full-size point
estimates; the paired differences between variants at equal size are what the rounds are for. Reported: point
estimates on the full sets; per round, the paired differences V-cycle - no-restriction and V-cycle - single
pass (mean, 2.5-97.5 percentiles, share of rounds won). Features are cached, so reruns skip the GPU.

python experiments/lm1b_mauve.py --size medium --boot 50   (--boot = subsampling rounds)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="medium")
ap.add_argument("--variants", default="baseline,vcycle|m=6,no-restrict|m=6,vcycle|m=8,no-restrict|m=8")
ap.add_argument("--ref", default="openai-community/gpt2-large")
ap.add_argument("--boot", type=int, default=50)
ap.add_argument("--features_only", action="store_true", help="compute and cache the features (GPU), then stop")
a = ap.parse_args()
variants = [v for v in a.variants.split(",") if v]
feat_path = os.path.join(RESULTS, "cache", f"lm1b_mauve_feats_{a.size}.pt")

if os.path.exists(feat_path):
    F_ = torch.load(feat_path, weights_only=False)
    print(f"features from {feat_path}", flush=True)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    dev = device()
    btok = AutoTokenizer.from_pretrained("bert-base-uncased")  # LM1B's pre-tokenizer (bert-base-uncased ids)
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s*.pt"))):
        c = torch.load(path, weights_only=False)
        got = {v: c["ids"] if v == "baseline" else c.get("variant_ids", {}).get(v) for v in variants}
        if all(x is not None for x in got.values()):
            runs[int(re.search(r"_s(\d+)\.pt$", path).group(1))] = got
    seeds = sorted(runs)
    n = len(runs[seeds[0]]["baseline"])
    val = np.memmap(os.path.join(DATA, "lm1b", "valid_bert_30522.bin"), dtype=np.uint16, mode="r")
    pick = np.random.default_rng(0).permutation(len(val) // 128)[:n * len(seeds)]  # as in lm1b_quality.py
    human = torch.from_numpy(np.stack([np.asarray(val[i * 128:(i + 1) * 128], dtype=np.int64) for i in pick]))
    rtok = AutoTokenizer.from_pretrained(a.ref)
    rtok.pad_token = rtok.pad_token or rtok.eos_token
    ref = AutoModelForCausalLM.from_pretrained(a.ref).to(dev).eval()

    def features(ids, bs=16):
        out = []
        for i in range(0, len(ids), bs):
            tx = [t.strip() for t in btok.batch_decode(ids[i:i + bs].tolist(), skip_special_tokens=True)]
            enc = rtok(tx, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True, max_length=256)
            h = ref(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1]
            last = enc["attention_mask"].sum(1).to(dev) - 1
            out.append(h[torch.arange(len(last), device=dev), last].float().cpu())
        return torch.cat(out).numpy()

    F_ = {"seeds": seeds, "n": n, "human": features(human),
          "variants": {v: np.concatenate([features(runs[s][v]) for s in seeds]) for v in variants}}
    torch.save(F_, feat_path)
    del ref
    torch.cuda.empty_cache()

if a.features_only:
    print(f"features cached in {feat_path}", flush=True)
    sys.exit(0)

import mauve  # noqa: E402

H, V = F_["human"], F_["variants"]
N, M = len(next(iter(V.values()))), len(H)
mv = lambda p, q, seed: float(mauve.compute_mauve(p_features=p, q_features=q, verbose=False, seed=seed).mauve)
out = {"size": a.size, "seeds": F_["seeds"], "n_pooled": N, "n_human": M, "boot": a.boot,
       "scheme": "half-size subsamples without replacement, paired across variants", "point": {}, "boot_values": {}}
with Timer() as t:
    for v in variants:
        out["point"][v] = mv(H, V[v], 25)
        print(f"MAUVE {v:18s} {out['point'][v]:.3f}", flush=True)
    rng = np.random.default_rng(0)
    bv = {v: [] for v in variants}
    for b in range(a.boot):
        iq, ip = rng.permutation(N)[:N // 2], rng.permutation(M)[:M // 2]
        for v in variants:
            bv[v].append(mv(H[ip], V[v][iq], 1000 + b))
        if (b + 1) % 10 == 0:
            print(f"  round {b + 1}/{a.boot}", flush=True)
            out["boot_values"] = bv
            save_json(out, f"lm1b_mauve_{a.size}.json")
out["seconds"] = t.s
out["boot_values"] = bv
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
out["half_size"] = {v: [float(np.mean(bv[v]))] + ci(bv[v]) for v in variants}  # mean and interval at N/2
out["paired"] = {}
for v in variants:
    if not v.startswith("vcycle"):
        continue
    for b_ in ("baseline", v.replace("vcycle", "no-restrict")):
        if b_ in bv:
            d = np.array(bv[v]) - np.array(bv[b_])
            out["paired"][f"{v} - {b_}"] = {"full": out["point"][v] - out["point"][b_], "half_mean": float(d.mean()),
                                            "half_ci": ci(d), "share_won": float((d > 0).mean())}
            r = out["paired"][f"{v} - {b_}"]
            print(f"{v} - {b_}: full {r['full']:+.3f}; half-size {r['half_mean']:+.3f} [{r['half_ci'][0]:+.3f}, "
                  f"{r['half_ci'][1]:+.3f}], won {100 * r['share_won']:.0f}% of rounds", flush=True)
save_json(out, f"lm1b_mauve_{a.size}.json")
