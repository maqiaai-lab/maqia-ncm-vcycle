"""LM1B arm of Phase 1 (+ a small V-cycle test): does the round-trip residual track bad text?

Samples follow scripts/eval_shard.py for lm1b-<size> exactly: level 0 from the level-0 unigram, levels
1..K-1 at temperature 1, tokens picked from the decoder logits with the frequency and adjacency penalties
in the released config. Per sample we measure the residual (round trip and level-local) and quality:
NLL under GPT-2 large (FLM protocol: special tokens kept, first 128 GPT-2 tokens), adjacent repeats,
distinct-2/3 and unigram entropy of the raw ids.

--vcycle m adds, for the same samples: V-cycle (keep encode(x)_{<m}, resample m..K-1 at T=1) and the
no-restriction control (keep generated r_{<m}), so the effect of the restriction step on text is
measured at equal compute.

python experiments/lm1b_diagnostic.py --size small --n 256 --seed 42 --vcycle 3,6
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, RESULTS, Timer, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import LM1B, auroc, corpus_distinct, spearman
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="small")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--vcycle", default="", help="comma-separated restart levels for the V-cycle comparison")
ap.add_argument("--ref", default="openai-community/gpt2-large")
ap.add_argument("--check", type=int, default=8, help="samples to compare against net.generate")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, "lm1b", a.size)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm)
lm = LM1B(ncm, dec, ck, ref=a.ref, bs=a.bs)

# 1. Equivalence with the released sampler (same seed -> same codes).
torch.manual_seed(1234)
ref = ncm.net.generate(lm.sample_l0(a.check), lm.prompt(a.check), temperature=lm.temp, top_k=0, top_p=1.0,
                       cfg_scale=0.0, cfg_null_eos=lm.eos)
torch.manual_seed(1234)
l0 = lm.sample_l0(a.check)
mine = torch.cat(dec.run(dec.prefix(lm.prompt(a.check)), start=1, codes=[l0], temperature=lm.temp)[0], 1)
assert torch.equal(ref, mine), "LevelDecoder differs from generate() on LM1B"
print(f"LevelDecoder == generate() on {a.check} LM1B samples (temperature {lm.temp})", flush=True)


def summarize(ids, nl, nt):
    r = lm.corpus(nl, nt, lm.quality(ids))
    r["corpus_distinct2"], r["corpus_distinct3"] = corpus_distinct(ids, lm.pad, 2), corpus_distinct(ids, lm.pad, 3)
    return r


# 2. Samples, residuals, quality. Resumes from the cache if an interrupted run already scored the baseline.
cache_path = os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s{a.seed}.pt")
prev = torch.load(cache_path, weights_only=False) if os.path.exists(cache_path) else None
if prev is not None and len(prev["ids"]) == a.n:
    codes, ids, (nl, nt) = prev["codes"], prev["ids"], prev["per_sample_nll"]["baseline"]
    gen_s = score_s = None
    print(f"resuming from {cache_path}", flush=True)
else:
    prev = None
    with Timer() as t:
        codes, ids = lm.generate(a.n, a.seed)
    gen_s = t.s
    with Timer() as t:
        nl, nt = lm.nll(ids)
    score_s = t.s
rt, loc = lm.residuals(codes, ids)
q = lm.quality(ids)
ppl_s = nl / np.maximum(nt, 1)
res = {"size": a.size, "n": a.n, "seed": a.seed, "levels": dec.levels, "seconds_generate": gen_s, "seconds_score": score_s,
       "baseline": summarize(ids, nl, nt)}
res["fixed_point_rate"] = float(((rt.sum(1) == 0)).float().mean())
res["disagree_rate_per_level"] = {"roundtrip": rt.mean(0).tolist(), "local": loc.mean(0).tolist()}
score = {"roundtrip_total": rt.sum(1).numpy(), "local_total": loc.sum(1).numpy(),
         **{f"local_level{k}": loc[:, k].numpy() for k in range(dec.K)}}
bad_nll = ppl_s >= np.quantile(ppl_s, 0.9)
bad_rep = q["adjacent_repeat"] >= np.quantile(q["adjacent_repeat"], 0.9)
res["spearman"] = {name: {"per_token_nll": spearman(s, ppl_s), "adjacent_repeat": spearman(s, q["adjacent_repeat"]),
                          "distinct2": spearman(s, q["distinct2"]), "entropy": spearman(s, q["entropy"])}
                   for name, s in score.items()}
res["auroc_worst_decile"] = {name: {"by_nll": auroc(s, bad_nll), "by_repeat": auroc(s, bad_rep)} for name, s in score.items()}
print("baseline:", res["baseline"], flush=True)
print("fixed-point rate:", res["fixed_point_rate"])
for name in ("roundtrip_total", "local_total"):
    print(name, "spearman", res["spearman"][name], "auroc", res["auroc_worst_decile"][name], flush=True)

# 3. V-cycle vs no-restriction at equal compute (one cycle from level m, temperature 1).
res["vcycle"] = {}
per_sample = {"baseline": (nl, nt)}
variant_ids = {}


def checkpoint():
    save_json(res, f"lm1b_{a.size}_s{a.seed}.json")
    torch.save({"codes": codes, "ids": ids, "rt": rt, "loc": loc, "ppl": ppl_s, "per_sample_nll": per_sample,
                "variant_ids": variant_ids}, cache_path)


checkpoint()
for m in [int(v) for v in a.vcycle.split(",") if v]:
    for label, reencode in (("vcycle", True), ("no-restrict", False)):
        key = f"{label}|m={m}"
        if prev is not None and key in prev.get("variant_ids", {}):
            new_ids = prev["variant_ids"][key]
            nl2, nt2 = prev["per_sample_nll"][key]
        else:
            torch.manual_seed(a.seed + 1000 * m)
            _, new_ids = lm.restart(codes, ids, m, reencode=reencode)
            nl2, nt2 = lm.nll(new_ids)
        r = summarize(new_ids, nl2, nt2)
        r["extra_nfe"] = dec.K - m
        res["vcycle"][key] = r
        per_sample[key] = (nl2, nt2)
        variant_ids[key] = new_ids
        print(f"{label:12s} m={m}: {r}", flush=True)
        checkpoint()
