"""LM1B: V-cycle vs no-restriction across restart levels, and iterated V-cycles.

Reuses the baseline samples cached by lm1b_diagnostic.py for the same size and seed. A V-cycle with
c cycles re-encodes the newest sample and restarts from level m again, c times (c * (K-m) extra NFE).
No-restriction keeps the generated r_{<m}; repeating it only redraws the same conditional, so it runs once.
--temps adds plain NCM samples at lower temperatures: lowering T also lowers perplexity, so these points
show whether the V-cycle beats simply sampling colder (perplexity at matched entropy).

python experiments/lm1b_sweep.py --size small --seed 42 --levels 4,6,8,10 --cycles 6:3,8:2
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, RESULTS, Timer, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import LM1B, corpus_distinct
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="small")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--levels", default="4,6,8,10", help="restart levels: one V-cycle and one no-restriction run each")
ap.add_argument("--cycles", default="6:3,8:2", help="m:c pairs for iterated V-cycles (reports every cycle up to c)")
ap.add_argument("--temps", default="0.8,0.9", help="baseline sampling temperatures for the PPL/entropy frontier")
ap.add_argument("--ref", default="openai-community/gpt2-large")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, "lm1b", a.size)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm)
lm = LM1B(ncm, dec, ck, ref=a.ref, bs=a.bs)
base = torch.load(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s{a.seed}.pt"), weights_only=False)
codes, ids = base["codes"], base["ids"]
out = {"size": a.size, "seed": a.seed, "n": len(ids), "variants": {}}
per_sample = {"baseline": base["per_sample_nll"]["baseline"]}
variant_ids = {}


def record(name, new_ids, extra_nfe):
    nl, nt = lm.nll(new_ids)
    r = lm.corpus(nl, nt, lm.quality(new_ids))
    r["corpus_distinct2"], r["corpus_distinct3"] = corpus_distinct(new_ids, lm.pad, 2), corpus_distinct(new_ids, lm.pad, 3)
    r["extra_nfe"] = extra_nfe
    out["variants"][name] = r
    per_sample[name], variant_ids[name] = (nl, nt), new_ids
    print(f"{name:20s} {r}", flush=True)
    save_json(out, f"lm1b_sweep_{a.size}_s{a.seed}.json")
    torch.save({"per_sample_nll": per_sample, "variant_ids": variant_ids},
               os.path.join(RESULTS, "cache", f"lm1b_sweep_{a.size}_s{a.seed}.pt"))


nb = base["ids"]
r0 = lm.corpus(*base["per_sample_nll"]["baseline"], lm.quality(nb))
r0["corpus_distinct2"], r0["corpus_distinct3"] = corpus_distinct(nb, lm.pad, 2), corpus_distinct(nb, lm.pad, 3)
r0["extra_nfe"] = 0
out["variants"]["baseline"] = r0

with Timer() as t:
    for m in [int(v) for v in a.levels.split(",") if v]:
        for label, reencode in (("vcycle", True), ("no-restrict", False)):
            torch.manual_seed(a.seed + 1000 * m + (0 if reencode else 1))
            _, new_ids = lm.restart(codes, ids, m, reencode=reencode)
            record(f"{label}|m={m}|c=1", new_ids, dec.K - m)
    for pair in [p for p in a.cycles.split(",") if p]:
        m, c = (int(v) for v in pair.split(":"))
        torch.manual_seed(a.seed + 7000 + m)
        cc, ci = codes, ids
        for i in range(1, c + 1):
            cc, ci = lm.restart(cc, ci, m, reencode=True)
            if i > 1:
                record(f"vcycle|m={m}|c={i}", ci, i * (dec.K - m))
    for T in [float(v) for v in a.temps.split(",") if v]:
        _, t_ids = lm.generate(len(ids), a.seed, temperature=T)
        record(f"baseline|T={T}", t_ids, 0)
out["seconds"] = t.s
save_json(out, f"lm1b_sweep_{a.size}_s{a.seed}.json")
