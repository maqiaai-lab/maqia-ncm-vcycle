"""LM1B: iterate the V-cycle towards convergence and track quality, diversity and drift after every cycle.

Starts from the baseline samples cached by lm1b_diagnostic.py (same size and seed). For each m:c pair it
re-encodes the newest sample, keeps encode(x)_{<m} and resamples levels m..K-1 at T=1, c times. After every
cycle it records:
- gen. PPL (per-sample NLL kept for paired bootstraps), entropy, per-sample and corpus distinct-n;
- the round-trip residual per level, the fixed-point rate, and the coarse fixed-point rate
  (encode(x)_{<m} == the r_{<m} the sample was generated from). Multigrid predicts the coarse codes settle;
- drift: token agreement and token-set Jaccard with the original draft and with the previous cycle.
Falling corpus distinct-2 with rising agreement between cycles would mean drift toward generic text.

One no-restriction run per m (keep the generated r_{<m}, resample once) is the control at the same level.
Repeating it only redraws the same conditional, so its expected PPL does not depend on c.
Resumable: codes, ids and the RNG state are saved after every cycle.

python experiments/lm1b_converge.py --size small --seed 42 --cycles 6:6,8:6
"""
import argparse
import os
import sys

import numpy as np
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
ap.add_argument("--cycles", default="6:6,8:6", help="m:c pairs: restart level and number of cycles")
ap.add_argument("--ref", default="openai-community/gpt2-large")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, "lm1b", a.size)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm)
lm = LM1B(ncm, dec, ck, ref=a.ref, bs=a.bs)
base = torch.load(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s{a.seed}.pt"), weights_only=False)
codes0, ids0 = base["codes"], base["ids"]

name = f"lm1b_converge_{a.size}_s{a.seed}"
state_path = os.path.join(RESULTS, "cache", name + ".pt")
if os.path.exists(state_path):
    st = torch.load(state_path, weights_only=False)
    print(f"resuming from {state_path}", flush=True)
else:
    st = {"out": {"size": a.size, "seed": a.seed, "n": len(ids0), "K": dec.K, "device": device(), "runs": {}},
          "per_sample_nll": {}, "ids": {}, "resume": {}}
out = st["out"]


def save():
    torch.save(st, state_path)
    save_json(out, name + ".json")


def overlap(x, y, pad):
    """Mean per-sample positional token agreement (ignoring positions that are [PAD] in both) and token-set Jaccard."""
    both_pad = (x == pad) & (y == pad)
    pos = ((x == y) & ~both_pad).sum(1).float() / (~both_pad).sum(1).clamp(min=1).float()
    jac = []
    for u, v in zip(x.tolist(), y.tolist()):
        su, sv = set(u) - {pad}, set(v) - {pad}
        jac.append(len(su & sv) / max(len(su | sv), 1))
    return float(pos.mean()), float(np.mean(jac))


def measure(codes, ids, m, prev_ids):
    nl, nt = lm.nll(ids)
    r = lm.corpus(nl, nt, lm.quality(ids))
    r["corpus_distinct2"], r["corpus_distinct3"] = corpus_distinct(ids, lm.pad, 2, lm.cut_eos), corpus_distinct(ids, lm.pad, 3, lm.cut_eos)
    rt, loc = lm.residuals(codes, ids)
    r["fixed_point_rate"] = float((rt.sum(1) == 0).float().mean())
    r["coarse_fixed_point_rate"] = float((rt[:, :m].sum(1) == 0).float().mean())
    r["roundtrip_disagree_per_level"] = rt.mean(0).tolist()
    r["local_disagree_per_level"] = loc.mean(0).tolist()
    r["agree_draft"], r["jaccard_draft"] = overlap(ids, ids0, lm.pad)
    r["agree_prev"], r["jaccard_prev"] = overlap(ids, prev_ids, lm.pad)
    return r, (nl, nt)


def show(tag, r):
    print(f"{tag:22s} ppl {r['gen_ppl']:6.1f}  H {r['entropy']:.3f}  corpus-d2 {r['corpus_distinct2']:.3f}  "
          f"fp {r['fixed_point_rate']:.3f}  coarse-fp {r['coarse_fixed_point_rate']:.3f}  "
          f"agree draft/prev {r['agree_draft']:.3f}/{r['agree_prev']:.3f}", flush=True)


with Timer() as t_all:
    for pair in [p for p in a.cycles.split(",") if p]:
        m, c = (int(v) for v in pair.split(":"))
        key = f"m={m}"
        run = out["runs"].setdefault(key, {"m": m, "extra_nfe_per_cycle": dec.K - m, "cycles": []})
        if "no_restrict" not in run:
            torch.manual_seed(a.seed + 1000 * m + 1)
            nr_codes, nr_ids = lm.restart(codes0, ids0, m, reencode=False)
            r, ps = measure(nr_codes, nr_ids, m, ids0)
            r["extra_nfe"] = dec.K - m
            run["no_restrict"] = r
            st["per_sample_nll"][f"no-restrict|{key}"], st["ids"][f"no-restrict|{key}"] = ps, nr_ids.to(torch.int16)
            show(f"no-restrict {key}", r)
            save()
        if not run["cycles"]:
            r, ps = measure(codes0, ids0, m, ids0)
            r.update(cycle=0, extra_nfe=0)
            run["cycles"].append(r)
            st["per_sample_nll"][f"{key}|c=0"] = ps
            show(f"{key} c=0", r)
            save()
        res = st["resume"].get(key)
        if res is None:
            torch.manual_seed(a.seed + 7000 + m)
            cc, ci = codes0, ids0
        else:
            cc, ci = [x.long() for x in res["codes"]], res["ids"].long()
            torch.set_rng_state(res["rng"])
            if res["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(res["cuda_rng"])
        for i in range(len(run["cycles"]), c + 1):
            prev = ci
            with Timer() as t:
                cc, ci = lm.restart(cc, ci, m, reencode=True)
            r, ps = measure(cc, ci, m, prev)
            r.update(cycle=i, extra_nfe=i * (dec.K - m), seconds_cycle=t.s)
            run["cycles"].append(r)
            st["per_sample_nll"][f"{key}|c={i}"], st["ids"][f"{key}|c={i}"] = ps, ci.to(torch.int16)
            st["resume"][key] = {"codes": [x.to(torch.int32) for x in cc], "ids": ci.to(torch.int16),
                                 "rng": torch.get_rng_state(),
                                 "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
            show(f"{key} c={i}", r)
            save()
out["seconds_last_invocation"] = t_all.s
save()
