"""Error diffusion within a level (Floyd & Steinberg 1976) for unconditional NCM text: LevelDecoder.sample_ed.

Stage 1 (single pass, 12 NFE, no V-cycle): TinyStories and LM1B-small, seeds x n samples per arm. Arms: the
reference (T = 1, gamma = 0), three lower temperatures (the temperature frontier) and gamma x rho. Every arm samples
with Gumbel-max noise drawn per level from the same seed, so arms are coupled sample by sample and gamma = 0 at T = 1
is an exact sample of the released model. Pass bar (decided in advance):
a setting passes on a task if the pooled paired-bootstrap perplexity ratio vs the reference has its 95% CI below 0.97
with at most 1% loss of corpus distinct-2, or if it lies beyond the temperature frontier (lower perplexity than the
frontier at its distinct-2) on both tasks.

Stage 2 (LM1B-medium, inside the V-cycle from level 6): reference, error diffusion alone, V-cycle, V-cycle with error
diffusion in the regenerated levels, and error diffusion in both the draft and the cycle. The setting is the best one
that passed stage 1 (lowest mean log ratio over the two tasks); stage 2 is skipped if none passed.

python experiments/ed_sweep.py --stage 1
python experiments/ed_sweep.py --stage 2
"""
import argparse
import json
import math
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
ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--tasks", default="tinystories,lm1b_small")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--gammas", default="0.5,1,2")
ap.add_argument("--rhos", default="0.5,1")
ap.add_argument("--temps", default="0.95,0.9,0.85")
ap.add_argument("--m", type=int, default=6, help="stage 2: V-cycle restart level")
ap.add_argument("--ed", default="", help="stage 2: gamma,rho (default: best passing setting of stage 1)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.set_grad_enabled(False)
N_BOOT = 1000


def setup(task):
    ck = os.path.join(ARTIFACTS, "lm1b", task.split("_")[1]) if task.startswith("lm1b") else os.path.join(ARTIFACTS, task)
    ncm = load(ck, device=device())
    dec = LevelDecoder(ncm)
    lm = LM1B(ncm, dec, ck, bs=a.bs)
    # The released sampler path is untouched: ed=None must still reproduce generate() exactly.
    torch.manual_seed(1234)
    ref = ncm.net.generate(lm.sample_l0(8), lm.prompt(8), temperature=lm.temp, top_k=0, top_p=1.0, cfg_scale=0.0,
                           cfg_null_eos=lm.eos)
    torch.manual_seed(1234)
    l0 = lm.sample_l0(8)
    mine = torch.cat(dec.run(dec.prefix(lm.prompt(8)), start=1, codes=[l0], temperature=lm.temp)[0], 1)
    assert torch.equal(ref, mine), f"LevelDecoder differs from generate() on {task}"
    print(f"[{task}] LevelDecoder == generate() on 8 samples", flush=True)
    return ncm, dec, lm


def measure(lm, ids):
    nl, nt = lm.nll(ids)
    q = lm.quality(ids)
    r = lm.corpus(nl, nt, q)
    r["corpus_distinct2"] = corpus_distinct(ids, lm.pad, 2, lm.cut_eos)
    return r, nl, nt


def ratio_ci(nl_a, nt_a, nl_b, nt_b, seed=0):
    """Pooled perplexity ratio a / b with a paired bootstrap over samples."""
    f = lambda i: math.exp(nl_a[i].sum() / nt_a[i].sum() - nl_b[i].sum() / nt_b[i].sum())
    rng = np.random.default_rng(seed)
    n = len(nl_a)
    boots = [f(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return f(np.arange(n)), [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def compare(per, ref, arms, seeds):
    """Per arm vs ref: pooled ratio + CI, mean relative change of corpus distinct-2 over seeds, other metrics."""
    out = {}
    cat = lambda arm, j: np.concatenate([per[arm][s][j] for s in seeds])
    for arm in arms:
        r, ci = ratio_ci(cat(arm, 1), cat(arm, 2), cat(ref, 1), cat(ref, 2))
        d2 = float(np.mean([per[arm][s][0]["corpus_distinct2"] / per[ref][s][0]["corpus_distinct2"] - 1 for s in seeds]))
        avg = lambda key: float(np.mean([per[arm][s][0][key] for s in seeds]))
        out[arm] = {"ppl_ratio": r, "ci": ci, "corpus_d2_change": d2,
                    **{k: avg(k) for k in ("gen_ppl", "corpus_distinct2", "distinct2", "entropy", "adjacent_repeat")}}
    return out


def frontier_ratio(cmp, temp_arms, d2):
    """Perplexity ratio of the temperature frontier at corpus-d2 change d2 (linear in log ratio; clamped)."""
    pts = sorted([(cmp[t]["corpus_d2_change"], math.log(cmp[t]["ppl_ratio"])) for t in temp_arms] + [(0.0, 0.0)])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return math.exp(float(np.interp(d2, xs, ys)))


def stage1():
    seeds = [int(s) for s in a.seeds.split(",")]
    eds = [(float(g), float(r)) for g in a.gammas.split(",") for r in a.rhos.split(",")]
    temps = [float(t) for t in a.temps.split(",")]
    arms = {"T=1": (1.0, (0.0, 0.0)), **{f"T={t}": (t, (0.0, 0.0)) for t in temps},
            **{f"g={g},r={r}": (1.0, (g, r)) for g, r in eds}}
    res = {"seeds": seeds, "n": a.n, "arms": {k: {"temperature": v[0], "gamma": v[1][0], "rho": v[1][1]} for k, v in arms.items()},
           "tasks": {}}
    for task in a.tasks.split(","):
        ncm, dec, lm = setup(task)
        per = {arm: {} for arm in arms}
        secs = {}
        for s in seeds:
            for arm, (temp, ed) in arms.items():
                with Timer() as t:
                    _, ids = lm.generate(a.n, s, temperature=temp * lm.temp, ed=ed)
                secs[arm] = secs.get(arm, 0.0) + t.s
                per[arm][s] = measure(lm, ids)
                r = per[arm][s][0]
                print(f"[{task} s{s}] {arm:14s} ppl {r['gen_ppl']:.2f} cd2 {r['corpus_distinct2']:.4f} "
                      f"rep {r['adjacent_repeat']:.4f} ({t.s:.0f}s)", flush=True)
        cmp = compare(per, "T=1", [k for k in arms if k != "T=1"], seeds)
        temp_arms = [f"T={t}" for t in temps]
        for arm, c in cmp.items():
            c["seconds_generate"] = secs[arm]
            if arm in temp_arms:
                continue
            c["strict_pass"] = bool(c["ci"][1] < 0.97 and c["corpus_d2_change"] >= -0.01)
            c["frontier_ratio"] = frontier_ratio(cmp, temp_arms, c["corpus_d2_change"])
            c["beyond_frontier"] = bool(c["ppl_ratio"] < c["frontier_ratio"])
        res["tasks"][task] = {"reference": {k: float(np.mean([per["T=1"][s][0][k] for s in seeds]))
                                            for k in ("gen_ppl", "corpus_distinct2", "distinct2", "entropy", "adjacent_repeat")},
                              "seconds_generate_reference": secs["T=1"], "arms": cmp}
        save_json(res, f"ed_stage1{a.tag}.json")
        del ncm, dec, lm
        torch.cuda.empty_cache()
    # Decision for stage 2, by the rule fixed in advance.
    tasks = list(res["tasks"])
    cands = []
    for arm in arms:
        if not arm.startswith("g="):
            continue
        cs = [res["tasks"][t]["arms"][arm] for t in tasks]
        if any(c["strict_pass"] for c in cs) or all(c["beyond_frontier"] for c in cs):
            cands.append((float(np.mean([math.log(c["ppl_ratio"]) for c in cs])), arm))
    res["passing"] = [arm for _, arm in sorted(cands)]
    res["best"] = sorted(cands)[0][1] if cands else None
    save_json(res, f"ed_stage1{a.tag}.json")
    print("passing:", res["passing"], "best:", res["best"], flush=True)


def stage2():
    if a.ed:
        g, r = (float(v) for v in a.ed.split(","))
    else:
        s1 = json.load(open(os.path.join(RESULTS, f"ed_stage1{a.tag}.json")))
        if not s1.get("best"):
            print("stage 2 skipped: no setting passed stage 1", flush=True)
            save_json({"skipped": "no setting passed stage 1"}, f"ed_stage2{a.tag}.json")
            return
        g, r = s1["arms"][s1["best"]]["gamma"], s1["arms"][s1["best"]]["rho"]
    ed, off = (g, r), (0.0, 0.0)
    seeds = [int(s) for s in a.seeds.split(",")]
    ncm, dec, lm = setup("lm1b_medium")
    names = ["reference", "ed", f"vcycle|m={a.m}", f"vcycle+ed|m={a.m}", f"ed+vcycle+ed|m={a.m}"]
    per = {k: {} for k in names}
    for s in seeds:
        codes, ids = lm.generate(a.n, s, ed=off)
        codes_e, ids_e = lm.generate(a.n, s, ed=ed)
        out = {"reference": ids, "ed": ids_e}
        for name, (c0, i0, e) in {f"vcycle|m={a.m}": (codes, ids, off), f"vcycle+ed|m={a.m}": (codes, ids, ed),
                                  f"ed+vcycle+ed|m={a.m}": (codes_e, ids_e, ed)}.items():
            torch.manual_seed(s + 1000 * a.m)
            out[name] = lm.restart(c0, i0, a.m, ed=e)[1]
        for name in names:
            per[name][s] = measure(lm, out[name])
            rr = per[name][s][0]
            print(f"[medium s{s}] {name:18s} ppl {rr['gen_ppl']:.2f} cd2 {rr['corpus_distinct2']:.4f}", flush=True)
    res = {"gamma": g, "rho": r, "m": a.m, "seeds": seeds, "n": a.n,
           "vs_reference": compare(per, "reference", names[1:], seeds),
           "vs_vcycle": compare(per, f"vcycle|m={a.m}", names[3:], seeds)}
    v = res["vs_vcycle"][f"vcycle+ed|m={a.m}"]
    res["adds_to_vcycle"] = bool(v["ci"][1] < 1.0 and v["corpus_d2_change"] >= -0.01)
    save_json(res, f"ed_stage2{a.tag}.json")
    print(json.dumps(res, indent=1), flush=True)


stage1() if a.stage == 1 else stage2()
