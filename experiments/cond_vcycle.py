"""Prompted tasks (XSum, WMT14): does a V-cycle improve NCM's outputs when the prompt pins the content?

Single pass exactly as the released evaluation runs it (both released configs sample the levels greedily, top_k = 1).
V-cycle from level m: re-encode the output with the frozen tokenizer, keep encode(x)_{<m}, regenerate levels m..K-1
with the same prompt and settings, pick tokens again. The output is re-encoded in canonical form (text up to the first
EOS, then EOS, then PAD), as the tokenizer saw text in training; --raw_restrict re-encodes the raw picked ids. Under greedy decoding, regenerating from the *generated*
r_{<m} (no restriction) reproduces the single pass exactly, so the single pass is itself the equal-compute control;
the script checks that it is reproduced. Repeated cycles are a deterministic map under greedy decoding: the
fixed-point rate is the share of documents a cycle leaves unchanged. With --top_k 0 (sampling), no restriction is a
real control and is scored. Also: the single pass's round-trip residual as a per-document error detector.

Metrics as scripts/eval_merge.py: XSum ROUGE-1/2/L (+ GPT-2 gen. PPL, words); WMT14 BLEU, lowercased BLEU, chrF,
length ratio. Paired bootstrap over documents for every comparison with the single pass. Resumable.

python experiments/cond_vcycle.py --task xsum --levels 3,4,5,6 --cycles 3
python experiments/cond_vcycle.py --task wmt14 --levels 3,5,7 --cycles 3
"""
import argparse
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
from vcycle.cond import Conditional, bleu_from_stats
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import auroc, spearman
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="xsum or wmt14")
ap.add_argument("--n", type=int, default=0, help="documents (0 = the config's n_docs)")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--levels", default="3,5", help="restart levels m")
ap.add_argument("--cycles", type=int, default=3, help="V-cycles per restart level")
ap.add_argument("--check", type=int, default=64, help="documents to compare against net.generate")
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None, help="override the config (0 = sample from the full distribution)")
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--uncached", action="store_true", help="recompute the prompt at every level (slow; for checking)")
ap.add_argument("--raw_restrict", action="store_true", help="re-encode the raw picked ids, junk after EOS included (ablation)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm, cache_prefix=not a.uncached, cache_padded=True)
task = Conditional(ncm, dec, ck, os.path.join(DATA, a.task), n=a.n, bs=a.bs, temperature=a.temp, top_k=a.top_k,
                   top_p=a.top_p, raw_restrict=a.raw_restrict)
greedy = task.top_k == 1 or task.temp == 0
idxs = list(range(len(task)))
N, K = len(idxs), dec.K
name = f"{a.task}_vcycle{a.tag}"
state_path = os.path.join(RESULTS, "cache", name + ".pt")
st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"variants": {}, "chain": {}}
out = st.get("out") or {"task": a.task, "n": N, "levels": dec.levels, "K": K, "device": device(),
                        "sampling": {"temperature": task.temp, "top_k": task.top_k, "top_p": task.top_p}, "greedy": greedy,
                        "prefix_cache": "uncached" if a.uncached else "cached (pad-aware)",
                        "restriction": "raw picked ids" if a.raw_restrict else "canonical: text, EOS, PAD", "variants": {}}
st["out"] = out
seed_of = lambda key: zlib.crc32(f"{name}|{key}".encode())


def save():
    torch.save(st, state_path)
    save_json(out, name + ".json")


# 1. LevelDecoder == generate() on the first documents, with the prompt cache and without it.
if "check" not in out:
    chk = idxs[:min(a.check, N)]
    torch.manual_seed(0)
    bad_cached = task.check(chk)
    task.dec = LevelDecoder(ncm, cache_prefix=False)
    torch.manual_seed(0)
    bad_uncached = task.check(chk)
    task.dec = dec
    out["check"] = {"docs": len(chk), "mismatch_cached": int(bad_cached.sum()), "mismatch_uncached": int(bad_uncached.sum())}
    print(f"LevelDecoder vs generate() on {len(chk)} {a.task} documents: {int(bad_cached.sum())} differ with the prompt "
          f"cache, {int(bad_uncached.sum())} without", flush=True)
    if greedy:
        assert bad_uncached.sum() == 0, "uncached LevelDecoder differs from generate()"
        assert bad_cached.float().mean() <= 0.02, "the pad-aware prompt cache differs from generate() too often"
    save()


def scored(key, ids, extra_nfe, seconds=None, ref_ids=None):
    per, corpus, hyps = task.score(ids)
    r = dict(corpus, extra_nfe=extra_nfe, seconds=seconds)
    if ref_ids is not None:
        r["changed_vs_prev"] = float((ids != ref_ids).any(1).float().mean())
    out["variants"][key] = r
    st["variants"][key] = {"ids": ids.to(torch.int32), "per": per}
    print(f"{key:22s} {({k: round(v, 3) for k, v in r.items() if isinstance(v, float)})}", flush=True)
    save()


def paired(key, base="baseline"):
    """Paired bootstrap over documents: key vs base."""
    pv, pb = st["variants"][key]["per"], st["variants"][base]["per"]
    idx = np.random.default_rng(seed_of(key)).integers(0, N, size=(1000, N))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {}
    if task.metric == "rouge":
        for k in ("rouge1", "rouge2", "rougeL"):
            d = pv[k] - pb[k]
            res[k] = [100 * float(d.mean())] + ci(100 * d[idx].mean(1))
        pp = lambda p, ix: np.exp(p["nll"][ix].sum(-1) / p["ntok"][ix].sum(-1))
        res["gen_ppl_ratio"] = [float(pp(pv, np.arange(N)) / pp(pb, np.arange(N)))] + ci(pp(pv, idx) / pp(pb, idx))
        res["words"] = [float((pv["words"] - pb["words"]).mean())] + ci((pv["words"] - pb["words"])[idx].mean(1))
    else:
        sv, sb = pv["bleu_stats"], pb["bleu_stats"]
        bv = np.array([bleu_from_stats(sv[ix].sum(0)) - bleu_from_stats(sb[ix].sum(0)) for ix in idx])
        res["bleu"] = [bleu_from_stats(sv.sum(0)) - bleu_from_stats(sb.sum(0))] + ci(bv)
        d = pv["chrf"] - pb["chrf"]
        res["chrf_sentence"] = [float(d.mean())] + ci(d[idx].mean(1))
    out["variants"][key]["vs_" + base] = res
    save()
    return res


# 2. Single pass.
if "baseline" not in st["variants"]:
    torch.manual_seed(0)
    with Timer() as t:
        codes, ids = task.generate(idxs)
    st["baseline_codes"] = [c.to(torch.int32) for c in codes]
    scored("baseline", ids, 0, t.s)
codes = [c.long() for c in st["baseline_codes"]]
ids = st["variants"]["baseline"]["ids"].long()

# 3. Residual of the single pass vs per-document quality (worst decile: lowest ROUGE-L / sentence chrF).
if "residual" not in out:
    rt, loc = task.residuals(codes, ids)
    q = st["variants"]["baseline"]["per"]["rougeL" if task.metric == "rouge" else "chrf"]
    worst = q <= np.quantile(q, 0.1)
    res = {"fixed_point_rate": float((rt.sum(1) == 0).float().mean()),
           "disagree_rate_per_level": {"roundtrip": rt.mean(0).tolist(), "local": loc.mean(0).tolist()}}
    for nm, s in (("roundtrip_total", rt.sum(1).numpy()), ("local_total", loc.sum(1).numpy())):
        res[nm] = {"spearman_with_quality": spearman(s, q), "auroc_worst_decile": auroc(s, worst)}
    out["residual"] = res
    st["residual"] = {"rt": rt, "loc": loc}
    print("residual:", {k: v for k, v in res.items() if k != "disagree_rate_per_level"}, flush=True)
    save()

# 4. Per restart level: no restriction (a check under greedy decoding, a control otherwise), then iterated V-cycles.
# Under greedy decoding the check is structural (same context, same argmax), so it runs at the first level only.
levels = [int(v) for v in a.levels.split(",") if v]
for m in levels:
    key = f"no-restrict|m={m}"
    if key not in st["variants"] and (not greedy or m == levels[0]):
        torch.manual_seed(seed_of(key))
        with Timer() as t:
            _, nr_ids = task.restart(codes, ids, idxs, m, reencode=False)
        same = float((nr_ids == ids).all(1).float().mean())
        if greedy:  # reproduces the single pass; scoring it again would only repeat the baseline row
            out["variants"][key] = {"identical_to_single_pass": same, "extra_nfe": K - m, "seconds": t.s}
            st["variants"][key] = {"identical": same}
            print(f"{key:22s} identical to the single pass on {100 * same:.2f}% of documents", flush=True)
            save()
        else:
            scored(key, nr_ids, K - m, t.s)
            paired(key)
    ch = st["chain"].get(m)
    if ch is None:
        cc, ci_, start = codes, ids, 1
        torch.manual_seed(seed_of(f"vcycle|m={m}"))
    else:
        cc, ci_, start = [c.long() for c in ch["codes"]], ch["ids"].long(), ch["cycle"] + 1
        torch.set_rng_state(ch["rng"])
        if ch["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ch["cuda_rng"])
    for i in range(start, a.cycles + 1):
        key = f"vcycle|m={m}|c={i}"
        with Timer() as t:
            cc, new_ids = task.restart(cc, ci_, idxs, m, reencode=True)
        scored(key, new_ids, i * (K - m), t.s, ref_ids=ci_)
        paired(key)
        if not greedy and i == 1:
            paired(key, f"no-restrict|m={m}")
        ci_ = new_ids
        st["chain"][m] = {"cycle": i, "codes": [c.to(torch.int32) for c in cc], "ids": ci_.to(torch.int32),
                          "rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        save()

# 5. The first documents' texts, kept in the JSON (the resumable state in results/cache/ is large and not committed).
ex = list(range(min(3, N)))
if "examples" not in out:
    src = [task.items[i] if isinstance(task.items[i], str) else task.tok.decode(list(task.items[i])[:120]) + " ..." for i in ex]
    out["examples"] = {"source": src, "reference": [task.refs[i] for i in ex]}
for key, v in st["variants"].items():
    if "ids" in v and "examples" not in out["variants"][key]:
        out["variants"][key]["examples"] = [task.text(r) for r in v["ids"][ex].long().tolist()]
save()
