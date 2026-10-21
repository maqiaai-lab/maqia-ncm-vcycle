"""Phase 1: is the tokenizer round-trip residual an error signal?

A sample is self-consistent (a fixed point) when encode(decode(r)) == r. For every greedy baseline
board we measure per-level agreement between the generated codes r and
  * r'  = encode(x)                      (plain round trip; a level-j miss cascades to finer levels)
  * r'' = encode_given_coarse(x, r)      (level-local: residual taken against the generated coarse levels)
and ask whether disagreement separates wrong boards from solved ones (AUROC). Decoder and generator
confidences are reported as the fallback residual from the plan.

python experiments/phase1_fixed_point.py --difficulty hard
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import split_levels, sudoku_pass
from vcycle.common import ARTIFACTS, DATA, RESULTS, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load


def auroc(score, positive):
    """P(score of a random positive > score of a random negative), ties count half."""
    s, y = score.double(), positive.bool()
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).double().mean()
    eq = (pos[:, None] == neg[None, :]).double().mean()
    return (gt + 0.5 * eq).item()


ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=125)
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
dec = LevelDecoder(ncm)
puzzles, solutions = sudoku.load_split(DATA, a.difficulty, a.n)
base = sudoku_pass(dec, puzzles, a.bs, name=f"sudoku_{a.difficulty}_greedy_n{a.n}")
toks = base["tokens"].long()
m = sudoku.metrics(toks, puzzles, solutions)
wrong = ~m["solved"]
print(f"{a.difficulty}: solved {m['solved'].float().mean():.4f} on {a.n} boards ({base['seconds']:.0f}s)")

codes = split_levels(dec, base["codes"])
rt, loc, rt_proj = [[] for _ in range(dec.K)], [[] for _ in range(dec.K)], [[] for _ in range(dec.K)]
roundtrip_tokens = []
for sl in batches(a.n, a.bs):
    x = toks[sl].to(dev)
    c = [ck[sl].to(dev) for ck in codes]
    r1 = dec.encode(x)
    r2 = dec.encode_given_coarse(x, c)
    r3 = dec.encode(sudoku.project(x, puzzles[sl].to(dev)))
    roundtrip_tokens.append(dec.decode(dec.z_from_codes(r1))[0].cpu())
    for k in range(dec.K):
        rt[k].append((r1[k] != c[k]).cpu())
        loc[k].append((r2[k] != c[k]).cpu())
        rt_proj[k].append((r3[k] != c[k]).cpu())
rt = [torch.cat(v) for v in rt]
loc = [torch.cat(v) for v in loc]
rt_proj = [torch.cat(v) for v in rt_proj]
roundtrip_tokens = torch.cat(roundtrip_tokens)

levels = dec.levels
res = {"difficulty": a.difficulty, "n": a.n, "levels": levels,
       "solved": m["solved"].float().mean().item(), "n_wrong": int(wrong.sum())}

# Fixed-point rates.
fp = torch.stack([d.any(1) for d in rt], 1).any(1).logical_not()
res["fixed_point_rate"] = fp.float().mean().item()
res["fixed_point_rate_solved"] = fp[~wrong].float().mean().item()
res["fixed_point_rate_wrong"] = fp[wrong].float().mean().item() if wrong.any() else None
res["roundtrip_changes_tokens"] = (roundtrip_tokens != toks).any(1).float().mean().item()
res["projection_changes_codes"] = any(bool((p != r).any()) for p, r in zip(rt_proj, rt))


def first_level(d):
    anyk = torch.stack([v.any(1) for v in d], 1)  # (N, K)
    idx = torch.where(anyk.any(1), anyk.float().argmax(1), torch.full((anyk.shape[0],), dec.K))
    return idx


for name, d in (("roundtrip", rt), ("local", loc)):
    per_level = {
        "disagree_rate_solved": [v[~wrong].float().mean().item() for v in d],
        "disagree_rate_wrong": [v[wrong].float().mean().item() if wrong.any() else None for v in d],
    }
    fl = first_level(d)
    per_level["first_level_hist_solved"] = torch.bincount(fl[~wrong], minlength=dec.K + 1).tolist()
    per_level["first_level_hist_wrong"] = torch.bincount(fl[wrong], minlength=dec.K + 1).tolist()
    count = torch.stack([v.float().sum(1) for v in d], 1)  # (N, K) disagreeing positions per level
    frac = torch.stack([v.float().mean(1) for v in d], 1)
    per_level["auroc"] = {
        "any_disagreement": auroc(count.sum(1) > 0, wrong),
        "n_disagree_positions": auroc(count.sum(1), wrong),
        "mean_level_disagree_frac": auroc(frac.mean(1), wrong),
        "coarseness(K - first level)": auroc(dec.K - fl, wrong),
        "per_level_frac": [auroc(frac[:, k], wrong) for k in range(dec.K)],
    }
    res[name] = per_level

gconf = base["gen_conf"].float()
dconf = base["dec_conf"].float()
gl = list(gconf.split(levels, 1))
res["confidence_auroc"] = {
    "1 - min decoder prob": auroc(1 - dconf.min(1).values, wrong),
    "1 - mean decoder prob": auroc(1 - dconf.mean(1), wrong),
    "1 - min generator prob (all levels)": auroc(1 - gconf.min(1).values, wrong),
    "1 - min generator prob per level": [auroc(1 - g.min(1).values, wrong) for g in gl],
    "unit violations (verifier)": auroc(m["unit_violations"].float(), wrong),
}
res["mean_min_gen_prob_per_level"] = {
    "solved": [g[~wrong].min(1).values.mean().item() for g in gl],
    "wrong": [g[wrong].min(1).values.mean().item() if wrong.any() else None for g in gl],
}
print(f"fixed-point rate: solved {res['fixed_point_rate_solved']:.3f} wrong {res['fixed_point_rate_wrong']}")
print("roundtrip AUROC", res["roundtrip"]["auroc"]["n_disagree_positions"],
      "local AUROC", res["local"]["auroc"]["n_disagree_positions"])
print("confidence AUROC", {k: v for k, v in res["confidence_auroc"].items() if not isinstance(v, list)})
save_json(res, f"phase1_sudoku_{a.difficulty}.json")
torch.save({"roundtrip_count": torch.stack([v.sum(1) for v in rt], 1).short(),
            "local_count": torch.stack([v.sum(1) for v in loc], 1).short(),
            "dec_conf_mean": dconf.mean(1), "solved": m["solved"]},
           os.path.join(RESULTS, "cache", f"phase1_scores_{a.difficulty}.pt"))
