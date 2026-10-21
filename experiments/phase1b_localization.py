"""Phase 1b: does the residual point at *where* a board is wrong?

Level-k position i summarizes board span [i*L/l_k, (i+1)*L/l_k) (area pooling). On wrong boards we
ask, per level, whether positions whose re-encoded code disagrees cover more wrong cells than
positions that agree (position-level AUROC and lift). Uses the cached greedy pass; no generation.

python experiments/phase1b_localization.py --difficulty hard
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import split_levels, sudoku_pass
from vcycle.common import ARTIFACTS, DATA, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--bs", type=int, default=250)
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
dec = LevelDecoder(ncm)
puzzles, solutions = sudoku.load_split(DATA, a.difficulty, 2000)
base = sudoku_pass(dec, puzzles, 125, name=f"sudoku_{a.difficulty}_greedy_n2000")
toks = base["tokens"].long()
codes = split_levels(dec, base["codes"])
wrong_cell = toks != solutions  # (N, 89) board positions that are wrong
wrong = wrong_cell.any(1)
L = dec.levels[-1]

res = {"difficulty": a.difficulty, "n_wrong_boards": int(wrong.sum()), "levels": dec.levels, "per_level": []}
dis_rt = [[] for _ in range(dec.K)]
dis_loc = [[] for _ in range(dec.K)]
for sl in batches(len(toks), a.bs):
    x = toks[sl].to(dev)
    c = [ck[sl].to(dev) for ck in codes]
    r1, r2 = dec.encode(x), dec.encode_given_coarse(x, c)
    for k in range(dec.K):
        dis_rt[k].append((r1[k] != c[k]).cpu())
        dis_loc[k].append((r2[k] != c[k]).cpu())

for k, l in enumerate(dec.levels):
    # fraction of wrong cells under each level-k position (area pooling over its span)
    span_err = torch.nn.functional.adaptive_avg_pool1d(wrong_cell.float()[:, None, :], l)[:, 0, :]  # (N, l)
    row = {"level": k, "length": l}
    for name, d in (("roundtrip", dis_rt), ("local", dis_loc)):
        dk = torch.cat(d[k])[wrong]  # (W, l)
        e = span_err[wrong]
        has_err = e > 0
        pos, neg = dk[has_err], dk[~has_err]
        row[name] = {
            "P(disagree | span has error)": pos.float().mean().item() if pos.numel() else None,
            "P(disagree | span clean)": neg.float().mean().item() if neg.numel() else None,
            "P(span has error | disagree)": has_err[dk].float().mean().item() if dk.any() else None,
            "P(span has error)": has_err.float().mean().item(),
        }
    res["per_level"].append(row)
    r = row["local"]
    fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
    print(f"level {k:2d} (l={l:2d}) local: P(dis|err) {fmt(r['P(disagree | span has error)'])} "
          f"P(dis|clean) {fmt(r['P(disagree | span clean)'])} "
          f"P(err|dis) {fmt(r['P(span has error | disagree)'])} base {fmt(r['P(span has error)'])}")
save_json(res, f"phase1b_localization_{a.difficulty}.json")
