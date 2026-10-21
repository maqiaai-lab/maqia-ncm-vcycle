"""Check that the cached-prompt path gives the same codes as NextConceptTransformer.generate.

python experiments/check_prefix_cache.py --difficulty hard --n 250
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.common import ARTIFACTS, DATA, Timer, batches, device
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--bs", type=int, default=125)
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
cached, plain = LevelDecoder(ncm, cache_prefix=True), LevelDecoder(ncm, cache_prefix=False)
puzzles, _ = sudoku.load_split(DATA, a.difficulty, a.n)

ref, mine, gaps = [], [], []
with Timer() as t_ref:
    for sl in batches(a.n, a.bs):
        ref.append(ncm.net.generate(None, puzzles[sl].to(dev), temperature=0.0, top_k=0, top_p=1.0))
with Timer() as t_mine:
    for sl in batches(a.n, a.bs):
        mine.append(torch.cat(cached.run(cached.prefix(puzzles[sl].to(dev)))[0], 1))
ref, mine = torch.cat(ref), torch.cat(mine)

# Logit gap between the two paths at every level, on the reference codes.
p = puzzles[:a.bs].to(dev)
pc, pp = cached.prefix(p), plain.prefix(p)
codes = list(ref[:a.bs].split(cached.levels, 1))
z = torch.zeros(a.bs, cached.levels[-1], cached.vq.embed_dim, device=dev)
for k in range(cached.K):
    gaps.append((cached.logits(k, pc, z) - plain.logits(k, pp, z)).abs().max().item())
    z = z + cached.recon(k, codes[k])

n_diff = (ref != mine).any(1).sum().item()
print(f"boards with any code difference: {n_diff}/{a.n}")
print(f"max |logit gap| per level: {[f'{g:.1e}' for g in gaps]}")
print(f"generate(): {t_ref.s:.1f}s   cached LevelDecoder: {t_mine.s:.1f}s   speedup {t_ref.s / t_mine.s:.2f}x")
