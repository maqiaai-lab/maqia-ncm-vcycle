"""Phase 0: reproduce NCM Sudoku accuracy and check LevelDecoder against the released generate().

python experiments/phase0_baseline.py --difficulty hard --n 2000
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.common import ARTIFACTS, DATA, Timer, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=125)
ap.add_argument("--check", type=int, default=250, help="boards on which to compare against net.generate")
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
dec = LevelDecoder(ncm)
puzzles, solutions = sudoku.load_split(DATA, a.difficulty, a.n)

# 1. LevelDecoder reproduces generate() code for code.
n_check = min(a.check, a.n)
p = puzzles[:n_check].to(dev)
ref = ncm.net.generate(None, p, temperature=0.0, top_k=0, top_p=1.0)
codes, _ = dec.run(dec.prefix(p))
mine = torch.cat(codes, 1)
assert torch.equal(ref, mine), f"code mismatch on {(ref != mine).any(1).sum().item()} boards"
print(f"LevelDecoder == generate() on {n_check} boards")

# 2. Greedy baseline (the paper's setting: temperature 0, 12 levels = 12 NFE).
toks = []
dec.nfe = 0
with Timer() as t:
    for sl in batches(a.n, a.bs):
        _, z = dec.run(dec.prefix(puzzles[sl].to(dev)))
        toks.append(dec.decode(z)[0].cpu())
toks = torch.cat(toks)
m = sudoku.metrics(toks, puzzles, solutions)
res = {
    "difficulty": a.difficulty, "n": a.n, "device": dev,
    "solved": m["solved"].float().mean().item(), "cell_acc": m["cell_acc"].mean().item(),
    "valid": m["valid"].float().mean().item(),
    "mean_unit_violations": m["unit_violations"].float().mean().item(),
    "boards_with_clue_violations": (m["clue_violations"] > 0).float().mean().item(),
    "nfe_per_board": dec.nfe / (a.n / a.bs) if a.n % a.bs == 0 else None,
    "seconds": t.s,
}
print(res)
save_json(res, f"phase0_sudoku_{a.difficulty}.json")
