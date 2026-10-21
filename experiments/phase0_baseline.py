"""Phase 0: reproduce NCM Sudoku accuracy and check LevelDecoder against the released generate().

python experiments/phase0_baseline.py --difficulty hard --n 2000
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import sudoku_pass
from vcycle.common import ARTIFACTS, DATA, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=125)
ap.add_argument("--check", type=int, default=500, help="boards on which to compare against net.generate")
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
dec = LevelDecoder(ncm)
puzzles, solutions = sudoku.load_split(DATA, a.difficulty, a.n)

# 1. LevelDecoder (cached prompt keys/values) reproduces generate() code for code.
n_check = min(a.check, a.n)
ref, mine = [], []
for sl in batches(n_check, a.bs):
    p = puzzles[sl].to(dev)
    ref.append(ncm.net.generate(None, p, temperature=0.0, top_k=0, top_p=1.0))
    mine.append(torch.cat(dec.run(dec.prefix(p))[0], 1))
ref, mine = torch.cat(ref), torch.cat(mine)
assert torch.equal(ref, mine), f"code mismatch on {(ref != mine).any(1).sum().item()} boards"
print(f"LevelDecoder == generate() on {n_check} boards")

# 2. Greedy baseline (the paper's setting: temperature 0, 12 levels = 12 NFE), cached for later phases.
base = sudoku_pass(dec, puzzles, a.bs, name=f"sudoku_{a.difficulty}_greedy_n{a.n}")
m = sudoku.metrics(base["tokens"].long(), puzzles, solutions)
res = {
    "difficulty": a.difficulty, "n": a.n, "device": dev, "checked_against_generate": n_check,
    "solved": m["solved"].float().mean().item(), "cell_acc": m["cell_acc"].mean().item(),
    "valid": m["valid"].float().mean().item(),
    "mean_unit_violations": m["unit_violations"].float().mean().item(),
    "boards_with_clue_violations": (m["clue_violations"] > 0).float().mean().item(),
    "nfe_per_board": dec.K,
    "seconds_per_board": base["seconds"] / a.n,
}
print(res)
save_json(res, f"phase0_sudoku_{a.difficulty}.json")
