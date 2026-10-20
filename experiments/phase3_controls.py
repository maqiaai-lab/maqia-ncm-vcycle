"""Phase 3: compute-matched controls with a stochastic correction budget.

Every method starts from the greedy NCM pass (12 NFE). Boards the Sudoku checker rejects (a free,
solution-blind test at inference time) get up to R further rounds, at temperature T, until a round
yields a valid board. Methods differ only in what a round does:

  fresh            new full sample from scratch                                         12 NFE
  vcycle m         V-cycle: keep encode(x)_{<m} of the current board, resample m..K-1   K-m NFE
  vcycle-V2 m      as vcycle, but at level m only resample positions where r'_m != r_m  K-m NFE
  no-restrict m    keep the *generated* codes r_{<m} (no restriction), resample m..K-1  K-m NFE

Accuracy is reported against the cumulative mean NFE per board, so curves compare at equal compute.

python experiments/phase3_controls.py --difficulty hard --n 500 --rounds 8 --temp 1.0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import split_levels, sudoku_pass
from vcycle.common import ARTIFACTS, DATA, Timer, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load
from vcycle.vcycle import cycle

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=125)
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--levels", default="1,3,5,7", help="restart levels m for the partial methods")
ap.add_argument("--methods", default="fresh,vcycle,vcycle-V2,no-restrict")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ncm = load(os.path.join(ARTIFACTS, "sudoku", a.difficulty), device=dev)
dec = LevelDecoder(ncm)
puzzles, solutions = sudoku.load_split(DATA, a.difficulty, 2000)
base = sudoku_pass(dec, puzzles, a.bs, name=f"sudoku_{a.difficulty}_greedy_n2000")
puzzles, solutions = puzzles[:a.n], solutions[:a.n]
codes0 = [c[:a.n] for c in split_levels(dec, base["codes"])]
x0 = base["tokens"][:a.n].long()
solved0 = sudoku.metrics(x0, puzzles, solutions)["solved"]
K = dec.K


def valid(x, pz):
    return sudoku.metrics(x.cpu(), pz.cpu(), pz.cpu()[:, 1:90])["valid"].to(x.device)


def run_method(method, m):
    torch.manual_seed(a.seed)
    nfe = torch.full((a.n,), K, dtype=torch.long)
    curve = [{"round": 0, "nfe": float(K), "solved": solved0.float().mean().item(),
              "valid": valid(x0, puzzles).float().mean().item()}]
    xs, cs = x0.clone(), [c.clone() for c in codes0]
    with Timer() as t:
        for r in range(1, a.rounds + 1):
            for sl in batches(a.n, a.bs):
                pz = puzzles[sl].to(dev)
                x, c = xs[sl].to(dev), [ck[sl].to(dev) for ck in cs]
                active = ~valid(x, pz)
                if not active.any():
                    continue
                pe = dec.prefix(pz)
                if method == "fresh":
                    idx = active.nonzero().squeeze(1)
                    new_c, z = dec.run(pe[idx], temperature=a.temp)
                    x[idx] = dec.decode(z)[0]
                    for k in range(K):
                        c[k][idx] = new_c[k]
                    n = torch.where(active, K, 0)
                else:
                    c, x, _, n = cycle(dec, pe, c, x, ("fixed", m), temperature=a.temp, puzzles=pz,
                                       reencode=method != "no-restrict", masked=method == "vcycle-V2",
                                       active=active)
                xs[sl] = x.cpu()
                for k in range(K):
                    cs[k][sl] = c[k].cpu()
                nfe[sl] += n.cpu()
            mt = sudoku.metrics(xs, puzzles, solutions)
            curve.append({"round": r, "nfe": nfe.float().mean().item(), "solved": mt["solved"].float().mean().item(),
                          "valid": mt["valid"].float().mean().item()})
    print(f"{method:12s} m={m}: " + " ".join(f"{p['solved']:.3f}@{p['nfe']:.1f}" for p in curve) + f" [{t.s:.0f}s]",
          flush=True)
    return {"method": method, "m": m, "temperature": a.temp, "curve": curve, "seconds": t.s}


out = {"difficulty": a.difficulty, "n": a.n, "temperature": a.temp, "rounds": a.rounds,
       "baseline_solved": solved0.float().mean().item(), "runs": []}
for method in a.methods.split(","):
    for m in ([0] if method == "fresh" else [int(v) for v in a.levels.split(",")]):
        out["runs"].append(run_method(method, m))
        save_json(out, f"phase3_sudoku_{a.tag or f'{a.difficulty}_n{a.n}_T{a.temp}'}.json")
