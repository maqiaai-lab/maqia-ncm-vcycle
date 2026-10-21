"""Phase 3: compute-matched controls with a stochastic correction budget.

Every method starts from the greedy NCM pass (12 NFE). Boards the Sudoku checker rejects (a free,
solution-blind test at inference time) get up to R further rounds, at temperature T, until a round
yields a valid board. Methods differ only in what a round does:

  fresh            new full sample from scratch                                         12 NFE
  vcycle m         V-cycle: keep encode(x)_{<m} of the current board, resample m..K-1   K-m NFE
  vcycle-V2 m      as vcycle, but at level m only resample positions where r'_m != r_m  K-m NFE
  no-restrict m    keep the *generated* codes r_{<m} (no restriction), resample m..K-1  K-m NFE
  vcycle-tau       restart each board at its coarsest disagreeing level j (keep r'_{<j})
  vcycle-accept    restart each board at j+1 (keep r'_{<=j}: accept the restriction at j)

The checker accepts a board when it is a valid grid that keeps every clue; it never sees the solution.
Per-board outcomes (round first solved, NFE) go to results/cache for paired statistics.

Accuracy is reported against the cumulative mean NFE per board, so curves compare at equal compute.

python experiments/phase3_controls.py --difficulty hard --n 500 --rounds 8 --temp 1.0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import split_levels, sudoku_pass
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
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
ap.add_argument("--methods", default="fresh,vcycle,vcycle-V2,no-restrict,vcycle-tau,vcycle-accept")
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


RULES = {"vcycle-tau": ("tau", 0.0), "vcycle-accept": ("accept", 0.0)}
ADAPTIVE = tuple(RULES) + ("fresh",)


def checker(x):
    """Solution-blind acceptance test: a valid grid that keeps every clue."""
    mt = sudoku.metrics(x, puzzles, puzzles[:, 1:90])
    return mt["valid"] & (mt["clue_violations"] == 0)


def run_method(method, m):
    torch.manual_seed(a.seed)
    nfe = torch.full((a.n,), K, dtype=torch.long)
    solved_round = torch.where(solved0, 0, -1)
    xs, cs = x0.clone(), [c.clone() for c in codes0]
    curve = [{"round": 0, "nfe": float(K), "solved": solved0.float().mean().item()}]
    nfe_rounds = [nfe.clone()]
    with Timer() as t:
        for r in range(1, a.rounds + 1):
            todo = (~checker(xs)).nonzero().squeeze(1)
            for s in range(0, len(todo), a.bs):
                idx = todo[s:s + a.bs]
                pz, x, c = puzzles[idx].to(dev), xs[idx].to(dev), [ck[idx].to(dev) for ck in cs]
                pre = dec.prefix(pz)
                if method == "fresh":
                    c, z = dec.run(pre, temperature=a.temp)
                    x, n = dec.decode(z)[0], torch.full((len(idx),), K)
                else:
                    c, x, _, n = cycle(dec, pre, c, x, RULES.get(method, ("fixed", m)), temperature=a.temp,
                                       puzzles=pz, reencode=method != "no-restrict", masked=method == "vcycle-V2")
                xs[idx] = x.cpu()
                for k in range(K):
                    cs[k][idx] = c[k].cpu()
                nfe[idx] += n.cpu()
            solved = sudoku.metrics(xs, puzzles, solutions)["solved"]
            solved_round = torch.where((solved_round < 0) & solved, r, solved_round)
            nfe_rounds.append(nfe.clone())
            curve.append({"round": r, "nfe": nfe.float().mean().item(), "solved": solved.float().mean().item()})
    print(f"{method:13s} m={m}: " + " ".join(f"{p['solved']:.3f}@{p['nfe']:.1f}" for p in curve) + f" [{t.s:.0f}s]",
          flush=True)
    per_board[f"{method}|m={m}"] = {"solved_round": solved_round.short(), "nfe": torch.stack(nfe_rounds).short()}
    return {"method": method, "m": m, "temperature": a.temp, "seed": a.seed, "curve": curve, "seconds": t.s}


tag = a.tag or f"{a.difficulty}_n{a.n}_T{a.temp}_s{a.seed}"
out = {"difficulty": a.difficulty, "n": a.n, "temperature": a.temp, "rounds": a.rounds, "seed": a.seed,
       "baseline_solved": solved0.float().mean().item(), "runs": []}
per_board = {"baseline_solved": solved0}
# Resume: every run reseeds at its start, so runs finished by an interrupted job are reused as they are.
json_path, pb_path = os.path.join(RESULTS, f"phase3_sudoku_{tag}.json"), os.path.join(RESULTS, "cache", f"phase3_perboard_{tag}.pt")
if os.path.exists(json_path) and os.path.exists(pb_path):
    old, old_pb = json.load(open(json_path)), torch.load(pb_path, weights_only=False)
    for run in old["runs"]:
        if f"{run['method']}|m={run['m']}" in old_pb and run.get("rounds", a.rounds) == a.rounds:
            out["runs"].append(run)
            per_board[f"{run['method']}|m={run['m']}"] = old_pb[f"{run['method']}|m={run['m']}"]
    print(f"resuming {tag}: {len(out['runs'])} runs already done", flush=True)
done = {(r["method"], r["m"]) for r in out["runs"]}
for method in a.methods.split(","):
    for m in ([0] if method in ADAPTIVE else [int(v) for v in a.levels.split(",")]):
        if (method, m) in done:
            continue
        out["runs"].append(run_method(method, m))
        save_json(out, f"phase3_sudoku_{tag}.json")
        torch.save(per_board, os.path.join(RESULTS, "cache", f"phase3_perboard_{tag}.pt"))
