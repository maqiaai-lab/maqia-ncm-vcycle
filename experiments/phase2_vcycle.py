"""Phase 2: greedy V-cycle variants on Sudoku (the paper's decoding setting, temperature 0).

Every config runs on all boards without a verifier. Greedy cycles are deterministic per board, so
verifier-gated and residual-gated numbers are computed afterwards from the per-board outcomes.

python experiments/phase2_vcycle.py --difficulty hard --n 500 --cycles 1
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle import sudoku
from vcycle.cache import split_levels, sudoku_pass
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.loading import load
from vcycle.vcycle import v_cycle_decode

CONFIGS = {
    # V1 as written in the plan: keep re-encoded levels < j, resample from j
    "V1 tau=0": dict(rule=("tau", 0.0)),
    "V1 tau=0.05": dict(rule=("tau", 0.05)),
    "V1 tau=0.1": dict(rule=("tau", 0.1)),
    "V1 tau=0.2": dict(rule=("tau", 0.2)),
    # V1 accepting the restriction at the first disagreeing level j (resample from j+1)
    "V1-accept tau=0": dict(rule=("accept", 0.0)),
    "V1-accept tau=0.1": dict(rule=("accept", 0.1)),
    # V2: masked resample at the restart level
    "V2-accept tau=0": dict(rule=("accept", 0.0), masked=True),
    # V3: pin clues + separators before re-encoding
    "V3 accept tau=0": dict(rule=("accept", 0.0), project=True),
    # V4: fixed restart level m
    **{f"V4 m={m}": dict(rule=("fixed", m)) for m in (1, 2, 3, 4, 5, 6, 8, 10)},
    # control: restart at m without the restriction step (keeps generated coarse codes)
    "no-restrict m=3": dict(rule=("fixed", 3), reencode=False),
}

ap = argparse.ArgumentParser()
ap.add_argument("--difficulty", default="hard")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=125)
ap.add_argument("--cycles", type=int, default=1)
ap.add_argument("--configs", default="", help="comma-separated subset of config names")
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
m0 = sudoku.metrics(x0, puzzles, solutions)
print(f"{a.difficulty} n={a.n}: baseline solved {m0['solved'].float().mean():.4f}", flush=True)

names = [s.strip() for s in a.configs.split(",")] if a.configs else list(CONFIGS)
tag = a.tag or f"{a.difficulty}_n{a.n}_c{a.cycles}"
json_path, pb_path = os.path.join(RESULTS, f"phase2_sudoku_{tag}.json"), os.path.join(RESULTS, "cache", f"phase2_perboard_{tag}.pt")
summary = {"difficulty": a.difficulty, "n": a.n, "baseline_solved": m0["solved"].float().mean().item(), "configs": {}}
per_board = {"baseline_solved": m0["solved"]}
# Resume: greedy cycles are deterministic, so configs finished by an interrupted job are reused as they are.
if os.path.exists(json_path) and os.path.exists(pb_path):
    summary["configs"] = json.load(open(json_path))["configs"]
    per_board = torch.load(pb_path, weights_only=False)
    print(f"resuming {tag}: {len(summary['configs'])} configs already done", flush=True)
for name in names:
    if name in summary["configs"]:
        continue
    kw = CONFIGS[name]
    solved = [[] for _ in range(a.cycles)]
    changed = [[] for _ in range(a.cycles)]
    nfe_c = [[] for _ in range(a.cycles)]
    with Timer() as t:
        for sl in batches(a.n, a.bs):
            pz = puzzles[sl].to(dev)
            pe = dec.prefix(pz)
            _, _, _, hist = v_cycle_decode(dec, pe, [c[sl].to(dev) for c in codes0], x0[sl].to(dev),
                                           n_cycles=a.cycles, puzzles=pz, **kw)
            for i, h in enumerate(hist):
                solved[i].append(sudoku.metrics(h["x"].cpu(), puzzles[sl], solutions[sl])["solved"])
                changed[i].append(h["changed"].cpu())
                nfe_c[i].append(dec.board_nfe(h["start"]).cpu())
    rows = []
    for i in range(a.cycles):
        s, ch, nf = torch.cat(solved[i]), torch.cat(changed[i]), torch.cat(nfe_c[i])
        per_board[f"{name}|c{i + 1}"] = s
        rows.append({
            "cycle": i + 1,
            "solved": s.float().mean().item(),
            "fixed": (~m0["solved"] & s).sum().item(),
            "broken": (m0["solved"] & ~s).sum().item(),
            "verifier_gated_solved": (m0["solved"] | s).float().mean().item(),
            "changed_this_cycle": ch.float().mean().item(),
            "extra_nfe_mean": nf.float().mean().item(),
            "extra_nfe_on_wrong": nf[~m0["solved"]].float().mean().item() if (~m0["solved"]).any() else 0.0,
        })
    summary["configs"][name] = {"seconds": t.s, "cycles": rows}
    r = rows[-1]
    print(f"{name:22s} solved {r['solved']:.4f} (fixed {r['fixed']}, broken {r['broken']}) "
          f"gated {r['verifier_gated_solved']:.4f} nfe+{r['extra_nfe_mean']:.2f} [{t.s:.0f}s]", flush=True)
    save_json(summary, f"phase2_sudoku_{tag}.json")
    torch.save(per_board, pb_path)
