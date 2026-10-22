"""Sudoku stacking pilot: soft-decision first pass -> singles propagation -> symmetry-twisted V-cycle retries.

Every arm starts with one 12-NFE pass. Boards the checker rejects (valid grid + keeps every clue; it never sees the
solution) first get naked/hidden-singles propagation from the clues (0 generator NFE; accepted only when it completes
the grid; SINGLES=0 disables it), then up to ROUNDS retries (12 NFE for a fresh restart, 12 - M for a V-cycle):

  N  greedy -> best-of-N: fresh restarts at T=1
  F  greedy -> fresh greedy restarts, each in a random symmetry frame g of the puzzle
  E  greedy -> V-cycle from level M at T=1 (identity frame; the retry method of experiments/phase3_controls.py)
  A  greedy -> twisted V-cycle: random symmetry g, encode(g(x)), keep levels < M, regenerate at T=0 on
     g(puzzle), map back with g^-1
  B  soft   -> twisted V-cycle (as A)
  D  soft   -> twisted V-cycle whose regeneration is also soft
  A0 / C are round 0 of A / B (12 NFE).

Soft-decision ("softdec"): at every level the generator reads the posterior-mean code embedding
sum_c softmax(logits / Ts)(c) e(c) instead of the argmax code, and the decoder reads the posterior-mean latent.
Ts is chosen from TS by the first-pass solved rate on boards 0..N/2-1 and reported on the rest (and on all boards).

Writes results/sudoku_stack_<diff>.json and results/cache/sudoku_stack_<diff>_perboard.pt.
  python experiments/sudoku_stack.py            (env: DIFF=hard N=2000 SEEDS=0,1,2 ROUNDS=32 M=5 TS=1,2,4,8)
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcycle import sudoku  # noqa: E402
from vcycle.common import ARTIFACTS, DATA, RESULTS, device  # noqa: E402
from vcycle.decoding import LevelDecoder  # noqa: E402
from vcycle.loading import load  # noqa: E402

torch.set_grad_enabled(False)
DIFF = os.environ.get("DIFF", "hard")
N = int(os.environ.get("N", "2000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "32"))
M = int(os.environ.get("M", "5"))
TS = [float(t) for t in os.environ.get("TS", "1,2,4,8").split(",")]
BS = int(os.environ.get("BS", "250"))
NBOOT = int(os.environ.get("NBOOT", "1000"))
SINGLES = os.environ.get("SINGLES", "1") == "1"
dev = device()
try:
    GPU_NAME = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
except (AssertionError, RuntimeError):
    GPU_NAME = "unknown"
ncm = load(os.path.join(ARTIFACTS, "sudoku", DIFF), device=dev)
dec = LevelDecoder(ncm)
K, LV = dec.K, dec.levels
puzzles, solutions = sudoku.load_split(DATA, DIFF, N)
CP = sudoku.CELL_POS
P_LEN = int(dec.net.prefix_len)
print(f"sudoku-{DIFF} N={N} device={dev} levels={LV} prompt={P_LEN}", flush=True)


# ---------------------------------------------------------------- symmetry group (as in the pilot sym_views.py) ------
def rand_line_perm(g):
    bands = torch.randperm(3, generator=g)
    return torch.cat([b * 3 + torch.randperm(3, generator=g) for b in bands])


def rand_elem(g):
    """Uniform element of the full group: digit relabelling, row/column permutations within bands/stacks,
    band/stack permutations, optional transpose."""
    sigma = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(9, generator=g)])  # blank stays 0
    rows, cols = rand_line_perm(g), rand_line_perm(g)
    return sigma, rows, cols, bool(torch.randint(2, (1,), generator=g).item())


def stack_elems(elems):
    return (torch.stack([e[0] for e in elems]), torch.stack([e[1] for e in elems]),
            torch.stack([e[2] for e in elems]), torch.tensor([e[3] for e in elems]))


def ident_el(n):
    return (torch.arange(10).expand(n, 10).clone(), torch.arange(9).expand(n, 9).clone(),
            torch.arange(9).expand(n, 9).clone(), torch.zeros(n, dtype=torch.bool))


def _grid(b):
    return b[:, CP].view(-1, 9, 9)


def _ungrid(G, like):
    out = like.clone()
    out[:, CP] = G.reshape(-1, 81)
    return out


def apply(board, el):
    s, r, c, t = el
    G, B = _grid(board), board.shape[0]
    G = torch.gather(G, 1, r[:, :, None].expand(B, 9, 9))
    G = torch.gather(G, 2, c[:, None, :].expand(B, 9, 9))
    G = torch.where(G <= 9, torch.gather(s, 1, G.clamp(max=9).reshape(B, 81)).view(B, 9, 9), G)
    return _ungrid(torch.where(t[:, None, None], G.transpose(1, 2), G), board)


def apply_inv(board, el):
    s, r, c, t = el
    G, B = _grid(board), board.shape[0]
    G = torch.where(t[:, None, None], G.transpose(1, 2), G)
    G = torch.where(G <= 9, torch.gather(torch.argsort(s, 1), 1, G.clamp(max=9).reshape(B, 81)).view(B, 9, 9), G)
    G = torch.gather(G, 1, torch.argsort(r, 1)[:, :, None].expand(B, 9, 9))
    G = torch.gather(G, 2, torch.argsort(c, 1)[:, None, :].expand(B, 9, 9))
    return _ungrid(G, board)


def puzzle_apply(pz, el):
    out = pz.clone()
    out[:, 1:90] = apply(pz[:, 1:90], el)
    return out


def checker(x, pz):
    mt = sudoku.metrics(x, pz, pz[:, 1:90])
    return mt["valid"] & (mt["clue_violations"] == 0)


_g = torch.Generator().manual_seed(123)
_el = stack_elems([rand_elem(_g) for _ in range(N)])
assert torch.equal(apply_inv(apply(solutions, _el), _el), solutions)
assert checker(apply(solutions, _el), puzzle_apply(puzzles, _el)).all(), "g(solution) must solve g(puzzle)"


# ---------------------------------------------------------------- singles propagation (0 generator NFE) ---------------
UNITS = sudoku.UNITS.tolist()
PEERS = [set() for _ in range(81)]
for u in UNITS:
    for i in u:
        PEERS[i] |= set(u)
for i in range(81):
    PEERS[i].discard(i)


def singles(g):
    g = list(g)
    cand = [set(range(1, 10)) if v == 0 else {v} for v in g]
    for i in range(81):
        if g[i]:
            for p in PEERS[i]:
                cand[p].discard(g[i])
    changed = True
    while changed:
        changed = False
        for i in range(81):
            if g[i] == 0 and len(cand[i]) == 1:
                v = next(iter(cand[i]))
                g[i], changed = v, True
                for p in PEERS[i]:
                    cand[p].discard(v)
        for u in UNITS:
            for v in range(1, 10):
                if any(g[i] == v for i in u):
                    continue
                spots = [i for i in u if g[i] == 0 and v in cand[i]]
                if len(spots) == 1:
                    i = spots[0]
                    g[i], cand[i], changed = v, {v}, True
                    for p in PEERS[i]:
                        cand[p].discard(v)
    return g


t0 = time.perf_counter()
sg = torch.tensor([singles(p) for p in puzzles[:, 1:90][:, CP].tolist()])
sing_board = _ungrid(sg.view(-1, 9, 9), puzzles[:, 1:90])
sing_ok = (sg > 0).all(1) & checker(sing_board, puzzles)  # complete and accepted by the checker
print(f"singles: completes {sing_ok.float().mean()*100:.2f}% of boards, correct on "
      f"{(sing_ok & (sing_board == solutions).all(1)).float().mean()*100:.2f}% [{time.perf_counter()-t0:.0f}s]",
      flush=True)


# ---------------------------------------------------------------- generation ------------------------------------------
def soft_recon(k, p):
    q = p @ dec.vq.embeddings[k].weight.float()
    if k != K - 1:
        q = dec.vq._up_to_full(q, k, LV[-1])
    return dec.vq._apply_quant_resi(q, k)


def generate(pz, el, start=0, codes=None, mode="hard", T=0.0, Ts=4.0):
    """Levels start..K-1 on g(puzzle) (levels < start from codes), decoded and mapped back with g^-1."""
    pf = dec.prefix(puzzle_apply(pz, el).to(dev))
    if mode == "hard":
        _, z = dec.run(pf, start=start, codes=codes, temperature=T)
    else:
        z = torch.zeros(pz.shape[0], LV[-1], dec.vq.embed_dim, device=dev)
        for k in range(start):
            z = z + dec.recon(k, codes[k])
        for k in range(start, K):
            z = z + soft_recon(k, (dec.logits(k, pf, z).float() / Ts).softmax(-1))
    return apply_inv(dec.decode(z)[0].cpu(), el)


def first_pass(mode, Ts=4.0):
    return torch.cat([generate(puzzles[s:s + BS], ident_el(min(BS, N - s)), mode=mode, Ts=Ts)
                      for s in range(0, N, BS)])


def retry(x, pz, el, m, mode, T, Ts):
    """One retry in frame el: m = 0 restarts from scratch on g(puzzle); m > 0 is a V-cycle (encode(g(x)), keep
    levels < m, regenerate m..K-1)."""
    if m == 0:
        return generate(pz, el, mode=mode, T=T, Ts=Ts)
    re = dec.encode(apply(x, el).to(dev))
    return generate(pz, el, start=m, codes=re, mode=mode, T=T, Ts=Ts)


# FLOP proxy in token positions: a pass processes its prompt (once) and the positions of the levels it samples. A
# retry in a new frame must recompute the prompt; in the identity frame the prompt's keys/values can be reused.
FULL = P_LEN + sum(LV)


def retry_cost(m, frames):
    return ((P_LEN if frames == "twisted" else 0) + sum(LV[m:])) / FULL


def run_arm(x1, frames, m, regen, T, seed):
    """Checker-gated policy from a first pass x1 (12 NFE): accept, else singles (if enabled), else up to ROUNDS
    retries. Returns per board the round in which it was first solved (-1 never) and the round after which the
    checker accepted it (-1 never), plus the duplicate rate of retries."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    xs = x1.clone()
    acc = checker(xs, puzzles)
    if SINGLES:
        use_s = ~acc & sing_ok
        xs[use_s] = sing_board[use_s]
        acc = checker(xs, puzzles)
    solved = (xs == solutions).all(1)
    solved_round = torch.where(solved, 0, -1)
    acc_round = torch.where(acc, 0, -1)
    seen = [{bytes(xs[i].to(torch.uint8).tolist())} for i in range(N)]
    dup = tot = 0
    for r in range(1, ROUNDS + 1):
        todo = (~acc).nonzero().squeeze(1)
        for s in range(0, len(todo), BS):
            idx = todo[s:s + BS]
            el = stack_elems([rand_elem(g) for _ in range(len(idx))]) if frames == "twisted" else ident_el(len(idx))
            xn = retry(xs[idx], puzzles[idx], el, m, regen, T, TS_STAR)
            for j, i in enumerate(idx.tolist()):
                key = bytes(xn[j].to(torch.uint8).tolist())
                tot += 1
                dup += key in seen[i]
                seen[i].add(key)
            xs[idx] = xn
        acc = checker(xs, puzzles)
        solved = (xs == solutions).all(1)
        solved_round = torch.where((solved_round < 0) & solved, r, solved_round)
        acc_round = torch.where((acc_round < 0) & acc, r, acc_round)
    return {"solved_round": solved_round.short().numpy(), "acc_round": acc_round.short().numpy(),
            "m": m, "frames": frames, "dup": dup / max(tot, 1), "retries": tot,
            "wrong_accepted": int((acc & ~solved).sum()), "unaccepted_end": int((~acc).sum())}


def cost_matrix(run, cost="nfe"):
    """(ROUNDS + 1, N) cumulative cost per board after each round: a board accepted after round a stops retrying."""
    a = np.where(run["acc_round"] >= 0, run["acc_round"], ROUNDS).astype(np.int64)
    n_retries = np.minimum(np.arange(ROUNDS + 1)[:, None], a[None, :])
    if cost == "nfe":
        return K + (K - run["m"]) * n_retries
    return 1.0 + retry_cost(run["m"], run["frames"]) * n_retries


# ---------------------------------------------------------------- curves and paired bootstrap ---------------------------
def acc_at(run, idx, budget, cost="nfe"):
    sr, c = run["solved_round"][idx], cost_matrix(run, cost)[:, idx]
    xs = c.mean(1)
    ys = np.array([((sr >= 0) & (sr <= r)).mean() for r in range(c.shape[0])])
    # Past the last round: once every board is accepted no more compute is spent, so accuracy stays final;
    # if boards were still being retried when the rounds ran out, the curve is truncated (nan).
    return float(np.interp(budget, xs, ys, right=ys[-1] if run["unaccepted_end"] == 0 else np.nan))


def arm_acc(runs, idx, budget, cost="nfe"):
    return float(np.nanmean([acc_at(r, idx, budget, cost) for r in runs]))


def paired(ra, rb, idx, budget, cost="nfe", seed=0):
    rng = np.random.default_rng(seed)
    d0 = arm_acc(ra, idx, budget, cost) - arm_acc(rb, idx, budget, cost)
    ds = []
    for _ in range(NBOOT):
        ix = idx[rng.integers(0, len(idx), len(idx))]
        ds.append(arm_acc(ra, ix, budget, cost) - arm_acc(rb, ix, budget, cost))
    ds = np.array(ds)
    ds = ds[~np.isnan(ds)]
    if len(ds) < NBOOT // 2:
        return {"diff": d0 * 100, "lo": float("nan"), "hi": float("nan"), "n": len(ds)}
    return {"diff": d0 * 100, "lo": float(np.percentile(ds, 2.5) * 100),
            "hi": float(np.percentile(ds, 97.5) * 100), "n": len(ds)}


if __name__ == "__main__":
    TAG = DIFF + ("" if SINGLES else "_nosingles")
    out = {"diff": DIFF, "n": N, "seeds": SEEDS, "rounds": ROUNDS, "m": M, "ts_grid": TS, "device": str(dev),
           "gpu": GPU_NAME, "singles": bool(SINGLES),
           "singles_complete": float(sing_ok.float().mean()), "flop_full_positions": FULL}
    t0 = time.perf_counter()
    x_hard = first_pass("hard")
    s_hard = (x_hard == solutions).all(1)
    print(f"greedy first pass: {s_hard.float().mean()*100:.2f}% [{time.perf_counter()-t0:.0f}s]", flush=True)
    half = N // 2
    held, all_ = np.arange(half, N), np.arange(N)
    soft_x, soft_rate = {}, {}
    for Ts in TS:
        t0 = time.perf_counter()
        soft_x[Ts] = first_pass("soft", Ts)
        s = (soft_x[Ts] == solutions).all(1)
        soft_rate[Ts] = {"select": float(s[:half].float().mean()), "held": float(s[half:].float().mean()),
                         "all": float(s.float().mean()), "fixed": int((s & ~s_hard).sum()),
                         "broken": int((~s & s_hard).sum())}
        print(f"soft Ts={Ts}: select {soft_rate[Ts]['select']*100:.2f}% held {soft_rate[Ts]['held']*100:.2f}% "
              f"all {soft_rate[Ts]['all']*100:.2f}% (fixed {soft_rate[Ts]['fixed']}, broken {soft_rate[Ts]['broken']}) "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    TS_STAR = max(TS, key=lambda t: soft_rate[t]["select"])
    out.update({"greedy_first_pass": float(s_hard.float().mean()), "soft": {str(k): v for k, v in soft_rate.items()},
                "ts_star": TS_STAR})
    print(f"Ts* = {TS_STAR} (chosen on boards 0..{half-1})", flush=True)

    # name: (first pass, frames, restart level, regeneration, temperature)
    arms = {"N": ("hard", "ident", 0, "hard", 1.0),     # best-of-N: fresh restarts at T = 1
            "F": ("hard", "twisted", 0, "hard", 0.0),   # fresh greedy restarts, each in a random symmetry frame
            "E": ("hard", "ident", M, "hard", 1.0),     # V-cycle retries from level M (the repo's method)
            "A": ("hard", "twisted", M, "hard", 0.0),   # symmetry-twisted V-cycle
            "B": ("soft", "twisted", M, "hard", 0.0),   # soft-decision first pass + twisted V-cycle
            "D": ("soft", "twisted", M, "soft", 0.0)}   # soft decisions in the first pass and in every retry
    runs = {}
    for name, (first, frames, m, regen, T) in arms.items():
        x1 = x_hard if first == "hard" else soft_x[TS_STAR]
        runs[name] = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            res = run_arm(x1, frames, m, regen, T, seed)
            runs[name].append(res)
            print(f"arm {name} s{seed}: " + " ".join(f"{acc_at(res, all_, b)*100:.1f}@{b}" for b in (12, 13, 14, 15, 16))
                  + f" | final {((res['solved_round'] >= 0).mean())*100:.1f}@{cost_matrix(res)[-1].mean():.2f}"
                  + f" | dup {res['dup']:.3f} of {res['retries']} | wrong accepted {res['wrong_accepted']}"
                  + f" [{time.perf_counter()-t0:.0f}s]", flush=True)

    budgets = (12, 13, 14, 15, 16, 18, 20)
    out["arms"] = {n: {split: {str(b): arm_acc(r, ix, b) * 100 for b in budgets}
                       for split, ix in (("all", all_), ("held", held))} for n, r in runs.items()}
    out["arms_flop"] = {n: {str(f): arm_acc(r, all_, f, "flop") * 100 for f in (1.0, 1.1, 1.2, 1.3, 1.4, 1.6)}
                        for n, r in runs.items()}
    out["final"] = {n: {"solved": float(np.mean([(x["solved_round"] >= 0).mean() for x in r])) * 100,
                        "mean_nfe": float(np.mean([cost_matrix(x)[-1].mean() for x in r]))} for n, r in runs.items()}
    out["dup"] = {n: float(np.mean([x["dup"] for x in r])) for n, r in runs.items()}
    out["wrong_accepted"] = {n: [x["wrong_accepted"] for x in r] for n, r in runs.items()}
    out["curves"] = {n: [{"nfe": cost_matrix(x)[:, all_].mean(1).tolist(),
                          "flop": cost_matrix(x, "flop")[:, all_].mean(1).tolist(),
                          "solved": [float(((x["solved_round"] >= 0) & (x["solved_round"] <= k)).mean())
                                     for k in range(ROUNDS + 1)]} for x in r] for n, r in runs.items()}
    comps = [("C-A0", "B", "A", 12), ("E-N", "E", "N", 14), ("F-N", "F", "N", 14), ("A-E", "A", "E", 13),
             ("A-E", "A", "E", 14), ("A-E", "A", "E", 16), ("A-F", "A", "F", 13), ("A-F", "A", "F", 14),
             ("A-F", "A", "F", 16), ("B-A", "B", "A", 13), ("B-A", "B", "A", 14), ("D-B", "D", "B", 14),
             ("D-A", "D", "A", 13), ("D-A", "D", "A", 14), ("D-E", "D", "E", 13), ("D-E", "D", "E", 14),
             ("D-E", "D", "E", 16)]
    out["paired"] = {}
    for label, a, b, bud in comps:
        for split, ix in (("all", all_), ("held", held)):
            r = paired(runs[a], runs[b], ix, bud)
            out["paired"][f"{label}@{bud}|{split}"] = r
            print(f"{label:5s} @{bud} NFE [{split:4s}]: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]", flush=True)
    for a, b in (("A", "E"), ("D", "E"), ("A", "F")):
        for f in (1.2, 1.4):
            r = paired(runs[a], runs[b], all_, f, "flop")
            out["paired"][f"{a}-{b}@{f}FLOP|all"] = r
            print(f"{a}-{b}   @{f} full-pass FLOP equivalents [all ]: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]",
                  flush=True)
    with open(os.path.join(RESULTS, f"sudoku_stack_{TAG}.json"), "w") as f:
        json.dump(out, f, indent=1)
    os.makedirs(os.path.join(RESULTS, "cache"), exist_ok=True)
    torch.save({"runs": runs, "x_hard": x_hard.to(torch.uint8), "soft_x": {k: v.to(torch.uint8) for k, v in soft_x.items()},
                "sing_ok": sing_ok}, os.path.join(RESULTS, "cache", f"sudoku_stack_{TAG}_perboard.pt"))
    print("done", flush=True)
