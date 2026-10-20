"""Multigrid V-cycle decoding for NCM (training-free).

One cycle, per board:
  restriction   r' = encode(project(x))                  frozen tokenizer encoder + quantizer
  residual      d_k = mean_i [r'_k,i != r_k,i]            per level
  pick level    m (rule below); boards with m = K are left untouched
  correction    keep r'_{<m} (or r_{<m} with reencode=False), resample levels m..K-1, decode

Restart-level rules (`rule`):
  ("fixed", m)   V4: restart every board at level m
  ("tau", t)     V1: m = coarsest level with d_k > t      (t = 0 is a no-op under greedy decoding:
                 levels < m then agree, so the same context resamples the same codes)
  ("accept", t)  V1 with the restriction accepted at the disagreeing level itself:
                 m = (coarsest level with d_k > t) + 1, so r'_j replaces r_j
masked=True gives V2: at the restart level only positions with r'_m != r_m are resampled.
"""
import torch

from vcycle import sudoku


def disagreement(codes, codes_re):
    return torch.stack([(a != b).float().mean(1) for a, b in zip(codes, codes_re)], 1)  # (B, K)


def restart_level(rule, d, K):
    kind, v = rule
    B = d.shape[0]
    if kind == "fixed":
        return torch.full((B,), int(v), device=d.device)
    over = d > float(v)
    first = torch.where(over.any(1), over.float().argmax(1), torch.full((B,), K, device=d.device))
    if kind == "tau":
        return first
    if kind == "accept":
        return torch.where(first < K, first + 1, first).clamp(max=K)
    raise ValueError(kind)


@torch.no_grad()
def cycle(dec, prefix, codes, x, rule, temperature=0.0, top_k=0, top_p=1.0, puzzles=None,
          project=False, reencode=True, masked=False, active=None):
    """One V-cycle on a batch. Returns (codes, tokens, start, nfe) with per-board start and NFE."""
    K = dec.K
    xin = sudoku.project(x, puzzles) if project else x
    codes_re = dec.encode(xin)
    start = restart_level(rule, disagreement(codes, codes_re), K)
    if active is not None:
        start = torch.where(active.to(start.device), start, torch.full_like(start, K))
    base = codes_re if reencode else codes
    new_codes = [c.clone() for c in codes]
    new_x = x.clone()
    for s in torch.unique(start).tolist():
        if s >= K:
            continue
        idx = (start == s).nonzero().squeeze(1)
        sub = [c[idx] for c in base]
        keep = None
        if masked:
            keep = {s: codes_re[s][idx] == codes[s][idx]}
            sub[s] = codes[s][idx]
        c, z = dec.run(prefix[idx], start=s, codes=sub, keep=keep,
                       temperature=temperature, top_k=top_k, top_p=top_p)
        for k in range(K):
            new_codes[k][idx] = c[k]
        new_x[idx] = dec.decode(z)[0]
    return new_codes, new_x, start, dec.board_nfe(start)


@torch.no_grad()
def v_cycle_decode(dec, prefix, codes, x, rule, n_cycles=1, stop_when=None, **kw):
    """Repeat cycles; stop_when(x) -> bool (B,) marks boards to freeze (e.g. a verifier says valid).
    Returns final codes, tokens, total NFE per board (excluding the first pass) and per-cycle history."""
    nfe = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    hist = []
    for _ in range(n_cycles):
        active = ~stop_when(x) if stop_when is not None else None
        new_codes, new_x, start, n = cycle(dec, prefix, codes, x, rule, active=active, **kw)
        changed = (new_x != x).any(1)
        codes, x = new_codes, new_x
        nfe += n
        hist.append({"x": x.clone(), "start": start.clone(), "changed": changed})
    return codes, x, nfe, hist
