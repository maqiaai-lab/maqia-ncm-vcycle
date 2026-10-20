"""Cached single-pass NCM samples (codes, tokens, confidences) so later phases reuse the first pass."""
import os

import torch

from vcycle import sudoku
from vcycle.common import RESULTS, Timer, batches


def sudoku_pass(dec, puzzles, bs=125, temperature=0.0, seed=0, name=None):
    path = os.path.join(RESULTS, "cache", f"{name}.pt") if name else None
    if path and os.path.exists(path):
        return torch.load(path, weights_only=True)
    torch.manual_seed(seed)
    dev = dec.ncm.device
    codes, toks, gconf, dconf = [], [], [], []
    with Timer() as t:
        for sl in batches(len(puzzles), bs):
            conf = []
            c, z = dec.run(dec.prefix(puzzles[sl].to(dev)), temperature=temperature, conf=conf)
            x, probs = dec.decode(z)
            codes.append(torch.cat(c, 1).cpu().to(torch.uint8))
            toks.append(x.cpu().to(torch.uint8))
            gconf.append(torch.cat(conf, 1).cpu().half())
            dconf.append(probs.max(-1).values.cpu().half())
    out = {"codes": torch.cat(codes), "tokens": torch.cat(toks), "gen_conf": torch.cat(gconf),
           "dec_conf": torch.cat(dconf), "seconds": t.s, "temperature": temperature, "seed": seed}
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(out, path)
    return out


def split_levels(dec, flat):
    return list(flat.long().split(dec.levels, dim=1))
