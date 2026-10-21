"""Check a new machine against the committed results: decode and re-score committed LM1B samples.

Sampling uses a different RNG stream on another device, so seeds cannot be compared directly. Everything after
sampling is deterministic and is compared with the committed cache: decoding the committed codes and picking tokens,
the round-trip residuals, and GPT-2-large NLL (per sample and corpus PPL).

python experiments/check_gpu_scoring.py --size small --seed 42
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, RESULTS, device
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import LM1B
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="small")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, "lm1b", a.size)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm)
lm = LM1B(ncm, dec, ck)
c = torch.load(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s{a.seed}.pt"), weights_only=False)
codes, ids = c["codes"], c["ids"]
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"device {device()} ({name}); {len(ids)} committed samples of lm1b_{a.size}_s{a.seed}", flush=True)

out = []
for s in range(0, len(ids), 32):
    cc = [x[s:s + 32].to(dec.ncm.device) for x in codes]
    out.append(lm.pick_tokens(ncm.vqvae.decoder(dec.z_from_codes(cc))).cpu())
new_ids = torch.cat(out)
print(f"decode + pick: {int((new_ids != ids).any(1).sum())}/{len(ids)} samples differ, "
      f"{float((new_ids != ids).float().mean()):.5f} of tokens (near-ties in token picking)", flush=True)
rt, loc = lm.residuals(codes, ids)
print(f"residual max |diff|: round trip {float((rt - c['rt']).abs().max()):.4g}, level-local {float((loc - c['loc']).abs().max()):.4g}")
nl, nt = lm.nll(ids)
nl0, nt0 = c["per_sample_nll"]["baseline"]
rel = float(np.max(np.abs(nl - nl0) / np.abs(nl0)))
print(f"token counts equal: {np.array_equal(nt, nt0)}; per-sample NLL max relative difference {rel:.2e}")
print(f"corpus PPL: committed {np.exp(nl0.sum() / nt0.sum()):.3f}, here {np.exp(nl.sum() / nt.sum()):.3f}")
assert np.array_equal(nt, nt0) and rel < 1e-4, "scoring differs from the committed results"
print("scoring check passed")
