# Multigrid V-Cycle Decoding for NCM — Working Plan

**Goal:** Show that NCM samples improve if we add multigrid-style correction cycles at inference time, with **no retraining**.

**Target paper:** *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation* (Kumbong, Khare et al., arXiv:2610.14162, Oct 2026)
**Code:** https://github.com/HazyResearch/ncm (code, checkpoints and datasets are public)
**Contacts (later):** kumboh@stanford.edu, iskhare@stanford.edu

---

## 1. The idea in one paragraph

NCM decodes once from coarse to fine and never revisits earlier levels. In multigrid terms, it does one "prolongation" sweep and no "restriction." Classical multigrid theory (Brandt, 1977) shows that local corrections can't remove global, low-frequency errors. You remove them by projecting the residual back to a coarse grid, fixing it there, and propagating the fix back up (a V-cycle). NCM has the piece diffusion models lack: a **frozen encoder**. After decoding, we can re-encode the output, see where the coarse codes disagree with what was generated, resample from that level down, and repeat. Hypothesis: one or two cycles reduce repetition artifacts and constraint violations (Sudoku) at a small extra cost.

## 2. Mapping multigrid → NCM

| Multigrid | NCM |
|---|---|
| Coarse grid | Coarse code levels (k = 1, 2, …), which are short |
| Prolongation (coarse → fine) | Normal generation of levels k+1…K |
| Restriction (fine → coarse) | Re-encoding decoded text with the frozen tokenizer encoder |
| Residual | Disagreement between generated codes `r_k` and re-encoded codes `r'_k` |
| Smoothing | Resampling fine levels conditioned on a corrected coarse reconstruction |
| V-cycle | Decode → re-encode → resample from level j → decode |

## 3. Key notation (from the paper)

- `r_k`: codes at level k, length `ℓ_k` (ℓ_1 < … < ℓ_K = L)
- `ẑ^(k)`: cumulative reconstruction after level k = Σ_{j≤k} R_j(e_j(r_j))
- Generator: `p_θ(r_k | ẑ^(k-1))`, with codes **independent within a level**
- Decoder: `p_ψ(x | ẑ^(K))`, argmax → tokens
- Encoder + quantizer: text `x` → codes `r_{1:K}`

---

## 4. Phase 0 — Setup and reproduction (Day 1–2)

- [ ] Clone the repo, install, download checkpoints for **Sudoku** and **LM1B** first (small, fast).
- [ ] Reproduce one number from the paper on each (Sudoku accuracy, LM1B gen. perplexity + unigram entropy).
- [ ] Find these entry points in the code. I could not inspect the repo; fill these in.
  - [ ] Tokenizer `encode(text) -> codes per level`: ____________
  - [ ] Tokenizer `decode(ẑ) -> tokens`: ____________
  - [ ] Generation loop over levels: ____________
  - [ ] Can generation **start from level j** given `ẑ^(j-1)`? If not, write a small wrapper.
  - [ ] How Sudoku clues and prompts are fed (prefix-LM conditioning): ____________
- [ ] Log the per-level codes, not just final text, for every sample.

## 5. Phase 1 — Go/no-go diagnostic: the fixed-point test (Day 2–3)

This is the cheapest test of whether the idea has signal at all, and it needs no new decoding.

**Definition:** a sample is *self-consistent* if `encode(decode(r)) == r`, i.e., it is a fixed point of the tokenizer round-trip.

- [ ] For N baseline samples (e.g., 2,000 Sudoku puzzles, 2,000 LM1B samples), compute:
  - Fixed-point rate overall and **per level** (fraction of positions where `r'_k == r_k`).
  - Where disagreements concentrate (which levels, which positions).
- [ ] Correlate disagreement with errors:
  - Sudoku: disagreement rate in **wrong** vs **correct** solutions; disagreement vs the number of violated rows, columns and boxes.
  - LM1B: disagreement vs repetition rate (repeated adjacent tokens, distinct-2/3).

**Go if:** wrong or degenerate samples show clearly more coarse-level disagreement than good samples. That means the residual carries an error signal.
**If it fails:** the round-trip may be near-identity even for bad samples. Switch the residual signal to decoder confidence instead: token-level entropy or max-probability from `p_ψ(x | ẑ^(K))`, mapped back to the levels and positions that influence those tokens.

## 6. Phase 2 — V-cycle decoding (Day 3–7)

### Pseudocode (adapt to the repo's API)

```python
def ncm_decode(model, tok, prompt, start_level=1, z_prev=None, codes_prev=None):
    """Standard NCM generation, optionally starting from level `start_level`."""
    codes = list(codes_prev[:start_level-1]) if codes_prev else []
    z = z_prev if z_prev is not None else tok.zero_recon()
    for k in range(start_level, tok.K + 1):
        r_k = model.sample_level(k, z, prompt)      # parallel within level
        codes.append(r_k)
        z = z + tok.reconstruct(k, r_k)             # ẑ^(k) = ẑ^(k-1) + R_k(e_k(r_k))
    x = tok.decode_argmax(z)
    return x, codes

def v_cycle_decode(model, tok, prompt, n_cycles=2, project=None, pick_level=None):
    x, codes = ncm_decode(model, tok, prompt)
    for _ in range(n_cycles):
        if project is not None:               # optional: enforce hard constraints
            x = project(x, prompt)            # e.g., overwrite Sudoku clue cells
        codes_re = tok.encode(x)              # restriction: text -> codes
        j = pick_level(codes, codes_re)       # coarsest level with large disagreement
        if j is None:                         # fixed point reached
            break
        # Keep re-encoded (self-consistent) coarse levels 1..j-1, resample j..K
        z_keep = tok.recon_from_codes(codes_re[:j-1])
        x, codes = ncm_decode(model, tok, prompt, start_level=j,
                              z_prev=z_keep, codes_prev=codes_re)
    return x
```

### Variants to try

1. **V1 — Restart from level j:** keep re-encoded levels 1…j-1 and resample j…K (as above).
2. **V2 — Masked resample:** at level j, resample only positions where `r'_j ≠ r_j` and keep the rest. This is valid because within-level codes are conditionally independent.
3. **V3 — Constraint projection (Sudoku):** before re-encoding, overwrite clue cells with the given clues (the "pinned constraints"), then V-cycle. This is alternating projections between "what the tokenizer considers valid" and "what the constraints require."
4. **V4 — Fixed j:** always restart at j = 1, 2 or 3. This is a simpler baseline for choosing `pick_level`.

### `pick_level` rules
- Coarsest level where disagreement > τ (try τ ∈ {0, 0.05, 0.1}).
- Or fixed j (V4).

## 7. Phase 3 — Controls and ablations (Week 2)

**Compute-matched controls (critical).** The V-cycle must beat simply spending extra compute.

| Control | What it tests |
|---|---|
| Best-of-N at equal NFE / wall clock | Is it just more samples? |
| MBR at equal NFE (the paper already uses MBR) | Is it just selection? |
| Fresh restart with a new seed at equal NFE | Is it just re-rolling? |
| Re-encode only (no resampling) | Is the round-trip alone a fix? |
| Resample fine levels **without** re-encoding coarse levels | Does the restriction step matter? |

**Ablations:** number of cycles (1/2/3), start level j, threshold τ, sampling temperature, V1 vs V2.

## 8. Metrics

- **Sudoku:** accuracy per difficulty (the paper's main reasoning result), number of constraint violations, accuracy vs number of cycles.
- **LM1B / OpenWebText:** gen. perplexity **and** unigram entropy (report both; perplexity alone can be gamed by low diversity), repetition rate, distinct-n.
- **WMT14 De→En (optional):** BLEU.
- **Cost:** NFE, wall-clock time per sample, % overhead vs a single pass.
- **Convergence:** fixed-point rate after each cycle (does it converge, like multigrid?).

### Results table template

| Method | NFE | Time | Sudoku (easy/med/hard) | LM1B PPL | Entropy | Rep. rate |
|---|---|---|---|---|---|---|
| NCM (paper) | | | | | | |
| NCM (reproduced) | | | | | | |
| + Best-of-N (matched) | | | | | | |
| + V-cycle ×1 (V1) | | | | | | |
| + V-cycle ×2 (V1) | | | | | | |
| + V-cycle + projection (V3) | | | | | | |

## 9. Phase 4 — Stretch: global editing (if Phase 2 works)

- Pin a constraint (a required term in a translation, a fixed sentence in a story) and V-cycle until the whole output is consistent with it.
- This frames NCM as a **constraint solver**, which is a bigger claim than "better samples."

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Round-trip is near-identity, so there's no residual signal | Use decoder confidence or entropy as the residual (Phase 1 fallback) |
| Re-encoding argmax tokens loses information | Also try re-encoding from the soft decoder distribution, if the encoder accepts embeddings |
| Gains vanish under compute-matched controls | Report honestly; the fixed-point diagnostic is still a publishable observation |
| Cycles oscillate instead of converging | Damp: resample only V2-masked positions; lower the temperature on later cycles |
| Can't start generation mid-hierarchy in the repo | Write a wrapper that injects `ẑ^(j-1)` and codes for levels < j |

## 11. Prior-art check (do before outreach)

Search Google Scholar, arXiv and Semantic Scholar for:
- [ ] "multigrid" + "discrete diffusion" / "next-scale prediction" / "VAR"
- [ ] "V-cycle" + "generative model" / "autoregressive" / "VQ-VAE"
- [ ] "encode-decode consistency" / "cycle consistency" + "non-autoregressive decoding"
- [ ] "re-encoding" + "iterative refinement" + "VQ"
- [ ] Self-correction in VAR models (e.g., Infinity's bitwise self-correction; note it is a *training-time* fix, while ours is *inference-time*)
- [ ] HMAR (Kumbong et al., CVPR 2025), the NCM lead author's earlier image work. Check what refinement it already does.

Record anything close here and state clearly how this differs.

## 12. Timeline

| When | Milestone |
|---|---|
| Days 1–2 | Setup, reproduce Sudoku + LM1B |
| Days 2–3 | Fixed-point diagnostic → go/no-go |
| Days 3–7 | V-cycle variants V1–V4 on Sudoku |
| Week 2 | Compute-matched controls, LM1B, ablations |
| Week 3 | Prior-art check, short write-up, one key chart, GitHub repo |
| Then | Email the authors (result in the body, links to the repo) |

## 13. Outreach checklist

- [ ] One chart: Sudoku accuracy vs NFE, baseline vs V-cycle vs best-of-N.
- [ ] Two or three key numbers written in the email body.
- [ ] A public GitHub repo with code and results (full URL, no shorteners, no zips).
- [ ] A specific subject line, e.g., *"NCM: multigrid V-cycle decoding improves Sudoku accuracy (short result)"*.
- [ ] A clear ask (feedback / call / collaboration) and what you want (credit, co-authorship, just sharing).

## 14. Notes and log

_Use this section as a lab notebook: dates, commands, results, surprises._
