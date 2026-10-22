# Multigrid V-Cycle Decoding for NCM — Working Plan

**Goal:** Show that NCM samples improve if we add multigrid-style correction cycles at inference time, with **no retraining**.

**Target paper:** *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation* (Kumbong, Khare et al., arXiv:2610.14162, Oct 2026)
**Code:** https://github.com/HazyResearch/ncm (code, checkpoints and datasets are public)

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

- [x] Clone the repo, install, download checkpoints for **Sudoku** first. `setup.sh` pins NCM at commit `c702742`. LM1B is not done yet; it is a stretch goal because this machine is CPU-only.
- [x] Reproduce one number from the paper. Sudoku-hard: **86.30%** solved on 2,000 boards vs **86.1%** in the paper, at 12 NFE with greedy decoding (`results/phase0_sudoku_hard.json`). LM1B is not done.
- [x] Find these entry points in the code (HazyResearch/ncm @ c702742):
  - [x] **Tokenizer `encode(text) -> codes per level`.** In NCM this is `VQVAE.encode` (`models/vqvae.py`), which runs `_encode_pre_quant` and then `MultiscaleResidualVectorQuantizer.z_to_idx` (`models/quant.py`). The Sudoku encoder is just token + position embeddings followed by a GroupNorm (`enc_layers=0`), so restriction costs almost nothing. Our wrapper is `LevelDecoder.encode`, which splits the 322 codes into the 12 levels (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89).
  - [x] **Tokenizer `decode(ẑ) -> tokens`.** In NCM this is `VQVAE.decoder(z_hat)` (`models/vae.py`, one transformer block) followed by argmax. LM1B instead picks tokens with frequency and adjacency penalties (`pick_tokens` in `scripts/eval_shard.py`). Our wrapper is `LevelDecoder.decode`.
  - [x] **Generation loop over levels.** `NextConceptTransformer.generate` (`models/next_concept.py`). Level 0 comes from learned queries (`query_level0`); level k+1 reads `area_interp(ẑ^(k), l_{k+1})` (`vq.get_next_autoregressive_input`) through `_decode_level_logits`.
  - [x] **Can generation start from level j?** Not in NCM. Our wrapper `LevelDecoder.run(prefix, start=j, codes=..., keep=...)` supports a per-board start level and masked resampling. It reproduces `generate()` code for code on 500 boards per difficulty. It also caches the prompt's per-layer keys and values, since the prompt never attends to the targets, which makes it 3.5× faster with identical codes (`experiments/check_prefix_cache.py`).
  - [x] **How Sudoku clues and prompts are fed.** The prompt is prefix-LM conditioning on 91 raw tokens (BOS, 89 board tokens with 0 for blank, BOS), embedded with the tokenizer's token table (`raw_token_prefix`). Every level attends to it, and it attends only to itself. Clues are not hard-enforced: 0.1% of greedy boards overwrite a clue.
- [x] Log the per-level codes, not just final text, for every sample. `vcycle/cache.py` stores codes (N×322), tokens and the generator and decoder confidences.

## 5. Phase 1 — Go/no-go diagnostic: the fixed-point test (Day 2–3)

This is the cheapest test of whether the idea has signal at all, and it needs no new decoding.

**Definition:** a sample is *self-consistent* if `encode(decode(r)) == r`, i.e., it is a fixed point of the tokenizer round-trip.

- [x] For N baseline samples (e.g., 2,000 Sudoku puzzles, 2,000 LM1B samples), compute: *(done: 2,000 boards per difficulty, 768 LM1B-small + 256 LM1B-medium texts; `results/phase1_*.json`, `results/lm1b_*.json`)*
  - Fixed-point rate overall and **per level** (fraction of positions where `r'_k == r_k`).
  - Where disagreements concentrate (which levels, which positions).
- [x] Correlate disagreement with errors: *(level-local residual AUROC 0.87 / 0.88 / 0.92 on Sudoku; 0.74 for the worst-NLL decile on LM1B-small)*
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
- [x] "multigrid" + "discrete diffusion" / "next-scale prediction" / "VAR"
- [x] "V-cycle" + "generative model" / "autoregressive" / "VQ-VAE"
- [x] "encode-decode consistency" / "cycle consistency" + "non-autoregressive decoding"
- [x] "re-encoding" + "iterative refinement" + "VQ" *(closest: DiverseVAR "scale-travel", arXiv:2511.21415; see `docs/prior_art.md`)*
- [x] Self-correction in VAR models (e.g., Infinity's bitwise self-correction; note it is a *training-time* fix, while ours is *inference-time*)
- [x] HMAR (Kumbong et al., CVPR 2025), the NCM lead author's earlier image work. Check what refinement it already does. *(masked multi-step sampling within a scale; never revisits coarser scales)*

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

- [x] Charts: `figures/fig1_restriction_effect.png` (text) and `figures/fig6_sudoku_retries.png` (Sudoku); all figures are drawn by `experiments/figures.py`.
- [ ] Two or three key numbers written in the email body.
- [ ] A public GitHub repo with code and results (full URL, no shorteners, no zips).
- [ ] A specific subject line, e.g., *"NCM: multigrid V-cycle decoding improves Sudoku accuracy (short result)"*.
- [ ] A clear ask (feedback / call / collaboration) and what you want (credit, co-authorship, just sharing).

## 14. Notes and log

_Use this section as a lab notebook: dates, commands, results, surprises._

### 2026-10-20 (setup, Phase 0, Phase 1 on hard)

- **Environment:** CPU only (4 cores, 15 GB RAM, no GPU). The code runs unchanged on GPU (`vcycle.common.device()`).
- **Checkpoint safety:** all checkpoints load with `weights_only=True`. The dataset pickles contain only integer lists (checked with `pickletools`).
- **Sudoku-hard reproduced:** 86.30% solved on 2,000 boards (paper: 86.1%). The baseline never violates a clue on the first 250 boards and does on 0.1% of all 2,000.
- **Surprise 1 (fixed points):** the fixed-point rate is **0%**. Every greedy sample's codes differ from `encode(decode(r))` somewhere, yet `decode(encode(x)) == x` holds for every board. The generator writes a *non-canonical* code for the same board.
  - The plan's yes/no fixed-point test therefore carries no signal (AUROC 0.5).
  - The *amount* of disagreement does: **AUROC 0.86** for spotting wrong boards (0.92 with the level-local residual, `encode_given_coarse`). For comparison, decoder confidence scores 0.87 and generator confidence 0.73. **Go.**
- **Surprise 2 (level 4):** at level 4 (9 positions, about one row each) the generator is 100% confident on every board, yet the residual fires on 0.5% of clean spans and on 25% of spans that contain an error. When it fires, the span is wrong **97.5%** of the time (base rate 43%). The restriction sees errors the generator cannot (`results/phase1b_localization_hard.json`).
- **Plan correction:** V1 as written (keep re-encoded levels < j, resample from j, with τ = 0) is a guaranteed **no-op** under greedy decoding. Levels < j agree, so the same context gives the same argmax. I added *V1-accept*, which also keeps the re-encoded level j and resamples from j+1.
- **Plan correction:** V3 clue projection changes the codes of only the 0.1% of boards that violate a clue.
- **Prior art:** see `docs/prior_art.md`. The closest work is DiverseVAR's "scale-travel" (arXiv:2511.21415): re-encode and resume at a fixed scale, once, for images.

### 2026-10-21 (engineering notes, Phase 2/3 smoke tests)

- **Memory bug and fix:** `getattr(omegaconf_cfg, "missing_key", default)` raises and catches an exception internally. The caught exception's traceback cycle pinned the caller's frame, and with it about 750 MB of cached prompt keys/values per batch, until the process was OOM-killed at 14 GB. Fixed by reading config flags once with `.get()`.
- **Speed:** with the prompt cache, a greedy pass costs 0.077 s/board on CPU (2,000 boards in about 2.6 min).
- **Phase 2 smoke test** (250 hard boards, greedy, 2 cycles):
  - Two configurations are exact no-ops, as predicted: V1 τ=0, and no-restrict at m=3.
  - V1-accept: 84.8 → 86.0% (4 boards fixed, 1 broken).
  - Gated by the Sudoku checker: 86.4%, for +5 NFE/board.
- **Phase 3 smoke test** (250 boards, temperature 1, retries gated by the checker, 3 rounds):
  - Partial restarts are cheaper than fresh ones: 92.4% at 15.2 NFE vs 91.6% at 16.5 NFE.
  - Restriction vs no restriction at m=3 differs by 1 board, so the full 2,000-board, 3-seed run is needed.

### 2026-10-21 (LM1B arm: the V-cycle works on text)

Setup: `experiments/lm1b_diagnostic.py` and `lm1b_sweep.py`.
- **Sampling** matches NCM's LM1B eval exactly: level 0 from the unigram, T=1, frequency- and adjacency-penalized token picking. `LevelDecoder` reproduces `generate()` code for code under the same RNG seed.
- **Scoring** uses GPT-2 large with the FLM protocol (the paper's). Entropy is the mean per-sample unigram entropy of the raw ids.
- **Model:** LM1B-small, 256 samples per seed.

Results:
- **Baseline (NCM single pass):** gen. PPL 95.8 / 94.6 / 91.0 (seeds 42 / 0 / 1), entropy 4.317. The paper's LM1B point is about 92 at H = 4.31.
- **The residual on text:** the fixed-point rate is again 0%. The level-local residual flags the worst-NLL decile with AUROC 0.74 (mean of 3 seeds; Spearman ρ with per-token NLL +0.25).
  - The generated codes are 86–100% off the encoder's manifold at levels 0–6, and 25% at the finest level.
- **V-cycle, one cycle from level 6** (+6 NFE): gen. PPL **79.9 / 80.7 / 78.9**. No-restriction at the same levels and compute: 93.8 / 94.7 / 91.1.
  - Pooled paired bootstrap, V-cycle / no-restriction: **0.857 [0.836, 0.876]**.
  - Entropy rises from 4.314 to 4.344.
  - Restarting from level 3 does nothing (0.985 [0.951, 1.024]).
- **Sweep** (seed 42):
  - The V-cycle beats no-restriction at every level: m=4: 0.93×, m=6: 0.87×, m=8: 0.87×, m=10: 0.92×.
  - Iterating helps: m=6 gives 79.9 → 75.2 → 72.8 over 1/2/3 cycles; m=8 × 2 gives 75.2 at +8 NFE.
- **Not just colder sampling:**
  - NCM at T=0.9 reaches PPL 82.5, but with lower entropy (4.289), lower corpus distinct-2 (0.464) and more repeats (0.61%).
  - The V-cycle from m=8 at the same PPL (82.8) has entropy 4.337, distinct-2 0.489 and 0.35% repeats.
  - The V-cycle points lie beyond NCM's temperature frontier (`results/lm1b_small_ppl_vs_entropy.png`).
- **Caveat:** cross-sample distinct-2 does drop vs the T=1 baseline (0.505 → 0.477 for m=6). The V-cycle text is more fluent and somewhat more generic, though less so than colder sampling.
- **Mechanism (from the examples):** the V-cycle keeps the draft's content and repairs its fluency, like an editing pass. Resampling the same levels from the *generated* coarse codes produces an unrelated text: off-manifold coarse codes barely pin down content, while the encoder's codes do. Restarting from level 3 does not help, because levels 3–5 are regenerated off-manifold again.

### 2026-10-21 (Sudoku Phases 2–3, LM1B-medium, wrap-up)

Two container restarts interrupted the runs. Phase 2, Phase 3 and the LM1B diagnostic now checkpoint and resume, and no finished run was lost.

**Sudoku Phase 3** (hard, 2,000 boards × 3 seeds, checker-gated retries at T=1, compared at matched mean NFE, paired bootstrap):
- **V-cycle from level 5 vs fresh restarts (best-of-N):** +0.93 / +1.19 / +0.93 / +0.92 points at 13 / 14 / 15 / 16 NFE. All CIs exclude 0, e.g. 90.4% vs 89.3% at 14 NFE.
- **Restriction step alone** (V-cycle − no-restriction at level 5): +0.14 to +0.47 points, with CIs that include 0. At level 7 it is +0.6 and significant, but level 7 is a poor restart level.
- **V2 masked resampling** is significantly *worse* than full resampling (−0.7 to −1.5 points at level 5).
- **V1-accept** (residual-chosen level) is competitive but not better than the best fixed level.
- **Conclusion:** on Sudoku, most of the gain is "keep the coarse levels, resample the fine ones". Re-encoding adds little, because the generated codes are mostly on the encoder's manifold.

**Sudoku Phase 2** (greedy, no checker, 1,000 hard boards, 3 cycles):
- **Predicted no-ops:** V1 at τ ≤ 0.1 and no-restriction m=3 change nothing.
- **V1-accept:** 84.3 → 85.2% (10 fixed, 1 broken).
- **V4 m=6:** 85.3% after one cycle (10 fixed, 0 broken; +6 NFE), 85.5% after three.
- **Residual-gated:** cycling only the top-20% residual boards keeps 85.1% at a fifth of the compute.

**LM1B-medium** (317M parameters, seed 42, 256 samples):
- **Single pass:** 90.7.
- **V-cycle m=6:** **69.6**, against no-restriction 87.5. The ratio is 0.796 [0.765, 0.825], stronger than on small.
- **V-cycle m=8:** 78.9 vs 89.4.
- Entropy is unchanged (4.343 → 4.346), and corpus distinct-2 drops 0.586 → 0.554.

**Overall answer to the plan's question** ("do NCM samples improve with multigrid correction cycles, without retraining?"):
- **Text: yes, clearly.** The improvement comes specifically from the restriction step.
- **Sudoku: modestly.** The gain comes mostly from partial restarts, not from re-encoding.
- **The residual is a useful training-free error detector** on both.

All numbers are in `results/SUMMARY.md` and the README.

### 2026-10-21 (local GPU session: medium seeds, iteration, quality checks, gating)

**Environment.** Resumed locally: Windows 10, RTX 4050 Laptop (6 GB), torch 2.11.0+cu128, transformers 5.20.
- **GPU equivalence.** Sudoku-hard: the uncached `LevelDecoder` matches `generate()` code for code on all 2,000 boards. The cached-prompt path differs on 1 board, at a near-tie (top-2 logit margin 2.9e-6), so `phase0_baseline.py`'s assert fires on GPU; nothing was written. GPU vs the committed CPU greedy pass: 165 boards differ in codes but only 12 in tokens; solved 86.25% vs 86.30%. LM1B: `LevelDecoder == generate()` holds on GPU for small, medium and large (8 samples, T=1).
- **GPU scoring = CPU scoring.** Re-scoring the committed CPU samples (small, seed 42) on GPU with transformers 5.20: per-sample NLL max relative difference 3.5e-7, corpus PPL 95.808 on both. Decoding the committed codes on GPU changes 0.07% of tokens (near-ties in token picking). GPU and CPU seeds can be pooled; the same seed does *not* give the same samples (CUDA RNG).
- **VRAM.** Generator + tokenizer + GPT-2 large exceed 6 GB, and on Windows the driver spills to system memory instead of failing. `LM1B.nll` now parks the generator on the CPU while scoring (automatic below 16 GB of VRAM). LM1B-large then runs without spilling (generation 63 s vs 25 s for medium per 256 samples).
- **Speed.** One LM1B-medium seed with `--vcycle 6,8`: about 6 min (43 min on 4 CPU cores).

**LM1B-medium, 3 seeds** (42 on CPU from the cloud run; 0 and 1 on GPU), pooled paired bootstrap over 768 samples:
- Gen. PPL: single pass 90.7 / 87.1 / 89.9 (mean 89.2). V-cycle m=6: 69.6 / 65.9 / 68.5 (68.0). No-restrict m=6: 87.5 / 86.9 / 86.3 (86.9).
- **V-cycle / no-restrict: m=6 0.783 [0.765, 0.802]; m=8 0.871 [0.858, 0.886].** The single-seed m=6 figure was 0.796 [0.765, 0.825].
- Residual AUROC (worst-NLL decile, level-local): 0.640 (mean of 3 seeds).
- Bootstrap note: the LM1B-small CIs moved in the third decimal (e.g. [0.839, 0.877] → [0.839, 0.876]) without any change in data, because every section drew from one RNG and the medium section now draws more (N_BOOT = 1000 Monte Carlo noise). `analyze.py` now gives each text section its own generator, seeded from its name, so a section's CIs no longer depend on which other results exist. The Sudoku sections still use the shared generator first and are byte-identical to before.

**LM1B-large, 3 seeds** (GPU; the same tokenizer as small and medium):
- Gen. PPL: single pass 80.1 / 81.4 / 85.1 (seeds 42 / 0 / 1; mean 82.2). V-cycle m=6: 63.4 / 63.1 / 65.2 (63.9). No-restrict m=6: 80.4 / 79.5 / 79.3 (79.8).
- **V-cycle / no-restrict: m=6 0.801 [0.784, 0.818]; m=8 0.888 [0.874, 0.901].** Entropy 4.340 → 4.344; corpus distinct-2 0.582 → 0.555.
- The ratio does not shrink with scale: 0.857 (small), 0.783 (medium), 0.801 (large).
- Residual AUROC (worst-NLL decile, level-local): 0.678.

**TinyStories, 3 seeds** (step 5; `lm1b_diagnostic.py --task tinystories`; genppl protocol; GPU):
- `LevelDecoder == generate()` on 8 TinyStories samples, the first check of the `query_level0: false` path.
- Gen. PPL: single pass 15.9 / 16.3 / 17.1 (seeds 42 / 0 / 1; mean 16.4). V-cycle m=6: 13.1 / 13.3 / 13.6 (13.3). No-restrict m=6: 16.1 / 16.2 / 17.1 (16.5).
- **V-cycle / no-restrict: m=6 0.809 [0.797, 0.821]; m=8 0.915 [0.905, 0.924].** Entropy 3.960 → 3.956; per-sample distinct-2 0.895 → 0.890; corpus distinct-2 0.329 → 0.317; adjacent repeats 0.65% → 0.48%.
- **The residual is a stronger detector here:** round-trip disagreement vs per-token NLL ρ = +0.52, AUROC 0.81 for the worst-NLL decile. Fixed-point rate 0%.
- **Gating is significant but small,** as on LM1B-small: residual / random gate 0.981 [0.972, 0.990] at q = 0.2.
- **Here no-restrict keeps more of the content** than on LM1B (the generated coarse codes disagree with the encoder less: 0.6–0.8 at levels 2–10, against 0.85–0.97 on LM1B), yet it still does not improve fluency.
- Example edits (seed 0): "there made a little girl" → "there was a little girl"; "so sad and and scared" → "so sad and scared"; `said "ooIs?"` → `said "What happened?"`. Not always consistent: another story's "thin bug named Tim" becomes "thin cat" while his friend stays "Sam the bug".
- `analyze.py` now decodes the TinyStories examples with its BPE tokenizer (decoding them with BERT's vocabulary gave `[unused…]` tokens).
- The NCM README links an OWT checkpoint, but `hazyresearch/ncm-checkpoints` has only `owt/TODO.md`.
- No paper TinyStories number was checked against this baseline.

**Step 4: is lower GPT-2 PPL better text?** (`experiments/lm1b_quality.py`, medium, 3 seeds × 256.)
- **Human anchor:** 768 held-out LM1B validation chunks score gen. PPL **53.5** under the same FLM protocol (Qwen2.5-1.5B: **40.3**), entropy 4.311, corpus distinct-2 0.718 per 256 chunks (NCM: about 0.58). One V-cycle (68.0) is still well above human; it repairs, it does not over-smooth. See the iteration results below for where that changes.
- **Second scorer, a different model family** (Qwen2.5-1.5B, text without special tokens, ≤ 256 tokens): single pass 105.2, no-restrict m=6 102.6, **V-cycle m=6 74.9**; V-cycle / no-restrict **0.730 [0.713, 0.747]** (m=8: 0.840 [0.825, 0.854]). Larger than under GPT-2: the gain is not a GPT-2 artifact.
- **MAUVE** vs human chunks (GPT-2-large features, pooled 768 vs 768): single pass 0.458, no-restrict m=6 0.452, **V-cycle m=6 0.626**, V-cycle m=8 0.612, no-restrict m=8 0.475. Per seed (256 vs 256) it is noisy: single pass 0.66 / 0.71 / 0.71 vs V-cycle 0.64 / 0.73 / 0.84.
- **MAUVE uncertainty (`experiments/lm1b_mauve.py`): directional, not significant.**
  - A first attempt bootstrapped *with replacement* and was discarded. Its intervals excluded their own point estimates (V-cycle m=6: 0.626, interval [0.406, 0.599]): duplicated points distort MAUVE's k-means quantization and bias it down.
  - Replacement: 50 rounds of paired half-size subsamples without replacement, fresh k-means seed each round. V-cycle − no-restrict, m=6: full sets +0.174; half-size mean +0.058 [−0.103, +0.189]; the V-cycle is ahead in 80% of rounds. At m=8: +0.138; +0.037 [−0.147, +0.183]; 66%. V-cycle − single pass, m=6: +0.098 [−0.052, +0.239], 88%.
  - Conclusion: MAUVE agrees in direction with both perplexity scorers, but 768 samples cannot resolve it. That would need several thousand per variant.

**Step 2: iterating the V-cycle** (`experiments/lm1b_converge.py`, seed 42, 256 samples, 6 cycles; GPU draws, so cycle 1 differs from the CPU sweep: small m=6 82.7 here vs 79.9 there).

| | Cycle 0 → 1 → 3 → 6: gen. PPL | Corpus distinct-2, cycle 0 → 6 | Token agreement with draft / previous cycle, cycle 6 | Kept-level round-trip disagreement, cycle 0 → 1 → 6 |
|---|---|---|---|---|
| small, m=6 | 95.8 → 82.7 → 75.8 → 72.1 | 0.505 → 0.441 (−13%) | 0.14 / 0.49 (no-restrict: 0.15) | 0.94 → 0.69 → 0.55 |
| small, m=8 | 95.8 → 83.2 → 72.1 → 66.4 | 0.505 → 0.457 (−10%) | 0.55 / 0.89 | 0.90 → 0.43 → 0.24 |
| medium, m=6 | 90.7 → 70.1 → 58.7 → 51.9 | 0.586 → 0.510 (−13%) | 0.27 / 0.70 (no-restrict: 0.21) | 0.87 → 0.55 → 0.38 |
| medium, m=8 | 90.7 → 78.5 → 70.0 → 65.1 | 0.586 → 0.552 (−6%) | 0.73 / 0.95 | 0.84 → 0.28 → 0.12 |

- **No exact fixed points** (0% of samples at every cycle; at most 2.7% have the kept coarse levels exactly round-trip). But the *graded* residual contracts, roughly geometrically, toward a floor: multigrid-like.
- **The restart level decides the regime.** From m=8 the chain converges: successive cycles agree on 95% of tokens (medium), most of the draft survives, and the diversity cost stays at 6%. From m=6 it keeps drifting: after 6 cycles the text shares no more with the draft than the no-restrict control does, diversity falls 13%, and medium's PPL (51.9) ends *below* human text (53.5), a sign of over-smoothing.
- **Still beyond the temperature frontier.** NCM cannot buy lower PPL with temperature: T=0.9 gives 82.5, T=0.8 gives 82.6 (with distinct-2 0.464 and 0.433). At distinct-2 0.464, four V-cycles from m=8 reach 70.0.
- `analyze.py` adds a table per run and `results/lm1b_{size}_converge.png`.

**Step 3: residual-gated V-cycle on text** (computed from the diagnostic caches, no new sampling; gate = top-q by level-local residual summed over levels, as in Sudoku Phase 2).
- **Small (3 seeds):** the residual gate beats a random gate at equal compute, but only by about 1.5% PPL: ratio 0.985 [0.974, 0.994] at q = 0.2 (89.4 vs 90.8). Cycling the top 20% captures 30% of the full log-PPL gain; an oracle gate on the sample's own GPT-2 NLL captures 47%.
- **Medium (3 seeds):** not significant: 0.991 [0.980, 1.002] at q = 0.2.
- **Gating is no better than a finer restart level** (small, seed 42): at +2 NFE, every sample from m=10 gives 88.2 and the top third from m=6 gives 88.1; at +4 NFE, every sample from m=8 gives 82.8 and the top two thirds from m=6 give 83.9.
- **Null result:** on text the V-cycle helps nearly every sample, so a gate cannot concentrate the gain. The residual remains a useful detector (AUROC 0.64–0.74) but not a useful compute gate.

**Step 5 groundwork.** OWT has no released checkpoint (`ncm-checkpoints` holds only `owt/TODO.md`). TinyStories is released. `LevelDecoder` now supports `query_level0: false` (offsets from level 1; heads, RoPE and post-upsample convs indexed k − 1, as in `generate()`). `load()` points BPE tokenizers at their files. The text helpers follow the config's `eval.metric` (`genppl` for TinyStories: cut at the first EOS, whole text scored). The BPE pickles were checked with `pickletools`: dicts of int → bytes and lists of byte tuples, no import or call opcodes. On LM1B paths the new code is identical to the old (checked on random logits and ids).

### 2026-10-21 (phase A: conditional tasks on the laptop, tooling for a rented GPU)

**Code.**
- `vcycle/cond.py` and `experiments/cond_vcycle.py`: XSum and WMT14, sampled exactly as `scripts/eval_shard.py` and scored as `scripts/eval_merge.py` (ROUGE-1/2/L plus GPT-2 gen. PPL; sacreBLEU, lowercased BLEU, chrF, length ratio). Paired bootstraps over documents; BLEU's is exact, from per-sentence sufficient statistics (`bleu_from_stats` reproduces sacreBLEU's corpus score).
- `LevelDecoder`:
  - `predict_level0` (WMT14 reads level 0 off the prompt's last non-pad positions);
  - an opt-in pad-aware prompt cache (`cache_padded=True`). It is off by default, so the LM1B and TinyStories paths are unchanged.
- **Checks.** Both decoders, cached and uncached, match `generate()` code for code: XSum 32/32 and WMT14 64/64 documents on the GPU.
- Token picking moved to a shared `pick_tokens` function, identical to the 78ff268 code under LM1B and XSum settings (random-input test).
- **The released conditional configs decode greedily** (`top_k: 1`). Without re-encoding, the same regeneration therefore reproduces the single pass exactly (checked: 100% of documents at every level), so the single pass is the equal-compute control.
- **Surprise: the restriction must re-encode the output in canonical form.** After the first EOS, the picked ids are decoder noise (XSum: only 39% of the tail is EOS, because the frequency and adjacency penalties push the picker off repeated EOS). Round trip on 48 outputs per task, share of texts reproduced exactly:

  | Form | XSum | WMT14 |
  |---|---|---|
  | Raw ids | 79% | 40% |
  | Text + EOS + PAD (the released tokenizer pads with PAD) | 79% | 92% |
  | Text + EOS repeated | 81% | 21% |
  | Text + PAD | 71% | 77% |

  `Conditional.canonical` re-encodes text + EOS + PAD; `--raw_restrict` keeps the raw form as an ablation.

**WMT14 German → English, all 3,003 sentences** (laptop GPU, greedy as released). Single pass: BLEU 21.73, lowercased 22.62, chrF 49.59, length ratio 0.938.

| Restart level (extra NFE per cycle) | Δ BLEU after 1 / 2 / 3 cycles [95% CI] | Sentences fixed after cycle 2 / 3 |
|---|---|---|
| m = 3 (+8) | +0.32 [+0.05, +0.59] / +0.49 [+0.21, +0.77] / +0.55 [+0.25, +0.83] | 18% / 33% |
| m = 5 (+6) | +0.29 [+0.10, +0.50] / +0.23 [+0.02, +0.45] / +0.27 [+0.05, +0.51] | 23% / 41% |
| m = 7 (+4) | −0.09 [−0.21, +0.01] / −0.04 / −0.09 (sentence chrF −0.18 [−0.28, −0.09]) | 49% / 74% |

- **Small but significant gains from coarse and middle restarts; none from the finest.** Under greedy decoding the V-cycle is a deterministic map, and it converges: the share of fixed points grows every cycle. Length is unchanged.
- **Raw re-encoding (m = 5, one cycle): −4.28 BLEU [−4.77, −3.81]**, length ratio 1.053. The noisy tail lengthens and degrades the translations. Canonical form is essential.
- **The residual does not flag bad translations** (worst-decile sentence chrF: AUROC 0.51). With the prompt fixing the content, the reference score mostly reflects content choices.
- **Example (sentence 2).** The single pass gives "…in the new B 33 pedestrian traffic was moved into operation in the village premises". One cycle from m = 5 gives "…the new B 33 pedestrian light was put into operation at the village park", but "another another" becomes "one one".

**XSum, smoke test only** (200 articles; the full 11,334 run is planned for the rented GPU). ROUGE stays flat (R-1 34.06 → 33.83–34.03, R-2 12.44 → 12.69–13.16). Summary perplexity under GPT-2 falls 78.9 → 61–67. Content kept, fluency up.

**LLM judge** (`experiments/judge.py`). A local open model; next-token probabilities for "A" and "B"; both orders averaged; order agreement and letter-A share reported as diagnostics. The plumbing was tested with Qwen2.5-0.5B-Instruct, which answers "A" 92–99% of the time (order agreement 0%), so it is useless as a judge, and its outputs were deleted. The real judge is Qwen2.5-7B-Instruct on the rented GPU. Its human-vs-NCM sanity check runs first and decides whether its verdicts mean anything.

**Rented-GPU tooling** (`experiments/box/`):
- `setup.sh`, `queue.sh`, `push.sh`, `pull.sh`. Code and results travel by SSH from the laptop; there is no git and no token on the box.
- `queue.sh` runs a scoring check first, then two lanes in parallel: 17 more LM1B-medium seeds (20 in all) plus MAUVE features and quality checks, and full XSum. The judge runs after both.
- The judged comparisons are fixed in advance: one cycle from m = 6 (LM1B) or m = 5 (XSum, WMT14).
- `MAX_MINUTES` schedules a hard shutdown, a cost cap.
- MAUVE's CPU-only statistics run on the laptop from the pulled features.
- `experiments/check_gpu_scoring.py` re-scores committed samples on a new machine.
- `*.sh` files are forced to LF (`.gitattributes`), since CRLF breaks bash on Linux.
- The large resumable state (`results/cache/*_vcycle*.pt`, 28 MB for WMT14, about 85 MB for XSum) is not committed. Every reported number and three examples per variant are in the JSON outputs.

### 2026-10-21/22 (phase B: full XSum, 20 seeds, MAUVE at scale, the LLM judge)

**Where it ran.**
- **Laptop (RTX 4050, 6 GB):** XSum on all 11,334 articles plus the raw-restriction ablation, LM1B-medium seeds 2–18, and the MAUVE features and statistics.
- **One AWS g5.2xlarge (A10G, 24 GB, us-east-1d):** the GPU scoring check, the five judge runs (`experiments/box/judges.sh`), then the quality checks on 20 seeds (`after_judges.sh`). The quality job was moved there from the laptop to free the laptop's CPU.
- `analyze.py` ran on the laptop: on the box it stopped at the TinyStories config, which was not copied.

**Memory-paging fix.** The first XSum passes took about 60 minutes each. GPT-2 large stayed on the GPU after scoring, the GPU sat at 5.5 of 6 GB, and Windows silently paged GPU memory to RAM. `Conditional.genppl_nll` now moves GPT-2 to the GPU only while it scores. With `XSUM_BS=16` a pass takes about 7.5 minutes.

**XSum, all 11,334 articles** (greedy as released; single pass R-1/R-2/R-L 34.92/12.71/27.93, GPT-2 PPL 82.6, 20.7 words):
- Three cycles from m = 5 (+12 NFE): R-1/R-2/R-L +0.24 / +0.25 / +0.19, all CIs above zero; PPL 62.5 (×0.756 [0.749, 0.764]); length unchanged. 47% of summaries are fixed points by cycle 3.
- One cycle from m = 5: +0.19 / +0.18 / +0.15, PPL ×0.828.
- m = 3 is weaker on ROUGE (+0.13 / +0.20 / +0.11 after three cycles) with a similar PPL drop.
- **Raw re-encoding** (m = 5, one cycle) costs little here (R-2 +0.13 instead of +0.18; 20.9 words), as the 79% round trip for both forms predicted.
- **The residual barely flags the worst summaries** (AUROC 0.60–0.63).

**LM1B-medium, 20 seeds × 256 samples.** Single pass 88.8, no restriction (m = 6) 88.4, V-cycle **68.6**: ratio 0.776 [0.769, 0.783], better on 20 of 20 seeds. At m = 8: 76.8 (0.866 [0.861, 0.871]). The 3-seed estimates (0.783, 0.871) held.

**MAUVE at 5,120 samples per variant** (5,120 human chunks, GPT-2-large features):
- Single pass 0.312, no restriction 0.307, **V-cycle 0.410**.
- Over 20 rounds of paired half-size subsamples, the V-cycle leads no restriction by **+0.077 [+0.024, +0.130]**, ahead in 20 of 20 rounds. At 768 samples the same comparison was +0.06 [−0.10, +0.19].
- Absolute MAUVE falls as the sample count grows (more clusters), so only differences compare across sizes.

**Quality on 20 seeds.**
- Qwen2.5-1.5B ratio vs no restriction: 0.724 [0.717, 0.730] at m = 6 and 0.837 [0.832, 0.843] at m = 8.
- Human text: 53.5 (GPT-2) and 40.8 (Qwen).
- One V-cycle closes **51%** of the log-perplexity gap between NCM and human text: (ln 88.8 − ln 68.6) / (ln 88.8 − ln 53.5). **Correction:** a status digest to the user said 57%, which is the same ratio on the linear perplexity scale; the log scale is the right one.

**LLM judge** (Qwen2.5-7B-Instruct, both orders, mean of P(A) / (P(A) + P(B)) over the orders):

| A vs B | Items | Mean preference for A [95% CI] | Share preferring A | Order agreement | Letter-A share |
|---|---|---|---|---|---|
| human vs NCM single pass (sanity check) | 5,120 | 0.669 [0.664, 0.674] | 97% | 33% | 0.18 |
| V-cycle m = 6 vs no restriction m = 6 | 5,120 | 0.589 [0.581, 0.597] | 64% | 48% | 0.44 |
| V-cycle m = 6 vs single pass | 5,120 | 0.633 [0.626, 0.639] | 76% | 37% | 0.26 |
| XSum: one cycle from m = 5 vs single pass | 2,000 | 0.526 [0.516, 0.535] | 44% | 23% | 0.80 |
| WMT14: one cycle from m = 5 vs single pass | 3,003 | 0.515 [0.506, 0.525] | 44% | 33% | 0.72 |

- **The sanity check passes**, so its verdicts mean something.
- **Strong position bias, in opposite directions:** the judge favors the second text on LM1B and the first on XSum and WMT14. Averaging over both orders removes it from the means, but single verdicts are noisy (order agreement 23–48%).
- **On free-running text the judge agrees with the other three measures.** On the prompted tasks it is a near-tie: the mean leans to the V-cycle while the share leans to the single pass, which is what a small fluency polish on mostly identical outputs looks like.
- `judge.py` now asks the model for the logits of the answer position only (`logits_to_keep=1`; same outputs, much less memory).

**AWS operations.**
- **Capacity:** g7e, g6e and g6 had no capacity in us-east-1. A script tried 20 type and zone pairs every 2 minutes, with a 60-minute limit, and launched the first that worked: a g5.2xlarge.
- **Launch:** explicit parameters, not the launch template (its disk names the old image's snapshot): DLAMI base Ubuntu 24.04 of 2026-10-16, a 150 GB gp3 disk deleted on termination, shutdown behavior "terminate", and a hard shutdown scheduled on the box.
- **Cost and cleanup:** ran 21:43–23:29 UTC, about $2.15. Results were pulled selectively (the box's MAUVE and quality inputs were older than the laptop's), then the instance was terminated by hand. Afterwards no instance and no volume was left in us-east-1.

**Verdict after phase B.**
- **Free-running text:** a clear, robust win. Four independent measures agree: two scorers from different model families, MAUVE and a judge. Medium plus one V-cycle beats the 2.3× larger model's single pass at less compute.
- **Prompted tasks:** a small, significant polish on automatic metrics that a judge barely notices; classifier-free guidance is the bigger lever on WMT14.
- **Still missing:** a human evaluation and the CFG and MBR comparisons at equal compute.

### 2026-10-22 (literature search, zero-compute pilots, error diffusion, Sudoku symmetry frames)

**Literature search for a partner method** (two rounds of parallel searches over multigrid, Monte Carlo, coding
theory, combinatorial search, inverse problems, NLP decoding, physics and control, plus a prior-art check of VAR,
RVQ-audio, MaskGIT and diffusion-LM methods). Every candidate was checked against the full text of the NCM paper.
- Reranking the draft and V-cycle outputs by NCM's own ELBO (maximum mutual information decoding; Li et al., 2016)
  was the best-supported candidate on WMT14: a GPT-2-small per-sentence choice between draft and V-cycle gave
  21.73 → 22.56 cased BLEU. It was **rejected as extra compute**; the remaining work targeted zero-compute ideas.
- Closest prior art found: DiverseVAR scale-travel (latent re-quantization), TTS-VAR and beam search over scales,
  VPG (guidance with a corrupted prefix), Lattice Deduction Transformers (random symmetry frames for Sudoku), POMO,
  Mixture of Inputs (posterior-mean inputs). Details in `docs/prior_art.md`.

**Error diffusion within a level** (Floyd & Steinberg, 1976; `experiments/ed_sweep.py`, `LevelDecoder.sample_ed`;
run on an AWS g5.2xlarge, about 28 minutes). Stage 1 (3 seeds × 256): best setting γ = 2, ρ = 0.5 gave LM1B-small
×0.994 [0.973, 1.016] and TinyStories ×0.971 [0.949, 0.994]; no setting met the pre-registered bar. Stage 2
(LM1B-medium): V-cycle with error diffusion / V-cycle ×1.003 [0.996, 1.011]. **Dead.**

**Zero-compute pilots** (single seed, exploratory; scripts kept outside the repository). All negative on text:
Voronoi-consistent sampling (×0.998), within-level Potts and copula couplings fitted on real codes (×0.96–0.995, CIs
include 1; within-level dependence is only 0.3–0.6% of the code NLL), inversion of the training-time corruption
channel (on NCM's temperature frontier), small-model draft + restriction + large-model fine levels (0.961 of the
frontier, n.s.). On Sudoku: neighbourhood search and keep-best acceptance 0.0 ± 0.3 points; limited discrepancy search
+0.65 points. Two Sudoku ideas survived: random symmetry frames and soft (posterior-mean) decisions.

**Sudoku stacking run** (`experiments/sudoku_stack.py`; first on an AWS g5.xlarge, about 13 minutes, then the final
run on the laptop GPU; 3 seeds × 2,000 boards per difficulty). Accuracy on hard boards at 14 mean NFE, NCM alone /
with singles propagation:
- best-of-N 89.5 / 94.5; V-cycle retries 90.2 / 94.8; fresh restarts in random frames 92.8 / 96.4;
- symmetry-twisted V-cycle 94.8 / 97.2; soft decisions + twisted V-cycle **96.5 / 97.9**.
- The frame change is the main lever (+3.35 points over best-of-N, NCM alone). The twisted V-cycle adds +2.03 over
  fresh restarts in random frames per NFE but ties them at matched FLOPs. Soft decisions add +3.75 in a single pass
  (held-out half +3.80) and +1.70 on top of the twisted retries.
- No wrong board was accepted in any run. The A10G replication agrees with the laptop run to within 0.2 points.

**Write-up.** README rewritten in a paper-like structure; figures moved to `figures/` and drawn by
`experiments/figures.py` with one style (`vcycle/plotstyle.py`, PNG at 300 dpi and PDF); `experiments/analyze.py` now
writes tables only, including the Sudoku stacking and error-diffusion sections. The old charts under `results/*.png`
were removed.
