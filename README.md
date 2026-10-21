# Multigrid V-cycle decoding for Next-Concept Models (training-free)

This repo tests whether samples from **Next-Concept Modeling** (NCM; Kumbong, Khare et al.,
[arXiv:2610.14162](https://arxiv.org/abs/2610.14162), code at
[HazyResearch/ncm](https://github.com/HazyResearch/ncm)) improve when inference adds
multigrid-style correction cycles. The released checkpoints are used unchanged; nothing is retrained.

NCM decodes once, coarse to fine. A V-cycle adds the missing "restriction" step:

1. Re-encode the decoded output with the frozen tokenizer encoder.
2. Compare the re-encoded codes with the generated ones, level by level. This difference is the *residual*.
3. Keep the re-encoded coarse levels and resample the finer ones, then repeat.

The full research plan and the lab notebook are in [`PLAN.md`](PLAN.md). The prior-art check is in
[`docs/prior_art.md`](docs/prior_art.md).

## Results

| | Result |
|---|---|
| **Text (LM1B)** | **Small** (3 seeds × 256 samples): one V-cycle cuts generative perplexity from 93.8 to **79.9** at +6 NFE. Resampling the same levels *without* re-encoding costs the same and gives 93.2, so the re-encode step is what matters (paired ratio **0.857**, 95% CI [0.836, 0.876]). Iterating keeps helping (72.8 after 3 cycles), entropy *rises*, and the V-cycle beats simply sampling colder. **Medium** (1 seed): 90.7 → **69.6**, against 87.5 for no restriction (ratio **0.796** [0.765, 0.825]). |
| **Residual = free error detector** | NCM samples are **never** fixed points of the tokenizer round trip (0% of 6,000 Sudoku boards and 1,024 texts). The *amount* of disagreement flags wrong Sudoku boards with AUROC **0.92** on hard, beating the decoder's own confidence (0.87). On text it flags the worst-perplexity tenth of samples with AUROC 0.74 on LM1B-small (0.64 on medium). |
| **Reproduction** | Sudoku: 99.4 / 96.4 / **86.3%** solved on easy / medium / hard (paper, hard: 86.1%). LM1B-small: gen. PPL 95.8 / 94.6 / 91.0 over three seeds, H = 4.32 (paper's NCM point: about 92, H = 4.31). |
| **Sudoku correction** | With the Sudoku checker deciding which boards to retry, V-cycle retries from level 5 beat fresh resampling (best-of-N) by **+1.2 points** at equal compute (90.4% vs 89.3% at 14 NFE; 95% CI [+0.6, +1.7]; 3 seeds × 2,000 hard boards). Most of the gain comes from keeping the coarse levels. Here the re-encode step adds only +0.3 to +0.5 points, with a CI that includes 0. |

### 1. Text: the V-cycle repairs NCM samples, and the repair comes from the restriction step

**Setup** (`experiments/lm1b_diagnostic.py`, `experiments/lm1b_sweep.py`):
- Samples are drawn exactly as NCM's own LM1B evaluation does it. Level 0 comes from the released unigram, the levels are sampled at T = 1, and tokens are picked with the released frequency and adjacency penalties. Our decoder reproduces `generate()` code for code under the same seed.
- Scoring is the paper's FLM protocol: corpus perplexity under GPT-2 large.
- **V-cycle:** re-encode the sample, keep the encoder's codes for levels < m, and resample levels m..11 at T = 1.
- **No restriction:** keep the *generated* codes for levels < m and resample the same levels. It costs the same.

| Variant (LM1B-small, 3 seeds × 256) | Extra NFE | Gen. PPL ↓ (per seed) | Mean | Entropy ↑ | Adjacent repeats |
|---|---|---|---|---|---|
| NCM single pass | 0 | 94.6, 91.0, 95.8 | 93.8 | 4.317 | 0.47% |
| No restriction, m = 6 | +6 | 94.7, 91.1, 93.8 | 93.2 | 4.314 | 0.45% |
| **V-cycle, m = 6** | **+6** | **80.7, 78.9, 79.9** | **79.9** | **4.344** | 0.45% |
| V-cycle, m = 3 | +9 | 95.0, 88.9, 94.2 | 92.7 | 4.324 | 0.57% |

Paired bootstrap over the 768 samples: V-cycle (m = 6) / no restriction = **0.857 [0.836, 0.876]**; V-cycle / single pass = 0.852 [0.836, 0.868]. Restarting from level 3 has no significant effect (0.985 [0.951, 1.024]).

**A bigger model shows the same effect, and it is stronger** (LM1B-medium, 317M parameters, seed 42, 256 samples):

| Variant | Extra NFE | Gen. PPL ↓ | Ratio vs no restriction [95% CI] | Entropy | Corpus distinct-2 |
|---|---|---|---|---|---|
| NCM single pass | 0 | 90.7 | – | 4.343 | 0.586 |
| No restriction, m = 6 | +6 | 87.5 | – | 4.336 | 0.583 |
| **V-cycle, m = 6** | **+6** | **69.6** | **0.796 [0.765, 0.825]** | 4.346 | 0.554 |
| No restriction, m = 8 | +4 | 89.4 | – | 4.340 | 0.583 |
| **V-cycle, m = 8** | **+4** | **78.9** | **0.883 [0.859, 0.909]** | 4.347 | 0.569 |

![LM1B: V-cycle vs resampling at equal compute](results/lm1b_small_ppl_vs_nfe.png)

The sweep below covers seed 42, restart levels m = 4, 6, 8 and 10, and repeated cycles:
- **Every restart level helps.** The V-cycle beats no restriction at all four levels, with ratios 0.93, 0.87, 0.87 and 0.92.
- **Repeating helps further.** Two cycles from level 8 (+8 NFE) give 75.2 (0.785 × no restriction). Three cycles from level 6 give 72.8.
- **It beats colder sampling.** Lowering NCM's sampling temperature also reduces perplexity (T = 0.9: 82.5), but it loses diversity: entropy falls to 4.289, corpus distinct-2 to 0.464 and adjacent repeats rise to 0.61%. At the same perplexity (82.8, from level 8), the V-cycle has entropy 4.337, distinct-2 0.489 and 0.35% repeats. Its points lie beyond NCM's own temperature frontier.

![LM1B: quality vs diversity](results/lm1b_small_ppl_vs_entropy.png)

**Why it works.** The generator's codes are mostly not the codes the encoder would produce for the same text: 86–100% of positions at levels 0–6 disagree, and about 25% at the finest level. The decoder still reads them; its released config trains it with heavy code corruption, which is a likely reason. But the generator conditions its fine levels on these off-manifold coarse codes.
- **Re-encoding** puts the coarse levels back on the encoder's manifold before the fine levels are regenerated. This works as a training-free fix for exposure bias.
- **Restarting from level 3 doesn't help,** because it regenerates levels 3–5 off-manifold again.
- **The examples show an editing pass** (`results/SUMMARY.md`). The V-cycle keeps the draft's content and repairs local fluency: *"included the students at least 30, 000 in more than five years"* becomes *"the students lost at least 30, 000 or more than five years"*. Resampling from the *generated* coarse codes instead produces an unrelated text. In generated samples, the coarse codes barely pin down content.

**Caveats.**
- Cross-sample diversity drops somewhat: corpus distinct-2 falls about 5–6% at m = 6 (0.505 → 0.477 on small, 0.586 → 0.554 on medium). That is less than colder sampling costs.
- Coverage is two model sizes of one dataset, and LM1B-medium has a single seed.
- Quality is measured only by GPT-2-large perplexity under the paper's protocol: no MAUVE and no human evaluation.

### 2. The round-trip residual is a training-free error detector

A sample is a *fixed point* if `encode(decode(r)) == r`. The plan's yes/no test carries no signal, because **no NCM sample is a fixed point**. Yet `decode(encode(x)) == x` holds for every Sudoku board: the generator writes a non-canonical code for the board it means. The *amount* of disagreement does carry signal, especially the **level-local** version. At each level it asks which code the quantizer would pick given the *generated* coarser levels, so one miss does not cascade.

| Sudoku (2,000 boards each) | Solved | Residual AUROC, level-local (round trip) | Decoder confidence | Generator confidence |
|---|---|---|---|---|
| easy | 99.4% | **0.872** (0.797) | 0.848 | 0.701 |
| medium | 96.4% | **0.883** (0.867) | 0.847 | 0.627 |
| hard | 86.3% | **0.920** (0.859) | 0.875 | 0.730 |

**The residual also localizes errors** (`results/phase1b_localization_hard.json`). At level 4 (9 positions, about one board row each), the generator is 100% confident on every board. Yet when the residual fires, the covered span contains a wrong cell **97.5%** of the time (base rate 43%). Clean spans almost never fire (0.5%).

### 3. Sudoku: correction cycles help a little, mostly by keeping the coarse levels

**Setup** (`experiments/phase3_controls.py`):
- Every method starts from NCM's greedy pass, which costs 12 NFE and solves 86.3%.
- The Sudoku checker accepts a board if it is a valid grid that keeps every clue. It never sees the solution. Boards it rejects get up to 8 retry rounds at T = 1.
- Methods are compared when their *mean* compute per board reaches the same budget.
- The 95% intervals are paired bootstraps over boards, pooled over 3 seeds.

| Accuracy at a compute budget (hard, 2,000 boards × 3 seeds) | 13 NFE | 14 NFE | 15 NFE | 16 NFE |
|---|---|---|---|---|
| Fresh restart (best-of-N with the checker) | 87.8 | 89.3 | 90.6 | 91.3 |
| No restriction, level 5 (keep generated `r_{<5}`) | 88.6 | 90.2 | 91.0 | 91.8 |
| **V-cycle, level 5** (keep re-encoded `r'_{<5}`) | **88.7** | **90.4** | **91.5** | **92.3** |
| V-cycle, level chosen by the residual (V1-accept) | 88.5 | 89.8 | 90.8 | 91.6 |
| V2: masked resample at level 5 | 88.1 | 89.3 | 90.1 | 90.7 |

| Paired difference, percentage points [95% CI] | 13 NFE | 14 NFE | 15 NFE | 16 NFE |
|---|---|---|---|---|
| V-cycle (level 5) − fresh restart | +0.93 [+0.51, +1.33] | +1.19 [+0.62, +1.73] | +0.93 [+0.33, +1.56] | +0.92 [+0.32, +1.48] |
| V-cycle (level 5) − no restriction (level 5) | +0.14 [−0.16, +0.43] | +0.27 [−0.20, +0.71] | +0.45 [−0.09, +0.93] | +0.47 [−0.09, +1.01] |
| V-cycle (level 7) − no restriction (level 7) | +0.56 [+0.16, +0.99] | +0.67 [+0.20, +1.21] | +0.57 [+0.07, +1.12] | +0.68 [+0.14, +1.19] |
| V2 masked (level 5) − V-cycle (level 5) | −0.66 [−0.94, −0.38] | −1.14 [−1.55, −0.70] | −1.39 [−1.84, −0.93] | −1.54 [−2.04, −1.06] |

![Sudoku (hard): accuracy vs compute](results/sudoku_hard_n2000_T1.0_acc_vs_nfe.png)

- **Partial restarts beat best-of-N.** Restarting from level 5 instead of from scratch is worth about +1 point at equal compute.
- **On Sudoku, the restriction step adds only a little.** It gives +0.3 to +0.5 points at level 5, with an interval that includes 0. At level 7 it is a significant +0.6, but level 7 is a poor restart point overall. This fits the LM1B mechanism: Sudoku's generated codes are mostly *on* the encoder's manifold (0–20% disagreement per level on solved boards). So re-encoding changes much less than on text, where 86–100% of coarse codes are off-manifold.
- **Masked resampling (V2) hurts.** Resampling only the disagreeing positions at the restart level is significantly worse than resampling the whole level.
- **The residual-chosen restart level (V1-accept) is competitive** but does not beat the best fixed level.
- **Greedy V-cycles** (Phase 2: temperature 0, no checker, 1,000 hard boards, baseline 84.3%) move little.
  - The plan's V1 at τ ≤ 0.1 changes nothing, and neither does the no-restriction control. Both were predicted.
  - Restarting from level 6 fixes 10 boards and breaks none in one cycle (85.3%, +6 NFE), reaching 85.5% after three.
  - Cycling only the 20% of boards with the highest residual keeps most of the gain (85.1%) at a fifth of the compute, with no checker at all.
  - With a checker available, stochastic retries (table above) are far stronger.

### What changed from the plan

- **V1 as written is a guaranteed no-op under greedy decoding.** With τ = 0, the levels below j agree, so the same context resamples the same codes. We added *V1-accept*, which also keeps the re-encoded level j.
- **V3 clue projection rarely changes anything.** The greedy model overwrites a clue on only 0.1% of hard boards.
- **The yes/no fixed-point criterion is replaced by graded residuals.**
- **Prior art:** DiverseVAR's "scale-travel" (arXiv:2511.21415) already re-encodes with a multiscale tokenizer and resumes generation at a fixed scale, once, for images. The residual-chosen restart level, the iteration, the controls and the text setting are what is new here. Details are in [`docs/prior_art.md`](docs/prior_art.md).

### Suggested next steps

- **More text coverage.** Run LM1B-medium and -large with more seeds, plus OWT and TinyStories. Add MAUVE or human preference checks alongside gen. perplexity.
- **Iterate to convergence on text.** Measure perplexity and diversity per cycle and the fixed-point rate after each cycle. The ×3 result suggests there is more to gain.
- **Use the residual as a gate on text.** Cycle only the flagged samples, as Phase 2 does for Sudoku, to cut the extra compute.
- **Conditional tasks.** WMT14 and XSum, where the prompt pins content and the V-cycle's "keep content, fix fluency" behavior should matter most.
- **Training-side follow-up for the NCM authors.** Teacher-force on the generator's own re-encoded codes, so its coarse levels land on the encoder's manifold in the first place.

## How the pieces map

| Multigrid | NCM | Code |
|---|---|---|
| Coarse grid | code levels k = 0..K-1 (Sudoku: lengths 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89) | `LevelDecoder.levels` |
| Prolongation | generate levels k+1..K-1 given `ẑ^(k)` | `LevelDecoder.run(prefix, start=j, codes=...)` |
| Restriction | re-encode tokens with the frozen encoder + quantizer | `LevelDecoder.encode` |
| Residual | per-level disagreement between `r_k` and the re-encoded `r'_k` | `vcycle.vcycle.disagreement` |
| Level-local residual | at each level, the code the quantizer would pick given the *generated* coarser levels, so a miss does not cascade | `LevelDecoder.encode_given_coarse` |
| V-cycle | decode → re-encode → keep `r'_{<m}` → resample m..K-1 → decode | `vcycle.vcycle.cycle` |

## Reproducing

```bash
./setup.sh                          # NCM code (pinned), Sudoku checkpoints and boards
experiments/run_sudoku.sh           # Phases 0, 1, 1b, 2, 3 on Sudoku, one job at a time
hf download hazyresearch/ncm-checkpoints lm1b/tokenizer.pt lm1b/small/generator.pt lm1b/small/config.yaml --local-dir artifacts
experiments/run_lm1b.sh small       # LM1B: 3 seeds, residual diagnostic + V-cycle vs no restriction
python experiments/lm1b_sweep.py --size small --seed 42   # restart levels, repeated cycles, lower temperatures
python experiments/analyze.py       # writes results/SUMMARY.md, results/summary.json and the charts
```

- **Scoring model:** LM1B scoring downloads GPT-2 large (about 3 GB) on first use.
- **Interruptions:** Phase 2, Phase 3 and `lm1b_diagnostic.py` checkpoint and resume after one.
- **Full tables** (every variant, every budget, the LM1B examples) are in [`results/SUMMARY.md`](results/SUMMARY.md). Raw numbers are in `results/*.json`.

- **CPU-only works.** Every number here was produced on 4 CPU cores. A greedy Sudoku pass over all 2,000 boards takes about 2.6 minutes with the cached-prompt decoder.
- **GPU:** `vcycle.common.device()` picks CUDA automatically when it is available.
- **Checkpoints** are loaded with `torch.load(weights_only=True)`.

| Path | What it is |
|---|---|
| `vcycle/loading.py` | builds the tokenizer and generator from a released checkpoint directory |
| `vcycle/decoding.py` | `LevelDecoder`: level-by-level sampling, start mid-hierarchy, masked resampling, cached prompt keys/values, encode/decode. It is checked to match `NextConceptTransformer.generate` code for code |
| `vcycle/vcycle.py` | restart-level rules (V1/V4 and V1-accept), masked V2, constraint projection (V3), iterated cycles |
| `vcycle/sudoku.py` | board format, clue projection, metrics (solved, valid, unit and clue violations) |
| `experiments/phase*.py` | one script per phase of the plan; outputs go to `results/*.json` |
| `vcycle/lm1b.py` | LM1B sampling exactly as the released eval does it, FLM-protocol scoring, diversity metrics |
| `experiments/lm1b_diagnostic.py` | LM1B residual diagnostic, plus V-cycle vs no-restriction at equal compute |
| `experiments/lm1b_sweep.py` | restart levels, repeated cycles, lower-temperature baselines |
| `experiments/analyze.py` | tables, paired-bootstrap intervals and the charts |
