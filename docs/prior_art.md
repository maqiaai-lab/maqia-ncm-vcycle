# Prior-art check (plan §11)

Scan date: 2026-10-20. Sources: arXiv, Semantic Scholar, the NCM paper, its blog and its repo.
**Verified** means I opened the arXiv abstract page myself. The other entries come from an
automated search pass and should be checked before they are cited.

## The NCM paper itself (arXiv:2610.14162, verified)

- Decoding is a single coarse-to-fine pass, one forward per level. The abstract says nothing about
  re-encoding, self-correction or revisiting coarse levels.
- According to the search pass, the paper's robustness mechanism is *training-time* conditioning
  corruption (§3.4). App. F.3 names "corruption based on the model's actual errors" as future work,
  and App. I lists "occasional artifacts at coarse-level transitions" as a limitation.
- This repo's code confirms the corruption: the LM1B tokenizer config has `aux.code_corruption`.

## Close

| Work | What it does | How the V-cycle differs |
|---|---|---|
| **DiverseVAR, "scale-travel"**. Park, Phunyaphibarn, Lee, Sung. arXiv:2511.21415 (v1 Nov 2025). **Verified.** | Training-free, for text-to-image VAR. It uses the multiscale autoencoder to extract coarse-scale tokens from a partial result, then resumes generation at an intermediate scale. Its goal is to recover the quality lost to diversity noise. | Runs once, at a fixed scale. There is no disagreement residual, no adaptive restart level, no masked resampling, no iteration to a fixed point and no constraint projection, and it works on images, not text. **This is the precedent for restrict-then-regenerate; cite it.** |
| **RePaint**. Lugmayr et al., CVPR 2022, arXiv:2201.09865 | Inpainting that repeatedly jumps back to a higher noise level and re-denoises, re-imposing the known pixels each time. | Works in continuous diffusion time. Its "restriction" is forward noising rather than an encoder, and its schedule is fixed. |
| **Predict-Project-Renoise**. Rochman-Sharabi & Louppe, arXiv:2601.21033 | Repeats predict → project onto hard constraints → renoise with a pretrained diffusion model. | Continuous and noise-based. This is the continuous analogue of our V3 projection variant. |

## Related

- **VAR self-correction, all training-time or search-based; none revisits a coarse scale chosen by a residual:**
  - Infinity bitwise self-correction (arXiv:2412.04431)
  - SAR (arXiv:2512.06421)
  - LSRS (arXiv:2512.03796)
  - VAR-Scaling (arXiv:2601.07293)
  - AID-VAR (arXiv:2605.24843)
  - SSG (arXiv:2602.05534)
- **HMAR** (Kumbong et al., CVPR 2025, arXiv:2506.04421), by NCM's lead author. It does multi-step masked sampling *within* a scale and never re-encodes or revisits earlier scales.
- **reAR** (arXiv:2510.04450) is a training-time regularizer that makes the generator and tokenizer consistent. It is the closest relative of our "fixed point" diagnostic, but it is used as a loss, not at inference.
- **Watermarking autoregressive image generation** (arXiv:2506.16349) defines "reverse cycle-consistency": the token match after decode → re-tokenize. It is a precedent for the residual metric, but not for sampling.
- **Text diffusion that revisits tokens:**
  - ReMDM (arXiv:2503.00307)
  - Informed correctors (arXiv:2407.21243)
  - P2 path planning (arXiv:2502.03540)
  - RemeDi, PRISM, Corrective DLMs

  All of these revisit tokens in a flat sequence. None restarts from a coarse level of a code hierarchy.
- **Multilevel background:**
  - "Multi-scale AR models are Laplacian, Discrete, and Latent Diffusion Models in Disguise" (arXiv:2510.02826)
  - Multilevel Diffusion (arXiv:2303.04772)
  - V-cycle *training* of transformers (arXiv:2404.07999)
  - Multigrid Monte Carlo (Goodman & Sokal)

## Novelty assessment

- **Not new:** "re-encode with the frozen multiscale encoder, keep the coarse levels, regenerate the fine levels." DiverseVAR's scale-travel does exactly this, once and at a fixed level, for images. Present the V-cycle as a residual-driven, iterated generalization of it.
- **Probably new:**
  - choosing the restart level from the per-level round-trip disagreement;
  - resampling only the disagreeing positions;
  - iterating toward a fixed point, framed as multigrid;
  - applying any of this to text or reasoning (NCM).
- **Probably new as a diagnostic:** using the round-trip residual as a training-free error detector. Our measured AUROC is 0.92 on hard Sudoku, against 0.87 for decoder confidence. The nearest prior uses of the round-trip are a training loss (reAR) and a watermark test.
- **Constraint projection before re-encoding:** frame it as the discrete, multiscale analogue of RePaint and PPR.
