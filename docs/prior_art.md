# Prior art

Scans: 2026-10-20 (V-cycle decoding) and 2026-10-22 (the zero-compute extensions and the Sudoku methods). Sources:
arXiv, Semantic Scholar, the NCM paper, its blog and its repository. *Verified* means that the abstract or full text
was read; other entries come from search results and should be checked before they are cited.

## 1. The NCM paper (arXiv:2610.14162, verified)

- Decoding is a single coarse-to-fine pass, one generator evaluation per level. The paper does not mention
  re-encoding, self-correction, revisiting coarse levels, multigrid methods, symmetry augmentation or soft code
  inputs (case-insensitive search of the full text).
- Robustness to its own coarse-level errors comes from *training-time* conditioning corruption (§3.4). Appendix F.3
  names "corruption based on the model's actual errors" as future work; Appendix I lists "occasional artifacts at
  coarse-level transitions" as a limitation.
- Inference-time improvements in the paper are classifier-free guidance (in the style of VAR) and minimum-Bayes-risk
  selection (Appendix F.4). Appendix A derives an ELBO, which is used only to motivate the training objective.

## 2. Restrict-then-regenerate

| Work | What it does | Difference from the V-cycle |
|---|---|---|
| **DiverseVAR, "scale-travel"** (Park, Phunyaphibarn, Lee, Sung; arXiv:2511.21415, v1 Nov. 2025; verified) | Training-free, for text-to-image VAR. Re-quantizes an accumulated latent with the multiscale quantizer and resumes generation at an intermediate scale, to recover quality lost to diversity noise. | Re-quantizes in latent space, once, at a fixed scale, for images. The V-cycle decodes to discrete text and re-encodes with the full encoder, and is evaluated with equal-compute controls, iteration and a residual diagnostic. **The closest precedent; cite it.** |
| **GAZER** (arXiv:2606.22550; abstract verified) | Rewinds a multiscale image generator to an earlier scale when a multimodal LLM flags an error, then resamples. | External verifier; no re-encoding of the output. |
| **RePaint** (Lugmayr et al., CVPR 2022; arXiv:2201.09865) | Inpainting that repeatedly jumps back to a higher noise level and re-denoises. | Continuous diffusion time; the "restriction" is forward noising with a fixed schedule. |
| **Predict–project–renoise** (Rochman-Sharabi & Louppe; arXiv:2601.21033) | Repeats prediction, projection onto hard constraints and renoising with a pretrained diffusion model. | Continuous and noise-based; the analogue of the constraint-projection variant (V3). |

## 3. Self-correction and search in multiscale and masked generators

- **Training-time or trained modules:** Infinity's bitwise self-correction (arXiv:2412.04431), SAR (arXiv:2512.06421),
  reAR (arXiv:2510.04450, a generator–tokenizer consistency loss), Logit Refiner (arXiv:2609.11804, which describes
  within-scale parallel decoding as a mean-field approximation).
- **Search over scales with a verifier:** TTS-VAR (arXiv:2507.18537), beam search over Infinity's scales
  (arXiv:2510.16751), LSRS (arXiv:2512.03796), VAR-Scaling (arXiv:2601.07293).
- **Guidance variants:** VPG (arXiv:2605.30317) contrasts a model's own generated prefix with a corrupted copy, SWG
  (arXiv:2511.22991), SSG (arXiv:2602.05534).
- **Same author line:** HMAR (Kumbong et al., CVPR 2025; arXiv:2506.04421) samples with several masked steps
  *within* a scale and never revisits earlier scales.
- **Masked and diffusion language models that revisit tokens:** ReMDM (arXiv:2503.00307), informed correctors
  (arXiv:2407.21243), path planning (arXiv:2502.03540). All revisit tokens of a flat sequence; none restarts from a
  coarse level of a code hierarchy.
- **Round-trip consistency as a metric:** watermarking for autoregressive image generation (arXiv:2506.16349) uses
  the token match after decode → re-tokenize; LILAC (arXiv:2608.05727) reports that audio codecs rewrite at least 15%
  of tokens per decode–re-encode pass.

## 4. Sudoku: symmetry frames and soft decisions

| Work | Relation |
|---|---|
| **Lattice Deduction Transformers** (arXiv:2605.08605, May 2026; abstract verified) | Runs each step of a neural Sudoku solver in a fresh random symmetry frame to decorrelate its chains. The closest precedent for frame randomization. |
| **POMO** (Kwon et al., NeurIPS 2020) | Uses symmetric instance views as multiple greedy starts for combinatorial optimization. |
| ARC solvers ("The LLM ARChitect", 2024; HRM, 2025; TRM, 2025) | Vote over augmented views of a task at test time. |
| **Multiple coarse grids** (Hackbusch, 1989; Mulder, 1989; Naik & Van Rosendale, 1993) | Multigrid methods that use several distinct coarse spaces to make the coarse correction robust. The symmetry-twisted V-cycle uses a different coarse representation of the same problem in every cycle. |
| **Mixture of Inputs** (Zhuang et al., 2025; arXiv:2505.14827; abstract verified) | Feeds the posterior-expected token embedding back into an autoregressive model, training-free. The soft-decision first pass applies the same idea to code levels and to the decoder input. |
| Latent Refinement Decoding (Zhu et al., 2025; arXiv:2510.11052) | Keeps uncertain positions of a diffusion language model as probability mixtures of embeddings. |

What we did not find: a restriction step that re-encodes a rejected solution in a transformed frame and keeps its
coarse codes (Section 3.7 of the README), or soft code decisions in a multiscale discrete generator.

## 5. Work behind the negative results (README Section 4)

- Error diffusion: Floyd & Steinberg (1976); temporal herding for discrete diffusion, Suzuki et al. (arXiv:2509.20896;
  verified), reports large gains for a uniform-noise model, which did not transfer to NCM.
- Dependence within a parallel step: Bansal et al. (arXiv:2606.22976; verified) bound the error of parallel sampling by
  the marginal error plus the conditional total correlation. NCM's measured within-level dependence is small, which
  caps every zero-compute coupling method.
- Consistent reconstruction: Thao & Vetterli (1994) and Goyal, Vetterli & Thao (1998) motivated the
  Voronoi-consistent sampler.
- Copula sampling with an auxiliary model: Discrete Copula Diffusion (ICLR 2025); a zero-compute copula sampler for
  diffusion language models, SAC-Copula (arXiv:2608.20839), was built for watermarking.

## 6. Novelty assessment

- **Not new:** re-encoding with the frozen multiscale encoder, keeping the coarse levels and regenerating the fine
  ones (DiverseVAR's scale-travel does this once, in latent space, for images); randomizing the symmetry frame of a
  neural Sudoku solver (Lattice Deduction Transformers, POMO); posterior-mean inputs (Mixture of Inputs).
- **New to our knowledge:**
  - restriction through the discrete output and the full encoder, applied to text and evaluated against an
    equal-compute control that isolates the re-encoding step;
  - the round-trip residual as a training-free error detector (AUROC 0.92 on hard Sudoku, against 0.87 for decoder
    confidence);
  - iterated V-cycles and the contraction of the coarse residual;
  - restriction in a transformed frame (the symmetry-twisted V-cycle) and soft code decisions in a multiscale
    discrete generator.
