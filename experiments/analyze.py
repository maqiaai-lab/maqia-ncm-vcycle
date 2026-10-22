"""Aggregate all experiments into results/SUMMARY.md and results/summary.json (figures: experiments/figures.py).

Phase 3 comparisons are made at matched compute: for every method and seed, accuracy is linearly
interpolated at fixed mean-NFE budgets along its round-by-round curve. Confidence intervals come
from a paired bootstrap over boards (the same resampled boards for every method and seed).

python experiments/analyze.py
"""
import glob
import json
import os
import re
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import RESULTS

sys.stdout.reconfigure(encoding="utf-8")  # the tables use non-ASCII (−, ×, …); Windows consoles default to cp1252

PAPER = {"hard": 86.1}  # NCM Sudoku-hard solved %, paper Fig. (12 NFE)
BUDGETS = [13, 14, 15, 16]  # every method and seed reaches 16 NFE within its 8 rounds
N_BOOT = 1000
rng = np.random.default_rng(0)  # Sudoku sections, in file order


def section_rng(name):
    """Bootstrap generator for one text section, so its CIs do not depend on which other results exist."""
    return np.random.default_rng(zlib.crc32(name.encode()))


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def show(name):
    """Variant names use '|' as a separator, which would split a Markdown table cell."""
    return name.replace("|", " ")


def pct(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f}"


lines, summary = [], {}

# ---- Phase 0 / 1 -------------------------------------------------------------------------------
lines += ["## Phase 0–1: reproduction and the round-trip residual", "",
          "| Difficulty | Solved (ours) | Paper | Fixed-point rate | Residual AUROC (round trip / level-local) | Decoder-confidence AUROC | Generator-confidence AUROC |",
          "|---|---|---|---|---|---|---|"]
for d in ("easy", "medium", "hard"):
    p0, p1 = load(f"phase0_sudoku_{d}.json"), load(f"phase1_sudoku_{d}.json")
    if not p0 or not p1:
        continue
    ca = p1["confidence_auroc"]
    lines.append(f"| {d} | {pct(p0['solved'])}% | {PAPER.get(d, '–')}{'%' if d in PAPER else ''} | "
                 f"{pct(p1['fixed_point_rate'])}% | {p1['roundtrip']['auroc']['n_disagree_positions']:.3f} / "
                 f"{p1['local']['auroc']['n_disagree_positions']:.3f} | {ca['1 - mean decoder prob']:.3f} | "
                 f"{ca['1 - min generator prob (all levels)']:.3f} |")
    summary[f"phase01_{d}"] = {"solved": p0["solved"], "fixed_point_rate": p1["fixed_point_rate"],
                               "auroc_roundtrip": p1["roundtrip"]["auroc"]["n_disagree_positions"],
                               "auroc_local": p1["local"]["auroc"]["n_disagree_positions"],
                               "auroc_decoder_conf": ca["1 - mean decoder prob"],
                               "auroc_generator_conf": ca["1 - min generator prob (all levels)"]}
lines.append("")

# ---- Phase 2 -----------------------------------------------------------------------------------
GATE_Q = (0.1, 0.2)  # residual gating: cycle only the boards with the top-q level-local residual
for path in sorted(glob.glob(os.path.join(RESULTS, "phase2_sudoku_*.json"))):
    p2 = json.load(open(path))
    tag = os.path.basename(path)[len("phase2_sudoku_"):-5]
    pb_path = os.path.join(RESULTS, "cache", f"phase2_perboard_{tag}.pt")
    sc_path = os.path.join(RESULTS, "cache", f"phase1_scores_{p2['difficulty']}.pt")
    pb = torch.load(pb_path, weights_only=False) if os.path.exists(pb_path) else None
    flags = {}
    if pb is not None and os.path.exists(sc_path):
        score = torch.load(sc_path, weights_only=False)["local_count"][:p2["n"]].float().sum(1)
        for q in GATE_Q:
            flags[q] = score >= torch.quantile(score, 1 - q)
    lines += [f"## Phase 2: greedy V-cycles ({p2['difficulty']}, {p2['n']} boards, baseline {pct(p2['baseline_solved'])}%)", "",
              "Each cycle re-encodes the current board and regenerates from the restart level at temperature 0. "
              "*Checker-gated* cycles only boards the Sudoku checker rejects. *Residual-gated* cycles only the boards whose "
              "level-local residual is in the top q (no checker, no solution). Extra NFE is the mean per board over all "
              "cycled boards, summed over cycles (gating would lower it).", "",
              "| Variant | Cycle | Solved | Fixed / broken | Checker-gated | " + " | ".join(f"Residual-gated top {int(100 * q)}%" for q in flags) +
              (" | " if flags else "") + "Extra NFE (cumulative) |",
              "|---|---|---|---|---|" + "---|" * len(flags) + "---|"]
    for name, cfg in p2["configs"].items():
        tot = 0.0
        for row in cfg["cycles"]:
            tot += row["extra_nfe_mean"]
            gated = []
            for q, fl in flags.items():
                after = pb.get(f"{name}|c{row['cycle']}")
                gated.append("–" if after is None else pct(torch.where(fl, after, pb["baseline_solved"]).float().mean().item()) + "%")
            lines.append(f"| {name} | {row['cycle']} | {pct(row['solved'])}% | {row['fixed']} / {row['broken']} | "
                         f"{pct(row['verifier_gated_solved'])}% | " + "".join(g + " | " for g in gated) + f"+{tot:.2f} |")
    lines.append("")
    summary[f"phase2_{tag}"] = p2

# ---- Phase 3 -----------------------------------------------------------------------------------
def curves(nfe_rounds, solved_round, idx):
    """Mean NFE and accuracy after each round, for boards idx (B, n) -> two (B, R+1) arrays."""
    R = nfe_rounds.shape[0] - 1
    x = np.stack([nfe_rounds[r][idx].mean(-1) for r in range(R + 1)], 1)
    sr = solved_round[idx]
    y = np.stack([((sr >= 0) & (sr <= r)).mean(-1) for r in range(R + 1)], 1)
    return x, y


def interp_at(xy, budget):
    """Accuracy at a mean-NFE budget along each curve (NaN past the last round)."""
    x, y = xy
    return np.array([np.interp(budget, x[b], y[b], right=np.nan) for b in range(x.shape[0])])


groups = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "cache", "phase3_perboard_*.pt"))):
    tag = os.path.basename(path)[len("phase3_perboard_"):-3]
    m = re.match(r"(.+)_s(\d+)$", tag)
    if not m:
        continue
    groups.setdefault(m.group(1), []).append(torch.load(path, weights_only=False))

for gname, seeds in groups.items():
    meta = load(f"phase3_sudoku_{gname}_s0.json") or {}
    keys = [k for k in seeds[0] if k != "baseline_solved" and all(k in s for s in seeds)]
    n = len(seeds[0]["baseline_solved"])
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    full_idx = np.arange(n)[None, :]
    data = {k: [(s[k]["nfe"].numpy().astype(np.int64), s[k]["solved_round"].numpy().astype(np.int64)) for s in seeds]
            for k in keys}

    full_xy = {k: [curves(nr, sr, full_idx) for nr, sr in data[k]] for k in keys}
    boot_xy = {k: [curves(nr, sr, boot_idx) for nr, sr in data[k]] for k in keys}
    at_budget = lambda xy, budget: np.mean(np.stack([interp_at(c, budget) for c in xy]), 0)  # NaN if any seed falls short
    point = {k: {B: float(at_budget(full_xy[k], B)[0]) for B in BUDGETS} for k in keys}
    boot = {k: {B: at_budget(boot_xy[k], B) for B in BUDGETS} for k in keys}
    lines += [f"## Phase 3: correction rounds at matched compute ({gname}; {len(seeds)} seed{'s' if len(seeds) > 1 else ''}; {n} boards)", "",
              "Accuracy (%) when the mean compute reaches the given NFE per board (12 = one greedy pass). "
              "Brackets: 95% paired-bootstrap interval over boards.", "",
              "| Method | " + " | ".join(f"{B} NFE" for B in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for k in keys:
        cells = []
        for B in BUDGETS:
            v, bs = point[k][B], boot[k][B]
            cells.append("–" if np.isnan(v) else f"{100 * v:.1f} [{100 * np.nanpercentile(bs, 2.5):.1f}, {100 * np.nanpercentile(bs, 97.5):.1f}]")
        lines.append(f"| {show(k)} | " + " | ".join(cells) + " |")
    lines.append("")

    # Paired differences that answer the plan's control questions.
    pairs = [(f"vcycle|m={mm}", f"no-restrict|m={mm}", "does the restriction step matter?") for mm in (1, 3, 5, 7)]
    pairs += [(f"vcycle-V2|m={mm}", f"vcycle|m={mm}", "masked vs full resample at the restart level") for mm in (1, 3, 5, 7)]
    pairs += [(k, "fresh|m=0", "vs fresh restarts (best-of-N with the checker)") for k in keys if k != "fresh|m=0"]
    lines += ["Paired differences in accuracy (percentage points) at matched NFE, mean [95% CI]:", "",
              "| A − B | Question | " + " | ".join(f"{B} NFE" for B in BUDGETS) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    diffs = {}
    for a_, b_, q in pairs:
        if a_ not in boot or b_ not in boot:
            continue
        cells = []
        for B in BUDGETS:
            dlt = boot[a_][B] - boot[b_][B]
            v = point[a_][B] - point[b_][B]
            lo, hi = np.nanpercentile(dlt, 2.5), np.nanpercentile(dlt, 97.5)
            cells.append("–" if np.isnan(v) else f"{100 * v:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
            diffs.setdefault(f"{a_} - {b_}", {})[B] = {"mean": v, "lo": lo, "hi": hi}
        lines.append(f"| {show(a_)} − {show(b_)} | {q} | " + " | ".join(cells) + " |")
    lines.append("")
    summary[f"phase3_{gname}"] = {"point": point, "diffs": diffs, "seeds": len(seeds), "n": n, "meta": meta.get("temperature")}

# ---- LM1B ---------------------------------------------------------------------------------------
lm = {}
TEXT_NAME = lambda size: "TinyStories" if size == "tinystories" else f"LM1B-{size}"
for path in sorted(glob.glob(os.path.join(RESULTS, "*_s*.json"))):
    if not re.fullmatch(r"(lm1b_[a-z]+|tinystories)_s\d+\.json", os.path.basename(path)):  # diagnostic runs, not sweep/converge
        continue
    r = json.load(open(path))
    cache = os.path.join(RESULTS, "cache", os.path.basename(path)[:-5] + ".pt")
    lm.setdefault(r["size"], []).append((r, torch.load(cache, weights_only=False) if os.path.exists(cache) else None))
for size, runs in lm.items():
    variants = ["baseline"] + list(runs[0][0]["vcycle"])
    protocol = "genppl protocol: samples cut at the first EOS" if size == "tinystories" else "FLM protocol"
    lines += [f"## {TEXT_NAME(size)}: residual diagnostic and V-cycle ({len(runs)} seed{'s' if len(runs) > 1 else ''} × {runs[0][0]['n']} samples)", "",
              f"Gen. PPL is corpus-level under GPT-2 large ({protocol}, as in the paper); entropy is the mean per-sample "
              f"unigram entropy of the {'tokens before the first EOS' if size == 'tinystories' else 'raw ids'}. V-cycle and "
              "no-restriction resample levels m..K-1 once, at T=1, from the same "
              "baseline samples, so they cost the same (K-m extra NFE).", "",
              "| Variant | Extra NFE | Gen. PPL (per seed) | Gen. PPL mean | Entropy | Adjacent repeats | distinct-2 | [PAD] fraction |",
              "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        rows = [rr["baseline"] if v == "baseline" else rr["vcycle"][v] for rr, _ in runs]
        ppl = [x["gen_ppl"] for x in rows]
        lines.append(f"| {show(v)} | {0 if v == 'baseline' else rows[0]['extra_nfe']} | {', '.join(f'{p_:.1f}' for p_ in ppl)} | "
                     f"{np.mean(ppl):.1f} | {np.mean([x['entropy'] for x in rows]):.3f} | "
                     f"{100 * np.mean([x['adjacent_repeat'] for x in rows]):.2f}% | {np.mean([x['distinct2'] for x in rows]):.3f} | "
                     f"{100 * np.mean([x.get('pad_fraction', float('nan')) for x in rows]):.1f}% |")
    lines.append("")
    # Paired bootstrap over samples (pooled over seeds) for corpus PPL ratios.
    if all(c is not None and "per_sample_nll" in c for _, c in runs):
        pooled = {v: (np.concatenate([c["per_sample_nll"][v][0] for _, c in runs]),
                      np.concatenate([c["per_sample_nll"][v][1] for _, c in runs])) for v in variants}
        N = len(pooled["baseline"][0])
        idx = section_rng(f"diag_{size}").integers(0, N, size=(N_BOOT, N))
        cppl = lambda v, ix: np.exp(pooled[v][0][ix].sum(-1) / pooled[v][1][ix].sum(-1))
        comps = [(v, "baseline") for v in variants[1:]]
        comps += [(v, v.replace("vcycle", "no-restrict")) for v in variants if v.startswith("vcycle")]
        lines += ["Paired bootstrap over samples (seeds pooled): corpus gen. PPL ratio A / B, mean [95% CI]:", "",
                  "| A / B | Ratio |", "|---|---|"]
        for a_, b_ in comps:
            if b_ not in pooled:
                continue
            ratio = cppl(a_, idx) / cppl(b_, idx)
            pt = cppl(a_, np.arange(N)) / cppl(b_, np.arange(N))
            summary.setdefault("text_ratios", {}).setdefault(size, {})[f"{a_} / {b_}"] = [
                float(pt), float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]
            lines.append(f"| {show(a_)} / {show(b_)} | {pt:.3f} [{np.percentile(ratio, 2.5):.3f}, {np.percentile(ratio, 97.5):.3f}] |")
        lines.append("")
    diag = [rr for rr, _ in runs]
    lines += ["Residual vs per-sample quality (Spearman ρ, mean over seeds; AUROC for flagging the worst-decile samples):", "",
              "| Residual | ρ with per-token NLL | ρ with adjacent repeats | ρ with entropy | AUROC worst-NLL decile | AUROC worst-repeat decile |",
              "|---|---|---|---|---|---|"]
    for name in ("roundtrip_total", "local_total"):
        sp = lambda k: np.mean([d["spearman"][name][k] for d in diag])
        au = lambda k: np.mean([d["auroc_worst_decile"][name][k] for d in diag])
        lines.append(f"| {name} | {sp('per_token_nll'):+.3f} | {sp('adjacent_repeat'):+.3f} | {sp('entropy'):+.3f} | "
                     f"{au('by_nll'):.3f} | {au('by_repeat'):.3f} |")
    lines.append(f"\nFixed-point rate: {np.mean([d['fixed_point_rate'] for d in diag]):.3f}.\n")
    # Cross-sample diversity and a few paired examples (needs the saved variant samples).
    with_ids = [(rr, c) for rr, c in runs if c is not None and c.get("variant_ids")]
    if with_ids:
        rr, c = with_ids[0]
        lines += [f"Cross-sample diversity (seed {rr['seed']}): corpus distinct-2 / distinct-3 over all samples:", "",
                  "| Variant | distinct-2 | distinct-3 |", "|---|---|---|"]
        for v in variants:
            x = rr["baseline"] if v == "baseline" else rr["vcycle"][v]
            if "corpus_distinct2" in x:
                lines.append(f"| {show(v)} | {x['corpus_distinct2']:.3f} | {x['corpus_distinct3']:.3f} |")
        lines.append("")
        from transformers import AutoTokenizer
        if size == "tinystories":  # the released BPE tokenizer (its pickles were checked with pickletools)
            from omegaconf import OmegaConf
            from vcycle.common import ARTIFACTS
            from vcycle.loading import import_ncm
            import_ncm()
            from tokenization.bpe_tokenizer import BPETokenizer
            ck_ = os.path.join(ARTIFACTS, "tinystories")
            cf_ = OmegaConf.load(os.path.join(ck_, "config.yaml"))
            pt_ = cf_.tokenizer.pre_tokenizer
            pt_.vocab_file_path, pt_.merges_file_path = (os.path.join(ck_, cf_.checkpoints.bpe_vocab),
                                                         os.path.join(ck_, cf_.checkpoints.bpe_merges))
            bpe = BPETokenizer.from_config(pt_)
            show_text = lambda ids: bpe.decode([t for t in ids if t not in (bpe.eos_token_id, bpe.pad_token_id)]).replace("\n", " / ")
        else:
            btok = AutoTokenizer.from_pretrained("bert-base-uncased")
            show_text = lambda ids: btok.decode(ids).replace("[CLS]", " / ").replace("[PAD]", "")
        best = min((v for v in c["variant_ids"] if v.startswith("vcycle")),
                   key=lambda v: rr["vcycle"][v]["gen_ppl"])
        ctrl = best.replace("vcycle", "no-restrict")
        lines += [f"Examples (seed {rr['seed']}, first 3 samples; same starting sample in each row, text truncated):", ""]
        for i in range(3):
            for name, ids_ in (("baseline", c["ids"]), (best, c["variant_ids"][best]), (ctrl, c["variant_ids"].get(ctrl))):
                if ids_ is None:
                    continue
                txt = show_text(ids_[i].tolist()).strip()
                lines.append(f"- *{show(name)}*: {txt[:300]}")
            lines.append("")
    summary[f"lm1b_{size}"] = [rr for rr, _ in runs]

# ---- LM1B sweep -----------------------------------------------------------------------------------
for path in sorted(glob.glob(os.path.join(RESULTS, "lm1b_sweep_*_s*.json"))):
    sw = json.load(open(path))
    cache = os.path.join(RESULTS, "cache", os.path.basename(path)[:-5] + ".pt")
    ps = torch.load(cache, weights_only=False)["per_sample_nll"] if os.path.exists(cache) else {}
    lines += [f"## LM1B-{sw['size']} sweep: restart level and number of cycles (seed {sw['seed']}, {sw['n']} samples)", "",
              "Same baseline samples for every row. Extra NFE is relative to one NCM pass (11 generator levels; "
              "level 0 is drawn from a unigram). Ratio: corpus gen. PPL vs the no-restriction run at the same level, "
              "paired bootstrap over samples.", "",
              "| Variant | Extra NFE | Gen. PPL | Ratio vs no-restrict [95% CI] | Entropy | distinct-2 (corpus) | Adjacent repeats |",
              "|---|---|---|---|---|---|---|"]
    N = len(next(iter(ps.values()))[0]) if ps else 0
    idx = section_rng(os.path.basename(path)).integers(0, N, size=(N_BOOT, N)) if N else None
    for name, r in sw["variants"].items():
        ratio = "–"
        ctrl = re.sub(r"^vcycle\|m=(\d+)\|c=\d+$", r"no-restrict|m=\1|c=1", name)
        if name.startswith("vcycle") and ctrl in ps and name in ps:
            cp = lambda v, ix: np.exp(ps[v][0][ix].sum(-1) / ps[v][1][ix].sum(-1))
            b = cp(name, idx) / cp(ctrl, idx)
            ratio = f"{cp(name, np.arange(N)) / cp(ctrl, np.arange(N)):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
        lines.append(f"| {show(name)} | +{r['extra_nfe']} | {r['gen_ppl']:.1f} | {ratio} | {r['entropy']:.3f} | "
                     f"{r.get('corpus_distinct2', float('nan')):.3f} | {100 * r['adjacent_repeat']:.2f}% |")
    lines.append("")
    summary[f"lm1b_sweep_{sw['size']}_s{sw['seed']}"] = sw

# ---- LM1B iterated V-cycles (convergence) -----------------------------------------------------------
for path in sorted(glob.glob(os.path.join(RESULTS, "lm1b_converge_*_s*.json"))):
    cv = json.load(open(path))
    cache = os.path.join(RESULTS, "cache", os.path.basename(path)[:-5] + ".pt")
    ps = torch.load(cache, weights_only=False)["per_sample_nll"] if os.path.exists(cache) else {}
    N = cv["n"]
    idx = section_rng(os.path.basename(path)).integers(0, N, size=(N_BOOT, N))
    cp = lambda v, ix: np.exp(ps[v][0][ix].sum(-1) / ps[v][1][ix].sum(-1))

    def ratio(a_, b_):
        if a_ not in ps or b_ not in ps:
            return "–"
        b = cp(a_, idx) / cp(b_, idx)
        return f"{cp(a_, np.arange(N)) / cp(b_, np.arange(N)):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"

    lines += [f"## LM1B-{cv['size']}: iterated V-cycles (seed {cv['seed']}, {N} samples, {cv.get('device', 'cpu')})", "",
              "Each cycle re-encodes the newest sample, keeps encode(x)_{<m} and resamples levels m..K-1 at T=1. Ratios "
              "compare corpus gen. PPL, paired bootstrap over samples. *Coarse FP*: share of samples whose re-encoded "
              "levels < m equal the codes they were generated from (a coarse fixed point). *Agree*: positional token "
              "agreement with the original draft / with the previous cycle.", "",
              "| m | Cycle | Extra NFE | Gen. PPL | Ratio vs no-restrict [95% CI] | Ratio vs previous cycle [95% CI] | Entropy "
              "| distinct-2 (corpus) | Coarse FP | Agree draft / prev |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key, run in cv["runs"].items():
        nr = run.get("no_restrict")
        if nr:
            lines.append(f"| {run['m']} | no-restrict | +{nr['extra_nfe']} | {nr['gen_ppl']:.1f} | – | – | {nr['entropy']:.3f} | "
                         f"{nr['corpus_distinct2']:.3f} | {pct(nr['coarse_fixed_point_rate'])}% | {nr['agree_draft']:.3f} / – |")
        for r in run["cycles"]:
            i = r["cycle"]
            vs_nr = ratio(f"{key}|c={i}", f"no-restrict|{key}") if i else "–"
            vs_prev = ratio(f"{key}|c={i}", f"{key}|c={i - 1}") if i else "–"
            lines.append(f"| {run['m']} | {i} | +{r['extra_nfe']} | {r['gen_ppl']:.1f} | {vs_nr} | {vs_prev} | {r['entropy']:.3f} | "
                         f"{r['corpus_distinct2']:.3f} | {pct(r['coarse_fixed_point_rate'])}% | "
                         f"{r['agree_draft']:.3f} / {r['agree_prev']:.3f} |")
    lines.append("")
    summary[f"lm1b_converge_{cv['size']}_s{cv['seed']}"] = cv

# ---- LM1B residual-gated V-cycle (from the diagnostic caches, no new sampling) ----------------------
# The gate cycles only the samples whose level-local residual (summed over levels, as in Sudoku Phase 2) is in
# the top q of their seed; the rest keep the single-pass sample. Controls at equal expected compute: a random
# gate of rate q (ratio of expectations over the gate) and, as an upper bound that is not free, an oracle gate
# on the single-pass sample's own GPT-2 NLL.
GATE_LM = (0.1, 0.2, 0.3, 0.5)
for size, runs in lm.items():
    caches = [c for _, c in runs if c is not None and "loc" in c]
    m_gate, K_lm = 6, len(runs[0][0]["levels"])
    if not caches or not all(f"vcycle|m={m_gate}" in c["per_sample_nll"] for c in caches):
        continue
    cat = lambda k, j: np.concatenate([np.asarray(c["per_sample_nll"][k][j], float) for c in caches])
    nb, tb, nv, tv = cat("baseline", 0), cat("baseline", 1), cat(f"vcycle|m={m_gate}", 0), cat(f"vcycle|m={m_gate}", 1)
    score = [c["loc"].sum(1).numpy() for c in caches]
    own = [np.asarray(c["per_sample_nll"]["baseline"][0], float) / np.maximum(np.asarray(c["per_sample_nll"]["baseline"][1], float), 1)
           for c in caches]
    top = lambda xs, q: np.concatenate([x >= np.quantile(x, 1 - q) for x in xs])
    N = len(nb)
    idx = section_rng(f"gate_{size}").integers(0, N, size=(N_BOOT, N))
    mixed = lambda g, ix: np.exp(np.where(g, nv, nb)[ix].sum(-1) / np.where(g, tv, tb)[ix].sum(-1))
    rand = lambda q, ix: np.exp(((1 - q) * nb + q * nv)[ix].sum(-1) / ((1 - q) * tb + q * tv)[ix].sum(-1))
    full = np.arange(N)
    p_base, p_all = mixed(np.zeros(N, bool), full), mixed(np.ones(N, bool), full)
    lines += [f"## {TEXT_NAME(size)}: residual-gated V-cycle (m = {m_gate}; {len(caches)} seed{'s' if len(caches) > 1 else ''} × "
              f"{N // len(caches)} samples)", "",
              f"Only the top-q samples by level-local residual are cycled; extra NFE is the mean per sample. Single pass "
              f"{p_base:.1f}, every sample cycled {p_all:.1f}. *Share of gain*: fraction of the log-PPL gain of cycling every "
              "sample. Ratio: residual gate / random gate at the same q, paired bootstrap over samples.", "",
              "| q | Extra NFE | Residual gate | Random gate | Oracle gate (own NLL) | Residual / random [95% CI] | Share of gain: residual / oracle |",
              "|---|---|---|---|---|---|---|"]
    gain = lambda p: np.log(p_base / p) / np.log(p_base / p_all)
    gate_summary = {}
    for q in GATE_LM:
        g, go = top(score, q), top(own, q)
        pr, pq, po = mixed(g, full), rand(q, full), mixed(go, full)
        b = mixed(g, idx) / rand(q, idx)
        ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        lines.append(f"| {q:g} | +{q * (K_lm - m_gate):.1f} | {pr:.1f} | {pq:.1f} | {po:.1f} | {pr / pq:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{gain(pr):.2f} / {gain(po):.2f} |")
        gate_summary[q] = {"residual": pr, "random": pq, "oracle": po, "ratio": pr / pq, "ci": ci}
    lines.append("")
    summary[f"lm1b_gate_{size}"] = gate_summary

# Same seed as the sweep: is gating at m = 6 a better use of compute than restarting every sample at a finer level?
for path in sorted(glob.glob(os.path.join(RESULTS, "cache", "lm1b_sweep_*_s*.pt"))):
    size, seed = re.search(r"lm1b_sweep_(\w+?)_s(\d+)\.pt", os.path.basename(path)).groups()
    diag = os.path.join(RESULTS, "cache", f"lm1b_{size}_s{seed}.pt")
    if not os.path.exists(diag):
        continue
    ps, s = torch.load(path, weights_only=False)["per_sample_nll"], torch.load(diag, weights_only=False)["loc"].sum(1).numpy()
    K_lm = len(json.load(open(os.path.join(RESULTS, f"lm1b_{size}_s{seed}.json")))["levels"])
    nb, tb = (np.asarray(x, float) for x in ps["baseline"])
    lv = sorted(int(re.search(r"m=(\d+)", k).group(1)) for k in ps if re.fullmatch(r"vcycle\|m=\d+\|c=1", k))
    qs = (0.2, 1 / 3, 0.5, 2 / 3, 1.0)
    lines += [f"Gating vs a finer restart level (LM1B-{size}, seed {seed}): gen. PPL (extra NFE per sample) when the top-q "
              f"residual samples are cycled from level m; q = 1 cycles every sample. Single pass: {np.exp(nb.sum() / tb.sum()):.1f}.", "",
              "| m | " + " | ".join(f"q = {q:.2f}" for q in qs) + " |", "|---|" + "---|" * len(qs)]
    for m_ in lv:
        nv, tv = (np.asarray(x, float) for x in ps[f"vcycle|m={m_}|c=1"])
        cells = []
        for q in qs:
            g = s >= np.quantile(s, 1 - q)
            cells.append(f"{np.exp(np.where(g, nv, nb).sum() / np.where(g, tv, tb).sum()):.1f} (+{q * (K_lm - m_):.1f})")
        lines.append(f"| {m_} | " + " | ".join(cells) + " |")
    lines.append("")

# ---- Text: does the V-cycle's gain grow with the coarse residual? (diagnostic caches, no new sampling) -----
# If the gain comes from replacing off-manifold coarse codes, samples whose kept levels change more on the round trip
# should gain more. Dose: round-trip disagreement on levels < m (mean over levels). Gain: log-PPL of the
# no-restriction control minus that of the V-cycle, per sample. Partial rho removes the draft's own log-PPL.
def ranks(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt + 1) / 2)[inv]  # average ranks for ties


def rho(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def partial_out(y, z):
    A = np.c_[np.ones_like(z), z]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


m_dose, dose_rows, dose_summary = 6, [], {}
for size, runs in lm.items():
    caches = [c for _, c in runs if c is not None and "rt" in c]
    if not caches or not all(f"no-restrict|m={m_dose}" in c["per_sample_nll"] for c in caches):
        continue
    lp = lambda k: np.concatenate([np.asarray(c["per_sample_nll"][k][0], float) / np.maximum(np.asarray(c["per_sample_nll"][k][1], float), 1)
                                   for c in caches])
    cat = lambda k, j: np.concatenate([np.asarray(c["per_sample_nll"][k][j], float) for c in caches])
    x = np.concatenate([c["rt"][:, :m_dose].mean(1).numpy().astype(float) for c in caches])
    gain, draft = lp(f"no-restrict|m={m_dose}") - lp(f"vcycle|m={m_dose}"), lp("baseline")
    rx, rg, rd = ranks(x), ranks(gain), ranks(draft)
    r, r_part = rho(rx, rg), rho(partial_out(rx, rd), partial_out(rg, rd))
    idx = section_rng(f"dose_{size}").integers(0, len(x), size=(N_BOOT, len(x)))
    b = [rho(ranks(x[i]), ranks(gain[i])) for i in idx]
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    nv, tv, nn = cat(f"vcycle|m={m_dose}", 0), cat(f"vcycle|m={m_dose}", 1), cat(f"no-restrict|m={m_dose}", 0)
    qb = np.clip(np.searchsorted(np.quantile(x, [0.2, 0.4, 0.6, 0.8]), x, side="right"), 0, 4)
    quint = [float(np.exp((nv[qb == k].sum() - nn[qb == k].sum()) / tv[qb == k].sum())) for k in range(5)]
    lo, hi = np.percentile(x, [10, 90])
    dose_rows.append(f"| {TEXT_NAME(size)} | {len(x)} | {lo:.2f}–{hi:.2f} | {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r_part:+.3f} | "
                     + " / ".join(f"{q_:.3f}" for q_ in quint) + " |")
    dose_summary[size] = {"n": len(x), "residual_p10_p90": [float(lo), float(hi)], "rho": r, "ci": ci,
                          "rho_partial_draft": r_part, "ratio_by_quintile": quint}
if dose_rows:
    lines += [f"## Text: V-cycle gain against the coarse residual (m = {m_dose}; pooled seeds, no new sampling)", "",
              f"Per sample: residual = round-trip disagreement on the kept levels 0–{m_dose - 1} (mean over levels); gain = "
              "log-PPL of the no-restriction control minus that of the V-cycle. Spearman rho with a 95% bootstrap CI over "
              "samples; partial rho given the draft's own log-PPL; corpus gen. PPL ratio V-cycle / control within each "
              "residual quintile, lowest to highest.", "",
              "| Model | Samples | Residual, 10th–90th pct. | Spearman rho [95% CI] | Partial rho given draft | V-cycle / control by residual quintile |",
              "|---|---|---|---|---|---|"] + dose_rows + [""]
    summary["text_dose_response"] = dose_summary

# ---- LM1B quality beyond GPT-2 perplexity (lm1b_quality.py, lm1b_mauve.py) --------------------------
for path in sorted(glob.glob(os.path.join(RESULTS, "lm1b_quality_*.json"))):
    qj = json.load(open(path))
    size = qj["size"]
    mv_path = os.path.join(RESULTS, f"lm1b_mauve_{size}.json")
    mj = json.load(open(mv_path)) if os.path.exists(mv_path) else None
    mj = mj if mj and "half_size" in mj else None  # a run still in progress has no summary yet
    gpt2 = {v: np.mean([rr["baseline"]["gen_ppl"] if v == "baseline" else rr["vcycle"][v]["gen_ppl"] for rr, _ in lm.get(size, [])])
            for v in qj["variants"]} if size in lm else {}
    ref2 = qj["ref2"].split("/")[-1]
    lines += [f"## LM1B-{size}: quality beyond GPT-2 perplexity (seeds {qj['seeds']}, pooled {qj['n_per_seed'] * len(qj['seeds'])} samples)", "",
              f"*Human*: held-out LM1B validation chunks (128 BERT ids, like the samples), scored the same way. {ref2}: gen. PPL "
              "of the text without special tokens, up to 256 tokens. MAUVE: against the same number of human chunks, GPT-2-large "
              "features, full sets" + (f". Uncertainty from {mj['boot']} rounds of paired half-size subsamples (without "
                                      "replacement; fresh k-means seed): the last column gives the full-set difference, then the "
                                      "half-size mean difference with its 2.5–97.5 percentiles and the share of rounds the "
                                      "V-cycle is ahead. Half-size MAUVE is not on the full-size scale; only differences compare."
                                      if mj else "."), "",
              f"| Text | GPT-2 gen. PPL (FLM) | {ref2} gen. PPL | {ref2} ratio vs no-restrict [95% CI] | MAUVE ↑ | MAUVE − no-restrict |",
              "|---|---|---|---|---|---|"]
    lines.append(f"| human | {qj['human']['gen_ppl']:.1f} | {qj['ref2_ppl']['human']:.1f} | – | – | – |")
    for v in qj["variants"]:
        r2 = qj["ref2_ratios"].get(f"{v} / {v.replace('vcycle', 'no-restrict')}") if v.startswith("vcycle") else None
        mval = mj["point"].get(v) if mj else qj.get("mauve", {}).get(v, {}).get("pooled")
        mcell = f"{mval:.3f}" if mval is not None else "–"
        pd = mj["paired"].get(f"{v} - {v.replace('vcycle', 'no-restrict')}") if mj and v.startswith("vcycle") else None
        lines.append(f"| {show(v)} | {gpt2.get(v, float('nan')):.1f} | {qj['ref2_ppl'][v]['pooled']:.1f} | "
                     f"{f'{r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}]' if r2 else '–'} | {mcell} | "
                     + (f"{pd['full']:+.3f}; half-size {pd['half_mean']:+.3f} [{pd['half_ci'][0]:+.3f}, {pd['half_ci'][1]:+.3f}], "
                        f"ahead in {100 * pd['share_won']:.0f}% |" if pd else "– |"))
    lines += ["", f"Human cross-sample diversity: corpus distinct-2 {qj['human']['corpus_distinct2_per_seed_block'][0]:.3f} per "
              f"{qj['n_per_seed']} chunks (samples: see the table above, per seed); entropy {qj['human']['entropy']:.3f}.", ""]
    summary[f"lm1b_quality_{size}"] = {"quality": qj, "mauve": mj}

# ---- Prompted tasks: XSum and WMT14 (cond_vcycle.py) ----------------------------------------------
fmt_ci = lambda r, d=2, sign=True: (f"{r[0]:+.{d}f} [{r[1]:+.{d}f}, {r[2]:+.{d}f}]" if sign else
                                    f"{r[0]:.{d}f} [{r[1]:.{d}f}, {r[2]:.{d}f}]") if r else "–"
for task_ in ("xsum", "wmt14"):
    cj = load(f"{task_}_vcycle.json")
    if not cj or "baseline" not in cj.get("variants", {}):
        continue
    rawj = load(f"{task_}_vcycle_raw.json")
    V = cj["variants"]
    samp = cj["sampling"]
    title = {"xsum": "XSum (summaries)", "wmt14": "WMT14 German → English"}[task_]
    lines += [f"## {title}: V-cycles on the released greedy decoding ({cj['n']} documents)", "",
              f"Single pass exactly as the released evaluation (levels sampled with temperature {samp['temperature']:g}, "
              f"top_k {samp['top_k']}: greedy). V-cycle from level m: re-encode the output ({cj.get('restriction', '')}), keep "
              f"the encoder's levels < m, regenerate levels m..{cj['K'] - 1} greedily with the same prompt. Without "
              "re-encoding, the same regeneration reproduces the single pass exactly (checked below), so the single pass is "
              "the equal-compute control. Differences: paired bootstrap over documents. *Changed*: share of documents whose "
              f"output differs from the previous cycle's (1 − fixed-point rate). Decoder vs generate(): "
              f"{cj['check']['mismatch_cached']}/{cj['check']['docs']} differ with the prompt cache, "
              f"{cj['check']['mismatch_uncached']}/{cj['check']['docs']} without.", ""]
    rows = [("baseline", V["baseline"])] + [(k, v) for k, v in V.items() if k != "baseline"]
    if rawj and "vcycle|m=5|c=1" in rawj.get("variants", {}):
        rows.append(("vcycle|m=5|c=1 (raw re-encoding)", rawj["variants"]["vcycle|m=5|c=1"]))
    if task_ == "xsum":
        lines += ["| Variant | Extra NFE | R-1 | R-2 | R-L | Δ R-1 [95% CI] | Δ R-2 | Δ R-L | Gen. PPL (GPT-2) | PPL ratio [95% CI] | Words | Changed |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in rows:
            if "identical_to_single_pass" in r:
                lines.append(f"| {show(k)} | +{r['extra_nfe']} | identical to the single pass on {100 * r['identical_to_single_pass']:.1f}% of documents ||||||||||")
                continue
            d = r.get("vs_baseline", {})
            lines.append(f"| {show(k)} | +{r['extra_nfe']} | {r['rouge1']:.2f} | {r['rouge2']:.2f} | {r['rougeL']:.2f} | "
                         f"{fmt_ci(d.get('rouge1'))} | {fmt_ci(d.get('rouge2'))} | {fmt_ci(d.get('rougeL'))} | {r['gen_ppl']:.1f} | "
                         f"{fmt_ci(d.get('gen_ppl_ratio'), 3, False)} | {r['words']:.1f} | "
                         f"{pct(r.get('changed_vs_prev')) + '%' if r.get('changed_vs_prev') is not None else '–'} |")
    else:
        lines += ["| Variant | Extra NFE | BLEU | Δ BLEU [95% CI] | BLEU (lc) | chrF | Δ sentence chrF [95% CI] | Length ratio | Changed |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k, r in rows:
            if "identical_to_single_pass" in r:
                lines.append(f"| {show(k)} | +{r['extra_nfe']} | identical to the single pass on {100 * r['identical_to_single_pass']:.1f}% of documents |||||||")
                continue
            d = r.get("vs_baseline", {})
            lines.append(f"| {show(k)} | +{r['extra_nfe']} | {r['bleu']:.2f} | {fmt_ci(d.get('bleu'))} | {r['bleu_lc']:.2f} | "
                         f"{r['chrf']:.2f} | {fmt_ci(d.get('chrf_sentence'))} | {r['len_ratio']:.3f} | "
                         f"{pct(r.get('changed_vs_prev')) + '%' if r.get('changed_vs_prev') is not None else '–'} |")
    rs = cj.get("residual")
    if rs:
        lines += ["", f"Residual of the single pass: fixed-point rate {rs['fixed_point_rate']:.3f}; flagging the worst tenth of "
                  f"documents ({'ROUGE-L' if task_ == 'xsum' else 'sentence chrF'}): AUROC {rs['roundtrip_total']['auroc_worst_decile']:.3f} "
                  f"(round trip), {rs['local_total']['auroc_worst_decile']:.3f} (level-local)."]
    lines.append("")
    summary[f"{task_}_vcycle"] = {"main": cj, "raw": rawj}

# ---- LLM judge (judge.py) ---------------------------------------------------------------------------
jfiles = sorted(p_ for p_ in glob.glob(os.path.join(RESULTS, "judge_*.json")) if "_smoke" not in p_)
if jfiles:
    lines += ["## Pairwise preference by a local LLM judge", "",
              "Each item pairs two texts made from the same single-pass sample (LM1B) or for the same source (XSum, WMT14). "
              "The judge sees both orders; an item's preference for A is the mean over the orders of P(A's letter) / "
              "(P(A) + P(B)) from its next-token probabilities. CIs: bootstrap over items. *Order agreement*: share of items "
              "on which the judge picks the same text in both orders.", "",
              "| Task | A | B | Items | Mean preference for A [95% CI] | Share preferring A [95% CI] | Order agreement | Judge |",
              "|---|---|---|---|---|---|---|---|"]
    for p_ in jfiles:
        j = json.load(open(p_))
        task_name = f"LM1B-{j['size']}" if j["task"] == "lm1b" else {"xsum": "XSum", "wmt14": "WMT14"}.get(j["task"], j["task"])
        lines.append(f"| {task_name} | {show(j['a'])} | {show(j['b'])} | {j['items']} | "
                     f"{fmt_ci(j['mean_preference_for_a'], 3, False)} | {fmt_ci(j['share_preferring_a'], 3, False)} | "
                     f"{100 * j['order_agreement']:.0f}% | {j['judge'].split('/')[-1]} |")
        summary[os.path.basename(p_)[:-5]] = j
    lines.append("")

# ---- Sudoku: soft decisions, singles propagation and symmetry-twisted V-cycles (sudoku_stack.py) ----------
SUDOKU_LEVELS = [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]  # positions per level, all released Sudoku tokenizers
STACK_NAMES = {"N": "Best-of-N (fresh restarts, T = 1)", "F": "Fresh greedy restarts in random symmetry frames",
               "E": "V-cycle retries from level 5 (T = 1)", "A": "Symmetry-twisted V-cycle (level 5, T = 0)",
               "B": "Soft-decision first pass + twisted V-cycle", "D": "Soft decisions throughout + twisted V-cycle"}
for path in sorted(glob.glob(os.path.join(RESULTS, "sudoku_stack_*.json"))):
    sj = json.load(open(path))
    tag = os.path.basename(path)[len("sudoku_stack_"):-5]
    budgets = [b for b in ("12", "13", "14", "15", "16", "18", "20") if b in next(iter(sj["arms"].values()))["all"]]
    lines += [f"## Sudoku ({sj['diff']}): checker-gated retries, {'with' if sj.get('singles', True) else 'without'} singles propagation "
              f"({sj['n']} boards, seeds {sj['seeds']}, {sj.get('gpu') or ('NVIDIA A10G, replication' if 'a10g' in tag else sj['device'])})", "",
              f"Greedy first pass {pct(sj['greedy_first_pass'], 2)}%. Soft-decision first pass at Ts = {sj['ts_star']:g} (chosen on "
              f"boards 0–{sj['n'] // 2 - 1}): {pct(sj['soft'][str(sj['ts_star'])]['all'], 2)}% "
              f"(held-out half {pct(sj['soft'][str(sj['ts_star'])]['held'], 2)}%). Singles propagation alone completes "
              f"{pct(sj['singles_complete'], 2)}% of boards. Accuracy (%) at a mean NFE budget per board; "
              "\u2013 where the arm's retries ran out before the budget.", "",
              "| Method | " + " | ".join(f"{b} NFE" for b in budgets) + " | Final (mean NFE) | Duplicate retries |",
              "|---|" + "---|" * (len(budgets) + 2)]
    for k, name in STACK_NAMES.items():
        if k not in sj["arms"]:
            continue
        a = sj["arms"][k]["all"]
        cells = ["\u2013" if a[b] != a[b] else f"{a[b]:.1f}" for b in budgets]
        fin = sj["final"][k] if "final" in sj else None
        lines.append(f"| {name} | " + " | ".join(cells) + " | " +
                     (f"{fin['solved']:.1f} ({fin['mean_nfe']:.1f}) | " if fin else "\u2013 | ") + f"{100 * sj['dup'][k]:.0f}% |")
    lines += ["", "Paired differences (percentage points, mean [95% CI], bootstrap over boards, seeds averaged):", "",
              "| Comparison | Boards | Difference |", "|---|---|---|"]
    for key, r in sj["paired"].items():
        label, split = key.split("|")
        if r["diff"] != r["diff"] or (r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0):
            continue
        lines.append(f"| {label} | {'all' if split == 'all' else 'held-out half'} | "
                     f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] |")
    # Extra paired comparisons from the per-board file: equal NFE and equal FLOPs (token positions processed).
    pb_path = os.path.join(RESULTS, "cache", f"sudoku_stack_{tag}_perboard.pt")
    if os.path.exists(pb_path) and "acc_round" in next(iter(torch.load(pb_path, weights_only=False)["runs"].values()))[0]:
        runs_ = torch.load(pb_path, weights_only=False)["runs"]
        R_, K_, full_ = sj["rounds"], 12, sj["flop_full_positions"]
        prompt_ = full_ - sum(SUDOKU_LEVELS)

        def cost_(run, kind):
            a = np.where(run["acc_round"] >= 0, run["acc_round"], R_).astype(np.int64)
            n_ = np.minimum(np.arange(R_ + 1)[:, None], a[None, :])
            if kind == "nfe":
                return K_ + (K_ - run["m"]) * n_
            per = ((prompt_ if run["frames"] == "twisted" else 0) + sum(SUDOKU_LEVELS[run["m"]:])) / full_
            return 1.0 + per * n_

        def acc_(arm, idx, budget, kind):
            vals = []
            for run in runs_[arm]:
                c, sr = cost_(run, kind)[:, idx], run["solved_round"][idx]
                ys = np.array([((sr >= 0) & (sr <= r)).mean() for r in range(R_ + 1)])
                vals.append(np.interp(budget, c.mean(1), ys, right=ys[-1] if run["unaccepted_end"] == 0 else np.nan))
            return float(np.nanmean(vals))

        n_b = len(runs_["A"][0]["solved_round"])
        boot_ = section_rng(f"stack_{tag}").integers(0, n_b, size=(400, n_b))
        lines += ["", "Further paired differences (percentage points, mean [95% CI]; FLOPs counted as token positions "
                  "processed, one full pass = 1, a retry in a new frame recomputes the prompt):", "",
                  "| Comparison | Budget | Difference |", "|---|---|---|"]
        extra = {}
        for a_, b_ in (("D", "F"), ("A", "F"), ("D", "E"), ("F", "N"), ("E", "N")):
            if a_ not in runs_ or b_ not in runs_:
                continue
            for kind, budget in (("nfe", 13), ("nfe", 14), ("flop", 1.1), ("flop", 1.2), ("flop", 1.4)):
                d0 = acc_(a_, np.arange(n_b), budget, kind) - acc_(b_, np.arange(n_b), budget, kind)
                ds = np.array([acc_(a_, ix, budget, kind) - acc_(b_, ix, budget, kind) for ix in boot_])
                ds = ds[~np.isnan(ds)]
                if np.isnan(d0) or len(ds) < 200 or (d0 == 0 and np.all(ds == 0)):
                    continue
                lo, hi = np.percentile(ds, 2.5), np.percentile(ds, 97.5)
                unit = "NFE" if kind == "nfe" else "full-pass FLOPs"
                lines.append(f"| {a_} − {b_} | {budget:g} {unit} | {100 * d0:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
                extra[f"{a_}-{b_}@{budget}{kind}"] = [100 * d0, 100 * lo, 100 * hi]
        sj["paired_extra"] = extra
    lines.append("")
    summary[f"sudoku_stack_{tag}"] = sj

# ---- Error diffusion within a level (ed_sweep.py): a negative result --------------------------------------------
ed1, ed2 = load("ed_stage1.json"), load("ed_stage2.json")
if ed1:
    lines += [f"## Error diffusion within a level (Floyd–Steinberg), stage 1 ({len(ed1['seeds'])} seeds × {ed1['n']} samples)", "",
              "Each arm samples every level with error diffusion of strength γ and carry ρ (γ = 0 is NCM). Ratio: corpus "
              "gen. PPL against the reference sampler with the same seeds, paired bootstrap over samples; *frontier*: "
              "ratio against NCM's own temperature curve at the same corpus distinct-2.", "",
              "| Task | Arm | Gen. PPL | Ratio [95% CI] | Corpus distinct-2 change | Frontier ratio |", "|---|---|---|---|---|---|"]
    for task_, tj in ed1["tasks"].items():
        lines.append(f"| {task_} | reference (NCM) | {tj['reference']['gen_ppl']:.2f} | – | – | – |")
        for arm, r in tj["arms"].items():
            fr = f"{r['frontier_ratio']:.3f}" if "frontier_ratio" in r else "–"
            lines.append(f"| {task_} | {arm} | {r['gen_ppl']:.2f} | {r['ppl_ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                         f"{100 * r['corpus_d2_change']:+.1f}% | {fr} |")
    lines.append("")
    summary["error_diffusion_stage1"] = ed1
if ed2:
    lines += [f"Stage 2 (LM1B-medium, γ = {ed2['gamma']:g}, ρ = {ed2['rho']:g}, m = {ed2['m']}, {len(ed2['seeds'])} seeds × "
              f"{ed2['n']}): ratios against the reference sampler.", "",
              "| Arm | Gen. PPL | Ratio [95% CI] |", "|---|---|---|"]
    for arm, r in ed2["vs_reference"].items():
        lines.append(f"| {arm} | {r['gen_ppl']:.2f} | {r['ppl_ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] |")
    lines += ["", "Against the plain V-cycle (does error diffusion add to it?):", "", "| Arm | Ratio vs V-cycle [95% CI] |",
              "|---|---|"]
    for arm, r in ed2.get("vs_vcycle", {}).items():
        lines.append(f"| {arm} | {r['ppl_ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] |")
    lines.append("")
    summary["error_diffusion_stage2"] = ed2

with open(os.path.join(RESULTS, "SUMMARY.md"), "w", encoding="utf-8") as f:
    f.write("# Results summary (generated by experiments/analyze.py)\n\n" + "\n".join(lines) + "\n")
with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=float)
print("\n".join(lines))
