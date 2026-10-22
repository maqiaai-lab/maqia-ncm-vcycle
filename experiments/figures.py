"""Draw every figure of the README from the result files (run experiments/analyze.py first; it writes the paired
ratios this script reads from results/summary.json). Output: figures/*.png (300 dpi) and figures/*.pdf.

python experiments/figures.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcycle import plotstyle as ps  # noqa: E402
from vcycle.common import RESULTS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (plotstyle selects the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
ps.setup()


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


summary = load("summary.json") or {}
HUMAN_PPL = (load("lm1b_quality_medium.json") or {}).get("human", {}).get("gen_ppl")


# ---- Figure 1: the restriction effect at every model size ------------------------------------------------------------
def fig_restriction():
    ratios = summary.get("text_ratios", {})
    rows = [("small", "LM1B-small", "97M · 3 × 256 samples"), ("medium", "LM1B-medium", "317M · 20 × 256 samples"),
            ("large", "LM1B-large", "725M · 3 × 256 samples"), ("tinystories", "TinyStories", "3 × 256 samples")]
    rows = [r for r in rows if r[0] in ratios]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.0))
    ys = np.arange(len(rows))[::-1]
    for y, (key, name, sub) in zip(ys, rows):
        for m, marker, fill, dy in ((6, "o", ps.BLUE, 0.12), (8, "D", ps.SURFACE, -0.12)):
            r = ratios[key].get(f"vcycle|m={m} / no-restrict|m={m}")
            if not r:
                continue
            ax.plot([r[1], r[2]], [y + dy, y + dy], color=ps.BLUE, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot([r[0]], [y + dy], marker=marker, ms=6 if marker == "o" else 5, color=ps.BLUE, mfc=fill,
                    mec=ps.BLUE, mew=1.2, zorder=3)
            if m == 6:
                ps.end_label(ax, r[2], y + dy, f"{r[0]:.3f}", dx=5, size=7, color=ps.INK2)
    ax.axvline(1.0, color=ps.AXIS, lw=0.9, zorder=1)
    ax.annotate("no effect", (1.0, ys[0] + 0.45), xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7, color=ps.MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{name}\n{sub}" for _, name, sub in rows], fontsize=7.5, color=ps.INK)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.7)
    ax.set_xlim(0.72, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Generative perplexity of the V-cycle relative to the equal-compute control (ratio, 95% CI)")
    ax.set_title("Re-encoding the coarse levels is what lowers perplexity")
    ax.legend(handles=[Line2D([], [], marker="o", color=ps.BLUE, ls="", ms=6, mec=ps.SURFACE, label="restart level m = 6 (+6 NFE)"),
                       Line2D([], [], marker="D", color=ps.BLUE, mfc=ps.SURFACE, ls="", ms=5, label="restart level m = 8 (+4 NFE)")],
              loc="upper left")
    ps.caption(fig, "Control: the same levels resampled from the generated (not re-encoded) coarse codes. Paired bootstrap "
                    "over samples; GPT-2-large scorer, the paper's protocols.")
    ps.save(fig, os.path.join(FIG, "fig1_restriction_effect"))


# ---- Figure 2: quality against diversity (LM1B-small, seed 42) ------------------------------------------------------
def fig_frontier():
    sw, cv = load("lm1b_sweep_small_s42.json"), load("lm1b_converge_small_s42.json")
    if not sw or not cv:
        return
    V = sw["variants"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    temps = sorted([(1.0, V["baseline"])] + [(float(k.split("T=")[1]), r) for k, r in V.items() if k.startswith("baseline|T=")],
                   reverse=True)
    ps.line(ax, [r["corpus_distinct2"] for _, r in temps], [r["gen_ppl"] for _, r in temps], ps.GRAY,
            label="NCM, lower sampling temperature", z=2)
    for T, r in temps:
        ps.end_label(ax, r["corpus_distinct2"], r["gen_ppl"], f"T = {T:g}" + (" (released)" if T == 1.0 else ""),
                     dx=0, dy=8, ha="center", va="bottom", size=7)
    for key, ls in (("m=8", "-"), ("m=6", (0, (4, 2)))):
        run = cv["runs"].get(key)
        if not run:
            continue
        cy = run["cycles"]
        ps.line(ax, [c["corpus_distinct2"] for c in cy], [c["gen_ppl"] for c in cy], ps.BLUE, ls=ls,
                label=f"V-cycle, restart level {run['m']}, cycles 0–6", z=4)
        ps.end_label(ax, cy[-1]["corpus_distinct2"], cy[-1]["gen_ppl"], f"m = {run['m']}, cycle 6", dx=-7, dy=0,
                     ha="right", size=7)
    nr = [(r["corpus_distinct2"], r["gen_ppl"]) for k, r in V.items() if k.startswith("no-restrict|")]
    ax.scatter([x for x, _ in nr], [y for _, y in nr], marker="s", s=26, color=ps.ORANGE, edgecolor=ps.SURFACE,
               linewidth=1, zorder=3, label="No restriction, m = 4–10 (one cycle)")
    lo = min(c["gen_ppl"] for r in cv["runs"].values() for c in r["cycles"])
    ax.set_ylim(lo - 4, max(r["gen_ppl"] for _, r in temps) + 5)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.004, x1 + 0.004)
    ax.set_xlabel("Corpus distinct-2 across samples (higher = more diverse)")
    ax.set_ylabel("Generative perplexity (GPT-2 large)")
    ax.set_title("The V-cycle moves beyond NCM's own temperature frontier")
    ax.legend(loc="upper left")
    ps.caption(fig, "LM1B-small, seed 42, 256 samples; one point per cycle. Lower and further right is better: at equal "
                    "diversity the V-cycle reaches lower perplexity than any temperature. Human text: perplexity 53.5, "
                    "corpus distinct-2 0.72 (off the chart).")
    ps.save(fig, os.path.join(FIG, "fig2_quality_vs_diversity"))


# ---- Figure 3: iterating the V-cycle (LM1B-medium, seed 42) ---------------------------------------------------------
def fig_iteration():
    cv = load("lm1b_converge_medium_s42.json")
    if not cv:
        return
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
    for key, ls in (("m=8", "-"), ("m=6", (0, (4, 2)))):
        run = cv["runs"][key]
        m, cy = run["m"], run["cycles"]
        x = [c["cycle"] for c in cy]
        ps.line(axl, x, [c["gen_ppl"] for c in cy], ps.BLUE, ls=ls, label=f"V-cycle, restart level {m}")
        ps.end_label(axl, x[-1], cy[-1]["gen_ppl"], f"m = {m}", dx=6, size=7)
        kept = [float(np.mean(c["roundtrip_disagree_per_level"][:m])) for c in cy]
        ps.line(axr, x, kept, ps.BLUE, ls=ls, label=f"restart level {m}")
        ps.end_label(axr, x[-1], kept[-1], f"m = {m}", dx=6, size=7)
        nr = run.get("no_restrict")
        if nr:
            axl.scatter([1], [nr["gen_ppl"]], marker="s", s=26, color=ps.ORANGE, edgecolor=ps.SURFACE, linewidth=1, zorder=4,
                        label="No restriction, one cycle" if key == "m=8" else None)
    if HUMAN_PPL:
        axl.axhline(HUMAN_PPL, color=ps.GRAY, lw=1, ls=(0, (3, 2)), zorder=1)
        axl.annotate(f"human text ({HUMAN_PPL:.1f})", (0, HUMAN_PPL), xytext=(0, 3), textcoords="offset points",
                     ha="left", va="bottom", fontsize=7, color=ps.MUTED)
    axl.set_xlabel("Cycle (0 = single NCM pass)")
    axl.set_ylabel("Generative perplexity (GPT-2 large)")
    axl.set_title("a  Perplexity")
    h, l = axl.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: ("No restriction" in l[i], l[i]), reverse=False)
    axl.legend([h[i] for i in order], [l[i] for i in order], loc="upper right")
    axr.set_xlabel("Cycle")
    axr.set_ylabel("Round-trip disagreement, kept levels")
    axr.set_title("b  Coarse residual")
    axr.set_ylim(0, 1)
    for a in (axl, axr):
        a.set_xticks(range(0, 7))
        a.set_xlim(-0.3, 6.9)
    ps.caption(fig, "LM1B-medium, seed 42, 256 samples, +4 (m = 8) or +6 (m = 6) NFE per cycle. From level 8 the residual "
                    "contracts and the chain settles; from level 6 it keeps drifting, below human-text perplexity.", y=-0.04)
    ps.save(fig, os.path.join(FIG, "fig3_iteration"))


# ---- Figure 4: Sudoku retries at matched compute ----------------------------------------------------------------------
SUDOKU_SERIES = [("N", "Best-of-N (fresh restarts)", ps.GRAY, "o", "-"),
                 ("E", "V-cycle retries (level 5)", ps.BLUE, "o", "-"),
                 ("F", "Fresh restarts in random symmetry frames", ps.ORANGE, "s", "-"),
                 ("A", "Symmetry-twisted V-cycle", ps.AQUA, "D", "-"),
                 ("D", "Soft decisions + symmetry-twisted V-cycle", ps.YELLOW, "^", "-")]


def mean_curve(curves, xmax):
    x = np.mean([c["nfe"] for c in curves], 0)
    y = 100 * np.mean([c["solved"] for c in curves], 0)
    keep = x <= xmax + 1e-9
    return x[keep], y[keep]


def fig_sudoku():
    panels = [("sudoku_stack_hard_nosingles.json", "a  NCM alone"), ("sudoku_stack_hard.json", "b  With singles propagation (0 NFE)")]
    data = [(load(f), t) for f, t in panels]
    data = [(d, t) for d, t in data if d]
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(7.2, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for ax, (d, title) in zip(axes, data):
        for key, label, color, marker, ls in SUDOKU_SERIES:
            if key not in d["curves"]:
                continue
            x, y = mean_curve(d["curves"][key], 20.0)
            ps.line(ax, x, y, color, label=label, marker=marker, ms=4.2, ls=ls, markevery=list(range(min(len(x), 9))))
        ax.axhline(86.1, color=ps.AXIS, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.annotate("NCM paper, single pass (86.1%)", (20, 86.1), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=6.8, color=ps.MUTED)
        ax.set_xlim(11.6, 20.2)
        ax.set_xticks(range(12, 21, 2))
        ax.set_xlabel("Mean generator forward passes per board (NFE)")
        ax.set_title(title)
    axes[0].set_ylabel("Hard boards solved (%)")
    axes[0].set_ylim(84.5, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.135), fontsize=7.2)
    ps.caption(fig, "2,000 hard boards, 3 seeds. A solution-blind checker (valid grid, clues kept) decides which boards are "
                    "retried; one point per retry round. Singles propagation fills the cells forced by row, column and box "
                    "constraints. Markers show the first eight retry rounds.", y=-0.17)
    ps.save(fig, os.path.join(FIG, "fig6_sudoku_retries"))


# ---- Figure 5: the residual as an error detector (Sudoku) ------------------------------------------------------------
def fig_residual():
    rows = []
    for d in ("easy", "medium", "hard"):
        p1 = load(f"phase1_sudoku_{d}.json")
        if p1:
            ca = p1["confidence_auroc"]
            rows.append((d, p1["local"]["auroc"]["n_disagree_positions"], ca["1 - mean decoder prob"],
                         ca["1 - min generator prob (all levels)"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ys = np.arange(len(rows))[::-1]
    series = [(1, "Round-trip residual (level-local)", ps.BLUE, "o"), (2, "Decoder confidence", ps.ORANGE, "s"),
              (3, "Generator confidence", ps.AQUA, "D")]
    for y, row in zip(ys, rows):
        vals = [row[i] for i, *_ in series]
        ax.plot([min(vals), max(vals)], [y, y], color=ps.GRID, lw=2.2, zorder=1, solid_capstyle="round")
    for i, label, color, marker in series:
        ax.scatter([r[i] for r in rows], ys, s=34, marker=marker, color=color, edgecolor=ps.SURFACE, linewidth=1,
                   zorder=3, label=label)
    for y, row in zip(ys, rows):
        ps.end_label(ax, row[1], y, f"{row[1]:.2f}", dx=7, size=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}" for r in rows], color=ps.INK)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlim(0.55, 0.98)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC for flagging wrong boards (0.5 = chance)")
    ax.set_title("The round-trip residual flags wrong Sudoku boards better than confidence", pad=22)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, handletextpad=0.3, columnspacing=1.6)
    ps.caption(fig, "2,000 boards per difficulty, NCM greedy pass. No solution is used to compute any detector.", y=-0.06)
    ps.save(fig, os.path.join(FIG, "fig5_residual_detector"))


# ---- Figure 6: prompted tasks -----------------------------------------------------------------------------------------
def fig_prompted():
    w, x = load("wmt14_vcycle.json"), load("xsum_vcycle.json")
    if not w or not x:
        return
    ramp = {3: "#104281", 5: ps.BLUE, 7: "#86b6ef"}  # one hue, ordered by restart level
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3})

    def draw(ax, V, metric, scale):
        for m in (3, 5, 7):
            pts = []
            for c in (1, 2, 3):
                r = V.get(f"vcycle|m={m}|c={c}")
                if r and metric in r.get("vs_baseline", {}):
                    d = r["vs_baseline"][metric]
                    pts.append((r["extra_nfe"], scale * d[0], scale * d[1], scale * d[2]))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ax.errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], fmt="none",
                        ecolor=ramp[m], elinewidth=1, capsize=0, zorder=2)
            ps.line(ax, xs, [p[1] for p in pts], ramp[m], label=f"V-cycle, restart level {m}")
        ax.axhline(0, color=ps.AXIS, lw=0.9, zorder=1)

    draw(axl, w["variants"], "bleu", 1.0)
    axl.scatter([11], [1.3], marker="D", s=30, color=ps.GRAY, edgecolor=ps.SURFACE, linewidth=1, zorder=4,
                label="Classifier-free guidance, NCM paper")
    ps.end_label(axl, 11, 1.3, "CFG (lowercased BLEU)", dx=7, size=6.8)
    axl.set_xlabel("Extra NFE per sentence")
    axl.set_ylabel("Change in BLEU vs single pass")
    axl.set_title("a  WMT14 German→English")

    draw(axr, x["variants"], "rouge2", 1.0)
    axr.set_xlabel("Extra NFE per article")
    axr.set_ylabel("Change in ROUGE-2 vs single pass")
    axr.set_title("b  XSum summarization")
    h, l = axl.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=7.2)
    ps.caption(fig, "Released greedy decoding; one point per cycle (1–3), 95% paired-bootstrap intervals over all 3,003 "
                    "sentences and 11,334 articles.", y=-0.14)
    ps.save(fig, os.path.join(FIG, "fig4_prompted_tasks"))


for f in (fig_restriction, fig_frontier, fig_iteration, fig_sudoku, fig_residual, fig_prompted):
    f()
