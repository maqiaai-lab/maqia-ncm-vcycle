"""Shared figure style: one typeface, one palette, thin marks, recessive axes.

Categorical colours follow a fixed slot order and stay attached to the same method in every figure (the V-cycle is
always slot 1). The palette was checked for colour-vision-deficiency separation; the two light slots (aqua, yellow)
fall below 3:1 contrast on white, so every figure that uses them also labels the series directly.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e6e5df", "#bdbcb4", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY = "#8c8b85"  # baselines and reference curves

# Method -> colour, fixed across figures.
METHOD = {
    "vcycle": BLUE,
    "no_restriction": ORANGE,
    "temperature": GRAY,
    "single_pass": GRAY,
}

FONTS = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONTS[1],
        "mathtext.it": f"{FONTS[1]}:italic",
        "mathtext.bf": f"{FONTS[1]}:bold",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.labelcolor": INK2,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,  # embed TrueType, so text stays text in the PDF
        "axes.unicode_minus": True,
    })


def line(ax, x, y, color, label=None, marker="o", ls="-", z=3, ms=4.6, **kw):
    """A 2 px line with markers that carry a white ring, so crossings stay legible."""
    return ax.plot(x, y, color=color, ls=ls, marker=marker, ms=ms, mec=SURFACE, mew=1.0, label=label, zorder=z, **kw)


def end_label(ax, x, y, text, dx=5, dy=0, ha="left", va="center", size=7.5, color=INK2):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size, color=color)


def caption(fig, text, y=-0.02):
    """A short source note under the axes, in muted ink, wrapped to the figure width."""
    import textwrap
    width = int(fig.get_figwidth() * 19)  # characters per line at 6.8 pt
    fig.text(0.0, y, "\n".join(textwrap.wrap(text, width)), ha="left", va="top", fontsize=6.8, color=MUTED,
             linespacing=1.35, transform=fig.transFigure)


def save(fig, path_stem):
    """PNG (300 dpi) for the README and a vector PDF next to it."""
    fig.savefig(path_stem + ".png")
    fig.savefig(path_stem + ".pdf")
    plt.close(fig)
    print("figure:", path_stem + ".png")
