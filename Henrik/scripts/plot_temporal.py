# Reads:  results/temporal_accretion.csv (one row per year)
# Writes: results/figures/temporal_accretion.pdf
# Does:   plots (a) recall@K on each year's new GWAS edges and (b) training and test set size over time

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

INPUT, OUTPUT, TOP_K = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = f"recall@{TOP_K}"

TEXTWIDTH = 6.5
USE_TEX = False
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5, "annot": 5.5}
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.01, facecolor="white")
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
C_LIGHT, C_DARK = SUNSET(0.82), SUNSET(0.17)   # ends of the style guide's 3-color sample
plt.rcParams.update({
    "text.usetex": USE_TEX, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"] if USE_TEX else ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2,
})


def style_years(ax, title):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("Test year", fontsize=FS["label"], labelpad=6)
    ax.set_title(title, fontsize=FS["title"] - 1, fontweight="semibold", color="#222222", pad=8)
    ax.tick_params(labelsize=FS["tick"])


def plot_recall(ax, df):
    """(a) Recall@K on each year's new edges, SVD vs popularity baseline."""
    ax.plot(df["year"], df[R], "-o", color=C_LIGHT, ms=2.5, label="SVD")
    ax.plot(df["year"], df[f"{R}_popularity"], "-o", color=C_DARK, ms=2.5, label="Popularity baseline")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, None)
    ax.set_ylabel(f"Recall@{TOP_K} on new edges", fontsize=FS["label"], labelpad=6)
    ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc="upper right")
    style_years(ax, "(a)  Predicting the next year")


def plot_sizes(ax, df):
    """(b) Training edges (everything before the year) and test edges (that year's new pairs)."""
    ax.plot(df["year"], df["train_edges"], "-o", color=C_LIGHT, ms=2.5, label="Training edges\n(first reported before)")
    ax.plot(df["year"], df["test_edges"], "-o", color=C_DARK, ms=2.5, label="Test edges\n(new that year)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k" if y >= 1000 else f"{y:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(df["test_edges"].min() / 2, df["train_edges"].max() * 20)   # headroom for the legend
    ax.set_ylabel("Edges", fontsize=FS["label"], labelpad=6)
    ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc="upper left", labelspacing=0.8)
    style_years(ax, "(b)  Graph and test set size")


df = pd.read_csv(INPUT)
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.8))
plot_recall(axes[0], df)
plot_sizes(axes[1], df)

fig.suptitle("Temporal replay: predicting each year's new GWAS edges",
             fontsize=FS["title"], fontweight="bold", color="#111111", y=1.0)
fig.text(0.5, 0.915,
         f"Train: pairs reported before the year  |  Test: new pairs between known nodes  |  "
         f"{int(df['year'].min())}–{int(df['year'].max())}",
         ha="center", fontsize=FS["subtitle"], color="#555555")
fig.subplots_adjust(top=0.80, wspace=0.5)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, **SAVE_KW)
print(f"[plot_temporal] Saved {OUTPUT}")
