# Reads:  results/temporal_years.csv
# Writes: results/figures/temporal_accretion.pdf
# Does:   plots recall on each year's new GWAS edges at fixed K and at K = 1% of known phenotypes

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

YEARS, OUTPUT, TOP_K = sys.argv[1], sys.argv[2], int(sys.argv[3])

TEXTWIDTH = 6.5
USE_TEX = False
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5, "annot": 5.5}
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.01, facecolor="white")
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
C_LIGHT, C_DARK = SUNSET(0.82), SUNSET(0.17)   # ends of the style guide's 3-color sample
PCT = FuncFormatter(lambda y, _: f"{y:.0%}")
plt.rcParams.update({
    "text.usetex": USE_TEX, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"] if USE_TEX else ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2,
})


def finish(ax, title, xlabel, ylabel, legend_loc):
    ax.set_title(title, fontsize=FS["title"] - 1, fontweight="semibold", color="#222222", pad=8)
    ax.set_xlabel(xlabel, fontsize=FS["label"], labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS["label"], labelpad=6)
    ax.tick_params(labelsize=FS["tick"])
    ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc=legend_loc)


def plot_recall(ax, years, label, title, ylabel):
    """Recall on each year's new edges, SVD vs popularity baseline, for one choice of K."""
    ax.plot(years["year"], years[f"recall_{label}"], "-o", color=C_LIGHT, ms=2.5, label="SVD")
    ax.plot(years["year"], years[f"popularity_{label}"], "-o", color=C_DARK, ms=2.5, label="Popularity baseline")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, max(years[f"recall_{label}"].max(), years[f"popularity_{label}"].max()) * 1.45)
    finish(ax, title, "Test year", ylabel, "upper right")


years = pd.read_csv(YEARS)
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.4))
plot_recall(axes[0], years, "k_fixed", f"(a)  Fixed K = {TOP_K}", f"Recall@{TOP_K}")
plot_recall(axes[1], years, "k_fraction", "(b)  K = 1% of known phenotypes", "Recall at K = 1%")

fig.suptitle("Recall on each year's new GWAS edges",
             fontsize=FS["title"], fontweight="bold", color="#111111", y=1.0)
fig.text(0.5, 0.915, "Train: pairs reported before the year  |  Test: new pairs that year",
         ha="center", fontsize=FS["subtitle"], color="#555555")
fig.subplots_adjust(top=0.80, wspace=0.45)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, **SAVE_KW)
print(f"[plot_temporal] Saved {OUTPUT}")
