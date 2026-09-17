# Reads:  results/temporal_years.csv, results/temporal_excluding_top_studies.csv
# Writes: results/figures/temporal_accretion.pdf
# Does:   plots recall on each year's new GWAS edges at fixed K and at K = 1% of known phenotypes,
#         graph/test size over time, and the effect of excluding each year's top publications

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

YEARS, EXCLUDED, OUTPUT, TOP_K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

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


def plot_sizes(ax, years):
    """Training edges (first reported before the year) and test edges (new that year)."""
    ax.plot(years["year"], years["train_edges"], "-o", color=C_LIGHT, ms=2.5, label="Training edges")
    ax.plot(years["year"], years["test_edges"], "-o", color=C_DARK, ms=2.5, label="Test edges")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:,.0f}k" if y >= 1000 else f"{y:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(years["test_edges"].min() / 2, years["train_edges"].max() * 20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    finish(ax, "(c)  Graph and test set size", "Test year", "Edges", "upper left")


def plot_exclusion(ax, years, excluded):
    """Recall at K = 1% with all publications (circles) vs without the year's top 5 (open squares)."""
    both = years.merge(excluded, on="year", suffixes=("", "_excl"))
    x = range(len(both))
    for col, color, offset, label in [("recall_k_fraction", C_LIGHT, -0.12, "SVD"),
                                      ("popularity_k_fraction", C_DARK, 0.12, "Popularity")]:
        xs = [i + offset for i in x]
        ax.vlines(xs, both[col], both[f"{col}_excl"], color=color, lw=0.8)
        ax.plot(xs, both[col], "o", color=color, ms=4, label=f"{label}, all publications")
        ax.plot(xs, both[f"{col}_excl"], "s", mfc="white", color=color, ms=4, label=f"{label}, without top 5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(both["year"])
    ax.set_xlim(-0.6, len(both) - 0.4)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, both.filter(like="_k_fraction").max().max() * 1.6)
    finish(ax, "(d)  Without top 5 publications", "Test year", "Recall at K = 1%", "upper left")


years, excluded = pd.read_csv(YEARS), pd.read_csv(EXCLUDED)
fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 6.2))
plot_recall(axes[0, 0], years, "k_fixed", f"(a)  Fixed K = {TOP_K}", f"Recall@{TOP_K} on new edges")
plot_recall(axes[0, 1], years, "k_fraction", "(b)  K = 1% of known phenotypes", "Recall at K = 1%")
plot_sizes(axes[1, 0], years)
plot_exclusion(axes[1, 1], years, excluded)

fig.suptitle("Temporal replay: predicting each year's new GWAS edges",
             fontsize=FS["title"], fontweight="bold", color="#111111", y=1.0)
fig.text(0.5, 0.955,
         f"Train: pairs reported before the year  |  Test: new pairs between known nodes  |  "
         f"{int(years['year'].min())}–{int(years['year'].max())}",
         ha="center", fontsize=FS["subtitle"], color="#555555")
fig.subplots_adjust(top=0.89, wspace=0.45, hspace=0.55)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, **SAVE_KW)
print(f"[plot_temporal] Saved {OUTPUT}")
