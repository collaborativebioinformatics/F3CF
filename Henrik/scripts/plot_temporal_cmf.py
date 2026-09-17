# Reads:  results/temporal_cmf.csv (recall per year x method x gene-degree group)
# Writes: results/figures/temporal_cmf.pdf
# Does:   plots (a) recall at K = 1% over years for SVD, CMF, CMF + STRING and (b) pooled recall by gene degree

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

INPUT, OUTPUT = sys.argv[1], sys.argv[2]

TEXTWIDTH = 6.5
USE_TEX = False
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5, "annot": 5.5}
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.01, facecolor="white")
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
METHODS = ["SVD", "CMF", "CMF + STRING"]
COLORS = dict(zip(METHODS, [SUNSET(0.82 - 0.65 * i / 2) for i in range(3)]))
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


def finish(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS["title"] - 1, fontweight="semibold", color="#222222", pad=8)
    ax.set_xlabel(xlabel, fontsize=FS["label"], labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS["label"], labelpad=6)
    ax.yaxis.set_major_formatter(PCT)
    ax.tick_params(labelsize=FS["tick"])
    ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc="upper left")


def plot_years(ax, table):
    """(a) Recall on each year's new edges, one line per method."""
    overall = table[table["degree_group"] == "all"]
    for method in METHODS:
        rows = overall[overall["method"] == method]
        ax.plot(rows["year"], rows["recall"], "-o", color=COLORS[method], ms=2.5, label=method)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_ylim(0, overall[overall["method"].isin(METHODS)]["recall"].max() * 1.35)
    finish(ax, "(a)  Recall over time", "Test year", "Recall at K = 1% of known phenotypes")


def plot_degree(ax, table):
    """(b) Recall pooled over all years, by the gene's number of phenotypes in training."""
    groups = [g for g in ["<5", "5-20", ">20"] if g in set(table["degree_group"])]
    pooled = table[table["method"].isin(METHODS)].groupby(["degree_group", "method"])[["hits", "n_test"]].sum()
    width = 0.26
    for i, method in enumerate(METHODS):
        recall = [pooled.loc[(g, method), "hits"] / pooled.loc[(g, method), "n_test"] for g in groups]
        ax.bar([j + (i - 1) * width for j in range(len(groups))], recall, width * 0.92,
               color=COLORS[method], label=method)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, (pooled["hits"] / pooled["n_test"]).max() * 1.45)
    finish(ax, "(b)  By gene degree, all years", "Gene's known phenotypes", "Recall at K = 1%")


table = pd.read_csv(INPUT)
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.6))
plot_years(axes[0], table)
plot_degree(axes[1], table)

fig.suptitle("Does STRING help predict next year's GWAS edges?",
             fontsize=FS["title"], fontweight="bold", color="#111111", y=1.0)
fig.text(0.5, 0.915, "Temporal replay  |  K = 1% of known phenotypes",
         ha="center", fontsize=FS["subtitle"], color="#555555")
fig.subplots_adjust(top=0.80, wspace=0.45)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, **SAVE_KW)
print(f"[plot_temporal_cmf] Saved {OUTPUT}")
