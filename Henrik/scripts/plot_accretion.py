# Reads:  results/accretion.csv (accretion summary across seeds)
# Writes: results/figures/accretion.pdf
# Does:   plots (a) recall vs GWAS training size and (b) paired change when other data is added

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

INPUT, OUTPUT, TOP_K = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = f"recall@{TOP_K}"

TEXTWIDTH = 6.5
USE_TEX = False
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5, "annot": 5.5}
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.01, facecolor="white")
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
C_SVD, C_POP = SUNSET(0.82), SUNSET(0.17)   # ends of the style guide's 3-color sample
ACCENT = "#CC5A20"
PRETTY = {"opentargets": "OpenTargets", "clinvar": "ClinVar"}
plt.rcParams.update({
    "text.usetex": USE_TEX, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"] if USE_TEX else ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2,
})


def short_label(name):
    """'GWAS + opentargets + clinvar' -> '+ OpenTargets\\n+ ClinVar'"""
    for key, value in PRETTY.items():
        name = name.replace(key, value)
    return name.replace("GWAS ", "").replace(" + ", "\n+ ").lstrip("\n")


def panel_title(ax, text):
    ax.set_title(text, fontsize=FS["title"] - 1, fontweight="semibold", color="#222222", pad=8)


def plot_gwas_curve(ax, gwas):
    """(a) Recall vs GWAS training edges, with 95% bands across seeds."""
    for col, color, label in [(R, C_SVD, "SVD"), (f"{R}_popularity", C_POP, "Popularity baseline")]:
        ax.fill_between(gwas["train_edges"], gwas[f"{col}_ci_low"], gwas[f"{col}_ci_high"],
                        color=color, alpha=0.25, lw=0)
        ax.plot(gwas["train_edges"], gwas[col], "-o", color=color, ms=2.5, label=label)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k" if x >= 1000 else f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, None)
    ax.set_xlabel("GWAS training edges", fontsize=FS["label"])
    ax.set_ylabel(f"Recall@{TOP_K} on held-out GWAS edges", fontsize=FS["label"], labelpad=6)
    panel_title(ax, "(a)  Accretion of GWAS edges")
    ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc="lower right")


def plot_source_changes(ax, cross):
    """(b) Paired change vs all GWAS (same test set per seed), with 95% intervals across seeds."""
    d = "delta_vs_gwas"
    ypos = list(range(len(cross)))[::-1]
    for y, (_, row) in zip(ypos, cross.iterrows()):
        err = [[(row[d] - row[f"{d}_ci_low"]) * 100], [(row[f"{d}_ci_high"] - row[d]) * 100]]
        ax.errorbar(row[d] * 100, y, xerr=err, fmt="o", color=ACCENT, ms=4, elinewidth=1.0, capsize=2)
        ax.annotate(f"{row[d] * 100:+.2f} pp  ({int(row['seeds_improved'])}/{int(row['n_seeds'])} seeds)",
                    (row[f"{d}_ci_high"] * 100, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=FS["annot"] + 0.5, color="#333333")
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([short_label(name) for name in cross["data"]], linespacing=1.3)
    ax.set_ylim(-0.6, len(cross) - 0.4)
    lo = min(cross[f"{d}_ci_low"].min(), 0) * 100
    hi = max(cross[f"{d}_ci_high"].max(), 0) * 100
    span = max(hi - lo, 0.5)
    ax.set_xlim(lo - 0.15 * span, hi + 0.9 * span)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlabel(f"Change in recall@{TOP_K} vs all GWAS (pp)", fontsize=FS["label"])
    panel_title(ax, "(b)  Adding other sources")


df = pd.read_csv(INPUT)
gwas = df[df["kind"] == "gwas_fraction"]
cross = df[df["kind"] != "gwas_fraction"]

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.6))
plot_gwas_curve(axes[0], gwas)
plot_source_changes(axes[1], cross)
for ax in axes:
    ax.tick_params(labelsize=FS["tick"])

last = gwas.iloc[-1]
fig.suptitle("Recall on held-out GWAS edges as datasets accrete",
             fontsize=FS["title"], fontweight="bold", color="#111111", y=1.0)
fig.text(0.5, 0.915,
         f"SVD  |  fixed test set of {int(last['test_edges']):,} GWAS edges  |  "
         f"{int(last['n_seeds'])} seeds, 95% t-interval",
         ha="center", fontsize=FS["subtitle"], color="#555555")
fig.subplots_adjust(top=0.80, wspace=0.45)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, **SAVE_KW)
print(f"[plot_accretion] Saved {OUTPUT}")
