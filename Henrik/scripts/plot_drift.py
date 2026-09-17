# Reads:  results/drift.csv (mean embedding drift per year)
# Writes: results/figures/drift.pdf
# Does:   plots mean cosine drift of SNPs, genes and phenotypes over years

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

INPUT, OUTPUT = sys.argv[1], sys.argv[2]

TEXTWIDTH = 6.5
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5}
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
})

df = pd.read_csv(INPUT).dropna(subset=["drift_snp"])
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, 2.8))
for i, (col, label) in enumerate([("drift_snp", "SNPs"), ("drift_gene", "Genes"), ("drift_pheno", "Phenotypes")]):
    ax.plot(df["year"], df[col], "-o", ms=2.5, color=SUNSET(0.82 - 0.65 * i / 2), label=label)

ax.set_ylim(0, df[["drift_snp", "drift_gene", "drift_pheno"]].max().max() * 1.15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.set_xlabel("Year", fontsize=FS["label"], labelpad=6)
ax.set_ylabel("Mean cosine drift since previous year", fontsize=FS["label"], labelpad=6)
ax.tick_params(labelsize=FS["tick"])
ax.legend(frameon=False, labelcolor="#333333", fontsize=FS["legend"], loc="upper left")
fig.suptitle("How much each year reshapes the GWAS graph", fontsize=FS["title"], fontweight="bold",
             color="#111111", y=1.02)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.01, facecolor="white")
print(f"[plot_drift] Saved {OUTPUT}")
