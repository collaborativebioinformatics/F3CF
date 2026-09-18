# Reads:  results/prs_accretion.csv (federated PRS results for one trait)
# Writes: results/figures/prs_accretion.pdf
# Does:   plots PRS accuracy in held-out individuals against the number of sites in the meta-analysis

# The line takes the best p-value threshold at each step; the thin lines behind it are the individual thresholds,
# which show how much of the gain depends on that choice. Points are labelled with the number of individuals
# contributing to the meta-analysis. The dashed line is the trait's genetic variance share: the share of trait
# variance the simulation put in the genotypes, so no genotype-based predictor can pass it.

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

INPUT, OUTPUT = sys.argv[1], sys.argv[2]
TRAIT = sys.argv[4] if len(sys.argv) > 4 else "Height"
CEILING = float(sys.argv[3]) if len(sys.argv) > 3 else None   # trait's genetic variance share
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
})

table = pd.read_csv(INPUT)
per_threshold = table.groupby(["threshold", "n_sites"])["r2"].mean().reset_index()
best = (table.groupby(["repeat", "n_sites", "train_individuals"])["r2"].max()
        .groupby(["n_sites", "train_individuals"]).mean().reset_index())

fig, ax = plt.subplots(figsize=(6.5, 3.2))
fig.subplots_adjust(left=0.115, right=0.975, top=0.866, bottom=0.175)
for threshold, rows in per_threshold.groupby("threshold"):
    ax.plot(rows["n_sites"], rows["r2"], "-", lw=0.6, color="#dddddd", zorder=1)
ax.plot(best["n_sites"], best["r2"], "-o", ms=3.5, color=SUNSET(0.3), zorder=2, label="Best of 9 inclusion thresholds (grey lines show each)")

if CEILING is not None:
    ax.axhline(CEILING, color="#999999", lw=0.9, ls="--", zorder=0)
    ax.annotate(f"heritability ceiling ({CEILING:.3f})", (ax.get_xlim()[0], CEILING), xytext=(2, 4),
                textcoords="offset points", fontsize=6.5, color="#555555")
ax.set_ylim(0, max(best["r2"].max() * 1.25, (CEILING or 0) * 1.12))
steps = sorted(table["n_sites"].unique())
train_n = best.set_index("n_sites")["train_individuals"].round().astype(int)
ax.set_xticks(steps)
ax.set_xticklabels([f"{step}\n(N = {train_n[step]:,})" for step in steps])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.set_xlabel("Sites in the meta-analysis", fontsize=8, labelpad=6)
ax.set_ylabel("PRS R² in held-out individuals", fontsize=8, labelpad=6)
ax.tick_params(labelsize=6.8)
ax.legend(frameon=False, labelcolor="#555555", fontsize=6.0, loc="lower right",
          handlelength=1.8, handletextpad=0.6, borderpad=0.1, borderaxespad=0.6)
fig.suptitle(f"Federated PRS accuracy for {TRAIT} as sites join the meta-analysis", fontsize=9,
             fontweight="bold", color="#111111", x=0.5, y=0.972)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
PNG = os.path.join(os.path.dirname(OUTPUT), "png",
                   os.path.basename(OUTPUT).replace(".pdf", ".png"))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
for path, dots in [(OUTPUT, None), (PNG, 300)]:
    fig.savefig(path, dpi=dots, bbox_inches="tight", pad_inches=0.01, facecolor="white")
print(f"[plot_prs_accretion] Saved {OUTPUT}")
