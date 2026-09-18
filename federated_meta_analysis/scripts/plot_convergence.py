# Reads:  results/prs_pheno*.csv (or the finer-grained prs_fine_pheno*.csv), data/syn_rel1_chr.pheno*
# Writes: results/figures/convergence.pdf and results/figures/png/convergence.png
# Does:   plots how much of each trait's genetic signal is still unclaimed as the federation grows

# For each trait, 1 - R2/h2 is the share of its genetic variance the polygenic score has not captured. The bold
# line is the mean of that across the six traits; each trait is also drawn faintly behind it, because the traits
# are not close together - one of them is still holding most of its signal at the largest sample size. The lower
# panel is the derivative per 1,000 additional individuals, negative while the federation is still gaining and
# approaching zero once it has taken what this much data can give. The x-axis is cumulative training individuals
# on a continuous scale, so uneven gaps between site joins stay visible.

# Style follows plot_styleguide/plot_style_guide.md

import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

RESULTS_GLOB, PHENOTYPE_DIR, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
BOOTSTRAP = sys.argv[4] if len(sys.argv) > 4 else None   # 95% band from resampling the held-out individuals
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
LINE, GHOST = SUNSET(0.15), "#999999"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.4,
})

table = pd.concat([pd.read_csv(path) for path in sorted(glob.glob(RESULTS_GLOB))], ignore_index=True)
best = (table.groupby(["trait", "repeat", "n_sites"])["r2"].max()
        .groupby(["trait", "n_sites"]).mean().reset_index())
train_n = table.groupby("n_sites")["train_individuals"].mean().round().astype(int)
steps = sorted(best["n_sites"].unique())

heritability = {}
for path in glob.glob(f"{PHENOTYPE_DIR}/syn_rel1_chr.pheno*"):
    trait = "pheno" + re.search(r"pheno(\d+)$", path).group(1)
    values = pd.read_csv(path, sep="\t")
    heritability[trait] = values["GenoEff"].var() / values["Phenotype(liability)"].var()

best["remaining"] = 1 - best["r2"] / best["trait"].map(heritability)
mean = best.groupby("n_sites")["remaining"].mean()
x = np.array([train_n[s] for s in steps], dtype=float)
centre = mean.reindex(steps).to_numpy()

band_lo = band_hi = None
if BOOTSTRAP:                                   # a high R2 is a low remaining fraction, so the bounds swap
    draws = pd.read_csv(BOOTSTRAP)
    draws["lo"] = 1 - draws["r2_hi"] / draws["trait"].map(heritability)
    draws["hi"] = 1 - draws["r2_lo"] / draws["trait"].map(heritability)
    available = draws[draws["n_sites"].isin(steps)]
    band_lo = available.groupby("n_sites")["lo"].mean().reindex(steps).to_numpy()
    band_hi = available.groupby("n_sites")["hi"].mean().reindex(steps).to_numpy()

rate_x = x[1:]
rate = np.diff(centre) / (np.diff(x) / 1000)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True,
                                  gridspec_kw={"height_ratios": [1.55, 1.0]})
fig.subplots_adjust(left=0.105, right=0.975, top=0.855, bottom=0.10, hspace=0.13)

for trait in sorted(best["trait"].unique()):        # one unlabelled ghost line per trait, behind the mean
    trace = best[best["trait"] == trait].set_index("n_sites")["remaining"].reindex(steps).to_numpy()
    top.plot(x, trace, "-", lw=0.9, color=GHOST, alpha=0.20, zorder=1)
    bottom.plot(rate_x, np.diff(trace) / (np.diff(x) / 1000), "-", lw=0.9, color=GHOST, alpha=0.20, zorder=2)
if band_lo is not None:
    top.fill_between(x, band_lo, band_hi, color=LINE, alpha=0.15, lw=0, zorder=2)
top.plot(x, centre, "-o", ms=3.5, color=LINE, zorder=3)
top.set_ylim(0, 1.02)
bottom.axhline(0, color="#dddddd", lw=0.6, zorder=1)
bottom.plot(rate_x, rate, "-o", ms=3.5, color=LINE, zorder=3)

for axis in (top, bottom):
    for step in steps:                                  # a light rule wherever a site joins
        axis.axvline(train_n[step], color="#ececec", lw=0.7, zorder=0)
for step in steps:
    if float(step).is_integer():
        top.annotate(f"site {int(step)}", (train_n[step], 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=5.8,
                     color="#999999", rotation=90, annotation_clip=False)

top.set_ylabel("Fraction of genetic signal remaining", fontsize=8, labelpad=6)
bottom.set_ylabel("Δ fraction per 1,000", fontsize=8, labelpad=6)
bottom.set_xlabel("Cumulative training individuals", fontsize=8, labelpad=6)
bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
bottom.set_xticks([train_n[s] for s in steps if float(s).is_integer()])
for axis in (top, bottom):
    axis.tick_params(labelsize=7)

fig.suptitle("Convergence of federated genotype-phenotype discovery", fontsize=9, fontweight="bold",
             color="#111111", x=0.5, y=0.995)
fig.text(0.5, 0.947, "Mean across six traits (grey lines each an individual trait)",
         ha="center", va="bottom", fontsize=7.5, color="#888888")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
PNG = os.path.join(os.path.dirname(OUTPUT), "png", os.path.basename(OUTPUT).replace(".pdf", ".png"))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
for path, dots in [(OUTPUT, None), (PNG, 300)]:
    fig.savefig(path, dpi=dots, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"[plot_convergence] Saved {OUTPUT}  ({len(steps)} accretion points)")
for step, value, rate_value in zip(steps, centre, [np.nan, *rate]):
    print(f"[plot_convergence]   N {train_n[step]:>6,}  remaining {value:.3f}"
          + (f"   rate {rate_value:+.4f}" if np.isfinite(rate_value) else ""))
