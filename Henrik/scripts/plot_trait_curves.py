# Reads:  results/prs_pheno*.csv, data/syn_rel1_chr.pheno* (for each trait's heritability)
# Writes: results/figures/trait_curves.pdf
# Does:   plots PRS accuracy against number of federated sites, one coloured line per trait

# For each trait and accretion step the best p-value threshold is taken (chosen on the held-out individuals, so
# these are slightly optimistic in level; the shape and the ordering between traits are unaffected). Each trait's
# dashed line, in its own colour, is its heritability: the share of trait variance the simulation put in the
# genotypes, which no genotype-based predictor can pass. Traits are ordered by heritability in the legend.

# Style follows plot_styleguide/plot_style_guide.md

import glob
import os
import re
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

RESULTS_GLOB, PHENOTYPE_DIR, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
BOOTSTRAP = sys.argv[4] if len(sys.argv) > 4 else None   # 95% band from resampling the held-out individuals
NAMES = {"pheno1": "Height", "pheno2": "BMI", "pheno3": "HDL cholesterol", "pheno4": "Diastolic BP",
         "pheno5": "Triglycerides", "pheno6": "Body weight", "pheno7": "Heart rate",
         "pheno8": "Alcohol consumption", "pheno9": "HbA1c", "pheno10": "Lymphocyte count"}
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
})

table = pd.concat([pd.read_csv(path) for path in sorted(glob.glob(RESULTS_GLOB))], ignore_index=True)
best = (table.groupby(["trait", "repeat", "n_sites"])["r2"].max()
        .groupby(["trait", "n_sites"]).mean().reset_index())
train_n = table.groupby("n_sites")["train_individuals"].mean().round().astype(int)   # varies slightly by split

heritability = {}
for path in glob.glob(f"{PHENOTYPE_DIR}/syn_rel1_chr.pheno*"):
    trait = "pheno" + re.search(r"pheno(\d+)$", path).group(1)
    values = pd.read_csv(path, sep="\t")
    heritability[trait] = values["GenoEff"].var() / values["Phenotype(liability)"].var()

traits = sorted(best["trait"].unique(), key=lambda t: -heritability.get(t, 0))
colours = [SUNSET(0.05 + 0.83 * i / max(len(traits) - 1, 1)) for i in range(len(traits))]


def readable(colour):
    """The pale end of the ramp is fine as a line but unreadable as text, so cap the text luminance."""
    red, green, blue = colour[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    limit = 0.42
    return colour if luminance <= limit else tuple(channel * limit / luminance for channel in (red, green, blue))

fig, ax = plt.subplots(figsize=(6.5, 3.2))
fig.subplots_adjust(left=0.09, right=0.71, top=0.866, bottom=0.16)   # reserve the right band for the line labels
bands = pd.read_csv(BOOTSTRAP).set_index(["trait", "n_sites"]) if BOOTSTRAP else None
entries = []                                   # colour encodes heritability rank, legend follows the lines
for trait, colour in zip(traits, colours):
    rows = best[best["trait"] == trait].sort_values("n_sites")
    ceiling = heritability.get(trait)
    ax.axhline(ceiling, color=colour, lw=0.7, ls="--", alpha=0.55, zorder=0)
    if bands is not None:
        window = bands.reindex([(trait, step) for step in rows["n_sites"]])
        ax.fill_between(rows["n_sites"], window["r2_lo"].to_numpy(), window["r2_hi"].to_numpy(),
                        color=colour, alpha=0.16, lw=0, zorder=1)
    line, = ax.plot(rows["n_sites"], rows["r2"], "-o", ms=3, color=colour, zorder=2,
                    label=f"{NAMES.get(trait, trait)}  (h² = {ceiling:.2f})")
    entries.append((rows["r2"].iloc[-1], colour, f"{NAMES.get(trait, trait)}  (h\u00b2 = {ceiling:.2f})"))

entries.sort(key=lambda e: -e[0])                  # label each line at its right-hand end, nudged apart
placed, GAP = [], 0.034
for value, colour, text in entries:
    y = value if not placed else min(value, placed[-1][0] - GAP)
    placed.append((y, colour, text))

ax.set_ylim(0, max(heritability[t] for t in traits) * 1.10)
steps = sorted(best["n_sites"].unique())
ax.set_xticks(steps)
ax.set_xticklabels([f"{step}\n({train_n[step]:,})" for step in steps])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.set_xlabel("Sites in the meta-analysis (training individuals)", fontsize=8, labelpad=6)
ax.set_ylabel("PRS R²", fontsize=8, labelpad=6)
ax.tick_params(labelsize=7)
for y, colour, text in placed:
    ax.annotate(text, (steps[-1], y), xytext=(21, 0), textcoords="offset points", ha="left", va="center",
                fontsize=7.5, color=readable(colour), annotation_clip=False)
fig.suptitle("PRS accretion across six traits", fontsize=9,
             fontweight="bold", color="#111111", x=0.5, y=0.97)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
PNG = os.path.join(os.path.dirname(OUTPUT), "png",
                   os.path.basename(OUTPUT).replace(".pdf", ".png"))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
for path, dots in [(OUTPUT, None), (PNG, 300)]:
    fig.savefig(path, dpi=dots, bbox_inches="tight", pad_inches=0.01, facecolor="white")
print(f"[plot_trait_curves] Saved {OUTPUT}")
for trait in traits:
    rows = best[best["trait"] == trait].sort_values("n_sites")
    first, last = rows["r2"].iloc[0], rows["r2"].iloc[-1]
    print(f"[plot_trait_curves] {trait:8s} h2 {heritability[trait]:.3f}  1 site {first:.3f} -> 6 sites {last:.3f}"
          f"  gain {last - first:+.3f}  reaches {last / heritability[trait]:.0%} of ceiling")
