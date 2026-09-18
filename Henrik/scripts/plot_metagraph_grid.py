# Reads:  results/_prs_work_pheno*/r0_s*.meta.txt, data/hapnest/all/all.bim, results/prs_pheno1.csv
# Writes: results/figures/metagraph_grid.pdf
# Does:   draws which loci are associated with which trait at each accretion step, one panel per step

# Nodes on the outer ring are 250kb genomic loci, placed at their real position along the genome (chromosome
# boundaries are the grey ticks). The six inner nodes are the traits, sized by how many loci they carry. An edge
# means at least one variant in that locus passed 5e-8 for that trait in the federated meta-analysis at that step.
# Loci rather than variants keep one signal in linkage disequilibrium from being drawn dozens of times.
# Repeat 0 is shown; the other two repeats give the same picture.

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

WORK_DIR, BFILE, ACCRETION, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
THRESHOLD, BLOCK, STEPS = 5e-8, 250_000, [1, 2, 3, 4, 5, 6]
NAMES = {"pheno1": "Height", "pheno2": "BMI", "pheno3": "HDL", "pheno6": "Body weight",
         "pheno7": "Heart rate", "pheno9": "HbA1c"}
HERITABILITY_ORDER = ["pheno1", "pheno9", "pheno6", "pheno2", "pheno3", "pheno7"]
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20", "#E09040", "#EDBC70", "#F5DCA8",
])
COLOUR = {t: SUNSET(0.05 + 0.83 * i / 5) for i, t in enumerate(HERITABILITY_ORDER)}
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

bim = pd.read_csv(f"{BFILE}.bim", sep="\t", header=None, usecols=[0, 3], names=["chrom", "pos"])
lengths = bim.groupby("chrom")["pos"].max()
offset, running = {}, 0
for chromosome in sorted(lengths.index):
    offset[chromosome] = running
    running += lengths[chromosome]
genome = running


def angle_of(block_key):
    chromosome, block = int(block_key.split(":")[0]), int(block_key.split(":")[1])
    return 2 * np.pi * (offset[chromosome] + block * BLOCK + BLOCK / 2) / genome


def loci_at(trait, step):
    table = pd.read_csv(f"{WORK_DIR}/_prs_work_{trait}/r0_s{step}.meta.txt", sep="\t", usecols=["ID", "P"])
    significant = table.loc[table["P"] < THRESHOLD, "ID"]
    return {f'{v.split(":")[0]}:{int(v.split(":")[1]) // BLOCK}' for v in significant}


train_n = (pd.read_csv(ACCRETION).groupby("n_sites")["train_individuals"].mean().round().astype(int))
trait_angle = {t: 2 * np.pi * i / 6 + np.pi / 6 for i, t in enumerate(HERITABILITY_ORDER)}
trait_xy = {t: (0.56 * np.cos(a), 0.56 * np.sin(a)) for t, a in trait_angle.items()}

fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.7))
fig.subplots_adjust(left=0.01, right=0.99, top=0.846, bottom=0.05, wspace=0.10, hspace=0.32)
for ax, step in zip(axes.ravel(), STEPS):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22)
    ring = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(ring), np.sin(ring), color="#e4e4e4", lw=0.7, zorder=0)
    for chromosome in sorted(offset):                       # chromosome boundaries
        a = 2 * np.pi * offset[chromosome] / genome
        ax.plot([0.97 * np.cos(a), 1.04 * np.cos(a)], [0.97 * np.sin(a), 1.04 * np.sin(a)],
                color="#c8c8c8", lw=0.5, zorder=1)
    total = 0
    for trait in HERITABILITY_ORDER:
        blocks = loci_at(trait, step)
        total += len(blocks)
        tx, ty = trait_xy[trait]
        for block in blocks:
            a = angle_of(block)
            ax.plot([np.cos(a), tx], [np.sin(a), ty], color=COLOUR[trait], lw=0.52, alpha=0.22, zorder=2)
            ax.plot([np.cos(a)], [np.sin(a)], ".", ms=1.6, color=COLOUR[trait], alpha=0.55, zorder=3)
        radius = 2.6 + 0.85 * np.sqrt(len(blocks))
        ax.plot([tx], [ty], "o", ms=radius, color=COLOUR[trait],
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        direction = trait_angle[trait]                      # push the label straight out from the centre
        dx, dy = np.cos(direction), np.sin(direction)
        # edges radiate across the inside of the ring, so the label carries its own background
        ax.annotate(NAMES[trait], (tx, ty), xytext=((radius * 0.5 + 7) * dx, (radius * 0.5 + 7) * dy),
                    textcoords="offset points", fontsize=5.8, color="#222222", zorder=6,
                    ha="left" if dx > 0.25 else "right" if dx < -0.25 else "center",
                    va="bottom" if dy > 0.25 else "top" if dy < -0.25 else "center",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.75))
    ax.set_title(f"{step} site{'s' if step > 1 else ''}   (N = {train_n[step]:,})",
                 fontsize=8, fontweight="normal", color="#111111", pad=14)
    ax.annotate(f"{total} loci", (0.5, 1.005), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=7, color="#888888")

fig.suptitle("Locus-trait associations by genome position at each meta-analysis step",
             fontsize=9, fontweight="bold", color="#111111", x=0.5, y=0.955)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
PNG = os.path.join(os.path.dirname(OUTPUT), "png",
                   os.path.basename(OUTPUT).replace(".pdf", ".png"))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
for path, dots in [(OUTPUT, None), (PNG, 300)]:
    fig.savefig(path, dpi=dots, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"[plot_metagraph_grid] Saved {OUTPUT}")
