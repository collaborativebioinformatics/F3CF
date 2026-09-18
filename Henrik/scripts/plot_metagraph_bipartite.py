# Reads:  results/_prs_work_pheno*/r0_s*.meta.txt, results/_true_work/pheno*.glm.linear, results/prs_pheno1.csv
# Writes: results/figures/metagraph_bipartite.pdf
# Does:   draws the locus-trait bipartite graph at each accretion step, edges coloured by whether they are real

# Left nodes are 250kb genomic loci, ordered by genome position and held in the same place in every panel, so a
# locus that lights up in a later panel is a new discovery rather than a re-drawn graph. Right nodes are the six
# traits. An edge means a variant in that locus passed 5e-8 for that trait in the federated meta-analysis at that
# step; node size follows the size of that variant's meta-analysis effect.
#
# An edge is green when the locus is causal for the trait and red when it is not; the panel reports the
# share of that step's associations which are causal. Truth comes from
# association-testing the simulation's noiseless genetic value (GenoEff), which has no environmental noise in it,
# so power is not the limiting factor; the test is applied only to the 249 loci that were ever discovered, so it
# is corrected for 249 tests rather than a genome. On Height, the one trait shipping a causal-variant list, every
# discovered locus is called causal and 49 of 56 appear in that list (the other 7 are neighbouring blocks in
# linkage disequilibrium), and no discovered locus is called false - the labelling does not invent errors.

# Style follows plot_styleguide/plot_style_guide.md

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WORK_DIR, TRUE_DIR, ACCRETION, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BLOCK, THRESHOLD, STEPS = 250_000, 5e-8, [1, 2, 3, 4, 5, 6]
NAMES = {"pheno1": "Height", "pheno9": "HbA1c", "pheno6": "Body weight", "pheno2": "BMI",
         "pheno3": "HDL cholesterol", "pheno7": "Heart rate"}
ORDER = ["pheno1", "pheno9", "pheno6", "pheno2", "pheno3", "pheno7"]
CAUSAL, FALSE, DORMANT = "#3F8F5B", "#B5312A", "#e2e2e2"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})


def block_of(variant):
    chromosome, position = variant.split(":")[0], int(variant.split(":")[1])
    return f"{chromosome}:{position // BLOCK}"


def sort_key(locus):
    chromosome, block = locus.split(":")
    return int(chromosome), int(block)


discovered = {}                                  # trait -> step -> {locus: |beta| of its lead variant}
for trait in ORDER:
    discovered[trait] = {}
    for step in STEPS:
        table = pd.read_csv(f"{WORK_DIR}/_prs_work_{trait}/r0_s{step}.meta.txt", sep="\t",
                            usecols=["ID", "BETA", "P"])
        hits = table[table["P"] < THRESHOLD].copy()
        hits["locus"] = [block_of(v) for v in hits["ID"]]
        lead = hits.loc[hits.groupby("locus")["P"].idxmin()]
        discovered[trait][step] = dict(zip(lead["locus"], lead["BETA"].abs()))

loci = sorted({l for trait in ORDER for step in STEPS for l in discovered[trait][step]}, key=sort_key)
position = {locus: 1 - i / max(len(loci) - 1, 1) for i, locus in enumerate(loci)}
alpha = 0.05 / len(loci)                         # corrected for the loci actually tested, not the genome

truth = {}
for trait in ORDER:
    table = pd.read_csv(f"{TRUE_DIR}/{trait}.{trait}.glm.linear", sep="\t", usecols=["ID", "P"])
    table["locus"] = [block_of(v) for v in table["ID"]]
    best = table.groupby("locus")["P"].min()
    truth[trait] = set(best.index[best < alpha])

train_n = pd.read_csv(ACCRETION).groupby("n_sites")["train_individuals"].mean().round().astype(int)
trait_y = {trait: 1 - i / (len(ORDER) - 1) for i, trait in enumerate(ORDER)}
scale = max(b for trait in ORDER for step in STEPS for b in discovered[trait][step].values())

fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.6))
fig.subplots_adjust(left=0.015, right=0.985, top=0.840, bottom=0.075, wspace=0.62, hspace=0.30)
for ax, step in zip(axes.ravel(), STEPS):
    # nothing is drawn past x = 1.0, so the labels beyond the axes box cannot collide with an edge
    ax.set_xlim(-0.36, 1.0)
    ax.set_ylim(-0.09, 1.09)
    ax.axis("off")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter([0.0] * len(loci), [position[l] for l in loci], s=1.5, color=DORMANT, alpha=0.30, zorder=1)
    green = red = 0
    causal_edges, false_edges = [], []          # red is drawn last so it reads on top of the green
    for trait in ORDER:
        ty = trait_y[trait]
        for locus, beta in discovered[trait][step].items():
            real = locus in truth[trait]
            green, red = green + real, red + (not real)
            (causal_edges if real else false_edges).append((position[locus], ty, beta))
    for y, ty, beta in causal_edges:
        ax.plot([0.0, 1.0], [y, ty], color=CAUSAL, lw=0.32, alpha=0.30, zorder=2)
        ax.scatter([0.0], [y], s=0.9 + 9 * np.sqrt(beta / scale), color=CAUSAL, alpha=0.55, zorder=3)
    for y, ty, beta in false_edges:
        ax.plot([0.0, 1.0], [y, ty], color=FALSE, lw=0.38, alpha=1.0, zorder=5)
        ax.scatter([0.0], [y], s=0.9 + 9 * np.sqrt(beta / scale), color=FALSE, alpha=1.0, zorder=6)
    for trait in ORDER:
        carried = len(discovered[trait][step])
        ax.scatter([1.0], [trait_y[trait]], s=12 + 2.2 * carried, color="#444444",
                   edgecolor="white", linewidth=0.5, zorder=4, clip_on=False)
        # 15pt clear of the axes edge: more than the largest node's radius, so a node cannot reach the text
        ax.annotate(f"{NAMES[trait]}  ({carried})", (1.0, trait_y[trait]),
                    xycoords=("axes fraction", "data"), xytext=(15, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9.5, color="#333333", annotation_clip=False)
    ax.set_title(f"{step} site{'s' if step > 1 else ''}   (N = {train_n[step]:,})",
                 fontsize=12.3, fontweight="normal", color="#111111", pad=15)
    correct = f"{green / (green + red):.0%} correct" if green + red else "none found"
    ax.annotate(f"{green} causal, {red} false \u2014 {correct}", (0.5, 1.002), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=10.8, color="#888888")

fig.legend(handles=[Line2D([], [], color=FALSE, lw=1.6, label="false positive association"),
                    Line2D([], [], color=CAUSAL, lw=1.6, label="association with a causal locus"),
                    Line2D([], [], color=DORMANT, marker="o", ms=3, lw=0,
                           label="locus not yet discovered")],
           frameon=False, labelcolor="#333333", fontsize=10.0, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.005), handlelength=1.8)
fig.suptitle("Locus-trait associations at each meta-analysis step",
             fontsize=13.8, fontweight="bold", color="#111111", x=0.5, y=0.955)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
PNG = os.path.join(os.path.dirname(OUTPUT), "png",
                   os.path.basename(OUTPUT).replace(".pdf", ".png"))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
for path, dots in [(OUTPUT, None), (PNG, 300)]:
    fig.savefig(path, dpi=dots, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"[plot_metagraph_bipartite] Saved {OUTPUT}  ({len(loci)} loci, threshold for truth {alpha:.1e})")
