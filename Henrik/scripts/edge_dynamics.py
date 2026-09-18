# Reads:  results/_prs_work_pheno*/r*_s*.meta.txt (federated meta-analysis at every repeat x accretion step)
# Writes: results/edge_dynamics.csv, results/edges_step.csv
# Does:   turns each accretion step into an edge set (variant significant for a trait) and counts how that
#         edge set changes between consecutive steps: edges that appear, disappear, or persist.

# An edge is a variant-trait pair passing 5e-8 in the meta-analysis at that step. "Locus" collapses variants
# into 250kb blocks so that one signal in linkage disequilibrium is not counted many times. Whether a
# disappearing edge was actually wrong needs a truth standard; the six-site meta-analysis is the best estimate
# available, so an edge is called surviving if it is still significant at step 6. That is a replication
# standard, not ground truth - it is validated against the real causal variants for Height in validate_truth().

import glob
import os
import re
import sys
import numpy as np
import pandas as pd

WORK_GLOB, OUTPUT_DIR = sys.argv[1], sys.argv[2]
THRESHOLD, BLOCK = 5e-8, 250_000
STEPS = [1, 2, 3, 4, 5, 6]


def locus(variant):
    chromosome, position = variant.split(":")[0], int(variant.split(":")[1])
    return f"{chromosome}:{position // BLOCK}"


def hits(trait, repeat, step):
    path = f"results/_prs_work_{trait}/r{repeat}_s{step}.meta.txt"
    table = pd.read_csv(path, sep="\t", usecols=["ID", "P"])
    return set(table.loc[table["P"] < THRESHOLD, "ID"])


traits = sorted({re.search(r"_prs_work_(pheno\d+)", p).group(1) for p in glob.glob(WORK_GLOB)},
                key=lambda t: int(t.replace("pheno", "")))
repeats = sorted({int(re.search(r"/r(\d+)_s", p).group(1)) for p in glob.glob(f"{WORK_GLOB}/r*_s*.meta.txt")})
print(f"[edge_dynamics] {len(traits)} traits, {len(repeats)} repeats, {len(STEPS)} steps")

rows, sizes = [], []
for trait in traits:
    for repeat in repeats:
        edges = {step: hits(trait, repeat, step) for step in STEPS}
        final = edges[STEPS[-1]]
        for step in STEPS:
            sizes.append({"trait": trait, "repeat": repeat, "step": step,
                          "variants": len(edges[step]), "loci": len({locus(v) for v in edges[step]}),
                          "survive_to_final": len(edges[step] & final)})
        for previous, current in zip(STEPS, STEPS[1:]):
            before, after = edges[previous], edges[current]
            rows.append({"trait": trait, "repeat": repeat, "from_step": previous, "to_step": current,
                         "appear": len(after - before), "disappear": len(before - after),
                         "persist": len(before & after),
                         "disappear_absent_at_final": len((before - after) - final)})
    print(f"[edge_dynamics]   {trait} done")

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/edge_dynamics.csv", index=False)
pd.DataFrame(sizes).to_csv(f"{OUTPUT_DIR}/edges_step.csv", index=False)
print(f"[edge_dynamics] Saved {OUTPUT_DIR}/edge_dynamics.csv and {OUTPUT_DIR}/edges_step.csv")

summary = (pd.DataFrame(rows).groupby(["trait", "to_step"])[["appear", "disappear", "persist"]]
           .mean().round(1))
print()
print(summary.to_string())
