# Reads:  data/edges.csv (merged gene-phenotype pairs with their supporting sources)
# Writes: results/predictions.csv (top-K predicted phenotypes per gene; held_out = hidden real edge)
# Does:   masks a random share of edges, runs SVD collaborative filtering, reports recall overall and per source

import os
import sys
import numpy as np
import pandas as pd
from cf import mask_edges, evaluate, bootstrap_ci

INPUT, OUTPUT = sys.argv[1], sys.argv[2]
MASK_FRACTION, N_COMPONENTS, TOP_K = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
SEED, N_BOOT = int(sys.argv[6]), int(sys.argv[7])

rng = np.random.default_rng(SEED)

print(f"[collab_filter] Loading {INPUT} ...")
edges = pd.read_csv(INPUT, keep_default_na=False)
train, test = mask_edges(edges, MASK_FRACTION, rng)
print(f"[collab_filter] Masked {len(test):,} of {len(edges):,} edges ({MASK_FRACTION:.0%})")

stats, hits, predictions = evaluate(train, test, N_COMPONENTS, TOP_K, SEED, keep_predictions=True,
                                    log=lambda msg: print(f"[collab_filter] {msg}"))

print("\n[collab_filter] Results")
for key, value in stats.items():
    print(f"[collab_filter]   {key:<28} {value:,.4f}" if isinstance(value, float) else
          f"[collab_filter]   {key:<28} {value:,}")

# Recall per supporting source (a pair supported by several sources counts for each of them)
print(f"\n[collab_filter] Recall@{TOP_K} by source [95% bootstrap CI]")
for source in sorted(set(";".join(hits["sources"]).split(";"))):
    h = hits.loc[hits["sources"].str.split(";").map(lambda s: source in s), "hit_svd"].to_numpy()
    lo, hi = bootstrap_ci(h, rng, N_BOOT)
    print(f"[collab_filter]   {source:<14} {h.mean():.1%} [{lo:.1%}, {hi:.1%}]  ({len(h):,} test edges)")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
predictions.to_csv(OUTPUT, index=False)
print(f"\n[collab_filter] Saved {len(predictions):,} predictions to {OUTPUT}")
