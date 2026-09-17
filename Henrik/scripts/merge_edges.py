# Concatenate per-source edge lists, keep the strongest score per gene-phenotype pair
# Usage: python scripts/merge_edges.py data/edges.csv data/edges_gwas.csv [data/edges_other.csv ...]

import sys
import pandas as pd

OUTPUT = sys.argv[1]
INPUTS = sys.argv[2:]

frames = []
for path in INPUTS:
    # keep_default_na=False: never turn gene names like "NA" or "NULL" into missing values
    df = pd.read_csv(path, keep_default_na=False)
    print(f"[merge_edges] {path}: {len(df):,} edges")
    frames.append(df)

edges = pd.concat(frames, ignore_index=True)
print(f"[merge_edges] {len(edges):,} edges before dedup")

# Sort strongest first, then keep the first row per pair (so `source` is the source of the best score).
# Note: scores from different sources are on different scales; revisit when a second source is added.
edges = edges.sort_values("score", ascending=False)
edges = edges.drop_duplicates(["gene", "phenotype"], keep="first")
edges = edges.sort_values(["gene", "phenotype"])

print(f"[merge_edges] {len(edges):,} edges after dedup  "
      f"(genes: {edges['gene'].nunique():,}, phenotypes: {edges['phenotype'].nunique():,})")
print("[merge_edges] edges per source:")
print(edges["source"].value_counts().to_string())

edges.to_csv(OUTPUT, index=False)
print(f"[merge_edges] Saved {OUTPUT}")
