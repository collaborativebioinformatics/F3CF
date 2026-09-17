# Reads:  data/edges_{source}.csv for every gene-phenotype source
# Writes: data/edges.csv (gene, phenotype, sources, n_sources)
# Does:   merges sources into one row per gene-phenotype pair, recording which sources support it

# Scores are NOT merged: each source's score is on its own scale (GWAS -log10 p, OpenTargets 0-1,
# ClinVar variant counts), so comparing them would be meaningless. The merged file records which
# sources support each pair; per-source scores stay in data/edges_{source}.csv.

import sys
import pandas as pd

OUTPUT, INPUTS = sys.argv[1], sys.argv[2:]

frames = []
for path in INPUTS:
    # keep_default_na=False: never turn gene names like "NA" or "NULL" into missing values
    df = pd.read_csv(path, keep_default_na=False)
    print(f"[merge_edges] {path}: {len(df):,} edges")
    frames.append(df[["gene", "phenotype", "source"]])

edges = pd.concat(frames, ignore_index=True).drop_duplicates()
print(f"[merge_edges] {len(edges):,} edges before merging pairs")

edges = (
    edges.groupby(["gene", "phenotype"], as_index=False)["source"]
    .agg(lambda s: ";".join(sorted(s)))
    .rename(columns={"source": "sources"})
)
edges["n_sources"] = edges["sources"].str.count(";") + 1

print(f"[merge_edges] {len(edges):,} unique pairs  "
      f"(genes: {edges['gene'].nunique():,}, phenotypes: {edges['phenotype'].nunique():,})")
print("[merge_edges] pairs by supporting sources:")
print(edges["sources"].value_counts().to_string())

edges.to_csv(OUTPUT, index=False)
print(f"[merge_edges] Saved {OUTPUT}")
