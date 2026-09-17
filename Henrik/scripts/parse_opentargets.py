# Reads:  data/opentargets/association_overall_direct/*.parquet and data/opentargets/target/*.parquet
# Writes: data/edges_opentargets.csv (gene, phenotype, score = overall association score, source)
# Does:   keeps associations with score >= threshold, maps Ensembl IDs to gene symbols

import sys
import glob
import pandas as pd

# Most OpenTargets associations are weak literature co-mentions (median score ~0.035), so MIN_SCORE
# keeps only reasonably supported ones so they don't swamp the GWAS edges.
INPUT_DIR, OUTPUT, MIN_SCORE = sys.argv[1], sys.argv[2], float(sys.argv[3])

# Load associations: one row per (target, disease), score in [0, 1]
files = sorted(glob.glob(f"{INPUT_DIR}/association_overall_direct/*.parquet"))
print(f"[parse_opentargets] Loading {len(files)} association files ...")
df = pd.concat(
    [pd.read_parquet(f, columns=["targetId", "diseaseId", "associationScore"]) for f in files],
    ignore_index=True,
)
print(f"[parse_opentargets]   {len(df):,} associations")

df = df.dropna()
df = df[df["associationScore"] >= MIN_SCORE]
print(f"[parse_opentargets]   {len(df):,} with score >= {MIN_SCORE}")

# Map Ensembl gene IDs to gene symbols, to match GWAS Catalog MAPPED_GENE
targets = pd.concat(
    [pd.read_parquet(f, columns=["id", "approvedSymbol"])
     for f in sorted(glob.glob(f"{INPUT_DIR}/target/*.parquet"))],
    ignore_index=True,
)
df = df.merge(targets, left_on="targetId", right_on="id", how="inner")
df = df[df["approvedSymbol"].notna() & (df["approvedSymbol"] != "")]
print(f"[parse_opentargets]   {len(df):,} after mapping Ensembl IDs to gene symbols")

# Two Ensembl IDs can share a symbol: keep the strongest score per gene-phenotype pair
edges = (
    df.rename(columns={"approvedSymbol": "gene", "diseaseId": "phenotype", "associationScore": "score"})
    .groupby(["gene", "phenotype"], as_index=False)["score"]
    .max()
)
edges["source"] = "opentargets"

print(f"[parse_opentargets]   genes: {edges['gene'].nunique():,}  "
      f"phenotypes: {edges['phenotype'].nunique():,}  edges: {len(edges):,}")

edges.to_csv(OUTPUT, index=False)
print(f"[parse_opentargets] Saved {OUTPUT}")
