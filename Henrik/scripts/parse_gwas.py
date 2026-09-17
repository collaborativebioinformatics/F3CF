# Reads:  data/gwas_catalog.tsv (GWAS Catalog associations with ontology annotations)
# Writes: data/edges_gwas.csv (gene, phenotype, score = strongest -log10 p, source, date/year = first report)
# Does:   keeps associations with p < threshold, splits multi-gene / multi-trait rows, one edge per pair

import math
import sys
import pandas as pd

INPUT, OUTPUT, P_THRESHOLD = sys.argv[1], sys.argv[2], float(sys.argv[3])

# Load
print(f"[parse_gwas] Loading {INPUT} ...")
df = pd.read_csv(INPUT, sep="\t", low_memory=False)
print(f"[parse_gwas]   {len(df):,} associations, {df.shape[1]} columns")

# Keep only the columns we need, drop rows with missing values
df = df[["MAPPED_GENE", "MAPPED_TRAIT_URI", "PVALUE_MLOG", "DATE"]]  # DATE = study publication date
df = df.dropna()
df["PVALUE_MLOG"] = pd.to_numeric(df["PVALUE_MLOG"], errors="coerce")
df = df.dropna()
print(f"[parse_gwas]   {len(df):,} rows after dropping missing values")

# The Catalog lists hits down to p < 1e-5; keep only p < P_THRESHOLD (5e-8 = genome-wide significance)
df = df[df["PVALUE_MLOG"] > -math.log10(P_THRESHOLD)]
print(f"[parse_gwas]   {len(df):,} rows with p < {P_THRESHOLD:g}")

# Split multi-gene rows into one row per gene.
# MAPPED_GENE uses "A, B" (several genes), "A - B" (SNP between two genes),
# "A; B" and "A x B" (SNP-SNP interactions). All mean "these genes", so split on all of them.
print("[parse_gwas] Splitting multi-gene rows ...")
df["gene"] = df["MAPPED_GENE"].str.split(r",|;| - | x ", regex=True)
df = df.explode("gene")
df["gene"] = df["gene"].str.strip()

# A row can also map to several traits ("uri1, uri2"), so split those too
df["phenotype"] = df["MAPPED_TRAIT_URI"].str.split(",")
df = df.explode("phenotype")
# Keep the short ontology ID ("http://www.ebi.ac.uk/efo/EFO_0000180" -> "EFO_0000180"),
# the format OpenTargets and most other sources use
df["phenotype"] = df["phenotype"].str.strip().str.split("/").str[-1]

# "NA - GENE" means no gene on that side of the SNP, so "NA" is a placeholder, not a gene
df = df[(df["gene"] != "") & (df["gene"] != "NA") & (df["phenotype"] != "")]
print(f"[parse_gwas]   {len(df):,} gene-phenotype rows")

# Same gene-phenotype pair is often reported by many studies: keep the strongest p-value, and date the
# pair by its first report (a pair found in 2012 and again in 2018 is a 2012 edge)
edges = (
    df.groupby(["gene", "phenotype"], as_index=False)
    .agg(score=("PVALUE_MLOG", "max"), date=("DATE", "min"))
)
edges.insert(3, "source", "gwas_catalog")
edges["year"] = edges["date"].str[:4].astype(int)

print(f"[parse_gwas]   genes: {edges['gene'].nunique():,}  "
      f"phenotypes: {edges['phenotype'].nunique():,}  edges: {len(edges):,}")

edges.to_csv(OUTPUT, index=False)
print(f"[parse_gwas] Saved {OUTPUT}")
