# GWAS Catalog TSV -> gene-phenotype edge list
# Usage: python scripts/parse_gwas.py data/gwas_catalog.tsv data/edges_gwas.csv

import sys
import pandas as pd

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

# Load
print(f"[parse_gwas] Loading {INPUT} ...")
df = pd.read_csv(INPUT, sep="\t", low_memory=False)
print(f"[parse_gwas]   {len(df):,} associations, {df.shape[1]} columns")

# Keep only the columns we need, drop rows with missing values
df = df[["MAPPED_GENE", "MAPPED_TRAIT_URI", "PVALUE_MLOG"]]
df = df.dropna()
df["PVALUE_MLOG"] = pd.to_numeric(df["PVALUE_MLOG"], errors="coerce")
df = df.dropna()
print(f"[parse_gwas]   {len(df):,} rows after dropping missing values")

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
df["phenotype"] = df["phenotype"].str.strip()

# "NA - GENE" means no gene on that side of the SNP, so "NA" is a placeholder, not a gene
df = df[(df["gene"] != "") & (df["gene"] != "NA") & (df["phenotype"] != "")]
print(f"[parse_gwas]   {len(df):,} gene-phenotype rows")

# Same gene-phenotype pair is often reported by many studies: keep the strongest p-value
edges = (
    df.groupby(["gene", "phenotype"], as_index=False)["PVALUE_MLOG"]
    .max()
    .rename(columns={"PVALUE_MLOG": "score"})
)
edges["source"] = "gwas_catalog"

print(f"[parse_gwas]   genes: {edges['gene'].nunique():,}  "
      f"phenotypes: {edges['phenotype'].nunique():,}  edges: {len(edges):,}")

edges.to_csv(OUTPUT, index=False)
print(f"[parse_gwas] Saved {OUTPUT}")
