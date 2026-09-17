# Reads:  data/gwas_catalog.tsv (GWAS Catalog associations with ontology annotations)
# Writes: data/edges_gwas.csv (snp, target, target_type = gene | phenotype, date and year of first report)
# Does:   turns genome-wide significant single-SNP associations into SNP-gene and SNP-phenotype edges

import math
import sys
import pandas as pd

INPUT, OUTPUT, P_THRESHOLD = sys.argv[1], sys.argv[2], float(sys.argv[3])

print(f"[parse_gwas] Loading {INPUT} ...")
df = pd.read_csv(INPUT, sep="\t", low_memory=False,
                 usecols=["SNPS", "MAPPED_GENE", "MAPPED_TRAIT_URI", "PVALUE_MLOG", "DATE"])
print(f"[parse_gwas]   {len(df):,} associations")

df["PVALUE_MLOG"] = pd.to_numeric(df["PVALUE_MLOG"], errors="coerce")
df = df[df["PVALUE_MLOG"] > -math.log10(P_THRESHOLD)]
print(f"[parse_gwas]   {len(df):,} with p < {P_THRESHOLD:g}")

# One rs ID per row. Rows listing several SNPs (haplotypes, SNP-SNP interactions) can't be paired with
# their genes, and positional IDs like "chr12:111446804" are not rs IDs: both are dropped.
rs_ids = df["SNPS"].str.findall(r"rs\d+")
df = df[rs_ids.str.len() == 1].assign(snp=rs_ids.str[0])
print(f"[parse_gwas]   {len(df):,} with exactly one rs ID")

# SNP-gene edges. MAPPED_GENE uses "A, B" (several genes) and "A - B" (SNP between two genes);
# "NA - GENE" means no gene on that side, so "NA" is not a gene.
genes = df[["snp", "DATE"]].assign(target=df["MAPPED_GENE"].str.split(r",|;| - | x ", regex=True))
genes = genes.explode("target").dropna(subset=["target"])
genes = genes.assign(target=genes["target"].str.strip(), target_type="gene")
genes = genes[(genes["target"] != "") & (genes["target"] != "NA")]

# SNP-phenotype edges, as short ontology IDs ("http://www.ebi.ac.uk/efo/EFO_0000180" -> "EFO_0000180")
phenos = df[["snp", "DATE"]].assign(target=df["MAPPED_TRAIT_URI"].str.split(","))
phenos = phenos.explode("target").dropna(subset=["target"])
phenos = phenos.assign(target=phenos["target"].str.strip().str.split("/").str[-1], target_type="phenotype")
phenos = phenos[phenos["target"] != ""]

# Date each edge by its first report (an edge found in 2012 and again in 2018 is a 2012 edge)
edges = (pd.concat([genes, phenos])
         .groupby(["snp", "target", "target_type"], as_index=False)["DATE"].min()
         .rename(columns={"DATE": "date"}))
edges["year"] = edges["date"].str[:4].astype(int)

for target_type, part in edges.groupby("target_type"):
    print(f"[parse_gwas]   SNP-{target_type} edges: {len(part):,}  "
          f"(SNPs {part['snp'].nunique():,}, {target_type}s {part['target'].nunique():,})")
edges.to_csv(OUTPUT, index=False)
print(f"[parse_gwas] Saved {OUTPUT}")
