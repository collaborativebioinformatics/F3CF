# Reads:  data/clinvar_variant_summary.txt.gz (ClinVar variant summary)
# Writes: data/edges_clinvar.csv (gene, phenotype, score = number of distinct qualifying variants, source)
# Does:   edge = >=1 reviewed, germline, pathogenic/likely pathogenic single-gene variant for that condition

import re
import sys
import pandas as pd

INPUT, OUTPUT = sys.argv[1], sys.argv[2]

# Review statuses with actual assertion criteria and no conflicts
TRUSTED_REVIEW = {
    "criteria provided, single submitter",
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}
# Ontologies that match the IDs used by GWAS Catalog / OpenTargets, in order of preference
ID_PREFERENCE = ["MONDO", "Orphanet", "Human Phenotype Ontology", "EFO"]


def pick_id(condition):
    """Pick one ontology ID per condition, in the short format the other sources use."""
    refs = {}
    for ref in condition.split(","):
        prefix, _, value = ref.partition(":")
        refs.setdefault(prefix, value)
    for prefix in ID_PREFERENCE:
        if prefix in refs:
            value = refs[prefix]
            if prefix == "Orphanet":
                return f"Orphanet_{value}"          # "306511" -> Orphanet_306511
            # "MONDO:0013342", "HP:0000992", " The Experimental Factor Ontology:EFO_0004269"
            match = re.search(r"(MONDO|HP|EFO)[:_](\d+)$", value)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
    return ""  # only MedGen / OMIM / MeSH: no matching ontology ID


print(f"[parse_clinvar] Loading {INPUT} ...")
df = pd.read_csv(
    INPUT, sep="\t", low_memory=False, keep_default_na=False,
    usecols=["VariationID", "GeneSymbol", "ClinSigSimple", "ReviewStatus", "OriginSimple",
             "Assembly", "PhenotypeIDS"],
)
print(f"[parse_clinvar]   {len(df):,} rows")

# Each variant is listed once per genome assembly: keep one
df = df[df["Assembly"] == "GRCh38"]
print(f"[parse_clinvar]   {len(df):,} GRCh38 rows")

df = df[df["ClinSigSimple"] == 1]
print(f"[parse_clinvar]   {len(df):,} pathogenic / likely pathogenic")

df = df[df["ReviewStatus"].isin(TRUSTED_REVIEW)]
print(f"[parse_clinvar]   {len(df):,} with review criteria and no conflicts")

df = df[df["OriginSimple"].isin(["germline", "germline/somatic"])]
print(f"[parse_clinvar]   {len(df):,} germline")

# Variants spanning several genes (e.g. large deletions, "GENE1;GENE2") don't say which gene matters.
# GeneSymbol can also be free text such as "covers 10 genes, none of which curated to show dosage
# sensitivity": real gene symbols never contain whitespace, so drop anything that does.
df = df[(df["GeneSymbol"] != "") & (df["GeneSymbol"] != "-") & ~df["GeneSymbol"].str.contains(";")
        & ~df["GeneSymbol"].str.contains(r"\s", regex=True)]
print(f"[parse_clinvar]   {len(df):,} in a single gene")

# PhenotypeIDS: conditions separated by "|", cross-references for one condition separated by ","
# e.g. "MONDO:MONDO:0013342,MedGen:C3150901,OMIM:613647,Orphanet:306511|MedGen:C3661900"
df["condition"] = df["PhenotypeIDS"].str.split("|")
df = df.explode("condition")

print("[parse_clinvar] Picking ontology IDs ...")
df["phenotype"] = df["condition"].map(pick_id)
df = df[df["phenotype"] != ""]
print(f"[parse_clinvar]   {len(df):,} variant-condition rows with a MONDO/Orphanet/HP/EFO ID")

edges = (
    df.rename(columns={"GeneSymbol": "gene"})
    .groupby(["gene", "phenotype"], as_index=False)["VariationID"]
    .nunique()
    .rename(columns={"VariationID": "score"})
)
edges["source"] = "clinvar"

print(f"[parse_clinvar]   genes: {edges['gene'].nunique():,}  "
      f"phenotypes: {edges['phenotype'].nunique():,}  edges: {len(edges):,}")
print("[parse_clinvar]   phenotype ID types: "
      + ", ".join(f"{k} {v:,}" for k, v in edges["phenotype"].str.split("_").str[0].value_counts().items()))

edges.to_csv(OUTPUT, index=False)
print(f"[parse_clinvar] Saved {OUTPUT}")
