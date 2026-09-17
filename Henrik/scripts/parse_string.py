# Reads:  data/string/9606.protein.links.v12.0.txt.gz and 9606.protein.info.v12.0.txt.gz
# Writes: data/edges_gene_gene.csv (gene_a < gene_b, score = STRING combined score 0-1000)
# Does:   keeps links with combined score > min_score, maps proteins to gene symbols, one row per gene pair

import sys
import numpy as np
import pandas as pd

LINKS, INFO, OUTPUT, MIN_SCORE = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])  # 700 = "high confidence"

print(f"[parse_string] Loading {LINKS} ...")
links = pd.read_csv(LINKS, sep=" ")
print(f"[parse_string]   {len(links):,} protein links (each pair listed in both directions)")

links = links[links["combined_score"] > MIN_SCORE]
print(f"[parse_string]   {len(links):,} with combined score > {MIN_SCORE}")

# Map STRING protein IDs (9606.ENSP...) to gene symbols with STRING's own protein info table
info = pd.read_csv(INFO, sep="\t", usecols=["#string_protein_id", "preferred_name"])
symbol = dict(zip(info["#string_protein_id"], info["preferred_name"]))
links["gene_a"] = links["protein1"].map(symbol)
links["gene_b"] = links["protein2"].map(symbol)
unmapped = links["gene_a"].isna() | links["gene_b"].isna()
print(f"[parse_string]   {unmapped.sum():,} links with an unmapped protein (dropped)")
links = links[~unmapped]

# Several proteins can map to one symbol: drop self-pairs, keep one row per unordered pair (max score)
links = links[links["gene_a"] != links["gene_b"]]
a = np.minimum(links["gene_a"], links["gene_b"])
b = np.maximum(links["gene_a"], links["gene_b"])
edges = (
    pd.DataFrame({"gene_a": a, "gene_b": b, "score": links["combined_score"]})
    .groupby(["gene_a", "gene_b"], as_index=False)["score"]
    .max()
)

degree = pd.concat([edges["gene_a"], edges["gene_b"]]).value_counts()
print(f"[parse_string]   genes: {len(degree):,}  gene-gene edges: {len(edges):,}  "
      f"neighbors per gene: median {degree.median():.0f}, max {degree.max():,} ({degree.idxmax()})")

edges.to_csv(OUTPUT, index=False)
print(f"[parse_string] Saved {OUTPUT}")
