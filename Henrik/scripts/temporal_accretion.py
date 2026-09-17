# Reads:  data/edges_gwas.csv (GWAS edges, each dated by the first study that reported the pair)
# Writes: results/temporal_accretion.csv (one row per year)
# Does:   replays history: train on pairs first reported before year Y, test on new pairs reported in Y

# Test set for year Y = pairs whose FIRST report is in Y and whose gene and phenotype were both already
# seen before Y (new edges between known nodes). A pair reported in 2012 and again in 2018 is only ever a
# 2012 edge, so it is training data from 2013 on and never a test edge. No random masking, no seeds;
# the SVD solver uses a fixed random_state so reruns give the same numbers.

import os
import sys
import pandas as pd
from cf import evaluate

INPUT, OUTPUT = sys.argv[1], sys.argv[2]
FIRST_YEAR, LAST_YEAR = int(sys.argv[3]), int(sys.argv[4])
N_COMPONENTS, TOP_K, SVD_SEED = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
R = f"recall@{TOP_K}"


def split_year(edges, year):
    """(train, all new pairs in year, test = new pairs between a gene and phenotype already in train)."""
    train = edges[edges["year"] < year]
    new = edges[edges["year"] == year]
    known_nodes = new["gene"].isin(set(train["gene"])) & new["phenotype"].isin(set(train["phenotype"]))
    return train, new, new[known_nodes]


edges = pd.read_csv(INPUT, keep_default_na=False)
print(f"[temporal] {len(edges):,} GWAS edges, first reported {edges['year'].min()}-{edges['year'].max()}")

rows = []
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    train, new, test = split_year(edges, year)
    print(f"\n[temporal] {year}: {len(new):,} new pairs, {len(test):,} between genes and phenotypes seen before {year}")
    stats, _, _ = evaluate(train, test, N_COMPONENTS, TOP_K, SVD_SEED, log=lambda msg: print(f"[temporal] {msg}"))
    rows.append({"year": year, "new_pairs": len(new), "test_share_of_new": len(test) / len(new), **stats})
    print(f"[temporal]   {R}: {stats[R]:.2%}   popularity: {stats[R + '_popularity']:.2%}")

table = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
table.to_csv(OUTPUT, index=False)

print(f"\n[temporal] Summary: train = pairs first reported before the year; test = new pairs that year "
      f"between known genes and phenotypes")
print(pd.DataFrame({
    "year": table["year"],
    "train edges": table["train_edges"].map("{:,}".format),
    "new pairs": table["new_pairs"].map("{:,}".format),
    "test edges": table["test_edges"].map("{:,}".format),
    "test / new": table["test_share_of_new"].map("{:.0%}".format),
    R: table[R].map("{:.1%}".format),
    "popularity": table[f"{R}_popularity"].map("{:.1%}".format),
}).to_string(index=False))
print(f"\n[temporal] Saved {OUTPUT}")
