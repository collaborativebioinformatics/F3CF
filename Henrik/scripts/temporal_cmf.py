# Reads:  data/edges_gwas.csv (GWAS edges dated by first report), data/edges_gene_gene.csv (STRING)
# Writes: results/temporal_cmf.csv (recall per year x method x gene-degree group)
# Does:   temporal replay comparing SVD, CMF without STRING and CMF with STRING at K = 1% of known phenotypes

# Same yearly splits as temporal_accretion.py. Gene degree = number of phenotypes the gene has in training
# (pairs first reported before the year). Groups: under DEGREE_LOW, DEGREE_LOW to DEGREE_HIGH, over DEGREE_HIGH.

import os
import sys
import numpy as np
import pandas as pd
from cf import evaluate
from cmf import cmf_factorizer
from replay import split_year

EDGES, GENE_GENE, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
FIRST_YEAR, LAST_YEAR = int(sys.argv[4]), int(sys.argv[5])
N_COMPONENTS, K_FRACTION, SEED = int(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])
ALPHA, LAMBDA, N_ITER = float(sys.argv[9]), float(sys.argv[10]), int(sys.argv[11])
DEGREE_LOW, DEGREE_HIGH = int(sys.argv[12]), int(sys.argv[13])


def degree_group(degree):
    return np.select([degree < DEGREE_LOW, degree > DEGREE_HIGH],
                     [f"<{DEGREE_LOW}", f">{DEGREE_HIGH}"], f"{DEGREE_LOW}-{DEGREE_HIGH}")


def recall_rows(year, k, method, hits, hit_col):
    """Recall overall and per gene-degree group; hits and n_test kept so groups can be pooled across years."""
    rows = []
    for group, part in [("all", hits)] + list(hits.groupby("degree_group")):
        rows.append({"year": year, "k": k, "method": method, "degree_group": group,
                     "n_test": len(part), "hits": int(part[hit_col].sum()), "recall": part[hit_col].mean()})
    return rows


def replay_year(edges, gene_gene, year):
    """All methods on one year's split; popularity baseline taken from the SVD run (same ranking rule)."""
    train, _, test = split_year(edges, year)
    k = max(1, round(K_FRACTION * train["phenotype"].nunique()))
    degree = train.groupby("gene")["phenotype"].nunique()
    methods = [("SVD", None),
               ("CMF", cmf_factorizer(None, N_COMPONENTS, ALPHA, LAMBDA, N_ITER, SEED)),
               ("CMF + STRING", cmf_factorizer(gene_gene, N_COMPONENTS, ALPHA, LAMBDA, N_ITER, SEED))]
    rows = []
    for method, factorize in methods:
        _, hits, _ = evaluate(train, test, N_COMPONENTS, k, SEED, factorize=factorize, log=lambda msg: None)
        hits = hits.assign(degree_group=degree_group(hits["gene"].map(degree).to_numpy()))
        rows += recall_rows(year, k, method, hits, "hit_svd")
        if factorize is None:
            rows += recall_rows(year, k, "Popularity", hits, "hit_pop")
    overall = {r["method"]: r["recall"] for r in rows if r["degree_group"] == "all"}
    print(f"[temporal_cmf] {year}: K={k}  test {len(test):,}  " + "  ".join(f"{m} {v:.1%}" for m, v in overall.items()))
    return rows


def print_summary(table):
    pct = "{:.1%}".format
    wide = table[table["degree_group"] == "all"].pivot(index="year", columns="method", values="recall")
    print(f"\n[temporal_cmf] Recall at K = {K_FRACTION:.0%} of known phenotypes, all test edges")
    print(wide.to_string(formatters={c: pct for c in wide.columns}))
    for label, years in [(f"{FIRST_YEAR}-{LAST_YEAR}", table["year"] > 0), ("2019-2025", table["year"] >= 2019)]:
        pooled = table[years].groupby(["degree_group", "method"])[["hits", "n_test"]].sum()
        pooled = (pooled["hits"] / pooled["n_test"]).unstack("method")
        counts = table[years & (table["method"] == "SVD")].groupby("degree_group")["n_test"].sum()
        print(f"\n[temporal_cmf] Pooled over {label}, by gene degree in training (test edges per group: "
              + ", ".join(f"{g} {n:,}" for g, n in counts.items()) + ")")
        print(pooled.to_string(formatters={c: pct for c in pooled.columns}))


edges = pd.read_csv(EDGES, keep_default_na=False, dtype={"first_pmids": str})
gene_gene = pd.read_csv(GENE_GENE, keep_default_na=False)
table = pd.DataFrame([row for year in range(FIRST_YEAR, LAST_YEAR + 1) for row in replay_year(edges, gene_gene, year)])
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
table.to_csv(OUTPUT, index=False)
print_summary(table)
print(f"\n[temporal_cmf] Saved {OUTPUT}")
