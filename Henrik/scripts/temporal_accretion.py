# Reads:  data/edges_gwas.csv (GWAS edges dated by first report), data/gwas_catalog.tsv (publication authors)
# Writes: results/temporal_years.csv, results/temporal_top_studies.csv, results/temporal_excluding_top_studies.csv
# Does:   replays history: train on pairs first reported before year Y, test on new pairs reported in Y

# Test set for year Y = pairs whose FIRST report is in Y and whose gene and phenotype were both seen before Y.
# A pair reported in 2012 and again in 2018 is only ever a 2012 edge. Recall is measured at a fixed K and at
# K = K_FRACTION of the phenotypes known before Y. "Study" = publication (PubMed ID): biobank papers bundle
# thousands of trait analyses. For EXCLUSION_YEARS, the year is rerun without its top publications.

import os
import sys
import pandas as pd
from cf import evaluate

EDGES, CATALOG = sys.argv[1], sys.argv[2]
OUT_YEARS, OUT_STUDIES, OUT_EXCLUDED = sys.argv[3], sys.argv[4], sys.argv[5]
FIRST_YEAR, LAST_YEAR = int(sys.argv[6]), int(sys.argv[7])
N_COMPONENTS, TOP_K, K_FRACTION, SVD_SEED = int(sys.argv[8]), int(sys.argv[9]), float(sys.argv[10]), int(sys.argv[11])
N_TOP_STUDIES = int(sys.argv[12])
EXCLUSION_YEARS = [int(y) for y in sys.argv[13].split(",")]


def split_year(edges, year, excluded_pmids=frozenset()):
    """(train, new pairs in year, test = new pairs between a gene and phenotype already in train).
    Pairs whose first-year reports all come from excluded publications are not new in that year."""
    train = edges[edges["year"] < year]
    new = edges[edges["year"] == year]
    if excluded_pmids:
        new = new[new["first_pmids"].str.split(";").map(lambda pmids: bool(set(pmids) - excluded_pmids))]
    known_nodes = new["gene"].isin(set(train["gene"])) & new["phenotype"].isin(set(train["phenotype"]))
    return train, new, new[known_nodes]


def recall_at_both_k(train, test):
    """Recall and popularity baseline at fixed TOP_K and at K = K_FRACTION of known phenotypes."""
    k_fraction = max(1, round(K_FRACTION * train["phenotype"].nunique()))
    row = {"k_fraction": k_fraction}
    for label, k in [("k_fixed", TOP_K), ("k_fraction", k_fraction)]:
        stats, _, _ = evaluate(train, test, N_COMPONENTS, k, SVD_SEED, log=lambda msg: None)
        row[f"recall_{label}"] = stats[f"recall@{k}"]
        row[f"popularity_{label}"] = stats[f"recall@{k}_popularity"]
    row.update(train_edges=stats["train_edges"], genes=stats["genes"], phenotypes=stats["phenotypes"],
               test_edges=stats["test_edges"])
    return row


def top_studies(new, authors, year):
    """The N_TOP_STUDIES publications that first reported the most of this year's new pairs."""
    counts = new["first_pmids"].str.split(";").explode().value_counts().head(N_TOP_STUDIES)
    return pd.DataFrame({
        "year": year, "rank": range(1, len(counts) + 1), "pmid": counts.index,
        "first_author": counts.index.map(authors), "new_pairs": counts.to_numpy(),
        "share_of_new": counts.to_numpy() / len(new),
    })


def replay_year(edges, authors, year):
    """Year row, top-studies table, and (for EXCLUSION_YEARS) the rerun without those studies."""
    train, new, test = split_year(edges, year)
    row = {"year": year, "new_pairs": len(new), "test_share_of_new": len(test) / len(new),
           **recall_at_both_k(train, test)}
    studies = top_studies(new, authors, year)
    print(f"[temporal] {year}: train {row['train_edges']:,}  test {row['test_edges']:,}  "
          f"recall K={TOP_K} {row['recall_k_fixed']:.1%}  K={row['k_fraction']} {row['recall_k_fraction']:.1%}  "
          f"top {N_TOP_STUDIES} publications = {studies['share_of_new'].sum():.0%} of new pairs")
    excluded = None
    if year in EXCLUSION_YEARS:
        _, new_x, test_x = split_year(edges, year, frozenset(studies["pmid"]))
        excluded = {"year": year, "excluded_pmids": ";".join(studies["pmid"]), "new_pairs": len(new_x),
                    **recall_at_both_k(train, test_x)}
    return row, studies, excluded


def print_tables(years, studies, excluded):
    pct = "{:.1%}".format
    show = years[["year", "train_edges", "test_edges", "k_fraction", "recall_k_fixed", "popularity_k_fixed",
                  "recall_k_fraction", "popularity_k_fraction"]].rename(columns={
        "k_fraction": "K (1%)", "recall_k_fixed": f"SVD K={TOP_K}", "popularity_k_fixed": f"pop K={TOP_K}",
        "recall_k_fraction": "SVD K=1%", "popularity_k_fraction": "pop K=1%"})
    print("\n[temporal] Recall on each year's new pairs between known genes and phenotypes")
    print(show.to_string(index=False, formatters={c: pct for c in show.columns if "SVD" in c or "pop" in c}))
    print(f"\n[temporal] Top {N_TOP_STUDIES} publications by new pairs, exclusion years")
    print(studies[studies["year"].isin(EXCLUSION_YEARS)].to_string(index=False, formatters={"share_of_new": pct}))
    both = years.merge(excluded, on="year", suffixes=("", "_excl"))
    cols = ["test_edges", "recall_k_fixed", "popularity_k_fixed", "recall_k_fraction", "popularity_k_fraction"]
    print(f"\n[temporal] All publications vs without that year's top {N_TOP_STUDIES}")
    print(both[["year"] + [c for col in cols for c in (col, f"{col}_excl")]].to_string(
        index=False, formatters={c: pct for c in both.columns if "recall" in c or "popularity" in c}))


edges = pd.read_csv(EDGES, keep_default_na=False, dtype={"first_pmids": str})
authors = (pd.read_csv(CATALOG, sep="\t", usecols=["PUBMEDID", "FIRST AUTHOR"], dtype=str)
           .drop_duplicates("PUBMEDID").set_index("PUBMEDID")["FIRST AUTHOR"])
results = [replay_year(edges, authors, year) for year in range(FIRST_YEAR, LAST_YEAR + 1)]
years = pd.DataFrame([r[0] for r in results])
studies = pd.concat([r[1] for r in results], ignore_index=True)
excluded = pd.DataFrame([r[2] for r in results if r[2] is not None])

for path, table in [(OUT_YEARS, years), (OUT_STUDIES, studies), (OUT_EXCLUDED, excluded)]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
print_tables(years, studies, excluded)
print(f"\n[temporal] Saved {OUT_YEARS}, {OUT_STUDIES}, {OUT_EXCLUDED}")
