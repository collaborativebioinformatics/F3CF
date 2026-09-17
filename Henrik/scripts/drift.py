# Reads:  data/edges_gwas.csv (SNP-gene and SNP-phenotype edges dated by first report)
# Writes: results/drift.csv (one row per year), results/drift_entities.csv (drift per entity per year)
# Does:   replays GWAS history, factorizes the graph at each year, and measures how far embeddings move

# Graph at year Y = every edge first reported in or before Y. Model, over the entities in that graph:
#   SNP x gene A ~ S G^T,  SNP x phenotype B ~ S P^T,  shared SNP embeddings S
#   minimize ||A - S G^T||^2 + ||B - S P^T||^2 + lambda (||S||^2 + ||G||^2 + ||P||^2)   (absent edges count as 0)
# Fitted by alternating least squares. Year Y+1 starts from year Y's embeddings; entities appearing for the
# first time start from a fixed random draw. Drift in year Y = cosine distance between an entity's embedding
# after fitting Y-1 and after fitting Y, for entities present in both. Entities in small disconnected parts of
# the graph shrink toward a zero embedding (64 dimensions can't represent them); their direction is meaningless,
# so entities whose norm is below MIN_RELATIVE_NORM x the median norm of their type, in either year, are skipped.

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

EDGES, OUT_YEARS, OUT_ENTITIES = sys.argv[1], sys.argv[2], sys.argv[3]
FIRST_YEAR, LAST_YEAR = int(sys.argv[4]), int(sys.argv[5])
DIM, LAMBDA, N_ITER, SEED = int(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
MIN_RELATIVE_NORM = float(sys.argv[10])
TYPES = ["snp", "gene", "pheno"]


def solve(target, gram):
    """Least-squares embedding update: target @ (gram + lambda I)^-1."""
    return np.linalg.solve(gram + LAMBDA * np.eye(DIM), target.T).T


def fit(a, b, s, g, p):
    """Alternating least squares from the given starting embeddings."""
    for _ in range(N_ITER):
        s = solve(a @ g + b @ p, g.T @ g + p.T @ p)
        g = solve(a.T @ s, s.T @ s)
        p = solve(b.T @ s, s.T @ s)
    return s, g, p


def graph_until(edges, year):
    """IDs of the entities present by `year`, and the SNP x gene and SNP x phenotype matrices over them."""
    sub = edges[edges["year"] <= year]
    by_type = {"gene": sub[sub["target_type"] == "gene"], "pheno": sub[sub["target_type"] == "phenotype"]}
    present = {"snp": np.unique(sub["snp_id"]), **{t: np.unique(part["target_id"]) for t, part in by_type.items()}}
    matrices = []
    for t, part in by_type.items():
        rows = np.searchsorted(present["snp"], part["snp_id"])
        cols = np.searchsorted(present[t], part["target_id"])
        shape = (len(present["snp"]), len(present[t]))
        matrices.append(sp.csr_matrix((np.ones(len(part)), (rows, cols)), shape=shape))
    return present, matrices[0], matrices[1], len(sub)


def drift(before, after):
    """Cosine distance per row, and a mask of rows whose norm is large enough in both years to have a direction."""
    norm_before, norm_after = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    usable = (norm_before > MIN_RELATIVE_NORM * np.median(norm_before)) & \
             (norm_after > MIN_RELATIVE_NORM * np.median(norm_after))
    cosine = np.sum(before[usable] * after[usable], axis=1) / (norm_before[usable] * norm_after[usable])
    return 1 - cosine, usable


def load_edges(path):
    """Edges with integer IDs into a global per-type index, so an entity keeps its embedding row across years."""
    edges = pd.read_csv(path, keep_default_na=False)
    names = {"snp": pd.Index(edges["snp"].unique()),
             "gene": pd.Index(edges.loc[edges["target_type"] == "gene", "target"].unique()),
             "pheno": pd.Index(edges.loc[edges["target_type"] == "phenotype", "target"].unique())}
    edges["snp_id"] = names["snp"].get_indexer(edges["snp"])
    edges["target_id"] = np.where(edges["target_type"] == "gene", names["gene"].get_indexer(edges["target"]),
                                  names["pheno"].get_indexer(edges["target"]))
    return edges, names


edges, names = load_edges(EDGES)
rng = np.random.default_rng(SEED)
embeddings = {t: rng.normal(0, 0.1, (len(names[t]), DIM)) for t in TYPES}

year_rows, entity_tables, previous = [], [], None
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    present, a, b, n_edges = graph_until(edges, year)
    fitted = fit(a, b, *(embeddings[t][present[t]] for t in TYPES))
    row = {"year": year, "n_snps": len(present["snp"]), "n_genes": len(present["gene"]),
           "n_phenos": len(present["pheno"]), "n_edges": n_edges}
    skipped = {}
    for t, emb in zip(TYPES, fitted):
        embeddings[t][present[t]] = emb
        if previous is None:
            row[f"drift_{t}"], row[f"new_{t}s"] = np.nan, np.nan
            continue
        old_ids, old_emb = previous[t]   # entities never leave the cumulative graph
        distance, usable = drift(old_emb, embeddings[t][old_ids])
        row[f"drift_{t}"], row[f"new_{t}s"] = distance.mean(), len(present[t]) - len(old_ids)
        skipped[t] = 1 - usable.mean()
        entity_tables.append(pd.DataFrame({"year": year, "entity_type": t, "entity": names[t][old_ids[usable]],
                                           "drift": distance}))
    previous = {t: (present[t], embeddings[t][present[t]].copy()) for t in TYPES}
    year_rows.append(row)
    print(f"[drift] {year}: {n_edges:,} edges  drift SNP {row['drift_snp']:.4f}  gene {row['drift_gene']:.4f}  "
          f"phenotype {row['drift_pheno']:.4f}  |  skipped (near-zero embedding): "
          + ", ".join(f"{t} {share:.1%}" for t, share in skipped.items()))

years = pd.DataFrame(year_rows)[["year", "n_snps", "n_genes", "n_phenos", "n_edges", "drift_snp", "drift_gene",
                                 "drift_pheno", "new_snps", "new_genes", "new_phenos"]]
os.makedirs(os.path.dirname(OUT_YEARS), exist_ok=True)
years.to_csv(OUT_YEARS, index=False)
pd.concat(entity_tables, ignore_index=True).to_csv(OUT_ENTITIES, index=False)
print(f"[drift] Saved {OUT_YEARS} and {OUT_ENTITIES}")
