# SVD collaborative filtering on the gene x phenotype matrix, validated by masking edges
# Usage: python scripts/collab_filter.py data/edges.csv results/predictions.csv MASK_FRACTION N_COMPONENTS TOP_K SEED

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]
MASK_FRACTION = float(sys.argv[3])
N_COMPONENTS = int(sys.argv[4])
TOP_K = int(sys.argv[5])
SEED = int(sys.argv[6])

rng = np.random.default_rng(SEED)

# Load edges and index genes / phenotypes
print(f"[collab_filter] Loading {INPUT} ...")
edges = pd.read_csv(INPUT, keep_default_na=False)
gene_codes, genes = pd.factorize(edges["gene"])
pheno_codes, phenotypes = pd.factorize(edges["phenotype"])
n_genes, n_phenos, n_edges = len(genes), len(phenotypes), len(edges)

print(f"[collab_filter]   genes:      {n_genes:,}")
print(f"[collab_filter]   phenotypes: {n_phenos:,}")
print(f"[collab_filter]   edges:      {n_edges:,}")
print(f"[collab_filter]   sparsity:   {1 - n_edges / (n_genes * n_phenos):.4%}")

# Mask a random 10% of edges: they are hidden from training and used as the test set
n_mask = int(n_edges * MASK_FRACTION)
is_masked = np.zeros(n_edges, dtype=bool)
is_masked[rng.choice(n_edges, size=n_mask, replace=False)] = True
print(f"[collab_filter] Masked {n_mask:,} edges ({MASK_FRACTION:.0%})")

# Binary training matrix: 1 = known association. Scores are not used yet.
train = sp.csr_matrix(
    (np.ones((~is_masked).sum()), (gene_codes[~is_masked], pheno_codes[~is_masked])),
    shape=(n_genes, n_phenos),
)

# Factorize: train ~ gene_factors @ pheno_factors
print(f"[collab_filter] Running TruncatedSVD with {N_COMPONENTS} components ...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
gene_factors = svd.fit_transform(train)   # n_genes x k
pheno_factors = svd.components_           # k x n_phenos
print(f"[collab_filter]   explained variance: {svd.explained_variance_ratio_.sum():.1%}")

# Popularity baseline: rank phenotypes by how many genes they have in training
pheno_degree = np.asarray(train.sum(axis=0)).ravel()

# Masked edges grouped by gene, for fast lookup
masked_by_gene = {}
for g, p in zip(gene_codes[is_masked], pheno_codes[is_masked]):
    masked_by_gene.setdefault(g, set()).add(p)

# Score all phenotypes for each gene in batches (the full dense matrix would not fit in memory),
# exclude phenotypes already known in training, keep the top K
print(f"[collab_filter] Predicting top {TOP_K} phenotypes per gene ...")
hits_svd = 0
hits_pop = 0
rows = []
batch_size = 1000
for start in range(0, n_genes, batch_size):
    stop = min(start + batch_size, n_genes)
    scores = gene_factors[start:stop] @ pheno_factors
    known = train[start:stop].toarray() > 0
    scores[known] = -np.inf
    pop_scores = np.where(known, -np.inf, pheno_degree)

    top = np.argpartition(-scores, TOP_K, axis=1)[:, :TOP_K]
    top_pop = np.argpartition(-pop_scores, TOP_K, axis=1)[:, :TOP_K]

    for i, g in enumerate(range(start, stop)):
        held_out = masked_by_gene.get(g, set())
        hits_svd += len(held_out & set(top[i]))
        hits_pop += len(held_out & set(top_pop[i]))
        for p in top[i]:
            rows.append((genes[g], phenotypes[p], scores[i, p], p in held_out))

recall_svd = hits_svd / n_mask
recall_pop = hits_pop / n_mask
print(f"\n[collab_filter] Recall@{TOP_K} on {n_mask:,} masked edges")
print(f"[collab_filter]   SVD:                 {recall_svd:.1%}  ({hits_svd:,} recovered)")
print(f"[collab_filter]   popularity baseline: {recall_pop:.1%}  ({hits_pop:,} recovered)")

# held_out = True means this prediction is a real edge that was hidden during training
predictions = pd.DataFrame(rows, columns=["gene", "phenotype", "pred_score", "held_out"])
predictions = predictions.sort_values(["gene", "pred_score"], ascending=[True, False])
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
predictions.to_csv(OUTPUT, index=False)
print(f"\n[collab_filter] Saved {len(predictions):,} predictions to {OUTPUT}")
