# Reads:  nothing directly (imported by collab_filter.py and accretion_demo.py)
# Writes: nothing
# Does:   SVD collaborative filtering on a gene x phenotype matrix, evaluated on held-out edges

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

BATCH_SIZE = 500          # genes scored at once (the dense score matrix would not fit in memory)
N_RANDOM_PAIRS = 10_000   # random gene-phenotype pairs scored as a reference


def pair_keys(df):
    """One string per gene-phenotype pair, for fast set operations."""
    return df["gene"] + "\t" + df["phenotype"]


def mask_edges(edges, mask_fraction, rng):
    """Randomly split edges into (train, test)."""
    n_mask = int(len(edges) * mask_fraction)
    is_masked = np.zeros(len(edges), dtype=bool)
    is_masked[rng.choice(len(edges), size=n_mask, replace=False)] = True
    return edges[~is_masked].reset_index(drop=True), edges[is_masked].reset_index(drop=True)


def bootstrap_ci(hits, rng, n_boot):
    """95% CI of the mean of a boolean array, by resampling with replacement."""
    boot = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def deduplicate(train, test):
    """One row per gene-phenotype pair in each set; refuse test edges that are also in training."""
    train = train.drop_duplicates(["gene", "phenotype"])
    test = test.drop_duplicates(["gene", "phenotype"]).reset_index(drop=True)
    if pair_keys(test).isin(set(pair_keys(train))).any():
        raise ValueError("test edges are also in the training edges")
    return train, test


def presence_matrix(train):
    """Binary gene x phenotype matrix of training edges. Scores are ignored: every cell is 0 or 1."""
    genes = pd.Index(train["gene"].unique())
    phenotypes = pd.Index(train["phenotype"].unique())
    rows, cols = genes.get_indexer(train["gene"]), phenotypes.get_indexer(train["phenotype"])
    matrix = sp.csr_matrix((np.ones(len(train)), (rows, cols)), shape=(len(genes), len(phenotypes)))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return genes, phenotypes, matrix


def propagate(genes, real, gene_gene, weight):
    """Give each gene its neighbors' phenotypes that it lacks, at `weight`.
    Gene order: training genes first, then STRING-only genes; genes with no entries are dropped.
    Returns (genes, matrix, real edges, propagated cells, number of STRING-only genes kept)."""
    string_genes = pd.Index(pd.unique(pd.concat([gene_gene["gene_a"], gene_gene["gene_b"]])))
    all_genes = genes.append(string_genes.difference(genes))
    a, b = all_genes.get_indexer(gene_gene["gene_a"]), all_genes.get_indexer(gene_gene["gene_b"])
    n = len(all_genes)
    adjacency = sp.csr_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    real = sp.vstack([real, sp.csr_matrix((n - len(genes), real.shape[1]))]).tocsr()
    reach = (adjacency @ real).tocsr()   # nonzero = at least one neighbor has this phenotype
    reach.data[:] = 1.0
    propagated = (reach - reach.multiply(real)).tocsr()
    propagated.eliminate_zeros()

    matrix = (real + weight * propagated).tocsr()
    keep = np.flatnonzero(matrix.getnnz(axis=1) > 0)
    n_string_only = len(keep) - (real.getnnz(axis=1) > 0).sum()
    return all_genes[keep], matrix[keep], real[keep], propagated[keep], n_string_only


def score_pairs(gene_factors, pheno_factors, test_g, test_p, covered, seed):
    """Predicted score of each covered test edge, and of random gene-phenotype pairs."""
    rng = np.random.default_rng(seed)
    test_score = np.full(len(test_g), np.nan)
    cg, cp = test_g[covered], test_p[covered]
    test_score[covered] = np.einsum("ij,ji->i", gene_factors[cg], pheno_factors[:, cp])
    rand_g = rng.integers(0, len(gene_factors), N_RANDOM_PAIRS)
    rand_p = rng.integers(0, pheno_factors.shape[1], N_RANDOM_PAIRS)
    random_score = np.einsum("ij,ji->i", gene_factors[rand_g], pheno_factors[:, rand_p])
    return test_score, random_score


def top_k_mask(scores, known, top_k):
    """Each row's top-K columns, ignoring known cells. Returns (masked scores, top indices, boolean mask)."""
    scores = np.where(known, -np.inf, scores)
    top = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
    in_top = np.zeros_like(known)
    in_top[np.arange(len(top))[:, None], top] = True
    return scores, top, in_top


def batch_predictions(genes, phenotypes, start, scores, top, g_local, p_local):
    """Top-K predictions for one batch of genes; held_out marks real test edges."""
    is_test = np.zeros(scores.shape, dtype=bool)
    is_test[g_local, p_local] = True
    return pd.DataFrame({
        "gene": genes[np.repeat(np.arange(start, start + len(top)), top.shape[1])],
        "phenotype": phenotypes[top.ravel()],
        "pred_score": np.take_along_axis(scores, top, axis=1).ravel(),
        "held_out": np.take_along_axis(is_test, top, axis=1).ravel(),
    })


def rank_test_edges(factors, known, genes, phenotypes, test_g, test_p, covered, top_k, keep_predictions):
    """Mark test edges that land in their gene's top-K (SVD and popularity baseline)."""
    gene_factors, pheno_factors = factors
    pheno_degree = np.asarray(known.sum(axis=0)).ravel()
    idx = np.flatnonzero(covered)
    idx = idx[np.argsort(test_g[idx], kind="stable")]   # covered test edges, sorted by gene
    sorted_g = test_g[idx]
    hit_svd, hit_pop = np.zeros(len(test_g), dtype=bool), np.zeros(len(test_g), dtype=bool)
    predictions = []
    for start in range(0, len(genes), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(genes))
        known_rows = known[start:stop].toarray() > 0
        scores, top, in_top = top_k_mask(gene_factors[start:stop] @ pheno_factors, known_rows, top_k)
        _, _, in_top_pop = top_k_mask(pheno_degree, known_rows, top_k)
        lo, hi = np.searchsorted(sorted_g, [start, stop])
        g_local, p_local = sorted_g[lo:hi] - start, test_p[idx[lo:hi]]
        hit_svd[idx[lo:hi]] = in_top[g_local, p_local]
        hit_pop[idx[lo:hi]] = in_top_pop[g_local, p_local]
        if keep_predictions:
            predictions.append(batch_predictions(genes, phenotypes, start, scores, top, g_local, p_local))
    if keep_predictions:
        predictions = pd.concat(predictions, ignore_index=True)
        predictions = predictions.sort_values(["gene", "pred_score"], ascending=[True, False])
    return hit_svd, hit_pop, predictions if keep_predictions else None


def evaluate(train, test, n_components, top_k, seed, keep_predictions=False, log=print,
             gene_gene=None, propagation_weight=None):
    """Fit SVD on train; recall = share of test edges in their gene's top-K (unseen gene/phenotype = miss).
    Propagated cells (gene_gene) shape the SVD but are not known edges. Returns (stats, hits, predictions)."""
    train, test = deduplicate(train, test)
    genes, phenotypes, known = presence_matrix(train)
    matrix, propagated = known, None
    if gene_gene is not None:
        genes, matrix, known, propagated, n_new = propagate(genes, known, gene_gene, propagation_weight)
        log(f"  STRING propagation: +{propagated.nnz:,} cells at weight {propagation_weight} "
            f"({n_new:,} genes only reachable through STRING)")
    test_g, test_p = genes.get_indexer(test["gene"]), phenotypes.get_indexer(test["phenotype"])
    covered = (test_g >= 0) & (test_p >= 0)
    log(f"  train: {len(train):,} edges, {len(genes):,} genes, {len(phenotypes):,} phenotypes  |  "
        f"test: {len(test):,} edges, {covered.mean():.1%} with gene and phenotype seen in training")

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    factors = (svd.fit_transform(matrix), svd.components_)
    test_score, random_score = score_pairs(*factors, test_g, test_p, covered, seed)
    hit_svd, hit_pop, predictions = rank_test_edges(factors, known, genes, phenotypes, test_g, test_p,
                                                    covered, top_k, keep_predictions)
    stats = {"train_edges": len(train), "genes": len(genes), "phenotypes": len(phenotypes),
             "test_edges": len(test), "test_covered": covered.mean(), f"recall@{top_k}": hit_svd.mean(),
             f"recall@{top_k}_popularity": hit_pop.mean(), "mean_score_test": np.nanmean(test_score),
             "mean_score_random": random_score.mean(), "explained_variance": svd.explained_variance_ratio_.sum()}
    filled = filled_by_propagation(propagated, test_g, test_p, covered)
    if propagated is not None:
        stats.update(propagated_cells=propagated.nnz, test_propagated=filled.mean())
    hits = test.assign(hit_svd=hit_svd, hit_pop=hit_pop, pred_score=test_score, filled_by_propagation=filled)
    return stats, hits, predictions


def filled_by_propagation(propagated, test_g, test_p, covered):
    """True for test edges whose cell was directly filled by propagation (from training edges only)."""
    filled = np.zeros(len(test_g), dtype=bool)
    if propagated is not None and covered.any():
        filled[covered] = np.asarray(propagated[test_g[covered], test_p[covered]]).ravel() != 0
    return filled
