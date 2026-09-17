# Reads:  nothing directly (imported by temporal_cmf.py)
# Writes: nothing
# Does:   collective matrix factorization: gene x phenotype X ~ U V^T and gene x gene S ~ U W^T, shared gene factors U

# Fitted by alternating least squares on the same unweighted squared loss over all cells that SVD minimizes:
#   ||X - U V^T||^2 + alpha ||S - U W^T||^2 + lambda (||U||^2 + ||V||^2 + ||W||^2)
# Initialized from TruncatedSVD of X, so with S = None it starts at the SVD solution and only adds lambda.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def gene_adjacency(genes, gene_gene):
    """Symmetric binary gene x gene matrix over `genes` (pairs with a gene outside `genes` are dropped)."""
    a, b = genes.get_indexer(gene_gene["gene_a"]), genes.get_indexer(gene_gene["gene_b"])
    keep = (a >= 0) & (b >= 0) & (a != b)
    a, b = a[keep], b[keep]
    n = len(genes)
    adjacency = sp.csr_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def solve(target, gram, lam):
    """Least-squares factor update: target @ (gram + lam I)^-1 (gram is symmetric)."""
    return np.linalg.solve(gram + lam * np.eye(len(gram)), target.T).T


def fit_cmf(x, s, n_components, alpha, lam, n_iter, seed):
    """Return (gene factors U, phenotype factors V^T). s = None fits X alone."""
    u = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x)
    for _ in range(n_iter):
        v = solve(x.T @ u, u.T @ u, lam)
        target, gram = x @ v, v.T @ v
        if s is not None:
            w = solve(s.T @ u, u.T @ u, lam)
            target, gram = target + alpha * (s @ w), gram + alpha * (w.T @ w)
        u = solve(target, gram, lam)
    v = solve(x.T @ u, u.T @ u, lam)
    return u, v.T


def cmf_factorizer(gene_gene, n_components, alpha, lam, n_iter, seed):
    """factorize(matrix, genes) for cf.evaluate; gene_gene = None gives CMF without STRING."""
    def factorize(matrix, genes):
        s = None if gene_gene is None else gene_adjacency(pd.Index(genes), gene_gene)
        return fit_cmf(matrix, s, n_components, alpha, lam, n_iter, seed)
    return factorize
