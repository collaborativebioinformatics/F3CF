#!/usr/bin/env python3
"""Turn Sebastian's data/synthgen (PRS, phenotypes, drugs) into Davor's federated site layout.

    python Elakiya/synthgen_prs/build_sites.py            # writes data/synthgen_federated

Deterministic (seed 0). For every site it writes, in the layout read by
scripts/federated_cf_data.py:load_site():

* ``patient_phenotypes.csv``: binary patients x phenotypes (rows ``<site>_<synthgen id>``), taken
  directly from data/synthgen/phenotypes.csv for the patients assigned to the site, minus the
  site's dropped phenotype columns.
* ``genome_phenotypes.csv``: PRS-to-phenotype effect sizes estimated locally at the site. Rows are
  the 100 polygenic scores (``<site>_prs_<trait>``), columns are phenotypes, a cell is the log odds
  ratio per standard deviation of PRS k for phenotype j, from a univariate logistic regression
  ``logit P(y_j = 1) = a + b * z_k`` on that site's patients, with ``z_k`` the PRS standardised
  within the site. A small ridge penalty on ``b`` (``lambda = 1``, negligible at n >= 700) keeps
  the estimate finite under complete separation or a zero-prevalence column; such columns are
  written as 0 because Davor's genome loss is an unmasked MSE and cannot take NaN. Site-3 gets no
  genome file (genome-less site, as in scripts/generate_federated_sites.py).
* ``patient_drugs.csv``: binary patients x drugs sidecar (not read by the current prototype).

Genome rows are site-local entities in the prototype (their embeddings never leave the client, only
the P x k phenotype tables are FedAvg'd), so row ids carry the site prefix like Davor's
``site-1_rs0000``. At the top level it writes ``phenotype_ids.txt``, ``sites.json`` and a
``true_embeddings.pt`` that is only a POOLED-DATA REFERENCE (synthgen's generative factors are not
published): nongenetic = top-k eigenvectors of the pooled phenotype correlation matrix scaled by
sqrt(eigenvalue); genetic = top-k right singular vectors of the pooled PRS x phenotype log-OR
matrix scaled by the singular values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from federated_cf_data import write_id_list, write_labeled_matrix  # noqa: E402

SYNTHGEN = ROOT / "data" / "synthgen"
SITES = [
    {"site_id": "site-1", "n_patients": 1000, "has_genome": True, "drop": 8},
    {"site_id": "site-2", "n_patients": 800, "has_genome": True, "drop": 10},
    {"site_id": "site-3", "n_patients": 700, "has_genome": False, "drop": 6},
]
RIDGE = 1.0
N_FACTORS = 32


def load_synthgen(src: Path = SYNTHGEN) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prs = pd.read_csv(src / "prs.csv", index_col="id")
    phen = pd.read_csv(src / "phenotypes.csv", index_col="id")
    drugs = pd.read_csv(src / "drugs.csv", index_col="id")
    if not (prs.index.equals(phen.index) and phen.index.equals(drugs.index)):
        raise ValueError("synthgen files do not share the same patient order")
    if list(prs.columns) != list(phen.columns):
        raise ValueError("prs.csv and phenotypes.csv columns differ")
    return prs, phen, drugs


def standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def logistic_effects(z: np.ndarray, y: np.ndarray, ridge: float = RIDGE, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Per-SD log-OR (and Wald SE) of every column of ``y`` (n x J, binary) on every column of ``z`` (n x K).

    Fits ``logit P(y_j) = a + b z_k`` for all (k, j) pairs by vectorised Newton-Raphson on the
    ridge-penalised log-likelihood ``l(a, b) - ridge * b**2 / 2`` (intercept unpenalised).
    Returns ``beta`` and ``se`` of shape (K, J). Columns with no cases or no controls give beta 0.
    """
    n, k = z.shape
    beta = np.zeros((k, y.shape[1]))
    se = np.zeros_like(beta)
    for j in range(y.shape[1]):
        yj = y[:, j].astype(float)
        prev = yj.mean()
        if prev <= 0 or prev >= 1:
            continue
        a = np.full(k, np.log(prev / (1 - prev)))
        b = np.zeros(k)
        for _ in range(iters):
            eta = a[:, None] + b[:, None] * z.T  # K x n
            p = 1.0 / (1.0 + np.exp(-eta))
            w = p * (1 - p)
            r = yj[None, :] - p
            ga, gb = r.sum(1), (r * z.T).sum(1) - ridge * b
            haa, hab, hbb = w.sum(1), (w * z.T).sum(1), (w * z.T**2).sum(1) + ridge
            det = haa * hbb - hab**2
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
            a, b = a + da, b + db
            if np.max(np.abs(db)) < 1e-8:
                break
        beta[:, j] = b
        se[:, j] = np.sqrt(haa / det)
    return beta, se


def _write_csv(path: Path, row_ids, col_ids, values: np.ndarray, decimals: int | None = None) -> None:
    vals = np.round(values, decimals) if decimals is not None else values
    write_labeled_matrix(path, list(row_ids), list(col_ids), torch.as_tensor(np.asarray(vals, dtype=np.float64)))


def build(output_dir: Path, seed: int = 0, src: Path = SYNTHGEN) -> Path:
    prs, phen, drugs = load_synthgen(src)
    rng = np.random.default_rng(seed)
    phenotype_ids = list(phen.columns)
    prs_ids = [f"prs_{c}" for c in prs.columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_id_list(output_dir / "phenotype_ids.txt", phenotype_ids)

    order = rng.permutation(len(phen))
    start = 0
    manifest = []
    for spec in SITES:
        rows = order[start : start + spec["n_patients"]]
        start += spec["n_patients"]
        rows.sort()
        dropped = sorted(rng.choice(len(phenotype_ids), size=spec["drop"], replace=False).tolist())
        cols = [c for i, c in enumerate(phenotype_ids) if i not in set(dropped)]
        site_dir = output_dir / spec["site_id"]
        site_dir.mkdir(parents=True, exist_ok=True)
        pids = [f"{spec['site_id']}_{i}" for i in phen.index[rows]]
        _write_csv(site_dir / "patient_phenotypes.csv", pids, cols, phen.iloc[rows][cols].to_numpy())
        _write_csv(site_dir / "patient_drugs.csv", pids, drugs.columns, drugs.iloc[rows].to_numpy())
        genome_path = site_dir / "genome_phenotypes.csv"
        if spec["has_genome"]:
            z = standardize(prs.iloc[rows].to_numpy())
            beta, _ = logistic_effects(z, phen.iloc[rows][cols].to_numpy())
            _write_csv(genome_path, [f"{spec['site_id']}_{p}" for p in prs_ids], cols, beta, decimals=5)
        elif genome_path.exists():
            genome_path.unlink()
        manifest.append({
            **spec,
            "n_genomes": len(prs_ids) if spec["has_genome"] else 0,
            "genome_rows": "per-SD log-OR of each phenotype on each PRS (site-local logistic fit)" if spec["has_genome"] else None,
            "dropped_phenotypes": [phenotype_ids[i] for i in dropped],
        })

    # Pooled-data reference embeddings (not the unknown synthgen generative factors).
    yall = phen.to_numpy()
    corr = np.corrcoef(yall.T)
    evals, evecs = np.linalg.eigh(corr)
    top = np.argsort(evals)[::-1][:N_FACTORS]
    ref_ng = evecs[:, top] * np.sqrt(np.clip(evals[top], 0, None))
    beta_all, _ = logistic_effects(standardize(prs.to_numpy()), yall)
    _, s, vt = np.linalg.svd(beta_all, full_matrices=False)
    ref_g = (vt[:N_FACTORS].T * s[:N_FACTORS])
    torch.save(
        {
            "nongenetic": torch.tensor(ref_ng, dtype=torch.float32),
            "genetic": torch.tensor(ref_g, dtype=torch.float32),
            "phenotype_ids": phenotype_ids,
            "note": "POOLED-DATA REFERENCE, not synthgen ground truth: nongenetic = top-32 eigvecs of pooled "
                    "phenotype correlation * sqrt(eigval); genetic = top-32 right singular vectors of pooled "
                    "PRS x phenotype per-SD log-OR * singular values.",
        },
        output_dir / "true_embeddings.pt",
    )
    (output_dir / "sites.json").write_text(json.dumps({
        "source": "data/synthgen (prs.csv, phenotypes.csv, drugs.csv) on main, commits 422661e and 7f85d90",
        "builder": "Elakiya/synthgen_prs/build_sites.py",
        "seed": seed,
        "ridge_lambda": RIDGE,
        "sites": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default=str(ROOT / "data" / "synthgen_federated"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"Wrote {build(Path(args.output_dir), args.seed)}")


if __name__ == "__main__":
    main()
