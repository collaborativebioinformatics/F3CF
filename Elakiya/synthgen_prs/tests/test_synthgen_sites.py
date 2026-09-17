"""Checks for data/synthgen_federated (run: python -m pytest Elakiya/synthgen_prs/tests -q)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "Elakiya" / "synthgen_prs"))

from build_sites import SITES, build, load_synthgen, logistic_effects, standardize  # noqa: E402
from federated_cf_data import load_site, read_id_list, read_labeled_matrix  # noqa: E402

DATA = ROOT / "data" / "synthgen_federated"
IDS = read_id_list(DATA / "phenotype_ids.txt")


@pytest.mark.parametrize("spec", SITES, ids=lambda s: s["site_id"])
def test_site_loads_with_davor_loader(spec):
    site = load_site(DATA / spec["site_id"], IDS)
    n_cols = len(IDS) - spec["drop"]
    assert site.patient_phenotypes.shape == (spec["n_patients"], n_cols)
    assert not torch.isnan(site.patient_phenotypes).any()
    assert set(site.patient_phenotypes.unique().tolist()) <= {0.0, 1.0}
    assert all(p.startswith(spec["site_id"] + "_id") for p in site.patient_ids)
    if spec["has_genome"]:
        assert site.genome_phenotypes.shape == (100, n_cols)
        assert not torch.isnan(site.genome_phenotypes).any()
        assert torch.equal(site.genome_col_index, site.patient_col_index)
        assert all(g.startswith(spec["site_id"] + "_prs_") for g in site.genome_ids)
    else:
        assert site.genome_phenotypes is None


def test_sites_partition_patients_and_drugs_align():
    seen = []
    for spec in SITES:
        rows, cols, drugs = read_labeled_matrix(DATA / spec["site_id"] / "patient_drugs.csv")
        prow, _, _ = read_labeled_matrix(DATA / spec["site_id"] / "patient_phenotypes.csv")
        assert rows == prow and drugs.shape == (spec["n_patients"], 12)
        seen += [r.split("_", 1)[1] for r in rows]
    assert len(seen) == len(set(seen)) == 2500


def test_own_prs_effect_positive():
    """synthgen: PRS_k is the strongest predictor of pheno_k (corr ~0.36), so its per-SD log-OR must be > 0."""
    for spec in SITES:
        if not spec["has_genome"]:
            continue
        site = load_site(DATA / spec["site_id"], IDS)
        g = site.genome_phenotypes.numpy()
        for c, gi in enumerate(site.genome_col_index.tolist()):
            row = site.genome_ids.index(f"{spec['site_id']}_prs_{IDS[gi]}")
            assert g[row, c] > 0.5, (spec["site_id"], IDS[gi], g[row, c])
            assert g[row, c] == g[:, c].max()


def test_logistic_matches_scipy():
    from scipy.optimize import minimize

    prs, phen, _ = load_synthgen()
    z = standardize(prs.iloc[:300, :3].to_numpy())
    y = phen.iloc[:300, :2].to_numpy()
    beta, se = logistic_effects(z, y, ridge=0.0)
    for k in range(3):
        for j in range(2):
            def nll(t):
                eta = t[0] + t[1] * z[:, k]
                return np.sum(np.logaddexp(0, eta) - y[:, j] * eta)
            assert abs(minimize(nll, [0.0, 0.0], method="BFGS").x[1] - beta[k, j]) < 1e-4
    assert (se > 0).all()


def test_separation_stays_finite():
    z = np.linspace(-2, 2, 50)[:, None]
    y = np.stack([(z[:, 0] > 0).astype(int), np.zeros(50, int)], axis=1)
    beta, _ = logistic_effects(z, y)
    assert np.isfinite(beta).all() and beta[0, 0] > 0 and beta[0, 1] == 0


def test_build_is_deterministic(tmp_path):
    build(tmp_path)
    for f in ["phenotype_ids.txt", "sites.json", "site-1/genome_phenotypes.csv", "site-2/patient_phenotypes.csv",
              "site-3/patient_drugs.csv"]:
        assert (tmp_path / f).read_bytes() == (DATA / f).read_bytes(), f
    assert not (tmp_path / "site-3" / "genome_phenotypes.csv").exists()
    ref = torch.load(tmp_path / "true_embeddings.pt", weights_only=False)
    assert ref["nongenetic"].shape == ref["genetic"].shape == (100, 32)
    assert "POOLED" in ref["note"]
    assert json.loads((DATA / "sites.json").read_text())["seed"] == 0
