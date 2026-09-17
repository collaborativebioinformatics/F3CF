"""data/ehr_lipids loads with Davor's loader and is internally consistent (network-free)."""

import csv
import filecmp
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ehr_lipids"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
from federated_cf_data import load_site, read_id_list  # noqa: E402

IDS = read_id_list(DATA / "phenotype_ids.txt")
HPO = {r["hpo_id"]: r for r in csv.DictReader((HERE / "hpo_terms.csv").open())}
SITES = json.loads((DATA / "sites.json").read_text())


def test_phenotype_ids_are_recorded_hpo_terms():
    assert len(IDS) == len(HPO) == 32
    for pid in IDS:
        assert re.fullmatch(r"HP:\d{7}", pid)
        assert pid in HPO and HPO[pid]["hpo_release"] == "hp/releases/2026-09-01"
    labels = {r["phenotype_id"]: r["label"] for r in csv.DictReader((DATA / "phenotype_labels.csv").open())}
    assert labels == {k: v["label"] for k, v in HPO.items()}


@pytest.mark.parametrize("spec", SITES, ids=[s["site_id"] for s in SITES])
def test_sites_load_with_davors_loader(spec):
    site = load_site(DATA / spec["site_id"], IDS)
    x = site.patient_phenotypes
    assert x.shape == (spec["n_patients"], len(IDS) - spec["drop"])
    assert set(torch.unique(x).tolist()) <= {0.0, 1.0}
    assert site.patient_ids[0] == f"{spec['site_id']}_p0000" and len(set(site.patient_ids)) == len(site.patient_ids)
    if spec["n_genomes"]:
        g = site.genome_phenotypes
        assert g.shape == (spec["n_genomes"], x.shape[1]) and torch.isfinite(g).all()
        assert all(r.startswith(f"{spec['site_id']}_rs") for r in site.genome_ids)
        assert torch.equal(site.genome_col_index, site.patient_col_index)
    else:
        assert site.genome_phenotypes is None


def test_hierarchy_and_clinical_rules_hold():
    implied = {"HP:0002155": "HP:0003077", "HP:0003141": "HP:0003077", "HP:0003077": "HP:0003119",
               "HP:0005978": "HP:0000819", "HP:0001658": "HP:0001677", "HP:0001681": "HP:0001677"}
    for spec in SITES:
        site = load_site(DATA / spec["site_id"], IDS)
        cols = [IDS[i] for i in site.patient_col_index.tolist()]
        x = site.patient_phenotypes
        for child, parent in implied.items():
            if child in cols and parent in cols:
                assert (x[:, cols.index(child)] <= x[:, cols.index(parent)]).all(), (spec["site_id"], child)


def test_effect_sources_match_genome_rows():
    rows = list(csv.DictReader((DATA / "effect_sources.csv").open()))
    rsids = [r["rsid"] for r in rows]
    assert len(rsids) == 14 and all(re.fullmatch(r"rs\d+", r) for r in rsids)
    assert all(r["source"] for r in rows)
    site = load_site(DATA / "site-1", IDS)
    assert [g.split("_", 1)[1] for g in site.genome_ids] == rsids
    ldl = site.genome_phenotypes[:, [IDS[i] for i in site.genome_col_index.tolist()].index("HP:0003141")]
    signs = {r["rsid"]: math.copysign(1, float(r["published_effect"])) for r in rows if r["trait"] == "ldl"}
    for i, rs in enumerate(rsids):
        if rs in signs:
            assert math.copysign(1, float(ldl[i])) == signs[rs]


def test_true_embeddings_like_davors():
    t = torch.load(DATA / "true_embeddings.pt", weights_only=False)
    assert list(t["phenotype_ids"]) == IDS
    assert t["nongenetic"].shape[0] == t["genetic"].shape[0] == len(IDS)


def test_build_is_deterministic(tmp_path):
    import build_sites

    build_sites.build(tmp_path, seed=0)
    for f in ["phenotype_ids.txt", "sites.json", "effect_sources.csv", "phenotype_labels.csv",
              "site-1/patient_phenotypes.csv", "site-1/genome_phenotypes.csv", "site-2/genome_phenotypes.csv",
              "site-3/patient_phenotypes.csv"]:
        assert filecmp.cmp(tmp_path / f, DATA / f, shallow=False), f
    assert not (tmp_path / "site-3" / "genome_phenotypes.csv").exists()
