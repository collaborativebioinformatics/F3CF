import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metametagraphs.ehr import hpo
from metametagraphs.ehr.__main__ import main
from metametagraphs.ehr.rules import load_nodes

REPO = Path(__file__).resolve().parents[3]


def test_crosswalk_valid_against_excerpt():
    obo = hpo.load_obo()
    assert obo["_version"] == "hp/releases/2026-09-01"
    cw = hpo.load_crosswalk()
    nodes = list(load_nodes()["node_id"])
    assert hpo.validate(cw, obo, nodes) == []
    assert set(cw["subphenotype_id"]) == set(nodes)        # every hand node is listed, mapped or not
    unmapped = set(cw.loc[cw["match"] == "unmapped", "subphenotype_id"])
    assert unmapped == {"dyslipidaemia.treated", "hypertension.treated"}
    for h in hpo.load_ids():
        assert h in obo and not obo[h]["obsolete"]


def test_validate_catches_errors():
    obo = hpo.load_obo()
    bad = pd.DataFrame([
        {"subphenotype_id": "t2d", "hpo_id": "HP:0005978", "hpo_label": "Diabetes", "match": "exact", "implies": "yes", "note": ""},
        {"subphenotype_id": "t2d", "hpo_id": "HP:9999999", "hpo_label": "x", "match": "exact", "implies": "yes", "note": ""},
        {"subphenotype_id": "t2d", "hpo_id": "HP:0000488", "hpo_label": "Retinopathy", "match": "related", "implies": "yes", "note": ""},
        {"subphenotype_id": "nope", "hpo_id": "HP:0002099", "hpo_label": "Asthma", "match": "sort of", "implies": "no", "note": ""},
    ])
    probs = hpo.validate(bad, obo, ["t2d"])
    assert len(probs) == 5


def test_ancestor_propagation_true_path_rule():
    obo = hpo.load_obo()
    assert {"HP:0000819", "HP:0000118"} <= hpo.ancestors("HP:0005978", obo)
    assert "HP:0003077" in hpo.ancestors("HP:0002155", obo)
    m = pd.DataFrame({"t2d": [1, 0, 0], "dyslipidaemia.high_tg": [0, 1, 0], "dyslipidaemia.treated": [0, 0, 1],
                      "t2d.ophthalmic": [1, 0, 1]}, index=pd.Index([11, 12, 13], name="person_id"))
    out, rep = hpo.to_hpo_matrix(m)
    assert out.loc[11, ["HP:0005978", "HP:0000819"]].tolist() == [1, 1]
    assert out.loc[12, ["HP:0002155", "HP:0003077", "HP:0003119"]].tolist() == [1, 1, 1]
    assert out.loc[13].sum() == 0                     # treatment and related-only rows export nothing
    assert "HP:0000488" not in out.columns or out["HP:0000488"].sum() == 0
    assert set(np.unique(out.to_numpy())) <= {0, 1}
    kinds = set(rep["kind"])
    assert {"mapping", "column"} <= kinds
    # columns follow catalog order and are all reachable
    cat = hpo.load_ids()
    assert list(out.columns) == [c for c in cat if c in set(out.columns)]


def test_alignment_to_given_id_list(tmp_path):
    ids = ["HP:0002099", "HP:0000819", "HP:0005978", "HP:0030828", "HP:0003124"]   # custom order; wheezing is never reached
    m = pd.DataFrame({"t2d": [1], "asthma": [1], "dyslipidaemia.high_tc": [1], "t2d.renal": [1]}, index=pd.Index([5], name="person_id"))
    out, rep = hpo.to_hpo_matrix(m, catalog=ids)
    assert list(out.columns) == ["HP:0002099", "HP:0000819", "HP:0005978", "HP:0003124"]
    note = rep[(rep["kind"] == "mapping") & (rep["subphenotype_id"] == "t2d.renal")]["note"].iloc[0]
    assert "not in the phenotype id list" in note
    hpo.write_davor(out, tmp_path, "site-9", ids)
    assert (tmp_path / "phenotype_ids.txt").read_text().split() == ids
    with pytest.raises(ValueError):
        hpo.write_davor(out, tmp_path, "site-9", ids[:2])


def test_toy_build_export_hpo_davor_format(tmp_path, capsys):
    assert main(["build", "--source", "toy", "--input", str(tmp_path / "toy"), "--out", str(tmp_path / "b"), "--n", "120",
                 "--phecodes", "none", "--hpo-site", "site-7"]) == 0
    ids = hpo.load_ids()[:32]
    (tmp_path / "ids.txt").write_text("\n".join(ids) + "\n")
    assert main(["export-hpo", "--build", str(tmp_path / "b"), "--out", str(tmp_path / "h"), "--site", "site-1",
                 "--phenotype-ids", str(tmp_path / "ids.txt"), "--format", "davor"]) == 0
    assert "HPO export for site-1" in capsys.readouterr().out
    rows = list(csv.reader((tmp_path / "h" / "patient_phenotypes.csv").open()))
    header = rows[0]
    assert header[0] == "row_id" and set(header[1:]) <= set(ids)
    assert header[1:] == [c for c in ids if c in set(header[1:])]
    sub = pd.read_parquet(tmp_path / "b" / "subphenotype_matrix.parquet")
    assert [r[0] for r in rows[1:]] == [f"site-1_{p}" for p in sub.index]
    assert all(v in ("0", "1") for r in rows[1:] for v in r[1:])
    body = pd.read_csv(tmp_path / "h" / "patient_phenotypes.csv", index_col=0)
    assert (body["HP:0005978"].to_numpy() == sub["t2d"].to_numpy()).all()
    assert (body["HP:0003233"].to_numpy() == sub["dyslipidaemia.low_hdl"].to_numpy()).all()
    assert (tmp_path / "h" / "hpo_mapping_report.csv").exists()
    assert (tmp_path / "b" / "hpo" / "patient_phenotypes.csv").read_text().splitlines()[1].startswith("site-7_")


def _davor_loader():
    candidates = [REPO / "scripts", Path(os.environ.get("METAMETAGRAPHS_MAIN", "/tmp/v2")) / "scripts"]
    pytest.importorskip("torch")
    for c in candidates:
        if (c / "federated_cf_data.py").exists():
            sys.path.insert(0, str(c))
            return importlib.import_module("federated_cf_data")
    pytest.skip("scripts/federated_cf_data.py not found")


def test_davor_load_site_reads_export(tmp_path):
    fcd = _davor_loader()
    assert main(["build", "--source", "toy", "--input", str(tmp_path / "toy"), "--out", str(tmp_path / "b"), "--n", "60",
                 "--phecodes", "none"]) == 0
    ids_file = next((p for p in [REPO / "data" / "ehr_lipids" / "phenotype_ids.txt",
                                 Path(os.environ.get("METAMETAGRAPHS_MAIN", "/tmp/v2")) / "data" / "ehr_lipids" / "phenotype_ids.txt"]
                     if p.exists()), None)
    if ids_file is None:
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("\n".join(hpo.load_ids()[:32]) + "\n")
    site_dir = tmp_path / "fed" / "site-1"
    main(["export-hpo", "--build", str(tmp_path / "b"), "--out", str(site_dir), "--site", "site-1",
          "--phenotype-ids", str(ids_file)])
    ids = fcd.read_id_list(ids_file)
    site = fcd.load_site(site_dir, ids)
    assert site.patient_phenotypes.shape[0] == 60
    assert site.patient_ids[0].startswith("site-1_")
    assert site.genome_phenotypes is None
