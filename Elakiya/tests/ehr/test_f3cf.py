import pandas as pd

from metametagraphs.ehr.__main__ import main
from metametagraphs.ehr.f3cf import DRUG_COLUMNS, PHENOTYPE_COLUMNS, drug_label, patient_drug


def test_drug_labels():
    assert drug_label("drug_name", "Simvastatin 40mg tablets") == "simvastatin"
    assert drug_label("drug_name", "atorvastatin 80 MG Oral Tablet") == "atorvastatin"
    assert drug_label("drug_name", "Ezetimibe 10mg tablets") == "ezetimibe"
    assert drug_label("atc", "c10aa") == "atc:C10AA"
    assert drug_label("bnf", "02120000") == "bnf:0212"
    assert drug_label("ukb_field", "6177=1") == "self_report:6177=1"
    assert drug_label("omop", "1539411") == "omop:1539411"


def test_patient_drug_scores_and_omop_dedup():
    ev = pd.DataFrame({
        "person_id": [1, 1, 1, 1, 2], "rule_type": "med", "status": "direct",
        "code_system": ["drug_name", "drug_name", "omop", "atc", "omop"],
        "code": ["simvastatin 20 mg", "simvastatin 10 mg", "1539411", "C10AA", "1545959"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-01-01", "2021-01-01"]),
    })
    out = patient_drug(ev, "clinic_a").set_index(["patient", "drug"])["score"].to_dict()
    assert out == {(1, "simvastatin"): 2, (1, "atc:C10AA"): 1, (2, "omop:1545959"): 1}


def test_cli_build_and_export(tmp_path, capsys):
    assert main(["build", "--source", "toy", "--input", str(tmp_path / "toy"), "--out", str(tmp_path / "b"), "--n", "150",
                 "--toy-schema", "omop", "--f3cf-site", "clinic_b"]) == 0
    assert main(["export-f3cf", "--build", str(tmp_path / "b"), "--out", str(tmp_path / "f"), "--site", "clinic_a"]) == 0
    assert "F3CF export for site clinic_a" in capsys.readouterr().out
    m = pd.read_parquet(tmp_path / "b" / "subphenotype_matrix.parquet")
    pp = pd.read_csv(tmp_path / "f" / "patient_phenotype.csv")
    dr = pd.read_csv(tmp_path / "f" / "patient_drug.csv")
    assert list(pp.columns) == PHENOTYPE_COLUMNS and list(dr.columns) == DRUG_COLUMNS
    assert len(pp) == int(m.to_numpy().sum()) and set(pp["score"]) == {1} and set(pp["source"]) == {"clinic_a"}
    assert set(pp["phenotype"]) <= set(m.columns)
    treated = set(m.index[m["dyslipidaemia.treated"] == 1])
    assert set(dr["patient"]) == treated
    assert {"simvastatin", "atorvastatin", "atc:C10AA"} <= set(dr["drug"]) and (dr["score"] >= 1).all()
    assert not dr["drug"].str.startswith("omop:").any()          # names available from the vocabulary
    b = pd.read_csv(tmp_path / "b" / "f3cf" / "patient_phenotype.csv")
    assert set(b["source"]) == {"clinic_b"} and len(b) == len(pp)
