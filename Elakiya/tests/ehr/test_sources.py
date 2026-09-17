import pandas as pd

from conftest import FIXTURES
from metametagraphs.ehr.sources import omop, ukb

SYN = FIXTURES / "omop_synthea27nj"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_ukb_formats(tmp_path):
    write(tmp_path / "tabular" / "a.tsv",
          "EID\t31-0.0\t41270-0.0\t41270-0.1\t30780-0.0\t6177-0.0\t6153-0.0\n"
          "1000001\t1\tE780\tI10\t5.4\t1\t\n"
          "1000002\t0\t\t\t\t\t2\n")
    # synthetic-dataset layout: no header, yyyymmdd
    write(tmp_path / "medrec" / "set3a1.txt",
          "1000001\t1\t20100105\tC10F.\t\t\t\t\n"
          "1000001\t1\t20100105\t44P6.\t\t6.10\t\t\n"
          "1000002\t3\t20110203\t\tXE0Uc\t\t\t\n"
          "1000002\t1\t20110203\t44P5.\t\tnot-a-number\t\t\n")
    # portal layout: header, dd/mm/yyyy
    write(tmp_path / "medrec" / "gp_clinical_extra.txt",
          "eid\tdata_provider\tevent_dt\tread_2\tread_3\tvalue1\tvalue2\tvalue3\n1000002\t2\t05/06/2012\tH33..\t\t\t\t\n")
    write(tmp_path / "gp_scripts.txt",
          "eid\tdata_provider\tissue_date\tread_2\tbnf_code\tdmd_code\tdrug_name\tquantity\n"
          "1000001\t1\t01/02/2010\t\t02.12.00.00\t\tSimvastatin 40mg tablets\t28\n")
    ppl, ev, info = ukb.load(tmp_path)
    assert ppl["sex"].to_dict() == {1000001: "male", 1000002: "female"}
    assert info["dropped_lab_rows"] == 1
    key = set(zip(ev["person_id"], ev["kind"], ev["code_system"], ev["code"]))
    assert {(1000001, "diagnosis", "icd10", "E780"), (1000001, "diagnosis", "icd10", "I10"),
            (1000001, "diagnosis", "read2", "C10F"), (1000002, "diagnosis", "ctv3", "XE0Uc"),
            (1000002, "diagnosis", "read2", "H33"), (1000001, "medication", "ukb_field", "6177=1"),
            (1000002, "medication", "ukb_field", "6153=2"), (1000001, "medication", "bnf", "02120000"),
            (1000001, "medication", "drug_name", "simvastatin 40mg tablets")} <= key
    labs = ev[ev["kind"] == "lab"]
    assert sorted(labs["value"]) == [5.4, 6.1] and set(labs["analyte"]) == {"ldl_c"}
    dated = ev[(ev["code"] == "H33")]
    assert dated["date"].iloc[0] == pd.Timestamp("2012-06-05")
    assert ev.loc[ev["source"] == "ukb_tabular", "date"].isna().all()


def test_omop_eunomia_fixture():
    ppl, ev, info = omop.load(SYN)
    raw_p = pd.read_csv(SYN / "PERSON.csv", dtype=str)
    exp = raw_p.set_index(raw_p["person_id"].astype(int))["gender_source_value"].map({"M": "male", "F": "female"})
    assert ppl["sex"].to_dict() == exp.to_dict()
    assert info["vocabulary"] is True
    raw_m = pd.read_csv(SYN / "MEASUREMENT.csv", dtype=str)
    ldl_raw = pd.to_numeric(raw_m.loc[raw_m["measurement_concept_id"] == "3009966", "value_as_number"])
    ldl = ev[(ev["kind"] == "lab") & (ev["analyte"] == "ldl_c")]
    assert len(ldl) == len(ldl_raw) and abs(ldl["value"].sum() - ldl_raw.sum() * 0.02586) < 1e-6
    assert (ev["kind"] == "lab").sum() == raw_m["measurement_concept_id"].isin(["3009966", "3007070", "3022192", "3027114"]).sum()
    # SNOMED source values via CONCEPT vocabulary; drug names from CONCEPT
    assert "snomed" in set(ev["code_system"]) and "icd10" not in set(ev["code_system"])
    names = set(ev.loc[ev["code_system"] == "drug_name", "code"])
    assert any("simvastatin" in n for n in names)


def test_omop_upper_case_parquet_and_atc(tmp_path):
    for f in SYN.glob("*.csv"):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df.columns = [c.upper() for c in df.columns]
        df.to_parquet(tmp_path / (f.stem.lower() + ".parquet"))
    # toy ATC vocabulary: ancestor of simvastatin 20 MG with a toy concept id
    pd.DataFrame({"ancestor_concept_id": ["2100000003"], "descendant_concept_id": ["1539411"]}).to_csv(tmp_path / "concept_ancestor.csv", index=False)
    c = pd.read_parquet(tmp_path / "concept.parquet")
    extra = pd.DataFrame([{k: "" for k in c.columns}])
    extra.loc[0, ["CONCEPT_ID", "CONCEPT_NAME", "VOCABULARY_ID", "CONCEPT_CODE"]] = ["2100000003", "toy ATC", "ATC", "C10AA"]
    pd.concat([c, extra]).to_parquet(tmp_path / "concept.parquet")
    ppl, ev, info = omop.load(tmp_path)
    assert len(ppl) == 12 and info["concept_ancestor_rows"] == 1
    raw_d = pd.read_csv(SYN / "DRUG_EXPOSURE.csv", dtype=str)
    assert (ev["code_system"] == "atc").sum() == (raw_d["drug_concept_id"] == "1539411").sum()


def test_omop_without_vocabulary_guesses_source_systems(tmp_path):
    pd.DataFrame({"person_id": [1, 2], "gender_concept_id": [8532, 0], "gender_source_value": ["", "M"]}).to_csv(tmp_path / "person.csv", index=False)
    pd.DataFrame({"person_id": [1, 1, 2], "condition_concept_id": [432867, 0, 0],
                  "condition_start_date": ["2020-01-01"] * 3, "condition_source_value": ["E78.0", "55822004", "free text"]}).to_csv(
        tmp_path / "condition_occurrence.csv", index=False)
    pd.DataFrame({"person_id": [1], "measurement_concept_id": [3004410], "measurement_date": ["2020-01-01"],
                  "value_as_number": [7.0], "unit_concept_id": [8554]}).to_csv(tmp_path / "measurement.csv", index=False)
    ppl, ev, _ = omop.load(tmp_path)
    assert ppl["sex"].to_dict() == {1: "female", 2: "male"}
    systems = set(zip(ev["code_system"], ev["code"]))
    assert ("icd10", "E780") in systems and ("snomed", "55822004") in systems and ("omop", "432867") in systems
    a1c = ev[ev["analyte"] == "hba1c"]["value"].iloc[0]
    assert abs(a1c - (7.0 - 2.15) * 10.929) < 1e-9
