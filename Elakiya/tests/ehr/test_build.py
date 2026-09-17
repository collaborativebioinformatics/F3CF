"""End to end: toy cohort in both layouts, the Eunomia fixture, output contract and CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from metametagraphs.ehr import OUTPUT_FILES, build
from metametagraphs.ehr.__main__ import main
from metametagraphs.ehr.rules import EVIDENCE_COLUMNS

LAB_NODES = ["dyslipidaemia.high_ldl.severe", "dyslipidaemia.high_ldl.severe.dlcn_possible",
             "dyslipidaemia.high_ldl.severe.dlcn_probable", "dyslipidaemia.high_tg.very_high", "dyslipidaemia.high_tc",
             "t2d.poor_control"]


@pytest.fixture(scope="module")
def ukb_res(toy_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_ukb")
    return build("ukb", toy_dir / "ukb", out, phecodes=("1.2", "X")), out


@pytest.fixture(scope="module")
def omop_res(toy_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_omop")
    return build("omop", toy_dir / "omop", out, phecodes=("1.2",)), out


def test_output_contract(ukb_res):
    res, out = ukb_res
    for f in OUTPUT_FILES + ["phecode_crosswalk_report.csv"]:
        assert (out / f).exists(), f
    m = pd.read_parquet(out / "subphenotype_matrix.parquet")
    assert m.index.name == "person_id" and m.index.dtype == np.int64 and len(m) == 300
    assert set(np.unique(m.to_numpy())) <= {0, 1} and all(t == np.int8 for t in m.dtypes)
    h = json.loads((out / "subphenotype_hierarchy.json").read_text())
    assert h["format"] == "metametagraphs.subphenotype_hierarchy/1"
    ids = [n["id"] for n in h["nodes"]]
    assert ids == list(m.columns)
    for n in h["nodes"]:
        assert {"id", "label", "parent", "level", "source_codes", "rule_types"} <= set(n)
        if n["parent"]:
            assert ids.index(n["parent"]) < ids.index(n["id"])
    ev = pd.read_parquet(out / "subphenotype_evidence.parquet")
    assert list(ev.columns) == EVIDENCE_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(ev["date"])
    # every positive cell has evidence and every kept evidence row is a positive cell
    pos = set(zip(*np.nonzero(m.to_numpy())))
    idx = {p: i for i, p in enumerate(m.index)}
    col = {c: j for j, c in enumerate(m.columns)}
    kept = ev[ev["status"].isin(["direct", "rollup"])]
    assert {(idx[p], col[s]) for p, s in zip(kept["person_id"], kept["subphenotype"])} == pos
    info = json.loads((out / "build_info.json").read_text())
    assert info["consistency_problems"] == [] and info["n_persons"] == 300


def test_toy_truth_recovered(ukb_res, toy_dir):
    res, _ = ukb_res
    m = res["matrix"]
    t = pd.read_csv(toy_dir / "toy_truth.csv").set_index("person_id")
    worst = t[["ldl1", "ldl2"]].max(axis=1)
    assert set(m.index[m["dyslipidaemia.high_ldl.severe.dlcn_probable"] == 1]) == set(t.index[worst > 8.5])
    assert set(m.index[m["dyslipidaemia.high_ldl.severe.dlcn_possible"] == 1]) == set(t.index[(worst >= 5.0) & (worst <= 8.5)])
    cut = np.where(t["sex"] == "male", 1.03, 1.29)
    lab_low = set(t.index[t["hdl"] < cut])
    coded = set(res["evidence"].query("subphenotype == 'dyslipidaemia.low_hdl' and rule_type == 'code'")["person_id"])
    assert set(m.index[m["dyslipidaemia.low_hdl"] == 1]) == lab_low | coded
    assert set(m.index[m["dyslipidaemia.treated"] == 1]) == set(t.index[t["n_scripts"] >= 1])  # >=2 scripts or self-report
    assert set(m.index[m["dyslipidaemia.high_lpa"] == 1]) == set(t.index[t["lpa"] > 105])
    assert m["dyslipidaemia.high_lpa.very_high"].sum() == 0
    # two-code-or-code-plus-treatment rule for the parent's own codes
    direct = set(res["evidence"].query("subphenotype == 'dyslipidaemia' and status == 'direct'")["person_id"])
    gap = (pd.to_datetime(t["date2"]) - pd.to_datetime(t["date1"])).dt.days
    gp2 = set(t.index[(t["dys_gp_codes"] >= 2) & ~t["ctv3_only"]])
    assert set(t.index[(gap <= 730)]) & gp2 <= direct                  # two codes inside the 730-day window
    far = set(t.index[(gap > 730) & ~t["dys_hosp_code"] & (t["n_scripts"] == 0)]) & gp2
    assert far and not (far & direct)                                  # two codes too far apart
    single = set(t.index[(t["dys_gp_codes"] == 1) & ~t["dys_hosp_code"] & (t["n_scripts"] == 0) & ~t["ctv3_only"]])
    assert single and not (single & direct)


def test_ukb_and_omop_layouts_agree_on_lab_nodes(ukb_res, omop_res):
    mu, mo = ukb_res[0]["matrix"], omop_res[0]["matrix"]
    for n in LAB_NODES:
        assert (mu[n] == mo[n]).all(), n
    lab_hdl = lambda r: set(r["evidence"].query("subphenotype == 'dyslipidaemia.low_hdl' and rule_type == 'lab'")["person_id"])  # noqa: E731
    assert lab_hdl(ukb_res[0]) == lab_hdl(omop_res[0])
    t_u = set(mu.index[mu["dyslipidaemia.treated"] == 1])
    t_o = set(mo.index[mo["dyslipidaemia.treated"] == 1])
    assert t_o < t_u          # OMOP has no self-report field, so single-script people are missing
    ev = omop_res[0]["evidence"]
    assert {"atc", "drug_name", "omop"} <= set(ev.loc[ev["subphenotype"] == "dyslipidaemia.treated", "code_system"])


def test_eunomia_fixture_build(tmp_path):
    res = build("omop", FIXTURES / "omop_synthea27nj", tmp_path, phecodes=())
    m = res["matrix"]
    raw = pd.read_csv(FIXTURES / "omop_synthea27nj" / "DRUG_EXPOSURE.csv", dtype=str)
    st = raw[raw["drug_concept_id"].isin(["1539463", "1539411", "1545959"])]
    two = st.groupby("person_id")["drug_exposure_start_date"].nunique()
    assert set(m.index[m["dyslipidaemia.treated"] == 1]) == set(two[two >= 2].index.astype(int))
    cond = pd.read_csv(FIXTURES / "omop_synthea27nj" / "CONDITION_OCCURRENCE.csv", dtype=str)
    htn = set(cond.loc[cond["condition_concept_id"] == "320128", "person_id"].astype(int))
    assert set(m.index[m["hypertension.essential"] == 1]) == htn and (m["hypertension"] >= m["hypertension.essential"]).all()
    assert res["info"]["consistency_problems"] == []


def test_cli_toy_rules_override_and_min_count(tmp_path, capsys):
    rc = main(["build", "--source", "toy", "--input", str(tmp_path / "toy"), "--out", str(tmp_path / "o1"), "--n", "120",
               "--phecodes", "none"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "hand hierarchy: 27 nodes" in text and "consistency problems: none" in text
    # custom rule file: only T2D codes
    from metametagraphs.ehr.rules import load_rules
    r = load_rules()
    r[r["node_id"].str.startswith("t2d")].drop(columns=["threshold_f", "min_count_i", "norm"]).to_csv(tmp_path / "rules.csv", index=False)
    assert main(["build", "--source", "ukb", "--input", str(tmp_path / "toy" / "ukb"), "--out", str(tmp_path / "o2"),
                 "--rules", str(tmp_path / "rules.csv"), "--phecodes", "none"]) == 0
    m2 = pd.read_parquet(tmp_path / "o2" / "subphenotype_matrix.parquet")
    assert m2["dyslipidaemia"].sum() == 0 and m2["t2d"].sum() > 0
    # --min-count 2 removes single-date T2D codes
    assert main(["build", "--source", "ukb", "--input", str(tmp_path / "toy" / "ukb"), "--out", str(tmp_path / "o3"),
                 "--rules", str(tmp_path / "rules.csv"), "--phecodes", "none", "--min-count", "2"]) == 0
    m3 = pd.read_parquet(tmp_path / "o3" / "subphenotype_matrix.parquet")
    assert m3["t2d"].sum() < m2["t2d"].sum()
    assert main(["validate-rules"]) == 0
    assert main(["toy", "--out", str(tmp_path / "toy2"), "--n", "20"]) == 0
    assert (tmp_path / "toy2" / "omop" / "person.csv").exists()
