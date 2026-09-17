import numpy as np
import pandas as pd

from conftest import make_events, persons
from metametagraphs.ehr import hierarchy as H
from metametagraphs.ehr import rules as R

NODES = R.load_nodes()
RULES = R.load_rules()


def ldl(pid, v):
    return dict(person_id=pid, kind="lab", code_system="ukb_field", code="30780", analyte="ldl_c", value=v, unit="mmol/L")


def run(ev, ppl):
    hits = R.apply_node_logic(R.evaluate_rules(ev, ppl, RULES), NODES)
    res = H.resolve(hits, NODES)
    return res, H.to_matrix(res, ppl.index, NODES)


def test_order_and_levels():
    order = H.order(NODES)
    pos = {n: i for i, n in enumerate(order)}
    assert set(order) == set(NODES["node_id"])
    for n, p in zip(NODES["node_id"], NODES["parent"]):
        if p:
            assert pos[p] < pos[n]
    lv = H.levels(NODES)
    assert lv["dyslipidaemia"] == 0 and lv["dyslipidaemia.high_ldl.severe.dlcn_probable"] == 3


def test_rollup_exclusivity_and_consistency():
    ppl = persons({1: "male", 2: "female", 3: "male", 4: "female"})
    ev = make_events([ldl(1, 9.0), ldl(2, 5.5), ldl(3, 4.95), ldl(4, 3.0)])
    res, m = run(ev, ppl)
    # person 1: DLCN probable wins over possible; both ancestors positive
    assert m.loc[1, "dyslipidaemia.high_ldl.severe.dlcn_probable"] == 1
    assert m.loc[1, "dyslipidaemia.high_ldl.severe.dlcn_possible"] == 0
    assert m.loc[1, ["dyslipidaemia.high_ldl.severe", "dyslipidaemia.high_ldl", "dyslipidaemia"]].tolist() == [1, 1, 1]
    excl = res[(res["person_id"] == 1) & res["status"].str.startswith("excluded_by")]
    assert set(excl["subphenotype"]) == {"dyslipidaemia.high_ldl.severe.dlcn_possible"}
    assert m.loc[2, "dyslipidaemia.high_ldl.severe.dlcn_possible"] == 1
    assert m.loc[3, "dyslipidaemia.high_ldl.severe"] == 1 and m.loc[3, "dyslipidaemia.high_ldl.severe.dlcn_possible"] == 0
    assert m.loc[4].sum() == 0
    # parent reached only through roll-up has rollup evidence naming the child
    roll = res[(res["person_id"] == 3) & (res["subphenotype"] == "dyslipidaemia")]
    assert set(roll["status"]) == {"rollup"} and "dyslipidaemia.high_ldl.severe" in set(roll["code"])
    assert H.check_consistency(m, NODES) == []
    assert m.dtypes.unique().tolist() == [np.int8] and m.index.name == "person_id"


def test_check_consistency_detects_problems():
    m = pd.DataFrame(0, index=[1], columns=H.order(NODES), dtype=np.int8)
    m.loc[1, "t2d.renal"] = 1
    m.loc[1, ["dyslipidaemia.high_ldl.severe.dlcn_possible", "dyslipidaemia.high_ldl.severe.dlcn_probable"]] = 1
    probs = H.check_consistency(m, NODES)
    assert any("t2d.renal" in p for p in probs) and any("dlcn_ldl" in p for p in probs)


def test_report_and_json():
    ppl = persons({1: "male", 2: "female"})
    ev = make_events([ldl(1, 9.0), ldl(2, 5.5),
                      dict(person_id=2, kind="diagnosis", code_system="omop", code="192279", source="omop_condition_occurrence")])
    res, m = run(ev, ppl)
    rep = H.report(m, res, NODES, ppl).set_index("subphenotype")
    assert rep.loc["dyslipidaemia.high_ldl", "n_positive"] == 2 and rep.loc["dyslipidaemia.high_ldl", "n_male"] == 1
    assert rep.loc["dyslipidaemia.high_ldl.severe.dlcn_possible", "n_excluded"] == 1
    assert rep.loc["t2d", "n_positive"] == 1 and rep.loc["t2d", "n_direct"] == 0     # only via roll-up
    assert "omop_condition_occurrence:1" in rep.loc["t2d.renal", "persons_by_source"]
    js = H.hierarchy_json(NODES, RULES)
    first = js["nodes"][0]
    assert {"id", "label", "parent", "level", "children", "rule_types", "source_codes", "lab_rules"} <= set(first)
    by_id = {n["id"]: n for n in js["nodes"]}
    hdl = by_id["dyslipidaemia.low_hdl"]
    assert {r["sex"] for r in hdl["lab_rules"]} == {"male", "female", "unknown"}
    assert by_id["dyslipidaemia.high_ldl"]["parent"] == "dyslipidaemia" and by_id["dyslipidaemia"]["parent"] is None
