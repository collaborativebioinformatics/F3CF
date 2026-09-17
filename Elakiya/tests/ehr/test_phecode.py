import pandas as pd

from conftest import make_events
from metametagraphs.ehr import phecode as P
from metametagraphs.ehr import rules as R


def test_truncation_rollup():
    assert P.ancestors("272.11") == ["272.1", "272"]
    assert P.ancestors("EM_239.11") == ["EM_239.1", "EM_239"]
    assert P.parent_of("495") == ""


def test_phecode12_and_x_maps():
    d, m = P.load("1.2")
    mp = dict(zip(m["icd10"], m["phecode"]))
    assert mp["E780"] == "272.11" and mp["E781"] == "272.12" and mp["E782"] == "272.13"
    assert dict(zip(d["phecode"], d["name"]))["272.1"] == "Hyperlipidemia"
    dx, mx = P.load("X")
    assert {"EM_239", "EM_239.1", "EM_239.11"} <= set(mx.loc[mx["icd10"] == "E780", "phecode"])


def test_build_nodes_and_evidence():
    ev = make_events([dict(person_id=1, kind="diagnosis", code_system="icd10", code="E78.0"),
                      dict(person_id=2, kind="diagnosis", code_system="icd10", code="I10"),
                      dict(person_id=3, kind="diagnosis", code_system="read2", code="C320."),     # not ICD10: ignored
                      dict(person_id=4, kind="diagnosis", code_system="icd10", code="K21.0")])    # unmapped chapter
    nodes, e = P.build(ev, "1.2")
    ids = set(nodes["node_id"])
    assert {"phecode1.2:272", "phecode1.2:272.1", "phecode1.2:272.11", "phecode1.2:401", "phecode1.2:401.1"} <= ids
    parent = dict(zip(nodes["node_id"], nodes["parent"]))
    assert parent["phecode1.2:272.11"] == "phecode1.2:272.1" and parent["phecode1.2:272"] == ""
    assert set(e["person_id"]) == {1, 2} and list(e.columns) == R.EVIDENCE_COLUMNS
    # combined with the hand table the node schema validates
    R.validate(pd.concat([R.load_nodes(), nodes], ignore_index=True), R.load_rules())


def test_three_character_fallback():
    ev = make_events([dict(person_id=1, kind="diagnosis", code_system="icd10", code="E11.8")])
    _, m = P.load("1.2")
    got = P.map_icd10(ev, m)
    assert len(got) >= 1


def test_crosswalk_is_consistent_with_phecode_files():
    xw = P.load_crosswalk()
    nodes = set(R.load_nodes()["node_id"])
    assert set(xw["node_id"]) <= nodes
    for v in ("1.2", "X"):
        d, _ = P.load(v)
        names = dict(zip(d["phecode"], d["name"]))
        for r in xw[xw["phecode_version"] == v].itertuples():
            assert r.phecode in names, r.phecode
            assert r.phecode_label == f"{r.phecode} {names[r.phecode]}"


def test_crosswalk_agreement():
    m = pd.DataFrame({"t2d": [1, 1, 0, 0], "phecode1.2:250.2": [1, 0, 1, 0]}, index=[1, 2, 3, 4])
    xw = pd.DataFrame([{"node_id": "t2d", "phecode_version": "1.2", "phecode": "250.2", "relation": "equivalent", "phecode_label": ""}])
    a = P.crosswalk_agreement(m, xw, "1.2").iloc[0]
    assert (a["n_hand"], a["n_phecode"], a["n_both"]) == (2, 2, 1) and abs(a["jaccard"] - 1 / 3) < 1e-9
