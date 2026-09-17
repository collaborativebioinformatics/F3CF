import numpy as np
import pandas as pd
import pytest

from conftest import make_events, persons
from metametagraphs.ehr import rules as R

NODE_COLS = ["node_id", "label", "parent", "exclusive_group", "priority", "logic", "window_days", "description", "source"]
RULE_COLS = ["rule_id", "node_id", "rule_type", "code_system", "code", "match", "analyte", "op", "threshold", "unit",
             "sex", "aggregation", "min_count", "verified", "source"]


def tables(tmp_path, nodes, rules):
    n = pd.DataFrame(nodes, columns=NODE_COLS).fillna("")
    r = pd.DataFrame(rules, columns=RULE_COLS).fillna("")
    n.to_csv(tmp_path / "n.csv", index=False)
    r.to_csv(tmp_path / "r.csv", index=False)
    return R.load_nodes(tmp_path / "n.csv"), R.load_rules(tmp_path / "r.csv")


def node(i, parent="", logic="any", window="", group="", prio=""):
    return [i, i, parent, group, prio, logic, window, "", ""]


def code_rule(i, n, system, code, match="exact", min_count="", rtype="code"):
    return [i, n, rtype, system, code, match, "", "", "", "", "any", "", min_count, "yes", ""]


def lab_rule(i, n, analyte, op, thr, agg="any", sex="any"):
    return [i, n, "lab", "", "", "", analyte, op, thr, "mmol/L", sex, agg, "", "yes", ""]


def hit_persons(ev, node_id):
    return set(ev.loc[ev["subphenotype"] == node_id, "person_id"])


def test_bundled_tables_validate():
    nodes, rules = R.load_nodes(), R.load_rules()
    R.validate(nodes, rules)
    assert {"code", "lab", "med"} <= set(rules["rule_type"])
    assert (rules["source"] != "").all(), "every rule must cite a source"
    assert (nodes["source"] != "").all()


@pytest.mark.parametrize("bad", ["parent", "cycle", "logic", "rule_node", "lab"])
def test_validate_rejects(tmp_path, bad):
    nodes = [node("a"), node("b", "a")]
    rules = [code_rule("r1", "a", "icd10", "E78")]
    if bad == "parent":
        nodes.append(node("c", "zzz"))
    if bad == "cycle":
        nodes = [node("a", "b"), node("b", "a")]
    if bad == "logic":
        nodes[0] = node("a", logic="code>2")
    if bad == "rule_node":
        rules.append(code_rule("r2", "nope", "icd10", "E78"))
    if bad == "lab":
        rules.append(lab_rule("r3", "a", "ldl_c", "=>", "4"))
    n, r = tables(tmp_path, nodes, rules)
    with pytest.raises(R.RuleError):
        R.validate(n, r)


def test_parse_logic():
    assert R.parse_logic("any") == [[("any", 1)]]
    assert R.parse_logic("code>=2 | code>=1 & med>=1") == [[("code", 2)], [("code", 1), ("med", 1)]]


def test_code_rules_exact_prefix_regex_and_kind(tmp_path):
    n, r = tables(tmp_path, [node("a")], [code_rule("r1", "a", "icd10", "E78.0"), code_rule("r2", "a", "read2", "C32..", "prefix"),
                                          code_rule("r3", "a", "drug_name", r"\bsimvastatin\b", "regex", rtype="med")])
    ev = make_events([
        dict(person_id=1, kind="diagnosis", code_system="icd10", code="E78.0"),
        dict(person_id=2, kind="diagnosis", code_system="icd10", code="E78.00"),     # exact rule: no
        dict(person_id=3, kind="diagnosis", code_system="read2", code="C3200"),      # prefix C32: yes
        dict(person_id=4, kind="medication", code_system="drug_name", code="Simvastatin 20mg tablets"),
        dict(person_id=5, kind="diagnosis", code_system="drug_name", code="simvastatin"),  # wrong kind for med rule
    ])
    out = R.evaluate_rules(ev, persons({i: "male" for i in range(1, 6)}), r)
    assert hit_persons(out, "a") == {1, 3, 4}
    assert list(out.columns) == R.EVIDENCE_COLUMNS


def test_min_count_is_pooled_over_codes_and_uses_distinct_dates(tmp_path):
    n, r = tables(tmp_path, [node("t")], [code_rule("r1", "t", "omop", "111", min_count="2", rtype="med"),
                                          code_rule("r2", "t", "omop", "222", min_count="2", rtype="med")])
    ev = make_events([
        dict(person_id=1, kind="medication", code_system="omop", code="111", date="2020-01-01"),
        dict(person_id=1, kind="medication", code_system="omop", code="222", date="2020-02-01"),   # two drugs -> 2
        dict(person_id=2, kind="medication", code_system="omop", code="111", date="2020-01-01"),
        dict(person_id=2, kind="medication", code_system="omop", code="222", date="2020-01-01"),   # same date -> 1
        dict(person_id=3, kind="medication", code_system="omop", code="111", date="2020-01-01"),
    ])
    out = R.evaluate_rules(ev, persons({1: "male", 2: "male", 3: "male"}), r)
    assert hit_persons(out, "t") == {1}
    # CLI default applies when the rule has no min_count
    n2, r2 = tables(tmp_path, [node("t")], [code_rule("r1", "t", "omop", "111", rtype="med")])
    assert hit_persons(R.evaluate_rules(ev, persons({1: "m", 2: "m", 3: "m"}), r2, default_min_count=2), "t") == set()
    assert hit_persons(R.evaluate_rules(ev, persons({1: "m", 2: "m", 3: "m"}), r2, default_min_count=1), "t") == {1, 2, 3}


def _ldl(pid, values_dates):
    return [dict(person_id=pid, kind="lab", code_system="omop", code="x", analyte="ldl_c", value=v, unit="mmol/L", date=d)
            for v, d in values_dates]


@pytest.mark.parametrize("agg,expected", [("any", {1, 2}), ("worst", {1, 2}), ("most_recent", {2}), ("mean", {2})])
def test_lab_aggregation(tmp_path, agg, expected):
    n, r = tables(tmp_path, [node("h")], [lab_rule("r1", "h", "ldl_c", ">=", "5.0", agg)])
    ev = make_events(_ldl(1, [(6.0, "2019-01-01"), (3.0, "2020-01-01"), (3.0, "")]) +   # treated: high then low
                     _ldl(2, [(4.0, "2019-01-01"), (6.5, "2021-01-01")]) +
                     _ldl(3, [(4.9, "2019-01-01")]))
    out = R.evaluate_rules(ev, persons({1: "male", 2: "male", 3: "male"}), r)
    assert hit_persons(out, "h") == expected
    if agg == "worst":
        assert out.set_index("person_id").loc[1, "value"] == 6.0
    if agg == "mean":
        assert out.set_index("person_id").loc[2, "code"] == "mean(2)"


def test_sex_specific_hdl_thresholds():
    rules = R.load_rules()
    rules = rules[rules["node_id"] == "dyslipidaemia.low_hdl"]
    ev = make_events([dict(person_id=p, kind="lab", code_system="ukb_field", code="30760", analyte="hdl_c", value=v, unit="mmol/L")
                      for p, v in ((1, 1.15), (2, 1.15), (3, 1.15), (4, 0.9), (5, 1.4))])
    ppl = persons({1: "male", 2: "female", 3: "unknown", 4: "male", 5: "female"})
    out = R.evaluate_rules(ev, ppl, rules)
    assert hit_persons(out, "dyslipidaemia.low_hdl") == {2, 4}


def test_node_logic_counts_and_window(tmp_path):
    nodes = [node("d", logic="code>=2 | code>=1 & med>=1", window="365"), node("e", logic="code>=2")]
    rules = [code_rule("c", "d", "icd10", "E78", "prefix"), code_rule("m", "d", "atc", "C10", "prefix", rtype="med"),
             code_rule("c2", "e", "icd10", "E78", "prefix")]
    n, r = tables(tmp_path, nodes, rules)
    dx = lambda p, d, code="E78.5": dict(person_id=p, kind="diagnosis", code_system="icd10", code=code, date=d)  # noqa: E731
    rx = lambda p, d: dict(person_id=p, kind="medication", code_system="atc", code="C10AA01", date=d)  # noqa: E731
    ev = make_events([
        dx(1, "2010-01-01"), dx(1, "2010-06-01"),                  # 2 codes within a year
        dx(2, "2010-01-01"), dx(2, "2013-01-01"),                  # 2 codes, 3 years apart
        dx(3, "2010-01-01"), rx(3, "2010-03-01"),                  # code + med in window
        dx(4, "2010-01-01"), dx(4, "2010-01-01"),                  # same code same day counts once
        dx(5, ""), dx(5, "2015-01-01", "E78.0"),                   # undated event fits any window
        rx(6, "2010-01-01"), rx(6, "2010-02-01"),                  # meds only
    ])
    hits = R.evaluate_rules(ev, persons({i: "male" for i in range(1, 7)}), r)
    out = R.apply_node_logic(hits, n)
    assert hit_persons(out, "d") == {1, 3, 5}
    assert hit_persons(out, "e") == {1, 2, 5}                      # no window on e
