"""Hierarchy consistency: exclusivity, roll-up, per-node report, JSON export.

Order of operations (see :func:`resolve`):

1. **Exclusive siblings.** Within an ``exclusive_group`` a person keeps only
   the positive node with the highest ``priority``; evidence of the other
   nodes is kept with ``status = excluded_by:<winner>``.
2. **Roll-up.** A positive child makes every ancestor positive. The
   ancestor gets one evidence row per contributing child with
   ``status = rollup``, ``rule_id = rollup:<child>``.
3. **Checks.** Every child set is a subset of its parent set and exclusive
   siblings are disjoint (:func:`check_consistency`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.rules import EVIDENCE_COLUMNS


def parents(nodes: pd.DataFrame) -> dict:
    return dict(zip(nodes["node_id"], nodes["parent"]))


def ancestors(node: str, parent: dict) -> list[str]:
    out, cur = [], parent.get(node, "")
    while cur:
        out.append(cur)
        cur = parent.get(cur, "")
    return out


def order(nodes: pd.DataFrame) -> list[str]:
    """Depth-first order: each parent before its children, table order among siblings."""
    kids: dict[str, list[str]] = {}
    for n, p in zip(nodes["node_id"], nodes["parent"]):
        kids.setdefault(p, []).append(n)
    out = []

    def walk(n):
        out.append(n)
        for k in kids.get(n, []):
            walk(k)

    for root in kids.get("", []):
        walk(root)
    return out


def levels(nodes: pd.DataFrame) -> dict:
    parent = parents(nodes)
    return {n: len(ancestors(n, parent)) for n in nodes["node_id"]}


def apply_exclusivity(evidence: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    groups = nodes[nodes["exclusive_group"] != ""]
    if groups.empty or evidence.empty:
        return evidence
    ev = evidence.copy()
    prio = dict(zip(nodes["node_id"], nodes["priority"]))
    for g, grp in groups.groupby("exclusive_group"):
        members = set(grp["node_id"])
        sub = ev[ev["subphenotype"].isin(members) & (ev["status"] == "direct")]
        if sub.empty:
            continue
        best = (sub.assign(_p=sub["subphenotype"].map(prio))
                   .sort_values(["person_id", "_p"], ascending=[True, False])
                   .drop_duplicates("person_id").set_index("person_id")["subphenotype"])
        winner = sub["person_id"].map(best)
        lose = sub.index[(sub["subphenotype"] != winner).to_numpy()]
        ev.loc[lose, "status"] = "excluded_by:" + winner.loc[lose]
    return ev


def rollup(evidence: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    parent = parents(nodes)
    pos = evidence[evidence["status"] == "direct"][["person_id", "subphenotype"]].drop_duplicates()
    rows = []
    for child, grp in pos.groupby("subphenotype"):
        for anc in ancestors(child, parent):
            rows.append(pd.DataFrame({"person_id": grp["person_id"].to_numpy(), "subphenotype": anc,
                                      "rule_id": f"rollup:{child}", "rule_type": "rollup", "source": "hierarchy",
                                      "code_system": "subphenotype", "code": child, "value": np.nan, "unit": "",
                                      "date": pd.NaT, "status": "rollup"}))
    if not rows:
        return evidence
    return pd.concat([evidence] + rows, ignore_index=True)[EVIDENCE_COLUMNS]


def resolve(evidence: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    return rollup(apply_exclusivity(evidence, nodes), nodes)


def to_matrix(evidence: pd.DataFrame, persons: pd.Index, nodes: pd.DataFrame) -> pd.DataFrame:
    cols = order(nodes)
    m = pd.DataFrame(0, index=pd.Index(persons, name="person_id"), columns=cols, dtype=np.int8)
    if evidence.empty:
        return m
    pos = evidence[evidence["status"].isin(["direct", "rollup"])][["person_id", "subphenotype"]].drop_duplicates()
    pos = pos[pos["person_id"].isin(m.index) & pos["subphenotype"].isin(cols)]
    if len(pos):
        r = m.index.get_indexer(pos["person_id"])
        c = m.columns.get_indexer(pos["subphenotype"])
        arr = m.to_numpy().copy()
        arr[r, c] = 1
        m = pd.DataFrame(arr, index=m.index, columns=cols)
    return m


def check_consistency(matrix: pd.DataFrame, nodes: pd.DataFrame) -> list[str]:
    problems = []
    for n, p in zip(nodes["node_id"], nodes["parent"]):
        if p and n in matrix and p in matrix and (matrix[n] > matrix[p]).any():
            problems.append(f"{n} positive without parent {p}")
    for g, grp in nodes[nodes["exclusive_group"] != ""].groupby("exclusive_group"):
        cols = [c for c in grp["node_id"] if c in matrix]
        if cols and (matrix[cols].sum(axis=1) > 1).any():
            problems.append(f"exclusive group {g} has overlapping members")
    return problems


def report(matrix: pd.DataFrame, evidence: pd.DataFrame, nodes: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    lv = levels(nodes)
    lab = dict(zip(nodes["node_id"], nodes["label"]))
    par = parents(nodes)
    direct = evidence[evidence["status"] == "direct"] if len(evidence) else evidence
    excl = evidence[evidence["status"].astype(str).str.startswith("excluded_by")] if len(evidence) else evidence
    sex = persons["sex"].reindex(matrix.index).fillna("unknown")
    rows = []
    for n in matrix.columns:
        d = direct[direct["subphenotype"] == n] if len(direct) else direct
        by_type = d.drop_duplicates(["person_id", "rule_type"])["rule_type"].value_counts() if len(d) else pd.Series(dtype=int)
        by_src = d.drop_duplicates(["person_id", "source"])["source"].value_counts() if len(d) else pd.Series(dtype=int)
        rows.append({
            "subphenotype": n, "label": lab[n], "parent": par[n], "level": lv[n],
            "n_positive": int(matrix[n].sum()), "n_direct": int(d["person_id"].nunique()) if len(d) else 0,
            "n_excluded": int(excl.loc[excl["subphenotype"] == n, "person_id"].nunique()) if len(excl) else 0,
            "n_male": int(matrix.loc[sex == "male", n].sum()), "n_female": int(matrix.loc[sex == "female", n].sum()),
            "prevalence": float(matrix[n].mean()) if len(matrix) else 0.0,
            "persons_by_rule_type": ";".join(f"{k}:{v}" for k, v in by_type.items()),
            "persons_by_source": ";".join(f"{k}:{v}" for k, v in by_src.items()),
        })
    return pd.DataFrame(rows)


def hierarchy_json(nodes: pd.DataFrame, rules: pd.DataFrame, crosswalk: pd.DataFrame | None = None,
                   extra_nodes: list[dict] | None = None) -> dict:
    lv = levels(nodes)
    kids: dict[str, list[str]] = {}
    for n, p in zip(nodes["node_id"], nodes["parent"]):
        kids.setdefault(p, []).append(n)
    out = []
    for r in nodes.set_index("node_id").loc[order(nodes)].reset_index().itertuples(index=False):
        rr = rules[rules["node_id"] == r.node_id]
        codes = [{"rule_id": x.rule_id, "rule_type": x.rule_type, "system": x.code_system, "code": x.code, "match": x.match,
                  "min_count": None if np.isnan(x.min_count_i) else int(x.min_count_i), "verified": x.verified == "yes",
                  "source": x.source} for x in rr[rr["rule_type"] != "lab"].itertuples(index=False)]
        labs = [{"rule_id": x.rule_id, "rule_type": "lab", "analyte": x.analyte, "op": x.op, "threshold": x.threshold_f,
                 "unit": x.unit, "sex": x.sex, "aggregation": x.aggregation, "verified": x.verified == "yes",
                 "source": x.source} for x in rr[rr["rule_type"] == "lab"].itertuples(index=False)]
        xw = [] if crosswalk is None else crosswalk[crosswalk["node_id"] == r.node_id].drop(columns="node_id").to_dict("records")
        out.append({
            "id": r.node_id, "label": r.label, "parent": r.parent or None, "level": lv[r.node_id],
            "children": kids.get(r.node_id, []), "system": "hand", "rule_types": sorted(set(rr["rule_type"])),
            "logic": r.logic, "window_days": None if np.isnan(r.window_days) else int(r.window_days),
            "exclusive_group": r.exclusive_group or None, "priority": int(r.priority),
            "description": r.description, "source": r.source,
            "source_codes": codes, "lab_rules": labs, "phecode_crosswalk": xw,
        })
    return {"format": "metametagraphs.subphenotype_hierarchy/1", "nodes": out + (extra_nodes or [])}


def write_json(obj: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: None if o is pd.NaT else str(o)) + "\n")
