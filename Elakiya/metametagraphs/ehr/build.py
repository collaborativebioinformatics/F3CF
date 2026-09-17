"""Build the subphenotype outputs from an EHR source.

Output contract (all written to ``out``):

``subphenotype_matrix.parquet``
    index ``person_id`` (int64, every person in the source, including people
    without any evidence); one int8 column per subphenotype id, 0/1, after
    exclusivity and roll-up. Columns are in hierarchy order (parent before
    children): hand-table nodes first, then phecode nodes if built.
``subphenotype_hierarchy.json``
    ``{"format": "metametagraphs.subphenotype_hierarchy/1", "nodes": [...]}``;
    each node has ``id, label, parent, level, children, system (hand |
    phecode1.2 | phecodeX), rule_types, logic, window_days,
    exclusive_group, priority, description, source, source_codes,
    lab_rules, phecode_crosswalk``.
``subphenotype_evidence.parquet``
    long format, one row per piece of evidence:
    ``person_id, subphenotype, rule_id, rule_type (code | lab | med |
    rollup), source (source table), code_system, code, value, unit, date,
    status (direct | rollup | excluded_by:<node>)``. Only evidence that
    passed each node's logic is kept.
``subphenotype_report.csv``
    per node: n_positive, n_direct, n_excluded, n_male, n_female,
    prevalence, persons by rule type and by source table.
``phecode_crosswalk_report.csv`` (with phecodes)
    agreement between hand nodes and mapped phecodes.
``build_info.json``
    inputs, rule files, counts, consistency problems (should be empty).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from metametagraphs.ehr import hierarchy as H
from metametagraphs.ehr import phecode as P
from metametagraphs.ehr import rules as R
from metametagraphs.ehr.events import load_lab_codes
from metametagraphs.ehr.sources import omop as omop_src
from metametagraphs.ehr.sources import ukb as ukb_src

OUTPUT_FILES = ["subphenotype_matrix.parquet", "subphenotype_hierarchy.json", "subphenotype_evidence.parquet",
                "subphenotype_report.csv", "build_info.json"]


def load_source(source: str, path: str | Path, subset: int | None = None, lab_codes: str | Path | None = None):
    lc = load_lab_codes(lab_codes)
    if source == "ukb":
        return ukb_src.load(path, subset, lc)
    if source == "omop":
        return omop_src.load(path, subset, lc)
    raise ValueError(f"unknown source {source!r}")


def build(source: str, path: str | Path, out: str | Path, rules: str | Path | None = None, nodes: str | Path | None = None,
          lab_codes: str | Path | None = None, min_count: int = 1, phecodes: tuple[str, ...] = ("1.2",),
          subset: int | None = None, phecode_dir: str | Path | None = None) -> dict:
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    node_df, rule_df = R.load_nodes(nodes), R.load_rules(rules)
    R.validate(node_df, rule_df)
    persons, events, info = load_source(source, path, subset, lab_codes)

    evidence = R.evaluate_rules(events, persons, rule_df, default_min_count=min_count)
    evidence = R.apply_node_logic(evidence, node_df)
    all_nodes, extra_json = node_df, []
    for version in phecodes:
        pn, pe = P.build(events, version, phecode_dir)
        if len(pn):
            all_nodes = pd.concat([all_nodes, pn], ignore_index=True)
            evidence = pd.concat([evidence, pe], ignore_index=True) if len(evidence) else pe
    evidence = H.resolve(evidence, all_nodes)
    matrix = H.to_matrix(evidence, persons.index, all_nodes)
    problems = H.check_consistency(matrix, all_nodes)
    rep = H.report(matrix, evidence, all_nodes, persons)

    crosswalk = P.load_crosswalk()
    lv = H.levels(all_nodes)
    for version in phecodes:
        extra_json += P.json_nodes(all_nodes[all_nodes["node_id"].str.startswith(P.PREFIX[version])], lv, version)
    hjson = H.hierarchy_json(node_df, rule_df, crosswalk, extra_json)

    ev_out = evidence.copy()
    ev_out["date"] = pd.to_datetime(ev_out["date"])
    ev_out["value"] = pd.to_numeric(ev_out["value"], errors="coerce")
    ev_out = ev_out.astype({c: str for c in ["subphenotype", "rule_id", "rule_type", "source", "code_system", "code", "unit", "status"]})
    matrix.to_parquet(out / "subphenotype_matrix.parquet")
    ev_out.to_parquet(out / "subphenotype_evidence.parquet", index=False)
    H.write_json(hjson, out / "subphenotype_hierarchy.json")
    rep.to_csv(out / "subphenotype_report.csv", index=False)
    xw = pd.concat([P.crosswalk_agreement(matrix, crosswalk, v) for v in phecodes], ignore_index=True) if phecodes else pd.DataFrame()
    if len(xw):
        xw.to_csv(out / "phecode_crosswalk_report.csv", index=False)
    build_info = {
        "source": source, "input": str(path), "rules": str(rules or R.RESOURCES / "rules.csv"),
        "nodes": str(nodes or R.RESOURCES / "nodes.csv"), "min_count_default": min_count, "phecodes": list(phecodes),
        "n_persons": int(len(persons)), "n_events": int(len(events)),
        "events_by_source": {k: int(v) for k, v in events["source"].value_counts().items()},
        "n_subphenotypes": int(matrix.shape[1]), "n_evidence_rows": int(len(evidence)),
        "consistency_problems": problems, "seconds": round(time.time() - t0, 2), **{f"source_{k}": v for k, v in info.items()},
    }
    (out / "build_info.json").write_text(json.dumps(build_info, indent=2) + "\n")
    return {"matrix": matrix, "evidence": evidence, "report": rep, "hierarchy": hjson, "persons": persons,
            "events": events, "nodes": all_nodes, "info": build_info, "crosswalk_report": xw}
