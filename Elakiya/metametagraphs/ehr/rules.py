"""Rules engine: events + rule tables -> direct subphenotype evidence.

Two tables (defaults in ``resources/``, both overridable):

``nodes.csv``: one row per subphenotype node
    ``node_id, label, parent, exclusive_group, priority, logic, window_days, description, source``

    * ``logic``: how a node's rule hits combine per person. ``any`` (one
      qualifying event of any type) or a disjunction of conjunctions over
      event-type counts, e.g. ``code>=2 | code>=1 & med>=1`` (two diagnosis
      events, or one diagnosis plus one medication event). Types: ``code``,
      ``lab``, ``med``, ``any``. Counts are distinct (date, code) pairs;
      undated events count individually.
    * ``window_days``: if set, a clause has to be satisfied by events that
      all fall within one window of that many days. Undated events (for
      example UKB tabular fields) are treated as compatible with any window.
    * ``exclusive_group`` / ``priority``: siblings in the same group are
      mutually exclusive; the highest priority positive node wins
      (see :mod:`metametagraphs.ehr.hierarchy`).

``rules.csv``: one row per rule
    ``rule_id, node_id, rule_type, code_system, code, match, analyte, op,
    threshold, unit, sex, aggregation, min_count, verified, source``

    * ``rule_type=code``: diagnosis events with ``code_system`` whose code
      matches ``code`` (``exact``, ``prefix`` or ``regex``, after
      normalisation, see :func:`metametagraphs.ehr.events.normalise_code`).
    * ``rule_type=med``: the same over medication events.
    * ``rule_type=lab``: lab events of ``analyte`` (already in ``unit``)
      with ``value op threshold``; ``sex`` in {any, male, female, unknown}
      restricts the rule to persons of that sex; ``aggregation`` is
      ``any`` (every qualifying measurement is evidence), ``worst`` (the most
      extreme measurement in the direction of ``op``), ``most_recent`` (the
      latest dated measurement) or ``mean`` (the person's mean).
    * ``min_count``: a person needs at least this many distinct-date
      matching events (default from the CLI ``--min-count``, normally 1).
      Applies to code and med rules, pooled over all rules of the same node,
      rule type and code system (two different statins count as two).
"""

from __future__ import annotations

import operator
import re
from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.events import RESOURCES, normalise_code

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
KIND_OF = {"code": "diagnosis", "med": "medication", "lab": "lab"}
EVIDENCE_COLUMNS = ["person_id", "subphenotype", "rule_id", "rule_type", "source", "code_system", "code", "value",
                    "unit", "date", "status"]
_TERM = re.compile(r"^(code|lab|med|any)\s*>=\s*(\d+)$")


class RuleError(ValueError):
    pass


def load_nodes(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RESOURCES / "nodes.csv", dtype=str, keep_default_na=False)
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0)
    df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce")
    df["logic"] = df["logic"].replace("", "any")
    return df


def load_rules(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RESOURCES / "rules.csv", dtype=str, keep_default_na=False)
    df["threshold_f"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["min_count_i"] = pd.to_numeric(df["min_count"], errors="coerce")
    df["sex"] = df["sex"].replace("", "any")
    df["aggregation"] = df["aggregation"].replace("", "any")
    df["match"] = df["match"].replace("", "exact")
    df["norm"] = [c if m == "regex" else normalise_code(c, s) for c, s, m in zip(df["code"], df["code_system"], df["match"])]
    return df


def parse_logic(expr: str) -> list[list[tuple[str, int]]]:
    """``"code>=2 | code>=1 & med>=1"`` -> [[("code", 2)], [("code", 1), ("med", 1)]]."""
    expr = (expr or "any").strip()
    if expr == "any":
        return [[("any", 1)]]
    clauses = []
    for clause in expr.split("|"):
        terms = []
        for t in clause.split("&"):
            m = _TERM.match(t.strip())
            if not m:
                raise RuleError(f"bad logic term {t.strip()!r} in {expr!r}")
            terms.append((m.group(1), int(m.group(2))))
        clauses.append(terms)
    return clauses


def validate(nodes: pd.DataFrame, rules: pd.DataFrame) -> None:
    ids = set(nodes["node_id"])
    if len(ids) != len(nodes):
        raise RuleError("duplicate node_id in nodes table")
    for n, p in zip(nodes["node_id"], nodes["parent"]):
        if p and p not in ids:
            raise RuleError(f"node {n}: unknown parent {p}")
    parent = dict(zip(nodes["node_id"], nodes["parent"]))
    for n in ids:
        seen, cur = set(), n
        while parent.get(cur):
            if cur in seen:
                raise RuleError(f"cycle in hierarchy at {n}")
            seen.add(cur)
            cur = parent[cur]
    for g, grp in nodes[nodes["exclusive_group"] != ""].groupby("exclusive_group"):
        if grp["parent"].nunique() != 1:
            raise RuleError(f"exclusive group {g} spans several parents")
    for logic in nodes["logic"]:
        parse_logic(logic)
    bad = set(rules["node_id"]) - ids
    if bad:
        raise RuleError(f"rules reference unknown nodes {sorted(bad)}")
    bad_type = set(rules["rule_type"]) - set(KIND_OF)
    if bad_type:
        raise RuleError(f"unknown rule_type {sorted(bad_type)}")
    labs = rules[rules["rule_type"] == "lab"]
    if labs["threshold_f"].isna().any() or (~labs["op"].isin(OPS)).any() or (labs["analyte"] == "").any():
        raise RuleError("lab rules need analyte, op in >,>=,<,<= and a numeric threshold")
    if (~labs["aggregation"].isin(["any", "worst", "most_recent", "mean"])).any():
        raise RuleError("lab aggregation must be any, worst, most_recent or mean")
    if (~rules["sex"].isin(["any", "male", "female", "unknown"])).any():
        raise RuleError("sex must be any, male, female or unknown")
    if (~rules["match"].isin(["exact", "prefix", "regex"])).any():
        raise RuleError("match must be exact, prefix or regex")


# --------------------------------------------------------------------------
# Rule evaluation
# --------------------------------------------------------------------------
def _evidence(ev: pd.DataFrame, rule, status: str = "direct") -> pd.DataFrame:
    return pd.DataFrame({
        "person_id": ev["person_id"].to_numpy(), "subphenotype": rule.node_id, "rule_id": rule.rule_id,
        "rule_type": rule.rule_type, "source": ev["source"].to_numpy(), "code_system": ev["code_system"].to_numpy(),
        "code": ev["code"].to_numpy(), "value": ev["value"].to_numpy(), "unit": ev["unit"].to_numpy(),
        "date": ev["date"].to_numpy(), "status": status,
    })


def _code_hits(ev: pd.DataFrame, rule) -> pd.DataFrame:
    codes = ev["code"]
    if rule.match == "exact":
        m = codes == rule.norm
    elif rule.match == "prefix":
        m = codes.str.startswith(rule.norm)
    else:
        m = codes.str.contains(rule.norm, flags=re.IGNORECASE, regex=True)
    return ev[m.to_numpy()]


def _lab_hits(ev: pd.DataFrame, rule, sex: pd.Series) -> pd.DataFrame:
    if rule.sex != "any":
        ev = ev[ev["person_id"].map(sex).fillna("unknown").to_numpy() == rule.sex]
    if ev.empty:
        return ev
    op, thr = OPS[rule.op], rule.threshold_f
    agg = rule.aggregation
    if agg == "any":
        return ev[op(ev["value"], thr).to_numpy()]
    if agg == "worst":
        high = rule.op in (">", ">=")
        idx = ev.groupby("person_id")["value"].idxmax() if high else ev.groupby("person_id")["value"].idxmin()
        pick = ev.loc[idx]
    elif agg == "most_recent":
        s = ev.assign(_has=ev["date"].notna()).sort_values(["person_id", "_has", "date"], kind="stable")
        pick = s.groupby("person_id").tail(1).drop(columns="_has")
    else:  # mean
        g = ev.groupby("person_id")
        pick = g.tail(1).set_index("person_id")
        pick["value"] = g["value"].mean()
        pick["date"] = g["date"].max()
        pick["code"] = "mean(" + g.size().astype(str) + ")"
        pick = pick.reset_index()
    return pick[op(pick["value"], thr).to_numpy()]


def evaluate_rules(events: pd.DataFrame, persons: pd.DataFrame, rules: pd.DataFrame, default_min_count: int = 1) -> pd.DataFrame:
    """All rule hits as evidence rows (status ``direct``), before node logic.

    ``min_count`` is applied to the pooled hits of all code/med rules that
    share (node, rule_type, code_system, min_count), so "two prescriptions"
    counts two different statin concepts as well as two of the same.
    """
    out, pooled = [], {}
    by_kind = {k: events[events["kind"] == k] for k in KIND_OF.values()}
    sex = persons["sex"] if "sex" in persons else pd.Series(dtype=str)
    for rule in rules.itertuples(index=False):
        ev = by_kind[KIND_OF[rule.rule_type]]
        if rule.rule_type == "lab":
            hits = _lab_hits(ev[ev["analyte"] == rule.analyte], rule, sex)
            if len(hits):
                out.append(_evidence(hits, rule))
            continue
        hits = _code_hits(ev[ev["code_system"] == rule.code_system], rule)
        if len(hits):
            n = int(rule.min_count_i) if not np.isnan(rule.min_count_i) else default_min_count
            pooled.setdefault((rule.node_id, rule.rule_type, rule.code_system, n), []).append(_evidence(hits, rule))
    for (_, _, _, n), parts in pooled.items():
        ev = pd.concat(parts, ignore_index=True)
        if n > 1:
            k = ev["date"].astype("string").fillna(pd.Series(ev.index.astype(str), index=ev.index))
            counts = k.groupby(ev["person_id"]).nunique()
            ev = ev[ev["person_id"].isin(counts[counts >= n].index)]
        if len(ev):
            out.append(ev)
    if not out:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Node logic
# --------------------------------------------------------------------------
def _satisfied(counts: dict, clauses) -> bool:
    for clause in clauses:
        if all((sum(counts.values()) if t == "any" else counts.get(t, 0)) >= n for t, n in clause):
            return True
    return False


def _person_passes(ev: pd.DataFrame, clauses, window: float) -> bool:
    undated = ev[ev["date"].isna()]
    base = undated["rule_type"].value_counts().to_dict()
    dated = ev[ev["date"].notna()].sort_values("date")
    if np.isnan(window) or dated.empty:
        counts = dict(base)
        for k, v in dated.drop_duplicates(["rule_type", "code", "date"])["rule_type"].value_counts().items():
            counts[k] = counts.get(k, 0) + v
        return _satisfied(counts, clauses)
    dates = dated["date"].to_numpy()
    types = dated["rule_type"].to_numpy()
    keys = list(zip(types, dated["code"].to_numpy(), dates))
    span = np.timedelta64(int(window), "D")
    for i, start in enumerate(dates):
        counts = dict(base)
        seen = set()
        for j in range(i, len(dates)):
            if dates[j] - start > span:
                break
            if keys[j] in seen:
                continue
            seen.add(keys[j])
            counts[types[j]] = counts.get(types[j], 0) + 1
        if _satisfied(counts, clauses):
            return True
    return False


def apply_node_logic(evidence: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Keep evidence only for (person, node) pairs that satisfy the node's logic."""
    if evidence.empty:
        return evidence
    keep = []
    spec = {r.node_id: (parse_logic(r.logic), r.window_days) for r in nodes.itertuples(index=False)}
    for node, grp in evidence.groupby("subphenotype", sort=False):
        clauses, window = spec[node]
        if clauses == [[("any", 1)]]:
            keep.append(grp)
            continue
        ok = [pid for pid, g in grp.groupby("person_id") if _person_passes(g, clauses, window)]
        keep.append(grp[grp["person_id"].isin(ok)])
    return pd.concat(keep, ignore_index=True)
