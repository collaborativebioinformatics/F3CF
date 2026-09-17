"""Map subphenotypes to HPO terms and write the patient x HPO matrix used by the federated CF prototype.

Resources (``resources/``):

``hpo_crosswalk.csv``
    ``subphenotype_id, hpo_id, hpo_label, match, implies, note``. ``match`` is ``exact``,
    ``broader`` (the HPO term is broader than the node), ``narrower``, ``related`` or
    ``unmapped``. ``implies = yes`` only for exact and broader rows: a positive node then
    guarantees the HPO term. Narrower and related rows are reported but never exported.
``hpo_catalog.txt``
    default global phenotype list: the 32 ids of ``data/ehr_lipids/phenotype_ids.txt`` (PR #3)
    plus the crosswalk terms not in it.
``hpo_subset/hp_excerpt.obo``
    id, name and is_a of the catalog terms and all their ancestors, from hp.obo release
    ``hp/releases/2026-09-01`` (see ``hpo_subset/SOURCES.md``).

Conversion (:func:`to_hpo_matrix`): for every node with ``implies = yes`` rows, a positive
person gets the mapped HPO term and every HPO ancestor of it that is in the column set (true
path rule). Output columns are the catalog terms that can be reached this way, in catalog
order, so the file only has columns the EHR rules can actually observe; with a given
``phenotype_ids.txt`` the columns are a subset of, and ordered like, that list, which is what
``scripts/federated_cf_data.load_site`` requires.

Output (:func:`write_davor`), the layout of ``data/federated`` / ``data/ehr_lipids``:
``patient_phenotypes.csv`` (header ``row_id,<HP ids>``, rows ``<site>_<person_id>``, 0/1),
``phenotype_ids.txt`` (the global list) and ``hpo_mapping_report.csv``.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.events import RESOURCES

CROSSWALK = RESOURCES / "hpo_crosswalk.csv"
CATALOG = RESOURCES / "hpo_catalog.txt"
EXCERPT = RESOURCES / "hpo_subset" / "hp_excerpt.obo"
HPO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"
MATCHES = {"exact", "broader", "narrower", "related", "unmapped"}
REPORT_COLUMNS = ["kind", "subphenotype_id", "hpo_id", "hpo_label", "match", "implies", "n_positive", "exported_as", "note"]


def download_hpo(dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(HPO_URL, dest)
    return dest


def load_obo(path: str | Path | None = None) -> dict:
    """``{id: {"name": str, "parents": [ids], "obsolete": bool}}`` plus ``"_version"``."""
    terms, cur, version = {}, None, ""
    for line in Path(path or EXCERPT).read_text(encoding="utf-8").splitlines():
        if line.startswith("data-version: ") and cur is None:
            version = line.split(": ", 1)[1]
        if line == "[Term]":
            cur = {"parents": [], "obsolete": False}
            continue
        if line.startswith("["):
            cur = None
            continue
        if cur is None:
            continue
        if line.startswith("id: "):
            terms[line[4:]] = cur
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("is_a: "):
            cur["parents"].append(line[6:].split(" ! ")[0].strip())
        elif line.startswith("is_obsolete: true"):
            cur["obsolete"] = True
    terms["_version"] = version
    return terms


def ancestors(term: str, obo: dict) -> set[str]:
    out, stack = set(), list(obo.get(term, {}).get("parents", []))
    while stack:
        p = stack.pop()
        if p not in out:
            out.add(p)
            stack.extend(obo.get(p, {}).get("parents", []))
    return out


def load_crosswalk(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or CROSSWALK, dtype=str, keep_default_na=False)


def load_ids(path: str | Path | None = None) -> list[str]:
    ids = [x.strip() for x in Path(path or CATALOG).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {path}")
    return ids


def validate(crosswalk: pd.DataFrame, obo: dict, node_ids: list[str] | None = None) -> list[str]:
    """Problems with the crosswalk (empty list when valid)."""
    problems = []
    for r in crosswalk.itertuples(index=False):
        if r.match not in MATCHES:
            problems.append(f"{r.subphenotype_id}: bad match {r.match!r}")
        if (r.implies == "yes") != (r.match in ("exact", "broader")):
            problems.append(f"{r.subphenotype_id} {r.hpo_id}: implies must be yes exactly for exact/broader")
        if r.match == "unmapped":
            if r.hpo_id:
                problems.append(f"{r.subphenotype_id}: unmapped row with an HPO id")
            continue
        t = obo.get(r.hpo_id)
        if t is None:
            problems.append(f"{r.hpo_id} not in the HPO excerpt")
        elif t["obsolete"]:
            problems.append(f"{r.hpo_id} is obsolete")
        elif t["name"] != r.hpo_label:
            problems.append(f"{r.hpo_id}: label {r.hpo_label!r} != HPO {t['name']!r}")
        if node_ids is not None and r.subphenotype_id not in node_ids:
            problems.append(f"{r.subphenotype_id} is not a subphenotype node")
    return problems


def targets(crosswalk: pd.DataFrame, obo: dict, columns: list[str]) -> dict[str, list[str]]:
    """node -> exported HPO columns (mapped terms plus their in-column ancestors)."""
    colset, out = set(columns), {}
    for r in crosswalk[crosswalk["implies"] == "yes"].itertuples(index=False):
        reach = ({r.hpo_id} | ancestors(r.hpo_id, obo)) & colset
        out.setdefault(r.subphenotype_id, set()).update(reach)
    order = {c: i for i, c in enumerate(columns)}
    return {k: sorted(v, key=order.get) for k, v in out.items()}


def reachable_columns(crosswalk: pd.DataFrame, obo: dict, catalog: list[str]) -> list[str]:
    reach = set()
    for h in crosswalk.loc[crosswalk["implies"] == "yes", "hpo_id"]:
        reach |= {h} | ancestors(h, obo)
    return [c for c in catalog if c in reach]


def to_hpo_matrix(matrix: pd.DataFrame, crosswalk: pd.DataFrame | None = None, obo: dict | None = None,
                  catalog: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subphenotype matrix -> (person x HPO 0/1 matrix, mapping report)."""
    crosswalk = load_crosswalk() if crosswalk is None else crosswalk
    obo = load_obo() if obo is None else obo
    catalog = load_ids() if catalog is None else list(catalog)
    columns = reachable_columns(crosswalk, obo, catalog)
    tg = targets(crosswalk, obo, columns)
    out = np.zeros((len(matrix), len(columns)), dtype=np.int8)
    col = {c: j for j, c in enumerate(columns)}
    for node, hpos in tg.items():
        if node in matrix and hpos:
            pos = matrix[node].to_numpy() > 0
            for h in hpos:
                out[pos, col[h]] = 1
    hpo = pd.DataFrame(out, index=matrix.index, columns=columns)

    rows = []
    for r in crosswalk.itertuples(index=False):
        n = int(matrix[r.subphenotype_id].sum()) if r.subphenotype_id in matrix else 0
        exported = ";".join(tg.get(r.subphenotype_id, [])) if r.implies == "yes" else ""
        if r.implies == "yes" and r.hpo_id not in columns:
            note = (r.note + "; " if r.note else "") + "term not in the phenotype id list"
        else:
            note = r.note
        rows.append(["mapping", r.subphenotype_id, r.hpo_id, r.hpo_label, r.match, r.implies, n, exported, note])
    for node in matrix.columns:
        if node not in set(crosswalk["subphenotype_id"]):
            rows.append(["not_in_crosswalk", node, "", "", "", "no", int(matrix[node].sum()), "", ""])
    for c in columns:
        rows.append(["column", "", c, obo[c]["name"], "", "", int(hpo[c].sum()), c,
                     f"prevalence {hpo[c].mean():.3f}" if len(hpo) else ""])
    return hpo, pd.DataFrame(rows, columns=REPORT_COLUMNS)


def write_davor(hpo: pd.DataFrame, out: str | Path, site: str, phenotype_ids: list[str]) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing = [c for c in hpo.columns if c not in set(phenotype_ids)]
    if missing:
        raise ValueError(f"columns not in the phenotype id list: {missing}")
    with (out / "patient_phenotypes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", *hpo.columns])
        for pid, row in zip(hpo.index, hpo.to_numpy()):
            w.writerow([f"{site}_{pid}", *(int(x) for x in row)])
    (out / "phenotype_ids.txt").write_text("\n".join(phenotype_ids) + "\n", encoding="utf-8")
    return out


def export(build_dir: str | Path, out: str | Path, site: str, phenotype_ids: str | Path | None = None,
           crosswalk: str | Path | None = None, obo: str | Path | None = None) -> dict:
    build_dir, out = Path(build_dir), Path(out)
    matrix = pd.read_parquet(build_dir / "subphenotype_matrix.parquet")
    cw, ob = load_crosswalk(crosswalk), load_obo(obo)
    problems = validate(cw, ob)
    if problems:
        raise ValueError("invalid HPO crosswalk: " + "; ".join(problems))
    ids = load_ids(phenotype_ids)
    hpo, report = to_hpo_matrix(matrix, cw, ob, ids)
    write_davor(hpo, out, site, ids)
    report.to_csv(out / "hpo_mapping_report.csv", index=False)
    return {"matrix": hpo, "report": report, "phenotype_ids": ids, "out": out, "hpo_version": ob["_version"]}
