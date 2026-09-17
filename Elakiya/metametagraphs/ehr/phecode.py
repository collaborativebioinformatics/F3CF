"""PheWAS phecodes as a second, ontology-based subphenotype hierarchy.

Phecode 1.2 (Wu P et al. 2019, JMIR Med Inform 7(4):e14325, PMID 31553307)
    WHO ICD-10 map ``phecode_map_v1_2_icd10_beta.csv`` (ICD10, PHECODE, ...)
    and ``phecode_definitions1.2.csv`` (phecode, phenotype, ...), from
    phewascatalog.org as mirrored in https://github.com/spiros/phemap
    (Apache-2.0). Lipid family: 272 > 272.1 Hyperlipidemia > 272.11
    Hypercholesterolemia (E78.0), 272.12 Hyperglyceridemia (E78.1), 272.13
    Mixed hyperlipidemia (E78.2), 272.14 Hyperchylomicronemia (E78.3).
PhecodeX 1.0 (Shuey MM et al. 2023, Bioinformatics 39(11):btad655, PMID 37930895)
    ``phecodeX_unrolled_ICD_WHO.csv`` (phecode, ICD, vocabulary_id; already
    unrolled) and ``phecodeX_info.csv`` from
    https://github.com/PheWAS/PhecodeXVocabulary (no license file; only an
    excerpt is bundled, full files via :func:`download`). Lipids: EM_239
    Hyperlipidemia > EM_239.1 Hypercholesterolemia > EM_239.11 Pure
    hypercholesterolemia; EM_239.2 Hyperglyceridemia; EM_239.3 Mixed.

Roll-up is by truncating the last decimal (272.11 > 272.1 > 272;
EM_239.11 > EM_239.1 > EM_239). Phecode nodes get ids
``phecode1.2:<code>`` / ``phecodeX:<code>`` and use the same node schema as
``nodes.csv``, so exclusivity, roll-up and the matrix code are shared.

``resources/phecode_crosswalk.csv`` relates hand nodes to phecodes
(equivalent, close, broader); :func:`crosswalk_agreement` reports how many
people each pair shares. Low HDL, Lp(a), apoB and the lab-defined severity
nodes have no phecode counterpart, so the hand table stays the source for the
lipid demo.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.events import RESOURCES, normalise_code
from metametagraphs.ehr.rules import EVIDENCE_COLUMNS

SUBSET = RESOURCES / "phecode_subset"
PREFIX = {"1.2": "phecode1.2:", "X": "phecodeX:"}
FILES = {
    "1.2": ("phecode_definitions1.2.csv", "phecode_map_v1_2_icd10_beta.csv"),
    "X": ("phecodeX_info.csv", "phecodeX_unrolled_ICD_WHO.csv"),
}
URLS = {
    "1.2": ("https://raw.githubusercontent.com/spiros/phemap/master/data/phecode_definitions1.2.csv",
            "https://raw.githubusercontent.com/spiros/phemap/master/data/phecode_map_v1_2_icd10_beta.csv"),
    "X": tuple("https://raw.githubusercontent.com/PheWAS/PhecodeXVocabulary/main/" + urllib.parse.quote(f"PhecodeX (version 1.0)/{f}")
               for f in FILES["X"]),
}


def download(version: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for url, name in zip(URLS[version], FILES[version]):
        if not (dest / name).exists():
            urllib.request.urlretrieve(url, dest / name)
    return dest


def load(version: str = "1.2", src: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(definitions[phecode, name], map[icd10, phecode]) with dot-free ICD10 codes."""
    src = Path(src) if src else SUBSET
    dname, mname = FILES[version]
    if version == "1.2":
        d = pd.read_csv(src / dname, dtype=str, keep_default_na=False).rename(columns={"phenotype": "name"})
        m = pd.read_csv(src / mname, dtype=str, keep_default_na=False).rename(columns={"ICD10": "icd10", "PHECODE": "phecode"})
    elif version == "X":
        d = pd.read_csv(src / dname, dtype=str, keep_default_na=False, encoding="latin-1").rename(columns={"phecode_string": "name"})
        m = pd.read_csv(src / mname, dtype=str, keep_default_na=False, encoding="latin-1").rename(columns={"ICD": "icd10"})
    else:
        raise ValueError(f"unknown phecode version {version!r}")
    m = m[["icd10", "phecode"]].assign(icd10=lambda x: x["icd10"].map(lambda c: normalise_code(c, "icd10")),
                                       phecode=lambda x: x["phecode"].str.strip())
    return d[["phecode", "name"]], m[m["phecode"] != ""].drop_duplicates()


def parent_of(code: str) -> str:
    if "." not in code:
        return ""
    head, dec = code.split(".", 1)
    return head if len(dec) <= 1 else f"{head}.{dec[:-1]}"


def ancestors(code: str) -> list[str]:
    out, p = [], parent_of(code)
    while p:
        out.append(p)
        p = parent_of(p)
    return out


def map_icd10(events: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    """ICD10 diagnosis events -> (event row, phecode); 4-character codes fall back to their 3-character category."""
    icd = events[(events["kind"] == "diagnosis") & (events["code_system"] == "icd10")].reset_index(drop=True)
    if icd.empty:
        return icd.assign(phecode=pd.Series(dtype=str))
    hit = icd.merge(mp, left_on="code", right_on="icd10", how="inner")
    miss = icd[~icd["code"].isin(set(mp["icd10"]))]
    fb = miss.assign(_c3=miss["code"].str.slice(0, 3)).merge(mp, left_on="_c3", right_on="icd10", how="inner")
    return pd.concat([hit, fb.drop(columns="_c3")], ignore_index=True)


def build(events: pd.DataFrame, version: str = "1.2", src: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (phecode nodes in nodes.csv schema, direct evidence) for observed phecodes and their ancestors."""
    defs, mp = load(version, src)
    name = dict(zip(defs["phecode"], defs["name"]))
    pre = PREFIX[version]
    hits = map_icd10(events, mp)
    observed = set(hits["phecode"])
    keep = sorted(observed | {a for p in observed for a in ancestors(p)}, key=lambda c: (c.split(".")[0], c))
    nodes = pd.DataFrame({
        "node_id": [pre + c for c in keep], "label": [f"{c} {name.get(c, '')}".strip() for c in keep],
        "parent": [pre + parent_of(c) if parent_of(c) else "" for c in keep], "exclusive_group": "", "priority": 0.0,
        "logic": "any", "window_days": np.nan, "description": f"Phecode {version} from ICD10",
        "source": "Phecode 1.2 WHO ICD-10 map (Wu et al. 2019)" if version == "1.2" else "PhecodeX 1.0 WHO map (Shuey et al. 2023)",
    })
    ev = pd.DataFrame({
        "person_id": hits["person_id"].to_numpy(), "subphenotype": pre + hits["phecode"], "rule_id": f"phecode{version}_map",
        "rule_type": "code", "source": hits["source"].to_numpy(), "code_system": "icd10", "code": hits["code"].to_numpy(),
        "value": np.nan, "unit": "", "date": hits["date"].to_numpy(), "status": "direct",
    })[EVIDENCE_COLUMNS]
    return nodes, ev


def json_nodes(nodes: pd.DataFrame, levels: dict, version: str) -> list[dict]:
    kids: dict[str, list[str]] = {}
    for n, p in zip(nodes["node_id"], nodes["parent"]):
        kids.setdefault(p, []).append(n)
    return [{"id": r.node_id, "label": r.label, "parent": r.parent or None, "level": levels[r.node_id],
             "children": kids.get(r.node_id, []), "system": f"phecode{version}", "rule_types": ["code"], "logic": "any",
             "window_days": None, "exclusive_group": None, "priority": 0, "description": r.description, "source": r.source,
             "source_codes": [{"rule_type": "code", "system": "icd10", "code": "via phecode map", "match": "map"}],
             "lab_rules": [], "phecode_crosswalk": []} for r in nodes.itertuples(index=False)]


def load_crosswalk(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RESOURCES / "phecode_crosswalk.csv", dtype=str, keep_default_na=False)


def crosswalk_agreement(matrix: pd.DataFrame, crosswalk: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = []
    for r in crosswalk[crosswalk["phecode_version"] == version].itertuples(index=False):
        pcol = PREFIX[version] + r.phecode
        a = matrix[r.node_id] > 0 if r.node_id in matrix else pd.Series(False, index=matrix.index)
        b = matrix[pcol] > 0 if pcol in matrix else pd.Series(False, index=matrix.index)
        both, either = int((a & b).sum()), int((a | b).sum())
        rows.append({"node_id": r.node_id, "phecode": pcol, "relation": r.relation, "n_hand": int(a.sum()),
                     "n_phecode": int(b.sum()), "n_both": both, "jaccard": both / either if either else np.nan})
    return pd.DataFrame(rows)
