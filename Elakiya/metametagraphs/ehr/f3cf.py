"""Export build outputs as F3CF per-site relations.

F3CF (see the repository README) factorises, per clinic, a Patient x
Phenotype and a Patient x Drug relation into a shared latent space. This
module turns one ``build`` output directory into those two edge lists, in
the same long ``(row, column, score, source)`` shape as the team's GWAS edge
files (``gene, phenotype, score, source``):

``patient_phenotype.csv``: ``patient, phenotype, score, source``
    one row per positive cell of ``subphenotype_matrix.parquet``;
    ``phenotype`` is the subphenotype id (hand node or ``phecode1.2:...``),
    ``score`` is 1, ``source`` is the site name.
``patient_drug.csv``: ``patient, drug, score, source``
    from medication evidence in ``subphenotype_evidence.parquet``.
    ``drug`` is a normalised label:

    * ingredient name when the drug name matches an ATC C10AA statin
      (``simvastatin``), otherwise the lower-cased first word of the name;
    * ``atc:<code>`` for ATC classes, ``bnf:<first 4 digits>`` for BNF
      sections (``bnf:0212`` = lipid-regulating drugs);
    * ``self_report:<field>=<answer>`` for UKB touchscreen answers;
    * ``omop:<concept_id>`` only when the source had no drug names.

    ``score`` is the number of distinct dates on which the drug was
    recorded (undated records count once each), ``source`` the site name.

Only medications that matched a medication rule (currently lipid-lowering
therapy and self-reported blood pressure medication) are exported, since the
build keeps evidence for rule hits only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PHENOTYPE_COLUMNS = ["patient", "phenotype", "score", "source"]
DRUG_COLUMNS = ["patient", "drug", "score", "source"]
STATIN = re.compile(r"\b((?:simva|lova|prava|fluva|atorva|ceriva|rosuva|pitava)statin)\b", re.IGNORECASE)


def drug_label(code_system: str, code: str) -> str:
    code = str(code).strip()
    if code_system == "drug_name":
        m = STATIN.search(code)
        return m.group(1).lower() if m else (code.lower().split() or [""])[0]
    if code_system == "atc":
        return f"atc:{code.upper()}"
    if code_system == "bnf":
        return f"bnf:{code[:4]}"
    if code_system == "ukb_field":
        return f"self_report:{code}"
    return f"{code_system}:{code}"


def patient_phenotype(matrix: pd.DataFrame, site: str) -> pd.DataFrame:
    long = matrix.stack()
    long = long[long > 0].reset_index()
    long.columns = ["patient", "phenotype", "score"]
    long["score"] = 1
    long["source"] = site
    return long[PHENOTYPE_COLUMNS]


def patient_drug(evidence: pd.DataFrame, site: str) -> pd.DataFrame:
    med = evidence[(evidence["rule_type"] == "med") & (evidence["status"] == "direct")].copy()
    if med.empty:
        return pd.DataFrame(columns=DRUG_COLUMNS)
    named = set(med.loc[med["code_system"] == "drug_name", "person_id"])
    med = med[~((med["code_system"] == "omop") & med["person_id"].isin(named))]
    med["drug"] = [drug_label(s, c) for s, c in zip(med["code_system"], med["code"])]
    med = med[med["drug"] != ""].drop_duplicates(["person_id", "drug", "date", "code"])
    key = med["date"].astype("string").fillna(pd.Series(med.index.astype(str), index=med.index))
    out = key.groupby([med["person_id"], med["drug"]]).nunique().reset_index()
    out.columns = ["patient", "drug", "score"]
    out["source"] = site
    return out[DRUG_COLUMNS]


def export(build_dir: str | Path, out: str | Path, site: str) -> dict:
    build_dir, out = Path(build_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(build_dir / "subphenotype_matrix.parquet")
    evidence = pd.read_parquet(build_dir / "subphenotype_evidence.parquet")
    pp, pd_ = patient_phenotype(matrix, site), patient_drug(evidence, site)
    pp.to_csv(out / "patient_phenotype.csv", index=False)
    pd_.to_csv(out / "patient_drug.csv", index=False)
    return {"patient_phenotype": pp, "patient_drug": pd_, "out": out}
