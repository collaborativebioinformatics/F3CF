"""OMOP CDM sources -> common event table.

Reads ``person``, ``condition_occurrence``, ``measurement`` and (optionally)
``drug_exposure`` from a directory of CSV or parquet files, with any column
case (Eunomia GiBleed 5.3 uses upper case, Synthea27Nj 5.4 lower case).
If ``concept`` (and ``concept_ancestor``) are present they are used to

* name drugs (``drug_name`` events from ``concept_name``),
* map drugs to ATC classes (``atc`` events for every ATC ancestor's
  ``concept_code``, e.g. C10AA),
* tell SNOMED and ICD10 source values apart (``condition_source_concept_id``
  vocabulary). Without a vocabulary, a ``condition_source_value`` that looks
  like ICD10 (letter + two digits) is an ``icd10`` event and an all-digit one
  is a ``snomed`` event. ICD10CM codes are treated as ICD10 (prefix rules
  such as E78.0 also match E78.00).

Table and field names follow the OMOP CDM v5.4 field-level specification
(https://github.com/OHDSI/CommonDataModel, inst/csv/OMOP_CDMv5.4_Field_Level.csv).
Concept ids used below were checked in the Eunomia Synthea27Nj CONCEPT table
and data (https://github.com/OHDSI/EunomiaDatasets):

* gender: 8507 (source value M), 8532 (source value F)
* units: 8840 mg/dL, 8753 mmol/L, 9579 mmol/mol, 8554 %, 8736 nmol/L, 8636 g/L, 8751 mg/L
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.events import attach_labs, empty_events, finalize, load_lab_codes

GENDER = {"8507": "male", "8532": "female"}
UNITS = {"8840": "mg/dL", "8753": "mmol/L", "9579": "mmol/mol", "8554": "%", "8736": "nmol/L", "8636": "g/L", "8751": "mg/L"}
_ICD10 = re.compile(r"^[A-Z]\d\d")
_SNOMED = re.compile(r"^\d{6,18}$")
VOCAB_SYSTEM = {"SNOMED": "snomed", "ICD10": "icd10", "ICD10CM": "icd10", "Read": "read2"}


def find_table(root: Path, table: str, required: bool = True) -> Path | None:
    for p in sorted(Path(root).iterdir()):
        if p.is_file() and p.stem.lower() == table and p.suffix.lower() in {".csv", ".parquet", ".tsv"}:
            return p
    if required:
        raise FileNotFoundError(f"no {table}.csv / .parquet in {root}")
    return None


def read_table(path: Path, columns: list[str], optional: list[str] = ()) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path).astype(str)
    else:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing OMOP columns {missing}")
    for c in optional:
        if c not in df.columns:
            df[c] = ""
    df = df[list(columns) + list(optional)].copy()
    return df.replace({"nan": "", "None": "", "NaT": ""})


def load_vocabulary(root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cp = find_table(root, "concept", required=False)
    ap = find_table(root, "concept_ancestor", required=False)
    concept = read_table(cp, ["concept_id", "concept_name", "vocabulary_id", "concept_code"]) if cp else None
    ancestor = read_table(ap, ["ancestor_concept_id", "descendant_concept_id"]) if ap else None
    return concept, ancestor


def persons(root: Path, subset: int | None = None) -> pd.DataFrame:
    df = read_table(find_table(root, "person"), ["person_id", "gender_concept_id"], ["gender_source_value"])
    df["person_id"] = df["person_id"].astype(int)
    df = df.sort_values("person_id")
    df = df.head(subset) if subset else df
    sex = df["gender_concept_id"].map(GENDER)
    src = df["gender_source_value"].str.upper().map({"M": "male", "F": "female"})
    return pd.DataFrame({"sex": sex.fillna(src).fillna("unknown").to_numpy()}, index=pd.Index(df["person_id"], name="person_id"))


def condition_events(root: Path, concept: pd.DataFrame | None) -> pd.DataFrame:
    c = read_table(find_table(root, "condition_occurrence"), ["person_id", "condition_concept_id", "condition_start_date"],
                   ["condition_source_value", "condition_source_concept_id"])
    base = dict(person_id=c["person_id"].astype(int).to_numpy(), source="omop_condition_occurrence", kind="diagnosis",
                value=np.nan, unit="", raw_value="", date=c["condition_start_date"].to_numpy())
    a = pd.DataFrame({**base, "code_system": "omop", "code": c["condition_concept_id"].to_numpy()})
    src = c["condition_source_value"].str.strip()
    if concept is not None:
        vocab = c["condition_source_concept_id"].map(dict(zip(concept["concept_id"], concept["vocabulary_id"]))).map(VOCAB_SYSTEM)
    else:
        vocab = pd.Series(pd.NA, index=c.index, dtype="object")
    guess = np.where(src.str.match(_ICD10), "icd10", np.where(src.str.match(_SNOMED), "snomed", ""))
    system = vocab.fillna(pd.Series(guess, index=c.index)).fillna("")
    keep = (src != "") & (system != "")
    b = pd.DataFrame({k: (v[keep.to_numpy()] if isinstance(v, np.ndarray) else v) for k, v in base.items()})
    b["code_system"] = system[keep].to_numpy()
    b["code"] = src[keep].to_numpy()
    return pd.concat([a, b], ignore_index=True)


def measurement_events(root: Path, lab_codes: pd.DataFrame) -> pd.DataFrame:
    m = read_table(find_table(root, "measurement"), ["person_id", "measurement_concept_id", "measurement_date"],
                   ["value_as_number", "unit_concept_id", "unit_source_value"])
    known = set(lab_codes.loc[lab_codes["code_system"] == "omop", "norm"])
    m = m[m["measurement_concept_id"].isin(known)]
    unit = m["unit_concept_id"].map(UNITS).fillna(m["unit_source_value"])
    df = pd.DataFrame({"person_id": m["person_id"].astype(int).to_numpy(), "source": "omop_measurement", "kind": "lab",
                       "code_system": "omop", "code": m["measurement_concept_id"].to_numpy(),
                       "value": m["value_as_number"].to_numpy(), "unit": unit.to_numpy(), "raw_value": "",
                       "date": m["measurement_date"].to_numpy()})
    out = attach_labs(df, lab_codes)
    return out


def drug_events(root: Path, concept: pd.DataFrame | None, ancestor: pd.DataFrame | None) -> pd.DataFrame:
    path = find_table(root, "drug_exposure", required=False)
    if path is None:
        return empty_events()
    d = read_table(path, ["person_id", "drug_concept_id", "drug_exposure_start_date"], ["drug_source_value"])
    base = dict(source="omop_drug_exposure", kind="medication", value=np.nan, unit="")
    frames = [pd.DataFrame({**base, "person_id": d["person_id"].astype(int), "code_system": "omop",
                            "code": d["drug_concept_id"], "raw_value": d["drug_source_value"], "date": d["drug_exposure_start_date"]})]
    names = d["drug_source_value"].where(~d["drug_source_value"].str.fullmatch(r"\d*"), "")
    if concept is not None:
        cname = d["drug_concept_id"].map(dict(zip(concept["concept_id"], concept["concept_name"])))
        names = cname.fillna(names)
    m = names.fillna("") != ""
    frames.append(pd.DataFrame({**base, "person_id": d.loc[m, "person_id"].astype(int), "code_system": "drug_name",
                                "code": names[m], "raw_value": names[m], "date": d.loc[m, "drug_exposure_start_date"]}))
    if concept is not None and ancestor is not None and len(ancestor):
        atc = concept[concept["vocabulary_id"] == "ATC"][["concept_id", "concept_code"]]
        link = ancestor.merge(atc, left_on="ancestor_concept_id", right_on="concept_id")[["descendant_concept_id", "concept_code"]]
        j = d.merge(link, left_on="drug_concept_id", right_on="descendant_concept_id")
        if len(j):
            frames.append(pd.DataFrame({**base, "person_id": j["person_id"].astype(int), "code_system": "atc",
                                        "code": j["concept_code"], "raw_value": j["drug_source_value"], "date": j["drug_exposure_start_date"]}))
    return pd.concat(frames, ignore_index=True)


def load(root: str | Path, subset: int | None = None, lab_codes: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (persons, events, info) for an OMOP CDM directory."""
    root = Path(root)
    lab_codes = load_lab_codes() if lab_codes is None else lab_codes
    ppl = persons(root, subset)
    concept, ancestor = load_vocabulary(root)
    meas = measurement_events(root, lab_codes)
    parts = [condition_events(root, concept), meas, drug_events(root, concept, ancestor)]
    ev = finalize(pd.concat([p for p in parts if len(p)], ignore_index=True))
    ev = ev[ev["person_id"].isin(ppl.index)].reset_index(drop=True)
    info = {"cdm_dir": str(root), "vocabulary": concept is not None, "concept_ancestor_rows": 0 if ancestor is None else len(ancestor),
            "dropped_lab_rows": int(meas.attrs.get("dropped_lab_rows", 0))}
    return ppl, ev, info
