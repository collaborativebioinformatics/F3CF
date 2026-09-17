"""UK Biobank EHR sources -> common event table.

Supported inputs (any subset may be present under one directory):

Tabular participant data (``*.tsv``, searched recursively under ``tabular/`` or the top level)
    Tab separated with a header ``EID`` (or ``eid``) followed by
    ``FieldID-Instance.Array`` columns, as in the UKB synthetic dataset
    (https://biobank.ndph.ox.ac.uk/synthetic_dataset/). Fields used:

    ======  ==================================================  ==================
    field   UKB Showcase title                                  use
    ======  ==================================================  ==================
    31      Sex (data-coding 9: 0 = Female, 1 = Male)            person sex
    41270   Diagnoses - ICD10 (data-coding 19)                   diagnosis events
    30780   LDL direct (mmol/L)                                  lab
    30760   HDL cholesterol (mmol/L)                             lab
    30870   Triglycerides (mmol/L)                               lab
    30690   Cholesterol (mmol/L)                                 lab
    30790   Lipoprotein A (nmol/L)                               lab
    30640   Apolipoprotein B (g/L)                               lab
    30750   Glycated haemoglobin (HbA1c) (mmol/mol)              lab
    6177    Medication for cholesterol, blood pressure or        medication, men
            diabetes (data-coding 100625)
    6153    same question, women                                 medication, women
    ======  ==================================================  ==================

    Field titles and units were checked on the UKB Showcase. Tabular rows
    carry no event date (NaT); assessment-centre dates are not joined.

GP clinical records (``medrec/set3*.txt`` synthetic layout, or ``gp_clinical*``)
    ``eid, data_provider, event_dt, read_2, read_3, value1, value2, value3``.
    The synthetic files have no header; portal extracts do (detected).
    ``read_2`` becomes a ``read2`` event and ``read_3`` a ``ctv3`` event;
    ``value1`` is the recorded value. No unit column is documented, so lab
    values are taken in the unit given by ``lab_codes.csv`` (mmol/L for
    lipids). ``event_dt`` may be yyyymmdd or dd/mm/yyyy.

GP prescriptions (``gp_scripts*``)
    ``eid, data_provider, issue_date, read_2, bnf_code, dmd_code, drug_name,
    quantity`` (column list from UKB Showcase field 42039). Each row gives
    ``bnf`` (dots removed), ``drug_name`` (lower case) and, when present,
    ``read2`` medication events. The exact ``bnf_code`` formatting in the
    real table was not checked; codes are normalised by removing dots and
    spaces before prefix matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metametagraphs.ehr.events import attach_labs, empty_events, finalize, load_lab_codes, parse_dates

GP_CLINICAL_COLUMNS = ["eid", "data_provider", "event_dt", "read_2", "read_3", "value1", "value2", "value3"]
GP_SCRIPTS_COLUMNS = ["eid", "data_provider", "issue_date", "read_2", "bnf_code", "dmd_code", "drug_name", "quantity"]
FIELD_SEX = 31
FIELD_ICD10 = 41270
LAB_FIELDS = [30780, 30760, 30870, 30690, 30790, 30640, 30750]
MED_FIELDS = [6177, 6153]


def _field(col: str) -> int | None:
    head = str(col).split("-", 1)[0]
    return int(head) if head.isdigit() else None


# --------------------------------------------------------------------------
# Tabular
# --------------------------------------------------------------------------
def tabular_files(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    base = root / "tabular" if (root / "tabular").is_dir() else root
    return sorted(p for p in base.glob("*.tsv"))


def read_tabular(root: str | Path, fields: list[int], subset: int | None = None) -> pd.DataFrame:
    """Columns of ``fields`` (all instances/arrays) from every TSV that has them, indexed by eid."""
    wanted, parts = set(fields), []
    for path in tabular_files(Path(root)):
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        if not header or header[0].lower() != "eid":
            continue
        cols = [c for c in header[1:] if _field(c) in wanted]
        if not cols:
            continue
        df = pd.read_csv(path, sep="\t", usecols=[header[0]] + cols, nrows=subset, dtype=str, keep_default_na=False)
        parts.append(df.rename(columns={header[0]: "eid"}).assign(eid=lambda d: d["eid"].astype(int)).set_index("eid"))
    if not parts:
        return pd.DataFrame(index=pd.Index([], name="eid", dtype="int64"))
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    return out.fillna("")


def persons_from_tabular(tab: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in tab.columns if _field(c) == FIELD_SEX]
    sex = tab[cols[0]].map({"0": "female", "1": "male"}) if cols else pd.Series("unknown", index=tab.index)
    return pd.DataFrame({"sex": sex.fillna("unknown")}, index=pd.Index(tab.index, name="person_id"))


def tabular_events(tab: pd.DataFrame, lab_codes: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for col in tab.columns:
        f = _field(col)
        vals = tab[col]
        vals = vals[vals.astype(str).str.strip() != ""]
        if vals.empty:
            continue
        if f == FIELD_ICD10:
            rows.append(pd.DataFrame({"person_id": vals.index, "source": "ukb_tabular", "kind": "diagnosis",
                                      "code_system": "icd10", "code": vals.to_numpy(), "value": np.nan, "unit": "",
                                      "raw_value": "", "date": pd.NaT}))
        elif f in MED_FIELDS:
            rows.append(pd.DataFrame({"person_id": vals.index, "source": "ukb_tabular", "kind": "medication",
                                      "code_system": "ukb_field", "code": [f"{f}={v}" for v in vals], "value": np.nan,
                                      "unit": "", "raw_value": vals.to_numpy(), "date": pd.NaT}))
        elif f in LAB_FIELDS and col.endswith(".0"):
            rows.append(pd.DataFrame({"person_id": vals.index, "source": "ukb_tabular", "kind": "lab",
                                      "code_system": "ukb_field", "code": str(f), "value": vals.to_numpy(), "unit": "",
                                      "raw_value": "", "date": pd.NaT, "instance": col.split("-")[1]}))
    if not rows:
        return empty_events()
    df = pd.concat(rows, ignore_index=True)
    labs = df["kind"] == "lab"
    df = pd.concat([df[~labs], attach_labs(df[labs], lab_codes)], ignore_index=True)
    return finalize(df)


# --------------------------------------------------------------------------
# GP records
# --------------------------------------------------------------------------
def _read_headed_or_not(path: Path, columns: list[str]) -> pd.DataFrame:
    first = path.open().readline().rstrip("\n").split("\t")
    has_header = first and first[0].strip().lower() == "eid"
    df = pd.read_csv(path, sep="\t", header=0 if has_header else None, names=None if has_header else columns,
                     dtype=str, keep_default_na=False, quoting=3)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[columns]


def gp_clinical_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(set(root.glob("medrec/set3*.txt")) | set(root.glob("gp_clinical*")) | set(root.glob("medrec/gp_clinical*")))


def gp_scripts_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(set(root.glob("gp_scripts*")) | set(root.glob("medrec/gp_scripts*")))


def gp_clinical_events(paths: list[Path], lab_codes: pd.DataFrame | None = None, eids=None) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = _read_headed_or_not(Path(p), GP_CLINICAL_COLUMNS)
        df = df[pd.to_numeric(df["eid"], errors="coerce").notna()]
        if eids is not None:
            df = df[df["eid"].astype(int).isin(set(eids))]
        date = parse_dates(df["event_dt"])
        for col, system in (("read_2", "read2"), ("read_3", "ctv3")):
            m = df[col].str.strip() != ""
            frames.append(pd.DataFrame({"person_id": df.loc[m, "eid"].astype(int), "source": "ukb_gp_clinical",
                                        "kind": "diagnosis", "code_system": system, "code": df.loc[m, col].str.strip(),
                                        "value": df.loc[m, "value1"].str.strip(), "unit": "", "raw_value": "",
                                        "date": date[m]}))
    if not frames:
        return empty_events()
    ev = pd.concat(frames, ignore_index=True)
    ev = attach_labs(ev, lab_codes)
    dropped = ev.attrs.get("dropped_lab_rows", 0)
    ev.loc[ev["kind"] != "lab", "value"] = np.nan
    out = finalize(ev)
    out.attrs["dropped_lab_rows"] = dropped
    return out


def gp_scripts_events(paths: list[Path], eids=None) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = _read_headed_or_not(Path(p), GP_SCRIPTS_COLUMNS)
        df = df[pd.to_numeric(df["eid"], errors="coerce").notna()]
        if eids is not None:
            df = df[df["eid"].astype(int).isin(set(eids))]
        date = parse_dates(df["issue_date"])
        for col, system in (("bnf_code", "bnf"), ("drug_name", "drug_name"), ("read_2", "read2"), ("dmd_code", "dmd")):
            m = df[col].str.strip() != ""
            frames.append(pd.DataFrame({"person_id": df.loc[m, "eid"].astype(int), "source": "ukb_gp_scripts",
                                        "kind": "medication", "code_system": system, "code": df.loc[m, col].str.strip(),
                                        "value": np.nan, "unit": "", "raw_value": df.loc[m, "drug_name"], "date": date[m]}))
    return finalize(pd.concat(frames, ignore_index=True)) if frames else empty_events()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def load(root: str | Path, subset: int | None = None, lab_codes: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (persons, events, info) for a UKB-style directory."""
    root = Path(root)
    lab_codes = load_lab_codes() if lab_codes is None else lab_codes
    tab = read_tabular(root, [FIELD_SEX, FIELD_ICD10] + LAB_FIELDS + MED_FIELDS, subset=subset)
    persons = persons_from_tabular(tab)
    eids = persons.index if len(persons) else None
    clin_files, script_files = gp_clinical_files(root), gp_scripts_files(root)
    parts = [tabular_events(tab, lab_codes), gp_clinical_events(clin_files, lab_codes, eids), gp_scripts_events(script_files, eids)]
    events = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else empty_events()
    if eids is None:   # no tabular data: persons are whoever has records
        persons = pd.DataFrame({"sex": "unknown"}, index=pd.Index(sorted(events["person_id"].unique()), name="person_id"))
    info = {"tabular_files": [str(p) for p in tabular_files(root)], "gp_clinical_files": [str(p) for p in clin_files],
            "gp_scripts_files": [str(p) for p in script_files],
            "dropped_lab_rows": int(parts[1].attrs.get("dropped_lab_rows", 0))}
    return persons, finalize(events), info
