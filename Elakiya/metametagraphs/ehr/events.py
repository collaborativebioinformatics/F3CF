"""Common long-format event table that every EHR source is converted into.

One row per clinical fact::

    person_id    int
    source       str   e.g. ukb_gp_clinical, ukb_gp_scripts, ukb_tabular,
                       omop_condition_occurrence, omop_measurement, omop_drug_exposure
    kind         str   diagnosis | lab | medication
    code_system  str   icd10 | read2 | ctv3 | omop | snomed | bnf | atc | drug_name | ukb_field
    code         str   code as recorded (normalised by :func:`normalise_code`)
    analyte      str   lab analyte id (ldl_c, hdl_c, tg, tc, lpa_molar, lpa_mass, apob, hba1c), else ""
    value        float lab value in the analyte's canonical unit, else NaN
    unit         str   canonical unit of ``value``
    raw_value    str   value and unit as recorded
    date         datetime64[ns] (NaT when the source has no date)

Persons are a separate frame: ``person_id`` index with ``sex`` in
{male, female, unknown}.

Lab codes are mapped to analytes and canonical units by
``resources/lab_codes.csv``. Unit conversion (sources in that file):

* cholesterol (total, LDL, HDL) mg/dL x 0.02586 = mmol/L
* triglycerides mg/dL x 0.01129 = mmol/L
* apolipoprotein B mg/dL x 0.01 = g/L
* HbA1c: IFCC (mmol/mol) = (NGSP % - 2.15) x 10.929 (IFCC-NGSP master
  equation, as given in Medscape "Hemoglobin A1c Testing")
* Lp(a) is kept in the unit it was measured in (nmol/L and mg/dL are
  separate analytes) because the conversion depends on apo(a) isoform size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).parent / "resources"

EVENT_COLUMNS = ["person_id", "source", "kind", "code_system", "code", "analyte", "value", "unit", "raw_value", "date"]
SEXES = ("male", "female", "unknown")

# (from_unit, to_unit) -> function, per analyte family
_MG_DL_TO_MMOL_L = {"chol": 0.02586, "tg": 0.01129}


def empty_events() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in EVENT_COLUMNS})
    df["person_id"] = df["person_id"].astype("int64")
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def normalise_code(code: str, system: str) -> str:
    """ICD10: upper case without dot (E78.0 -> E780). Read2/CTV3: trailing dots removed. BNF: dots/spaces removed."""
    code = str(code).strip()
    if system == "icd10":
        return code.replace(".", "").upper()
    if system in ("read2", "ctv3"):
        return code.rstrip(".")
    if system == "bnf":
        return code.replace(".", "").replace(" ", "")
    if system in ("atc",):
        return code.upper()
    if system == "drug_name":
        return code.lower()
    return code


def parse_dates(s: pd.Series) -> pd.Series:
    """Accept yyyymmdd (UKB synthetic), yyyy-mm-dd (OMOP) and dd/mm/yyyy (UKB portal extracts)."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        miss = out.isna()
        if miss.any():
            out[miss] = pd.to_datetime(s[miss].str.slice(0, 10), format=fmt, errors="coerce")
    return out


def load_lab_codes(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RESOURCES / "lab_codes.csv", dtype=str, keep_default_na=False)
    df["norm"] = [normalise_code(c, s) for c, s in zip(df["code"], df["code_system"])]
    return df


def convert(values: pd.Series, units: pd.Series, analyte: str, canonical: str) -> pd.Series:
    """Convert recorded values to the analyte's canonical unit; unconvertible -> NaN."""
    v = pd.to_numeric(values, errors="coerce")
    u = units.fillna("").astype(str).str.strip().str.lower()
    target = canonical.lower()
    out = pd.Series(np.nan, index=v.index)
    same = (u == target) | (u == "")          # empty unit: assume the source's documented unit
    out[same] = v[same]
    if target == "mmol/l" and analyte in ("ldl_c", "hdl_c", "tc"):
        m = u == "mg/dl"
        out[m] = v[m] * _MG_DL_TO_MMOL_L["chol"]
    elif target == "mmol/l" and analyte == "tg":
        m = u == "mg/dl"
        out[m] = v[m] * _MG_DL_TO_MMOL_L["tg"]
    elif target == "g/l" and analyte == "apob":
        m = u == "mg/dl"
        out[m] = v[m] * 0.01
    elif target == "mmol/mol" and analyte == "hba1c":
        m = u == "%"
        out[m] = (v[m] - 2.15) * 10.929
    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, normalise codes, order columns."""
    if df.empty:
        return empty_events()
    df = df.copy()
    for c in EVENT_COLUMNS:
        if c not in df:
            df[c] = np.nan if c == "value" else ""
    df["person_id"] = df["person_id"].astype("int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        d = df["date"]
        df["date"] = parse_dates(d.astype(object).where(d.notna(), "").astype(str).replace({"NaT": "", "nan": ""}))
    for c in ("source", "kind", "code_system", "code", "analyte", "unit", "raw_value"):
        df[c] = df[c].fillna("").astype(str)
    df["code"] = [normalise_code(c, s) for c, s in zip(df["code"], df["code_system"])]
    return df[EVENT_COLUMNS].reset_index(drop=True)


def attach_labs(df: pd.DataFrame, lab_codes: pd.DataFrame | None = None, unit_col: str = "unit") -> pd.DataFrame:
    """For rows whose (code_system, code) is a known lab code, set kind/analyte/value/unit.

    ``df`` must carry ``value`` (as recorded) and ``unit_col`` (recorded unit,
    may be empty). Rows of lab codes whose value cannot be converted are
    dropped and counted in ``df.attrs['dropped_lab_rows']``.
    """
    lab_codes = load_lab_codes() if lab_codes is None else lab_codes
    df = df.copy()
    norm = [normalise_code(c, s) for c, s in zip(df["code"], df["code_system"])]
    key = pd.Series([f"{s}|{n}" for s, n in zip(df["code_system"], norm)], index=df.index)
    lut = {f"{s}|{n}": (a, u, du) for s, n, a, u, du in zip(lab_codes["code_system"], lab_codes["norm"], lab_codes["analyte"],
                                                             lab_codes["canonical_unit"], lab_codes["default_unit"])}
    hit = key.map(lambda k: k in lut).astype(bool)
    recorded = df["value"].astype(object)
    numeric = pd.to_numeric(recorded, errors="coerce").astype(float)
    for c in ("raw_value", "analyte", "unit", "kind"):
        df[c] = df[c].astype(object) if c in df else ""
    dropped = 0
    if hit.any():
        for k, (analyte, canon, default_unit) in lut.items():
            m = hit & (key == k)
            if not m.any():
                continue
            rec_unit = df.loc[m, unit_col].fillna("").astype(str)
            rec_unit = rec_unit.where(rec_unit.str.strip() != "", default_unit)
            df.loc[m, "raw_value"] = recorded[m].astype(str) + " " + rec_unit
            numeric[m] = convert(recorded[m], rec_unit, analyte, canon)
            df.loc[m, "analyte"] = analyte
            df.loc[m, "unit"] = canon
            df.loc[m, "kind"] = "lab"
    df["value"] = numeric
    if hit.any():
        bad = hit & numeric.isna()
        dropped = int(bad.sum())
        df = df[~bad]
    df.attrs["dropped_lab_rows"] = dropped
    return df
