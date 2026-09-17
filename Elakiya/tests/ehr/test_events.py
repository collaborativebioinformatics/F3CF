import numpy as np
import pandas as pd

from metametagraphs.ehr import events as E


def test_normalise_code():
    assert E.normalise_code("E78.0", "icd10") == "E780"
    assert E.normalise_code("e78.0", "icd10") == "E780"
    assert E.normalise_code("C10F.", "read2") == "C10F"
    assert E.normalise_code("H33..", "ctv3") == "H33"
    assert E.normalise_code("02.12.00.00", "bnf") == "02120000"
    assert E.normalise_code("Simvastatin 40mg", "drug_name") == "simvastatin 40mg"


def test_parse_dates_accepts_three_formats():
    d = E.parse_dates(pd.Series(["20100131", "2011-02-28", "15/03/2012", "", "junk"]))
    assert list(d[:3].dt.strftime("%Y-%m-%d")) == ["2010-01-31", "2011-02-28", "2012-03-15"]
    assert d[3:].isna().all()


def test_unit_conversion():
    v = pd.Series([100.0, 100.0, 100.0, 7.0, 5.0])
    assert abs(E.convert(v[:1], pd.Series(["mg/dL"]), "ldl_c", "mmol/L")[0] - 2.586) < 1e-9
    assert abs(E.convert(v[:1], pd.Series(["mg/dL"]), "tg", "mmol/L")[0] - 1.129) < 1e-9
    assert abs(E.convert(v[:1], pd.Series(["mg/dL"]), "apob", "g/L")[0] - 1.0) < 1e-9
    assert abs(E.convert(pd.Series([7.0]), pd.Series(["%"]), "hba1c", "mmol/mol")[0] - (7.0 - 2.15) * 10.929) < 1e-9
    assert E.convert(pd.Series([5.0]), pd.Series(["mmol/L"]), "ldl_c", "mmol/L")[0] == 5.0
    assert np.isnan(E.convert(pd.Series([50.0]), pd.Series(["mg/dL"]), "lpa_molar", "nmol/L")[0])   # Lp(a) never converted


def test_attach_labs_maps_and_drops_unconvertible():
    df = pd.DataFrame({"person_id": [1, 2, 3, 4], "kind": "diagnosis", "code_system": ["omop", "omop", "omop", "icd10"],
                       "code": ["3009966", "3009966", "3009966", "E780"], "value": ["200", "5.2", "7", ""],
                       "unit": ["mg/dL", "mmol/L", "furlongs", ""], "raw_value": "", "analyte": "", "source": "t", "date": ""})
    out = E.attach_labs(df)
    assert out.attrs["dropped_lab_rows"] == 1
    labs = out[out["kind"] == "lab"].set_index("person_id")
    assert abs(labs.loc[1, "value"] - 200 * 0.02586) < 1e-9 and labs.loc[2, "value"] == 5.2
    assert set(labs["analyte"]) == {"ldl_c"} and set(labs["unit"]) == {"mmol/L"}
    assert (out["kind"] == "diagnosis").sum() == 1
