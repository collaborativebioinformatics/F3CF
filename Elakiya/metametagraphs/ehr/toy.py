"""Toy cohort written in both UKB and OMOP CDM layouts from one latent truth.

``write_toy(dir, n, seed)`` creates::

    dir/ukb/tabular/{demographics,diagnoses,biochemistry,medication}.tsv
    dir/ukb/medrec/set3a1.txt        GP clinical records (synthetic-dataset layout, no header)
    dir/ukb/gp_scripts.txt           GP prescriptions (portal layout, with header)
    dir/omop/{person,condition_occurrence,measurement,drug_exposure,concept,concept_ancestor}.csv
    dir/toy_truth.csv                latent states per person

The cohort is built to exercise every rule type:

* two lipid panels per person with dates; treated people have a lower
  second LDL-C, so ``worst`` and ``most_recent`` aggregation differ;
* HDL-C values between the male (1.03) and female (1.29 mmol/L) cutoffs, so
  sex-specific rules matter;
* some people with one GP dyslipidaemia code only, some with two dates,
  some with one code plus statins (node logic ``code>=2 | code>=1 & med>=1``);
* some people with a single statin prescription (``min_count`` 2);
* LDL-C values in the DLCN possible and probable ranges (exclusive siblings);
* CTV3-only and ICD10-only diagnoses, SNOMED and ICD10 OMOP source values.

OMOP export: real concept ids for conditions, lipid and HbA1c measurements
(mg/dL and %, converted back by the reader) and three statin clinical drugs.
Lp(a) and apoB are UKB-only here because their OMOP concept ids could not be
verified. ``concept``/``concept_ancestor`` include toy-only ATC concepts
(ids >= 2,100,000,000, never real OMOP ids) so the ATC path can be tested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TOY_ATC_IDS = {"C10": 2100000001, "C10A": 2100000002, "C10AA": 2100000003}
STATINS = {  # OMOP concept id -> (name, BNF-style code as written in the toy gp_scripts)
    "1539463": ("simvastatin 10 MG Oral Tablet", "02.12.00.00"),
    "1539411": ("simvastatin 20 MG Oral Tablet", "02.12.00.00"),
    "1545959": ("atorvastatin 80 MG Oral Tablet", "02.12.00.00"),
}


def _dates(rng, n, start="2008-01-01", span_days=2500):
    return pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, span_days, n), unit="D")


def simulate(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"person_id": np.arange(1_000_001, 1_000_001 + n)})
    u = lambda: rng.random(n)  # noqa: E731
    t["sex"] = np.where(u() < 0.5, "male", "female")
    t["dys"] = u() < 0.5
    ldl_state = np.where(~t["dys"], "normal", rng.choice(["normal", "high", "severe", "dlcn_possible", "dlcn_probable"], n,
                                                          p=[0.2, 0.35, 0.15, 0.2, 0.1]))
    t["ldl_state"] = ldl_state
    base = {"normal": (3.0, 0.4), "high": (4.5, 0.2), "severe": (4.95, 0.02), "dlcn_possible": (6.0, 0.6), "dlcn_probable": (9.2, 0.4)}
    t["ldl1"] = [max(1.0, rng.normal(*base[s])) for s in ldl_state]
    t["treated"] = t["dys"] & (ldl_state != "normal") & (u() < 0.6)
    t["n_scripts"] = np.where(t["treated"], rng.choice([1, 2, 3, 4], n, p=[0.25, 0.25, 0.25, 0.25]), 0)
    t["ldl2"] = np.where(t["treated"], t["ldl1"] * 0.55, t["ldl1"] + rng.normal(0, 0.1, n))
    t["low_hdl"] = t["dys"] & (u() < 0.4)
    # half of the low-HDL people sit between the male and female cutoffs
    between = u() < 0.5
    t["hdl"] = np.where(t["low_hdl"], np.where(between, rng.uniform(1.06, 1.26, n), rng.uniform(0.7, 1.0, n)),
                        rng.uniform(1.35, 2.0, n))
    t["high_tg"] = t["dys"] & (u() < 0.35)
    t["very_high_tg"] = t["high_tg"] & (u() < 0.3)
    t["tg"] = np.where(t["very_high_tg"], rng.uniform(6.0, 9.0, n), np.where(t["high_tg"], rng.uniform(2.4, 4.5, n), rng.uniform(0.6, 1.9, n)))
    t["mixed"] = t["high_tg"] & (ldl_state != "normal") & (u() < 0.5)
    t["lpa"] = np.where(t["dys"] & (u() < 0.25), rng.uniform(110, 189, n), rng.uniform(4, 90, n))
    t["apob"] = np.clip(t["ldl1"] * 0.26 + rng.normal(0, 0.05, n), 0.4, None)
    t["tc"] = t["ldl1"] + t["hdl"] + t["tg"] / 2.2
    # coding pattern for dyslipidaemia in GP records: 0, 1 or 2 dated codes
    t["dys_gp_codes"] = np.where(t["dys"], rng.choice([0, 1, 2], n, p=[0.3, 0.35, 0.35]), 0)
    t["dys_hosp_code"] = t["dys"] & (u() < 0.3)
    t["t2d"] = u() < 0.2
    t["t2d_renal"] = t["t2d"] & (u() < 0.3)
    t["t2d_ophth"] = t["t2d"] & (u() < 0.25)
    t["hba1c1"] = np.where(t["t2d"], rng.uniform(60, 95, n), rng.uniform(30, 41, n))
    t["hba1c2"] = np.where(t["t2d"], rng.uniform(50, 90, n), rng.uniform(30, 41, n))
    t["htn"] = u() < 0.3
    t["htn_secondary"] = t["htn"] & (u() < 0.2)
    t["htn_organ"] = t["htn"] & (u() < 0.2)
    t["bp_med"] = t["htn"] & (u() < 0.6)
    t["asthma"] = u() < 0.15
    t["asthma_type"] = np.where(t["asthma"], rng.choice(["allergic", "non_allergic", "unspecified"], n, p=[0.5, 0.3, 0.2]), "")
    t["asthma_severe"] = t["asthma"] & (u() < 0.2)
    t["ctv3_only"] = u() < 0.25        # GP system codes only CTV3 for these people
    t["date1"] = _dates(rng, n)
    t["date2"] = t["date1"] + pd.to_timedelta(rng.integers(200, 900, n), unit="D")
    return t


def _gp_rows(t: pd.DataFrame, rng) -> list[tuple]:
    rows = []
    fmt = lambda d: d.strftime("%Y%m%d")  # noqa: E731

    def dx(r, read2, ctv3, date):
        if r.ctv3_only and ctv3:
            rows.append((r.person_id, 3, fmt(date), "", ctv3, "", "", ""))
        elif read2:
            rows.append((r.person_id, 1, fmt(date), read2, "", "", "", ""))

    for r in t.itertuples(index=False):
        for k in range(r.dys_gp_codes):
            code = "C320." if r.ldl_state != "normal" and not r.mixed else ("C322." if r.mixed else "C32..")
            dx(r, code, "", r.date1 if k == 0 else r.date2)
        for d, ldl, a1c in ((r.date1, r.ldl1, r.hba1c1), (r.date2, r.ldl2, r.hba1c2)):
            rows.append((r.person_id, 1, fmt(d), "44P6.", "", f"{ldl:.2f}", "", ""))
            rows.append((r.person_id, 1, fmt(d), "44P5.", "", f"{r.hdl:.2f}", "", ""))
            rows.append((r.person_id, 1, fmt(d), "44Q..", "", f"{r.tg:.2f}", "", ""))
            rows.append((r.person_id, 1, fmt(d), "44PJ.", "", f"{r.tc:.2f}", "", ""))
            if r.t2d:
                rows.append((r.person_id, 1, fmt(d), "42W5.", "", f"{a1c:.0f}", "", ""))
        if r.t2d:
            dx(r, "C10F.", "X40J5", r.date1)
            if r.t2d_renal:
                dx(r, "C10FC", "C1090", r.date2)
            if r.t2d_ophth:
                dx(r, "C10F6", "C1091", r.date2)
        if r.htn:
            dx(r, "G24.." if r.htn_secondary else "G20..", "XE0Ub" if r.htn_secondary else "XE0Uc", r.date1)
        if r.asthma:
            code = {"allergic": "H330.", "non_allergic": "H331.", "unspecified": "H33.."}[r.asthma_type]
            dx(r, code, "H33..", r.date1)
            if r.asthma_severe:
                dx(r, "H333.", "", r.date2)
    return rows


def _icd10(r) -> list[str]:
    out = []
    if r.dys_hosp_code:
        out.append("E782" if r.mixed else ("E781" if r.high_tg else ("E780" if r.ldl_state != "normal" else "E785")))
    if r.low_hdl and r.dys_hosp_code:
        out.append("E786")
    if r.t2d:
        out.append("E112" if r.t2d_renal else "E119")
    if r.htn_organ:
        out.append("I110")
    if r.htn_secondary:
        out.append("I158")
    if r.asthma_severe:
        out.append("J46")
    return out


def write_ukb(t: pd.DataFrame, root: Path, rng) -> Path:
    tab = root / "tabular"
    (root / "medrec").mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    eid = t["person_id"]
    pd.DataFrame({"EID": eid, "31-0.0": np.where(t["sex"] == "male", 1, 0)}).to_csv(tab / "demographics.tsv", sep="\t", index=False)
    icd = [_icd10(r) for r in t.itertuples(index=False)]
    width = max(3, max(len(x) for x in icd))
    pd.DataFrame({"EID": eid, **{f"41270-0.{k}": [x[k] if k < len(x) else "" for x in icd] for k in range(width)}}).to_csv(
        tab / "diagnoses.tsv", sep="\t", index=False)
    pd.DataFrame({"EID": eid, "30780-0.0": t["ldl1"].round(3), "30760-0.0": t["hdl"].round(3), "30870-0.0": t["tg"].round(3),
                  "30690-0.0": t["tc"].round(3), "30790-0.0": t["lpa"].round(1), "30640-0.0": t["apob"].round(3),
                  "30750-0.0": np.where(t["t2d"], t["hba1c1"].round(0), "")}).to_csv(tab / "biochemistry.tsv", sep="\t", index=False)
    med = []
    for r in t.itertuples(index=False):
        answers = ([1] if (r.treated and r.n_scripts == 1) else []) + ([2] if r.bp_med else [])
        answers = answers or [-7]
        med.append(answers + [""] * (3 - len(answers)))
    male = (t["sex"] == "male").to_numpy()
    cols = {}
    for fid, mask in ((6177, male), (6153, ~male)):
        for k in range(3):
            cols[f"{fid}-0.{k}"] = [m[k] if mask[i] else "" for i, m in enumerate(med)]
    pd.DataFrame({"EID": eid, **cols}).to_csv(tab / "medication.tsv", sep="\t", index=False)
    pd.DataFrame(_gp_rows(t, rng)).to_csv(root / "medrec" / "set3a1.txt", sep="\t", index=False, header=False)
    scripts = []
    for r in t.itertuples(index=False):
        for k in range(r.n_scripts):
            cid = list(STATINS)[k % 3]
            name, bnf = STATINS[cid]
            d = r.date1 + pd.Timedelta(days=30 * (k + 1))
            scripts.append((r.person_id, 1, d.strftime("%d/%m/%Y"), "", bnf, "", name.replace(" Oral Tablet", " tablets"), "28"))
    pd.DataFrame(scripts, columns=["eid", "data_provider", "issue_date", "read_2", "bnf_code", "dmd_code", "drug_name", "quantity"]).to_csv(
        root / "gp_scripts.txt", sep="\t", index=False)
    return root


def write_omop(t: pd.DataFrame, root: Path, rng) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"person_id": t["person_id"], "gender_concept_id": np.where(t["sex"] == "male", 8507, 8532),
                  "year_of_birth": 1950, "race_concept_id": 0, "ethnicity_concept_id": 0,
                  "gender_source_value": np.where(t["sex"] == "male", "M", "F")}).to_csv(root / "person.csv", index=False)
    cond = []

    def add(r, concept, icd10, snomed, date):
        src = icd10 if (r.person_id % 2 == 0 and icd10) else snomed
        cond.append((r.person_id, concept, date.strftime("%Y-%m-%d"), 32817, src, 0))

    for r in t.itertuples(index=False):
        if r.dys and (r.dys_gp_codes or r.dys_hosp_code):
            for k in range(max(1, r.dys_gp_codes)):
                add(r, 432867, "E78.5", "55822004", r.date1 if k == 0 else r.date2)
        if r.high_tg and r.dys_hosp_code:
            add(r, 4120314, "E78.1", "302870006", r.date1)
        if r.t2d:
            add(r, 201826, "E11.9", "44054006", r.date1)
            if r.t2d_renal:
                add(r, 192279, "E11.2", "", r.date2)
            if r.t2d_ophth:
                add(r, 4226121, "E11.3", "", r.date2)
        if r.htn and not r.htn_secondary:
            add(r, 320128, "I10", "59621000", r.date1)
        if r.asthma:
            add(r, 317009, "J45.9", "195967001", r.date1)
    pd.DataFrame(cond, columns=["person_id", "condition_concept_id", "condition_start_date", "condition_type_concept_id",
                                "condition_source_value", "condition_source_concept_id"]).assign(
        condition_occurrence_id=lambda d: np.arange(1, len(d) + 1)).to_csv(root / "condition_occurrence.csv", index=False)
    meas = []
    for r in t.itertuples(index=False):
        for d, ldl, a1c in ((r.date1, r.ldl1, r.hba1c1), (r.date2, r.ldl2, r.hba1c2)):
            ds = d.strftime("%Y-%m-%d")
            meas += [(r.person_id, 3009966, ds, round(ldl / 0.02586, 3), 8840, "mg/dL"),
                     (r.person_id, 3007070, ds, round(r.hdl / 0.02586, 3), 8840, "mg/dL"),
                     (r.person_id, 3022192, ds, round(r.tg / 0.01129, 3), 8840, "mg/dL"),
                     (r.person_id, 3027114, ds, round(r.tc / 0.02586, 3), 8840, "mg/dL")]
            if r.t2d:
                meas.append((r.person_id, 3004410, ds, round(a1c / 10.929 + 2.15, 4), 8554, "%"))
    pd.DataFrame(meas, columns=["person_id", "measurement_concept_id", "measurement_date", "value_as_number",
                                "unit_concept_id", "unit_source_value"]).assign(
        measurement_id=lambda d: np.arange(1, len(d) + 1), measurement_type_concept_id=32817).to_csv(root / "measurement.csv", index=False)
    drugs = []
    for r in t.itertuples(index=False):
        for k in range(r.n_scripts):
            cid = list(STATINS)[k % 3]
            drugs.append((r.person_id, int(cid), (r.date1 + pd.Timedelta(days=30 * (k + 1))).strftime("%Y-%m-%d"), 32869, cid))
    pd.DataFrame(drugs, columns=["person_id", "drug_concept_id", "drug_exposure_start_date", "drug_type_concept_id",
                                 "drug_source_value"]).assign(drug_exposure_id=lambda d: np.arange(1, len(d) + 1)).to_csv(
        root / "drug_exposure.csv", index=False)
    concept = [(int(c), n, "Drug", "RxNorm", "Clinical Drug", "S", "") for c, (n, _) in STATINS.items()]
    concept += [(TOY_ATC_IDS["C10"], "LIPID MODIFYING AGENTS (toy id)", "Drug", "ATC", "ATC 1st", "C", "C10"),
                (TOY_ATC_IDS["C10A"], "LIPID MODIFYING AGENTS, PLAIN (toy id)", "Drug", "ATC", "ATC 2nd", "C", "C10A"),
                (TOY_ATC_IDS["C10AA"], "HMG CoA reductase inhibitors (toy id)", "Drug", "ATC", "ATC 4th", "C", "C10AA")]
    pd.DataFrame(concept, columns=["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
                                   "standard_concept", "concept_code"]).to_csv(root / "concept.csv", index=False)
    anc = [(a, int(c)) for c in STATINS for a in TOY_ATC_IDS.values()]
    pd.DataFrame(anc, columns=["ancestor_concept_id", "descendant_concept_id"]).to_csv(root / "concept_ancestor.csv", index=False)
    return root


def write_toy(out: str | Path, n: int = 400, seed: int = 0) -> dict:
    out = Path(out)
    rng = np.random.default_rng(seed + 1)
    t = simulate(n, seed)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "toy_truth.csv", index=False)
    return {"ukb": write_ukb(t, out / "ukb", rng), "omop": write_omop(t, out / "omop", rng), "truth": t}
