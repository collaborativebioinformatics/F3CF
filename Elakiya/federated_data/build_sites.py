#!/usr/bin/env python3
"""Build ``data/ehr_lipids``: EHR-style HPO patient matrices + published-locus genome matrices.

Output layout is identical to ``data/federated`` (scripts/generate_federated_sites.py):
``phenotype_ids.txt``, ``sites.json``, ``site-k/patient_phenotypes.csv`` (binary, header
``row_id,<HPO ids>``), ``site-k/genome_phenotypes.csv`` (real-valued, rows ``site-k_<rsid>``;
site-3 has none) and ``true_embeddings.pt`` (``nongenetic``, ``genetic``, ``phenotype_ids``),
plus ``effect_sources.csv`` (per-locus provenance) and ``phenotype_labels.csv``.

How patients are simulated (all parameters below; see README.md for sources):

1. Genotypes: dosage ~ Binomial(2, effect-allele frequency) for 14 published lead variants.
2. Traits: LDL-C, HDL-C, TG, Lp(a), BMI, glycaemia, blood pressure and atopy are z-scores built
   from a polygenic part (published per-allele effects in trait-SD units, times ``GENETIC_SCALE``),
   a shared metabolic factor (raises TG, BMI, glycaemia, BP and lowers HDL), age and noise.
3. Labs: z-scores become lab values (mmol/L, nmol/L, g/L); intercepts are calibrated on a
   200,000-person reference sample so that high total cholesterol (>= 240 mg/dL) is 11.3%,
   low HDL-C (< 40 mg/dL) is 21.5% in men and 6.6% in women (CDC NCHS Data Brief 515) and
   hypertension is 47.7% (CDC NCHS Data Brief 511).
4. HPO terms: EHR-style rules on the labs (NCEP ATP III cut points; sex-specific HDL cut points,
   Table 8), with parents implied by children (Hypertriglyceridemia -> Hyperlipidemia ->
   Abnormal circulating lipid concentration) and comorbidity links (T2D and hypertension raise
   retinopathy, proteinuria and CKD; LDL-C, Lp(a), low HDL-C, T2D and hypertension raise
   coronary disease).
5. Genome matrix per site: variant x HPO effect = (published effect in SD units) x (trait-to-HPO
   loading) x GENETIC_SCALE + site sampling noise; 0 where the locus has no path to the term
   (Davor's MSE loss does not mask NaN, so no NaN is written).

Self-contained (numpy + torch). The full EHR -> subphenotype pipeline is in PR #2
(``Elakiya/metametagraphs/ehr`` on branch ``elakiya/ehr-subphenotypes``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAITS = ["ldl", "hdl", "tg", "lpa", "bmi", "gly", "bp", "atopy"]
GENETIC_SCALE = 4.0          # amplification of real per-allele effects so ~100s of patients carry signal
ASSUMED_SD_MGDL = {"ldl": 35.0, "hdl": 15.0, "tg": 90.0, "tc": 40.0}   # simulation assumption, not sourced
LOGIT_TO_SD = np.sqrt(3) / np.pi                                        # log-OR -> SD units (logistic SD)

# rsid, gene, trait, effect, effect unit, effect-allele frequency, frequency note, source
LOCI = [
    ("rs2479409", "PCSK9", "ldl", 2.01, "mg/dL per minor allele", 0.30, "MAF, Teslovich Table 1", "Teslovich 2010 Nature 466:707 (PMID 20686565), Table 1; sign read from PDF text layer"),
    ("rs629301", "SORT1", "ldl", -5.65, "mg/dL per minor allele", 0.22, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1"),
    ("rs6511720", "LDLR", "ldl", -6.99, "mg/dL per minor allele", 0.11, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1"),
    ("rs4420638", "APOE", "ldl", 7.14, "mg/dL per minor allele", 0.17, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs1367117", "APOB", "ldl", 4.05, "mg/dL per minor allele", 0.30, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs12916", "HMGCR", "ldl", 2.84, "mg/dL total cholesterol per minor allele (applied to LDL-C)", 0.39, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1 (lead trait TC, other trait LDL)"),
    ("rs3764261", "CETP", "hdl", 3.39, "mg/dL per minor allele", 0.32, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs1532085", "LIPC", "hdl", 1.45, "mg/dL per minor allele", 0.39, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs12678919", "LPL", "tg", -13.64, "mg/dL per minor allele", 0.12, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1"),
    ("rs964184", "APOA1 cluster", "tg", 16.95, "mg/dL per minor allele", 0.13, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs1260326", "GCKR", "tg", 8.76, "mg/dL per minor allele", 0.41, "MAF, Teslovich Table 1", "Teslovich 2010, Table 1; sign read from PDF text layer"),
    ("rs7903146", "TCF7L2", "gly", float(np.log(1.46)), "log-OR type 2 diabetes per T allele (OR 1.46)", 0.30, "simulation assumption", "Cauchi 2007 J Mol Med 85:777 (PMID 17476472), pooled allelic OR 1.46 [1.42-1.51]"),
    ("rs9939609", "FTO", "bmi", float(np.log(1.67) / 2), "log-OR obesity per risk allele (homozygote OR 1.67 / 2)", 0.40, "sqrt(16% homozygous), Frayling 2007", "Frayling 2007 Science 316:889 (PMID 17434869), abstract: homozygotes 16%, OR 1.67 for obesity"),
    ("rs10455872", "LPA", "lpa", float(np.log(1.7)), "log-OR coronary disease per copy (about 1.7), used as Lp(a) effect", 0.07, "simulation assumption", "Clarke 2009 NEJM 361:2518 (PMID 20032323), via BlueRipple summary 'odds ratios around 1.7 to 1.9'; exact value not verified"),
]

# HPO term -> trait loadings (how each term depends on the latent traits)
LOAD = {
    "HP:0003141": {"ldl": 1.0},
    "HP:0003124": {"ldl": 0.9, "tg": 0.2, "hdl": 0.1},
    "HP:0002155": {"tg": 1.0},
    "HP:0003233": {"hdl": -1.0},
    "HP:6000521": {"lpa": 1.0},
    "HP:0031798": {"ldl": 0.9},
    "HP:0003077": {"ldl": 0.6, "tg": 0.6},
    "HP:0003119": {"ldl": 0.5, "tg": 0.5, "hdl": -0.4, "lpa": 0.3},
    "HP:0010874": {"ldl": 1.0},
    "HP:0001084": {"ldl": 0.7},
    "HP:0001114": {"ldl": 0.7},
    "HP:0000822": {"bp": 1.0},
    "HP:0001677": {"ldl": 0.5, "hdl": -0.3, "lpa": 0.5, "bp": 0.3, "gly": 0.3},
    "HP:0001658": {"ldl": 0.5, "hdl": -0.3, "lpa": 0.5, "bp": 0.3, "gly": 0.3},
    "HP:0001681": {"ldl": 0.5, "hdl": -0.3, "lpa": 0.5, "bp": 0.3, "gly": 0.3},
    "HP:0002140": {"bp": 0.6, "gly": 0.3, "lpa": 0.2},
    "HP:0001712": {"bp": 0.8},
    "HP:0001095": {"bp": 0.8},
    "HP:0000819": {"gly": 1.0},
    "HP:0005978": {"gly": 1.0},
    "HP:0003074": {"gly": 1.0},
    "HP:0000855": {"bmi": 0.6, "gly": 0.5},
    "HP:0001513": {"bmi": 1.0},
    "HP:0001397": {"bmi": 0.7, "tg": 0.5},
    "HP:0000488": {"gly": 0.7, "bp": 0.3},
    "HP:0000093": {"gly": 0.6, "bp": 0.4},
    "HP:0012622": {"gly": 0.5, "bp": 0.5},
    "HP:0002099": {"atopy": 1.0},
    "HP:0030828": {"atopy": 0.9},
    "HP:0003193": {"atopy": 0.8},
    "HP:0000964": {"atopy": 0.6},
    "HP:0002094": {"atopy": 0.6, "bmi": 0.2},
}
# trait -> drivers [G_ldl, G_hdl, G_tg, G_lpa, G_bmi, G_gly, metabolic, atopy, age]
DRIVERS = {
    "ldl": [1, 0, 0, 0, 0, 0, 0.0, 0, 0.3], "hdl": [0, 1, 0, 0, 0, 0, -0.4, 0, 0.0],
    "tg": [0, 0, 1, 0, 0, 0, 0.5, 0, 0.0], "lpa": [0, 0, 0, 1, 0, 0, 0.0, 0, 0.0],
    "bmi": [0, 0, 0, 0, 1, 0, 0.6, 0, 0.0], "gly": [0, 0, 0, 0, 0, 1, 0.6, 0, 0.2],
    "bp": [0, 0, 0, 0, 0, 0, 0.4, 0, 0.4], "atopy": [0, 0, 0, 0, 0, 0, 0.0, 1, 0.0],
}
CORE = ["HP:0003141", "HP:0003124", "HP:0003233", "HP:0002155", "HP:0003077", "HP:0005978", "HP:0000822", "HP:0002099"]
SITES = [
    {"site_id": "site-1", "n_patients": 400, "n_genomes": len(LOCI), "drop": 3, "metabolic_shift": 0.0, "atopy_shift": 0.0, "age_mean": 55, "gwas_n": 50000},
    {"site_id": "site-2", "n_patients": 250, "n_genomes": len(LOCI), "drop": 5, "metabolic_shift": 0.3, "atopy_shift": 0.0, "age_mean": 60, "gwas_n": 20000},
    {"site_id": "site-3", "n_patients": 300, "n_genomes": 0, "drop": 4, "metabolic_shift": 0.0, "atopy_shift": 0.3, "age_mean": 50, "gwas_n": 0},
]


def load_terms() -> list[tuple[str, str]]:
    with (HERE / "hpo_terms.csv").open() as f:
        return [(r["hpo_id"], r["label"]) for r in csv.DictReader(f)]


def beta_sd() -> np.ndarray:
    """V x T matrix of per-allele effects in trait-SD units (before GENETIC_SCALE)."""
    b = np.zeros((len(LOCI), len(TRAITS)))
    for i, (_, _, trait, eff, unit, *_rest) in enumerate(LOCI):
        if unit.startswith("mg/dL total"):
            b[i, TRAITS.index(trait)] = eff / ASSUMED_SD_MGDL["tc"]
        elif unit.startswith("mg/dL"):
            b[i, TRAITS.index(trait)] = eff / ASSUMED_SD_MGDL[trait]
        else:
            b[i, TRAITS.index(trait)] = eff * LOGIT_TO_SD
    return b


def load_matrix(ids: list[str]) -> np.ndarray:
    return np.array([[LOAD[p].get(t, 0.0) for t in TRAITS] for p in ids])


def simulate_traits(n: int, rng: np.random.Generator, site: dict) -> dict:
    freqs = np.array([l[5] for l in LOCI])
    geno = rng.binomial(2, freqs, size=(n, len(LOCI))).astype(float)
    prs = (geno - 2 * freqs) @ beta_sd() * GENETIC_SCALE                 # n x T
    sex = np.where(rng.random(n) < 0.5, "male", "female")
    age = rng.normal(site["age_mean"], 8, n)
    age_z = (age - 55) / 8
    m = rng.normal(site["metabolic_shift"], 1, n)
    a = rng.normal(site["atopy_shift"], 1, n)
    e = lambda s: rng.normal(0, s, n)  # noqa: E731
    z = {
        "ldl": prs[:, 0] + 0.3 * age_z + e(0.8), "hdl": prs[:, 1] - 0.4 * m + e(0.8),
        "tg": prs[:, 2] + 0.5 * m + e(0.8), "lpa": prs[:, 3] + e(0.8),
        "bmi": prs[:, 4] + 0.6 * m + e(0.8), "gly": prs[:, 5] + 0.6 * m + 0.2 * age_z + e(0.8),
        "bp": 0.4 * m + 0.4 * age_z + e(0.8), "atopy": a + e(0.6),
    }
    return {"z": z, "sex": sex, "age_z": age_z, "m": m, "geno": geno}


def calibrate(seed: int = 12345) -> dict:
    """Intercepts / cut points from a 200k reference sample (site-1 settings)."""
    rng = np.random.default_rng(seed)
    t = simulate_traits(200_000, rng, SITES[0])
    z, male = t["z"], t["sex"] == "male"
    # HDL-C: P(HDL < 1.03 mmol/L) = 21.5% men, 6.6% women (CDC NCHS DB 515)
    hdl_sd = 0.38
    zc = z["hdl"] / z["hdl"].std() * hdl_sd
    mu_m = 1.03 - np.quantile(zc[male], 0.215)
    mu_f = 1.03 - np.quantile(zc[~male], 0.066)
    hdl = np.where(male, mu_m, mu_f) + zc
    tg = np.exp(np.log(1.3) + 0.5 * z["tg"] / z["tg"].std())
    ldl_dev = 0.9 * z["ldl"] / z["ldl"].std()
    # total cholesterol >= 6.21 mmol/L in 11.3% of adults (CDC NCHS DB 515)
    tc_wo = ldl_dev + hdl + tg / 2.2
    ldl_mu = 6.21 - np.quantile(tc_wo, 1 - 0.113)
    return {
        "sd": {k: float(v.std()) for k, v in z.items()}, "hdl_mu": {"male": float(mu_m), "female": float(mu_f)},
        "hdl_sd": hdl_sd, "ldl_mu": float(ldl_mu),
        "bp_cut": float(np.quantile(z["bp"] / z["bp"].std(), 1 - 0.477)),          # hypertension 47.7% (CDC NCHS DB 511)
        "gly_t2d": float(np.quantile(z["gly"] / z["gly"].std(), 0.90)),           # T2D about 10%: simulation assumption
        "gly_hyper": float(np.quantile(z["gly"] / z["gly"].std(), 0.80)),
        "bmi_obese": float(np.quantile(z["bmi"] / z["bmi"].std(), 0.70)),         # obesity about 30%: simulation assumption
        "atopy_asthma": float(np.quantile(z["atopy"] / z["atopy"].std(), 0.90)),  # asthma about 10%: simulation assumption
    }


def phenotypes(t: dict, cal: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    z = {k: v / cal["sd"][k] for k, v in t["z"].items()}
    n = len(t["sex"])
    u = lambda: rng.random(n)  # noqa: E731
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    male = t["sex"] == "male"
    ldl = cal["ldl_mu"] + 0.9 * z["ldl"]
    hdl = np.where(male, cal["hdl_mu"]["male"], cal["hdl_mu"]["female"]) + cal["hdl_sd"] * z["hdl"]
    tg = np.exp(np.log(1.3) + 0.5 * z["tg"])
    tc = ldl + hdl + tg / 2.2
    lpa = np.exp(np.log(20) + 1.2 * z["lpa"])
    apob = 0.26 * ldl + 0.1 + rng.normal(0, 0.08, n)
    h = {}
    h["HP:0003141"] = ldl >= 4.14                                   # LDL-C >= 160 mg/dL (NCEP ATP III)
    h["HP:0003124"] = (tc >= 6.21) | (ldl >= 4.91)                  # TC >= 240 mg/dL or LDL-C >= 190 mg/dL
    h["HP:0002155"] = tg >= 2.26                                    # TG >= 200 mg/dL
    h["HP:0003233"] = hdl < np.where(male, 1.03, 1.29)              # NCEP ATP III Table 8
    h["HP:6000521"] = lpa > 105                                     # ESC/EAS 2025: > 105 nmol/L
    h["HP:0031798"] = apob >= 1.30                                  # AHA/ACC 2018: apoB >= 130 mg/dL
    h["HP:0003077"] = h["HP:0003124"] | h["HP:0003141"] | h["HP:0002155"]
    h["HP:0003119"] = h["HP:0003077"] | h["HP:0003233"] | h["HP:6000521"] | h["HP:0031798"]
    h["HP:0010874"] = (ldl >= 6.5) & (u() < 0.3)
    h["HP:0001084"] = (ldl >= 5.0) & (u() < 0.15 + 0.1 * (t["age_z"] > 0))
    h["HP:0001114"] = ((ldl >= 5.0) & (u() < 0.1)) | (u() < 0.01)
    htn = z["bp"] > cal["bp_cut"]
    t2d = z["gly"] > cal["gly_t2d"]
    h["HP:0000822"] = htn
    h["HP:0005978"] = t2d
    h["HP:0000819"] = t2d
    h["HP:0003074"] = (z["gly"] > cal["gly_hyper"]) | t2d
    h["HP:0000855"] = ((t["m"] > 1.0) & (u() < 0.7)) | (t2d & (u() < 0.5))
    h["HP:0001513"] = z["bmi"] > cal["bmi_obese"]
    h["HP:0001397"] = u() < sig(-2.5 + 1.0 * z["bmi"] + 0.7 * z["tg"])
    cad = u() < sig(-4 + 0.8 * z["ldl"] - 0.4 * z["hdl"] + 0.7 * z["lpa"] + 0.8 * htn + 0.7 * t2d + 0.8 * t["age_z"])
    h["HP:0001677"] = cad
    h["HP:0001658"] = cad & (u() < 0.35)
    h["HP:0001681"] = cad & (u() < 0.5)
    h["HP:0002140"] = u() < sig(-5 + 1.0 * htn + 0.5 * t2d + 0.8 * t["age_z"] + 0.2 * z["lpa"])
    h["HP:0001712"] = htn & (u() < 0.15)
    h["HP:0001095"] = htn & (u() < 0.08)
    h["HP:0000488"] = (t2d & (u() < 0.2)) | (htn & (u() < 0.05))
    h["HP:0000093"] = (t2d & (u() < 0.2)) | (htn & (u() < 0.08))
    h["HP:0012622"] = (t2d & (u() < 0.15)) | (htn & (u() < 0.1)) | (h["HP:0000093"] & (u() < 0.4))
    asthma = z["atopy"] > cal["atopy_asthma"]
    h["HP:0002099"] = asthma
    h["HP:0030828"] = (asthma & (u() < 0.7)) | (u() < 0.03)
    h["HP:0003193"] = u() < sig(-1.5 + 1.2 * z["atopy"])
    h["HP:0000964"] = u() < sig(-2.5 + 1.0 * z["atopy"])
    h["HP:0002094"] = (asthma & (u() < 0.5)) | (h["HP:0001513"] & (u() < 0.1)) | (cad & (u() < 0.3))
    labs = {"ldl": ldl, "hdl": hdl, "tg": tg, "tc": tc, "lpa": lpa, "apob": apob}
    return {k: v.astype(int) for k, v in h.items()}, labs


def fmt(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.8g}"


def write_matrix(path: Path, rows: list[str], cols: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", *cols])
        for r, v in zip(rows, values):
            w.writerow([r, *(fmt(x) for x in v)])


def build(out: Path, seed: int = 0) -> dict:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    terms = load_terms()
    ids = [t for t, _ in terms]
    rng = np.random.default_rng(seed)
    cal = calibrate()
    L = load_matrix(ids)                                     # P x T
    B = beta_sd() * GENETIC_SCALE                            # V x T
    summary = {}
    for spec in SITES:
        site_dir = out / spec["site_id"]
        optional = [p for p in ids if p not in CORE]
        dropped = set(rng.choice(optional, size=spec["drop"], replace=False))
        cols = [p for p in ids if p not in dropped]
        t = simulate_traits(spec["n_patients"], rng, spec)
        h, labs = phenotypes(t, cal, rng)
        mat = np.stack([h[p] for p in cols], axis=1)
        pids = [f"{spec['site_id']}_p{i:04d}" for i in range(spec["n_patients"])]
        write_matrix(site_dir / "patient_phenotypes.csv", pids, cols, mat)
        gpath = site_dir / "genome_phenotypes.csv"
        if spec["n_genomes"]:
            colidx = [ids.index(p) for p in cols]
            G = B @ L[colidx].T
            se = 1.0 / np.sqrt(spec["gwas_n"] / 1000.0)      # site sampling noise, shrinks with GWAS size
            G = G + rng.normal(0, se, G.shape) * (G != 0)
            write_matrix(gpath, [f"{spec['site_id']}_{l[0]}" for l in LOCI], cols, np.round(G, 6))
        elif gpath.exists():
            gpath.unlink()
        summary[spec["site_id"]] = {
            "n_patients": spec["n_patients"], "n_columns": len(cols), "dropped": sorted(dropped),
            "prevalence": {p: round(float(h[p].mean()), 3) for p in ["HP:0003124", "HP:0003141", "HP:0003233", "HP:0002155", "HP:0005978", "HP:0000822", "HP:0002099"]},
            "male_low_hdl": round(float(h["HP:0003233"][t["sex"] == "male"].mean()), 3),
            "female_low_hdl": round(float(h["HP:0003233"][t["sex"] == "female"].mean()), 3),
            "female_hdl_below_1_03": round(float((labs["hdl"] < 1.03)[t["sex"] == "female"].mean()), 3),
            "mean_ldl_mmol_l": round(float(labs["ldl"].mean()), 2),
        }
    (out / "phenotype_ids.txt").write_text("\n".join(ids) + "\n")
    with (out / "phenotype_labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phenotype_id", "label"])
        w.writerows(terms)
    sites_json = [{k: s[k] for k in ("site_id", "n_patients", "n_genomes", "drop")} for s in SITES]
    (out / "sites.json").write_text(json.dumps(sites_json, indent=2) + "\n")
    with (out / "effect_sources.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rsid", "gene", "trait", "published_effect", "unit", "effect_allele_frequency", "frequency_source",
                    "effect_sd_units", "genetic_scale", "hpo_terms_with_nonzero_effect", "source"])
        bsd = beta_sd()
        for i, (rs, gene, trait, eff, unit, freq, fsrc, src) in enumerate(LOCI):
            hp = ";".join(p for p in ids if LOAD[p].get(trait, 0) != 0)
            w.writerow([rs, gene, trait, round(eff, 6), unit, freq, fsrc, round(float(bsd[i, TRAITS.index(trait)]), 6),
                        GENETIC_SCALE, hp, src])
    drivers = np.array([DRIVERS[t] for t in TRAITS], dtype=float)   # T x D
    torch.save({"nongenetic": torch.tensor(L @ drivers, dtype=torch.float32),
                "genetic": torch.tensor(L, dtype=torch.float32), "phenotype_ids": ids,
                "note": "nongenetic = HPO x latent drivers (6 polygenic traits, metabolic, atopy, age); "
                        "genetic = HPO x trait loadings (ldl, hdl, tg, lpa, bmi, gly, bp, atopy)"},
               out / "true_embeddings.pt")
    (out / "build_summary.json").write_text(json.dumps({"seed": seed, "genetic_scale": GENETIC_SCALE,
                                                        "calibration": cal, "sites": summary}, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default=str(ROOT / "data" / "ehr_lipids"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    summary = build(Path(args.output_dir), args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
