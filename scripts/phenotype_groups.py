#!/usr/bin/env python3
"""Phenotype grouping and presentation embeddings for the explorer prototype."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CATEGORY_COLORS = {
    "Cardiovascular": "#E63946",
    "Cancer": "#9B2226",
    "Metabolic / endocrine": "#F4A261",
    "Neurological": "#7B2CBF",
    "Psychiatric / behavioral": "#C77DFF",
    "Respiratory": "#4CC9F0",
    "Immune / infectious": "#2A9D8F",
    "Hematologic": "#D62828",
    "Renal": "#3A86FF",
    "Gastrointestinal / liver": "#588157",
    "Musculoskeletal": "#BC6C25",
    "Reproductive": "#FF70A6",
    "Sensory": "#FFB703",
    "Dermatologic": "#E9C46A",
    "Measurements / labs": "#8D99AE",
    "Other": "#6C757D",
}

_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Cancer",
        (
            "cancer", "neoplasm", "carcinoma", "tumor", "tumour", "lymphoma", "leukemia",
            "leukaemia", "melanoma", "sarcoma", "myeloma", "malignan", "glioma",
            "glioblastoma", "waldenstrom", "intraepithelial neoplasia", "macroglobulinemia",
        ),
    ),
    (
        "Cardiovascular",
        (
            "heart", "cardiac", "coronary", "myocardial", "atrial", "ventric", "aortic",
            "arter", "vascular", "stroke", "thrombo", "embolism", "hypertension",
            "hypertensive", "blood pressure", "angina", "atherosclero", "aneurysm",
            "ischemi", "infarction", "ejection fraction", "pulse", "cardiomyopath",
            "brugada", "ekg", "ecg", "electrocardi", "qt interval", "pr interval",
            "rr interval", "pp interval", "tachycardia", "varicose", "vein",
            "cholesterol", "ldl", "hdl", "triglyceride",
        ),
    ),
    (
        "Metabolic / endocrine",
        (
            "diabet", "glucose", "insulin", "homa-ir", "hypoglyc", "obes", "body mass",
            "bmi", "body fat", "body weight", "thyroid", "goiter", "goitre", "graves",
            "thyrotoxic", "metabol", "lipid", "lipoprotein", "adipos", "waist", "hba1c",
            "fatty acid", "docosahexaenoic",
        ),
    ),
    (
        "Neurological",
        (
            "alzheimer", "parkinson", "epilep", "dementia", "migraine", "seizure",
            "neuro", "brain", "hippocamp", "sclerosis", "cerebral", "cognit",
            "motor neuron", "ataxia", "headache", "vertigo", "sciatica", "narcolepsy",
            "parasomnia", "sleep apnea", "rem sleep", "peripheral nervous",
            "nervous system", "cerebrospinal",
        ),
    ),
    (
        "Psychiatric / behavioral",
        (
            "depress", "anxi", "schizophren", "bipolar", "psychiatr", "mental disorder",
            "autism", "adhd", "smok", "alcohol", "cannabis", "drug use", "substance",
            "insomni", "personality", "risk-taking", "educational attainment",
        ),
    ),
    (
        "Respiratory",
        (
            "lung", "pulmon", "asthma", "copd", "respirat", "fev", "bronch", "pneumon",
            "airway", "vital capacity", "expiratory", "rhinosinusitis", "rhinitis",
            "nasal polyp", "apnea",
        ),
    ),
    (
        "Immune / infectious",
        (
            "lupus", "sjogren", "rheumat", "autoimmune", "immun", "infect", "hepatitis",
            "covid", "hiv", "inflammat", "allerg", "celiac", "sarcoid", "herpes",
            "varicella", "zoster", "seropositivity", "ige", "cellulitis", "mucositis",
            "carbuncle", "furuncle", "wart",
        ),
    ),
    (
        "Hematologic",
        (
            "anemi", "hemoglobin", "haemoglob", "hematocrit", "erythrocyte", "platelet",
            "leukocyte", "blood cell", "hematol", "coagul", "neutrophil", "reticulocyte",
            "polycythemia", "red cell",
        ),
    ),
    (
        "Renal",
        (
            "kidney", "renal", "nephro", "creatinine", "albuminuria", "glomerul",
            "hematuria", "urolithiasis", "urea nitrogen", "urine potassium",
            "urinary retention",
        ),
    ),
    (
        "Gastrointestinal / liver",
        (
            "liver", "hepatic", "cirrhos", "alanine aminotransferase", "bowel", "crohn",
            "colitis", "intestin", "gastric", "stomach", "colon", "rectum", "anal",
            "appendic", "pancrea", "gallbladder", "gallstone", "cholecyst", "cholelith",
            "bile", "biliary", "diverticul", "hernia", "ulcer", "duoden", "digestive",
            "malabsorption", "hemorrhoid", "abdominal pain",
        ),
    ),
    (
        "Musculoskeletal",
        (
            "osteo", "arthritis", "arthropathy", "bone", "fracture", "osteopor",
            "skeletal", "muscle", "joint", "spine", "spondylo", "ankylosing",
            "vertebral", "epiphys", "hallux", "dupuytren", "chondrocalcin",
            "connective tissue", "synovium", "tendon", "bursa", "knee injury",
        ),
    ),
    (
        "Reproductive",
        (
            "prostat", "breast", "ovarian", "ovar", "uter", "pregnan", "menstrual",
            "menarche", "menopause", "fertil", "endometr", "testic", "cervix",
            "gyneco", "genital", "reproductive",
        ),
    ),
    (
        "Sensory",
        (
            "hearing", "deaf", "presbycusis", "tinnitus", "vision", "glaucoma",
            "cataract", "eye", "retina", "ear", "ocular", "myopia", "macular",
            "corneal", "keratoconus", "iritis", "iridocyclitis", "uveitis", "labyrinth",
        ),
    ),
    (
        "Dermatologic",
        (
            "skin", "dermat", "eczema", "psoriasis", "suntan", "sunburn", "alopecia",
            "acne", "vitiligo", "keratosis", "prurigo", "epidermal", "dermoid",
            "pilosebaceous", "cyst", "follicular", "hair color",
        ),
    ),
    (
        "Measurements / labs",
        (
            "measurement", "count", "volume", "ratio", "level", "concentration",
            "height", "weight", "anthropometr", "amount", "aging", "life span",
            "age at death", "function studies",
        ),
    ),
]

# Genetic space can re-home a few traits so the two maps are not identical.
_GENETIC_REMAP: list[tuple[tuple[str, ...], str]] = [
    (("stroke",), "Neurological"),
    (("cholesterol", "ldl", "hdl", "triglyceride"), "Metabolic / endocrine"),
    (("body mass", "bmi", "obes"), "Cardiovascular"),
    (("sleep apnea",), "Respiratory"),
    (("depression", "depress"), "Neurological"),
]


def classify_trait(label: str) -> str:
    text = label.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def classify_trait_genetic(label: str) -> str:
    text = label.lower()
    for keywords, category in _GENETIC_REMAP:
        if any(keyword in text for keyword in keywords):
            return category
    return classify_trait(label)


def display_name(label: str) -> str:
    return " · ".join(part.strip() for part in label.split("|") if part.strip()) or label


def load_labels(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["phenotype_id"]] = row["trait_label"]
    return mapping


def _unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, eps)).astype(np.float32)


def _scatter(centroid: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    noise = rng.normal(size=centroid.shape).astype(np.float32)
    noise *= scale / max(float(np.linalg.norm(noise)), 1e-8)
    return _unit(centroid + noise)


def structured_phenotype_embeddings(
    phenotype_ids: list[str],
    labels: dict[str, str],
    n_factors: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build two clustered P×k tables that are similar but not the same.

    Nongenetic space keeps clinical groups well separated. Genetic space pulls
    a few groups together (cardio–metabolic, neuro–psychiatric, cancer–immune)
    and reassigns selected traits, so UMAP/networks tell a related but different story.
    """
    rng = np.random.default_rng(seed)
    names = [labels.get(pid, pid) for pid in phenotype_ids]
    clinical = [classify_trait(name) for name in names]
    genetic_groups = [classify_trait_genetic(name) for name in names]

    centroids = {category: _unit(rng.normal(size=n_factors).astype(np.float32)) for category in CATEGORY_COLORS}

    genetic_centroids = dict(centroids)
    genetic_centroids["Metabolic / endocrine"] = _unit(
        0.76 * centroids["Cardiovascular"] + 0.24 * centroids["Metabolic / endocrine"]
    )
    genetic_centroids["Psychiatric / behavioral"] = _unit(
        0.7 * centroids["Neurological"] + 0.3 * centroids["Psychiatric / behavioral"]
    )
    genetic_centroids["Immune / infectious"] = _unit(
        0.62 * centroids["Cancer"] + 0.38 * centroids["Immune / infectious"]
    )
    genetic_centroids["Hematologic"] = _unit(
        0.5 * centroids["Cardiovascular"] + 0.5 * centroids["Hematologic"]
    )

    tight = {"Cardiovascular", "Cancer", "Respiratory", "Neurological", "Renal"}
    loose = {"Measurements / labs", "Other"}

    nongenetic = np.zeros((len(phenotype_ids), n_factors), dtype=np.float32)
    genetic = np.zeros_like(nongenetic)
    for i, (clinical_group, genetic_group) in enumerate(zip(clinical, genetic_groups)):
        n_scale = 0.85 if clinical_group in loose else 0.28 if clinical_group in tight else 0.38
        g_scale = 1.05 if genetic_group in loose else 0.42 if genetic_group in tight else 0.55
        nongenetic[i] = _scatter(centroids[clinical_group], rng, n_scale)
        genetic[i] = _scatter(genetic_centroids[genetic_group], rng, g_scale)

    return nongenetic, genetic, clinical, genetic_groups
