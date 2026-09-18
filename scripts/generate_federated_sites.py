#!/usr/bin/env python3
"""Generate federated sites: binary EHR plus optional patient-level PGS scores.

Nongenetic CF uses a binary patient × phenotype matrix. Genetic CF uses the
same patients and phenotypes, with each cell set to that patient's PGS for
the trait instead of 0/1. Sites without genetics omit the PGS matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from federated_cf_data import (
    GENOME_MATRIX_NAME,
    GENOME_MATRIX_NPZ_NAME,
    PATIENT_MATRIX_NAME,
    PHENOTYPE_IDS_NAME,
    read_labeled_matrix,
    read_sparse_labeled_matrix,
    write_id_list,
    write_labeled_matrix,
)
from phenotype_groups import CATEGORY_COLORS, load_labels, structured_phenotype_embeddings

DEFAULT_PGS_MATRIX = "data/pgs/pgs_phenotype_effects.csv"
DEFAULT_LABELS = "data/pgs/phenotype_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/federated")
    parser.add_argument("--pgs_matrix", default=DEFAULT_PGS_MATRIX, help="Trait catalog used for phenotype IDs")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="CSV with phenotype_id and trait_label")
    parser.add_argument("--n_phenotypes", type=int, default=0, help="0 = use all PGS traits")
    parser.add_argument("--n_factors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _site_columns(all_ids: list[str], drop: int, generator: torch.Generator) -> list[str]:
    if drop <= 0:
        return list(all_ids)
    perm = torch.randperm(len(all_ids), generator=generator).tolist()
    dropped = set(perm[:drop])
    return [phenotype_id for i, phenotype_id in enumerate(all_ids) if i not in dropped]


def _standardize_columns(matrix: torch.Tensor) -> torch.Tensor:
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (matrix - mean) / std


def generate(
    output_dir: str | Path,
    n_phenotypes: int = 0,
    n_factors: int = 32,
    seed: int = 0,
    pgs_matrix: str | Path | None = DEFAULT_PGS_MATRIX,
    max_snps: int = 0,
    labels_path: str | Path = DEFAULT_LABELS,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    pgs_path = Path(pgs_matrix) if pgs_matrix else None
    if pgs_path is not None and not pgs_path.exists() and pgs_path.suffix == ".csv":
        alt = pgs_path.with_suffix(".npz")
        if alt.exists():
            pgs_path = alt
    if pgs_path is not None and pgs_path.exists():
        if pgs_path.suffix == ".npz":
            _, phenotype_ids, _ = read_sparse_labeled_matrix(pgs_path)
        else:
            _, phenotype_ids, _ = read_labeled_matrix(pgs_path)
        if n_phenotypes and n_phenotypes < len(phenotype_ids):
            phenotype_ids = phenotype_ids[:n_phenotypes]
        print(f"Using PGS trait catalog {pgs_path} ({len(phenotype_ids)} traits)")
    else:
        n_keep = n_phenotypes if n_phenotypes else 40
        phenotype_ids = [f"PHENO_{i:04d}" for i in range(n_keep)]
        print("No PGS catalog found; using synthetic phenotype IDs")

    write_id_list(output_dir / PHENOTYPE_IDS_NAME, phenotype_ids)
    labels = load_labels(labels_path) if Path(labels_path).exists() else {}
    nongenetic_np, genetic_np, clinical_groups, _genetic_groups = structured_phenotype_embeddings(
        phenotype_ids, labels, n_factors, seed=seed
    )
    true_nongenetic = torch.from_numpy(nongenetic_np)
    true_genetic = torch.from_numpy(genetic_np)
    np.savez(
        output_dir / "global_phenotype_embeddings.npz",
        phenotype_ids=np.array(phenotype_ids),
        nongenetic_phenotype_embeddings=nongenetic_np,
        genetic_phenotype_embeddings=genetic_np,
    )
    used = {group: clinical_groups.count(group) for group in CATEGORY_COLORS if clinical_groups.count(group)}
    print(
        "Wrote clustered presentation embeddings "
        f"({sum(used.values())} traits in {len(used)} groups)"
    )

    sites = [
        {"site_id": "site-1", "n_patients": 120, "has_genetic": True, "drop": 4},
        {"site_id": "site-2", "n_patients": 90, "has_genetic": True, "drop": 6},
        {"site_id": "site-3", "n_patients": 110, "has_genetic": False, "drop": 5},
    ]

    for spec in sites:
        site_dir = output_dir / spec["site_id"]
        site_dir.mkdir(parents=True, exist_ok=True)
        col_ids = _site_columns(phenotype_ids, spec["drop"], generator)
        col_index = torch.tensor([phenotype_ids.index(pid) for pid in col_ids], dtype=torch.long)
        n_patients = spec["n_patients"]
        patient_ids = [f"{spec['site_id']}_p{i:04d}" for i in range(n_patients)]

        ehr_factors = torch.randn(n_patients, n_factors, generator=generator)
        logits = ehr_factors @ true_nongenetic[col_index].T
        patient_matrix = torch.bernoulli(torch.sigmoid(logits - 1.8), generator=generator)
        write_labeled_matrix(site_dir / PATIENT_MATRIX_NAME, patient_ids, col_ids, patient_matrix)

        genome_csv = site_dir / GENOME_MATRIX_NAME
        genome_npz = site_dir / GENOME_MATRIX_NPZ_NAME
        if spec["has_genetic"]:
            pgs_factors = torch.randn(n_patients, n_factors, generator=generator)
            pgs_matrix = _standardize_columns(pgs_factors @ true_genetic[col_index].T)
            write_labeled_matrix(genome_csv, patient_ids, col_ids, pgs_matrix)
            if genome_npz.exists():
                genome_npz.unlink()
            spec["n_genomes"] = n_patients
        else:
            spec["n_genomes"] = 0
            for leftover in (genome_csv, genome_npz):
                if leftover.exists():
                    leftover.unlink()

    (output_dir / "sites.json").write_text(json.dumps(sites, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = generate(
        args.output_dir,
        n_phenotypes=args.n_phenotypes,
        n_factors=args.n_factors,
        seed=args.seed,
        pgs_matrix=args.pgs_matrix,
        labels_path=args.labels,
    )
    print(f"Wrote federated site matrices under {output_dir}")


if __name__ == "__main__":
    main()
