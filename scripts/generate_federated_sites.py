#!/usr/bin/env python3
"""Generate synthetic federated sites with binary N×P EHR matrices and optional G×P PRS maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from federated_cf_data import (
    GENOME_MATRIX_NAME,
    PATIENT_MATRIX_NAME,
    PHENOTYPE_IDS_NAME,
    write_id_list,
    write_labeled_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/federated")
    parser.add_argument("--n_phenotypes", type=int, default=40)
    parser.add_argument("--n_factors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _site_columns(all_ids: list[str], drop: int, generator: torch.Generator) -> list[str]:
    if drop <= 0:
        return list(all_ids)
    perm = torch.randperm(len(all_ids), generator=generator).tolist()
    dropped = set(perm[:drop])
    return [phenotype_id for i, phenotype_id in enumerate(all_ids) if i not in dropped]


def generate(
    output_dir: str | Path,
    n_phenotypes: int = 40,
    n_factors: int = 32,
    seed: int = 0,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    phenotype_ids = [f"PHENO_{i:04d}" for i in range(n_phenotypes)]
    write_id_list(output_dir / PHENOTYPE_IDS_NAME, phenotype_ids)

    true_nongenetic = torch.randn(n_phenotypes, n_factors, generator=generator)
    true_genetic = torch.randn(n_phenotypes, n_factors, generator=generator)

    sites = [
        {"site_id": "site-1", "n_patients": 120, "n_genomes": 80, "drop": 4},
        {"site_id": "site-2", "n_patients": 90, "n_genomes": 70, "drop": 6},
        {"site_id": "site-3", "n_patients": 110, "n_genomes": 0, "drop": 5},
    ]

    for spec in sites:
        site_dir = output_dir / spec["site_id"]
        site_dir.mkdir(parents=True, exist_ok=True)
        col_ids = _site_columns(phenotype_ids, spec["drop"], generator)
        col_index = torch.tensor([phenotype_ids.index(pid) for pid in col_ids], dtype=torch.long)

        patients = torch.randn(spec["n_patients"], n_factors, generator=generator)
        logits = patients @ true_nongenetic[col_index].T
        # Shift so most entries are 0: a patient either has the phenotype or not.
        prevalence_bias = 1.8
        probs = torch.sigmoid(logits - prevalence_bias)
        patient_matrix = torch.bernoulli(probs, generator=generator)
        patient_ids = [f"{spec['site_id']}_p{i:04d}" for i in range(spec["n_patients"])]
        write_labeled_matrix(site_dir / PATIENT_MATRIX_NAME, patient_ids, col_ids, patient_matrix)

        if spec["n_genomes"] > 0:
            genomes = torch.randn(spec["n_genomes"], n_factors, generator=generator)
            genome_matrix = genomes @ true_genetic[col_index].T
            genome_matrix = genome_matrix + 0.05 * torch.randn(genome_matrix.shape, generator=generator)
            genome_ids = [f"{spec['site_id']}_rs{i:04d}" for i in range(spec["n_genomes"])]
            write_labeled_matrix(site_dir / GENOME_MATRIX_NAME, genome_ids, col_ids, genome_matrix)
        else:
            genome_path = site_dir / GENOME_MATRIX_NAME
            if genome_path.exists():
                genome_path.unlink()

    torch.save(
        {"nongenetic": true_nongenetic, "genetic": true_genetic, "phenotype_ids": phenotype_ids},
        output_dir / "true_embeddings.pt",
    )
    (output_dir / "sites.json").write_text(json.dumps(sites, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = generate(args.output_dir, args.n_phenotypes, args.n_factors, args.seed)
    print(f"Wrote federated site matrices under {output_dir}")


if __name__ == "__main__":
    main()
