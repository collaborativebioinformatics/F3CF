#!/usr/bin/env python3
"""NVFLARE client: local SVD collaborative filtering on N×P and optional G×P."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from federated_cf_data import load_site, read_id_list
from federated_cf_embeddings import (
    GENETIC_KEY,
    GENETIC_MASK_KEY,
    GENETIC_WEIGHT_KEY,
    NONGENETIC_KEY,
    NONGENETIC_MASK_KEY,
    NONGENETIC_WEIGHT_KEY,
    extract_site_phenotype_embeddings,
)

try:
    import nvflare.client as flare
except ImportError as exc:  # pragma: no cover
    raise SystemExit("nvflare is required to run the federated client") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Site directory with patient_phenotypes.csv")
    parser.add_argument("--phenotype_ids", required=True, help="Global phenotype ID catalog")
    parser.add_argument("--n_factors", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def local_update(data_dir: Path, phenotype_ids_path: Path, n_factors: int, device: str) -> dict[str, np.ndarray]:
    phenotype_ids = read_id_list(phenotype_ids_path)
    site = load_site(data_dir, phenotype_ids)
    n_phenotypes = len(phenotype_ids)
    torch_device = torch.device(device)

    nongenetic, nongenetic_mask, n_patients = extract_site_phenotype_embeddings(
        site.patient_phenotypes,
        site.patient_col_index,
        n_phenotypes=n_phenotypes,
        n_factors=n_factors,
        device=torch_device,
    )

    if site.genome_phenotypes is None or site.genome_col_index is None:
        genetic = np.zeros((n_phenotypes, n_factors), dtype=np.float32)
        genetic_mask = np.zeros((n_phenotypes,), dtype=np.float32)
        n_genomes = 0.0
    else:
        genetic, genetic_mask, n_genomes = extract_site_phenotype_embeddings(
            site.genome_phenotypes,
            site.genome_col_index,
            n_phenotypes=n_phenotypes,
            n_factors=n_factors,
            device=torch_device,
        )

    print(
        f"[{site.site_id}] N={len(site.patient_ids)} P_patient={site.patient_phenotypes.shape[1]} "
        f"G={'none' if site.genome_phenotypes is None else site.genome_phenotypes.shape[0]} "
        f"factors={n_factors}",
        flush=True,
    )
    return {
        NONGENETIC_KEY: nongenetic,
        NONGENETIC_MASK_KEY: nongenetic_mask,
        NONGENETIC_WEIGHT_KEY: np.float32(n_patients),
        GENETIC_KEY: genetic,
        GENETIC_MASK_KEY: genetic_mask,
        GENETIC_WEIGHT_KEY: np.float32(n_genomes),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    phenotype_ids_path = Path(args.phenotype_ids)

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        print(f"round={input_model.current_round}", flush=True)
        params = local_update(data_dir, phenotype_ids_path, args.n_factors, args.device)
        flare.send(
            flare.FLModel(
                params=params,
                metrics={
                    "n_patients": float(params[NONGENETIC_WEIGHT_KEY]),
                    "n_genomes": float(params[GENETIC_WEIGHT_KEY]),
                    "has_genetic": float(params[GENETIC_WEIGHT_KEY] > 0),
                },
                current_round=input_model.current_round,
            )
        )


if __name__ == "__main__":
    main()
