#!/usr/bin/env python3
"""NVFLARE client: local reconstruction training on shared phenotype embeddings."""

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
    l2_normalize_rows,
    l2_normalize_rows_torch,
    phenotype_mask,
    random_unit_params,
    to_numpy,
    train_reconstruction,
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
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _global_table(params: dict | None, key: str, n_phenotypes: int, n_factors: int, device: torch.device) -> torch.Tensor:
    if params and key in params:
        table = torch.as_tensor(to_numpy(params[key]), device=device)
    else:
        table = torch.as_tensor(random_unit_params(n_phenotypes, n_factors)[key], device=device)
    if table.shape != (n_phenotypes, n_factors):
        raise ValueError(f"{key} has shape {tuple(table.shape)}, expected {(n_phenotypes, n_factors)}")
    return l2_normalize_rows_torch(table)


def local_update(
    data_dir: Path,
    phenotype_ids_path: Path,
    n_factors: int,
    device: str,
    local_epochs: int,
    lr: float,
    global_params: dict | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    phenotype_ids = read_id_list(phenotype_ids_path)
    site = load_site(data_dir, phenotype_ids)
    n_phenotypes = len(phenotype_ids)
    torch_device = torch.device(device)

    nongenetic = _global_table(global_params, NONGENETIC_KEY, n_phenotypes, n_factors, torch_device)
    genetic = _global_table(global_params, GENETIC_KEY, n_phenotypes, n_factors, torch_device)

    nongenetic, nongenetic_loss = train_reconstruction(
        site.patient_phenotypes,
        site.patient_col_index,
        nongenetic,
        binary=True,
        local_epochs=local_epochs,
        lr=lr,
    )
    nongenetic_mask = phenotype_mask(n_phenotypes, site.patient_col_index)
    n_patients = float(site.patient_phenotypes.shape[0])

    if site.genome_phenotypes is None or site.genome_col_index is None:
        genetic_np = l2_normalize_rows(genetic.detach().cpu().numpy())
        genetic_mask = np.zeros((n_phenotypes,), dtype=np.float32)
        n_genomes = 0.0
        genetic_loss = 0.0
    else:
        genetic, genetic_loss = train_reconstruction(
            site.genome_phenotypes,
            site.genome_col_index,
            genetic,
            binary=False,
            local_epochs=local_epochs,
            lr=lr,
        )
        genetic_np = l2_normalize_rows(genetic.detach().cpu().numpy())
        genetic_mask = phenotype_mask(n_phenotypes, site.genome_col_index)
        n_genomes = float(site.genome_phenotypes.shape[0])

    print(
        f"[{site.site_id}] N={len(site.patient_ids)} P_patient={site.patient_phenotypes.shape[1]} "
        f"G={'none' if site.genome_phenotypes is None else site.genome_phenotypes.shape[0]} "
        f"nongenetic_loss={nongenetic_loss:.4f} genetic_loss={genetic_loss:.4f}",
        flush=True,
    )
    params = {
        NONGENETIC_KEY: l2_normalize_rows(nongenetic.detach().cpu().numpy()),
        NONGENETIC_MASK_KEY: nongenetic_mask,
        NONGENETIC_WEIGHT_KEY: np.float32(n_patients),
        GENETIC_KEY: genetic_np,
        GENETIC_MASK_KEY: genetic_mask,
        GENETIC_WEIGHT_KEY: np.float32(n_genomes),
    }
    metrics = {
        "n_patients": n_patients,
        "n_genomes": n_genomes,
        "has_genetic": float(n_genomes > 0),
        "nongenetic_reconstruction_loss": nongenetic_loss,
        "genetic_reconstruction_loss": genetic_loss,
    }
    return params, metrics


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    phenotype_ids_path = Path(args.phenotype_ids)

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        print(f"round={input_model.current_round}", flush=True)
        params, metrics = local_update(
            data_dir,
            phenotype_ids_path,
            args.n_factors,
            args.device,
            args.local_epochs,
            args.lr,
            input_model.params,
        )
        flare.send(
            flare.FLModel(
                params=params,
                metrics=metrics,
                current_round=input_model.current_round,
            )
        )


if __name__ == "__main__":
    main()
