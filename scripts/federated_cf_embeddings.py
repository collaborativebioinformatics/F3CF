#!/usr/bin/env python3
"""Shared phenotype embeddings, reconstruction training, and FedAvg.

All sites pull on the same global ``P × k`` tables. Local patient / genome
row embeddings stay at the site. Phenotype vectors are L2-normalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

NONGENETIC_KEY = "nongenetic_phenotype_embeddings"
GENETIC_KEY = "genetic_phenotype_embeddings"
NONGENETIC_MASK_KEY = "nongenetic_mask"
GENETIC_MASK_KEY = "genetic_mask"
NONGENETIC_WEIGHT_KEY = "nongenetic_weight"
GENETIC_WEIGHT_KEY = "genetic_weight"


@dataclass
class EmbeddingContribution:
    embeddings: np.ndarray  # P x k
    mask: np.ndarray  # P
    weight: float


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each embedding vector (row)."""
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def l2_normalize_rows_torch(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    return F.normalize(matrix, p=2, dim=-1, eps=eps)


def random_unit_params(n_phenotypes: int, n_factors: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Shared random unit-norm initialization for both phenotype tables."""
    rng = np.random.default_rng(seed)

    def draw() -> np.ndarray:
        return l2_normalize_rows(rng.normal(size=(n_phenotypes, n_factors)).astype(np.float32))

    return {NONGENETIC_KEY: draw(), GENETIC_KEY: draw()}


def empty_global_params(n_phenotypes: int, n_factors: int, seed: int = 0) -> dict[str, np.ndarray]:
    return random_unit_params(n_phenotypes, n_factors, seed=seed)


def to_numpy(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def phenotype_mask(n_phenotypes: int, local_col_index: Tensor) -> np.ndarray:
    mask = np.zeros((n_phenotypes,), dtype=np.float32)
    mask[local_col_index.detach().cpu().numpy()] = 1.0
    return mask


def train_reconstruction(
    matrix: Tensor,
    local_col_index: Tensor,
    phenotype_embeddings: Tensor,
    *,
    binary: bool,
    local_epochs: int = 20,
    lr: float = 0.05,
) -> tuple[Tensor, float]:
    """Fit local row embeddings and shared phenotype embeddings to reconstruct ``matrix``.

    ``phenotype_embeddings`` is the full global ``P × k`` table. Only columns in
    ``local_col_index`` receive gradients; other rows stay at the received values.
    Phenotype vectors are projected onto the unit sphere after every step.
    """
    n_rows = matrix.shape[0]
    n_phenotypes, n_factors = phenotype_embeddings.shape
    device = phenotype_embeddings.device
    target = matrix.to(device=device, dtype=torch.float32)
    columns = local_col_index.to(device=device)

    row_embeddings = nn.Parameter(0.1 * torch.randn(n_rows, n_factors, device=device))
    phenotypes = nn.Parameter(l2_normalize_rows_torch(phenotype_embeddings.detach().clone()))
    optimizer = torch.optim.Adam([row_embeddings, phenotypes], lr=lr)

    observed = torch.zeros((n_phenotypes, 1), device=device)
    observed[columns] = 1.0
    last_loss = 0.0

    for _ in range(local_epochs):
        optimizer.zero_grad()
        unit_phenotypes = l2_normalize_rows_torch(phenotypes)
        predicted = row_embeddings @ unit_phenotypes[columns].T
        if binary:
            loss = F.binary_cross_entropy_with_logits(predicted, target)
        else:
            loss = F.mse_loss(predicted, target)
        loss.backward()
        if phenotypes.grad is not None:
            phenotypes.grad.mul_(observed)
        optimizer.step()
        with torch.no_grad():
            phenotypes.copy_(l2_normalize_rows_torch(phenotypes))
        last_loss = float(loss.item())

    return l2_normalize_rows_torch(phenotypes).detach(), last_loss


def fedavg_normalized(
    contributions: list[EmbeddingContribution],
    previous: np.ndarray | None = None,
) -> np.ndarray:
    """Size-weighted FedAvg over observed phenotype rows, then L2-normalize."""
    valid = [item for item in contributions if item.weight > 0 and float(item.mask.sum()) > 0]
    if not valid:
        if previous is None:
            raise ValueError("no contributions to aggregate")
        return l2_normalize_rows(previous)

    n_phenotypes, n_factors = valid[0].embeddings.shape
    weighted_sum = np.zeros((n_phenotypes, n_factors), dtype=np.float64)
    weight_sum = np.zeros((n_phenotypes, 1), dtype=np.float64)
    for item in valid:
        mask = item.mask.reshape(-1, 1)
        weighted_sum += item.weight * mask * item.embeddings
        weight_sum += item.weight * mask

    averaged = weighted_sum / np.maximum(weight_sum, 1e-8)
    if previous is not None:
        unaveraged = weight_sum.squeeze(-1) <= 0
        averaged[unaveraged] = previous[unaveraged]
    return l2_normalize_rows(averaged)


def aggregate_contributions(
    nongenetic: list[EmbeddingContribution],
    genetic: list[EmbeddingContribution],
    previous: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    prev_nongenetic = None if previous is None else previous.get(NONGENETIC_KEY)
    prev_genetic = None if previous is None else previous.get(GENETIC_KEY)
    nongenetic_embeddings = fedavg_normalized(nongenetic, previous=prev_nongenetic)
    if any(item.weight > 0 for item in genetic):
        genetic_embeddings = fedavg_normalized(genetic, previous=prev_genetic)
    elif prev_genetic is not None:
        genetic_embeddings = l2_normalize_rows(prev_genetic)
    else:
        genetic_embeddings = np.zeros_like(nongenetic_embeddings)
    return {NONGENETIC_KEY: nongenetic_embeddings, GENETIC_KEY: genetic_embeddings}
