#!/usr/bin/env python3
"""Phenotype embedding extraction, alignment, and masked federated averaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

NONGENETIC_KEY = "nongenetic_phenotype_embeddings"
GENETIC_KEY = "genetic_phenotype_embeddings"
NONGENETIC_MASK_KEY = "nongenetic_mask"
GENETIC_MASK_KEY = "genetic_mask"
NONGENETIC_WEIGHT_KEY = "nongenetic_weight"
GENETIC_WEIGHT_KEY = "genetic_weight"

MatrixLike = Union[Tensor, list, tuple]


def _as_tensor(
    matrix: MatrixLike,
    device: Optional[torch.device | str],
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(matrix, Tensor):
        tensor = matrix.to(device=device or matrix.device, dtype=dtype)
    else:
        tensor = torch.as_tensor(matrix, dtype=dtype, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"matrix must be 2-dimensional, got shape {tuple(tensor.shape)}")
    return torch.nan_to_num(tensor, nan=0.0)


def pad_to_n_factors(embeddings: Tensor, n_factors: int) -> Tensor:
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    rank = embeddings.shape[-1]
    if rank == n_factors:
        return embeddings
    if rank > n_factors:
        return embeddings[..., :n_factors]
    pad_shape = embeddings.shape[:-1] + (n_factors - rank,)
    return torch.cat([embeddings, embeddings.new_zeros(pad_shape)], dim=-1)


def extract_column_embeddings(
    matrix: MatrixLike,
    n_factors: int,
    *,
    center: bool = False,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Truncated SVD column embeddings of shape ``(n_cols, n_factors)``."""
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    prepared = _as_tensor(matrix, device=device, dtype=dtype)
    if center:
        prepared = prepared - float(prepared.mean())
    _, s, vh = torch.linalg.svd(prepared, full_matrices=False)
    k = min(n_factors, s.numel())
    columns = vh[:k].T * torch.sqrt(s[:k])
    return pad_to_n_factors(columns, n_factors)


@dataclass
class EmbeddingContribution:
    embeddings: np.ndarray  # P x k
    mask: np.ndarray  # P
    weight: float


def scatter_to_global(local_embeddings: Tensor, local_col_index: Tensor, n_phenotypes: int) -> tuple[Tensor, Tensor]:
    """Place local ``P_site x k`` embeddings into a global ``P x k`` matrix with an observation mask."""
    n_factors = local_embeddings.shape[1]
    scattered = local_embeddings.new_zeros((n_phenotypes, n_factors))
    mask = local_embeddings.new_zeros((n_phenotypes,))
    scattered[local_col_index] = local_embeddings
    mask[local_col_index] = 1.0
    return scattered, mask


def extract_site_phenotype_embeddings(
    matrix: Tensor,
    local_col_index: Tensor,
    n_phenotypes: int,
    n_factors: int,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collaborative-filter a site matrix and return global-shaped phenotype embeddings."""
    column_embeddings = extract_column_embeddings(matrix, n_factors=n_factors, device=device)
    scattered, mask = scatter_to_global(column_embeddings, local_col_index, n_phenotypes)
    return (
        scattered.detach().cpu().numpy().astype(np.float32),
        mask.detach().cpu().numpy().astype(np.float32),
        float(matrix.shape[0]),
    )


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rotate ``source`` onto ``target`` using overlapping rows (SVD orthogonal Procrustes)."""
    overlap = mask > 0
    if int(overlap.sum()) < 2:
        return source
    src = source[overlap]
    tgt = target[overlap]
    matrix = src.T @ tgt
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    rotation = u @ vt
    return source @ rotation


def _is_zero_matrix(matrix: np.ndarray) -> bool:
    return bool(np.max(np.abs(matrix)) < 1e-8)


def align_and_average(
    contributions: list[EmbeddingContribution],
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Masked, weighted average of phenotype embeddings after Procrustes alignment."""
    valid = [item for item in contributions if item.weight > 0 and float(item.mask.sum()) > 0]
    if not valid:
        if reference is None:
            raise ValueError("no contributions to aggregate")
        return reference.astype(np.float32, copy=True)

    n_phenotypes, n_factors = valid[0].embeddings.shape
    if reference is None or _is_zero_matrix(reference):
        reference = max(valid, key=lambda item: item.weight).embeddings

    weighted_sum = np.zeros((n_phenotypes, n_factors), dtype=np.float64)
    weight_sum = np.zeros((n_phenotypes, 1), dtype=np.float64)
    for item in valid:
        aligned = orthogonal_procrustes(item.embeddings, reference, item.mask)
        mask = item.mask.reshape(-1, 1)
        weighted_sum += item.weight * mask * aligned
        weight_sum += item.weight * mask
    averaged = weighted_sum / np.maximum(weight_sum, 1e-8)
    unaveraged = weight_sum.squeeze(-1) <= 0
    if reference is not None:
        averaged[unaveraged] = reference[unaveraged]
    return averaged.astype(np.float32)


def empty_global_params(n_phenotypes: int, n_factors: int) -> dict[str, np.ndarray]:
    zeros = np.zeros((n_phenotypes, n_factors), dtype=np.float32)
    return {NONGENETIC_KEY: zeros.copy(), GENETIC_KEY: zeros.copy()}


def aggregate_contributions(
    nongenetic: list[EmbeddingContribution],
    genetic: list[EmbeddingContribution],
    previous: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    prev_nongenetic = None if previous is None else previous.get(NONGENETIC_KEY)
    prev_genetic = None if previous is None else previous.get(GENETIC_KEY)
    nongenetic_embeddings = align_and_average(nongenetic, reference=prev_nongenetic)
    if any(item.weight > 0 for item in genetic):
        genetic_embeddings = align_and_average(genetic, reference=prev_genetic)
    elif prev_genetic is not None:
        genetic_embeddings = prev_genetic.astype(np.float32, copy=True)
    else:
        genetic_embeddings = np.zeros_like(nongenetic_embeddings)
    return {NONGENETIC_KEY: nongenetic_embeddings, GENETIC_KEY: genetic_embeddings}
