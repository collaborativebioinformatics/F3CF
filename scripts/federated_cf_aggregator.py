#!/usr/bin/env python3
"""NVFLARE aggregator: FedAvg of shared normalized phenotype embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from federated_cf_embeddings import (
    GENETIC_KEY,
    GENETIC_MASK_KEY,
    GENETIC_WEIGHT_KEY,
    NONGENETIC_KEY,
    NONGENETIC_MASK_KEY,
    NONGENETIC_WEIGHT_KEY,
    EmbeddingContribution,
    aggregate_contributions,
    l2_normalize_rows,
    to_numpy,
)

try:
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
except ImportError:  # pragma: no cover - unit tests without NVFLARE

    class ModelAggregator:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.fl_ctx = None

    class FLModel:  # type: ignore[no-redef]
        def __init__(self, params=None, metrics=None, meta=None) -> None:
            self.params = params
            self.metrics = metrics
            self.meta = meta or {}


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _contribution(params: dict[str, Any], embedding_key: str, mask_key: str, weight_key: str) -> EmbeddingContribution:
    embeddings = l2_normalize_rows(to_numpy(params[embedding_key]))
    mask = to_numpy(params.get(mask_key, np.ones(embeddings.shape[0], dtype=np.float32))).reshape(-1)
    weight = _scalar(params.get(weight_key, 0.0))
    if mask.shape != (embeddings.shape[0],):
        raise ValueError(f"{mask_key} must have shape ({embeddings.shape[0]},)")
    return EmbeddingContribution(embeddings=embeddings, mask=mask, weight=weight)


def aggregate_client_params(
    client_params: list[dict[str, Any]],
    previous: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    nongenetic = [
        _contribution(params, NONGENETIC_KEY, NONGENETIC_MASK_KEY, NONGENETIC_WEIGHT_KEY)
        for params in client_params
    ]
    genetic = [
        _contribution(params, GENETIC_KEY, GENETIC_MASK_KEY, GENETIC_WEIGHT_KEY) for params in client_params
    ]
    return aggregate_contributions(nongenetic, genetic, previous=previous)


class PhenotypeEmbeddingAggregator(ModelAggregator):
    """FedAvg the shared phenotype tables; skip genetic updates from sites without PGS."""

    def __init__(self) -> None:
        super().__init__()
        self._results: list[Any] = []
        self._previous: dict[str, np.ndarray] | None = None

    def accept_model(self, model: FLModel) -> None:
        if not model.params:
            return
        self._results.append(model)

    def reset_stats(self) -> None:
        self._results = []

    def aggregate_model(self) -> FLModel:
        if not self._results:
            raise ValueError("no client embeddings received for aggregation")

        params = aggregate_client_params(
            [result.params for result in self._results],
            previous=self._previous,
        )
        n_nongenetic = sum(_scalar(result.params.get(NONGENETIC_WEIGHT_KEY, 0.0)) > 0 for result in self._results)
        n_genetic = sum(_scalar(result.params.get(GENETIC_WEIGHT_KEY, 0.0)) > 0 for result in self._results)
        metrics = {
            "n_nongenetic_sites": float(n_nongenetic),
            "n_genetic_sites": float(n_genetic),
        }
        self._previous = params
        return FLModel(params=params, metrics=metrics)


try:
    from pathlib import Path as _Path

    from nvflare.app_common.workflows.fedavg import FedAvg

    class PhenotypeEmbeddingFedAvg(FedAvg):
        """FedAvg that also writes the two global embedding matrices as ``.npz``."""

        def save_model_file(self, model: FLModel, filepath: str) -> None:
            arrays = {
                key: l2_normalize_rows(to_numpy(value)) if "embeddings" in key else np.asarray(value)
                for key, value in (model.params or {}).items()
            }
            np.savez(str(_Path(filepath).with_suffix(".npz")), **arrays)
            super().save_model_file(model, filepath)

except ImportError:  # pragma: no cover
    PhenotypeEmbeddingFedAvg = None  # type: ignore[misc, assignment]
