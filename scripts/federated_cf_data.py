#!/usr/bin/env python3
"""Load per-site patient–phenotype and optional genome–phenotype matrices."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

PATIENT_MATRIX_NAME = "patient_phenotypes.csv"
GENOME_MATRIX_NAME = "genome_phenotypes.csv"
PHENOTYPE_IDS_NAME = "phenotype_ids.txt"


@dataclass
class SiteDataset:
    """One federated site's observed matrices, aligned to a global phenotype order."""

    site_id: str
    phenotype_ids: list[str]
    patient_ids: list[str]
    patient_phenotypes: Tensor  # binary N x P_local (1 = phenotype present)
    patient_col_index: Tensor  # maps local columns -> global phenotype index
    genome_ids: list[str]
    genome_phenotypes: Tensor | None  # G x P_local_genome or None
    genome_col_index: Tensor | None


def read_id_list(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"no IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return ids


def write_id_list(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _format_cell(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.8g}"


def write_labeled_matrix(path: Path, row_ids: Sequence[str], col_ids: Sequence[str], values: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = values.detach().cpu().tolist()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", *col_ids])
        for row_id, row in zip(row_ids, dense):
            writer.writerow([row_id, *(_format_cell(x) for x in row)])


def read_labeled_matrix(path: Path) -> tuple[list[str], list[str], Tensor]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"{path} must have a row-id column and at least one phenotype column")
        col_ids = header[1:]
        row_ids: list[str] = []
        rows: list[list[float]] = []
        for line in reader:
            if not line:
                continue
            row_ids.append(line[0])
            rows.append([float(x) if x != "" else float("nan") for x in line[1:]])
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return row_ids, col_ids, torch.tensor(rows, dtype=torch.float32)


def _column_index(local_ids: Sequence[str], global_ids: Sequence[str], source: Path) -> Tensor:
    lookup = {phenotype_id: i for i, phenotype_id in enumerate(global_ids)}
    missing = [phenotype_id for phenotype_id in local_ids if phenotype_id not in lookup]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"{source} has phenotypes not in the global catalog: {preview}")
    return torch.tensor([lookup[phenotype_id] for phenotype_id in local_ids], dtype=torch.long)


def load_site(data_dir: str | Path, phenotype_ids: Sequence[str], site_id: str | None = None) -> SiteDataset:
    """Load ``patient_phenotypes.csv`` and optional ``genome_phenotypes.csv`` from a site directory."""
    data_dir = Path(data_dir)
    patient_path = data_dir / PATIENT_MATRIX_NAME
    if not patient_path.exists():
        raise FileNotFoundError(f"required matrix not found: {patient_path}")

    patient_ids, patient_cols, patient_matrix = read_labeled_matrix(patient_path)
    patient_index = _column_index(patient_cols, phenotype_ids, patient_path)

    genome_path = data_dir / GENOME_MATRIX_NAME
    genome_ids: list[str] = []
    genome_matrix: Tensor | None = None
    genome_index: Tensor | None = None
    if genome_path.exists():
        genome_ids, genome_cols, genome_matrix = read_labeled_matrix(genome_path)
        genome_index = _column_index(genome_cols, phenotype_ids, genome_path)

    return SiteDataset(
        site_id=site_id or data_dir.name,
        phenotype_ids=list(phenotype_ids),
        patient_ids=patient_ids,
        patient_phenotypes=patient_matrix,
        patient_col_index=patient_index,
        genome_ids=genome_ids,
        genome_phenotypes=genome_matrix,
        genome_col_index=genome_index,
    )
