#!/usr/bin/env python3
"""Load per-site patient–phenotype matrices.

Nongenetic CF uses binary ``patient_phenotypes.csv``. Genetic CF, when present,
uses ``genome_phenotypes.csv``: the same patients and traits, with PGS values
instead of 0/1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

PATIENT_MATRIX_NAME = "patient_phenotypes.csv"
GENOME_MATRIX_NAME = "genome_phenotypes.csv"
GENOME_MATRIX_NPZ_NAME = "genome_phenotypes.npz"
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
    genome_phenotypes: Tensor | None  # patient × phenotype PGS scores, or None
    genome_col_index: Tensor | None
    genome_row_index: Tensor | None = None  # COO row of each SNP–trait weight
    genome_entry_col: Tensor | None = None  # COO col into local genome columns
    genome_values: Tensor | None = None

    def has_genetic(self) -> bool:
        if self.genome_col_index is None:
            return False
        if self.genome_values is not None and int(self.genome_values.numel()) > 0:
            return True
        return self.genome_phenotypes is not None and self.genome_phenotypes.numel() > 0


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


def write_sparse_labeled_matrix(
    path: Path,
    row_ids: Sequence[str],
    col_ids: Sequence[str],
    row_index: Sequence[int],
    col_index: Sequence[int],
    values: Sequence[float],
) -> None:
    """Save a sparse row×column matrix (COO) plus axis labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_ids=np.array(list(row_ids), dtype=object),
        col_ids=np.array(list(col_ids), dtype=object),
        row_index=np.asarray(row_index, dtype=np.int32),
        col_index=np.asarray(col_index, dtype=np.int32),
        values=np.asarray(values, dtype=np.float32),
        n_rows=np.int32(len(row_ids)),
        n_cols=np.int32(len(col_ids)),
    )


def read_sparse_coo(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path, allow_pickle=True)
    row_ids = [str(x) for x in blob["row_ids"].tolist()]
    col_ids = [str(x) for x in blob["col_ids"].tolist()]
    return (
        row_ids,
        col_ids,
        np.asarray(blob["row_index"], dtype=np.int32),
        np.asarray(blob["col_index"], dtype=np.int32),
        np.asarray(blob["values"], dtype=np.float32),
    )


def read_sparse_labeled_matrix(path: Path) -> tuple[list[str], list[str], Tensor]:
    row_ids, col_ids, row_index, col_index, values = read_sparse_coo(path)
    dense = np.zeros((len(row_ids), len(col_ids)), dtype=np.float32)
    if values.size:
        dense[row_index, col_index] = values
    return row_ids, col_ids, torch.from_numpy(dense)


def subset_sparse_matrix(
    row_ids: Sequence[str],
    col_ids: Sequence[str],
    row_index: np.ndarray,
    col_index: np.ndarray,
    values: np.ndarray,
    keep_cols: Sequence[str] | None = None,
    max_rows: int = 0,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Keep a column subset / leading rows of a COO matrix without densifying."""
    kept_rows = list(row_ids)
    kept_cols = list(col_ids)
    row_index = np.asarray(row_index, dtype=np.int32)
    col_index = np.asarray(col_index, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)

    if max_rows and max_rows < len(kept_rows):
        mask = row_index < max_rows
        kept_rows = kept_rows[:max_rows]
        row_index = row_index[mask]
        col_index = col_index[mask]
        values = values[mask]

    if keep_cols is not None:
        lookup = {name: i for i, name in enumerate(kept_cols)}
        missing = [name for name in keep_cols if name not in lookup]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"genome matrix is missing phenotype columns: {preview}")
        keep_old = np.array([lookup[name] for name in keep_cols], dtype=np.int32)
        remap_col = np.full(len(kept_cols), -1, dtype=np.int32)
        remap_col[keep_old] = np.arange(len(keep_cols), dtype=np.int32)
        new_col = remap_col[col_index]
        mask = new_col >= 0
        row_index = row_index[mask]
        col_index = new_col[mask]
        values = values[mask]
        kept_cols = list(keep_cols)

    if row_index.size == 0:
        return [], kept_cols, row_index, col_index, values

    used = np.unique(row_index)
    remap_row = np.full(len(kept_rows), -1, dtype=np.int32)
    remap_row[used] = np.arange(len(used), dtype=np.int32)
    return (
        [kept_rows[i] for i in used.tolist()],
        kept_cols,
        remap_row[row_index],
        col_index,
        values,
    )


def subset_matrix_columns(
    row_ids: Sequence[str],
    col_ids: Sequence[str],
    matrix: Tensor,
    keep_cols: Sequence[str],
) -> tuple[list[str], list[str], Tensor]:
    lookup = {name: i for i, name in enumerate(col_ids)}
    missing = [name for name in keep_cols if name not in lookup]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"genome matrix is missing phenotype columns: {preview}")
    indexes = [lookup[name] for name in keep_cols]
    selected = matrix[:, indexes]
    nonzero_rows = selected.abs().sum(dim=1) > 0
    kept_row_ids = [row_ids[i] for i, keep in enumerate(nonzero_rows.tolist()) if keep]
    return kept_row_ids, list(keep_cols), selected[nonzero_rows]


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

    genome_npz = data_dir / GENOME_MATRIX_NPZ_NAME
    genome_csv = data_dir / GENOME_MATRIX_NAME
    genome_ids: list[str] = []
    genome_matrix: Tensor | None = None
    genome_index: Tensor | None = None
    genome_row_index: Tensor | None = None
    genome_entry_col: Tensor | None = None
    genome_values: Tensor | None = None
    if genome_csv.exists():
        genome_ids, genome_cols, genome_matrix = read_labeled_matrix(genome_csv)
        genome_index = _column_index(genome_cols, phenotype_ids, genome_csv)
    elif genome_npz.exists():
        genome_ids, genome_cols, row_index, col_index, values = read_sparse_coo(genome_npz)
        genome_index = _column_index(genome_cols, phenotype_ids, genome_npz)
        genome_row_index = torch.from_numpy(np.asarray(row_index, dtype=np.int64))
        genome_entry_col = torch.from_numpy(np.asarray(col_index, dtype=np.int64))
        genome_values = torch.from_numpy(np.asarray(values, dtype=np.float32))

    return SiteDataset(
        site_id=site_id or data_dir.name,
        phenotype_ids=list(phenotype_ids),
        patient_ids=patient_ids,
        patient_phenotypes=patient_matrix,
        patient_col_index=patient_index,
        genome_ids=genome_ids,
        genome_phenotypes=genome_matrix,
        genome_col_index=genome_index,
        genome_row_index=genome_row_index,
        genome_entry_col=genome_entry_col,
        genome_values=genome_values,
    )
