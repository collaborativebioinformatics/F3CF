#!/usr/bin/env python3
"""Build a PGS × phenotype matrix by aggregating SNP effect weights.

Each row is one polygenic score. Each column is a Catalog trait. The cell is
the sum of that score's SNP ``effect_weight`` values on its mapped trait
(the PRS recipe collapsed to one number, with every variant treated as
dosage 1). Other traits are zero: Catalog scores are trait-specific.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import torch

from federated_cf_data import write_id_list, write_labeled_matrix

DEFAULT_TRAIT_MAP = "data/pgs/trait_to_score.csv"
DEFAULT_SCORING_DIR = "data/pgs/scoring_files"
DEFAULT_OUTPUT = "data/pgs/pgs_phenotype_effects.csv"


def sum_effect_weights(path: Path) -> tuple[float, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    total = 0.0
    n_variants = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header: list[str] | None = None
        for raw in handle:
            if raw.startswith("#") or not raw.strip():
                continue
            fields = raw.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            row = {name: fields[i] if i < len(fields) else "" for i, name in enumerate(header)}
            if row.get("is_haplotype", "").lower() == "true":
                continue
            try:
                weight = float(row["effect_weight"])
            except (KeyError, ValueError):
                continue
            total += weight
            n_variants += 1
    return total, n_variants


def unique_phenotype_ids(raw_ids: list[str]) -> list[str]:
    if len(raw_ids) == len(set(raw_ids)):
        return raw_ids
    seen: dict[str, int] = {}
    unique: list[str] = []
    for trait_id in raw_ids:
        count = seen.get(trait_id, 0)
        seen[trait_id] = count + 1
        unique.append(trait_id if count == 0 else f"{trait_id}__{count}")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait-map", default=DEFAULT_TRAIT_MAP)
    parser.add_argument("--scoring-dir", default=DEFAULT_SCORING_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trait_map = Path(args.trait_map)
    scoring_dir = Path(args.scoring_dir)
    output = Path(args.output)

    with trait_map.open(encoding="utf-8", newline="") as handle:
        traits = list(csv.DictReader(handle))
    if not traits:
        raise SystemExit(f"no traits in {trait_map}")

    phenotype_ids = unique_phenotype_ids([row["trait_id"].replace("|", "+") for row in traits])
    pgs_ids: list[str] = []
    row_index: list[int] = []
    col_index: list[int] = []
    values: list[float] = []
    n_snps: list[int] = []
    aggregates: list[float] = []

    for col, row in enumerate(traits):
        pgs_id = row["pgs_id"]
        scoring_path = scoring_dir / f"{pgs_id}.txt.gz"
        if not scoring_path.exists():
            scoring_path = scoring_dir / f"{pgs_id}.txt"
        if not scoring_path.exists():
            print(f"skip missing scoring file {pgs_id}", flush=True)
            continue
        total, n_variants = sum_effect_weights(scoring_path)
        if n_variants == 0:
            print(f"skip empty scoring file {pgs_id}", flush=True)
            continue
        pgs_ids.append(pgs_id)
        row_index.append(len(pgs_ids) - 1)
        col_index.append(col)
        values.append(total)
        n_snps.append(n_variants)
        aggregates.append(total)
        print(
            f"{pgs_id} -> {phenotype_ids[col]}  "
            f"sum(effect_weight)={total:.6g} over {n_variants} SNPs",
            flush=True,
        )

    matrix = torch.zeros(len(pgs_ids), len(phenotype_ids), dtype=torch.float32)
    if values:
        matrix[row_index, col_index] = torch.tensor(values, dtype=torch.float32)
    write_labeled_matrix(output, pgs_ids, phenotype_ids, matrix)
    write_id_list(output.with_name("phenotype_ids.txt"), phenotype_ids)
    write_id_list(output.with_name("pgs_ids.txt"), pgs_ids)
    labels_path = output.with_name("phenotype_labels.csv")
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phenotype_id", "trait_label", "pgs_id", "variants_used", "aggregated_effect"],
        )
        writer.writeheader()
        pgs_lookup = {pgs_id: i for i, pgs_id in enumerate(pgs_ids)}
        for phenotype_id, row in zip(phenotype_ids, traits):
            idx = pgs_lookup.get(row["pgs_id"])
            writer.writerow(
                {
                    "phenotype_id": phenotype_id,
                    "trait_label": row["trait_label"],
                    "pgs_id": row["pgs_id"],
                    "variants_used": "" if idx is None else n_snps[idx],
                    "aggregated_effect": "" if idx is None else f"{aggregates[idx]:.8g}",
                }
            )

    n_rows, n_cols, n_nz = len(pgs_ids), len(phenotype_ids), len(values)
    density = n_nz / max(n_rows * n_cols, 1)
    print(
        f"Wrote {output}  shape={n_rows} PGS × {n_cols} phenotypes  "
        f"nonzero={n_nz}  density={density:.6f}"
    )


if __name__ == "__main__":
    main()
