#!/usr/bin/env python3
"""Offline ablation of three model changes against the current prototype.

Runs the federated round loop without NVFLARE by calling the same local-training
and FedAvg math directly. Variants:

  base       current model: two independent phenotype tables, no bias, W = I
  bias       + per-phenotype bias on the patient relation
  coupled    one shared phenotype table serving both relations through learned W_r
  both       bias + coupled
  bias+mask  bias, with held-out cells masked out of the local training loss
  both+mask  both, with held-out cells masked out of the local training loss

The two ``+mask`` variants test whether the factorization term only looks harmful
because the unmasked loss trains held-out cells as explicit negatives, teaching
patient vectors to suppress the very cells that are later scored.

Every variant carrying a bias is scored twice: once with the full model, and once
as ``<name> (bias only)`` with the patient x phenotype term dropped. The gap
between those two rows is what the factorization is actually worth.

Evaluation follows PR #4: hide 10% of each site's patient x phenotype cells,
train on a copy with those cells zeroed, then freeze the learned table, refit
patient vectors on the visible cells only, and score the hidden cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from federated_cf_data import load_site, read_id_list  # noqa: E402

N_FACTORS = 32
N_ROUNDS = 5
LOCAL_EPOCHS = 20
LR = 0.05
HOLDOUT_FRACTION = 0.10
EVAL_EPOCHS = 300


def l2n(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC; returns nan when only one class is present."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = float(labels.sum())
    n_neg = float(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks within ties
    sorted_scores = scores[order]
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class Variant:
    name: str
    bias: bool
    coupled: bool
    mask_unobserved: bool = False


VARIANTS = [
    Variant("base", bias=False, coupled=False),
    Variant("bias", bias=True, coupled=False),
    Variant("coupled", bias=False, coupled=True),
    Variant("both", bias=True, coupled=True),
    Variant("bias+mask", bias=True, coupled=False, mask_unobserved=True),
    Variant("both+mask", bias=True, coupled=True, mask_unobserved=True),
]


@dataclass
class SiteSplit:
    site_id: str
    patient_ids: list[str]
    train: Tensor  # N x P_local, held-out cells forced to 0
    visible: Tensor  # N x P_local, 1 where the cell was not held out
    truth: Tensor  # N x P_local, original values
    cols_p: Tensor
    genome: Tensor | None
    cols_g: Tensor | None


def split_sites(data_root: Path, phenotype_ids: list[str], seed: int) -> list[SiteSplit]:
    rng = np.random.default_rng(seed)
    splits = []
    for site_dir in sorted(p for p in data_root.iterdir() if (p / "patient_phenotypes.csv").exists()):
        site = load_site(site_dir, phenotype_ids)
        truth = site.patient_phenotypes
        visible = torch.ones_like(truth)
        n_cells = truth.numel()
        hidden = rng.choice(n_cells, size=int(n_cells * HOLDOUT_FRACTION), replace=False)
        flat = visible.reshape(-1)
        flat[torch.as_tensor(hidden, dtype=torch.long)] = 0.0
        splits.append(
            SiteSplit(
                site_id=site.site_id,
                patient_ids=site.patient_ids,
                train=truth * visible,
                visible=visible,
                truth=truth,
                cols_p=site.patient_col_index,
                genome=site.genome_phenotypes,
                cols_g=site.genome_col_index,
            )
        )
    return splits


def local_update(
    split: SiteSplit,
    tables: dict[str, np.ndarray],
    variant: Variant,
    n_phenotypes: int,
    gen: torch.Generator,
) -> dict:
    """One site's local fit. Mirrors train_reconstruction, plus optional W_r and bias."""
    k = N_FACTORS
    shared = variant.coupled

    z_pat = nn.Parameter(l2n(torch.as_tensor(tables["z_pat"])))
    z_gen = z_pat if shared else nn.Parameter(l2n(torch.as_tensor(tables["z_gen"])))
    params: list[nn.Parameter] = [z_pat] if shared else [z_pat, z_gen]

    if shared:
        w_pat = nn.Parameter(torch.as_tensor(tables["w_pat"]).clone())
        w_gen = nn.Parameter(torch.as_tensor(tables["w_gen"]).clone())
        params += [w_pat, w_gen]
    else:
        w_pat = torch.eye(k)
        w_gen = torch.eye(k)

    if variant.bias:
        bias = nn.Parameter(torch.as_tensor(tables["bias"]).clone())
        params.append(bias)
    else:
        bias = torch.zeros(n_phenotypes)

    rows_p = nn.Parameter(0.1 * torch.randn(split.train.shape[0], k, generator=gen))
    params.append(rows_p)
    has_genome = split.genome is not None
    if has_genome:
        rows_g = nn.Parameter(0.1 * torch.randn(split.genome.shape[0], k, generator=gen))
        params.append(rows_g)

    optimizer = torch.optim.Adam(params, lr=LR)

    # Column masks: only phenotypes this site actually holds may receive gradient.
    observed_p = torch.zeros(n_phenotypes, 1)
    observed_p[split.cols_p] = 1.0
    observed_g = torch.zeros(n_phenotypes, 1)
    if has_genome:
        observed_g[split.cols_g] = 1.0
    observed_all = torch.maximum(observed_p, observed_g)

    for _ in range(LOCAL_EPOCHS):
        optimizer.zero_grad()
        zp = l2n(z_pat)
        logits = rows_p @ w_pat @ zp[split.cols_p].T + bias[split.cols_p]
        if variant.mask_unobserved:
            cell_loss = F.binary_cross_entropy_with_logits(logits, split.train, reduction="none")
            loss = (cell_loss * split.visible).sum() / split.visible.sum().clamp(min=1.0)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, split.train)
        if has_genome:
            zg = l2n(z_gen)
            predicted = rows_g @ w_gen @ zg[split.cols_g].T
            loss = loss + F.mse_loss(predicted, split.genome)
        loss.backward()
        if z_pat.grad is not None:
            z_pat.grad.mul_(observed_all if shared else observed_p)
        if not shared and z_gen.grad is not None:
            z_gen.grad.mul_(observed_g)
        if variant.bias and bias.grad is not None:
            bias.grad.mul_(observed_p.squeeze(-1))
        optimizer.step()
        with torch.no_grad():
            z_pat.copy_(l2n(z_pat))
            if not shared:
                z_gen.copy_(l2n(z_gen))

    n_patients = float(split.train.shape[0])
    n_genomes = float(split.genome.shape[0]) if has_genome else 0.0
    out = {
        "z_pat": l2n(z_pat).detach().numpy(),
        "mask_pat": observed_all.squeeze(-1).numpy() if shared else observed_p.squeeze(-1).numpy(),
        "w_pat_weight": n_patients,
        "w_gen_weight": n_genomes,
        "n_patients": n_patients,
        "n_genomes": n_genomes,
    }
    if not shared:
        out["z_gen"] = l2n(z_gen).detach().numpy()
        out["mask_gen"] = observed_g.squeeze(-1).numpy()
    if shared:
        out["w_pat"] = w_pat.detach().numpy()
        out["w_gen"] = w_gen.detach().numpy()
    if variant.bias:
        out["bias"] = bias.detach().numpy()
        out["mask_bias"] = observed_p.squeeze(-1).numpy()
    return out


def masked_average(values, masks, weights, previous):
    total = np.zeros_like(previous, dtype=np.float64)
    denom = np.zeros((previous.shape[0], 1), dtype=np.float64)
    for value, mask, weight in zip(values, masks, weights):
        if weight <= 0:
            continue
        m = mask.reshape(-1, 1)
        total += weight * m * value
        denom += weight * m
    averaged = np.where(denom > 0, total / np.maximum(denom, 1e-8), previous)
    return averaged


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def aggregate(updates: list[dict], tables: dict[str, np.ndarray], variant: Variant) -> dict[str, np.ndarray]:
    new = dict(tables)
    weights_pat = [u["n_patients"] + u["n_genomes"] if variant.coupled else u["n_patients"] for u in updates]
    new["z_pat"] = normalize_rows(
        masked_average([u["z_pat"] for u in updates], [u["mask_pat"] for u in updates], weights_pat, tables["z_pat"])
    )
    if not variant.coupled:
        new["z_gen"] = normalize_rows(
            masked_average(
                [u["z_gen"] for u in updates],
                [u["mask_gen"] for u in updates],
                [u["n_genomes"] for u in updates],
                tables["z_gen"],
            )
        )
    else:
        for key, weight_key in (("w_pat", "w_pat_weight"), ("w_gen", "w_gen_weight")):
            total_weight = sum(u[weight_key] for u in updates)
            if total_weight > 0:
                new[key] = sum(u[weight_key] * u[key] for u in updates) / total_weight
    if variant.bias:
        new["bias"] = masked_average(
            [u["bias"].reshape(-1, 1) for u in updates],
            [u["mask_bias"] for u in updates],
            [u["n_patients"] for u in updates],
            tables["bias"].reshape(-1, 1),
        ).reshape(-1).astype(np.float32)
    return new


def init_tables(n_phenotypes: int, variant: Variant, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tables = {"z_pat": normalize_rows(rng.normal(size=(n_phenotypes, N_FACTORS)).astype(np.float32))}
    if not variant.coupled:
        tables["z_gen"] = normalize_rows(rng.normal(size=(n_phenotypes, N_FACTORS)).astype(np.float32))
    else:
        tables["w_pat"] = np.eye(N_FACTORS, dtype=np.float32)
        tables["w_gen"] = np.eye(N_FACTORS, dtype=np.float32)
    tables["bias"] = np.zeros(n_phenotypes, dtype=np.float32)
    return tables


def run_federated(splits, tables, variant, n_phenotypes, seed):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(N_ROUNDS):
        updates = [local_update(s, tables, variant, n_phenotypes, gen) for s in splits]
        tables = aggregate(updates, tables, variant)
    return tables


def score_site(
    split: SiteSplit,
    tables: dict[str, np.ndarray],
    variant: Variant,
    seed: int,
    use_factors: bool = True,
) -> float:
    """Freeze the learned table, refit patient vectors on visible cells, score hidden cells.

    With ``use_factors=False`` the patient x phenotype term is dropped entirely and the
    hidden cells are ranked by the learned per-phenotype bias alone.
    """
    gen = torch.Generator().manual_seed(seed + 999)
    z = l2n(torch.as_tensor(tables["z_pat"]))[split.cols_p]
    w = torch.as_tensor(tables["w_pat"]) if variant.coupled else torch.eye(N_FACTORS)
    b = torch.as_tensor(tables["bias"])[split.cols_p] if variant.bias else torch.zeros(len(split.cols_p))
    basis = (w @ z.T).detach()

    hidden = split.visible.numpy() == 0
    if not use_factors:
        scores = np.tile(b.detach().numpy(), (split.train.shape[0], 1))
        return auc(split.truth.numpy()[hidden], scores[hidden])

    rows = nn.Parameter(0.1 * torch.randn(split.train.shape[0], N_FACTORS, generator=gen))
    optimizer = torch.optim.Adam([rows], lr=LR)
    for _ in range(EVAL_EPOCHS):
        optimizer.zero_grad()
        logits = rows @ basis + b
        losses = F.binary_cross_entropy_with_logits(logits, split.truth, reduction="none")
        (losses * split.visible).sum().div(split.visible.sum()).backward()
        optimizer.step()

    with torch.no_grad():
        scores = (rows @ basis + b).numpy()
    return auc(split.truth.numpy()[hidden], scores[hidden])


def prevalence_auc(split: SiteSplit) -> float:
    visible = split.visible.numpy()
    truth = split.truth.numpy()
    counts = (truth * visible).sum(axis=0)
    seen = np.maximum(visible.sum(axis=0), 1.0)
    prevalence = counts / seen
    scores = np.tile(prevalence, (truth.shape[0], 1))
    hidden = visible == 0
    return auc(truth[hidden], scores[hidden])


def prs_auc(split: SiteSplit, phenotype_ids: list[str], prs_path: Path) -> float:
    """Prevalence + own-PRS logistic baseline, fitted per phenotype on visible cells."""
    import csv

    if not prs_path.exists():
        return float("nan")
    with prs_path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)[1:]
        prs = {row[0]: np.asarray(row[1:], dtype=np.float64) for row in reader}
    column_of = {name: i for i, name in enumerate(header)}

    raw_ids = [pid.split("_", 1)[1] for pid in split.patient_ids]
    if not all(rid in prs for rid in raw_ids):
        return float("nan")
    matrix = np.stack([prs[rid] for rid in raw_ids])

    truth = split.truth.numpy()
    visible = split.visible.numpy()
    scores = np.zeros_like(truth, dtype=np.float64)
    for local_col, global_col in enumerate(split.cols_p.numpy()):
        name = phenotype_ids[global_col]
        if name not in column_of:
            continue
        x = matrix[:, column_of[name]]
        x = (x - x.mean()) / max(x.std(), 1e-8)
        seen = visible[:, local_col] == 1
        y = truth[seen, local_col]
        if y.sum() == 0 or y.sum() == y.size:
            continue
        # Newton-Raphson, ridge on the slope
        beta = np.zeros(2)
        design = np.column_stack([np.ones(seen.sum()), x[seen]])
        for _ in range(50):
            p = 1.0 / (1.0 + np.exp(-design @ beta))
            gradient = design.T @ (y - p) - np.array([0.0, beta[1]])
            hessian = design.T @ (design * (p * (1 - p))[:, None]) + np.diag([0.0, 1.0])
            beta += np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        scores[:, local_col] = beta[0] + beta[1] * x
    hidden = visible == 0
    return auc(truth[hidden], scores[hidden])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(ROOT / "data" / "synthgen_federated"))
    parser.add_argument("--prs", default=str(ROOT / "data" / "synthgen" / "prs.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "spike_results.json"))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    phenotype_ids = read_id_list(data_root / "phenotype_ids.txt")
    n_phenotypes = len(phenotype_ids)

    results: dict[str, dict[str, list[float]]] = {}
    baselines: dict[str, dict[str, list[float]]] = {"prevalence": {}, "prevalence+own PRS": {}, "random init": {}}

    for seed in args.seeds:
        splits = split_sites(data_root, phenotype_ids, seed)
        site_ids = [s.site_id for s in splits]

        for name, function in (("prevalence", prevalence_auc),):
            for split in splits:
                baselines[name].setdefault(split.site_id, []).append(function(split))
        for split in splits:
            baselines["prevalence+own PRS"].setdefault(split.site_id, []).append(
                prs_auc(split, phenotype_ids, Path(args.prs))
            )

        random_variant = Variant("random", bias=False, coupled=False)
        random_tables = init_tables(n_phenotypes, random_variant, seed)
        for split in splits:
            baselines["random init"].setdefault(split.site_id, []).append(
                score_site(split, random_tables, random_variant, seed)
            )

        for variant in VARIANTS:
            tables = init_tables(n_phenotypes, variant, seed)
            trained = run_federated(splits, tables, variant, n_phenotypes, seed)
            arms = [(variant.name, True)]
            if variant.bias:
                # Same trained tables, scored without the patient x phenotype term. The gap
                # between these two rows is what the factorization is worth.
                arms.append((f"{variant.name} (bias only)", False))
            for arm_name, use_factors in arms:
                for split in splits:
                    results.setdefault(arm_name, {}).setdefault(split.site_id, []).append(
                        score_site(split, trained, variant, seed, use_factors=use_factors)
                    )
                row = " ".join(f"{s}={results[arm_name][s][-1]:.3f}" for s in site_ids)
                print(f"seed {seed}  {arm_name:22s}  {row}", flush=True)

    def summarize(table: dict[str, list[float]]) -> str:
        cells = []
        pooled = []
        for site in sorted(table):
            values = np.asarray(table[site], dtype=float)
            pooled.extend(values.tolist())
            cells.append(f"{site}: {values.mean():.3f} +/- {values.std():.3f}")
        cells.append(f"mean: {np.mean(pooled):.3f}")
        return "   ".join(cells)

    print(f"\n=== held-out AUC on patient x phenotype cells, {data_root.name} ===")
    print(f"=== mean +/- sd over seeds {args.seeds} ===")
    report = [
        "random init",
        "base",
        "bias (bias only)",
        "bias",
        "coupled",
        "both (bias only)",
        "both",
        "bias+mask (bias only)",
        "bias+mask",
        "both+mask (bias only)",
        "both+mask",
        "prevalence",
        "prevalence+own PRS",
    ]
    for name in report:
        table = results.get(name) or baselines.get(name)
        if table:
            print(f"{name:22s} {summarize(table)}")

    Path(args.out).write_text(
        json.dumps({"variants": results, "baselines": baselines, "seeds": args.seeds}, indent=2) + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
