#!/usr/bin/env python3
"""Run Davor's federated CF job (scripts/run_federated_cf_job.py, unchanged) on a data dir and score it.

    python Elakiya/federated_data/run_federated_cf.py --data_root data/ehr_lipids
    python Elakiya/federated_data/run_federated_cf.py --data_root data/federated --eval_only --embeddings <npz>

The runner is called with ``--skip_generate`` (otherwise it regenerates ``--data_root``) and with
``--output``/``--workspace`` outside the repository data, so no existing file is touched.

scripts/ contains no evaluation, so this wrapper adds one that is invariant to the rotation of the
latent space: for each table (nongenetic, genetic) it correlates the pairwise cosine similarities
of the learned phenotype embeddings with those of ``true_embeddings.pt`` (representational
similarity analysis, Pearson r over the upper triangle) and reports the same score for the
runner's random initialisation (``random_unit_params``) as a floor. Genetic scores only use
phenotypes with a non-zero true genetic vector and a non-zero column in at least one genome file
(terms without any associated locus cannot be learned from genetics).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _cos(x: np.ndarray) -> np.ndarray:
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ x.T


def rsa(learned: np.ndarray, true: np.ndarray) -> float:
    iu = np.triu_indices(len(true), 1)
    a, b = _cos(learned)[iu], _cos(true)[iu]
    return float(np.corrcoef(a, b)[0, 1])


def genome_columns(data_root: Path, ids: list[str]) -> set[int]:
    """Phenotype columns with at least one non-zero value in any genome file."""
    from federated_cf_data import read_labeled_matrix

    cols = set()
    for f in data_root.glob("site-*/genome_phenotypes.csv"):
        _, header, values = read_labeled_matrix(f)
        nz = (values != 0).any(dim=0).tolist()
        cols |= {ids.index(c) for c, keep in zip(header, nz) if keep}
    return cols


def neighbours(emb: np.ndarray, ids: list[str], labels: dict, query: str, k: int = 3) -> list[str]:
    sims = _cos(emb)[ids.index(query)]
    order = [i for i in np.argsort(-sims) if ids[i] != query][:k]
    return [f"{labels.get(ids[i], ids[i])} ({sims[i]:.2f})" for i in order]


def evaluate(data_root: Path, embeddings: Path) -> dict:
    import torch
    from federated_cf_embeddings import GENETIC_KEY, NONGENETIC_KEY, random_unit_params

    truth = torch.load(data_root / "true_embeddings.pt", weights_only=False)
    ids = list(truth["phenotype_ids"])
    learned = np.load(embeddings)
    init = random_unit_params(len(ids), learned["nongenetic_phenotype_embeddings"].shape[1])
    t_ng = truth["nongenetic"].numpy()
    t_g = truth["genetic"].numpy()
    keep = sorted(i for i in genome_columns(data_root, ids) if np.linalg.norm(t_g[i]) > 0)
    out = {
        "n_phenotypes": len(ids),
        "nongenetic_rsa": rsa(learned["nongenetic_phenotype_embeddings"], t_ng),
        "nongenetic_rsa_random_init": rsa(init[NONGENETIC_KEY], t_ng),
        "genetic_rsa": rsa(learned["genetic_phenotype_embeddings"][keep], t_g[keep]),
        "genetic_rsa_random_init": rsa(init[GENETIC_KEY][keep], t_g[keep]),
        "genetic_phenotypes_scored": len(keep),
    }
    labels_path = data_root / "phenotype_labels.csv"
    if labels_path.exists():
        import csv
        labels = {r["phenotype_id"]: r["label"] for r in csv.DictReader(labels_path.open())}
        for q in ("HP:0003141", "HP:0003233", "HP:0005978", "HP:0002099"):
            out[f"nearest_nongenetic[{labels[q]}]"] = neighbours(learned["nongenetic_phenotype_embeddings"], ids, labels, q)
        out["nearest_genetic[Elevated circulating LDL-C concentration]"] = neighbours(
            learned["genetic_phenotype_embeddings"][keep], [ids[i] for i in keep], labels, "HP:0003141")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=str(ROOT / "data" / "ehr_lipids"))
    ap.add_argument("--work", default="/tmp/metametagraphs_fedcf")
    ap.add_argument("--embeddings", help="existing npz (with --eval_only)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--num_rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    data_root = Path(args.data_root).resolve()
    work = Path(args.work) / data_root.name
    emb = Path(args.embeddings) if args.embeddings else work / "global_phenotype_embeddings.npz"
    if not args.eval_only:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_federated_cf_job.py"), "--data_root", str(data_root),
               "--skip_generate", "--workspace", str(work / "workspace"), "--output", str(emb),
               "--num_rounds", str(args.num_rounds), "--local_epochs", str(args.local_epochs), "--lr", str(args.lr)]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print(json.dumps(evaluate(data_root, emb), indent=2))


if __name__ == "__main__":
    main()
