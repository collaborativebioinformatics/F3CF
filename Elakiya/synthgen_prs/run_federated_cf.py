#!/usr/bin/env python3
"""Run Davor's federated CF job (scripts/run_federated_cf_job.py, unchanged) on data/synthgen_federated and score it.

    python Elakiya/synthgen_prs/run_federated_cf.py --repeats 3

Steps (everything is written under ``--work``, never under data/):

1. Hold-out split: in every site's patient_phenotypes.csv, 10% of the cells (seed 0) are set to 0
   in a training copy. Davor's binary loss is an unmasked BCE and cannot take NaN, so hidden cells
   look like negatives during training (the usual implicit-feedback protocol). Genome files are
   copied unchanged; in the prototype the genetic and nongenetic tables are trained by separate
   losses, so the genome data cannot leak into the patient-phenotype evaluation.
2. Federated run: the runner is called with ``--skip_generate`` and ``--output``/``--workspace``
   under ``--work``.
3. Centralized baseline: the same runner on ONE pooled site (all 2500 patients, all 100 phenotype
   columns, same hidden cells zeroed) plus the pooled genome effects. It sees the columns that the
   sites dropped, so it is an optimistic upper bound for this model.
4. Evaluation, per site: with the learned global nongenetic table frozen, patient vectors are
   fitted locally on the non-hidden cells (masked BCE, Adam), then hidden cells are scored as
   ``u_i . v_j``. Reported: AUC over all hidden cells, and mean per-phenotype AUC (removes the
   prevalence signal). Baselines use the same hidden cells: random-init table (runner's
   ``random_unit_params``), site prevalence, and "prevalence + own PRS" = logit(prev_j) + beta_jj * z_ij
   using the site's own genome_phenotypes.csv effect (site-3 has none, so it is skipped there).
5. Pooled-reference RSA (as in PR #3's wrapper): Pearson r between pairwise cosine similarities of
   the learned tables and of ``true_embeddings.pt`` (a pooled-data reference, see build_sites.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from build_sites import load_synthgen, logistic_effects, standardize  # noqa: E402
from federated_cf_data import load_site, read_id_list, write_id_list, write_labeled_matrix  # noqa: E402
from federated_cf_embeddings import GENETIC_KEY, NONGENETIC_KEY, random_unit_params  # noqa: E402

DATA = ROOT / "data" / "synthgen_federated"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    labels = labels.astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean_col_auc(scores: np.ndarray, labels: np.ndarray, hidden: np.ndarray) -> float:
    vals = [auc(scores[hidden[:, j], j], labels[hidden[:, j], j]) for j in range(labels.shape[1])]
    return float(np.nanmean(vals))


def _cos(x: np.ndarray) -> np.ndarray:
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ x.T


def rsa(learned: np.ndarray, ref: np.ndarray) -> float:
    iu = np.triu_indices(len(ref), 1)
    return float(np.corrcoef(_cos(learned)[iu], _cos(ref)[iu])[0, 1])


def write_site(site_dir: Path, ids, cols, y: np.ndarray, genome: tuple | None) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    write_labeled_matrix(site_dir / "patient_phenotypes.csv", ids, cols, torch.as_tensor(y, dtype=torch.float64))
    if genome is not None:
        g_ids, g_cols, g = genome
        write_labeled_matrix(site_dir / "genome_phenotypes.csv", g_ids, g_cols, torch.as_tensor(g, dtype=torch.float64))


def prepare(work: Path, mask_frac: float, seed: int) -> dict:
    """Write masked federated and pooled copies; return hidden masks and site data."""
    ids = read_id_list(DATA / "phenotype_ids.txt")
    rng = np.random.default_rng(seed)
    fed, pooled = work / "data_federated", work / "data_pooled"
    for d in (fed, pooled):
        shutil.rmtree(d, ignore_errors=True)
        write_id_list(d / "phenotype_ids.txt", ids)
    shutil.copy(DATA / "true_embeddings.pt", fed / "true_embeddings.pt")
    prs, phen, _ = load_synthgen()
    sites = {}
    pooled_hidden = np.zeros(phen.shape, dtype=bool)
    for site_dir in sorted(p for p in DATA.iterdir() if (p / "patient_phenotypes.csv").exists()):
        s = load_site(site_dir, ids)
        y = s.patient_phenotypes.numpy()
        hidden = rng.random(y.shape) < mask_frac
        genome = None
        if s.genome_phenotypes is not None:
            genome = (s.genome_ids, [ids[i] for i in s.genome_col_index], s.genome_phenotypes.numpy())
        cols = [ids[i] for i in s.patient_col_index]
        write_site(fed / s.site_id, s.patient_ids, cols, np.where(hidden, 0, y), genome)
        raw = [p.split("_", 1)[1] for p in s.patient_ids]
        rows = phen.index.get_indexer(raw)
        pooled_hidden[np.ix_(rows, s.patient_col_index.numpy())] = hidden
        sites[s.site_id] = dict(y=y, hidden=hidden, cols=s.patient_col_index.numpy(), rows=rows,
                                z=standardize(prs.iloc[rows].to_numpy()),
                                genome=None if genome is None else dict(zip(genome[0], genome[2])), genome_cols=None if genome is None else s.genome_col_index.numpy())
    yall = phen.to_numpy()
    beta, _ = logistic_effects(standardize(prs.to_numpy()), yall)
    write_site(pooled / "site-pooled", [f"pooled_{i}" for i in phen.index], ids, np.where(pooled_hidden, 0, yall),
               ([f"pooled_prs_{c}" for c in prs.columns], ids, np.round(beta, 5)))
    return dict(ids=ids, sites=sites, pooled=dict(y=yall, hidden=pooled_hidden))


def run_job(data_root: Path, out: Path, args) -> Path:
    shutil.rmtree(out, ignore_errors=True)
    emb = out / "global_phenotype_embeddings.npz"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_federated_cf_job.py"), "--data_root", str(data_root),
           "--skip_generate", "--workspace", str(out / "workspace"), "--output", str(emb),
           "--num_rounds", str(args.num_rounds), "--local_epochs", str(args.local_epochs), "--lr", str(args.lr)]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL if args.quiet else None)
    return emb


def fold_in(table: np.ndarray, y: np.ndarray, train: np.ndarray, steps: int = 300, lr: float = 0.05) -> np.ndarray:
    """Fit local patient vectors against a frozen phenotype table on the training cells only; return logits."""
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.as_tensor(table, dtype=torch.float32), dim=1)
    t = torch.as_tensor(y, dtype=torch.float32)
    m = torch.as_tensor(train, dtype=torch.float32)
    u = torch.nn.Parameter(0.1 * torch.randn(y.shape[0], v.shape[1]))
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(u @ v.T, t, reduction="none") * m).sum() / m.sum()
        loss.backward()
        opt.step()
    return (u @ v.T).detach().numpy()


def score_sites(table: np.ndarray, prep: dict) -> dict:
    out, all_s, all_l = {}, [], []
    for name, s in prep["sites"].items():
        h = s["hidden"]
        logits = fold_in(table[s["cols"]], s["y"], ~h)
        out[name] = {"auc": auc(logits[h], s["y"][h]), "mean_pheno_auc": mean_col_auc(logits, s["y"], h)}
        all_s.append(logits[h]); all_l.append(s["y"][h])
    out["all_sites"] = {"auc": auc(np.concatenate(all_s), np.concatenate(all_l))}
    return out


def baselines(prep: dict) -> dict:
    ids = prep["ids"]
    out = {}
    for name, s in prep["sites"].items():
        h, y = s["hidden"], s["y"]
        prev = np.clip((y * ~h).sum(0) / np.maximum((~h).sum(0), 1), 1e-3, 1 - 1e-3)
        base = np.broadcast_to(np.log(prev / (1 - prev)), y.shape)
        row = {"prevalence_auc": auc(base[h], y[h])}
        if s["genome"] is not None:
            # own-PRS effect beta_jj from this site's genome_phenotypes.csv (row prs_<pheno j>, column j)
            bjj = np.array([s["genome"][f"{name}_prs_{ids[c]}"][k] for k, c in enumerate(s["genome_cols"])])
            zc = s["z"][:, s["cols"]]
            sc = base + bjj[None, :] * zc
            row.update({"prev_plus_own_prs_auc": auc(sc[h], y[h]), "own_prs_mean_pheno_auc": mean_col_auc(zc, y, h)})
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/metametagraphs_synthgen_prs")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mask_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--no_central", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="hide NVFlare simulator stdout")
    args = ap.parse_args()
    work = Path(args.work)
    prep = prepare(work, args.mask_frac, args.seed)
    ref = torch.load(DATA / "true_embeddings.pt", weights_only=False)
    ref_ng, ref_g = ref["nongenetic"].numpy(), ref["genetic"].numpy()
    init = random_unit_params(len(prep["ids"]), 32)
    report = {"baselines": baselines(prep), "random_init": score_sites(init[NONGENETIC_KEY], prep)}
    report["random_init"].update({"rsa_nongenetic": rsa(init[NONGENETIC_KEY], ref_ng), "rsa_genetic": rsa(init[GENETIC_KEY], ref_g)})
    runs = []
    for r in range(args.repeats):
        res = {}
        emb = np.load(run_job(work / "data_federated", work / f"run{r}_federated", args))
        res["federated"] = score_sites(emb[NONGENETIC_KEY], prep)
        res["federated"].update({"rsa_nongenetic": rsa(emb[NONGENETIC_KEY], ref_ng), "rsa_genetic": rsa(emb[GENETIC_KEY], ref_g)})
        if not args.no_central:
            emb = np.load(run_job(work / "data_pooled", work / f"run{r}_central", args))
            p = prep["pooled"]
            logits = fold_in(emb[NONGENETIC_KEY], p["y"], ~p["hidden"])
            res["central"] = score_sites(emb[NONGENETIC_KEY], prep)
            res["central"]["pooled_foldin_auc"] = auc(logits[p["hidden"]], p["y"][p["hidden"]])
            res["central"].update({"rsa_nongenetic": rsa(emb[NONGENETIC_KEY], ref_ng), "rsa_genetic": rsa(emb[GENETIC_KEY], ref_g)})
        runs.append(res)
        print(json.dumps({"run": r, **res}, indent=1), flush=True)
    report["runs"] = runs
    (work / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("baselines", "random_init")}, indent=1))
    print(f"full report: {work / 'report.json'}")


if __name__ == "__main__":
    main()
