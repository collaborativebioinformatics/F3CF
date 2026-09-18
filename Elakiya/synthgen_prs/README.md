# Synthgen PRS data in the federated CF site layout

This folder turns Sebastian's `data/synthgen/` (commits `422661e` "Added synthetic PRS" and
`7f85d90` "Create drugs.csv") into the site layout that Davor's prototype (`scripts/federated_cf_*.py`)
reads. It adds PRS-to-phenotype effect sizes (the G x P matrix) and patient-to-phenotype matrices,
then runs his NVFlare job unchanged and scores it. We assume `data/synthgen/` is the content of the
Slack file `multi_prs_benchmark_100_traits.zip` (we did not have the zip to check).

Nothing outside `data/synthgen_federated/` and `Elakiya/synthgen_prs/` is added or changed. The raw
synthgen CSVs are read from `data/synthgen/` and not copied.

## What synthgen looks like (measured with `build_synthgen_sites.py` inputs)

| file | shape | content |
|---|---|---|
| `prs.csv` | 2500 x 100 (`id1`..`id2500`, `pheno0`..`pheno99`) | PRS, each column mean 0, SD 1, no NaN |
| `phenotypes.csv` | 2500 x 100, same ids and columns | binary; prevalence cycles 0.05, 0.094, ..., 0.45 every 10 traits (exact quantile counts, consistent with a liability-threshold model); 13 to 41 traits per patient |
| `drugs.csv` | 2500 x 12 | binary; 8 of 12 drugs are exact copies of one phenotype (corr 1.0), the rest partly |

- Column `prs.pheno_k` corresponds to `phenotypes.pheno_k`: for all 100 traits the own PRS is the
  strongest predictor (corr 0.28 to 0.43, AUC 0.68 to 0.85, mean 0.75).
- Cross-trait signal is genetic: PRS columns are correlated with each other (|r| up to 0.52, 142
  pairs above 0.3), and the off-diagonal PRS-to-phenotype correlations follow that pattern (r = 0.79
  with the PRS-PRS correlations).
- Phenotype-phenotype correlations are weak (|r| <= 0.12). So the patient x phenotype matrix alone
  has little collaborative structure beyond prevalence; see Results.
- No README or commit message describes the generator; the above is inferred from the data only.

## Layout (`data/synthgen_federated/`, 0.8 MB total, largest file 196 KB)

```
phenotype_ids.txt              100 global ids (pheno0..pheno99)
sites.json                     sites, seed, ridge lambda, dropped phenotypes per site
true_embeddings.pt             POOLED-DATA REFERENCE (see below), not synthgen ground truth
site-1/ patient_phenotypes.csv 1000 x 92   rows site-1_id<k>
        genome_phenotypes.csv   100 x 92   rows site-1_prs_pheno<k>, per-SD log-OR
        patient_drugs.csv      1000 x 12   sidecar, ignored by the prototype
site-2/ (same)                  800 x 90, 100 x 90, 800 x 12
site-3/ patient_phenotypes.csv  700 x 94   genome-less site (no genome_phenotypes.csv)
        patient_drugs.csv       700 x 12
```

Patients are split 1000/800/700 by a seeded permutation (seed 0); each site drops 8/10/6 phenotype
columns (like `"drop"` in Davor's `sites.json`), and the same columns are dropped from its genome file.
Row ids keep the synthgen id so rows can be traced back. Genome rows are site-local entities in the
prototype (their row embeddings stay at the client; only the P x 32 phenotype tables are FedAvg'd),
so they carry the site prefix, like Davor's `site-1_rs0000`.

## PRS-to-phenotype effect sizes

Cell `(k, j)` of `genome_phenotypes.csv` is the log odds ratio per standard deviation of PRS k for
phenotype j, from the univariate logistic regression `logit P(y_j = 1) = a + b z_k` fitted on that
site's patients only, with `z_k` the PRS standardised within the site. Only this 100 x P summary
matrix is needed by the model, so no patient-level genotype data leaves a site.

- Why per-SD OR: it is the standard effect size for a PGS and a binary trait. The PGS Catalog
  documentation lists, under performance metrics, "Standardized effect sizes, per standard deviation
  [SD] change in PGS ... Odds ratios (OR) and/or Hazard ratios (HR) for dichotomous traits"
  (https://www.pgscatalog.org/docs/ ; catalog paper: Lambert et al. 2021, Nat Genet 53:420-425,
  https://doi.org/10.1038/s41588-021-00783-5). The PGS reporting standard (PRS-RS; Wand et al. 2021,
  Nature 591:211-219, https://doi.org/10.1038/s41586-021-03243-6) asks studies to report the PGS
  effect size and how the score was scaled; standardising within site makes the scale explicit.
  General PRS association practice: Choi, Mak and O'Reilly 2020, Nat Protoc 15:2759-2772,
  https://doi.org/10.1038/s41596-020-0353-1.
- Log scale (not OR) because Davor's genetic loss is an MSE on real values and log-OR is symmetric
  around 0.
- Fit: vectorised Newton-Raphson with a ridge penalty `lambda * b^2 / 2`, `lambda = 1` on the slope
  only (intercept free). At n = 800 to 1000 this changes estimates negligibly (the test checks the
  unpenalised fit against scipy BFGS to 1e-4) but keeps `b` finite under complete separation. A
  column with no cases or no controls is written as 0 because the genome loss cannot take NaN. For
  small sites, Firth's bias-reduced logistic regression (Firth 1993, Biometrika 80:27-38,
  https://doi.org/10.1093/biomet/80.1.27) is the principled alternative; it is not needed here
  (prevalence >= 5%, no separation observed).
- Sanity (tested): at both genome sites every own-trait effect `b(prs_pheno_k, pheno_k)` is > 0.5
  and is the largest value in its column. Example at site-1: `prs_pheno0 -> pheno0` = 1.52.

## Reference embeddings (`true_embeddings.pt`)

Synthgen's generative factors are not published, so this file holds a pooled-data reference with
the same keys as Davor's (`nongenetic`, `genetic`, `phenotype_ids`, plus `note`):
nongenetic = top-32 eigenvectors of the pooled 2500-patient phenotype correlation matrix times
sqrt(eigenvalue); genetic = top-32 right singular vectors of the pooled 100 x 100 per-SD log-OR
matrix times the singular values. It is only used for the similarity (RSA) score below.

## Evaluation (`run_federated_cf.py`)

1. 10% of each site's patient x phenotype cells (seed 0) are hidden: set to 0 in a training copy
   under `--work` (Davor's BCE is unmasked, NaN would break it).
2. Federated: `scripts/run_federated_cf_job.py --skip_generate` on the three masked sites, output
   and workspace under `--work`.
3. Centralized baseline: the same runner on one pooled site (2500 patients, all 100 columns, same
   hidden cells zeroed, pooled genome effects). It also sees the columns the sites dropped, so it is
   optimistic.
4. Scoring at each site: freeze the learned global nongenetic table, fit patient vectors on the
   visible cells (masked BCE, 300 Adam steps), score hidden cells. Reported: AUC over all hidden
   cells and mean per-phenotype AUC (which removes the prevalence signal). Baselines on the same
   cells: random-init table (`random_unit_params`), site prevalence, and prevalence + own PRS
   (`logit(prev_j) + b_jj z_ij` using the site's own genome file).
5. RSA as in PR #3: Pearson r between pairwise cosine similarities of learned tables and the
   pooled reference, with the random init as floor.

Davor's client uses unseeded `torch.randn`, so the job was run 3 times (hidden cells fixed).

## Commands

```
pip install -r requirements.txt pandas scipy pytest     # tested: torch 2.14 (CPU), nvflare 2.9.0
python Elakiya/synthgen_prs/build_synthgen_sites.py               # rebuild data/synthgen_federated (6 s)
python -m pytest Elakiya/synthgen_prs/tests -q           # 8 passed
python Elakiya/synthgen_prs/run_federated_cf.py --repeats 3 --quiet   # about 3.5 min on CPU
```

Results go to `/tmp/metametagraphs_synthgen_prs/report.json` (change with `--work`).

## Results (default runner settings: 5 rounds, 20 local epochs, lr 0.05, k = 32; mean +/- SD over 3 runs)

Held-out AUC on hidden patient x phenotype cells:

| model | site-1 | site-2 | site-3 | all sites | mean per-phenotype AUC (s1 / s2 / s3) |
|---|---|---|---|---|---|
| random-init table | 0.490 | 0.491 | 0.518 | 0.498 | 0.493 / 0.500 / 0.515 |
| federated (3 sites) | 0.570 +/- 0.007 | 0.573 +/- 0.004 | 0.570 +/- 0.006 | 0.571 +/- 0.002 | 0.505 / 0.513 / 0.504 |
| centralized (pooled, same model) | 0.569 +/- 0.001 | 0.568 +/- 0.014 | 0.576 +/- 0.002 | 0.571 +/- 0.005 | 0.500 / 0.498 / 0.500 |
| site prevalence | 0.685 | 0.696 | 0.697 | | 0.5 by construction |
| prevalence + own PRS | 0.791 | 0.792 | n/a (no genome) | | own PRS only: 0.744 / 0.739 |

Similarity to the pooled reference (RSA, Pearson r):

| table | random init | federated | centralized |
|---|---|---|---|
| nongenetic | 0.010 | 0.038 +/- 0.015 | 0.043 +/- 0.009 |
| genetic | 0.014 | 0.470 +/- 0.047 | 0.568 +/- 0.023 |

Reading:
- The genetic table learns the cross-trait PRS structure (RSA 0.47 federated vs 0.01 floor); the
  federated cost against pooling is about 0.10 RSA, with only 2 of 3 sites holding genome data.
- On patient x phenotype reconstruction the model beats random (0.57 vs 0.50) but is below plain
  prevalence (0.69), and per-phenotype AUC is about 0.5, federated and centralized alike. Two
  reasons: (a) synthgen phenotypes are nearly uncorrelated given no PRS, so there is little
  collaborative signal to find; (b) the model has no per-phenotype bias term and trains the
  genetic and nongenetic tables with separate losses, so PRS information, which is where the signal
  is (prevalence + own PRS reaches 0.79), cannot reach patient predictions.
- Federated equals centralized here, so the federation itself costs nothing on this task; the
  limitation is the model / data fit, not the FedAvg.

## Limitations and unverified points

- `data/synthgen` has no generator description; the liability-threshold reading is an inference.
  That it matches `multi_prs_benchmark_100_traits.zip` is an assumption.
- Patients in synthgen carry both PRS and phenotypes, but the prototype has no patient-to-PRS
  relation, so per-patient PRS is only used to compute site-level effects and the baseline.
- `patient_drugs.csv` is written for future use; the current loader ignores it.
- `true_embeddings.pt` is derived from the same pooled data (not independent ground truth).
- Hidden cells are shown to the model as 0 during training (implicit-feedback protocol).
- Only default runner hyperparameters were run; 3 repeats only.

## Relation to PR #3

PR #3 (`elakiya/ehr-data-for-federated-cf`) adds a lipid EHR dataset in the same layout and a
wrapper with the RSA score; this folder reuses that wrapper style and score and adds the held-out
AUC and centralized baseline. It does not depend on or modify PR #3's files.
