# EHR-style HPO data for the federated CF prototype (`data/ehr_lipids`)

This is a drop-in dataset for Davor's federated collaborative filtering job (`scripts/`). It has
the same layout as `data/federated`, but:

- the columns are real HPO terms;
- the patient matrices come from EHR-style rules (lab thresholds, then diagnoses);
- the genome matrices come from published lead-variant effect sizes.

No existing file is modified.

## Layout (identical to `data/federated`)

```
data/ehr_lipids/
  phenotype_ids.txt                 32 HPO ids (global catalog)
  sites.json                        site_id, n_patients, n_genomes, drop
  site-1/patient_phenotypes.csv     400 x 29 binary (row_id = site-1_pNNNN)
  site-1/genome_phenotypes.csv      14 x 29 effects (row_id = site-1_<rsid>)
  site-2/patient_phenotypes.csv     250 x 27
  site-2/genome_phenotypes.csv      14 x 27
  site-3/patient_phenotypes.csv     300 x 28 (patient-only, like data/federated)
  true_embeddings.pt                {"nongenetic": 32 x 9, "genetic": 32 x 8, "phenotype_ids", "note"}
  effect_sources.csv                provenance per locus (sidecar)
  phenotype_labels.csv              HPO id -> label (sidecar)
  build_summary.json                calibration constants and per-site prevalences (sidecar)
```

Rows in `genome_phenotypes.csv` are real rsIDs, prefixed with the site id as in Davor's files.
Cells are 0 where a locus has no path to the term. NaN is never written, because
`train_reconstruction` uses an unmasked MSE and a NaN cell would make the loss NaN.

## Regenerate, run, test

```bash
pip install -r requirements.txt                               # torch, numpy, nvflare (repo root)
python Elakiya/federated_data/build_sites.py                  # writes data/ehr_lipids (deterministic, seed 0)
python Elakiya/federated_data/run_federated_cf.py --data_root data/ehr_lipids
python -m pytest -q Elakiya/federated_data/tests              # 8 tests
```

To use Davor's runner directly, pass `--skip_generate` (otherwise it regenerates `--data_root` with
random data) and an `--output` outside `data/federated`:

```bash
python scripts/run_federated_cf_job.py --data_root data/ehr_lipids --skip_generate \
    --workspace /tmp/fedcf/ws --output /tmp/fedcf/ehr_lipids_embeddings.npz
```

## HPO terms

All 32 ids and labels were checked in `hp.obo` release `hp/releases/2026-09-01`
(https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo); none is
obsolete. They are listed in `hpo_terms.csv`.

| group | terms |
| --- | --- |
| lipids | Abnormal circulating lipid concentration HP:0003119, Hyperlipidemia HP:0003077, Hypercholesterolemia HP:0003124, Elevated circulating LDL-C concentration HP:0003141, Decreased circulating HDL-C concentration HP:0003233, Hypertriglyceridemia HP:0002155, Elevated circulating lipoprotein(a) concentration HP:6000521, Elevated circulating apolipoprotein B concentration HP:0031798, Tendon xanthomatosis HP:0010874, Corneal arcus HP:0001084, Xanthelasma HP:0001114 |
| cardiovascular | Hypertension HP:0000822, Coronary artery atherosclerosis HP:0001677, Myocardial infarction HP:0001658, Angina pectoris HP:0001681, Ischemic stroke HP:0002140, Left ventricular hypertrophy HP:0001712, Hypertensive retinopathy HP:0001095 |
| metabolic | Diabetes mellitus HP:0000819, Type II diabetes mellitus HP:0005978, Hyperglycemia HP:0003074, Insulin resistance HP:0000855, Obesity HP:0001513, Hepatic steatosis HP:0001397, Retinopathy HP:0000488, Proteinuria HP:0000093, Chronic kidney disease HP:0012622 |
| respiratory / allergy | Asthma HP:0002099, Wheezing HP:0030828, Allergic rhinitis HP:0003193, Eczematoid dermatitis HP:0000964, Dyspnea HP:0002094 |

Note: the current HPO label of HP:0003141 is "Elevated circulating LDL-C concentration". The label
"Increased LDL cholesterol concentration" is older.

## Effect sizes (genome x phenotype)

See `data/ehr_lipids/effect_sources.csv` for per-locus provenance.

| loci | effect | source |
| --- | --- | --- |
| PCSK9 rs2479409, SORT1 rs629301, LDLR rs6511720, APOE rs4420638, APOB rs1367117, HMGCR rs12916 (LDL-C / TC); CETP rs3764261, LIPC rs1532085 (HDL-C); LPL rs12678919, APOA1 cluster rs964184, GCKR rs1260326 (TG) | mg/dL per minor allele, minor allele frequency | Teslovich et al. 2010, Nature 466:707 (PMID 20686565), Table 1 |
| TCF7L2 rs7903146 (T2D) | pooled allelic OR 1.46 [1.42-1.51] | Cauchi et al. 2007, J Mol Med 85:777 (PMID 17476472) |
| FTO rs9939609 (obesity) | homozygote OR 1.67; 16% homozygous (allele frequency 0.40 derived under Hardy-Weinberg) | Frayling et al. 2007, Science 316:889 (PMID 17434869), abstract |
| LPA rs10455872 (Lp(a) / coronary disease) | OR about 1.7 per copy | Clarke et al. 2009, NEJM 361:2518 (PMID 20032323), via a secondary summary; not verified in the paper |

Converting each effect into a genome cell takes four steps:

1. **mg/dL effects to SD units.** Divide by an assumed population SD: LDL-C 35, HDL-C 15, TG 90, TC 40 mg/dL. These SDs are simulation assumptions.
2. **log-OR effects to SD units.** Multiply by sqrt(3)/pi, the logistic-distribution SD.
3. **Amplify.** Multiply by `GENETIC_SCALE = 4`, so that a few hundred patients carry detectable signal.
4. **Project onto HPO terms.** Multiply by the HPO term's loading on the trait (`LOAD` in `build_sites.py`; for example, Decreased HDL-C has loading -1 on HDL).

Each site then adds sampling noise with sd = 1 / sqrt(GWAS n / 1000), using GWAS n = 50,000 for site-1 and 20,000 for site-2.

No hypertension or asthma locus could be verified in the time available, so the blood-pressure
and atopy terms have zero genome columns.

## How the patients are simulated

1. **Genotypes.** Each person's 14 genotypes are drawn from the allele frequencies above.
2. **Latent traits.** A per-trait polygenic score (the same effects times `GENETIC_SCALE`) drives the latent LDL-C, HDL-C, TG and Lp(a) z-scores, and the BMI and glycaemia z-scores. The last two also depend on a shared metabolic factor, which additionally raises TG and blood pressure and lowers HDL-C. Age raises LDL-C, glycaemia and blood pressure. Atopy is its own factor.
3. **Lab values.** The z-scores become lab values. Intercepts are calibrated on a 200,000-person reference sample to match three CDC NCHS targets (NHANES August 2021 to August 2023):

   | target | calibrated value | source |
   | --- | --- | --- |
   | high total cholesterol (>= 240 mg/dL) | 11.3% | Data Brief 515 |
   | low HDL-C (< 40 mg/dL) | 21.5% of men, 6.6% of women | Data Brief 515 |
   | hypertension | 47.7% | Data Brief 511 |

   The T2D (10%), obesity (30%) and asthma (10%) base rates are simulation assumptions.
4. **HPO terms.** EHR rules assign HPO terms from the labs:

   | HPO term | rule | source |
   | --- | --- | --- |
   | Elevated LDL-C | LDL-C >= 4.14 mmol/L (160 mg/dL) | NCEP ATP III |
   | Hypercholesterolemia | total cholesterol >= 6.21 mmol/L, or LDL-C >= 4.91 mmol/L | NCEP ATP III |
   | Hypertriglyceridemia | TG >= 2.26 mmol/L | NCEP ATP III |
   | Decreased HDL-C | HDL-C < 1.03 mmol/L (men) or < 1.29 mmol/L (women) | NCEP ATP III Table 8 |
   | Elevated Lp(a) | > 105 nmol/L | ESC/EAS 2025 |
   | Elevated apoB | >= 1.30 g/L | AHA/ACC 2018 |

   Parent terms follow their children: Hyperlipidemia and Abnormal circulating lipid concentration, and Diabetes mellitus from T2D. Xanthomas and corneal arcus become likely at LDL-C >= 5.0 or 6.5 mmol/L. Retinopathy, proteinuria and CKD follow T2D and hypertension. Coronary disease follows LDL-C, low HDL-C, Lp(a), T2D, hypertension and age.
5. **Sites differ.**
   - site-2 is older and more metabolic.
   - site-3 is younger, has more atopy and no genome file.
   - Each site drops 3, 5 or 4 non-core columns at random.

Resulting prevalences (from `build_summary.json`):

| site | N | high TC | high LDL-C | low HDL-C (HPO rule) | high TG | T2D | hypertension | asthma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| site-1 | 400 | 11.0% | 7.5% | 22.5% | 15.2% | 9.0% | 49.7% | 13.8% |
| site-2 | 250 | 11.2% | 7.2% | 25.2% | 15.6% | 16.0% | 66.4% | 11.6% |
| site-3 | 300 | 8.3% | 5.0% | 20.0% | 13.0% | 8.0% | 35.3% | 17.7% |

## Results of Davor's job (5 rounds, 20 local epochs, lr 0.05, 32 factors, NVFlare 2.9.0 simulator, CPU)

`scripts/` has no evaluation, so `run_federated_cf.py` scores representational similarity. This is the Pearson r between pairwise cosine similarities of the learned and true phenotype embeddings; the score is invariant to rotations of the latent space. The runner's random initialisation is the floor. For the genetic table, only terms with a non-zero genome column are scored.

| data | runs | nongenetic RSA | genetic RSA | random-init floor (nongenetic / genetic) | wall time per run |
| --- | --- | --- | --- | --- | --- |
| `data/federated` (baseline) | 3 | 0.77-0.78 | 0.73-0.74 (40 terms) | 0.03 / -0.05 | 30 s |
| `data/ehr_lipids` | 4 | 0.35-0.44 | 0.58-0.71 (24 terms) | -0.04 / -0.02 | 31 s |

Runs differ because `train_reconstruction` initialises row embeddings with unseeded `torch.randn`.

The lower scores are expected. Here the truth is a nonlinear, thresholded clinical model rather
than the bilinear model the CF assumes, and there are only 14 genome rows per site (versus 70-80).

The learned structure is clinically coherent. Nearest nongenetic neighbours (first run):

| term | nearest neighbours (cosine) |
| --- | --- |
| Elevated LDL-C | Hypercholesterolemia (0.98), Elevated apoB (0.95), Coronary artery atherosclerosis (0.88) |
| Decreased HDL-C | Hypertriglyceridemia (0.66), Coronary artery atherosclerosis (0.58), Proteinuria (0.57) |
| Type II diabetes mellitus | Diabetes mellitus (0.96), Retinopathy (0.90), Proteinuria (0.88) |
| Asthma | Wheezing (0.99), Eczematoid dermatitis (0.84), Dyspnea (0.83) |

On the genetic side, the nearest neighbours of Elevated LDL-C are Corneal arcus, Tendon xanthomatosis and Elevated apoB.

## Not verified

- **Simulation assumptions:**
  - the trait SDs used to standardise mg/dL effects;
  - `GENETIC_SCALE`;
  - the TCF7L2 and LPA allele frequencies;
  - the T2D, obesity and asthma base rates;
  - all comorbidity coefficients and the per-site shifts.
- **Lp(a) effect:** the OR comes from a secondary summary, not the paper.
- **Teslovich positive effects:** their signs were read from the PDF text layer, where "+" rendered as "1".
- **Unreachable databases:** the GWAS Catalog, PGS Catalog and Harmonizome were unreachable from the build environment. No effect was taken from them, and no hypertension or asthma locus is included.
- **Other synthetic data on main:** `data/synthgen` (per-patient PRS, added on main by Sebastian) was not used.

## Relation to PR #2

PR #2 (`elakiya/ehr-subphenotypes`) contains the full EHR pipeline (`Elakiya/metametagraphs/ehr`).
It reads UK Biobank and OMOP data, applies sourced rules and writes a subphenotype matrix plus an
F3CF export. This folder is a small, self-contained generator that applies the same threshold
logic to simulated labs and writes Davor's layout, so it does not depend on PR #2. On real data,
the PR #2 output (`patient_phenotype.csv`) can be pivoted to `patient_phenotypes.csv` once
subphenotype ids are mapped to HPO terms.
