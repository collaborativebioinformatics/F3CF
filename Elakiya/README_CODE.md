# Elakiya/: EHR to subphenotypes for F3CF

This folder is Elakiya's part of Team 8 (README milestone 5): it turns EHR data (UK Biobank
tabular + GP records, or OMOP CDM tables) into sourced subphenotypes, and exports them as the
per-site relations that F3CF factorises.

```bash
cd Elakiya
pip install -r requirements-ehr.txt
python -m pytest -q                                   # 47 tests, network-free (one uses scripts/ and torch if present)
python -m metametagraphs.ehr build --source toy --input /tmp/toy --out /tmp/site_a --f3cf-site clinic_a
python -m metametagraphs.ehr build --source omop --input path/to/cdm --out out/site_b --f3cf-site clinic_b
python -m metametagraphs.ehr export-hpo --build /tmp/site_a --out ../data/my_sites/site-1 --site site-1 \
    --phenotype-ids ../data/ehr_lipids/phenotype_ids.txt      # patient x HPO for scripts/federated_cf_*.py
```

Where it plugs into F3CF (see the flowchart in the repository README):

| F3CF relation | file | produced from |
| --- | --- | --- |
| Clinic X: Patient x Phenotype | `OUT/f3cf/patient_phenotype.csv` (`patient, phenotype, score, source`) | `subphenotype_matrix.parquet` |
| Clinic X: Patient x Drug | `OUT/f3cf/patient_drug.csv` (`patient, drug, score, source`) | medication evidence |
| Federated CF prototype (`scripts/`): site patient x phenotype | `site-k/patient_phenotypes.csv` (`row_id,<HP ids>`, rows `site-k_<person>`) | `export-hpo`, via `resources/hpo_crosswalk.csv` (HPO 2026-09-01) |
| Phenotype x PRS / GWAS (biobank, `Henrik/`) | `gene, phenotype, score, source` | joins on phenotype once subphenotype ids are mapped to trait names |

Layout:

```
metametagraphs/ehr/        events, sources/{ukb,omop}, rules, hierarchy, phecode, hpo, build, f3cf, toy, CLI
metametagraphs/ehr/resources/  nodes.csv, rules.csv, lab_codes.csv, phecode_crosswalk.csv, phecode_subset/,
                               hpo_crosswalk.csv, hpo_catalog.txt, hpo_subset/
tests/ehr/                 tests and a 12-person Eunomia Synthea27Nj fixture
docs/ehr_subphenotypes.md  inputs, output contract, rule tables, sources, what is not verified
```
