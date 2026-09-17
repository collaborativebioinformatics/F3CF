# EHR to subphenotypes (`metametagraphs.ehr`)

This covers README milestone 5: "Model that extracts information from EHRs to enrich phenotype
definitions; decide on EHR data types (e.g. ICD10); define subphenotypes (phenotype ontologies,
ICD10 etc.)". It turns EHR data into a person x subphenotype matrix plus the hierarchy and the
evidence behind every cell. Collaborative filtering, the metagraph, genotype features and
federation are out of scope here; they consume these outputs.

## Run

All commands run from the `Elakiya/` folder (team convention: one folder per person).

```bash
cd Elakiya
pip install -r requirements-ehr.txt
python -m metametagraphs.ehr build --source toy  --input /tmp/mmg_toy --out /tmp/mmg_subpheno
python -m metametagraphs.ehr build --source toy  --input /tmp/mmg_toy --out /tmp/mmg_subpheno_omop --toy-schema omop
python -m metametagraphs.ehr build --source ukb  --input path/to/ukb_dir  --out out/ [--subset 5000]
python -m metametagraphs.ehr build --source omop --input path/to/cdm_dir  --out out/ --phecodes 1.2,X
python -m metametagraphs.ehr validate-rules [--rules my_rules.csv] [--nodes my_nodes.csv]
python -m metametagraphs.ehr export-f3cf --build /tmp/mmg_subpheno --out /tmp/mmg_f3cf --site clinic_a
python -m metametagraphs.ehr build --source toy --input /tmp/mmg_toy --out /tmp/mmg_subpheno --f3cf-site clinic_a   # build + export in one step
python -m metametagraphs.ehr export-hpo --build /tmp/mmg_subpheno --out /tmp/mmg_hpo/site-1 --site site-1 [--phenotype-ids ../data/ehr_lipids/phenotype_ids.txt]
python -m metametagraphs.ehr build --source toy --input /tmp/mmg_toy --out /tmp/mmg_subpheno --hpo-site site-1       # build + HPO export
python -m pytest -q                  # tests/ehr, network-free, about 6 s (pytest.ini sets pythonpath)
```

Options: `--rules FILE`, `--nodes FILE`, `--lab-codes FILE` (replace the bundled tables),
`--min-count N` (default minimum number of distinct-date events for code and medication rules
without their own `min_count`), `--phecodes` (`1.2`, `X`, `1.2,X` or `none`) (default `1.2`), `--subset N`.

## Inputs

| source | expected layout | tables / fields used |
| --- | --- | --- |
| `ukb` | `DIR/tabular/*.tsv` (or `DIR/*.tsv`), `DIR/medrec/set3*.txt` or `DIR/gp_clinical*`, `DIR/gp_scripts*` | tabular: 31 sex, 41270 ICD10, 30780 / 30760 / 30870 / 30690 / 30790 / 30640 / 30750 labs, 6177 / 6153 medication; GP clinical: read_2, read_3, value1, event_dt; GP scripts: bnf_code, drug_name, read_2, issue_date |
| `omop` | `DIR/{person,condition_occurrence,measurement}.{csv,parquet}`, optional `drug_exposure`, `concept`, `concept_ancestor`; any column case | person.gender_concept_id; condition_concept_id + condition_source_value; measurement_concept_id + value_as_number + unit_concept_id; drug_concept_id (+ concept names, ATC ancestors) |
| `toy` | written by the tool | both of the above, from one latent cohort |

Tested on the Eunomia Synthea27Nj (CDM 5.4, lower case, with vocabulary) and GiBleed (CDM 5.3,
upper case, no vocabulary; 2,694 persons, 266k events, about 3 s) sample datasets.

## Output contract

| file | content |
| --- | --- |
| `subphenotype_matrix.parquet` | index `person_id` (int64, every person in the source); one int8 0/1 column per subphenotype id, after exclusivity and roll-up; columns in hierarchy order (parents first), hand nodes then phecode nodes |
| `subphenotype_hierarchy.json` | `{"format": "metametagraphs.subphenotype_hierarchy/1", "nodes": [...]}`, each node: `id, label, parent, level, children, system (hand, phecode1.2, phecodeX), rule_types, logic, window_days, exclusive_group, priority, description, source, source_codes[], lab_rules[], phecode_crosswalk[]` |
| `subphenotype_evidence.parquet` | `person_id, subphenotype, rule_id, rule_type (code, lab, med, rollup), source (source table), code_system, code, value, unit, date, status (direct, rollup, excluded_by:<node>)` |
| `subphenotype_report.csv` | per node: n_positive, n_direct, n_excluded, n_male, n_female, prevalence, persons by rule type and by source table |
| `phecode_crosswalk_report.csv` | hand node vs phecode: n_hand, n_phecode, n_both, Jaccard |
| `build_info.json` | inputs, rule files, event counts by source, dropped lab rows, consistency problems (empty when all is well) |

Invariants (tested): every positive cell has at least one `direct` or `rollup` evidence row and
vice versa; a child is never positive without its parent; exclusive siblings never overlap;
matrix columns equal the hierarchy node ids in order.

## F3CF export (per-site relations)

F3CF (repository README) factorises, per clinic, a Patient x Phenotype and a Patient x Drug
relation into a shared latent space. `export-f3cf` (or `build --f3cf-site NAME`, which writes to
`OUT/f3cf/`) turns one build into those two relations, in the same long
`(row, column, score, source)` shape as the GWAS edge files in `Henrik/` (`gene, phenotype, score, source`):

| file | columns | content |
| --- | --- | --- |
| `patient_phenotype.csv` | `patient, phenotype, score, source` | one row per positive matrix cell; `phenotype` = subphenotype id (hand node or `phecode1.2:...`); `score` = 1; `source` = site name |
| `patient_drug.csv` | `patient, drug, score, source` | medication evidence; `drug` = statin ingredient (`simvastatin`), class (`atc:C10AA`, `bnf:0212`), UKB self-report (`self_report:6177=1`) or `omop:<id>` when no names exist; `score` = number of distinct prescription dates; `source` = site name |

Each clinic runs `build --f3cf-site clinic_a` on its own data (Clinic A, Clinic B, ... in the F3CF
flowchart) and keeps the two files local as its site relations; only the factorisation updates
leave the site. Patient ids are local to the site. Phenotype ids are shared across sites because
the rule tables are shared, and they can join the GWAS phenotype edges once mapped to trait
names (for example `dyslipidaemia.high_ldl` to LDL cholesterol). Only medications that matched a
medication rule are exported (currently lipid-lowering therapy and self-reported blood pressure
medication).

## HPO export (patient x HPO, federated CF layout)

`export-hpo` (or `build --hpo-site NAME`, which writes to `OUT/hpo/`) converts the subphenotype
matrix into the patient x HPO layout used by `scripts/federated_cf_*.py` and `data/federated`:

| file | content |
| --- | --- |
| `patient_phenotypes.csv` | header `row_id,<HP ids>`; rows `<site>_<person_id>`; 0/1 |
| `phenotype_ids.txt` | the global HPO id list (given with `--phenotype-ids`, or the bundled catalog) |
| `hpo_mapping_report.csv` | one row per crosswalk mapping (node positives, exported columns, notes), per unmapped node, and per exported column (positives, prevalence) |

How it works:

* **The crosswalk.** `resources/hpo_crosswalk.csv` maps every hand node to HPO. Each row has:
  * a `match` type: exact, broader, narrower, related or unmapped;
  * `implies = yes` for exact and broader rows only, because those are the only cases where a
    positive node guarantees the HPO term.

  Lipid-lowering therapy and blood pressure medication are treatments, so they stay unmapped.
* **Missing HPO terms.** HPO has no terms for familial hypercholesterolemia, mixed
  hyperlipidemia, essential or secondary hypertension, or allergic / non-allergic asthma, so
  those nodes map to the broader term.
* **Ancestor propagation.** Terms propagate to their HPO ancestors (true path rule), but only to
  ancestors present in the id list. HPO places Hypertriglyceridemia under Hyperlipidemia; it does
  not place Elevated LDL-C or Hypercholesterolemia there.
* **Output columns.** The columns are the ids the rules can reach, in id-list order. With
  `--phenotype-ids ../data/ehr_lipids/phenotype_ids.txt` (PR #3), the columns are a subset of that
  list, so `scripts/federated_cf_data.load_site()` reads the file directly.
* **Offline validation.** Terms are checked against `resources/hpo_subset/hp_excerpt.obo`
  (id, name and is_a of 157 terms from hp.obo release 2026-09-01; see its SOURCES.md).

Results with the PR #3 id list (12 of its 32 terms are reachable):

| input | patients | columns | prevalences |
| --- | --- | --- | --- |
| toy UKB layout | 400 | 12 | abnormal lipid concentration 50%, elevated LDL-C 41%, hypercholesterolemia 41%, decreased HDL-C 17%, hypertriglyceridemia 17%, hypertension 28%, type II diabetes 20%, asthma 15% |
| Eunomia fixture | 12 | 12 | abnormal lipid concentration 42%, hypertension 42%, hypertriglyceridemia 17%, decreased HDL-C 8%, type II diabetes 8% |

## How the rest of Team 8 plugs in

* **Federated CF prototype (`scripts/`)**: use `export-hpo --phenotype-ids <global list>` per site;
  the output is a `site-k/patient_phenotypes.csv` for `load_site()`.
* **F3CF**: use `patient_phenotype.csv` and `patient_drug.csv` (above) as each clinic's local
  relations.
* **Collaborative filtering / metagraph**: read `subphenotype_matrix.parquet` as the person x
  subphenotype interaction block and join it to the genotype feature matrix on `person_id` (UKB
  `eid` or OMOP `person_id`). Use `subphenotype_hierarchy.json` for node labels, parents and
  levels (for example, to contrast siblings such as `dyslipidaemia.high_ldl` and
  `dyslipidaemia.low_hdl`), and the evidence file to explain any association.
* **Team 5 (Everything_OMOP)**: `--source omop` reads standard CDM 5.4 tables, so any biobank
  they convert can be used directly. Rules keyed on OMOP concept ids live in `rules.csv`
  (`code_system = omop`), and lab codes in `lab_codes.csv`. Their repository currently only has a
  README (planned NVFlare OMOP dataloader), so there was no code to align with yet.
* **Federation**: the matrix is per site. Per-node counts in `subphenotype_report.csv` are what a
  site can share.

## Rule tables

All in `Elakiya/metametagraphs/ehr/resources/`, editable, validated by `validate-rules`, and every row
cites its source (`verified = no` marks rows not found in a checked source).

* `nodes.csv`: hierarchy. `logic` combines event types per person (`any`, or for example
  `code>=2 | code>=1 & med>=1`), `window_days` requires the combination within a time window,
  and `exclusive_group` + `priority` make siblings mutually exclusive (highest priority wins).
* `rules.csv`: `code` rules (diagnoses), `med` rules (prescriptions, self-report), `lab` rules
  (`analyte op threshold unit`, `sex` any, male, female or unknown, `aggregation` any, worst,
  most_recent or mean), and `min_count` (pooled over rules of the same node, type and code
  system).
* `lab_codes.csv`: Read2 / UKB field / OMOP concept to analyte, canonical unit and default unit.
  Conversions: cholesterol mg/dL x 0.02586, triglycerides mg/dL x 0.01129, apoB mg/dL x 0.01 to
  g/L, HbA1c mmol/mol = (% - 2.15) x 10.929. Lp(a) is never converted between nmol/L and mg/dL.
* `phecode_crosswalk.csv`: hand node to Phecode 1.2 / PhecodeX codes (equivalent, close,
  broader).
* `phecode_subset/`: offline excerpt of the Phecode 1.2 and PhecodeX maps (see its SOURCES.md).

### The lipid hierarchy

| node | definition | source |
| --- | --- | --- |
| `dyslipidaemia` | E78.0-E78.5, Read2 C32, OMOP 432867, SNOMED 55822004; two codes, or one code plus lipid-lowering therapy, within 730 days; any child | project rule |
| `.high_ldl` | LDL-C >= 4.14 mmol/L (160 mg/dL) at any measurement, or E78.0 | NCEP ATP III |
| `.high_ldl.severe` | highest LDL-C >= 4.91 mmol/L (190 mg/dL) | NCEP ATP III; 2018 AHA/ACC |
| `.severe.dlcn_possible` | highest LDL-C >= 5.0 mmol/L (DLCN 3 or 5 points: possible FH band on LDL-C alone) | DLCN criteria |
| `.severe.dlcn_probable` | highest LDL-C > 8.5 mmol/L (DLCN 8 points: probable FH band); exclusive with `dlcn_possible` | DLCN criteria |
| `.low_hdl` | most recent HDL-C < 1.03 mmol/L (men) or < 1.29 mmol/L (women), or E78.6 | NCEP ATP III Table 8 |
| `.high_tg` / `.very_high` | highest TG >= 2.26 / >= 5.65 mmol/L (200 / 500 mg/dL), or E78.1, E78.3, OMOP 4120314 | NCEP ATP III |
| `.mixed` | E78.2, Read2 C322. | WHO ICD-10 |
| `.high_tc` | total cholesterol >= 6.21 mmol/L (240 mg/dL) | NCEP ATP III |
| `.high_lpa` / `.very_high` | Lp(a) > 105 nmol/L or > 50 mg/dL / > 430 nmol/L or > 180 mg/dL | ESC/EAS 2025 focused update / ESC/EAS 2019 |
| `.high_apob` | apoB >= 1.30 g/L (130 mg/dL) | 2018 AHA/ACC risk-enhancing factors |
| `.treated` | >= 2 prescriptions: BNF 2.12, ATC C10, statin names (ATC C10AA), OMOP statin concepts; or UKB 6177/6153 = 1 | BNF, WHO ATC, UKB Showcase |

The full DLCN score also uses family history, clinical history, examination and genetics, which
are not in these EHR tables, so the tool only derives the LDL-C criterion. UKB field 30790
reports Lp(a) up to 189 nmol/L, so `.high_lpa.very_high` cannot be observed in UKB.

Other demo parents: `t2d` (renal, ophthalmic, poor control = most recent HbA1c >= 75 mmol/mol,
the NICE NG28 rec. 1.7.26 threshold), `hypertension` (essential, secondary, organ damage,
self-reported treatment) and `asthma` (allergic, non-allergic, severe), with ICD10, Read2, CTV3
(OpenSAFELY codelists) and OMOP rules.

## Sources (checked 2026-09-16/17)

| topic | source |
| --- | --- |
| DLCN criteria | CADTH Common Drug Review, Lomitapide (Juxtapid) 2015, Table 10, https://www.ncbi.nlm.nih.gov/books/NBK362539/table/T17/ ; also in the 2019 ESC/EAS guidelines |
| ESC/EAS 2019 | 2019 ESC/EAS Guidelines for the management of dyslipidaemias, Eur Heart J 41(1):111, doi:10.1093/eurheartj/ehz455 (Lp(a) > 180 mg/dL / > 430 nmol/L; TG < 1.7 mmol/L lower risk) |
| ESC/EAS 2025 focused update | Eur Heart J 2025;46(42):4359, doi:10.1093/eurheartj/ehaf190; Lp(a) > 50 mg/dL (105 nmol/L) as a risk factor (PACE-CME summary) |
| NCEP ATP III | NIH Publication 01-3670 (2001), Table 2 (LDL, total, HDL classes), triglyceride classes, Table 8 (HDL < 40 men / < 50 women), https://www.nhlbi.nih.gov/files/docs/guidelines/atp3xsum.pdf |
| 2018 AHA/ACC cholesterol guideline | 2018 AHA/ACC/.../PCNA Guideline on the Management of Blood Cholesterol, Circulation, doi:10.1161/CIR.0000000000000625; LDL-C >= 190 mg/dL and risk enhancers (LDL-C 160-189, TG >= 175, Lp(a) >= 50 mg/dL or >= 125 nmol/L, apoB >= 130) as summarised by the Family Heart Foundation |
| NICE NG28 | https://www.nice.org.uk/guidance/ng28 , rec. 1.7.26 (HbA1c 75 mmol/mol [9.0%]) |
| HbA1c conversion | IFCC = (NGSP - 2.15) x 10.929, Medscape "Hemoglobin A1c Testing" |
| UKB fields | UKB Showcase: 31 (coding 9), 41270 (coding 19), category 17518, 30790 (nmol/L), 30640 (g/L), 6177 / 6153 (coding 100625), 42039 (gp_scripts columns) |
| UKB synthetic layout | https://biobank.ndph.ox.ac.uk/synthetic_dataset/ |
| Read2 codes | ClinicalCodes.org, https://clinicalcodes.rss.mhs.man.ac.uk/ (articles 1, 2, 6, 166, 178) |
| CTV3 codes | OpenSAFELY codelists diabetes, hypertension, asthma-diagnosis, https://github.com/opensafely/risk-factors-research (MIT) |
| ATC | WHO Collaborating Centre, https://atcddd.fhi.no/atc_ddd_index/?code=C10AA |
| BNF 2.12 | "BNF 2.12: Lipid-regulating drugs", https://openprescribing.net/bnf/0212/ (page title) |
| OMOP CDM | https://github.com/OHDSI/CommonDataModel (CDM 5.4 field-level CSV) |
| OMOP concept ids | Eunomia Synthea27Nj CONCEPT table and data, https://github.com/OHDSI/EunomiaDatasets (Apache-2.0) |
| Phecode 1.2 | Wu P et al., JMIR Med Inform 2019;7(4):e14325; files via https://github.com/spiros/phemap (Apache-2.0) |
| PhecodeX | Shuey MM et al., Bioinformatics 2023;39(11):btad655; https://github.com/PheWAS/PhecodeXVocabulary |

## Not verified

* Read2 lipid diagnosis codes C32.., C320., C321., C322. and serum triglyceride 44Q.. were not
  found in a checked codelist (`verified = no`). No CTV3 lipid codes are included.
* OMOP concept 3028437 (calculated LDL) and any OMOP concepts for Lp(a) and apoB could not be
  checked (Athena was unreachable), so OMOP Lp(a)/apoB rules are absent.
* ATC concept ids in OMOP: the tool maps drugs to ATC through `concept`/`concept_ancestor` by
  ATC `concept_code`, so no ATC concept ids are hard-coded; the toy vocabulary uses toy ids.
  The real CONCEPT_ANCESTOR path is tested only with that toy vocabulary (Eunomia's is empty).
* The exact `bnf_code` formatting in the real UKB `gp_scripts` table was not checked (codes are
  compared after removing dots and spaces). The GP clinical `value2`/`value3` columns (units in
  some extracts) are not used.
* The UKB synthetic files themselves were never downloaded; the parsers follow the published
  layout. Real UKB portal extracts (header, dd/mm/yyyy) are handled but untested on real data.
* Thresholds are converted from mg/dL cut points at 2 decimals (for example 160 mg/dL becomes
  4.14 mmol/L); the DLCN cut points are native mmol/L.
* The dyslipidaemia two-code / code-plus-treatment rule and the 730-day window are a project
  choice, not taken from a published phenotype.
