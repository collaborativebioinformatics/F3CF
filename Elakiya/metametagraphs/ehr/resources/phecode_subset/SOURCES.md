# Phecode subset (offline fixture)

Rows for phecode families 272/250/401/495 (Phecode 1.2) and EM_239/EM_202/CV_401/RE_475 (PhecodeX 1.0),
copied unchanged (same header, same row text) from:

| file | source | license |
| --- | --- | --- |
| phecode_definitions1.2.csv | https://raw.githubusercontent.com/spiros/phemap/master/data/phecode_definitions1.2.csv (originally phewascatalog.org) | Apache-2.0 (spiros/phemap) |
| phecode_map_v1_2_icd10_beta.csv | https://raw.githubusercontent.com/spiros/phemap/master/data/phecode_map_v1_2_icd10_beta.csv (originally phewascatalog.org) | Apache-2.0 (spiros/phemap) |
| phecodeX_info.csv | https://github.com/PheWAS/PhecodeXVocabulary, `PhecodeX (version 1.0)/phecodeX_info.csv` | no license file in the repository; small excerpt for testing, cite Shuey et al. 2023 |
| phecodeX_unrolled_ICD_WHO.csv | https://github.com/PheWAS/PhecodeXVocabulary, `PhecodeX (version 1.0)/phecodeX_unrolled_ICD_WHO.csv` | as above |

Downloaded 2026-09-16. Use `metametagraphs.phenotypes.phecode.download_phecode12()` / `download_phecodex()`
for the full tables. PhecodeX files are latin-1 encoded.

Citations: Wu P et al. (2019) JMIR Med Inform 7(4):e14325 (Phecode 1.2 ICD-10 map);
Shuey MM et al. (2023) Bioinformatics 39(11):btad655 (PhecodeX).
