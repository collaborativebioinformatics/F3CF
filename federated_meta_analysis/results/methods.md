# Methods

**Data.** We used HAPNEST synthetic genotypes (European panel, release 1): 10,000 individuals and 1,329,052
variants across the 22 autosomes, distributed as per-chromosome VCFs and converted to PLINK binary format. Six
quantitative phenotypes were analysed — Height, BMI, HDL cholesterol, Body weight, Heart rate and HbA1c — each
supplied on the liability scale together with the simulation's noiseless genetic value (GenoEff) for every
individual. Narrow-sense heritability per trait was taken as var(GenoEff) / var(phenotype), giving 0.60, 0.29,
0.25, 0.29, 0.08 and 0.46 respectively. Individuals were partitioned into six disjoint sites of 500, 1,000,
1,500, 1,700, 2,300 and 3,000 people, deliberately unequal so that contributors differ in size as real ones do.
A random 20% of individuals (2,000) was held out as a common test set, excluded from every site's analysis and
from the linkage-disequilibrium reference; the remaining 8,000 formed the training pool. Results are averaged
over three draws of the held-out split (seed 42).

**Method.** Each site ran its own genome-wide association analysis locally with plink2 `--glm` on its own
individuals. Only per-variant effect estimates, standard errors and sample counts were shared; no genotypes or
phenotypes left a site. Summary statistics were combined by fixed-effect inverse-variance weighted
meta-analysis, beta_meta = sum(beta_i / se_i^2) / sum(1 / se_i^2) with se_meta = sqrt(1 / sum(1 / se_i^2)),
aligning effect alleles across sites and flipping signs where they disagreed. Meta-analysis results were
clumped (r^2 = 0.1, 250 kb window) using the largest participating site's genotypes as the LD reference, and
held-out individuals were scored with plink2 `--score` across nine p-value inclusion thresholds
(5x10^-8 to 1.0). Prediction accuracy is the squared Pearson correlation between the score and the phenotype
among held-out individuals, taking the best threshold at each step. Sites were accreted in half-site
increments, so that a partially enrolled contributor runs its local GWAS on half its cohort, giving twelve
points from 200 to 8,000 training individuals.

**Measurements.** Three quantities were recorded. First, PRS accuracy (R^2) as a function of the number of
participating sites and cumulative training individuals, per trait. Second, the set of variant-trait
associations passing 5x10^-8 at each accretion step, collapsed into 250 kb loci, with each discovered locus
labelled causal or false. Labels came from association-testing the noiseless genetic value GenoEff, which
carries no environmental noise, applied only to the 245 loci ever discovered and Bonferroni-corrected for that
number; on Height, the one trait shipping a causal-variant list, this labelling calls all 56 discovered loci
causal, 49 of which appear in the shipped list and 7 of which are neighbouring blocks in linkage
disequilibrium. Third, the fraction of each trait's genetic signal still unclaimed, 1 - R^2/h^2, and its
derivative per 1,000 additional individuals. Genomic inflation was checked at every step and trait
(lambda 0.96-1.02).
