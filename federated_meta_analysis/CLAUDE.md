# federated_meta_analysis — orientation

Federated clump-and-threshold PRS on HAPNEST synthetic data. Six simulated sites each run their own
GWAS; only summary statistics are pooled. The question is what federation buys — in prediction accuracy,
and in whether the discovered variant-trait associations are actually real.

**Nothing here needs the network.** Every command uses `uv run --offline`. The environment is built; do
not run `uv sync` or `uv add` unless you genuinely need a new package and have network.

---

## Rebuild everything

```sh
./run_all.sh          # all five figures from cached results, offline, ~3 min
```

That is the entry point. It does not rerun any GWAS. If the figures are what changed, this is all you need.

---

## Layout

```
scripts/           10 scripts, all reachable from run_all.sh or run_fine.sh
data/hapnest/      PLINK filesets: all/ (merged) and site1..site6/, plus site_assignment.tsv
data/*.vcf.gz      the 22 source chromosome VCFs (14 GB total with the PLINK conversions)
results/           CSVs, figures/ (PDF) and figures/png/ (300 dpi) — plus 52 GB of caches
results/_prs_work_pheno{N}/   per-trait GWAS + score cache, 8.5 GB each
results/_true_work/           GWAS against the noiseless genetic value, 726 MB
archive/           4 GB of superseded work. Gitignored. Do not commit, do not resurrect without asking.
bin/plink2         v2.0.0-a.6.9
```

Traits are HAPNEST phenotype numbers, not names: `pheno1` Height, `pheno2` BMI, `pheno3` HDL
cholesterol, `pheno6` Body weight, `pheno7` Heart rate, `pheno9` HbA1c. Heritabilities 0.60, 0.29,
0.25, 0.29, 0.08, 0.46 — computed as `var(GenoEff) / var(Phenotype(liability))`, not hardcoded.

---

## The pipeline

`scripts/prs_accretion.py` is the analysis. Everything else reads its output.

```
BFILE PHENOTYPES ASSIGNMENT OUTPUT PLINK TEST_FRACTION REPEATS SEED [TRAIT] [GRANULARITY] [TAG]
```

Per accretion step: each site runs `plink2 --glm` locally → inverse-variance meta-analysis of the
summary statistics (aligning effect alleles, flipping signs where sites disagree) → `--clump`
(r²=0.1, 250 kb) against the largest participating site's genotypes → `--score` over nine p-value
thresholds → R² on held-out individuals.

`GRANULARITY=2` makes sites join in halves: at step 1.5 site 1 contributes its whole cohort and
site 2 a fixed random half, which runs its own local GWAS on that half. Twelve points instead of six.
`run_fine.sh` does this for all traits (~45 min, 7–8 min per trait).

---

## Things that will bite you

**The GWAS cache is keyed on trait + repeat + site number, not on the site assignment.** If you change
`site_assignment.tsv` you MUST pass a distinct `TAG`, or the run silently reuses summary statistics
computed from different individuals and produces confident, wrong results. `WORK` is
`results/_prs_work_{TRAIT}{TAG}`.

**Never rebuild figures while `prs_accretion.py` is running.** The figure scripts read `.meta.txt`
files straight out of the work directories. Doing this mid-run produced a bipartite figure reporting
229 loci instead of 245. If a count looks off, check for a running job before believing it.

**`data/EUR_trait1_causal_variants.txt` is not usable as ground truth.** The positions are right
(221/230 match) but the allele coding is not — 68 of them have allele pairs that do not overlap the
genotypes at all. A PRS built from its effect sizes correlates **−0.026** with the true genetic value.
Ground truth comes from `GenoEff` instead (see below). The file is only used as a cross-check.

**PRS R² is the squared correlation between score and trait.** A polygenic score is a weighted allele
count on its own scale. Comparing it to the trait directly gives a large negative "R²" that is only a
units mismatch.

**The best p-value threshold is chosen on the held-out set.** Levels are therefore optimistic. Shape
and between-trait ordering are unaffected. Say so rather than letting someone find it.

**`results/` is gitignored as `results/*`, not `results/`.** Git cannot re-include a file whose parent
directory is excluded, so the `!results/figures/**` negations only work because the directory itself
is not excluded. If you "simplify" that pattern you will silently stop tracking the deliverable.

---

## Ground truth

`scripts/true_loci.py` runs a GWAS against `GenoEff`, HAPNEST's noiseless genetic value. With no
environmental noise, power is not the constraint, so a significant locus is causal by construction.

A discovered locus is called causal if its `GenoEff` p-value clears `0.05 / 245` — Bonferroni over
the 245 loci ever discovered, not genome-wide, because only those are being tested. Loci are 250 kb
blocks: linkage disequilibrium means neighbours of a causal variant are significant too, and no
association test separates them. Do not claim variant-level causality.

Validation on Height, the one trait with a shipped causal list: all 56 discovered loci are called
causal, 49 appear in that list, 7 are adjacent LD blocks, zero are called false. The labelling does
not invent errors.

---

## What the results are

PRS R² at six sites: Height 0.443, HbA1c 0.443, Body weight 0.266, HDL 0.223, BMI 0.079,
Heart rate 0.058.

Fraction of achievable signal captured: HbA1c 95%, Body weight 92%, HDL 88%, Height 74%,
Heart rate 69%, **BMI 28%**. Body weight and BMI have near-identical heritability (0.288 vs 0.286)
and wildly different federation payoff — that contrast survives the bootstrap bands and is the
strongest finding.

Association precision across the six steps: **10% → 38% → 81% → 97% → 99% → 100%**. HbA1c at one site
produces 162 genome-wide hits across 88 LD blocks with median |effect| 5.01; by six sites the median
is 0.115 and **0.6% of those first hits survive**. Not genomic inflation — λ is 0.96–1.02 everywhere.
This is the headline: federation does not only find more, it corrects what a small site got wrong.

Confidence bands (`scripts/bootstrap_r2.py`, 6 s) resample the 2,000 held-out individuals 200 times
using cached scores. They cover held-out sampling only — **not** variation in how individuals are
assigned to sites. Height and HbA1c overlap heavily and should be presented as tied.

---

## Already tried and rejected — do not redo without a reason

- **Ridge / GBLUP on all SNPs.** Replaced by clump-and-threshold PRS, which is the standard method here.
- **Pooling raw genotypes across sites.** Violates the federated constraint. Sites share summary
  statistics only.
- **A phenotype-phenotype graph.** HAPNEST draws causal variants independently per trait: true genetic
  values correlate at most 0.026 against a null 95% half-width of 0.0196. Only 3 of 15 trait pairs share
  any locus, all involving Height, each a single 250 kb block. The graph is empty by construction.
- **Marginal-gain-per-1,000 figure** and an **appear/disappear/persist edge stability chart** — both
  built, both scrapped as less informative than the bipartite figure. In `archive/`.
- **A 5-seed sweep over site assignments.** Correct idea for error bars, but each seed invalidates the
  entire GWAS cache: ~8.5 GB and ~48 min per seed, ~255 GB and ~4 h for five. Bootstrapping the scoring
  step was the cheap substitute.

---

## Plot conventions

Style follows `~/Desktop/Desktop_M2/Projects_25/plot_styleguide/plot_style_guide.md`. Serif, no top or
right spines, `_sunset` colormap, PDF plus 300 dpi PNG in `results/figures/png/`.

**Titles state what is measured, never the result, and carry no trailing period.** The one exception is
`metagraph_bipartite`, whose title the author chose deliberately.

Terminology is fixed: **trait** (not phenotype), **locus** (not variant or SNP — the figures plot 250 kb
blocks), **site** (not clinic), **association** (not edge). Broad language is allowed in titles —
"genotype-phenotype" appears in the convergence title on purpose — precise language inside figures.

Trait colours encode heritability rank via the sunset ramp and are consistent across figures.
`metagraph_bipartite` is the deliberate exception: green/red for causal/false, since sunset has no green.
