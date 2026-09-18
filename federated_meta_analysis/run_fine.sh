#!/bin/sh
# Reruns the federated PRS accretion at half-site granularity (12 points instead of 6) for every trait,
# then rebuilds the convergence figure from the finer curve. Offline; ~45 min.
set -e
cd "$(dirname "$0")"
for n in 1 2 3 6 7 9; do
  echo "=== pheno$n"
  uv run --offline python scripts/prs_accretion.py \
      data/hapnest/all/all data/hapnest/all/phenotypes_all.txt data/hapnest/site_assignment.tsv \
      "results/prs_fine_pheno$n.csv" ./bin/plink2 0.2 3 42 "pheno$n" 2
done
uv run --offline python scripts/plot_convergence.py 'results/prs_fine_pheno*.csv' data \
    results/figures/convergence.pdf
echo "[run_fine] done"
