#!/bin/sh
# Rebuilds every figure from the results already on disk. No network: uv runs with --offline and
# every input is local. Run from the federated_meta_analysis directory:  ./run_all.sh
set -e
cd "$(dirname "$0")"
UV="uv run --offline python"

$UV scripts/edge_dynamics.py 'results/_prs_work_pheno*' results
$UV scripts/true_loci.py data/hapnest/all/all data ./bin/plink2 results/true_loci.csv

$UV scripts/bootstrap_r2.py results data/hapnest/all/phenotypes_all.txt results/prs_bootstrap.csv 200

$UV scripts/plot_prs_accretion.py results/prs_accretion.csv results/figures/prs_accretion.pdf 0.597 Height
$UV scripts/plot_trait_curves.py 'results/prs_pheno*.csv' data results/figures/trait_curves.pdf \
    results/prs_bootstrap.csv
$UV scripts/plot_convergence.py 'results/prs_fine_pheno*.csv' data results/figures/convergence.pdf \
    results/prs_bootstrap.csv
$UV scripts/plot_metagraph_grid.py results data/hapnest/all/all results/prs_pheno1.csv \
    results/figures/metagraph_grid.pdf
$UV scripts/plot_metagraph_bipartite.py results results/_true_work results/prs_pheno1.csv \
    results/figures/metagraph_bipartite.pdf

echo "[run_all] all figures rebuilt:"
ls -1 results/figures/*.pdf results/figures/png/*.png
