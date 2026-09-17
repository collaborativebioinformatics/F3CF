# Reads:  data/edges_gwas.csv, data/edges_gene_gene.csv, and extra source edge lists (e.g. opentargets, clinvar)
# Writes: results/accretion.csv (summary across seeds), results/accretion_runs.csv (one row per seed x step)
# Does:   per seed, holds out a fixed set of GWAS edges and measures recall as training data accretes

# Steps per seed, all evaluated on the same held-out GWAS edges (removed from every source, no leakage):
#   gwas_fraction     growing nested subsets of the remaining GWAS edges
#   gwas_propagation  all GWAS + STRING neighbor propagation; test set split into filled vs untouched cells
#   cumulative        all GWAS + extra sources added one at a time;  gwas_plus_one: all GWAS + each source alone
# Summary: mean across seeds, 95% t-interval, paired change vs "GWAS 100%" (same test set per seed).

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as st
from cf import evaluate, pair_keys

OUTPUT, OUTPUT_RUNS = sys.argv[1], sys.argv[2]
TEST_FRACTION, N_COMPONENTS, TOP_K = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
SEEDS = [int(s) for s in sys.argv[6].split(",")]
FRACTIONS = [float(f) for f in sys.argv[7].split(",")]
PROPAGATION_WEIGHT = float(sys.argv[8])
GENE_GENE, GWAS, EXTRAS = sys.argv[9], sys.argv[10], sys.argv[11:]
R = f"recall@{TOP_K}"


def build_steps(gwas, extras, gene_gene, seed):
    """Fixed test set for this seed, and the list of (kind, name, train edges, gene_gene) steps."""
    order = np.random.default_rng(seed).permutation(len(gwas))
    n_test = int(len(gwas) * TEST_FRACTION)
    test, pool = gwas.iloc[order[:n_test]], gwas.iloc[order[n_test:]]  # pool prefixes = random nested subsets
    test_keys = set(pair_keys(test))
    clean = []
    for df in extras:
        in_test = pair_keys(df).isin(test_keys)
        print(f"[accretion] seed {seed}: removed {in_test.sum():,} test pairs from {df['source'].iloc[0]}")
        clean.append(df[~in_test])

    steps = [("gwas_fraction", f"GWAS {f:.0%}", pool.iloc[: round(f * len(pool))], None) for f in FRACTIONS]
    steps.append(("gwas_propagation", "GWAS + STRING propagation", pool, gene_gene))
    names = ["GWAS"]
    for i, df in enumerate(clean):
        names.append(df["source"].iloc[0])
        steps.append(("cumulative", " + ".join(names), pd.concat([pool] + clean[: i + 1]), None))
    for df in clean[1:]:  # "GWAS + first source" is already the first cumulative step
        steps.append(("gwas_plus_one", f"GWAS + {df['source'].iloc[0]}", pd.concat([pool, df]), None))
    return test, steps


def propagation_split(result, hits, baseline_hits):
    """Recall on test edges propagation filled vs left untouched, against all-GWAS recall on the same edges."""
    filled = hits["filled_by_propagation"].to_numpy()
    prop_hits = hits["hit_svd"].to_numpy()
    for part, mask in [("filled", filled), ("untouched", ~filled)]:
        result[f"{R}_{part}"] = prop_hits[mask].mean()
        result[f"{R}_{part}_gwas_only"] = baseline_hits[mask].mean()
        result[f"delta_{part}"] = prop_hits[mask].mean() - baseline_hits[mask].mean()
        print(f"[accretion]   {part:<9} ({mask.mean():.1%} of test): {R} "
              f"{baseline_hits[mask].mean():.2%} -> {prop_hits[mask].mean():.2%}")


def run_seed(gwas, extras, gene_gene, seed):
    """Evaluate every step for one seed; returns one result dict per step."""
    test, steps = build_steps(gwas, extras, gene_gene, seed)
    rows = []
    for step, (kind, name, train, gg) in enumerate(steps, start=1):
        print(f"\n[accretion] seed {seed}, step {step}/{len(steps)}: {name}")
        result, hits, _ = evaluate(train, test, N_COMPONENTS, TOP_K, seed, gene_gene=gg,
                                   propagation_weight=PROPAGATION_WEIGHT,
                                   log=lambda msg: print(f"[accretion] {msg}"))
        if name == "GWAS 100%":
            baseline_hits = hits["hit_svd"].to_numpy()
        if gg is not None:
            propagation_split(result, hits, baseline_hits)
        rows.append({"seed": seed, "step": step, "kind": kind, "data": name, **result})
        print(f"[accretion]   {R}: {result[R]:.2%}   popularity: {result[R + '_popularity']:.2%}")
    return rows


def t_interval(values):
    """Mean and 95% t-interval across seeds."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.mean(), np.nan, np.nan
    half = st.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), values.mean() - half, values.mean() + half


def summary_row(step, kind, name, group):
    """One summary row for a step: means, t-intervals and seeds improved."""
    row = {"step": step, "kind": kind, "data": name, "n_seeds": len(group)}
    for col in ["train_edges", "genes", "phenotypes", "test_edges", "test_covered"]:
        row[col] = group[col].mean()
    for col in [R, f"{R}_popularity", "delta_vs_gwas"]:
        row[col], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = t_interval(group[col])
    row["seeds_improved"] = int((group["delta_vs_gwas"] > 0).sum())
    for col in ["mean_score_test", "mean_score_random", "propagated_cells", "test_propagated"]:
        row[col] = group[col].mean()
    for part in ["filled", "untouched"]:
        if group[f"delta_{part}"].notna().all():
            for col in [f"{R}_{part}", f"{R}_{part}_gwas_only", f"delta_{part}"]:
                row[col], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = t_interval(group[col])
            row[f"seeds_improved_{part}"] = int((group[f"delta_{part}"] > 0).sum())
    return row


def fmt(row, col, unit="%", sign=""):
    """'mean [low, high]' in percent / percentage points."""
    return f"{row[col] * 100:{sign}.2f}{unit} [{row[col + '_ci_low'] * 100:{sign}.2f}, {row[col + '_ci_high'] * 100:{sign}.2f}]"


def print_summary(summary):
    """Step table, then the propagation filled/untouched split."""
    show = pd.DataFrame({
        "step": summary["step"], "data": summary["data"],
        "train edges": summary["train_edges"].map("{:,.0f}".format),
        "test covered": summary["test_covered"].map("{:.1%}".format),
        f"{R} [95% CI]": summary.apply(lambda row: fmt(row, R), axis=1),
        "vs all GWAS, pp [95% CI]": summary.apply(lambda row: fmt(row, "delta_vs_gwas", unit="", sign="+"), axis=1),
        "seeds better": np.where(summary["data"] == "GWAS 100%", "baseline",
                                 summary["seeds_improved"].astype(str) + "/" + summary["n_seeds"].astype(str)),
        "popularity [95% CI]": summary.apply(lambda row: fmt(row, f"{R}_popularity"), axis=1),
    })
    print(f"\n[accretion] Summary over seeds {SEEDS}; fixed test set of {int(summary['test_edges'].iloc[0]):,} "
          f"GWAS edges per seed; CI = 95% t-interval across seeds")
    print(show.to_string(index=False))
    row = summary[summary["kind"] == "gwas_propagation"].iloc[0]
    print(f"\n[accretion] STRING propagation, test set split ({row['test_propagated']:.1%} of test edges filled "
          f"by propagation; {row['propagated_cells']:,.0f} propagated cells)")
    for part in ["filled", "untouched"]:
        print(f"[accretion]   {part:<9} all GWAS {fmt(row, f'{R}_{part}_gwas_only')}  ->  "
              f"+ propagation {fmt(row, f'{R}_{part}')}   change {fmt(row, f'delta_{part}', unit=' pp', sign='+')}   "
              f"seeds better {int(row[f'seeds_improved_{part}'])}/{int(row['n_seeds'])}")


gwas = pd.read_csv(GWAS, keep_default_na=False)
extras = [pd.read_csv(path, keep_default_na=False) for path in EXTRAS]
gene_gene = pd.read_csv(GENE_GENE, keep_default_na=False)
print(f"[accretion] STRING gene-gene edges: {len(gene_gene):,}")
print(f"[accretion] GWAS edges: {len(gwas):,}")
for df in extras:
    print(f"[accretion] {df['source'].iloc[0]} edges: {len(df):,}")

runs = pd.DataFrame([row for seed in SEEDS for row in run_seed(gwas, extras, gene_gene, seed)])
baseline = runs[runs["data"] == "GWAS 100%"].set_index("seed")[R]
runs["delta_vs_gwas"] = runs[R] - runs["seed"].map(baseline)   # paired: same seed = same test set
summary = pd.DataFrame([summary_row(step, kind, name, group)
                        for (step, kind, name), group in runs.groupby(["step", "kind", "data"], sort=True)])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
runs.to_csv(OUTPUT_RUNS, index=False)
summary.to_csv(OUTPUT, index=False)
print_summary(summary)
print(f"\n[accretion] Saved {OUTPUT} and {OUTPUT_RUNS}")
