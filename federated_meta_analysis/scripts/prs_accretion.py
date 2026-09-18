# Reads:  data/hapnest/all/all.{bed,bim,fam}, data/hapnest/all/phenotype.txt, data/hapnest/site_assignment.tsv
# Writes: results/prs_accretion.csv (one row per repeat x accretion step x p-value threshold)
# Does:   federated clump-and-threshold PRS: per-site GWAS, meta-analysis of summary statistics, held-out scoring

# No raw genotypes are pooled. Each site runs its own GWAS on its own individuals; only the per-variant effect and
# standard error leave the site. Those are combined by inverse-variance weighting:
#   beta_meta = sum(beta_i / se_i^2) / sum(1 / se_i^2),  se_meta = sqrt(1 / sum(1 / se_i^2))
# which is what a fixed-effect federated GWAS produces. Sites do not always report the same effect allele for a
# variant (plink names it per dataset), so betas are aligned to the first site's allele and the sign flipped where
# they differ - the alignment step any real meta-analysis performs. Clumping needs linkage disequilibrium, which needs
# genotypes, so it uses the largest participating site's own genotypes as the LD reference - the choice a real
# federation has, short of an external panel. Test individuals are held out of every site's GWAS and of the LD
# reference, and are scored with the meta-analysed weights.
#
# GRANULARITY splits each join in two: at step 1.5 the first site contributes its whole cohort and the second
# contributes a fixed random half of its own, which is what a site that has enrolled half its patients would
# send. The half-cohort GWAS is still run locally and only its summary statistics are pooled, so the federated
# constraint is unchanged.
#
# A polygenic score is on its own scale (a weighted allele count), not the trait's, so R2 is the squared
# correlation between score and trait among held-out individuals, the standard PRS metric. Comparing the raw
# score with the trait would give a large negative "R2" that only reflects the difference in units.

import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import stats

BFILE, PHENOTYPES, ASSIGNMENT, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
PLINK, TEST_FRACTION, REPEATS, SEED = sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
TRAIT = sys.argv[9] if len(sys.argv) > 9 else "Height"
GRANULARITY = int(sys.argv[10]) if len(sys.argv) > 10 else 1   # 1 = whole sites, 2 = also half-joined sites
TAG = sys.argv[11] if len(sys.argv) > 11 else ""               # keeps cached GWAS of different site splits apart
WORK = os.path.join(os.path.dirname(OUTPUT), f"_prs_work_{TRAIT}{TAG}")
THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
CLUMP_R2, CLUMP_KB = 0.1, 250


def run(*arguments):
    result = subprocess.run([PLINK, *arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(" ".join(arguments)[:200] + "\n" + result.stderr.strip()[-1500:])


def write_keep(path, table):
    table[["fid", "iid"]].to_csv(path, sep="\t", header=False, index=False)
    return path


def site_gwas(members, tag):
    """One site's local GWAS. Only ID, effect allele, beta, SE and sample count leave the site."""
    prefix = f"{WORK}/{tag}"
    if not os.path.exists(f"{prefix}.gwas.{TRAIT}.glm.linear"):
        write_keep(f"{prefix}.keep", members)
        run("--bfile", BFILE, "--keep", f"{prefix}.keep", "--pheno", PHENOTYPES, "--pheno-name", TRAIT,
            "--glm", "allow-no-covars", "--out", f"{prefix}.gwas", "--silent")
    summary = pd.read_csv(f"{prefix}.gwas.{TRAIT}.glm.linear", sep="\t",
                          usecols=["ID", "A1", "BETA", "SE", "OBS_CT"])
    return summary[np.isfinite(summary["BETA"]) & np.isfinite(summary["SE"]) & (summary["SE"] > 0)]


def meta_analyse(summaries):
    """Inverse-variance weighted fixed-effect meta-analysis of the sites' summary statistics."""
    stacked = pd.concat([summary.assign(weight=1 / summary["SE"] ** 2) for summary in summaries],
                        ignore_index=True)
    reference = stacked.drop_duplicates("ID").set_index("ID")["A1"]        # first site's effect allele
    flipped = stacked["A1"].to_numpy() != reference.reindex(stacked["ID"]).to_numpy()
    stacked["BETA"] = np.where(flipped, -stacked["BETA"], stacked["BETA"])
    stacked["A1"] = reference.reindex(stacked["ID"]).to_numpy()
    stacked["weighted_beta"] = stacked["BETA"] * stacked["weight"]
    meta = stacked.groupby("ID", sort=False).agg(A1=("A1", "first"), weighted_beta=("weighted_beta", "sum"),
                                                 weight=("weight", "sum"), n=("OBS_CT", "sum")).reset_index()
    meta["BETA"] = meta["weighted_beta"] / meta["weight"]
    meta["SE"] = np.sqrt(1 / meta["weight"])
    meta["P"] = 2 * stats.norm.sf(np.abs(meta["BETA"] / meta["SE"]))
    return meta[["ID", "A1", "BETA", "SE", "P", "n"]]


def clump_and_score(meta, ld_members, test_table, tag):
    """Clump the meta-analysis against one site's genotypes, then score the held-out individuals."""
    prefix = f"{WORK}/{tag}"
    meta.to_csv(f"{prefix}.meta.txt", sep="\t", index=False)
    run("--bfile", BFILE, "--keep", write_keep(f"{prefix}.ld.keep", ld_members), "--clump", f"{prefix}.meta.txt",
        "--clump-p1", "1", "--clump-r2", str(CLUMP_R2), "--clump-kb", str(CLUMP_KB),
        "--out", f"{prefix}.clump", "--silent")
    index_variants = set(pd.read_csv(f"{prefix}.clump.clumps", sep="\t")["ID"])
    kept = meta[meta["ID"].isin(index_variants)]

    kept[["ID", "A1", "BETA"]].to_csv(f"{prefix}.score.txt", sep="\t", index=False)
    kept[["ID", "P"]].to_csv(f"{prefix}.pvalues.txt", sep="\t", header=False, index=False)
    with open(f"{prefix}.ranges.txt", "w") as handle:
        for threshold in THRESHOLDS:
            handle.write(f"{threshold:g}\t0\t{threshold:g}\n")
    run("--bfile", BFILE, "--keep", write_keep(f"{prefix}.test.keep", test_table), "--score",
        f"{prefix}.score.txt", "1", "2", "3", "header", "--q-score-range", f"{prefix}.ranges.txt",
        f"{prefix}.pvalues.txt", "--out", f"{prefix}.prs", "--silent")
    return len(index_variants), prefix


os.makedirs(WORK, exist_ok=True)
fam = pd.read_csv(f"{BFILE}.fam", sep=r"\s+", header=None, usecols=[0, 1], names=["fid", "iid"])
phenotype = pd.read_csv(PHENOTYPES, sep="\t").rename(columns={"#FID": "fid", "IID": "iid"})
fam = fam.merge(phenotype, on=["fid", "iid"])
fam["site"] = pd.read_csv(ASSIGNMENT, sep="\t").set_index("iid")["site"].reindex(fam["iid"]).to_numpy()
sites = sorted(fam["site"].dropna().unique())
print(f"[prs_accretion] {TRAIT}: {len(fam):,} individuals, {len(sites)} sites, federated meta-analysis")

rows = []
for repeat in range(REPEATS):
    rng = np.random.default_rng(SEED + repeat)
    test_rows = rng.choice(len(fam), int(TEST_FRACTION * len(fam)), replace=False)
    test_table = fam.iloc[test_rows]
    available = fam.drop(index=fam.index[test_rows])

    members, summaries = {}, {}
    for site in sites:                      # a site's GWAS depends only on its own data, so it is run once
        whole = available[available["site"] == site]
        members[(site, 1)] = whole
        summaries[(site, 1)] = site_gwas(whole, f"r{repeat}_site{int(site)}")
        print(f"[prs_accretion]   repeat {repeat}  site {int(site)}: local GWAS on {len(whole):,} individuals")
        if GRANULARITY > 1:
            part = whole.sample(frac=1 / GRANULARITY, random_state=SEED + repeat * 100 + int(site))
            members[(site, 0)] = part
            summaries[(site, 0)] = site_gwas(part, f"r{repeat}_site{int(site)}p")
            print(f"[prs_accretion]   repeat {repeat}  site {int(site)}: local GWAS on {len(part):,} "
                  f"individuals (partial cohort)")

    for index in range(1, GRANULARITY * len(sites) + 1):
        step = index / GRANULARITY
        joined = [(site, 1) for site in sites[:int(step)]]
        if step != int(step):                                  # the next site has enrolled part of its cohort
            joined.append((sites[int(step)], 0))
        meta = meta_analyse([summaries[key] for key in joined])
        frames = [members[key] for key in joined]
        reference = max(frames, key=len)                       # largest participating cohort is the LD panel
        largest = int(reference["site"].iloc[0])
        tag = f"s{int(step)}" if step == int(step) else f"s{step}"
        n_clumped, prefix = clump_and_score(meta, reference, test_table, f"r{repeat}_{tag}")
        train_total = int(sum(len(frame) for frame in frames))
        genome_wide = int((meta["P"] < 5e-8).sum())
        for threshold in THRESHOLDS:
            path = f"{prefix}.prs.{threshold:g}.sscore"
            shared = {"trait": TRAIT, "repeat": repeat, "n_sites": step, "train_individuals": train_total,
                      "threshold": threshold, "clumped_variants": n_clumped,
                      "meta_genome_wide_hits": genome_wide, "ld_reference_site": int(largest)}
            if not os.path.exists(path):                  # no variant passed this threshold
                rows.append({**shared, "variants_in_score": 0, "r2": 0.0, "correlation": 0.0})
                continue
            scores = pd.read_csv(path, sep="\t").rename(columns={"#FID": "fid", "#IID": "iid", "IID": "iid"})
            merged = test_table.merge(scores[["iid", "ALLELE_CT", "SCORE1_AVG"]], on="iid")
            correlation = np.corrcoef(merged[TRAIT].to_numpy(), merged["SCORE1_AVG"].to_numpy())[0, 1]
            rows.append({**shared, "variants_in_score": int(merged["ALLELE_CT"].iloc[0] // 2),
                         "r2": float(correlation ** 2), "correlation": correlation})
        best = max(r["r2"] for r in rows if r["repeat"] == repeat and r["n_sites"] == step)
        print(f"[prs_accretion]   repeat {repeat}  meta of {step} site(s), {train_total:,} individuals, "
              f"{genome_wide:,} genome-wide hits  best R2 {best:+.4f}")

table = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
table.to_csv(OUTPUT, index=False)
summary = table.pivot_table(index=["n_sites", "train_individuals"], columns="threshold", values="r2", aggfunc="mean")
print()
print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))
print(f"[prs_accretion] Saved {OUTPUT}")
