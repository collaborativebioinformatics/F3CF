# Reads:  data/syn_rel1_chr-*.vcf.gz, data/syn_rel1_chr.pheno1
# Writes: data/hapnest/site{1..6}/site{N}.{bed,bim,fam} and data/hapnest/site_assignment.tsv
# Does:   converts the arrived VCFs to PLINK, splits individuals into sites, and gives each site more chromosomes

# Every site holds all downloaded chromosomes and its own disjoint, randomly drawn individuals. Site sizes differ
# (SITE_SIZES) the way real contributors do: a large hospital brings several times the patients of a small one.
# Only chromosomes already downloaded are used; rerunning picks up new ones.
# Variant IDs are set to chr:pos:ref:alt to match the causal variant file, and the VCF's "syn1_syn1" sample
# names are split on "_" so the IID matches the phenotype file.

import os
import subprocess
import sys
import numpy as np
import pandas as pd

VCF_DIR, PHENOTYPES, OUTPUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
PLINK, N_SITES, SEED = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
SITE_SIZES = [500, 1000, 1500, 1700, 2300, 3000]   # individuals per site, smallest to largest


def run(*arguments):
    result = subprocess.run([PLINK, *arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-800:])


def complete_vcfs():
    """Downloaded chromosomes, skipping any file still being written (no gzip end-of-file marker)."""
    found = []
    for chromosome in range(1, 23):
        path = f"{VCF_DIR}/syn_rel1_chr-{chromosome}.vcf.gz"
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            handle.seek(-28, os.SEEK_END)
            if handle.read().endswith(bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")):
                found.append(chromosome)
            else:
                print(f"[prepare_sites]   chromosome {chromosome}: still downloading, skipped")
    return found


chromosomes = complete_vcfs()
print(f"[prepare_sites] chromosomes ready: {chromosomes}")
os.makedirs(f"{OUTPUT_DIR}/_chromosomes", exist_ok=True)

for chromosome in chromosomes:
    prefix = f"{OUTPUT_DIR}/_chromosomes/chr{chromosome}"
    if not os.path.exists(f"{prefix}.bed"):
        print(f"[prepare_sites] converting chromosome {chromosome} ...")
        run("--vcf", f"{VCF_DIR}/syn_rel1_chr-{chromosome}.vcf.gz", "--id-delim", "_",
            "--set-all-var-ids", "@:#:$r:$a", "--new-id-max-allele-len", "100", "--max-alleles", "2",
            "--make-bed", "--out", prefix, "--silent")
    variants = sum(1 for _ in open(f"{prefix}.bim"))
    samples = sum(1 for _ in open(f"{prefix}.fam"))
    print(f"[prepare_sites]   chromosome {chromosome}: {variants:,} variants, {samples:,} samples")

# Random split of the individuals, disjoint across sites
pheno = pd.read_csv(PHENOTYPES, sep="\t")
fam = pd.read_csv(f"{OUTPUT_DIR}/_chromosomes/chr{chromosomes[0]}.fam", sep=r"\s+", header=None,
                  names=["fid", "iid", "father", "mother", "sex", "phenotype"])
matched = fam["iid"].isin(set(pheno["Sample"])).mean()
print(f"[prepare_sites] genotype IIDs matching the phenotype file: {matched:.1%} "
      f"({len(fam):,} genotyped, {len(pheno):,} phenotyped)")

if sum(SITE_SIZES) > len(fam):
    raise ValueError(f"site sizes sum to {sum(SITE_SIZES):,}, more than the {len(fam):,} individuals available")
rng = np.random.default_rng(SEED)
order = rng.permutation(len(fam))
sites = np.zeros(len(fam), dtype=int)
start = 0
for site, size in enumerate(SITE_SIZES[:N_SITES], start=1):
    sites[order[start:start + size]] = site
    start += size
assignment = pd.DataFrame({"fid": fam["fid"], "iid": fam["iid"], "site": sites})
assignment = assignment[assignment["site"] > 0]
assignment.to_csv(f"{OUTPUT_DIR}/site_assignment.tsv", sep="\t", index=False)

summary = []
for site in range(1, N_SITES + 1):
    site_dir = f"{OUTPUT_DIR}/site{site}"
    os.makedirs(site_dir, exist_ok=True)
    members = assignment[assignment["site"] == site]
    keep_file = f"{site_dir}/keep.txt"
    members[["fid", "iid"]].to_csv(keep_file, sep="\t", header=False, index=False)

    site_chromosomes = chromosomes
    parts = []
    for chromosome in site_chromosomes:
        part = f"{site_dir}/_chr{chromosome}"
        run("--bfile", f"{OUTPUT_DIR}/_chromosomes/chr{chromosome}", "--keep", keep_file,
            "--make-bed", "--out", part, "--silent")
        parts.append(part)
    target = f"{site_dir}/site{site}"
    if len(parts) == 1:
        for extension in ("bed", "bim", "fam"):
            os.replace(f"{parts[0]}.{extension}", f"{target}.{extension}")
    else:
        merge_list = f"{site_dir}/merge_list.txt"
        with open(merge_list, "w") as handle:
            handle.write("\n".join(parts) + "\n")
        run("--pmerge-list", merge_list, "bfile", "--make-bed", "--out", target, "--silent")
    for leftover in os.listdir(site_dir):                      # per-chromosome temporaries
        if leftover.startswith("_chr") or leftover.endswith((".log", ".pgen", ".pvar", ".psam")):
            os.remove(f"{site_dir}/{leftover}")

    variants = sum(1 for _ in open(f"{target}.bim"))
    summary.append({"site": f"site{site}", "samples": len(members), "chromosomes": len(site_chromosomes),
                    "chromosome_range": f"1-{max(site_chromosomes)}" if site_chromosomes else "none",
                    "variants": variants})

print()
print(pd.DataFrame(summary).to_string(index=False))
print(f"[prepare_sites] Saved {OUTPUT_DIR} (assignment in {OUTPUT_DIR}/site_assignment.tsv)")
