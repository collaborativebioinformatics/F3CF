# Reads:  data/syn_rel1_chr.pheno*, data/hapnest/all/all.{bed,bim,fam}, data/EUR_trait1_causal_variants.txt
# Writes: results/true_loci.csv (the causal loci of every trait), data/hapnest/all/geno_eff.txt
# Does:   recovers each trait's true causal loci by association-testing the simulation's noiseless genetic value

# HAPNEST reports GenoEff, the genetic component of each individual's phenotype with no environmental noise in
# it. A GWAS against GenoEff therefore finds the causal variants with no power loss, so a locus significant here
# is causal, not a discovery. Only trait 1 ships a causal-variant list, and its allele coding is unusable (the
# effect signs do not reproduce GenoEff: r = -0.03), so that list is used only to check this recovery. Truth is
# recorded per 250kb locus, not per variant, because linkage disequilibrium makes neighbours of a causal variant
# significant too and no association test can separate them.

import os
import subprocess
import sys
import numpy as np
import pandas as pd

BFILE, PHENOTYPE_DIR, PLINK, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TRAITS = [1, 2, 3, 6, 7, 9]
BLOCK, THRESHOLD = 250_000, 5e-8
WORK = os.path.join(os.path.dirname(OUTPUT), "_true_work")


def block_of(variant):
    chromosome, position = variant.split(":")[0], int(variant.split(":")[1])
    return f"{chromosome}:{position // BLOCK}"


os.makedirs(WORK, exist_ok=True)
fam = pd.read_csv(f"{BFILE}.fam", sep=r"\s+", header=None, usecols=[0, 1], names=["#FID", "IID"])
combined = fam.copy()
for number in TRAITS:
    values = pd.read_csv(f"{PHENOTYPE_DIR}/syn_rel1_chr.pheno{number}", sep="\t")
    combined = combined.merge(values[["Sample", "GenoEff"]].rename(columns={"Sample": "IID",
                                                                           "GenoEff": f"pheno{number}"}),
                              on="IID")
geno_eff = f"{PHENOTYPE_DIR}/hapnest/all/geno_eff.txt"
combined.to_csv(geno_eff, sep="\t", index=False)
print(f"[true_loci] wrote {geno_eff} ({len(combined):,} individuals)")

rows = []
for number in TRAITS:
    trait = f"pheno{number}"
    out = f"{WORK}/{trait}"
    if not os.path.exists(f"{out}.{trait}.glm.linear"):
        result = subprocess.run([PLINK, "--bfile", BFILE, "--pheno", geno_eff, "--pheno-name", trait,
                                 "--glm", "allow-no-covars", "--out", out, "--silent"],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-1500:])
    table = pd.read_csv(f"{out}.{trait}.glm.linear", sep="\t", usecols=["ID", "BETA", "P"])
    significant = table[table["P"] < THRESHOLD].copy()
    significant["locus"] = [block_of(v) for v in significant["ID"]]
    lead = significant.loc[significant.groupby("locus")["P"].idxmin()]
    for row in lead.itertuples():
        rows.append({"trait": trait, "locus": row.locus, "lead_variant": row.ID,
                     "beta": row.BETA, "p": row.P})
    print(f"[true_loci] {trait}: {len(significant):,} variants significant against GenoEff "
          f"-> {len(lead):,} causal loci")

truth = pd.DataFrame(rows)
truth.to_csv(OUTPUT, index=False)
print(f"[true_loci] Saved {OUTPUT}")

# check the recovery against the one causal list that exists
causal = pd.read_csv(f"{PHENOTYPE_DIR}/EUR_trait1_causal_variants.txt", sep="\t", header=None,
                     names=["variant", "effects"])
listed = {block_of(v.replace("chr", "")) for v in causal["variant"]}
found = set(truth.loc[truth["trait"] == "pheno1", "locus"])
print(f"\n[true_loci] check on Height: {len(listed)} loci in the shipped causal list, {len(found)} recovered, "
      f"{len(listed & found)} in both ({len(listed & found) / len(listed):.0%} of the list)")
