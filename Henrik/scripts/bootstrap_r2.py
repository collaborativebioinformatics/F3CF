# Reads:  results/_prs_work_pheno*/r*_s*.prs.*.sscore, data/hapnest/all/phenotypes_all.txt
# Writes: results/prs_bootstrap.csv
# Does:   puts a confidence interval on every accretion point by resampling the held-out individuals

# No GWAS or meta-analysis is repeated: the polygenic scores already written for the held-out individuals are
# reused exactly as they are. Only the test set is resampled, with replacement, so the interval says how much of
# each point moves with the particular 2,000 people it was measured on. Each bootstrap draw repeats the whole
# scoring rule, taking the best of the nine p-value thresholds within that draw, so the interval includes the
# cost of choosing a threshold rather than pretending it was fixed in advance. Draws from the repeats are pooled,
# which folds the variation between held-out splits into the same interval.

import glob
import os
import re
import sys
import numpy as np
import pandas as pd

WORK_DIR, PHENOTYPES, OUTPUT = sys.argv[1], sys.argv[2], sys.argv[3]
DRAWS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
SEED = 7
THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]

phenotypes = pd.read_csv(PHENOTYPES, sep="\t").rename(columns={"#FID": "fid", "IID": "iid"})
traits = sorted({re.search(r"_prs_work_(pheno\d+)$", p).group(1) for p in glob.glob(f"{WORK_DIR}/_prs_work_pheno*")},
                key=lambda t: int(t.replace("pheno", "")))

rows = []
for trait in traits:
    folder = f"{WORK_DIR}/_prs_work_{trait}"
    prefixes = sorted({re.match(r"(r\d+_s[\d.]+)\.prs\.", os.path.basename(p)).group(1)
                       for p in glob.glob(f"{folder}/r*_s*.prs.*.sscore")})
    for prefix in prefixes:
        repeat = int(re.match(r"r(\d+)_s", prefix).group(1))
        step = float(re.search(r"_s([\d.]+)$", prefix).group(1))
        columns, table = [], None
        for threshold in THRESHOLDS:
            path = f"{folder}/{prefix}.prs.{threshold:g}.sscore"
            if not os.path.exists(path):                      # no variant passed this threshold
                continue
            scores = pd.read_csv(path, sep="\t").rename(columns={"#IID": "iid", "IID": "iid"})
            scores = scores[["iid", "SCORE1_AVG"]].rename(columns={"SCORE1_AVG": f"t{threshold:g}"})
            table = scores if table is None else table.merge(scores, on="iid")
            columns.append(f"t{threshold:g}")
        if table is None:
            continue
        merged = table.merge(phenotypes[["iid", trait]], on="iid")
        scores_matrix = merged[columns].to_numpy(float)
        outcome = merged[trait].to_numpy(float)
        keep = scores_matrix.std(axis=0) > 0                  # a constant score has no correlation
        scores_matrix, columns = scores_matrix[:, keep], list(np.array(columns)[keep])
        if not columns:
            continue

        rng = np.random.default_rng(SEED + repeat)
        picks = rng.integers(0, len(outcome), size=(DRAWS, len(outcome)))
        drawn_scores = scores_matrix[picks]                   # (draws, individuals, thresholds)
        drawn_outcome = outcome[picks]                        # (draws, individuals)
        centred_scores = drawn_scores - drawn_scores.mean(axis=1, keepdims=True)
        centred_outcome = drawn_outcome - drawn_outcome.mean(axis=1, keepdims=True)
        numerator = np.einsum("dij,di->dj", centred_scores, centred_outcome)
        denominator = (np.sqrt((centred_scores ** 2).sum(axis=1))
                       * np.sqrt((centred_outcome ** 2).sum(axis=1))[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.where(denominator > 0, numerator / denominator, 0.0)
        best = np.nanmax(correlation ** 2, axis=1)            # the scoring rule picks the best threshold
        rows.append({"trait": trait, "repeat": repeat, "n_sites": step,
                     **{f"draw{i}": value for i, value in enumerate(best)}})
    print(f"[bootstrap_r2] {trait}: {len(prefixes)} accretion points bootstrapped")

table = pd.DataFrame(rows)
draw_columns = [c for c in table.columns if re.fullmatch(r"draw\d+", c)]
summary = []
for (trait, step), group in table.groupby(["trait", "n_sites"]):
    pooled = group[draw_columns].to_numpy().ravel()           # pool the repeats' draws
    pooled = pooled[np.isfinite(pooled)]
    assert pooled.min() >= 0 and pooled.max() <= 1, f"R2 outside [0,1] for {trait} step {step}"
    summary.append({"trait": trait, "n_sites": step, "r2_mean": pooled.mean(),
                    "r2_lo": np.percentile(pooled, 2.5), "r2_hi": np.percentile(pooled, 97.5)})
frame = pd.DataFrame(summary)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
frame.to_csv(OUTPUT, index=False)
print(f"[bootstrap_r2] Saved {OUTPUT}  ({len(frame)} trait x step points, {DRAWS} draws each)")
