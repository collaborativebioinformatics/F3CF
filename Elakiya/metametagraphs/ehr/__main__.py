"""Command line interface.

    python -m metametagraphs.ehr build --source ukb  --input DATA_DIR --out OUT
    python -m metametagraphs.ehr build --source omop --input CDM_DIR  --out OUT
    python -m metametagraphs.ehr build --source toy  --input TOY_DIR  --out OUT [--toy-schema omop]
    python -m metametagraphs.ehr toy --out TOY_DIR [--n 400] [--seed 0]
    python -m metametagraphs.ehr validate-rules [--rules FILE] [--nodes FILE]
    python -m metametagraphs.ehr export-f3cf --build OUT --out F3CF_DIR --site CLINIC_A
    python -m metametagraphs.ehr build ... --f3cf-site CLINIC_A     (also writes OUT/f3cf/)
    python -m metametagraphs.ehr export-hpo --build OUT --out HPO_DIR --site site-1 [--phenotype-ids FILE]
    python -m metametagraphs.ehr build ... --hpo-site site-1        (also writes OUT/hpo/)

``--source toy`` writes a toy cohort to ``--input`` (both layouts) and
builds from its UKB layout (or OMOP with ``--toy-schema omop``).
``--min-count N`` is the default minimum number of distinct-date matching
events for code and medication rules that do not set ``min_count``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from metametagraphs.ehr import rules as R


def _build(args) -> int:
    from metametagraphs.ehr.build import build
    from metametagraphs.ehr.toy import write_toy

    source, path = args.source, args.input
    if source == "toy":
        paths = write_toy(args.input, n=args.n, seed=args.seed)
        source, path = args.toy_schema, paths[args.toy_schema]
        print(f"toy cohort ({args.n} people) written to {args.input}; building from {path}")
    phecodes = () if args.phecodes == "none" else tuple(args.phecodes.split(","))
    res = build(source, path, args.out, rules=args.rules, nodes=args.nodes, lab_codes=args.lab_codes,
                min_count=args.min_count, phecodes=phecodes, subset=args.subset)
    info, rep = res["info"], res["report"]
    print(f"persons: {info['n_persons']}  events: {info['n_events']}  evidence rows: {info['n_evidence_rows']}")
    print(f"events by source: {info['events_by_source']}")
    hand = rep[~rep["subphenotype"].str.startswith("phecode")]
    print(f"\nhand hierarchy: {len(hand)} nodes")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(hand[["subphenotype", "level", "n_positive", "n_direct", "n_excluded", "n_male", "n_female",
                    "persons_by_rule_type"]].to_string(index=False))
        ph = rep[rep["subphenotype"].str.startswith("phecode")]
        if len(ph):
            print(f"\nphecode nodes: {len(ph)} (observed in this cohort)")
            print(ph[["subphenotype", "label", "level", "n_positive"]].to_string(index=False))
        if len(res["crosswalk_report"]):
            print("\nhand vs phecode agreement:")
            print(res["crosswalk_report"].to_string(index=False))
    if args.f3cf_site:
        _export(args.out, str(__import__("pathlib").Path(args.out) / "f3cf"), args.f3cf_site)
    if args.hpo_site:
        _export_hpo(args.out, str(__import__("pathlib").Path(args.out) / "hpo"), args.hpo_site, args.hpo_phenotype_ids)
    print(f"\nconsistency problems: {info['consistency_problems'] or 'none'}")
    print(f"outputs in {args.out} ({info['seconds']} s)")
    return 1 if info["consistency_problems"] else 0


def _export(build_dir: str, out: str, site: str) -> int:
    from metametagraphs.ehr.f3cf import export

    res = export(build_dir, out, site)
    pp, dr = res["patient_phenotype"], res["patient_drug"]
    print(f"F3CF export for site {site}: patient_phenotype.csv {len(pp)} edges "
          f"({pp['patient'].nunique()} patients x {pp['phenotype'].nunique()} phenotypes), "
          f"patient_drug.csv {len(dr)} edges ({dr['patient'].nunique()} patients x {dr['drug'].nunique()} drugs) -> {out}")
    return 0


def _export_hpo(build_dir: str, out: str, site: str, phenotype_ids: str | None = None) -> int:
    from metametagraphs.ehr.hpo import export

    res = export(build_dir, out, site, phenotype_ids)
    m, rep = res["matrix"], res["report"]
    mapped = rep[(rep["kind"] == "mapping") & (rep["implies"] == "yes")]["subphenotype_id"].nunique()
    print(f"HPO export for {site} (HPO {res['hpo_version']}): patient_phenotypes.csv {m.shape[0]} patients x "
          f"{m.shape[1]} HPO columns (of {len(res['phenotype_ids'])} ids), {mapped} subphenotypes mapped -> {out}")
    with pd.option_context("display.width", 200):
        cols = rep[rep["kind"] == "column"][["hpo_id", "hpo_label", "n_positive", "note"]]
        print(cols.to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m metametagraphs.ehr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build subphenotype matrix, hierarchy and evidence")
    b.add_argument("--source", choices=["ukb", "omop", "toy"], required=True)
    b.add_argument("--input", required=True, help="UKB data dir, OMOP CDM dir, or (toy) where to write the toy cohort")
    b.add_argument("--out", required=True)
    b.add_argument("--rules", help="rules CSV (default: bundled resources/rules.csv)")
    b.add_argument("--nodes", help="nodes CSV (default: bundled resources/nodes.csv)")
    b.add_argument("--lab-codes", help="lab code map CSV (default: bundled resources/lab_codes.csv)")
    b.add_argument("--min-count", type=int, default=1, help="default minimum distinct-date events for code/med rules")
    b.add_argument("--phecodes", default="1.2", help="comma list of phecode systems to add (1.2, X) or 'none'")
    b.add_argument("--subset", type=int, help="first N persons")
    b.add_argument("--toy-schema", choices=["ukb", "omop"], default="ukb")
    b.add_argument("--n", type=int, default=400, help="toy cohort size")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--f3cf-site", help="also export F3CF relations to OUT/f3cf with this site name")
    b.add_argument("--hpo-site", help="also export the patient x HPO matrix to OUT/hpo with this site name")
    b.add_argument("--hpo-phenotype-ids", help="global HPO id list for --hpo-site (default: bundled catalog)")
    hp = sub.add_parser("export-hpo", help="write patient_phenotypes.csv (row_id,<HP ids>) from a build")
    hp.add_argument("--build", required=True, help="output directory of a previous build")
    hp.add_argument("--out", required=True)
    hp.add_argument("--site", required=True, help="site id used as row_id prefix, e.g. site-1")
    hp.add_argument("--phenotype-ids", help="global phenotype_ids.txt to align to (default: bundled catalog)")
    hp.add_argument("--format", choices=["davor"], default="davor", help="output layout (data/federated style)")
    x = sub.add_parser("export-f3cf", help="write F3CF patient_phenotype.csv / patient_drug.csv from a build")
    x.add_argument("--build", required=True, help="output directory of a previous build")
    x.add_argument("--out", required=True)
    x.add_argument("--site", required=True, help="site / clinic name written to the source column")
    t = sub.add_parser("toy", help="write a toy cohort in UKB and OMOP layouts")
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=400)
    t.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("validate-rules", help="check a nodes/rules table pair")
    v.add_argument("--rules")
    v.add_argument("--nodes")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        return _build(args)
    if args.cmd == "export-hpo":
        return _export_hpo(args.build, args.out, args.site, args.phenotype_ids)
    if args.cmd == "export-f3cf":
        return _export(args.build, args.out, args.site)
    if args.cmd == "toy":
        from metametagraphs.ehr.toy import write_toy
        paths = write_toy(args.out, n=args.n, seed=args.seed)
        print(f"wrote {paths['ukb']} and {paths['omop']}")
        return 0
    nodes, rules = R.load_nodes(args.nodes), R.load_rules(args.rules)
    R.validate(nodes, rules)
    print(f"ok: {len(nodes)} nodes, {len(rules)} rules "
          f"({', '.join(f'{k}={v}' for k, v in rules['rule_type'].value_counts().items())}); "
          f"unverified rules: {int((rules['verified'] != 'yes').sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
