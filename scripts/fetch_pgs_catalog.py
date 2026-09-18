#!/usr/bin/env python3
"""Download polygenic score files from the PGS Catalog.

Use ``--all-traits`` to take every mapped ontology trait that has a compact
scoring file (one PGS per trait: the smallest file under ``--max-variants``).
Without that flag, only the listed ``--traits`` are fetched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

API_BASE = "https://www.pgscatalog.org/rest"
FTP_METADATA_SCORES = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/pgs_all_metadata_scores.csv"
FTP_METADATA_TRAITS = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/pgs_all_metadata_efo_traits.csv"
USER_AGENT = "Metametagraphs/0.1 (PGS Catalog fetch; educational use)"
DEFAULT_TRAITS = ("EFO_0004612", "MONDO_0005148")
MAX_RETRIES = 4
RETRY_BACKOFF_S = 2.0
REQUEST_PAUSE_S = 0.08

SCORE_SUMMARY_FIELDS = [
    "pgs_id",
    "name",
    "trait_id",
    "trait_label",
    "trait_reported",
    "variants_number",
    "method_name",
    "publication_id",
    "pubmed_id",
    "ftp_scoring_file",
    "scoring_file",
]


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(REQUEST_PAUSE_S)
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_S * attempt)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    candidates = [url]
    if url.startswith("https://ftp.ebi.ac.uk/"):
        candidates.append("http://" + url[len("https://") :])
        candidates.append("ftp://" + url[len("https://") :])
    last_error: Exception | None = None
    for candidate in candidates:
        req = urllib.request.Request(candidate, headers={"User-Agent": USER_AGENT})
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    tmp.write_bytes(response.read())
                tmp.replace(dest)
                return
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_error = exc
                time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Failed to download {url} -> {dest}") from last_error


def slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    folded = "".join(keep).strip("_")
    while "__" in folded:
        folded = folded.replace("__", "_")
    return folded or "trait"


def split_mapped(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def load_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_scores_from_metadata(
    score_rows: list[dict[str, str]],
    *,
    max_variants: int,
    max_scores_per_trait: int,
    trait_filter: set[str] | None,
) -> list[dict[str, Any]]:
    """Pick compact scores per mapped ontology trait (smallest first)."""
    by_trait: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        n_variants = int(row.get("Number of Variants") or 0)
        ftp_url = row.get("FTP link") or ""
        if n_variants <= 0 or n_variants > max_variants or not ftp_url:
            continue
        trait_ids = split_mapped(row.get("Mapped Trait(s) (EFO ID)") or "")
        trait_labels = split_mapped(row.get("Mapped Trait(s) (EFO label)") or "")
        for index, trait_id in enumerate(trait_ids):
            if trait_filter is not None and trait_id not in trait_filter:
                continue
            label = trait_labels[index] if index < len(trait_labels) else (trait_labels[0] if trait_labels else "")
            by_trait.setdefault(trait_id, []).append(
                {
                    "pgs_id": row["Polygenic Score (PGS) ID"],
                    "name": row.get("PGS Name"),
                    "trait_id": trait_id,
                    "trait_label": label,
                    "trait_reported": row.get("Reported Trait"),
                    "variants_number": n_variants,
                    "method_name": row.get("PGS Development Method"),
                    "publication_id": row.get("PGS Publication (PGP) ID"),
                    "pubmed_id": row.get("Publication (PMID)"),
                    "ftp_scoring_file": ftp_url,
                    "row": row,
                }
            )

    selected: list[dict[str, Any]] = []
    for trait_id, items in sorted(by_trait.items()):
        items.sort(key=lambda item: item["variants_number"])
        selected.extend(items[:max_scores_per_trait])
    return selected


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SCORE_SUMMARY_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pgs")
    parser.add_argument("--traits", nargs="+", default=list(DEFAULT_TRAITS))
    parser.add_argument("--all-traits", action="store_true", help="One compact score for every Catalog trait")
    parser.add_argument("--max-variants", type=int, default=10_000)
    parser.add_argument("--max-scores-per-trait", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    meta_dir = output_dir / "metadata"
    files_dir = output_dir / "scoring_files"
    meta_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = meta_dir / "pgs_all_metadata_scores.csv"
    traits_csv = meta_dir / "pgs_all_metadata_efo_traits.csv"
    print("Downloading Catalog-wide score metadata…", flush=True)
    download_file(FTP_METADATA_SCORES, scores_csv)
    download_file(FTP_METADATA_TRAITS, traits_csv)

    trait_filter: set[str] | None = None if args.all_traits else set(args.traits)
    selected = select_scores_from_metadata(
        load_score_rows(scores_csv),
        max_variants=args.max_variants,
        max_scores_per_trait=args.max_scores_per_trait,
        trait_filter=trait_filter,
    )
    unique_urls = {item["pgs_id"]: item["ftp_scoring_file"] for item in selected}
    print(
        f"Selected {len(selected)} trait–score pairs "
        f"({len(unique_urls)} unique scoring files, {len({item['trait_id'] for item in selected})} traits)",
        flush=True,
    )

    def fetch_one(pgs_id: str, url: str) -> tuple[str, Path]:
        dest = files_dir / f"{pgs_id}.txt.gz"
        if dest.exists() and dest.stat().st_size > 0:
            return pgs_id, dest
        download_file(url, dest)
        return pgs_id, dest

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_one, pgs_id, url) for pgs_id, url in unique_urls.items()]
        done = 0
        for future in as_completed(futures):
            pgs_id, dest = future.result()
            downloaded[pgs_id] = dest
            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"  downloaded {done}/{len(futures)} scoring files", flush=True)

    summary_rows = []
    for item in selected:
        dest = downloaded[item["pgs_id"]]
        slim = {field: item.get(field, "") for field in SCORE_SUMMARY_FIELDS if field != "scoring_file"}
        slim["scoring_file"] = str(dest)
        summary_rows.append(slim)

    write_summary(output_dir / "scores_summary.csv", summary_rows)
    trait_index = [
        {
            "trait_id": item["trait_id"],
            "trait_label": item["trait_label"],
            "pgs_id": item["pgs_id"],
            "variants_number": item["variants_number"],
        }
        for item in selected
    ]
    trait_path = output_dir / "trait_to_score.csv"
    with trait_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trait_id", "trait_label", "pgs_id", "variants_number"])
        writer.writeheader()
        writer.writerows(trait_index)
    print(f"Wrote {len(summary_rows)} trait–score pairs ({len(downloaded)} unique files) under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise
