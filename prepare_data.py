#!/usr/bin/env python3
"""Build the FitSloth food-search take-home dataset from Food-101.

Downloads a fixed 20-class subset of the Food-101 validation split and writes:

    food_search_dataset/
        images/        catalog images        (20 classes x 100)
        catalog.csv    image_id, filename, dish_label
        queries/       held-out query images (20 classes x 3)
        queries.csv    query_id, filename, true_label

Every run produces the identical dataset, so all candidates work from the same
catalog and the same queries. About 150 MB of downloads.

Standard library only. Nothing to pip install.

    python3 prepare_data.py
    python3 prepare_data.py --out ./somewhere-else

Source: Food-101 (Bossard, Guillaumin & Van Gool, ETH Zurich, 2014), served via
the Hugging Face datasets-server. Research / non-commercial use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://datasets-server.huggingface.co"
DATASET = "ethz/food101"
CONFIG = "default"
SPLIT = "validation"

# The validation split stores each class as a contiguous block of 250 rows.
ROWS_PER_CLASS = 250
NUM_CLASSES = 101
MAX_ROWS_PER_REQUEST = 100  # datasets-server limit

# Twenty classes, picked deliberately:
#   - three clusters of visually confusable dishes, which is where an
#     off-the-shelf embedding starts making interesting mistakes
#   - a few visually distinct anchors, so every candidate ends up with a
#     working system and a non-zero score rather than stuck at square one
#
# The offsets are where each class's block starts in the validation split.
# They are cached here so a normal run costs ~60 API calls instead of ~200;
# they are verified at runtime and rediscovered automatically if upstream
# ever reshuffles the data.
CLASS_BLOCKS: dict[str, int] = {
    # soups
    "hot_and_sour_soup": 21500,
    "miso_soup": 3500,
    "french_onion_soup": 15500,
    "pho": 16750,
    # salads
    "caesar_salad": 6750,
    "greek_salad": 1750,
    "caprese_salad": 20250,
    "seaweed_salad": 10250,
    # cakes
    "cheesecake": 8250,
    "tiramisu": 6250,
    "chocolate_cake": 3250,
    "red_velvet_cake": 9000,
    # asian dishes present in Food-101
    "pad_thai": 8000,
    "fried_rice": 4250,
    "gyoza": 21250,
    "sushi": 22500,
    # visually distinct anchors
    "hamburger": 750,
    "pizza": 2500,
    "steak": 6000,
    "donuts": 19250,
}

_throttle = threading.Semaphore(3)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def fetch(url: str, *, timeout: int = 60, retries: int = 6) -> bytes:
    """GET a URL, backing off on failure.

    The datasets-server rate-limits and is occasionally slow on a cold cache;
    both are worth waiting out rather than failing the whole run.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fitsloth-takehome/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
                time.sleep(wait)
                continue
            if exc.code < 500:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}\n  last error: {last}")


def fetch_api(url: str) -> dict:
    with _throttle:
        data = fetch(url)
        time.sleep(0.2)
    return json.loads(data)


def rows_url(offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"{API}/rows?{query}"


def first_label_at(offset: int) -> int:
    return fetch_api(rows_url(offset, 1))["rows"][0]["row"]["label"]


# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------

def label_names() -> list[str]:
    info = fetch_api(f"{API}/info?dataset={urllib.parse.quote(DATASET, safe='')}")
    return info["dataset_info"][CONFIG]["features"]["label"]["names"]


def rediscover_blocks(names: list[str]) -> dict[str, int]:
    """Slow path: probe the first row of all 101 blocks to rebuild the map."""
    print("  cached offsets are stale - rediscovering (this takes a few minutes)")
    found: dict[str, int] = {}
    for i in range(NUM_CLASSES):
        offset = i * ROWS_PER_CLASS
        found[names[first_label_at(offset)]] = offset
    if len(found) != NUM_CLASSES:
        raise RuntimeError(
            f"expected {NUM_CLASSES} distinct class blocks, found {len(found)}. "
            "The upstream dataset layout has changed in a way this script does "
            "not understand."
        )
    return found


def resolve_blocks(names: list[str]) -> dict[str, int]:
    """Confirm the cached offsets still point at the right classes."""
    name_to_id = {name: i for i, name in enumerate(names)}
    missing = [c for c in CLASS_BLOCKS if c not in name_to_id]
    if missing:
        sys.exit(f"classes no longer present upstream: {missing}")

    print(f"Verifying {len(CLASS_BLOCKS)} class blocks...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        labels = list(pool.map(first_label_at, CLASS_BLOCKS.values()))

    ok = all(
        labels[i] == name_to_id[dish] for i, dish in enumerate(CLASS_BLOCKS)
    )
    if ok:
        return dict(CLASS_BLOCKS)
    return {dish: rediscover_blocks(names)[dish] for dish in CLASS_BLOCKS}


def image_urls_for_class(offset: int, count: int) -> list[str]:
    """Collect `count` image URLs starting at a class block's offset."""
    urls: list[str] = []
    while len(urls) < count:
        want = min(MAX_ROWS_PER_REQUEST, count - len(urls))
        rows = fetch_api(rows_url(offset + len(urls), want))["rows"]
        if not rows:
            raise RuntimeError(f"no rows returned at offset {offset + len(urls)}")
        urls.extend(row["row"]["image"]["src"] for row in rows)
    return urls[:count]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build(out_dir: Path, per_class: int, queries_per_class: int) -> None:
    total_per_class = per_class + queries_per_class
    if total_per_class > ROWS_PER_CLASS:
        sys.exit(
            f"--per-class + --queries-per-class must be <= {ROWS_PER_CLASS} "
            f"(the validation split holds {ROWS_PER_CLASS} images per class)"
        )

    print(f"Building dataset in {out_dir}")
    print(f"  {len(CLASS_BLOCKS)} classes x {per_class} catalog + {queries_per_class} query images")

    print("Reading dataset info...")
    blocks = resolve_blocks(label_names())

    images_dir = out_dir / "images"
    queries_dir = out_dir / "queries"
    images_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    catalog_rows: list[dict] = []
    query_rows: list[dict] = []

    print("Collecting image URLs...")
    for dish in CLASS_BLOCKS:
        urls = image_urls_for_class(blocks[dish], total_per_class)

        for i, url in enumerate(urls[:per_class]):
            image_id = f"{dish}_{i:03d}"
            filename = f"{image_id}.jpg"
            jobs.append((url, images_dir / filename))
            catalog_rows.append(
                {"image_id": image_id, "filename": filename, "dish_label": dish}
            )

        for i, url in enumerate(urls[per_class:]):
            query_id = f"q_{dish}_{i:02d}"
            filename = f"{query_id}.jpg"
            jobs.append((url, queries_dir / filename))
            query_rows.append(
                {"query_id": query_id, "filename": filename, "true_label": dish}
            )

    print(f"Downloading {len(jobs)} images...")
    done = 0
    lock = threading.Lock()

    def download(job: tuple[str, Path]) -> None:
        nonlocal done
        url, path = job
        if not path.exists() or path.stat().st_size == 0:
            path.write_bytes(fetch(url, timeout=60))
        with lock:
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, jobs))

    with (out_dir / "catalog.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "filename", "dish_label"])
        writer.writeheader()
        writer.writerows(catalog_rows)

    with (out_dir / "queries.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["query_id", "filename", "true_label"])
        writer.writeheader()
        writer.writerows(query_rows)

    catalog_bytes = sum(p.stat().st_size for p in images_dir.glob("*.jpg"))
    query_bytes = sum(p.stat().st_size for p in queries_dir.glob("*.jpg"))

    print()
    print(f"Done. {out_dir.resolve()}")
    print(f"  images/   {len(catalog_rows):>5} files  {catalog_bytes / 1e6:>7.1f} MB")
    print(f"  queries/  {len(query_rows):>5} files  {query_bytes / 1e6:>7.1f} MB")
    print("  catalog.csv, queries.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("food_search_dataset"))
    parser.add_argument("--per-class", type=int, default=100, help="catalog images per class")
    parser.add_argument("--queries-per-class", type=int, default=3, help="held-out queries per class")
    args = parser.parse_args()
    build(args.out, args.per_class, args.queries_per_class)


if __name__ == "__main__":
    main()
