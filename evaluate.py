"""Evaluate all held-out images with binary dish-label relevance at rank five."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

from food_search import DATASET, INDEX, ROOT, ImageSearch

METRICS = ("precision@5", "recall@5", "ndcg@5")


def metrics_at_5(results, true_label, relevant_in_catalog):
    """Scores determine ranking upstream; only label matches determine relevance."""
    if len(results) < 5:
        raise ValueError("Evaluation requires five retrieved results")
    if relevant_in_catalog < 1:
        raise ValueError(f"No catalog images are relevant to {true_label!r}")
    relevance = [int(r["dish_label"] == true_label) for r in results[:5]]
    hits = sum(relevance)
    discounts = [1 / math.log2(position + 1) for position in range(1, 6)]
    dcg = sum(rel * discount for rel, discount in zip(relevance, discounts))
    ideal_dcg = sum(discounts[:min(5, relevant_in_catalog)])
    return {"precision@5": hits / 5, "recall@5": hits / relevant_in_catalog,
            "ndcg@5": dcg / ideal_dcg}


def evaluate(dataset=DATASET, index=INDEX, out=ROOT / "artifacts/evaluation"):
    dataset, index, out = Path(dataset), Path(index), Path(out)
    with (dataset / "queries.csv").open(newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    if not queries or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Queries must be nonempty and have unique IDs")
    search = ImageSearch(dataset, index)
    counts = Counter(row["dish_label"] for row in search.rows)
    for query in queries:
        if query["true_label"] not in counts:
            raise ValueError(f"Unknown query label: {query['true_label']}")
        if not (dataset / "queries" / query["filename"]).is_file():
            raise FileNotFoundError(query["filename"])
    per_query = []
    for i, query in enumerate(queries, 1):
        results = search.search_image(dataset / "queries" / query["filename"], top_k=5)
        metrics = metrics_at_5(results, query["true_label"], counts[query["true_label"]])
        per_query.append({**query, "relevant_in_catalog": counts[query["true_label"]],
                          **metrics, "results": [dict(r, relevant=r["dish_label"] == query["true_label"])
                                                  for r in results]})
        print(f"Evaluated {i}/{len(queries)}: {query['query_id']}", file=sys.stderr)
    summary = {
        "query_count": len(queries), "catalog_count": len(search.rows), "k": 5,
        "averaging": "arithmetic mean over queries",
        **{metric: sum(row[metric] for row in per_query) / len(per_query) for metric in METRICS},
        "model": search.encoder.model.config._name_or_path,
        "model_revision": search.encoder.revision,
        "catalog_images_per_label": dict(sorted(counts.items())),
        "limitations": [
            "Labels do not measure ingredient, preparation, or dietary suitability.",
            "Rankings below five are ignored.",
            "Averages hide weak classes and individual failures; 60 queries are a small sample.",
            "These metrics do not measure latency or similarity-score calibration.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in [("summary.json", summary), ("results.json", per_query)]:
        (out / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    fields = ["query_id", "filename", "true_label", "relevant_in_catalog", *METRICS]
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_query)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/evaluation")
    args = parser.parse_args()
    try:
        summary = evaluate(args.dataset, args.index, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
