# FitSloth image search

Image-to-image food search using pretrained CLIP ViT-B/32 image embeddings and
an exact NumPy cosine-similarity index. No training or server is needed.

## Run

Use [uv](https://docs.astral.sh/uv/) and the downloaded `food_search_dataset/`.
uv manages Python 3.11, the `.venv` environment, and dependencies for this project.
No environment activation is needed.

```bash
uv sync --locked
uv run food_search.py build
uv run food_search.py search food_search_dataset/queries/q_pho_00.jpg --top-k 5 --preview artifacts/pho_results.png
```

Skip `build` when `artifacts/catalog.npz` already exists and the catalog has not
changed. If your shell cannot find `uv`, use `~/.local/bin/uv` in its place.

Dependencies are declared in `pyproject.toml`; `uv.lock` records the resolved
versions. Commit both files when changing dependencies. Use `uv add package-name`
to add a dependency and `uv remove package-name` to remove one. Run
`uv lock --upgrade` followed by `uv sync` to update dependencies within their
declared constraints. The four original direct dependencies retain their exact pins.

The first build downloads the model into `.cache/huggingface/`, embeds only the
2,000 catalog images, and saves `artifacts/catalog.npz`. Subsequent searches reuse
the saved vectors and cached model. Rebuild after changing the catalog or images.
The index records the model revision so query and catalog encoders match.
The CLI prints JSON containing rank, image ID, filename, dish label, cosine score,
and absolute image path. Progress goes to stderr. The optional PNG shows the
query and its ranked matches. Scores are similarities, not probabilities.

For repeated queries, keep one model loaded:

```python
from food_search import ImageSearch

search = ImageSearch()
results = search.search_image("food_search_dataset/queries/q_pho_00.jpg", top_k=5)
```

`search_image(...)` is also available as a standalone convenience function.
Use `--dataset` and `--index` before the subcommand to override default paths.
If querying a catalog image itself, that image is eligible to appear in results.

## Design

The [CLIP image encoder](https://huggingface.co/openai/clip-vit-base-patch32)
produces 512-dimensional vectors. We use its provided resize/crop/normalization
processor and L2-normalize embeddings. Dot products then equal cosine similarity.
CPU inference with eager attention keeps indexing and queries on the same backend.
Before saving, the build checks that a single-image embedding matches its batched vector.
NumPy scans all 2,000 vectors exactly: the float32 matrix is only about 4 MB,
so an approximate index or database adds unnecessary complexity at this size.
Tied scores retain catalog order. `top_k` larger than the catalog returns all images.

Dish labels are returned for inspection but never fed into the encoder or ranking.
The held-out query images are never indexed. This implements the initial search
baseline and 60-query metric evaluation; the three-failure analysis is future work.
Visual resemblance alone may miss ingredients and confuse similar soups or cakes;
one successful example is not a measure of overall quality.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```

## Evaluate all 60 queries

```bash
uv run evaluate.py
```

Loads the model/index once and searches every image in `queries.csv`. Saves
`artifacts/evaluation/summary.json` (mean metrics), `per_query.csv` (individual
metrics), and `results.json` (ranked matches, scores, and relevance).
Use `--dataset`, `--index`, or `--out` to override paths. For a cached offline
run, prefix the command with `HF_HUB_OFFLINE=1`.

Precision@5 is relevant hits divided by 5. Recall@5 divides hits by the matching
label's catalog count (100 here, so its maximum is 0.05). NDCG@5 discounts binary
relevance by `1 / log2(rank + 1)` and divides by ideal DCG (about 2.948 here).
Each reported number is the arithmetic mean across queries. These metrics measure
label matching, not ingredient suitability, user intent, latency, or score calibration.
Averages can hide poor classes and individual failures, especially with only 60 queries.

## Engineering notes

The initial prototype ran in the existing Python 3.12 environment. The project
setup now uses an isolated Python 3.11 virtual environment to match the brief.
The model weights and saved catalog vectors are reused across environments.

An initial accelerated indexing run produced embeddings inconsistent with CPU
queries. We rebuilt with CPU inference and eager attention, and verified that a
catalog image encoded again ranks itself first with cosine similarity near 1.
We retained the same libraries; no library replacement was necessary.
