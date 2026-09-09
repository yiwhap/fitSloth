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

## Show three best and three failure cases

After running the evaluation:

```bash
uv run show_cases.py
open artifacts/cases/best_cases.png
open artifacts/cases/failure_cases.png
```

The script reads `artifacts/evaluation/results.json` without loading CLIP again.
Rerun `uv run evaluate.py` after changing the model, index, or queries to refresh
those results. It saves six individual previews, two contact sheets, `cases.json`,
and `report.md` under `artifacts/cases/`.

Best cases rank by Precision@5, then NDCG@5, then top-1 cosine (descending).
Failure cases require a wrong top-1 dish label and rank by Precision@5 then
NDCG@5 (ascending). Final ties use query ID. This selects global extremes, so
multiple examples can share a dish label. Cosine is only a best-case tie breaker,
not a measure of correctness. Use `--results`, `--dataset`, and `--out` to override paths.

To show every top-1 failure in a scrollable image gallery:

```bash
uv run show_cases.py --all-failures --out artifacts/all_cases
open artifacts/all_cases/failures.html
```

This retains the same failure definition: the first result has a different dish
label from the query. Queries with a correct first result but mistakes farther
down the ranking are not included.

## Fine-tuned CLIP

A trained image projection is available in `artifacts/finetuned_clip/`. Search
with it using the same CLI; the index automatically loads its matching projection:

```bash
uv run food_search.py --index artifacts/finetuned_clip/catalog.npz search food_search_dataset/queries/q_pho_00.jpg --top-k 5 --preview artifacts/finetuned_clip/search.png
```

The original `artifacts/catalog.npz` remains the baseline. The training run also
preserves a copy as `artifacts/finetuned_clip/baseline_catalog.npz`.

### Method and split

`finetune_clip.py` learns a 512-by-512 linear transform A of the original CLIP image
embeddings, initialized to identity. It then folds A into CLIP's actual visual
projection: `W_new = A @ W_original`. This is **constrained projection fine-tuning**,
not full-backbone fine-tuning. The vision transformer is frozen, and the text
encoder is not trained or evaluated. Because output embeddings are L2-normalized,
training on cached normalized embeddings is equivalent to applying the folded
projection to the frozen backbone. Four image encodings verify this equivalence
before the new index is saved. No new dependency or GPU is required.

The loss is [supervised contrastive learning](https://arxiv.org/abs/2004.11362):
other images with the same dish label are positives and other labels are negatives.
Balanced batches sample eight images per class. AdamW uses learning rate 0.001,
temperature 0.07, and a 0.01 identity regularization coefficient; seed is 42.
The projection follows the [CLIP vision model architecture](https://huggingface.co/docs/transformers/model_doc/clip).

Before splitting, the script groups exact decoded-pixel duplicates and candidate
near-duplicates with 64-bit dHash distance at most four. Groups remain together;
groups touching query images or containing conflicting labels are excluded from
training/validation. This heuristic can miss duplicates and produce false positives.
The current audit found no candidates, yielding 1,600 training and 400 validation
catalog images (80/20 per class). Validation queries search only training images,
so there are no self-matches. Query labels and retrieval outcomes are never used
for training or checkpoint selection.

The checkpoint with highest validation NDCG@5 across 30 epochs is saved; ties keep
the earlier epoch. The unchanged initialization is eligible if training fails to
improve validation. The current run selected epoch 9. After selection, both models
are evaluated on the same 60 query images against the full 2,000-image catalog.
Validation recall uses 80 relevant training images; final recall uses 100 catalog
images, so those recall values should not be compared directly.

### Measured comparison

| Model | Precision@5 | Recall@5 | NDCG@5 |
|---|---:|---:|---:|
| Original CLIP | 0.796667 | 0.039833 | 0.791588 |
| Fine-tuned projection | 0.930000 | 0.046500 | 0.926616 |

NDCG@5 improved for 19 queries, decreased for 3, and stayed unchanged for 38.
The regressions are `q_hot_and_sour_soup_02`, `q_miso_soup_00`, and `q_pad_thai_02`.
This is one seed on a small known-category dataset. The held-out queries had
already been visually inspected during project exploration; they are not a
pristine unseen test set. These results do not establish performance on new foods
or guarantee preserved text/image alignment.

### Reproduce and inspect

Run a fresh experiment in a new output directory (existing run directories are
never overwritten):

```bash
uv run finetune_clip.py --out artifacts/finetuned_clip_run2
```

Use `HF_HUB_OFFLINE=1` before the command when the model is cached and you want an
offline run. `--epochs`, `--lr`, and `--seed` configure training. Choose settings
using catalog validation results, rather than repeatedly optimizing the 60 queries.

Re-evaluate the saved model or inspect all its top-1 failures:

```bash
uv run evaluate.py --index artifacts/finetuned_clip/catalog.npz --out artifacts/finetuned_clip/finetuned
uv run show_cases.py --all-failures --results artifacts/finetuned_clip/finetuned/results.json --out artifacts/finetuned_clip/cases
open artifacts/finetuned_clip/cases/failures.html
```

The run directory contains `projection.npy` (the trained CLIP projection),
`transform.npy`, the matching `catalog.npz`, split and duplicate audits,
`training.json` (epoch history), baseline/fine-tuned evaluation folders,
`comparison.md`, `comparison.json`, and `per_query_changes.json`.
Keep `projection.npy` beside its index; its hash is checked when loading.
Run artifacts are ignored by Git, so retain or share this directory separately
when sharing the trained model. If the catalog changes, rebuild the original
baseline index and start a new training run.

## Compare both models visually

```bash
uv run compare_models.py
open artifacts/model_comparison/index.html
```

This produces a local report for all 60 queries using the saved evaluations.
It shows mean Precision@5, Recall@5, and NDCG@5, plus each query's metrics and
baseline/fine-tuned top-five images. Filter by dish or improved/worse/unchanged
NDCG@5; click an image panel to open the full-size PNG. No server is needed.
A per-query CSV and summary JSON are also saved. The script pairs queries by ID,
checks catalog compatibility, and recomputes metrics from the ranked labels.
Use `--baseline` and `--finetuned` for alternate evaluation directories, and
`--out` for another output folder. Rerun evaluation first after changing a model.
