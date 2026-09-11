# Food image search — Junior AI Engineer take-home

Upload a food photo and retrieve the five most similar catalog images. This project uses CLIP image embeddings, with two experiments: **fine-tuning the image projection** and **searching automatic YOLO crops alongside the full image**.

The local UI supports baseline and fine-tuned CLIP, displays cosine scores, and shows which input image or crop produced each result's score.

## Scope and brief requirements

This submission implements image-to-image search over the supplied catalog. The
required baseline is pretrained CLIP with a persistent NumPy index. The two
extensions are projection fine-tuning and automatic region-based retrieval;
the UI is a way to use and inspect those experiments.

| Requirement in the brief | Implementation and evidence |
|---|---|
| Query to ranked results with scores | `image_search.py` CLI and `ImageSearch.search_image()`; the UI accepts uploaded images |
| Embeddings and an index | CLIP ViT-B/32, 512-dimensional normalized vectors, saved exact NumPy search index |
| Mean Precision@5, Recall@5, NDCG@5 on all 60 queries | `evaluate_search.py` and the results table below; relevant means `dish_label == true_label` |
| Three bad queries and why they might fail | Three illustrated baseline failures below, with observations, hypotheses, and follow-up checks |
| One or two substantive extensions | Projection fine-tuning; full-image/crop ablation using identical inputs across encoders |
| Reproducible running instructions | Setup, CLI, UI, and experiment commands below; Python 3.11 and `uv.lock` |
| Models used and specific AI involvement | Model/training sections and AI assistance section |
| 3–5 minute screen recording including a failure | Still to be recorded and attached; suggested demonstration sequence below |

The code does not estimate calories, portion sizes, ingredients, or medical
suitability. No personal data or external inference API keys are required. Use the
supplied food examples when recording the demonstration.

## Setup

Run commands from the project root. Requirements: **Python 3.11** and **uv**.
Dependencies are pinned in `uv.lock`; inference runs on CPU.

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install dependencies:

```bash
uv sync --locked
```

uv manages `.venv`; you do not need to activate it. Initial dataset and model downloads require internet access. If the supplied dataset and saved model indexes are already present, go directly to **Launch the UI**.

For a fresh checkout, prepare the dataset and baseline index:

```bash
uv run prepare_data.py
uv run image_search.py build
```

Keep `prepare_data.py` and its output structure unchanged. Skip preparation when `food_search_dataset/` is already complete.
The baseline is pretrained CLIP and requires indexing, not training.
To create the fine-tuned model as well:

```bash
uv run finetune_clip.py
```

This writes `outputs/tuned/`. If that folder already exists, use the saved model or choose a new output folder as described under **Reproduce the experiments**.

## Launch the UI

```bash
uv run search_ui.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and keep the terminal running.
Press **Ctrl+C** in that terminal to stop the server.

1. Upload or drop a JPEG, PNG, or WebP photo, or choose a dataset example.
2. Select **Baseline CLIP**, **Fine-tuned CLIP**, or **Both models**.
3. Choose **Full image**, **Random crops**, **YOLO crops**, or **Full image + YOLO crops**.
4. Inspect Top-5 results, cosine scores, and selected-input thumbnails. Expand **All input scores** or **Search results for each detected item** for details.

The default is fine-tuned CLIP with full image + YOLO. If you have only built the
baseline index, select **Baseline CLIP**. The UI uses `outputs/catalog.npz` and
`outputs/tuned/catalog.npz`.

Uploads are processed locally and temporary input files are removed after each
search. The limit is 16 MB and 20 million pixels. Models load on first use and stay
in memory; searches run one at a time. Displayed time includes model loading when
needed and is not a controlled latency benchmark.

New uploads have no ground-truth label. The UI therefore shows similarity scores,
not correctness, Precision@5, Recall@5, or NDCG@5. Those metrics belong to the
labeled dataset evaluation.

### Port already in use

If startup reports `Port 8000 is already in use`, an app is already listening there.
If it is this search UI, use the existing browser page instead of starting another server.
To use a different port:

```bash
uv run search_ui.py --port 8001
```

Then open [http://127.0.0.1:8001](http://127.0.0.1:8001).

### Editing the UI

Edit `ui/index.html`, save the file, and refresh the browser (Cmd+R on macOS).
The server reads this file on each page request and disables caching, but does not
automatically reload an already-open page. The static comparison report at
`outputs/compare/index.html` is a separate page.

Startup prints the local URL on its own line and the exact HTML path. Click the
URL in terminals that support links (some require Cmd+click). To open the default
browser automatically:

```bash
uv run search_ui.py --open
```

Changes to `search_ui.py` require stopping and restarting that server. An existing
server on port 8000 keeps running its loaded Python code until restarted.

## Search from the command line

Baseline search with a saved image preview:

```bash
uv run image_search.py search food_search_dataset/queries/q_pho_00.jpg --top-k 5 --preview outputs/search.png
```

Fine-tuned search:

```bash
uv run image_search.py --index outputs/tuned/catalog.npz search food_search_dataset/queries/q_pho_00.jpg --top-k 5
```

Replace the query path with your own image. Results are JSON containing rank,
image ID, dish label, cosine score, and image path. **Cosine is not a probability.**
The CLI searches the full image; use the UI for interactive crop-based search.

## Data and retrieval

The supplied preparation script builds a fixed Food-101 subset:

| Location | Contents |
|---|---|
| `food_search_dataset/images/` | 2,000 catalog images: 20 classes × 100 images |
| `food_search_dataset/catalog.csv` | `image_id`, `filename`, `dish_label` |
| `food_search_dataset/queries/` | 60 separate queries: three per class |
| `food_search_dataset/queries.csv` | `query_id`, `filename`, `true_label` |

Each image is EXIF-oriented, converted to RGB, and processed by the pretrained
CLIP image processor. `openai/clip-vit-base-patch32` produces a 512-dimensional
embedding, which is L2-normalized. The dot product of two normalized embeddings
is their cosine similarity.

CLIP supplies a pretrained visual representation without requiring annotations
beyond the supplied dish labels. It is a practical starting point for visually
similar dishes, but its general-purpose features are not guaranteed to separate
these categories. NumPy is sufficient for only 2,000 vectors: it gives exact,
inspectable rankings without approximate-search tuning or another server. These
are simplicity choices, not claims that CLIP or NumPy is optimal at larger scale.

Catalog embeddings are computed once and stored in a NumPy index. Each query
searches all 2,000 vectors exactly. Labels do not influence ranking. Both encoders
use the same catalog images, encoded with their respective weights. Index loading
checks dataset fingerprints and, for the fine-tuned model, the projection hash.

## Fine-tuning

`finetune_clip.py` freezes the vision backbone and learns a 512 × 512 transform `A`.
It folds that transform into CLIP's image projection:

```text
W_new = A @ W_original
```

This is **projection fine-tuning**, not training the entire CLIP model. Text
alignment is not evaluated. Limiting training to the projection keeps the experiment
feasible on a CPU and limits the number of parameters adapted to a small catalog.
Identity initialization and regularization keep the starting behavior close to CLIP.

The catalog is split into **1,600 training images** and **400 validation images**.
Exact decoded-pixel matches and dHash duplicate candidates are grouped before
splitting. Groups touching query images or containing conflicting labels are
excluded. The saved audit found no candidates; this heuristic can still miss
near-duplicates or group distinct images.

Supervised contrastive loss treats different images with the same dish label as
positives and other dish labels as negatives. Self-pairs are excluded.

| Training setting | Value |
|---|---|
| Optimizer / learning rate | AdamW / 0.001 |
| Epochs / seed | 30 / 42 |
| Batch composition | Eight images per class |
| Temperature / identity regularization | 0.07 / 0.01 |
| Checkpoint selection | Highest validation NDCG@5 |
| Selected epoch in the saved run | 9 |

Validation queries search training images only. Ties keep the earlier checkpoint;
the unchanged initialization is eligible. Query labels and results are not used
for training or checkpoint selection. Final evaluation searches the full catalog.

## Crop strategies and score selection

Shared-plate photos motivate the region experiment: one embedding of a whole meal
can mix the target dish with other objects. Automatic crops can isolate useful
content, while retaining the full image can compensate for missed detections.
Random crops provide a simple control for cropping without a detector. The
experiment measures dish-label retrieval, not exhaustive multi-item recognition.

Each encoder is evaluated with four strategies, giving eight pipelines:

| Strategy | How results are calculated |
|---|---|
| Full image | One query embedding produces one Top-5 list |
| Random crops | Five independently searched crops; average all trial metrics |
| YOLO crops | Maximum cosine across detected crops per catalog candidate |
| Full image + YOLO crops | Maximum cosine across the full image and detected crops per candidate |

For full + YOLO:

```text
score(candidate) = max(cosine(full_image, candidate),
                       cosine(crop_1, candidate), ...)
```

Sort these scores and keep five unique catalog images. Different results can
select different inputs. Exact ties use input order, with the full image first
when included. Compute metrics once on the fused Top-5; **do not average metrics
across YOLO crops**. Random mode instead keeps separate trials and averages their
metrics without selecting the best trial.

YOLO uses `yolo26n-seg.pt`, image size 640, confidence 0.25, minimum box area 1%,
and IoU 0.5 duplicate suppression. It proposes fixed food/container categories
without text prompts. Although masks are displayed, retrieval uses **unmasked
rectangular box crops**. Detector labels are not sent to CLIP and do not define
relevance. If no eligible regions are found, YOLO-only uses the full image;
this occurred for four of the 60 saved queries.

Random crop areas range from 25% to 90%, with log-uniform aspect ratios from 3/4
to 4/3 and random positions. Both encoders use identical crops in a comparison.
Five separate random trials and multi-region YOLO are not compute-matched experiments.

## Reproduce the experiments

After preparing the data and baseline index, run the following on a fresh checkout:

```bash
uv run evaluate_search.py
uv run finetune_clip.py
uv run compare_pipelines.py --seed 1968095357
uv run show_search_cases.py
```

Skip training if the saved `outputs/tuned/` is already present. Training and
comparison refuse to overwrite existing output folders. For a new training run:

```bash
uv run finetune_clip.py --out outputs/tuned_run2
uv run compare_pipelines.py --finetuned outputs/tuned_run2/catalog.npz --out outputs/compare_run2 --seed 1968095357
```

To reuse the saved crops and detections and rerun the searches:

```bash
uv run compare_pipelines.py --resume --out outputs/compare
```

Omit `--seed` for fresh random crops, or use `--random-trials` to change trial count.
Resume must use the same configuration as the existing run. To evaluate only the
fine-tuned full-image model:

```bash
uv run evaluate_search.py --index outputs/tuned/catalog.npz --out outputs/tuned/finetuned
```

## Metrics and saved results

A result is relevant when its catalog `dish_label` equals the query's `true_label`.
Cosine determines ranking; label matches determine evaluation scores.

| Metric | Definition for one query |
|---|---|
| Precision@5 | Number of relevant Top-5 results / 5 |
| Recall@5 | Number of relevant Top-5 results / 100 |
| NDCG@5 | Sum of `relevance / log2(rank + 1)`, divided by the ideal five-hit score, approximately 2.948 |

Average each metric across all 60 queries. **Recall@5 is at most 0.05**, because
there are 100 relevant catalog images per query. Validation recall uses 80 relevant
training images in this split and is not directly comparable with final recall.

Saved results use training seed 42 and crop seed 1968095357:

| Encoder | Query inputs | Precision@5 | Recall@5 | NDCG@5 |
|---|---|---:|---:|---:|
| Baseline | Full image | 0.796667 | 0.039833 | 0.791588 |
| Baseline | Random crops | 0.774000 | 0.038700 | 0.777571 |
| Baseline | YOLO crops | 0.713333 | 0.035667 | 0.715367 |
| Baseline | Full image + YOLO crops | 0.816667 | 0.040833 | 0.818405 |
| Fine-tuned | Full image | 0.930000 | 0.046500 | 0.926616 |
| Fine-tuned | Random crops | 0.898667 | 0.044933 | 0.900559 |
| Fine-tuned | YOLO crops | 0.806667 | 0.040333 | 0.807525 |
| Fine-tuned | Full image + YOLO crops | 0.946667 | 0.047333 | 0.946926 |

Projection fine-tuning improved full-image Precision@5 from 79.67% to 93.00%.
Adding YOLO crops while keeping the full image reached 94.67%. YOLO crops alone
reduced average performance. These findings apply to this experiment; the queries
were inspected during development and are not a pristine unseen test.

Open the [comparison report](outputs/compare/index.html) or
[summary CSV](outputs/compare/summary.csv) after generating or receiving the outputs.
The report shows crops, detections, rankings, and the winning input per result.
Expand **Cosine from every input** to inspect score selection. This is score
provenance, not an attention map. `manifest.json` records seeds and detections;
`results.json` retains trial rankings, `source`, and `source_scores`.

## Inspect success and failure cases

Export three best cases and three wrong-top-1 cases from the baseline evaluation:

```bash
uv run show_search_cases.py
```

Read `outputs/cases/report.md` and its image strips. To export every fine-tuned
wrong-top-1 case:

```bash
uv run show_search_cases.py --all-failures --results outputs/tuned/finetuned/results.json --out outputs/tuned/cases
```

Open `outputs/tuned/cases/failures.html`. Selection uses Precision@5 and NDCG@5,
with cosine breaking best-case ties. A failure here means the top-ranked catalog
image has the wrong supplied dish label. This is a case-selection rule; the metrics
still evaluate all five results for all 60 queries.

### Three baseline failures

The following images come from `outputs/eval/results.json` and were exported by
`show_search_cases.py`. Each strip places the query first, then the five retrieved
images. These are deliberately difficult examples, not estimates of average quality.

| Query / supplied label | Top-1 label | Precision@5 | Recall@5 | NDCG@5 |
|---|---|---:|---:|---:|
| `q_sushi_00` / sushi | caesar_salad | 0.00 | 0.000 | 0.000000 |
| `q_caesar_salad_02` / caesar_salad | greek_salad | 0.20 | 0.010 | 0.131205 |
| `q_caprese_salad_00` / caprese_salad | greek_salad | 0.20 | 0.010 | 0.131205 |

**1. Sushi: a busy shared-platter composition.**

![Sushi query and baseline Top-5 results](outputs/cases/failure_q_sushi_00.png)

Observation: the query contains a platter, green garnish, several food shapes,
and background containers. All five retrieved images are salads. Hypothesis:
the global representation emphasizes the overall arrangement and colors more
than the sushi-specific details. This is consistent with the result, but is not
proven by an attention analysis. A follow-up check is an independently annotated
target-region comparison on fresh images. In this saved query, fine-tuned full-image
Precision@5 reaches 1.00, YOLO-only gives 0.20, and full + YOLO returns to 1.00;
automatic cropping alone does not solve the missed-target problem.

**2. Caesar salad: category cues are not visually distinctive.**

![Caesar salad query and baseline Top-5 results](outputs/cases/failure_q_caesar_salad_02.png)

Observation: a small serving of greens and a pale round item sit on a large light
plate. Retrieved Greek and caprese salads share greens, pale ingredients, and
plated composition; the first relevant Caesar salad appears at rank five.
Hypothesis: the embedding does not reliably distinguish fine ingredient or
preparation cues from the shared salad appearance. A useful check would compare
class-wise errors on additional labeled salad photos with varied plating.

**3. Caprese salad: visual appearance and the supplied label may disagree.**

![Caprese salad query and baseline Top-5 results](outputs/cases/failure_q_caprese_salad_00.png)

Observation: the query includes tomatoes, dark olives, greens, and pale cubes,
visually resembling the Greek-salad results. The first four results are labeled
Greek salad. Hypothesis: the query may be atypical for its category or its label
may be ambiguous; a retrieval error is not necessarily an unreasonable visual
match. This needs an independent label review. The evaluation keeps the original
`caprese_salad` label unchanged, as required by the brief and supplied dataset.

These hypotheses are based on visual inspection, not established causal
explanations. Fine-tuning improves aggregate results but still has wrong-top-1
cases, including `q_miso_soup_00`, `q_caprese_salad_00`, and `q_chocolate_cake_00`.

## Code guide

Read the core files in this order:

| File | What to understand |
|---|---|
| `prepare_data.py` | Supplied dataset preparation; keep the script and output unchanged |
| `image_search.py` | Baseline encoder, index, normalization, and cosine ranking |
| `evaluate_search.py` | Precision, recall, NDCG, and averaging |
| `finetune_clip.py` | Split isolation, positive/negative pairs, projection training, checkpoint selection |
| `detect_food.py` | YOLO proposals and box filtering |
| `compare_pipelines.py` | Four input strategies × two encoders, score fusion, and report generation |
| `show_search_cases.py` | Selection and presentation of retrieval examples |
| `search_ui.py` | Local HTTP server, upload validation, and reusable search models |
| `ui/index.html` | Browser controls, crop previews, and result cards |
| `tests/` | Software correctness tests; separate from the image evaluation queries |

Run checks:

```bash
uv run python -m unittest discover -s tests -v
```

## Saved files and troubleshooting

| Path | Purpose |
|---|---|
| `outputs/catalog.npz` | Baseline catalog index |
| `outputs/eval/` | Baseline evaluation |
| `outputs/tuned/` | Projection, transform, matching index, split, audit, history, and evaluations |
| `outputs/compare/` | Eight-way results and linked images |
| `outputs/cases/` | Generated baseline examples |
| `outputs/audit/` | Dataset inspection notes and contact sheets |
| `.cache/` | Model and package caches |

- **`uv: command not found`:** use `~/.local/bin/uv` or add `$HOME/.local/bin` to PATH.
- **Missing index:** run `uv run image_search.py build`; fine-tuned search also
  needs `uv run finetune_clip.py` or the saved trained artifacts.
- **Output folder exists:** use a new `--out`; comparison also supports `--resume`.
- **Catalog/projection mismatch:** use matching data, weights, and index. For an
  intentionally changed catalog, rebuild the index and train a new model.
- **Offline loading:** after weights are cached, prefix commands with
  `HF_HUB_OFFLINE=1`. This controls Hugging Face loading, not YOLO or dataset downloads.

Keep `projection.npy` beside the fine-tuned index and retain `transform.npy` for
comparison. The dataset, outputs, caches, and `.venv` are excluded from Git. A fresh
checkout must regenerate them. To share saved reports, include their output folders
and the matching dataset; stored absolute paths may require regeneration on another
machine. Do not paste explanatory inline comments after commands in interactive zsh.

## What the metrics do not show

One dish label per image cannot evaluate every item in a shared meal. There are
no per-item ground-truth boxes or relevance labels. The dataset has no dedicated
drink categories, although drinks may appear in food photos. Generic YOLO classes
can miss dishes, and extra crops can introduce spurious high similarities.

The saved experiment uses 60 queries and one seed per experiment, without confidence
intervals or controlled latency comparisons. Averages hide individual failures;
fresh, independently annotated images are needed to assess real-world performance.
Dish-label relevance does not establish ingredients, nutrition, dietary suitability,
or portion size. Text alignment, unknown-category handling, and score calibration
are not evaluated.

## AI assistance and engineering decisions

AI assisted implementation, metric tests, projection training, experiment reports,
the upload UI, and documentation. Specific disagreements and corrections were:

- Manual crop coordinates were rejected because they would not support a new
  uploaded photo. The retained workflow proposes regions automatically.
- A prompt-based detector was rejected for the image-only workflow; the retained
  detector uses fixed pretrained YOLO classes without text prompts.
- An accelerated indexing mismatch was corrected by using consistent CPU inference
  and checking cached embeddings against fresh encodings.
- An early explanation blamed drinks for crop failures before inspecting the
  dataset. Inspection showed that the miso example's soup crop can dominate the
  drink crops; missed target regions and shared visual features also matter.
- Score provenance was added to show which full image or crop actually won for
  each retrieved candidate, instead of guessing from the final ranking.

Earlier GrabCut and DINO experiments are not part of the active implementation.
The negative YOLO-only result is retained alongside the positive fine-tuning and
full + YOLO results. AI-generated code must still be explainable and editable by
the candidate during the interview.

## Screen recording and submission

**The required 3–5 minute recording is not included yet.** Record the working
system and attach the video or an accessible video link with the submission.
A suggested sequence, using only supplied food examples:

1. **0:00–0:40:** show the UI and explain image input, CLIP, and the saved catalog index.
2. **0:40–1:30:** search `q_pho_00.jpg` with baseline/full image and show Top-5 scores.
3. **1:30–2:30:** search `q_sushi_00.jpg` with baseline/full image, show the failure,
   and explain the shared-platter hypothesis.
4. **2:30–3:40:** compare encoders and full + YOLO; expand the winning input scores
   and explain that YOLO can miss the intended target.
5. **3:40–4:30:** show the eight-way metrics, the evaluation limitations, and one
   concrete correction to AI-generated work.

Before sending, include the source, `pyproject.toml`, `uv.lock`, this README, and
the recording. Include saved result folders if the reviewer should see them
without recomputation, or run the documented commands to regenerate them. Check
that the three failure images and comparison report are accessible in the package.
The source-only repository ignores generated outputs; README image links require
those outputs. Keep `prepare_data.py` and its dataset unchanged.

