"""CLIP image retrieval with a persistent, exact NumPy cosine index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "food_search_dataset"
INDEX = ROOT / "outputs/catalog.npz"
MODEL = "openai/clip-vit-base-patch32"
CACHE = ROOT / ".cache/huggingface"


def catalog(dataset):
    with (dataset / "catalog.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Catalog must contain unique image IDs and at least one image")
    return rows


def fingerprint(dataset, rows):
    digest = hashlib.sha256((dataset / "catalog.csv").read_bytes())
    for row in rows:
        path = dataset / "images" / row["filename"]
        digest.update(path.read_bytes())
    return digest.hexdigest()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("Embeddings must be finite and nonzero")
    return vectors / norms


def rank(vectors, query, top_k):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scores = vectors @ normalize(query).reshape(-1)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return order, scores[order]


class Encoder:
    def __init__(self, model=MODEL, revision=None, projection=None):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch = torch
        # Use one backend for build and search, including sandboxed executions.
        self.device = "cpu"
        torch.set_num_threads(min(8, torch.get_num_threads()))
        options = dict(cache_dir=str(CACHE), revision=revision)
        self.processor = CLIPImageProcessor.from_pretrained(model, **options)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model, attn_implementation="eager", **options)
        self.model.to(self.device).eval()
        if projection is not None:
            weight = np.load(projection, allow_pickle=False)
            if weight.shape != tuple(self.model.visual_projection.weight.shape) or not np.isfinite(weight).all():
                raise ValueError("Invalid fine-tuned projection")
            with torch.no_grad():
                self.model.visual_projection.weight.copy_(torch.from_numpy(weight))
        self.revision = self.model.config._commit_hash
        print(f"Encoder: {model} on {self.device}", file=sys.stderr)

    def encode(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            vectors = self.model(**inputs).image_embeds.cpu().numpy()
        return normalize(vectors)


def build_index(dataset=DATASET, index=INDEX, batch_size=32):
    dataset, index = Path(dataset), Path(index)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = catalog(dataset)
    source_hash = fingerprint(dataset, rows)
    encoder = Encoder()
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        chunks.append(encoder.encode([dataset / "images" / r["filename"] for r in batch]))
        print(f"Embedded {min(start + batch_size, len(rows))}/{len(rows)}", file=sys.stderr)
    vectors = np.concatenate(chunks)
    probe = encoder.encode([dataset / "images" / rows[0]["filename"]])[0]
    if float(probe @ vectors[0]) < 0.999:
        raise ValueError("Batch and single-image embeddings disagree; index was not saved")
    metadata = dict(model=MODEL, revision=encoder.revision, fingerprint=source_hash,
                    rows=rows, format_version=1)
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, vectors=vectors, metadata=json.dumps(metadata))
    temporary.replace(index)
    print(f"Saved {index}", file=sys.stderr)


class ImageSearch:
    """Load once and reuse for multiple queries. Labels never affect ranking."""
    def __init__(self, dataset=DATASET, index=INDEX):
        self.dataset = Path(dataset)
        with np.load(index, allow_pickle=False) as data:
            self.vectors = normalize(data["vectors"])
            metadata = json.loads(str(data["metadata"]))
        self.rows = catalog(self.dataset)
        if (metadata["format_version"] != 1 or metadata["rows"] != self.rows
                or metadata["fingerprint"] != fingerprint(self.dataset, self.rows)):
            raise ValueError("Catalog changed; rebuild the index")
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.rows):
            raise ValueError("Invalid index dimensions; rebuild the index")
        projection = None
        if "projection" in metadata:
            projection = Path(index).resolve().parent / metadata["projection"]
            if hashlib.sha256(projection.read_bytes()).hexdigest() != metadata["projection_sha256"]:
                raise ValueError("Fine-tuned projection changed; rebuild the index")
        self.metadata = metadata
        self.encoder = Encoder(metadata["model"], metadata["revision"], projection)

    def search_image(self, image_path, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = self.encoder.encode([Path(image_path)])[0]
        order, scores = rank(self.vectors, query, top_k)
        return [dict(rank=i + 1, **self.rows[int(idx)], score=float(score),
                     path=str((self.dataset / "images" / self.rows[int(idx)]["filename"]).resolve()))
                for i, (idx, score) in enumerate(zip(order, scores))]


def search_image(image_path, top_k=5, dataset=DATASET, index=INDEX):
    return ImageSearch(dataset, index).search_image(image_path, top_k)


def save_preview(query, results, destination):
    width, height = 256, 310
    canvas = Image.new("RGB", (width * (len(results) + 1), height), "#f4f4f4")
    draw = ImageDraw.Draw(canvas)
    entries = [(str(query), "Query", Path(query).name)] + [
        (r["path"], f'{r["rank"]}. {r["dish_label"]}', f'Cosine: {r["score"]:.4f}') for r in results]
    for i, (path, title, subtitle) in enumerate(entries):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((240, 240))
            canvas.paste(image, (i * width + (width - image.width) // 2, 8 + (240 - image.height) // 2))
        draw.text((i * width + 8, 256), title, fill="black")
        draw.text((i * width + 8, 278), subtitle, fill="black")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--index", type=Path, default=INDEX)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Download model and embed catalog once")
    build.add_argument("--batch-size", type=int, default=32)
    search = commands.add_parser("search", help="Return ranked results as JSON")
    search.add_argument("image", type=Path)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--preview", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build_index(args.dataset, args.index, args.batch_size)
        else:
            results = search_image(args.image, args.top_k, args.dataset, args.index)
            if args.preview:
                save_preview(args.image, results, args.preview)
            print(json.dumps(results, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
