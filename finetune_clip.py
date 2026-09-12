"""Train a constrained CLIP image projection using catalog-only supervised contrastive loss."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_search import METRICS, evaluate, metrics_at_5
from image_search import DATASET, INDEX, ROOT, Encoder, catalog, fingerprint, normalize


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def duplicate_groups(paths, distance=4):
    """Conservative dHash grouping; heuristic candidates, not guaranteed duplicates."""
    hashes, exact = [], []
    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            exact.append(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
            gray = np.asarray(im.convert('L').resize((9, 8)))
            hashes.append(int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big'))
    parent = list(range(len(paths)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pairs = []
    for i in range(len(paths)):
        for j in range(i):
            dist = (hashes[i] ^ hashes[j]).bit_count()
            if exact[i] == exact[j] or dist <= distance:
                parent[root(i)] = root(j)
                pairs.append({'a': str(paths[j]), 'b': str(paths[i]),
                              'exact_pixels': exact[i] == exact[j], 'dhash_distance': dist})
    return [root(i) for i in range(len(paths))], pairs


def split_catalog(rows, groups, query_groups, seed):
    """Keep near-duplicate groups together and exclude groups touching held-out queries."""
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    by_label = defaultdict(list)
    excluded = []
    for group, ids in members.items():
        labels = {rows[i]['dish_label'] for i in ids}
        if group in query_groups or len(labels) != 1:
            excluded.extend(ids)
        else:
            by_label[next(iter(labels))].append(ids)
    train, validation = [], []
    for label in sorted({r['dish_label'] for r in rows}):
        clusters = by_label[label]
        if len(clusters) < 2:
            raise ValueError(f'Not enough independent groups for {label}')
        rng.shuffle(clusters)
        target = max(1, round(sum(map(len, clusters)) * .2))
        held = []
        for cluster in clusters[:-1]:
            if len(held) < target:
                held.extend(cluster)
            else:
                train.extend(cluster)
        train.extend(clusters[-1])
        validation.extend(held)
    return sorted(train), sorted(validation), sorted(excluded)


def supervised_contrastive(vectors, labels, temperature=.07):
    z = F.normalize(vectors, dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (labels[:, None] == labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('Every anchor needs another same-label example')
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    return -(((logits - denominator[:, None]) * positives).sum(1) / counts).mean()


def retrieval_metrics(vectors, labels, train, validation):
    scores = vectors[validation] @ vectors[train].T
    order = np.argsort(-scores, axis=1, kind='stable')[:, :5]
    counts = Counter(labels[train])
    values = []
    for i, neighbors in zip(validation, order):
        results = [{'dish_label': int(labels[train[j]])} for j in neighbors]
        values.append(metrics_at_5(results, int(labels[i]), counts[labels[i]]))
    return {m: float(np.mean([v[m] for v in values])) for m in METRICS}


def train(args):
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError('epochs and learning rate must be positive')
    if args.out.exists():
        raise ValueError(f'{args.out} already exists; choose a new --out to preserve previous runs')
    torch.set_num_threads(4)
    torch.manual_seed(args.train_seed)
    rng = np.random.default_rng(args.train_seed)
    rows = catalog(args.dataset)
    with np.load(args.index, allow_pickle=False) as f:
        vectors = normalize(f['vectors'])
        metadata = json.loads(str(f['metadata']))
    if ('projection' in metadata or metadata['rows'] != rows or
            metadata['fingerprint'] != fingerprint(args.dataset, rows)):
        raise ValueError('Need an unchanged original CLIP catalog index')
    with (args.dataset / 'queries.csv').open() as f:
        queries = list(csv.DictReader(f))
    args.out.mkdir(parents=True)
    started = time.monotonic()
    paths = [args.dataset / 'images' / r['filename'] for r in rows]
    query_paths = [args.dataset / 'queries' / q['filename'] for q in queries]
    groups, pairs = duplicate_groups(paths + query_paths)
    train_ids, val_ids, excluded = split_catalog(rows, groups[:len(rows)], set(groups[len(rows):]), args.split_seed)
    dump(args.out / 'duplicate_audit.json', {'method': 'EXIF-normalized exact pixels or 64-bit dHash distance <= 4',
         'limitation': 'Heuristic grouping can miss duplicates and can group distinct images.', 'pairs': pairs})
    dump(args.out / 'split.json', {
        'seed': args.split_seed, 'split_seed': args.split_seed, 'catalog_fingerprint': metadata['fingerprint'],
        'train': [rows[i]['image_id'] for i in train_ids],
        'validation': [rows[i]['image_id'] for i in val_ids],
        'excluded': [rows[i]['image_id'] for i in excluded],
        'held_out_queries': [q['query_id'] for q in queries],
        'note': 'Query pixels used only for duplicate audit; query labels/results never used in training or checkpoint selection.'})
    print(f'Split: {len(train_ids)} train, {len(val_ids)} validation, {len(excluded)} excluded; {len(pairs)} duplicate candidates', flush=True)
    classes = sorted({r['dish_label'] for r in rows})
    labels = np.array([classes.index(r['dish_label']) for r in rows])
    class_ids = [np.array([i for i in train_ids if labels[i] == k]) for k in range(len(classes))]
    if any(len(ids) < 2 for ids in class_ids):
        raise ValueError('Each training label needs at least two images')
    x = torch.from_numpy(vectors)
    y = torch.from_numpy(labels)
    transform = nn.Linear(vectors.shape[1], vectors.shape[1], bias=False)
    identity = torch.eye(vectors.shape[1])
    with torch.no_grad():
        transform.weight.copy_(identity)
    optimizer = torch.optim.AdamW(transform.parameters(), lr=args.lr, weight_decay=0)
    best = identity.clone()
    baseline = retrieval_metrics(vectors, labels, train_ids, val_ids)
    best_score, best_epoch = baseline['ndcg@5'], 0
    history = [{'epoch': 0, 'validation': baseline}]
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(max(1, len(train_ids) // (len(classes) * 8))):
            batch = np.concatenate([rng.choice(ids, min(8, len(ids)), replace=False) for ids in class_ids])
            optimizer.zero_grad()
            loss = supervised_contrastive(transform(x[batch]), y[batch])
            loss = loss + .01 * (transform.weight - identity).square().sum() / len(identity)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            transformed = F.normalize(transform(x), dim=1).numpy()
        metrics = retrieval_metrics(transformed, labels, train_ids, val_ids)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation': metrics})
        if metrics['ndcg@5'] > best_score + 1e-9:
            best_score, best_epoch = metrics['ndcg@5'], epoch
            best = transform.weight.detach().clone()
        print(f'Epoch {epoch}/{args.epochs}: loss={np.mean(losses):.4f} validation NDCG@5={metrics["ndcg@5"]:.4f}', flush=True)
    dump(args.out / 'training.json', {'method': 'Constrained projection fine-tuning: W_new = A @ W_original; frozen vision backbone',
        'epochs': args.epochs, 'train_seed': args.train_seed, 'split_seed': args.split_seed, 'lr': args.lr, 'temperature': .07,
        'identity_regularization': .01, 'selection': 'Highest validation NDCG@5; ties keep earlier epoch; epoch zero is eligible',
        'best_epoch': best_epoch, 'history': history, 'training_seconds': time.monotonic() - started})
    np.save(args.out / 'transform.npy', best.numpy())
    encoder = Encoder(metadata['model'], metadata['revision'])
    weight = best @ encoder.model.visual_projection.weight.detach().cpu()
    np.save(args.out / 'projection.npy', weight.numpy())
    updated = normalize(vectors @ best.numpy().T)
    # Fold the transform into the actual CLIP projection and check cached-index equivalence.
    with torch.no_grad():
        encoder.model.visual_projection.weight.copy_(weight)
    probe_ids = train_ids[:2] + val_ids[:2]
    probes = encoder.encode([paths[i] for i in probe_ids])
    if not np.allclose(probes, updated[probe_ids], atol=2e-5, rtol=2e-4):
        raise ValueError('Transformed catalog disagrees with fine-tuned image encoder')
    tuned_metadata = dict(metadata, projection='projection.npy',
        projection_sha256=hashlib.sha256((args.out / 'projection.npy').read_bytes()).hexdigest())
    np.savez_compressed(args.out / 'catalog.npz', vectors=updated, metadata=json.dumps(tuned_metadata))
    del encoder
    if getattr(args, 'skip_evaluation', False):
        print('Checkpoint saved; evaluation will be performed by the seed study.', flush=True)
        return
    print('Checkpoint selected. Running final held-out evaluation for both models.', flush=True)
    baseline_summary = evaluate(args.dataset, args.index, args.out / 'baseline')
    tuned_summary = evaluate(args.dataset, args.out / 'catalog.npz', args.out / 'finetuned')
    comparison = {m: {'baseline': baseline_summary[m], 'finetuned': tuned_summary[m],
                      'delta': tuned_summary[m] - baseline_summary[m]} for m in METRICS}
    dump(args.out / 'comparison.json', comparison)
    a = json.loads((args.out / 'baseline/results.json').read_text())
    b = json.loads((args.out / 'finetuned/results.json').read_text())
    changes = [{ 'query_id': q['query_id'], 'true_label': q['true_label'],
                 **{m: r[m] - q[m] for m in METRICS}} for q, r in zip(a, b)]
    dump(args.out / 'per_query_changes.json', changes)
    lines = ['# Original vs fine-tuned CLIP', '', f'Selected epoch: {best_epoch}, using catalog validation NDCG@5 only.', '',
             '| Metric | Baseline | Fine-tuned | Delta |', '|---|---:|---:|---:|']
    lines += [f'| {m} | {v["baseline"]:.6f} | {v["finetuned"]:.6f} | {v["delta"]:+.6f} |' for m, v in comparison.items()]
    lines += ['', '## Protocol and limits', '',
        '- Both models search the same full catalog and use the same 60 query images and binary dish-label relevance.',
        '- Training uses catalog labels; validation searches the training subset, with near-duplicate groups kept together.',
        '- Held-out queries were previously inspected during project exploration, so this is not a pristine unseen test.',
        '- Duplicate detection is heuristic. Catalog candidates related to query images remain in both final galleries, but their groups are excluded from training/validation.',
        '- Only a constrained image projection is trained. The vision backbone and text encoder are not fine-tuned; text alignment is not evaluated.',
        '- Recall@5 uses the number of matching catalog images (100), so its maximum is 0.05.',
        '- Results are one seed on a small, known-category dataset; improvement is not guaranteed on new foods.', '']
    (args.out / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--index', type=Path, default=INDEX)
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/tuned')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--train-seed', '--seed', dest='train_seed', type=int, default=42, help='Training batch seed; independent of the split')
    p.add_argument('--split-seed', type=int, default=42, help='Fixed train/validation split seed')
    p.add_argument('--skip-evaluation', action='store_true', help='Save checkpoint only; for separately orchestrated evaluation')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
