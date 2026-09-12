"""Evaluate eight CLIP pipelines: full image, random crops, YOLO crops, and full + YOLO."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from detect_food import DETECTOR, FOOD_CLASSES, FoodDetector, annotate
from evaluate_search import METRICS, metrics_at_5
from image_search import DATASET, INDEX, ROOT, ImageSearch, normalize, save_preview


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding="utf-8")


def sample_boxes(size, count, rng):
    """Random area (25%-90%) and aspect ratio (3/4-4/3); random position."""
    w, h = size
    boxes = []
    for _ in range(count):
        for attempt in range(100):
            area = rng.uniform(.25, .9) * w * h
            ratio = np.exp(rng.uniform(np.log(.75), np.log(4/3)))
            cw, ch = round(np.sqrt(area * ratio)), round(np.sqrt(area / ratio))
            if 1 <= cw <= w and 1 <= ch <= h:
                break
        else:
            cw, ch = max(1, w//2), max(1, h//2)
        x, y = int(rng.integers(0, w-cw+1)), int(rng.integers(0, h-ch+1))
        boxes.append([x, y, x+cw, y+ch])
    return boxes


def ranked(search, scores, source_scores, source_views):
    """Keep the winning input per candidate; ties select the first supplied view."""
    source_scores = np.asarray(source_scores)
    if source_scores.shape != (len(source_views), len(scores)) or not source_views:
        raise ValueError('One score row is required per source view')
    if not np.allclose(source_scores.max(axis=0), scores):
        raise ValueError('Ranking scores must equal the maximum source cosine')
    winners = source_scores.argmax(axis=0)
    order = np.argsort(-scores, kind='stable')[:5]
    return [dict(rank=i+1, **search.rows[int(idx)], score=float(scores[idx]),
                 source=dict(source_views[int(winners[idx])]),
                 source_scores=[dict(view, score=float(row[idx]))
                                for view, row in zip(source_views, source_scores)],
                 path=str((search.dataset / 'images' / search.rows[int(idx)]['filename']).resolve()))
            for i, idx in enumerate(order)]


def ranking_html(results, qid, true_label, out):
    """Show the actual winning input alongside each retrieved image."""
    import os
    cards = []
    for result in results:
        source = result['source']
        src = html.escape(f'{qid}/{source["file"]}', quote=True)
        catalog = html.escape(os.path.relpath(result['path'], out), quote=True)
        relevant = result['dish_label'] == true_label
        scores = ''.join(f'<li>{html.escape(v["label"])}: {v["score"]:.6f}</li>'
                         for v in result['source_scores'])
        cards.append(f'<article class="match"><strong>Rank {result["rank"]} · cosine {result["score"]:.6f}</strong>'
                     f'<p>{"✓ Relevant (1)" if relevant else "✗ Not relevant (0)"}</p>'
                     f'<div class="pair"><figure><img loading="lazy" src="{src}">'
                     f'<figcaption>Selected input: {html.escape(source["label"])}</figcaption></figure>'
                     f'<figure><img loading="lazy" src="{catalog}">'
                     f'<figcaption>Catalog: {html.escape(result["dish_label"])}</figcaption></figure></div>'
                     f'<details><summary>Cosine from every input</summary><ul>{scores}</ul></details></article>')
    return '<div class="matches">' + ''.join(cards) + '</div>'


def fuse_scores(scores, full_scores=None):
    """Maximum cosine per candidate; optionally include the full-image score vector."""
    if full_scores is not None:
        scores = np.concatenate([np.asarray(full_scores)[None, :], scores], axis=0)
    if scores.ndim != 2 or len(scores) < 1:
        raise ValueError('Expected at least one region score vector')
    return scores.max(axis=0)


def prepare(args, queries):
    paths = [args.dataset / 'queries' / q['filename'] for q in queries]
    query_hash = hashlib.sha256((args.dataset / 'queries.csv').read_bytes())
    for p in paths:
        query_hash.update(p.read_bytes())
    manifest_path = args.out / 'manifest.json'
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest['query_hash'] != query_hash.hexdigest() or manifest['random_trials'] != args.random_trials
                or manifest['detector'] != DETECTOR or manifest.get('classes') != sorted(FOOD_CLASSES) or manifest['query_ids'] != [q['query_id'] for q in queries]):
            raise ValueError('Resume configuration/data changed; start a new output directory')
        seed = manifest['seed']
        if args.seed is not None and args.seed != seed:
            raise ValueError('Resume seed differs from saved seed')
    else:
        if args.out.exists():
            raise ValueError('Output exists: use --resume or choose a new --out')
        args.out.mkdir(parents=True)
        seed = args.seed if args.seed is not None else 42
        manifest = {'seed': seed, 'random_trials': args.random_trials, 'query_hash': query_hash.hexdigest(),
                    'query_ids': [q['query_id'] for q in queries], 'detector': DETECTOR, 'classes': sorted(FOOD_CLASSES),
                    'box_threshold': .25, 'inference_size': 640, 'nms_iou': .5,
                    'min_box_area_fraction': .01, 'random_area': [.25,.9], 'random_aspect': [.75,4/3],
                    'fusion': {'random': 'independent crop rankings; mean trial metrics', 'yolo': 'max across YOLO regions', 'full_yolo': 'max across full image and YOLO regions'},
                    'no_detection': 'full-image fallback, explicitly flagged', 'cases': {}}
        dump(manifest_path, manifest)
    detector = None
    for i, q in enumerate(queries):
        qid = q['query_id']
        if qid in manifest['cases']:
            continue
        if detector is None:
            detector = FoodDetector()
            manifest['detector_revision'] = detector.revision
        with Image.open(paths[i]) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        case_dir = args.out / qid
        case_dir.mkdir(exist_ok=True)
        image.save(case_dir / 'full.png')
        qseed = seed + int.from_bytes(hashlib.sha256(qid.encode()).digest()[:4], 'big')
        boxes = sample_boxes(image.size, args.random_trials, np.random.default_rng(qseed))
        for trial, box in enumerate(boxes):
            image.crop(box).save(case_dir / f'random_{trial}.png')
        start = time.monotonic()
        detections = detector.detect(image)
        elapsed = time.monotonic()-start
        annotate(image, detections).save(case_dir / 'detections.png')
        for j, detection in enumerate(detections):
            image.crop(detection['box']).save(case_dir / f'yolo_{j}.png')
        manifest['cases'][qid] = {'random_boxes': boxes, 'detections': detections,
                                  'fallback': not bool(detections), 'detection_seconds': elapsed}
        dump(manifest_path, manifest)
        print(f'Detected {i+1}/{len(queries)} {qid}: {len(detections)} regions ({elapsed:.1f}s)', flush=True)
    del detector
    manifest['fusion'] = {'random': 'independent crop rankings; mean trial metrics', 'yolo': 'max across YOLO regions', 'full_yolo': 'max across full image and YOLO regions'}
    dump(manifest_path, manifest)
    return manifest


def evaluate_models(args, queries, manifest, embedding_cache=None):
    baseline = ImageSearch(args.dataset, args.baseline)
    tuned = ImageSearch(args.dataset, args.finetuned)
    if baseline.metadata['fingerprint'] != tuned.metadata['fingerprint']:
        raise ValueError('Both models must search the same catalog')
    # Reuse frozen-backbone features only when equivalence is verified for these exact weights/indexes.
    transform = np.load(args.finetuned.parent / 'transform.npy', allow_pickle=False)
    original_weight = baseline.encoder.model.visual_projection.weight.detach().numpy()
    tuned_weight = tuned.encoder.model.visual_projection.weight.detach().numpy()
    reuse = (transform.shape == (512,512) and
             np.allclose(transform @ original_weight, tuned_weight, atol=2e-5, rtol=2e-4) and
             np.allclose(normalize(baseline.vectors @ transform.T), tuned.vectors, atol=2e-5, rtol=2e-4))
    cache_identity = (baseline.metadata['model'], baseline.metadata['revision'],
                      hashlib.sha256(original_weight.tobytes()).hexdigest())
    counts = Counter(r['dish_label'] for r in baseline.rows)
    all_records = []
    for i, q in enumerate(queries):
        qid = q['query_id']; folder = args.out / qid; case = manifest['cases'][qid]
        paths = [folder / 'full.png'] + [folder / f'random_{j}.png' for j in range(args.random_trials)]
        paths += [folder / f'yolo_{j}.png' for j in range(len(case['detections']))]
        cache_key = (cache_identity, tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)) if embedding_cache is not None else None
        embeddings = embedding_cache.get(cache_key) if embedding_cache is not None else None
        if embeddings is None:
            embeddings = np.concatenate([baseline.encoder.encode(paths[j:j+8]) for j in range(0,len(paths),8)])
            if embedding_cache is not None:
                embedding_cache[cache_key] = embeddings
        tuned_embeddings = normalize(embeddings @ transform.T) if reuse else np.concatenate([
            tuned.encoder.encode(paths[j:j+8]) for j in range(0,len(paths),8)])
        if reuse and i == 0:
            probe = tuned.encoder.encode([paths[0]])[0]
            if not np.allclose(probe, tuned_embeddings[0], atol=2e-5, rtol=2e-4):
                raise ValueError('Cached query transform does not match fine-tuned encoder')
        records = []
        for name, search, vectors in [('baseline',baseline,embeddings), ('finetune',tuned,tuned_embeddings)]:
            scores = vectors @ search.vectors.T
            full_views = [{'label': 'Full image', 'file': 'full.png'}]
            yolo_views = [{'label': f'YOLO crop {j+1} ({d["label"]})', 'file': f'yolo_{j}.png'}
                          for j, d in enumerate(case['detections'])]
            full = ranked(search, scores[0], scores[:1], full_views)
            random_scores = scores[1:1+args.random_trials]
            random_views = [{'label': f'Random crop {j+1}', 'file': f'random_{j}.png'}
                            for j in range(args.random_trials)]
            random = [ranked(search, random_scores[j], random_scores[j:j+1], random_views[j:j+1])
                      for j in range(args.random_trials)]
            yolo_scores = scores[1+args.random_trials:]
            fused = ranked(search, fuse_scores(yolo_scores), yolo_scores, yolo_views) if len(yolo_scores) else full
            full_fused = ranked(search, fuse_scores(yolo_scores, scores[0]),
                                np.concatenate([scores[:1], yolo_scores]), full_views + yolo_views)
            for method, result_sets in [('full',[full]),('random',random),('yolo',[fused]),('full_yolo',[full_fused])]:
                values = [metrics_at_5(rs, q['true_label'], counts[q['true_label']]) for rs in result_sets]
                record = {'query_id':qid,'true_label':q['true_label'],'model':name,'method':method,
                          **{m:float(np.mean([v[m] for v in values])) for m in METRICS},
                          'trials':values,'results':result_sets}
                if method in ('yolo', 'full_yolo'):
                    record['regions'] = [dict(region=j, detection=d, results=ranked(search, yolo_scores[j], yolo_scores[j:j+1], yolo_views[j:j+1]))
                                         for j,d in enumerate(case['detections'])]
                    record['fallback'] = case['fallback'] if method == 'yolo' else False
                    record['no_regions'] = case['fallback']
                    record['includes_full_image'] = method == 'full_yolo'
                    for region in record['regions'] if method == 'yolo' and getattr(args, 'write_previews', True) else []:
                        save_preview(folder / f'yolo_{region["region"]}.png', region['results'],
                                     folder / f'{name}_region_{region["region"]}.png')
                records.append(record)
                if getattr(args, 'write_previews', True):
                    save_preview(paths[1] if method == 'random' else paths[0], result_sets[0], folder / f'{name}_{method}.png')
        dump(folder / 'results.json',records)
        all_records.extend(records)
        print(f'Searched {i+1}/{len(queries)} {qid}',flush=True)
    dump(args.out / 'results.json',all_records)
    return all_records


def report(args, queries, manifest, records):
    summary = []
    titles = [('baseline','full','1. Baseline: full image'),
              ('baseline','random','2. Baseline: random crop'),
              ('baseline','yolo','3. Baseline: YOLO crops'),
              ('baseline','full_yolo','4. Baseline: full image + YOLO crops'),
              ('finetune','full','5. Fine-tune: full image'),
              ('finetune','random','6. Fine-tune: random crop'),
              ('finetune','yolo','7. Fine-tune: YOLO crops'),
              ('finetune','full_yolo','8. Fine-tune: full image + YOLO crops')]
    for model,method,title in titles:
        subset = [r for r in records if r['model']==model and r['method']==method]
        summary.append({'pipeline':title, **{m:float(np.mean([r[m] for r in subset])) for m in METRICS}})
    dump(args.out / 'summary.json',{'query_count':len(queries),'pipelines':summary,
        'seed':manifest['seed'],'random_trials':args.random_trials,
        'yolo_full_image_fallbacks':sum(c['fallback'] for c in manifest['cases'].values())})
    for filename, rows, fields in [('summary.csv',summary,['pipeline',*METRICS]),
        ('per_query.csv',records,['query_id','true_label','model','method',*METRICS])]:
        with (args.out / filename).open('w',newline='', encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    table='<table><tr><th>Pipeline</th>'+''.join(f'<th>{m}</th>' for m in METRICS)+'</tr>'
    for r in summary:
        table+=f'<tr><th>{r["pipeline"]}</th>'+''.join(f'<td>{r[m]:.4f}</td>' for m in METRICS)+'</tr>'
    table+='</table>'
    protocol = (f'{len(queries)} queries. Random seed: {manifest["seed"]}; {args.random_trials} random crops per query, '
        'with randomly sampled sizes and positions. The same crops are used by both models. Random crops each produce a separate Top-5 ranking; metrics average all crop trials equally. '
        'YOLO crops: maximum cosine across detected regions per catalog candidate. Full + YOLO: maximum cosine across the full image AND all detected regions. Both produce five unique catalog images without label-based view selection. Each result displays its winning input and all input cosine scores. Ties choose the first input (full image first in Full + YOLO, otherwise crop order). Relevance for metrics is determined by the catalog dish label, not the YOLO label. '
        'No detections: full-image fallback, counted below. Random uses five crops; YOLO crop counts vary and detection adds compute. '
        'The YOLO segmentation model supplies boxes and mask contours; retrieval uses unmasked box crops, preserving context. No text prompts or manual boxes are used. Only query images are cropped; both models search the same original catalog. Per-region results are not scored against the whole-photo label, which may describe only one item. '
        'There are no dedicated drink categories in this catalog. These queries were previously inspected; detection quality on fresh multi-item images is not established.')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eight-way CLIP comparison</title>
<style>body{font:16px system-ui;max-width:1700px;margin:24px auto;padding:0 20px;color:#14213d;background:#f0f2f5}header,section{padding:20px;background:white;margin:20px 0;border-radius:12px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%;height:auto}.regions img{max-height:220px}details{margin:16px 0}p{line-height:1.6}.matches{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.match{border:1px solid #ccd2dc;border-radius:8px;padding:12px}.pair{display:flex;gap:8px}.pair figure{margin:0;flex:1;min-width:0}.pair img{width:100%;height:150px;object-fit:contain}.pair figcaption{font-size:13px}</style></head><body><header><h1>Eight-way image search comparison</h1>'''
    doc+=table+f'<p>{html.escape(protocol)}</p><p>Full-image fallback: {sum(c["fallback"] for c in manifest["cases"].values())}/{len(queries)} queries.</p></header>'
    for q in queries:
        qid=q['query_id'];case=manifest['cases'][qid]
        doc+=f'<section><h2>{html.escape(qid)} · {html.escape(q["true_label"])}</h2><details><summary>Automatic detections and all random inputs</summary><div class="regions">'
        doc+=f'<img loading="lazy" src="{qid}/detections.png" alt="Automatic detections">'
        for j in range(args.random_trials):doc+=f'<img loading="lazy" src="{qid}/random_{j}.png" alt="Random trial {j}">'
        doc+='</div></details>'
        for model,method,title in titles:
            r=next(r for r in records if r['query_id']==qid and r['model']==model and r['method']==method)
            doc+=f'<details><summary>{title} · '+ ' · '.join(f'{m}: {r[m]:.4f}' for m in METRICS)+'</summary>'
            if method=='random':doc+='<p>Five separate crop rankings. Metrics average all five trials equally; no best crop is selected.</p>'
            if method=='full_yolo':doc+='<p>Maximum cosine across full image + YOLO crops. The full image is always included.</p>'
            if method=='yolo':doc+='<p>Combined ranking shown first; separate item searches follow. '+('No region detected; used full image.' if case['fallback'] else '')+'</p>'
            for trial, results in enumerate(r['results']):
                if method == 'random':
                    doc+=f'<details><summary>Random trial {trial+1} · '+ ' · '.join(f'{m}: {r["trials"][trial][m]:.4f}' for m in METRICS)+'</summary>'
                doc+=ranking_html(results, qid, q['true_label'], args.out)
                if method == 'random':
                    doc+='</details>' 
            if method in ('yolo','full_yolo'):
                for region in r['regions']:
                    doc+=f'<p>Region {region["region"]+1}: {html.escape(region["detection"]["label"])}</p><img loading="lazy" src="{qid}/{model}_region_{region["region"]}.png">'
            doc+='</details>'
        doc+='</section>'
    (args.out/'index.html').write_text(doc+'</body></html>', encoding="utf-8")
    (args.out/'report.md').write_text('# Eight-way comparison\n\n'+protocol+'\n\n| Pipeline | Precision@5 | Recall@5 | NDCG@5 |\n|---|---:|---:|---:|\n'+
        '\n'.join('| '+r['pipeline']+' | '+' | '.join(f'{r[m]:.6f}' for m in METRICS)+' |' for r in summary)+'\n', encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--baseline',type=Path,default=INDEX)
    p.add_argument('--finetuned',type=Path,default=ROOT/'outputs/tuned/catalog.npz')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/compare')
    p.add_argument('--random-trials',type=int,default=5)
    p.add_argument('--seed',type=int,help='Crop seed for new runs (default: 42); resume keeps the saved seed')
    p.add_argument('--resume',action='store_true',help='Reuse completed detections and crops in this output directory')
    args=p.parse_args()
    if args.random_trials<1:p.error('--random-trials must be positive')
    with (args.dataset/'queries.csv').open(encoding="utf-8") as f:queries=list(csv.DictReader(f))
    manifest=prepare(args,queries)
    records=evaluate_models(args,queries,manifest)
    report(args,queries,manifest,records)


if __name__=='__main__':main()
