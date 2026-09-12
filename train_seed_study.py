"""Train five projection seeds on one fixed split and report mean ± sample SD."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare_pipelines import prepare, evaluate_models, report, dump
from evaluate_search import METRICS
from finetune_clip import train
from image_search import DATASET, INDEX, ROOT


def aggregate(runs):
    if len(runs) < 2 or len({r['train_seed'] for r in runs}) != len(runs):
        raise ValueError('Need at least two distinct training seeds')
    names = [p['pipeline'] for p in runs[0]['pipelines']]
    if any([p['pipeline'] for p in r['pipelines']] != names for r in runs):
        raise ValueError('Runs must contain identical pipelines in identical order')
    return [dict(pipeline=name, **{m: {
        'mean': float(np.mean([r['pipelines'][i][m] for r in runs])),
        'sd': float(np.std([r['pipelines'][i][m] for r in runs], ddof=1))
    } for m in METRICS}) for i, name in enumerate(names)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--index', type=Path, default=INDEX)
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/seed_study')
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--crop-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--evaluate-only', action='store_true', help='Re-evaluate existing checkpoints and saved crops without retraining')
    args = p.parse_args()
    if args.out.exists() and not args.evaluate_only:
        p.error('Output already exists; choose an unused --out')
    if args.evaluate_only:
        saved = json.loads((args.out / 'protocol.json').read_text(encoding="utf-8"))
        if any(saved[k] != v for k, v in [('train_seeds', list(range(5))), ('split_seed', args.split_seed), ('crop_seed', args.crop_seed), ('epochs', args.epochs)]):
            p.error('Existing study protocol does not match arguments')
    args.out.mkdir(parents=True, exist_ok=args.evaluate_only)
    seeds = list(range(5))
    dump(args.out / 'protocol.json', dict(train_seeds=seeds, split_seed=args.split_seed,
         crop_seed=args.crop_seed, epochs=args.epochs, representative_seed=0,
         sd='Sample SD (ddof=1) across five run-level query means; fixed split and crops'))
    reference = None
    for seed in seeds:
        out = args.out / f'seed_{seed}'
        if args.evaluate_only:
            config = json.loads((out / 'training.json').read_text(encoding="utf-8"))
            if config['train_seed'] != seed or config['split_seed'] != args.split_seed:
                raise ValueError('Checkpoint seed does not match study')
        else:
            print(f'=== Training seed {seed} ===', flush=True)
            train(SimpleNamespace(dataset=args.dataset, index=args.index, out=out,
                  epochs=args.epochs, lr=.001, train_seed=seed, split_seed=args.split_seed,
                  skip_evaluation=True))
        split = json.loads((out / 'split.json').read_text(encoding="utf-8"))
        if reference is not None and split != reference:
            raise ValueError('Data split changed across training seeds')
        reference = split
    with (args.dataset / 'queries.csv').open(encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    comparison = SimpleNamespace(dataset=args.dataset, baseline=args.index,
        out=args.out / 'comparison', seed=args.crop_seed, random_trials=5, resume=args.evaluate_only)
    manifest = prepare(comparison, queries)
    runs = []
    embedding_cache = {}
    # Keep one set of crop images. The final visual report represents seed 0,
    # selected in advance, never chosen from evaluation scores.
    for seed in reversed(seeds):
        print(f'=== Evaluating seed {seed} ===', flush=True)
        out = args.out / f'seed_{seed}'
        comparison.finetuned = out / 'catalog.npz'
        comparison.write_previews = seed == 0
        records = evaluate_models(comparison, queries, manifest, embedding_cache=embedding_cache)
        report(comparison, queries, manifest, records)
        summary = json.loads((comparison.out / 'summary.json').read_text(encoding="utf-8"))
        run = dict(train_seed=seed, best_epoch=json.loads((out / 'training.json').read_text(encoding="utf-8"))['best_epoch'],
                   pipelines=summary['pipelines'])
        dump(out / 'evaluation.json', run)
        dump(out / 'query_results.json', records)
        runs.append(run)
    runs.sort(key=lambda r: r['train_seed'])
    summary = aggregate(runs)
    dump(args.out / 'summary.json', dict(train_seeds=seeds, split_seed=args.split_seed,
         crop_seed=args.crop_seed, query_count=len(queries), sd_ddof=1, pipelines=summary, runs=runs))
    with (args.out / 'summary.csv').open('w', newline='', encoding="utf-8") as f:
        fields = ['pipeline'] + [f'{m}_{stat}' for m in METRICS for stat in ('mean', 'sd')]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(dict(pipeline=row['pipeline'], **{f'{m}_{s}': row[m][s] for m in METRICS for s in ('mean', 'sd')}))
    lines = ['# Five training seeds', '',
        f'Train seeds: {seeds}. Fixed split seed: {args.split_seed}; crop seed: {args.crop_seed}.', '',
        'Mean ± sample SD across five run-level means over all 60 queries. Random mode averages five separate crop-trial metrics per query. Baseline is unchanged across runs; its SD is zero. This measures training variability, not split/crop uncertainty or a confidence interval.', '',
        '| Pipeline | Precision@5 | Recall@5 | NDCG@5 |', '|---|---:|---:|---:|']
    lines += ['| ' + r['pipeline'] + ' | ' + ' | '.join(f"{r[m]['mean']:.6f} ± {r[m]['sd']:.6f}" for m in METRICS) + ' |' for r in summary]
    lines += ['', '[Visual comparison: preselected seed 0](comparison/index.html). Other seeds retain per-query rankings in their query_results.json.', '',
              'Selected epochs: ' + ', '.join(f"seed {r['train_seed']}: {r['best_epoch']}" for r in runs) + '.', '']
    (args.out / 'report.md').write_text('\n'.join(lines), encoding="utf-8")
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
