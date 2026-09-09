"""Create a local visual comparison of original and fine-tuned CLIP evaluations."""
import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image

from evaluate import METRICS, metrics_at_5
from food_search import DATASET, ROOT
from show_cases import render_case


def pair_results(baseline, finetuned):
    def keyed(rows):
        mapped = {q['query_id']: q for q in rows}
        if not rows or len(mapped) != len(rows):
            raise ValueError('Evaluation must contain unique, nonempty query IDs')
        return mapped
    a, b = keyed(baseline), keyed(finetuned)
    if a.keys() != b.keys():
        raise ValueError('Both evaluations must contain exactly the same queries')
    pairs = []
    for qid in sorted(a):
        left, right = a[qid], b[qid]
        for field in ('filename', 'true_label', 'relevant_in_catalog'):
            if left[field] != right[field]:
                raise ValueError(f'Mismatched {field} for {qid}')
        measured = []
        for q in (left, right):
            measured.append({**q, **metrics_at_5(q['results'], q['true_label'], q['relevant_in_catalog'])})
        left, right = measured
        delta = {m: right[m] - left[m] for m in METRICS}
        status = 'improved' if delta['ndcg@5'] > 1e-9 else 'worse' if delta['ndcg@5'] < -1e-9 else 'unchanged'
        pairs.append({'query_id': qid, 'baseline': left, 'finetuned': right, 'delta': delta, 'status': status})
    return sorted(pairs, key=lambda p: (-p['delta']['ndcg@5'], p['query_id']))


def export(baseline_dir, finetuned_dir, dataset, out):
    summaries = [json.loads((folder / 'summary.json').read_text()) for folder in (baseline_dir, finetuned_dir)]
    for field in ('catalog_fingerprint', 'catalog_count', 'k'):
        if summaries[0].get(field) is None or summaries[0][field] != summaries[1].get(field):
            raise ValueError(f'Evaluations must share {field}; rerun evaluate.py for both indexes')
    pairs = pair_results(*[json.loads((folder / 'results.json').read_text()) for folder in (baseline_dir, finetuned_dir)])
    out.mkdir(parents=True, exist_ok=True)
    means = {name: {m: sum(p[name][m] for p in pairs) / len(pairs) for m in METRICS}
             for name in ('baseline', 'finetuned')}
    table = '<table><thead><tr><th>Metric</th><th>Baseline</th><th>Fine-tuned</th><th>Delta</th></tr></thead><tbody>'
    for m in METRICS:
        a, b = means['baseline'][m], means['finetuned'][m]
        table += f'<tr><th>{m}</th><td>{a:.4f}</td><td>{b:.4f}</td><td>{b-a:+.4f}</td></tr>'
    table += '</tbody></table>'
    cards, csv_rows = [], []
    for i, pair in enumerate(pairs):
        strips = []
        for name in ('baseline', 'finetuned'):
            strips.append(render_case(pair[name], dataset, out / f'{i:02d}_{name}.png', name))
        combined = Image.new('RGB', (strips[0].width, sum(s.height for s in strips)), 'white')
        combined.paste(strips[0], (0, 0))
        combined.paste(strips[1], (0, strips[0].height))
        filename = f'{i:02d}_comparison.png'
        combined.save(out / filename)
        qid = html.escape(pair['query_id'])
        label = html.escape(pair['baseline']['true_label'])
        metric_rows = ''.join(f'<tr><th>{m}</th><td>{pair["baseline"][m]:.4f}</td>'
                              f'<td>{pair["finetuned"][m]:.4f}</td><td>{pair["delta"][m]:+.4f}</td></tr>' for m in METRICS)
        cards.append(f'<article data-status="{pair["status"]}" data-label="{label}">'
            f'<h2>{qid} <span class="{pair["status"]}">{pair["status"]}</span></h2>'
            f'<p>Ground truth: {label}</p><table><thead><tr><th>Metric</th><th>Baseline</th><th>Fine-tuned</th><th>Delta</th></tr></thead>'
            f'<tbody>{metric_rows}</tbody></table><a href="{filename}"><img src="{filename}" '
            f'alt="Baseline above fine-tuned results for {qid}" loading="lazy"></a></article>')
        csv_rows.append({'query_id': pair['query_id'], 'true_label': pair['baseline']['true_label'], 'status': pair['status'],
                         **{f'{name}_{m}': pair[name][m] for name in ('baseline', 'finetuned', 'delta') for m in METRICS}})
    labels = sorted({p['baseline']['true_label'] for p in pairs})
    options = ''.join(f'<option>{html.escape(label)}</option>' for label in labels)
    counts = {status: sum(p['status'] == status for p in pairs) for status in ('improved', 'worse', 'unchanged')}
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP model comparison</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1700px;color:#14213d;background:#eef1f5}
h1{margin-bottom:8px}article,header{background:white;padding:20px;margin-bottom:24px;border-radius:12px}
img{width:100%;height:auto}table{border-collapse:collapse;margin:16px 0}td,th{text-align:left;padding:8px 20px;border-bottom:1px solid #ddd}
select{padding:10px;margin:8px}span{font-size:14px;padding:6px;border-radius:4px}.improved{background:#d8f3e4}.worse{background:#ffe1df}.unchanged{background:#eee}
[hidden]{display:none}a{color:inherit}</style></head><body><header><h1>Original CLIP vs fine-tuned projection</h1>'''
    document += f'<p>{len(pairs)} paired queries • {counts["improved"]} improved • {counts["worse"]} worse • {counts["unchanged"]} unchanged by NDCG@5</p>' + table
    document += '''<p>Each image panel shows the query on the left and five ranked matches. Baseline is above fine-tuned.
Green means a matching dish label; red means a different label. Cosine is not a probability.</p>
<p>Recall@5 divides by all relevant catalog images (100 per dish here; maximum 0.05).
These queries were previously inspected. Results are one training seed, not proof of generalization.</p>
<label>Show <select id="status"><option value="all">All queries</option><option value="improved">Improved</option><option value="worse">Worse</option><option value="unchanged">Unchanged</option></select></label>
<label>Dish <select id="label"><option value="all">All dishes</option>'''
    document += options + '</select></label><p id="count" aria-live="polite"></p></header>' + ''.join(cards)
    document += '''<script>
const statusFilter=document.getElementById('status'), labelFilter=document.getElementById('label');
function filter(){let n=0;document.querySelectorAll('article').forEach(card=>{
card.hidden=!(statusFilter.value==='all'||card.dataset.status===statusFilter.value)||!(labelFilter.value==='all'||card.dataset.label===labelFilter.value);
if(!card.hidden)n++;});document.getElementById('count').textContent=`Showing ${n} queries`;}
statusFilter.addEventListener('change',filter);labelFilter.addEventListener('change',filter);filter();
</script></body></html>'''
    (out / 'index.html').write_text(document)
    with (out / 'per_query.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (out / 'summary.json').write_text(json.dumps({'means': means, 'counts_by_ndcg': counts,
        'baseline_source': str(baseline_dir.resolve()), 'finetuned_source': str(finetuned_dir.resolve())}, indent=2) + '\n')
    print(json.dumps(means, indent=2))
    print(f'Open: {out.resolve() / "index.html"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT / 'artifacts/finetuned_clip/baseline')
    p.add_argument('--finetuned', type=Path, default=ROOT / 'artifacts/finetuned_clip/finetuned')
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/model_comparison')
    args = p.parse_args()
    try:
        export(args.baseline, args.finetuned, args.dataset, args.out)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
