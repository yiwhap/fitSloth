"""Export three best and three failed CLIP retrieval examples from an evaluation."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from evaluate_search import metrics_at_5
from image_search import DATASET, ROOT


def select_cases(rows, all_failures=False, failure_queries=None):
    """Rank by label relevance, then NDCG; cosine only breaks best-case ties."""
    scored = []
    for row in rows:
        results = row['results'][:5]
        metrics = metrics_at_5(results, row['true_label'], row['relevant_in_catalog'])
        scored.append({**row, **metrics, 'results': results})
    best = sorted(scored, key=lambda q: (-q['precision@5'], -q['ndcg@5'],
                                         -q['results'][0]['score'], q['query_id']))[:3]
    failures = sorted((q for q in scored if q['results'][0]['dish_label'] != q['true_label']),
                      key=lambda q: (q['precision@5'], q['ndcg@5'], q['query_id']))
    if failure_queries is not None:
        if all_failures or len(failure_queries) != 3 or len(set(failure_queries)) != 3:
            raise ValueError('Choose three distinct failure queries, without --all-failures')
        by_id = {q['query_id']: q for q in failures}
        if any(qid not in by_id for qid in failure_queries):
            raise ValueError('Each selected query must exist and have a wrong top-1 label')
        failures = [by_id[qid] for qid in failure_queries]
    elif not all_failures:
        failures = failures[:3]
    if len(best) < 3 or (not all_failures and len(failures) < 3):
        raise ValueError('Need at least three queries and three top-1 label failures')
    return {'best': best, 'failure': failures}


def render_case(query, dataset, destination, category):
    width, height = 280, 390
    canvas = Image.new('RGB', (width * 6, height), '#f5f6f8')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    heading = ImageFont.load_default(size=22)
    title = (f"{category.upper()} | {query['query_id']} | "
             f"Correct: {query['true_label']} | P@5: {query['precision@5']:.0%}")
    draw.text((12, 12), title, font=heading, fill='#14213d')
    entries = [(dataset / 'queries' / query['filename'], 'QUERY', query['true_label'], None)]
    entries += [(dataset / 'images' / r['filename'], f"#{i} " +
                 ('MATCH' if r['dish_label'] == query['true_label'] else 'WRONG LABEL'),
                 r['dish_label'], r['score']) for i, r in enumerate(query['results'], 1)]
    for i, (path, status, label, score) in enumerate(entries):
        x = i * width
        with Image.open(path) as original:
            thumb = ImageOps.contain(ImageOps.exif_transpose(original).convert('RGB'), (264, 250))
        canvas.paste(thumb, (x + (width - thumb.width) // 2, 55 + (250 - thumb.height) // 2))
        color = '#aa2424' if status.endswith('WRONG LABEL') else '#176440'
        draw.text((x + 10, 316), status, font=font, fill=color)
        draw.text((x + 10, 341), label.replace('_', ' '), font=font, fill='#14213d')
        if score is not None:
            draw.text((x + 10, 365), f'Cosine: {score:.4f}', font=font, fill='#435065')
    canvas.save(destination)
    return canvas


def export_cases(results_path, dataset, out, all_failures=False, failure_queries=None):
    rows = json.loads(results_path.read_text())
    selected = select_cases(rows, all_failures, failure_queries)
    out.mkdir(parents=True, exist_ok=True)
    report = [f"# CLIP: three best and {len(selected['failure'])} failure cases", '',
              'Correctness means matching the supplied dish label. Failures have a wrong top-1 label.',
              'Best: descending Precision@5, then NDCG@5, then top-1 cosine. ' +
              ('Failures: explicitly selected query IDs, in the supplied order.' if failure_queries else
               'Failures: ascending Precision@5, then NDCG@5. Final ties use query ID.'),
              'Selections are global; multiple queries may share a dish. '
              'Cosine is not a probability. These selected extremes do not represent average performance.', '',
              f"Source: `{results_path.resolve()}` ({len(rows)} queries).", '']
    for category, cases in selected.items():
        strips = []
        for q in cases:
            filename = f"{category}_{q['query_id']}.png"
            strips.append(render_case(q, dataset, out / filename, category))
            labels = ', '.join(r['dish_label'] for r in q['results'])
            report += [f"## {category.title()}: {q['query_id']}", '',
                       f"Ground truth: **{q['true_label']}**. Precision@5: **{q['precision@5']:.0%}**; "
                       f"NDCG@5: **{q['ndcg@5']:.4f}**.", '', f'Top five labels: {labels}.', '',
                       f'![Query and top five results]({filename})', '']
        if not strips:
            continue
        sheet = Image.new('RGB', (strips[0].width, sum(s.height for s in strips)), 'white')
        for i, strip in enumerate(strips):
            sheet.paste(strip, (0, i * strip.height))
        sheet.save(out / f'{category}_cases.png')
    report += ['## Interpretation', '',
               'Wrong-label results can still look similar. Inspect composition, visible food, '
               'and label quality before proposing a cause; these images alone cannot prove what the model attends to.', '']
    (out / 'report.md').write_text('\n'.join(report))
    (out / 'cases.json').write_text(json.dumps(selected, indent=2) + '\n')
    cards = []
    for q in selected['failure']:
        name = html.escape(f"failure_{q['query_id']}.png", quote=True)
        label = html.escape(q['query_id'])
        cards.append(f'<figure><img src="{name}" alt="{label}" loading="lazy"></figure>')
    (out / 'failures.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>CLIP failures</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#f5f6f8}'
        'figure{margin:24px 0}img{width:100%;height:auto}</style>'
        f"<h1>{len(cards)} top-1 failures out of {len(rows)} queries</h1>"
        '<p>The left image is the query; the next five are ranked results. '
        'Correctness uses supplied dish labels. Red means wrong label, green means a match.</p>'
        + ''.join(cards))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs/eval/results.json')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/cases')
    parser.add_argument('--all-failures', action='store_true',
                        help='Export every wrong top-1 result instead of only three')
    parser.add_argument('--failure-queries', nargs=3, metavar='QUERY_ID',
                        help='Export three specified wrong-top-1 queries in this order')
    args = parser.parse_args()
    try:
        selected = export_cases(args.results, args.dataset, args.out, args.all_failures, args.failure_queries)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}. Run uv run evaluate_search.py first if results are missing.\n')
    for category, cases in selected.items():
        print(f"{category}: " + ', '.join(q['query_id'] for q in cases))
    print(f'Report: {args.out.resolve() / "report.md"}')


if __name__ == '__main__':
    main()
