"""Prepare missing assets, then verify both encoders and YOLO with a real query."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from image_search import DATASET, INDEX, ROOT, ImageSearch, build_index


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def validate_dataset(dataset):
    queries = []
    for csv_name, folder in [('catalog.csv', 'images'), ('queries.csv', 'queries')]:
        with (dataset / csv_name).open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'{csv_name} is empty')
        missing = [row['filename'] for row in rows if not (dataset / folder / row['filename']).is_file()]
        if missing:
            raise FileNotFoundError(f'Incomplete supplied dataset: {folder}/{missing[0]}. Restore the dataset before continuing.')
        if folder == 'queries':
            queries = rows
    return queries


def prepare(check_only=False):
    if not DATASET.exists() and not check_only:
        run('prepare_data.py', '--out', DATASET)
    queries = validate_dataset(DATASET)
    if not INDEX.exists() and not check_only:
        build_index()
    tuned = ROOT / 'outputs/tuned'
    if not (tuned / 'catalog.npz').exists() and not check_only:
        if tuned.exists():
            raise ValueError('outputs/tuned is incomplete. Restore its model files or move the incomplete folder before retrying.')
        run('finetune_clip.py', '--skip-evaluation')
    query = DATASET / 'queries' / queries[0]['filename']
    for name, index in [('baseline', INDEX), ('fine-tuned', tuned / 'catalog.npz')]:
        engine = ImageSearch(DATASET, index)
        results = engine.search_image(query, top_k=5)
        if len(results) != 5:
            raise ValueError(f'{name} did not return five results')
        print(f'OK: {name} loaded and returned five results.', flush=True)
        del engine
    from detect_food import FoodDetector
    from PIL import Image, ImageOps
    detector = FoodDetector()
    with Image.open(query) as image:
        regions = detector.detect(ImageOps.exif_transpose(image).convert('RGB'))
    print(f'OK: YOLO inference completed ({len(regions)} regions).', flush=True)
    print('Setup verified. Start the UI: uv run search_ui.py --open', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check existing artifacts without preparing data or training; missing pretrained weights may download')
    args = parser.parse_args()
    try:
        prepare(args.check)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Setup failed: {error}\n')


if __name__ == '__main__':
    main()
