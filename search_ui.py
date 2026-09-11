"""Local upload UI for full-image and crop-based CLIP retrieval. Run: uv run search_ui.py."""
import argparse
import base64
import csv
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import secrets
import tempfile
import threading
import time
import webbrowser
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from compare_pipelines import fuse_scores, ranked, sample_boxes
from detect_food import FoodDetector, annotate
from image_search import DATASET, INDEX, ROOT, ImageSearch

MAX_UPLOAD = 16 * 1024 * 1024
MODELS = {'baseline': INDEX, 'finetuned': ROOT / 'outputs/tuned/catalog.npz'}
METHODS = {'full', 'random', 'yolo', 'full_yolo'}


def thumbnail(image):
    image = image.copy()
    image.thumbnail((420, 320))
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


def decode_image(raw):
    if not raw or len(raw) > MAX_UPLOAD:
        raise ValueError('Choose an image smaller than 16 MB.')
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError('Choose an image with at most 20 million pixels.')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Cannot read this image. Try a JPEG, PNG, or WebP file.') from exc


class SearchApp:
    def __init__(self):
        self.engines = {}
        self.detector = None
        # Serialize model access and keep loaded models reusable between requests.
        self.lock = threading.Lock()
        with (DATASET / 'queries.csv').open() as stream:
            self.examples = {r['filename']: r for r in csv.DictReader(stream)}

    def search(self, raw, model, method, seed=None):
        if model not in {*MODELS, 'both'} or method not in METHODS:
            raise ValueError('Unknown model or search method.')
        image = decode_image(raw)
        seed = secrets.randbits(32) if seed is None else int(seed)
        if not 0 <= seed < 2**32:
            raise ValueError('Seed must be between 0 and 4294967295.')
        with self.lock:
            return self._search(image, model, method, seed)

    def _search(self, image, model, method, seed):
        start = time.monotonic()
        detections = []
        views = [{'label': 'Full image', 'file': 'full.png', 'image': image}]
        if method == 'random':
            views = [{'label': f'Random trial {j+1}', 'file': f'random_{j}.png',
                      'box': box, 'image': image.crop(box)}
                     for j, box in enumerate(sample_boxes(image.size, 5, np.random.default_rng(seed)))]
        elif method in {'yolo', 'full_yolo'}:
            if self.detector is None:
                self.detector = FoodDetector()
            detections = self.detector.detect(image)
            crops = [{'label': f'YOLO crop {j+1} ({d["label"]})', 'file': f'yolo_{j}.png',
                      'box': d['box'], 'image': image.crop(d['box'])}
                     for j, d in enumerate(detections)]
            if method == 'full_yolo':
                views += crops
            elif crops:
                views = crops
        fallback = method == 'yolo' and not detections
        descriptors = [{k: v for k, v in view.items() if k != 'image'} for view in views]
        models = ['baseline', 'finetuned'] if model == 'both' else [model]
        output = []
        with tempfile.TemporaryDirectory(prefix='fitsloth-ui-') as temporary:
            paths = []
            for view in views:
                path = Path(temporary) / view['file']
                view['image'].save(path)
                paths.append(path)
            for name in models:
                if name not in self.engines:
                    if not MODELS[name].is_file():
                        command = 'uv run image_search.py build' if name == 'baseline' else 'uv run finetune_clip.py'
                        raise ValueError(f'Missing {name} index. Run: {command}')
                    self.engines[name] = ImageSearch(DATASET, MODELS[name])
                engine = self.engines[name]
                embeddings = np.concatenate([engine.encoder.encode(paths[j:j+8]) for j in range(0, len(paths), 8)])
                scores = embeddings @ engine.vectors.T
                groups = [(view['label'], scores[j:j+1], descriptors[j:j+1])
                          for j, view in enumerate(views)] if method == 'random' else [('Top 5', scores, descriptors)]
                rankings = []
                for title, matrix, sources in groups:
                    results = ranked(engine, fuse_scores(matrix), matrix, sources)
                    for result in results:
                        with Image.open(result.pop('path')) as catalog_image:
                            result['image'] = thumbnail(ImageOps.exif_transpose(catalog_image).convert('RGB'))
                    rankings.append({'title': title, 'results': results})
                # Independent item rankings let users inspect each detected dish.
                items = []
                if method in {'yolo', 'full_yolo'} and detections:
                    for j, view in enumerate(views):
                        if view['file'] == 'full.png':
                            continue
                        results = ranked(engine, scores[j], scores[j:j+1], descriptors[j:j+1])
                        for result in results:
                            with Image.open(result.pop('path')) as catalog_image:
                                result['image'] = thumbnail(ImageOps.exif_transpose(catalog_image).convert('RGB'))
                        items.append({'title': view['label'], 'results': results})
                output.append({'model': name, 'rankings': rankings, 'items': items})
        return {'models': output, 'method': method, 'seed': seed, 'fallback': fallback,
                'detection_count': len(detections), 'seconds': round(time.monotonic()-start, 2),
                'query': thumbnail(image),
                'detections': thumbnail(annotate(image, detections)) if detections else None,
                'views': [dict(descriptor, image=thumbnail(view['image']))
                          for descriptor, view in zip(descriptors, views)]}


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json'):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                return self.send(200, (ROOT/'ui/index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/examples':
                names = ['q_pho_00.jpg', 'q_sushi_00.jpg', 'q_miso_soup_00.jpg', 'q_hot_and_sour_soup_02.jpg']
                return self.send(200, {'examples': [{'filename': n, 'label': app.examples[n]['true_label']}
                                                  for n in names if n in app.examples]})
            if path.startswith('/example/') and path[9:] in app.examples:
                return self.send(200, (DATASET/'queries'/path[9:]).read_bytes(), 'image/jpeg')
            self.send(404, {'error': 'Not found.'})

        def do_POST(self):
            if urlsplit(self.path).path != '/api/search':
                return self.send(404, {'error': 'Not found.'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.send(403, {'error': 'Use the local app to upload images.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_UPLOAD:
                    return self.send(413, {'error': 'Choose an image smaller than 16 MB.'})
                options = parse_qs(urlsplit(self.path).query)
                result = app.search(self.rfile.read(size), options.get('model', ['finetuned'])[0],
                                    options.get('method', ['full_yolo'])[0], options.get('seed', [None])[0])
                self.send(200, result)
            except ValueError as exc:
                self.send(400, {'error': str(exc)})
            except Exception as exc:
                print(f'Search failed: {exc}', flush=True)
                self.send(500, {'error': 'Search failed. Check the terminal for model or index errors.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--open', action='store_true', dest='open_browser',
                        help='Open the UI in your default browser after starting')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(SearchApp()))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        parser.exit(1, f'Port {args.port} is already in use.\n'
                    f'If FitSloth is already running, open http://127.0.0.1:{args.port}\n'
                    'Otherwise choose another port: uv run search_ui.py --port 8001\n')
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'\nFitSloth is running\n\n{url}\n\n'
          f'HTML: {ROOT / "ui/index.html"}\n'
          'Save HTML edits, then refresh the browser. Ctrl+C to stop.\n', flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
