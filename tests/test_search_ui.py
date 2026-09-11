import unittest
import errno
from contextlib import redirect_stderr
from io import StringIO
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile

import numpy as np
from PIL import Image

from search_ui import SearchApp, decode_image


class SearchUITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        (root/'images').mkdir()
        Image.new('RGB', (60, 40), 'orange').save(root/'images/food.png')
        buffer = BytesIO()
        Image.new('RGB', (60, 40), 'green').save(buffer, format='PNG')
        self.raw = buffer.getvalue()
        self.app = SearchApp()
        def encode(paths):
            return np.array([[0., 1.] if p.name.startswith('yolo') else [1., 0.] for p in paths])
        self.app.engines = {name: SimpleNamespace(
            encoder=SimpleNamespace(encode=encode), dataset=root,
            rows=[{'image_id':str(i), 'filename':'food.png','dish_label':'dish'} for i in range(5)],
            vectors=np.array([[1.,0.],[0.,1.],[.8,.6],[.6,.8],[.7,.7]]))
            for name in ['baseline','finetuned']}
        self.app.detector = Mock()
        self.app.detector.detect.return_value = [{'box':[0,0,30,30],'label':'bowl','score':.9}]

    def test_fused_results_preserve_different_winning_inputs_and_items(self):
        result = self.app.search(self.raw, 'both', 'full_yolo', 42)
        self.assertEqual(len(result['models']), 2)
        for model in result['models']:
            rows = model['rankings'][0]['results']
            self.assertEqual([r['source']['file'] for r in rows[:2]], ['full.png','yolo_0.png'])
            self.assertEqual(len(model['items']), 1)
            self.assertTrue(all('path' not in r and r['image'].startswith('data:image/jpeg') for r in rows))

    def test_random_keeps_five_independent_trials_and_seeded_boxes(self):
        a = self.app.search(self.raw, 'baseline', 'random', 42)
        b = self.app.search(self.raw, 'baseline', 'random', 42)
        self.assertEqual([v['box'] for v in a['views']], [v['box'] for v in b['views']])
        self.assertEqual(len(a['models'][0]['rankings']), 5)
        self.app.detector.detect.assert_not_called()
        for j, trial in enumerate(a['models'][0]['rankings']):
            self.assertTrue(all(r['source']['file']==f'random_{j}.png' for r in trial['results']))

    def test_yolo_without_detections_falls_back_to_full(self):
        self.app.detector.detect.return_value = []
        yolo = self.app.search(self.raw, 'baseline', 'yolo')
        full = self.app.search(self.raw, 'baseline', 'full')
        self.assertTrue(yolo['fallback'])
        self.assertEqual(yolo['models'][0]['rankings'], full['models'][0]['rankings'])

    def test_rejects_bad_uploads_and_options(self):
        with self.assertRaises(ValueError):decode_image(b'not an image')
        for model,method,seed in [('other','full',1),('baseline','other',1),('baseline','full',-1)]:
            with self.assertRaises(ValueError):self.app.search(self.raw,model,method,seed)


class StartupTests(unittest.TestCase):
    def test_occupied_port_has_actionable_message_without_traceback(self):
        from search_ui import main
        output = StringIO()
        with patch('sys.argv', ['search_ui.py']), patch('search_ui.SearchApp'), \
             patch('search_ui.ThreadingHTTPServer', side_effect=OSError(errno.EADDRINUSE, 'Address already in use')), \
             redirect_stderr(output), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn('http://127.0.0.1:8000', output.getvalue())
        self.assertIn('--port 8001', output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())
