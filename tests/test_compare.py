import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from compare_pipelines import sample_boxes, fuse_scores, ranked, ranking_html
from detect_food import filter_boxes


class ComparisonTests(unittest.TestCase):
    def test_random_crop_changes_size_and_position_and_is_reproducible(self):
        boxes = sample_boxes((384, 512), 20, np.random.default_rng(42))
        self.assertEqual(boxes, sample_boxes((384, 512), 20, np.random.default_rng(42)))
        self.assertNotEqual(boxes, sample_boxes((384, 512), 20, np.random.default_rng(43)))
        self.assertGreater(len({(r-l,b-t) for l,t,r,b in boxes}), 1)
        self.assertGreater(len({(l,t) for l,t,r,b in boxes}), 1)
        for l,t,r,b in boxes:
            self.assertTrue(0<=l<r<=384 and 0<=t<b<=512)

    def test_detection_clamping_and_duplicate_removal(self):
        boxes = [[-4,-3,50,50],[0,0,49,49],[70,70,90,90],[5,5,5,10],[0,0,1,1]]
        kept = filter_boxes(boxes,[.9,.8,.7,.9,.9],['food']*5,(100,100))
        self.assertEqual([d['box'] for d in kept],[[0,0,50,50],[70,70,90,90]])

    def test_segmentation_polygons_stay_aligned_after_filtering(self):
        polygons = [[[1,1],[2,1],[1,2]], [[71,71],[72,71],[71,72]]]
        kept = filter_boxes([[0,0,50,50],[70,70,90,90]], [.1,.9],
                            ['bowl','cup'], (100,100), polygons=polygons)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['label'], 'cup')
        self.assertEqual(kept[0]['polygon'], polygons[1])

    def test_fusion_keeps_one_score_per_catalog_image(self):
        scores = np.array([[.9,.2,.8],[.7,.95,.5]])
        np.testing.assert_allclose(fuse_scores(scores), [.9,.95,.8])
        with self.assertRaises(ValueError):
            fuse_scores(np.empty((0,3)))

    def test_full_image_is_included_when_crops_miss_target(self):
        full = np.array([.95, .3, .2])
        crops = np.array([[.1, .8, .4], [.2, .6, .7]])
        combined = fuse_scores(crops, full)
        np.testing.assert_allclose(combined, [.95, .8, .7])
        self.assertEqual(int(combined.argmax()), 0)

    def test_full_plus_yolo_without_detections_equals_full(self):
        full = np.array([.7, .8, .1])
        np.testing.assert_allclose(fuse_scores(np.empty((0,3)), full), full)


class ProvenanceTests(unittest.TestCase):
    def test_each_candidate_selects_its_own_source_and_full_wins_tie(self):
        search = SimpleNamespace(dataset=Path('/tmp'), rows=[
            {'filename': f'{i}.jpg', 'dish_label': 'soup'} for i in range(3)])
        views = [{'label': 'Full image', 'file': 'full.png'},
                 {'label': 'YOLO crop 1', 'file': 'yolo_0.png'}]
        scores = np.array([[.9, .2, .7], [.3, .95, .7]])
        results = ranked(search, scores.max(axis=0), scores, views)
        self.assertEqual([r['source']['file'] for r in results],
                         ['yolo_0.png', 'full.png', 'full.png'])
        for r in results:
            self.assertEqual(r['score'], max(v['score'] for v in r['source_scores']))
        doc = ranking_html(results, 'q01', 'soup', Path('/tmp/report'))
        self.assertIn('q01/yolo_0.png', doc)
        self.assertIn('Selected input: Full image', doc)
        self.assertEqual(doc.count('✓ Relevant (1)'), 3)

    def test_single_input_fallback_and_invalid_provenance(self):
        search = SimpleNamespace(dataset=Path('/tmp'), rows=[{'filename':'a.jpg','dish_label':'soup'}])
        views = [{'label': 'Full image', 'file': 'full.png'}]
        results = ranked(search, np.array([.8]), np.array([[.8]]), views)
        self.assertEqual(results[0]['source'], views[0])
        with self.assertRaises(ValueError):
            ranked(search, np.array([.9]), np.array([[.8]]), views)
