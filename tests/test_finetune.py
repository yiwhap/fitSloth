import unittest
import numpy as np
import torch

from finetune_clip import split_catalog, supervised_contrastive, retrieval_metrics
from food_search import normalize


class FineTuneTests(unittest.TestCase):
    def test_groups_and_query_overlap_never_leak(self):
        rows = [{'dish_label': label} for label in ['a'] * 8 + ['b'] * 8]
        groups = [0, 0, 2, 3, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13, 14, 15]
        train, validation, excluded = split_catalog(rows, groups, {7}, 42)
        self.assertIn(7, excluded)
        self.assertFalse(set(train) & set(validation))
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in validation})
        self.assertEqual(sorted(train + validation + excluded), list(range(16)))
        self.assertEqual((train, validation, excluded), split_catalog(rows, groups, {7}, 42))

    def test_contrastive_loss_prefers_correct_clusters(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1., 0.], [1., .1], [0., 1.], [.1, 1.]], requires_grad=True)
        bad = good.detach()[[0, 2, 1, 3]]
        loss = supervised_contrastive(good, labels)
        self.assertLess(loss.item(), supervised_contrastive(bad, labels).item())
        loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_folded_projection_matches_normalized_feature_transform(self):
        rng = np.random.default_rng(1)
        h, w, a = rng.normal(size=(4, 7)), rng.normal(size=(5, 7)), rng.normal(size=(5, 5))
        np.testing.assert_allclose(normalize(normalize(h @ w.T) @ a.T),
                                   normalize(h @ (a @ w).T), atol=1e-6)

    def test_validation_gallery_excludes_self(self):
        vectors = np.array([[1., 0.]] * 5 + [[0., 1.]] * 5 + [[1., 0.]])
        labels = np.array([0] * 5 + [1] * 5 + [0])
        metrics = retrieval_metrics(vectors, labels, list(range(10)), [10])
        self.assertEqual(metrics, {'precision@5': 1., 'recall@5': 1., 'ndcg@5': 1.})
