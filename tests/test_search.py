import unittest
import numpy as np
from food_search import normalize, rank


class RankingTests(unittest.TestCase):
    def test_cosine_ranking_and_scores(self):
        vectors = normalize([[0, 3], [8, 0], [1, 1], [-1, 0]])
        ids, scores = rank(vectors, np.array([5, 0]), 3)
        self.assertEqual(ids.tolist(), [1, 2, 0])
        np.testing.assert_allclose(scores, [1, 1 / np.sqrt(2), 0], atol=1e-6)

    def test_ties_and_large_k(self):
        ids, _ = rank(normalize([[1, 0], [1, 0]]), np.array([1, 0]), 5)
        self.assertEqual(ids.tolist(), [0, 1])

    def test_invalid_vectors_and_k(self):
        for vector in ([[0, 0]], [[float("nan"), 1]]):
            with self.assertRaises(ValueError):
                normalize(vector)
        with self.assertRaises(ValueError):
            rank(np.eye(2), np.array([1, 0]), 0)
