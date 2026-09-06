import math
import unittest
from evaluate import metrics_at_5


def results(labels):
    return [{"dish_label": label} for label in labels]


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(metrics_at_5(results(["pho"] * 5), "pho", 100),
                         {"precision@5": 1.0, "recall@5": 0.05, "ndcg@5": 1.0})

    def test_no_matches(self):
        self.assertTrue(all(v == 0 for v in
                            metrics_at_5(results(["pizza"] * 5), "pho", 100).values()))

    def test_worked_example(self):
        actual = metrics_at_5(results(["pho", "miso", "pho", "pho", "pizza"]), "pho", 100)
        self.assertEqual(actual["precision@5"], 0.6)
        self.assertEqual(actual["recall@5"], 0.03)
        ideal = 1 + 1 / math.log2(3) + 0.5 + 1 / math.log2(5) + 1 / math.log2(6)
        self.assertAlmostEqual(actual["ndcg@5"], (1 + 0.5 + 1 / math.log2(5)) / ideal)

    def test_rank_matters_only_for_ndcg(self):
        early = metrics_at_5(results(["pho"] + ["pizza"] * 4), "pho", 100)
        late = metrics_at_5(results(["pizza"] * 4 + ["pho"]), "pho", 100)
        self.assertEqual(early["precision@5"], late["precision@5"])
        self.assertEqual(early["recall@5"], late["recall@5"])
        self.assertGreater(early["ndcg@5"], late["ndcg@5"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            metrics_at_5(results(["pho"] * 4), "pho", 100)
        with self.assertRaises(ValueError):
            metrics_at_5(results(["pho"] * 5), "pho", 0)
