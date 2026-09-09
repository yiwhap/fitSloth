import unittest
from compare_models import pair_results


def query(qid, matches):
    return {'query_id': qid, 'filename': qid + '.jpg', 'true_label': 'pho',
            'relevant_in_catalog': 100, 'results': [{'dish_label': label} for label in matches]}


class ComparisonTests(unittest.TestCase):
    def test_pairs_by_id_not_position_and_recomputes_metrics(self):
        good, bad = ['pho'] * 5, ['sushi'] * 5
        pairs = pair_results([query('a', bad), query('b', good)],
                             [query('b', bad), query('a', good)])
        self.assertEqual([p['query_id'] for p in pairs], ['a', 'b'])
        self.assertEqual([p['status'] for p in pairs], ['improved', 'worse'])
        self.assertEqual(pairs[0]['delta']['recall@5'], .05)

    def test_rejects_different_queries_or_ground_truth(self):
        a = query('a', ['pho'] * 5)
        with self.assertRaises(ValueError):
            pair_results([a], [query('b', ['pho'] * 5)])
        with self.assertRaises(ValueError):
            pair_results([a], [dict(a, true_label='sushi')])
        with self.assertRaises(ValueError):
            pair_results([a, a], [a])
