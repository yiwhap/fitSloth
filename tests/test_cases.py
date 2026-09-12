import unittest
from show_search_cases import select_cases


class CaseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(query_id=qid, true_label='food', relevant_in_catalog=100,
                         results=[dict(dish_label='other', score=.9)] * 5)
                     for qid in ['chocolate', 'sushi', 'caesar', 'caprese']]

    def test_explicit_order(self):
        chosen = select_cases(self.rows, failure_queries=['sushi', 'chocolate', 'caesar'])
        self.assertEqual([q['query_id'] for q in chosen['failure']], ['sushi', 'chocolate', 'caesar'])

    def test_reject_invalid_selection(self):
        for ids in [['sushi', 'sushi', 'caesar'], ['unknown', 'sushi', 'caesar']]:
            with self.assertRaises(ValueError):select_cases(self.rows, failure_queries=ids)
        with self.assertRaises(ValueError):
            select_cases(self.rows, all_failures=True, failure_queries=['sushi','chocolate','caesar'])
        self.rows[0]['results'][0] = dict(dish_label='food', score=.9)
        with self.assertRaises(ValueError):
            select_cases(self.rows, failure_queries=['chocolate','sushi','caesar'])
