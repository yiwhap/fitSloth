import unittest
from train_seed_study import aggregate
from evaluate_search import METRICS


class SeedStudyTests(unittest.TestCase):
    def run_data(self, seed, value):
        return {'train_seed': seed, 'pipelines': [
            {'pipeline': 'baseline', **dict.fromkeys(METRICS, .5)},
            {'pipeline': 'finetune', **dict.fromkeys(METRICS, value)}]}

    def test_sample_sd_over_runs(self):
        result = aggregate([self.run_data(i, i / 10) for i in range(5)])
        for metric in METRICS:
            self.assertEqual(result[0][metric], {'mean': .5, 'sd': 0.})
            self.assertAlmostEqual(result[1][metric]['mean'], .2)
            self.assertAlmostEqual(result[1][metric]['sd'], .025 ** .5)

    def test_rejects_duplicate_seeds_and_mismatched_pipelines(self):
        with self.assertRaises(ValueError):
            aggregate([self.run_data(0, .1), self.run_data(0, .2)])
        other = self.run_data(1, .2)
        other['pipelines'].reverse()
        with self.assertRaises(ValueError):
            aggregate([self.run_data(0, .1), other])
