import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from setup_project import prepare, validate_dataset


class SetupTests(unittest.TestCase):
    def test_incomplete_existing_dataset_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('setup_project.DATASET', root), patch('setup_project.run') as run:
                with self.assertRaises(FileNotFoundError):
                    prepare()
                run.assert_not_called()

    def test_missing_image_is_reported_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'catalog.csv').write_text('image_id,filename,dish_label\na,a.jpg,soup\n', encoding='utf-8')
            with self.assertRaisesRegex(FileNotFoundError, 'images/a.jpg'):
                validate_dataset(root)

    def test_reads_dataset_at_unicode_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'ข้อมูล reviewer'
            (root / 'images').mkdir(parents=True)
            (root / 'queries').mkdir()
            (root / 'images/a.jpg').touch()
            (root / 'queries/q.jpg').touch()
            (root / 'catalog.csv').write_text('image_id,filename,dish_label\na,a.jpg,soup\n', encoding='utf-8')
            (root / 'queries.csv').write_text('query_id,filename,true_label\nq,q.jpg,soup\n', encoding='utf-8')
            self.assertEqual(validate_dataset(root)[0]['query_id'], 'q')
