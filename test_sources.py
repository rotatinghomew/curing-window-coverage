from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
import unittest

from check_sources import load_sources

DATA = Path(__file__).with_name('data')


class ExchangeTests(unittest.TestCase):
    def test_example_is_readable(self):
        tables = load_sources(DATA)
        self.assertTrue(all(tables.values()))
        by_lot = {r['id']: r for r in tables['lots']}
        for truck in tables['trucks']:
            self.assertTrue(set(truck['loaded'].split(',')) - {''} <= set(truck['lots'].split(',')))
            self.assertTrue(all(by_lot[lot]['period'] == truck['period'] for lot in truck['lots'].split(',')))
        early = {r['feed'] for r in tables['lots'] if r['period'] == 'train'}
        later = {r['feed'] for r in tables['lots'] if r['period'] == 'holdout'}
        self.assertFalse(early & later)
        self.assertGreater(len(later), 10)

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / 'copies'
            shutil.copytree(DATA, duplicate)
            path = duplicate / 'lots.csv'
            rows = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(rows + [rows[1]]) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '标识为空或重复'):
                load_sources(duplicate)
