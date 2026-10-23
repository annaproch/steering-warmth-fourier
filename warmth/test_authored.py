import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import prepare_authored as a


class AuthoredTests(unittest.TestCase):
    def setUp(self):
        self.source=a.ROOT/'data/authored_warmth_v1.jsonl'
        self.rows=[json.loads(x) for x in self.source.read_text().splitlines()]

    def test_integrity_and_meaning_change_rejected(self):
        report=a.check(self.rows)
        self.assertEqual(report['splits'],dict(train=18,validation=6,test=6))
        self.assertEqual(report['unique_openings'],90)
        bad=copy.deepcopy(self.rows);bad[0]['responses'][2]+=' You will succeed.'
        with self.assertRaisesRegex(ValueError,'Substantive answer changed'):a.check(bad)
        bad=copy.deepcopy(self.rows);bad[0]['split']='test'
        if self.rows[0]['split']=='test':bad[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'6/2/2'):a.check(bad)

    def test_import_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            run=Path(td)/'run';a.prepare(self.source,run)
            rows=json.loads((run/'accepted.json').read_text())
            self.assertEqual(len(rows),30)
            self.assertTrue(all(not r['human_reviewed'] and r['observed_warmth']==[None]*3 for r in rows))
            audit=(run/'audit.csv').read_text();(run/'audit.csv').write_text(audit+'\n')
            a.prepare(self.source,run)
            self.assertEqual((run/'audit.csv').read_text(),audit+'\n')
            (run/'accepted.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'changed'):a.prepare(self.source,run)
            other=Path(td)/'old';other.mkdir();(other/'existing.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'fresh'):a.prepare(self.source,other)


if __name__=='__main__':unittest.main()
