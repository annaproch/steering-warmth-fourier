import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import calibration as c


class CalibrationTests(unittest.TestCase):
    def test_balanced_controls_and_reverse_scoring(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            a=c.parser().parse_args(['prepare','--run',td])
            c.prepare(a)
            items=c.read(a.run/'items.json')
            key=c.read(a.dataset/'answer_key.json')['answers']
            self.assertEqual(len(items),32)
            self.assertEqual(len(key),24)
            for task in ['base','content']:
                values=[key[x['case_id']]['expected'] for x in items if x['task']==task]
                self.assertEqual(values.count(0),4);self.assertEqual(values.count(1),4)
            for item in items:
                self.assertNotIn('expected',item['input'])
                value=key[item['case_id']]['expected']
                if item['flipped']: value=c.swap(value)
                field={'base':'base_correct_0_1','content':'content_preserved_0_1','warmth':'warmer'}[item['task']]
                c.save(a.run/'verdicts'/(item['id']+'.json'),{field:value,'notes':'synthetic test','quality_1_5':5})
            c.save(a.run/'judge_settings.json',{'items_sha':c.digest(a.run/'items.json')})
            c.score(a)
            report=c.read(a.run/'scores.json')
            self.assertEqual(report['position_check'],dict(total_pairs=8,consistent=8,both_correct=8))
            for metrics in report['tasks'].values(): self.assertEqual(metrics['correct'],metrics['total'])
            # Always choosing A should disagree with itself after reversing a non-tie pair.
            pair=[x for x in items if x['case_id']=='warmth_disagreement']
            for item in pair:c.save(a.run/'verdicts'/(item['id']+'.json'),{'warmer':'A'})
            c.score(a)
            self.assertEqual(c.read(a.run/'scores.json')['position_check']['consistent'],7)
            (a.run/'verdicts'/(items[0]['id']+'.json')).unlink()
            with self.assertRaisesRegex(ValueError,'Missing verdict'): c.score(a)

    def test_bad_verdicts_fail(self):
        for value in [True,'1',2,None]:
            with self.assertRaises(ValueError):c.prediction('base',{'base_correct_0_1':value})
        with self.assertRaises(ValueError):c.prediction('warmth',{'warmer':'both'})


if __name__=='__main__': unittest.main()
