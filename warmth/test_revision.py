import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matched as m
import revision as v


class RevisionTests(unittest.TestCase):
    def test_structural_checks(self):
        s={'base_answer':'one two three four five six seven eight nine ten'}
        texts=[s['base_answer']+' '+str(k) for k in range(3)]
        self.assertEqual(v.structural_checks(s,texts),[])
        self.assertIn('duplicate_variants',v.structural_checks(s,[texts[0]]*3))
        self.assertIn('length_outside_20_percent',v.structural_checks(s,['short']+texts[1:]))

    def test_pilot_only_uses_training_scenarios(self):
        source=Path(m.ROOT/'data/curated_smoke_v1.jsonl')
        originals=m.prepare_scenarios([json.loads(x) for x in source.read_text().splitlines()],5,42)
        train={x['id'] for x in originals if x['split']=='train'}
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['seed','--run',td,'--input',str(source),'--per-category','5','--development-pilot'])
            m.seed(a)
            rows=m.read(a.run/'scenarios.json')
            self.assertEqual(len(rows),4)
            self.assertEqual({x['category'] for x in rows},set(m.CATEGORIES))
            self.assertTrue(all(x['id'] in train and x['split']=='train' for x in rows))

    def test_pair_blinding_ties_and_content_judge(self):
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['review','--run',td,'--judge-model','/models/judge'])
            scenario=dict(id='s',group='s',split='train',category='advice',prompt='A prompt',
                          base_answer='one two three four five six seven eight nine ten',invariants='Keep meaning')
            m.save(a.run/'scenarios.json',[scenario])
            m.save(a.run/'draft_settings.json',dict(model={'path':'/writer'},protocol='v2'))
            for k in range(3):
                m.save(a.run/'drafts'/f's_{k}.json',dict(text=scenario['base_answer']+' '+str(k),hit_cap=False))
            m.review(a)
            self.assertEqual(len(m.csv_rows(a.run/'pair_review.csv')),3)
            self.assertNotIn('levels',m.csv_rows(a.run/'pair_review.csv')[0])
            prompts=[]
            def complete(r,prompt):
                prompts.append(prompt)
                if prompt.startswith(v.BASE_RUBRIC):
                    result={'base_correct_0_1':1,'notes':'base checked'}
                elif prompt.startswith(v.CONTENT_RUBRIC):
                    result={'content_preserved_0_1':1,'quality_1_5':4,'notes':'content checked'}
                else: result={'warmer':'tie','notes':'no clear difference'}
                return dict(text=json.dumps(result),hit_cap=False,tokens=20)
            with patch.object(m,'model_id',return_value={'path':'/models/judge'}), \
                 patch.object(m,'runner',return_value=object()), patch.object(m,'complete',side_effect=complete):
                m.judge(a)
            self.assertEqual(sum(p.startswith(v.BASE_RUBRIC) for p in prompts),1)
            self.assertEqual(sum(p.startswith(v.CONTENT_RUBRIC) for p in prompts),3)
            self.assertEqual(sum(p.startswith(v.PAIR_RUBRIC) for p in prompts),3)
            self.assertTrue(all(not r['warmth_0_4'] for r in m.csv_rows(a.run/'review.csv')))
            m.validate(a)
            self.assertEqual(m.read(a.run/'accepted.json'),[])
            self.assertIn('pairwise_warmth_not_ordered',m.read(a.run/'validation_report.json')['rejected'][0]['reasons'])
            rows=m.csv_rows(a.run/'pair_review.csv');fields=list(rows[0])
            key={x['id']:x for x in m.read(a.run/'pair_review_key.json')}
            for r in rows:
                r['warmer']='A' if key[r['id']]['levels'][0]>key[r['id']]['levels'][1] else 'B'
            m.write_csv(a.run/'pair_review.csv',fields,rows)
            m.validate(a)
            self.assertEqual(len(m.read(a.run/'accepted.json')),1)
            rows[0]['response_a']='tampered'
            m.write_csv(a.run/'pair_review.csv',fields,rows)
            with self.assertRaisesRegex(ValueError,'Pair text changed'):m.validate(a)


if __name__=='__main__': unittest.main()
