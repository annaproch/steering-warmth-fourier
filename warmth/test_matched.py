import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import matched as m


class MatchedTests(unittest.TestCase):
    def test_judge_template_switch_does_not_change_default_prompts(self):
        from unittest.mock import Mock
        r=object.__new__(m.old.Runner)
        r.a=SimpleNamespace()
        r.tok=Mock()
        r.tok.apply_chat_template.return_value='formatted prompt'
        r.prompt('question')
        self.assertNotIn('enable_thinking',r.tok.apply_chat_template.call_args.kwargs)
        r.a.chat_template_kwargs={'enable_thinking':False}
        r.prompt('question')
        self.assertIs(r.tok.apply_chat_template.call_args.kwargs['enable_thinking'],False)

    def test_failed_legacy_judge_migration(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['judge','--run',td,'--judge-model','/judge'])
            m.save(a.run/'draft_settings.json',{'model':{'path':'/writer'}})
            m.write_csv(a.run/'review.csv',['reviewer','reviewer_kind'],
                        [{'reviewer':'','reviewer_kind':''}])
            legacy=dict(model={'path':'/judge'},rubric_version=1,max_tokens=a.max_tokens)
            m.save(a.run/'judge_settings.json',legacy)
            m.save(a.run/'judge_raw'/'failed.json',{'hit_cap':True})
            with patch.object(m,'model_id',return_value={'path':'/judge'}), \
                 patch.object(m,'runner',side_effect=RuntimeError('stop before model')):
                with self.assertRaisesRegex(RuntimeError,'stop before model'):m.judge(a)
            self.assertEqual(m.read(a.run/'judge_settings.json')['chat_template_kwargs'],
                             {'enable_thinking':False})
            self.assertEqual(len(list(a.run.glob('judge_legacy_*/judge_raw/failed.json'))),1)
            m.save(a.run/'judge_settings.json',legacy)
            m.write_csv(a.run/'review.csv',['reviewer','reviewer_kind'],
                        [{'reviewer':'previous judge','reviewer_kind':'llm'}])
            with patch.object(m,'model_id',return_value={'path':'/judge'}):
                with self.assertRaisesRegex(ValueError,'Configuration changed'):m.judge(a)

    def scenarios(self, n=5):
        return [dict(id=f'{c}_{i}',group=f'{c}_{i}',category=c,prompt=f'{c} question {i}',
                     base_answer='A correct base answer.',invariants='Keep the answer.')
                for c in m.CATEGORIES for i in range(n)]

    def test_split_counts_and_duplicate_groups(self):
        rows=m.prepare_scenarios(self.scenarios(75),75,42)
        self.assertEqual([sum(x['split']==s for x in rows) for s in ['train','validation','test']],[180,60,60])
        rows=self.scenarios();rows[1]['group']=rows[0]['group']
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.prepare_scenarios(rows,5,42)
        rows=self.scenarios();rows[0]['id']='../unsafe'
        with self.assertRaisesRegex(ValueError,'IDs'):m.prepare_scenarios(rows,5,42)

    def test_geometry_distinguishes_scaling_from_curvature(self):
        linear=torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]]*8)
        stretched=linear.clone();stretched[:,1,0]=.5
        curved=linear.clone();curved[:,1,1]=1
        self.assertEqual(m.geometry(linear)[0]['curvature'],0)
        self.assertEqual(m.geometry(stretched)[0]['curvature'],0)
        self.assertAlmostEqual(m.geometry(stretched)[0]['norm_ratio'],3)
        self.assertGreater(m.geometry(curved)[0]['curvature'],0)
        cats=m.CATEGORIES*2
        report=m.diagnostics(linear,linear,cats,cats,8,1)
        self.assertAlmostEqual(report['held_alignment']['cosines']['mean'],1)
        self.assertIsNone(report['train_held_orth_cosine'])
        # Opposite held-out direction must not be refitted into a positive result.
        reversed=m.diagnostics(linear,linear.flip(1),cats,cats,8,1)
        self.assertAlmostEqual(reversed['held_alignment']['cosines']['mean'],-1)
        json.dumps(report,allow_nan=False)

    def test_causal_suffix_and_batching_with_real_transformer(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(9)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
              num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
              max_position_embeddings=64,pad_token_id=0,eos_token_id=2)).eval()
        r=SimpleNamespace(model=model,tok=SimpleNamespace(eos_token_id=2),
                          a=SimpleNamespace(layers=[1,2],max_length=30,device='cpu'))
        encoded=[([4,5],[6,7,8]),([4],[9])]
        batch=m.extract_batch(r,encoded,[10,11])
        for j,e in enumerate(encoded):
            single=m.extract_batch(r,[e],[10,11])
            alternate=m.extract_batch(r,[e],[12,13])
            for method in batch:torch.testing.assert_close(batch[method][j],single[method][0])
            for method in ['last','mean']:torch.testing.assert_close(single[method],alternate[method])
            self.assertFalse(torch.allclose(single['anchor'],alternate['anchor']))
        self.assertTrue(all(not x._forward_hooks for x in model.model.layers))

    def test_review_gate_and_independent_rater(self):
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['review','--run',td])
            scenarios=m.prepare_scenarios(self.scenarios(),5,42)
            m.save(a.run/'scenarios.json',scenarios)
            m.save(a.run/'draft_settings.json',{'model':{'path':'/models/writer'}})
            for s in scenarios:
                for k in range(3):
                    m.save(a.run/'drafts'/f'{s["id"]}_{k}.json',dict(text=f'response {k}',hit_cap=False))
            m.review(a)
            key={x['id']:x for x in m.read(a.run/'review_key.json')}
            rows=m.csv_rows(a.run/'review.csv')
            with self.assertRaises(ValueError):m.validate(a)
            fields=list(rows[0])
            for row in rows:
                row.update(warmth_0_4=str(key[row['id']]['level']),content_preserved_0_1='1',
                           base_correct_0_1='1',quality_1_5='5',reviewer='unit-test',reviewer_kind='human')
            # Ties reject a whole triplet instead of inventing an order.
            bad=key[rows[0]['id']]['scenario_id']
            for row in rows:
                if key[row['id']]['scenario_id']==bad:row['warmth_0_4']='1'
            m.write_csv(a.run/'review.csv',fields,rows);m.validate(a)
            self.assertEqual(len(m.read(a.run/'accepted.json')),19)
            rows[0].update(reviewer_kind='llm',reviewer_model='writer')
            m.write_csv(a.run/'review.csv',fields,rows)
            with self.assertRaisesRegex(ValueError,'different model'):m.validate(a)

    def test_config_lock_and_training_only_vectors(self):
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['analyze','--run',td,'--layers','1','--bootstrap','3'])
            rows=m.prepare_scenarios(self.scenarios(),5,42)
            for s in rows:s['responses']=['a','b','c']
            m.save(a.run/'accepted.json',rows)
            cfg=m.collection_settings(a);sig=m.hash_json(cfg)
            m.save(a.run/'collection.json',dict(signature=sig,settings=cfg))
            cache=a.run/'cache'/sig;cache.mkdir(parents=True)
            for s in rows:
                for k in range(3):
                    # Held-out directions deliberately reversed; vectors must stay positive.
                    sign=1 if s['split']=='train' else -1
                    torch.save(dict(activations={method:torch.tensor([[sign*k,0.]]) for method in ['mean','last','anchor']}),cache/f'{s["id"]}_{k}.pt')
            m.analyze(a)
            vectors=torch.load(a.run/'vectors.pt',weights_only=True)['vectors']
            torch.testing.assert_close(vectors['last_1'],torch.tensor([2.,0.]))
            a.suffix='Changed suffix'
            with self.assertRaisesRegex(ValueError,'configuration'):m.load_activations(a)
            m.lock(a.run/'lock.json',{'a':1})
            with self.assertRaisesRegex(ValueError,'Configuration'):m.lock(a.run/'lock.json',{'a':2})



class SteeringProtocolTests(unittest.TestCase):
    def test_score_does_not_count_flat_as_control(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            outputs=[dict(id=str(i),scenario_id='s',method='baseline' if strength==0 else 'linear',
                          strength=strength,prompt='p',text='answer',tokens=4,hit_cap=False)
                     for i,strength in enumerate([-1,0,1])]
            m.save(folder/'outputs.json',outputs);m.save(folder/'settings.json',dict(strengths=[-1,1]))
            m.export_generation_review(folder,outputs,42)
            rows=m.csv_rows(folder/'ratings.csv');fields=list(rows[0])
            for row in rows:row.update(warmth_0_4='2',quality_1_5='5',incorrect_claim_0_1='0',
                false_agreement_0_1='0',refusal_0_1='0',reviewer='test',reviewer_kind='human')
            m.write_csv(folder/'ratings.csv',fields,rows)
            a=m.parser().parse_args(['score','--ratings',str(folder/'ratings.csv')]);m.score(a)
            self.assertEqual(m.read(folder/'scores.json')['monotonic_with_positive_span'],0)
            for row in rows:row['warmth_0_4']=row['id']
            m.write_csv(folder/'ratings.csv',fields,rows);m.score(a)
            self.assertEqual(m.read(folder/'scores.json')['monotonic_with_positive_span'],1)

    def test_selection_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            a=m.parser().parse_args(['select','--run',td,'--method','last','--layer','16'])
            torch.save(dict(collection_signature='same',vectors={'last_16':torch.ones(3)}),a.run/'vectors.pt')
            m.save(a.run/'collection.json',dict(signature='same'))
            m.select(a)
            a.strengths=[-1,1]
            with self.assertRaisesRegex(ValueError,'Configuration'):m.select(a)


if __name__=='__main__':unittest.main()
