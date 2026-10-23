import collections
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
import pipeline as p


class GeometryTests(unittest.TestCase):
    def test_response_pooling_and_decode_only_hook(self):
        class FakeModel:
            def __init__(self):
                self.model = SimpleNamespace(layers=[torch.nn.Identity()])
                self.config = SimpleNamespace(max_position_embeddings=100)
                self.observed = []
            def __call__(self, input_ids, **kwargs):
                h = input_ids.float().unsqueeze(-1).repeat(1,1,2)
                self.model.layers[0](h)
            def generate(self, input_ids, **kwargs):
                for length in [input_ids.shape[1], 1, 1]:
                    self.observed.append(self.model.layers[0](torch.zeros(1,length,2)))
                return torch.cat([input_ids,torch.tensor([[8,9]])],1)
        runner = p.Runner.__new__(p.Runner)
        runner.a = SimpleNamespace(layers=[1],max_length=20,device='cpu',max_tokens=2)
        runner.model = FakeModel()
        runner.prompt = lambda _: [100,100,100]
        runner.tok = SimpleNamespace(eos_token_id=999, encode=lambda *args,**kw:[2,4], decode=lambda *args,**kw:'reply')
        result = runner.extract({'prompt':'seeker','response':'reply'})
        torch.testing.assert_close(result['activations'],torch.tensor([[3.,3.]]))
        encoded = [([100,100,100], [2,4]), ([100], [6]), ([1]*20, [2])]
        batched = runner.extract_batch(encoded)
        torch.testing.assert_close(batched[0]['activations'], result['activations'])
        torch.testing.assert_close(batched[1]['activations'], torch.tensor([[6.,6.]]))
        self.assertEqual(batched[2]['skipped'], 'over_max_length')
        runner.generate('seeker',1,torch.tensor([1.,2.]))
        torch.testing.assert_close(runner.model.observed[0],torch.zeros(1,3,2))
        for observed in runner.model.observed[1:]:
            torch.testing.assert_close(observed,torch.tensor([[[1.,2.]]]))
        self.assertEqual(len(runner.model.model.layers[0]._forward_hooks),0)
        runner.a.steer_first_token=True
        runner.model.observed=[]
        runner.generate('seeker',1,torch.tensor([1.,2.]))
        expected=torch.zeros(1,3,2);expected[:,-1]=torch.tensor([1.,2.])
        torch.testing.assert_close(runner.model.observed[0],expected)
        for observed in runner.model.observed[1:]:
            torch.testing.assert_close(observed,torch.tensor([[[1.,2.]]]))
        # Zero shifts preserve every position, and each generate resets prefill state.
        runner.model.observed=[]
        runner.generate('seeker',1,torch.zeros(2))
        self.assertTrue(all(not x.any() for x in runner.model.observed))
        runner.model.observed=[]
        runner.generate('seeker',1,None)
        self.assertTrue(all(not x.any() for x in runner.model.observed))
        # A one-token prompt is still prefill, not a decode step.
        runner.prompt=lambda _: [100]
        runner.a.steer_first_token=False
        runner.model.observed=[]
        runner.generate('seeker',1,torch.tensor([1.,2.]))
        torch.testing.assert_close(runner.model.observed[0],torch.zeros(1,1,2))
        def fail(input_ids,**kwargs):
            runner.model.model.layers[0](torch.zeros(1,1,2))
            raise RuntimeError('generation failed')
        runner.model.generate=fail
        with self.assertRaisesRegex(RuntimeError,'generation failed'):
            runner.generate('seeker',1,torch.ones(2))
        self.assertEqual(len(runner.model.model.layers[0]._forward_hooks),0)

    def test_linear_and_curved_controls(self):
        labels = torch.arange(3).repeat_interleave(4)
        linear = torch.stack([labels.double(), labels.double()*2], 1)
        f, curve = p.fit_geometry(linear, labels, 2)
        self.assertEqual(f['basis'].shape[1], 1)
        for t in [0., .5, 1., 1.5, 2.]:
            torch.testing.assert_close(p.displacement(f,t,'spline'),p.displacement(f,t,'linear'))
        curved = torch.stack([labels.double(), (labels == 1).double()],1)
        f, curve = p.fit_geometry(curved, labels, 2)
        torch.testing.assert_close(curve.evaluate(torch.arange(3,dtype=torch.float64)[:,None]),f['centroids'])
        self.assertGreater(float((p.displacement(f,.5,'spline')-p.displacement(f,.5,'linear')).norm()),.1)
        torch.testing.assert_close(p.displacement(f,.5,'spline').norm(),p.displacement(f,.5,'linear_matched').norm())
        torch.testing.assert_close(p.displacement(f,1.,'spline'),torch.zeros(2,dtype=torch.float64))

    def test_real_data_split_and_matching(self):
        if not (p.ROOT/'data'/p.FILES[0]).exists():
            self.skipTest('EPITOME not downloaded')
        with tempfile.TemporaryDirectory() as td:
            a=p.parser().parse_args(['prepare','--run',td])
            p.prepare(a)
            rows=p.read(a.run/'dataset.json')['rows']
            group_splits=collections.defaultdict(set)
            texts=collections.defaultdict(set)
            strata=collections.defaultdict(collections.Counter)
            for r in rows:
                group_splits[r['group']].add(r['split'])
                texts[p.normalized(r['prompt'])].add(r['split'])
                strata[r['split'],r['stratum']][r['er']]+=1
            self.assertTrue(all(len(s)==1 for s in group_splits.values()))
            self.assertTrue(all(len(s)==1 for s in texts.values()))
            self.assertTrue(all(c[0]==c[1]==c[2] for c in strata.values()))
            self.assertEqual(len(rows),150)

    def test_fit_integration_and_cache_signature(self):
        with tempfile.TemporaryDirectory() as td:
            a=p.parser().parse_args(['fit','--run',td,'--layers','1','--ranks','2'])
            rows=[]
            for split in ['train','validation','test']:
                for k in range(3):
                    for i in range(4):
                        rows.append(dict(id=f'{split}_{k}_{i}',split=split,er=k,stratum='0:0:4'))
            p.save(a.run/'dataset.json',{'rows':rows})
            sig,_=p.signature(a)
            cache=a.run/'cache'/sig;cache.mkdir(parents=True)
            torch.manual_seed(8)
            for r in rows:
                x=torch.randn(1,8)*.05
                x[0,0]+=r['er'];x[0,1]+=int(r['er']==1)
                torch.save({'activations':x},cache/(r['id']+'.pt'))
            p.fit(a)
            self.assertTrue((a.run/'geometry.png').exists())
            self.assertEqual(p.read(a.run/'selected.json')['layer'],1)
            a.max_length+=1
            self.assertNotEqual(p.signature(a)[0],sig)
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                p.fit(a)


if __name__=='__main__':
    unittest.main()
