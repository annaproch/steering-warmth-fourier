"""CPU pilot: ordered prompting is an experimental variable, not a concision label.

Run with PYTHONPATH=.deps /home/agent/venv/bin/python conciseness/pilot.py STAGE.
The upstream pure-Torch spline module is loaded directly without importing the
full causalab application. No upstream code is changed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
MODEL = ROOT / 'models/llama-3.1-8b-instruct'
LAYERS = [12, 20, 28]  # 1-based block outputs
UPSTREAM = ROOT.parent / 'vendor/causalab/causalab/methods/spline/cubic.py'
spec = importlib.util.spec_from_file_location('goodfire_cubic', UPSTREAM)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Spline = module.CubicSpline1D
torch.set_num_threads(8)
torch.manual_seed(42)

# 0 = most concise, 4 = most verbose. These are requested styles only.
STYLES = [
    ['Give a very concise answer with only the essential information.',
     'Give a brief answer with the essential information and minimal explanation.',
     'Give a moderately detailed answer with the essential information and some explanation.',
     'Give a detailed answer with the essential information and supporting explanation.',
     'Give a very detailed answer with the essential information and extensive explanation.'],
    ['Respond as compactly as possible while retaining the key facts.',
     'Respond briefly, retaining the key facts and a little explanation.',
     'Respond with moderate detail, retaining the key facts and some explanation.',
     'Respond at length, retaining the key facts and substantial explanation.',
     'Respond very expansively, retaining the key facts and comprehensive explanation.'],
    ['Keep your answer minimal but informative.',
     'Keep your answer short, with only a little elaboration.',
     'Use a balanced amount of detail and elaboration in your answer.',
     'Elaborate on your answer with plenty of relevant detail.',
     'Elaborate extensively on your answer with thorough discussion of relevant details.'],
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare():
    import pyarrow.parquet as pq
    data = pq.read_table(ROOT / 'data/tonebank.parquet').to_pylist()
    # Deduplicate by actual question, not possibly reused prompt_id.
    data = list({r['prompt'].strip(): r for r in data}.values())
    data.sort(key=lambda r: r['prompt'])
    random.Random(42).shuffle(data)
    rows = []
    for i, r in enumerate(data[:30]):
        split = 'train' if i < 18 else 'validation' if i < 24 else 'test'
        rows.append(dict(r, split=split, question_id=hashlib.sha256(r['prompt'].encode()).hexdigest()[:12]))
    dump(ROOT / 'data/pilot.json', rows)
    dump(OUT / 'provenance.json', {
        'dataset': 'withmartian/TONEBANK', 'public_rows': len(data),
        'dataset_sha256': hashlib.sha256((ROOT / 'data/tonebank.parquet').read_bytes()).hexdigest(),
        'model': 'NousResearch/Meta-Llama-3.1-8B-Instruct',
        'upstream_commit': '1b6f43a509d2a4ae49fd1bc4252e13889d57e38b',
        'upstream_spline_sha256': hashlib.sha256(UPSTREAM.read_bytes()).hexdigest(),
        'layers': LAYERS, 'seed': 42, 'styles': STYLES,
        'caveat': 'Levels are prompted verbosity, not human-rated conciseness. Source responses are retained for coverage review, not assumed ground truth.'})
    print('Prepared', len(rows), 'questions', flush=True)


class Runner:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, local_files_only=True).eval()
        self.model.generation_config.pad_token_id = self.tok.eos_token_id

    def ids(self, question, level, template):
        messages = [dict(role='system', content=STYLES[template][level]),
                    dict(role='user', content=question)]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt', add_special_tokens=False)

    @torch.inference_mode()
    def extract(self, ids):
        acts, hooks = {}, []
        for layer in LAYERS:
            def hook(m, inp, out, layer=layer):
                h = out[0] if isinstance(out, tuple) else out
                acts[layer] = h[0, -1].float().cpu()
            hooks.append(self.model.model.layers[layer-1].register_forward_hook(hook))
        try:
            self.model(**ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return torch.stack([acts[l] for l in LAYERS])

    @torch.inference_mode()
    def generate(self, ids, layer=None, donor=None, delta=None, max_tokens=128):
        handle = None
        if layer is not None:
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Only prefill's final assistant-boundary token: no repeated
                # injection into a different activation population.
                if h.shape[1] != ids['input_ids'].shape[1]:
                    return out
                h = h.clone()
                if donor is not None:
                    h[0, -1] = donor.to(h)
                else:
                    h[0, -1] += delta.to(h)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            handle = self.model.model.layers[layer-1].register_forward_hook(hook)
        try:
            start = time.time()
            result = self.model.generate(**ids, do_sample=False, max_new_tokens=max_tokens,
                                         use_cache=True)
            new = result[0, ids['input_ids'].shape[1]:]
            return {'text': self.tok.decode(new, skip_special_tokens=True),
                    'tokens': len(new), 'hit_cap': len(new) == max_tokens,
                    'seconds': time.time()-start}
        finally:
            if handle is not None:
                handle.remove()


def collect():
    rows = json.loads((ROOT / 'data/pilot.json').read_text())
    runner = Runner()
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        path = cache / (row['question_id']+'.pt')
        if path.exists():
            continue
        template = {'train': 0, 'validation': 1, 'test': 2}[row['split']]
        acts, lengths = [], []
        for level in range(5):
            ids = runner.ids(row['prompt'], level, template)
            acts.append(runner.extract(ids))
            lengths.append(ids['input_ids'].shape[1])
        torch.save({'activations': torch.stack(acts), 'lengths': lengths,
                    'template': template, 'question_id': row['question_id']}, path)
        print(f'Collected {i+1}/{len(rows)} {row["split"]}', flush=True)


def fit_model(x, rank):
    # x: questions x levels x hidden; center within each question to reduce
    # topic variation. Transform is estimated exclusively on training data.
    centered = x - x.mean(1, keepdim=True)
    flat = centered.flatten(0,1).double()
    _, s, vh = torch.linalg.svd(flat, full_matrices=False)
    rank = min(rank, int((s > s[0]*1e-8).sum()))
    basis = vh[:rank].T
    projected = centered.double() @ basis
    centroids = projected.mean(0)
    u = torch.arange(x.shape[1], dtype=torch.float64)[:, None]
    curve = Spline(u, centroids, bc='natural')
    design = torch.cat([torch.ones_like(u), u], 1)
    line = torch.linalg.lstsq(design, centroids).solution
    return {'basis': basis, 'centroids': centroids, 'line': line,
            'variance_fraction': float(s[:rank].square().sum()/s.square().sum())}, curve


def analyze():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((ROOT / 'data/pilot.json').read_text())
    x = torch.stack([torch.load(ROOT/'cache'/(r['question_id']+'.pt'), weights_only=True)['activations'] for r in rows])
    masks = {s: torch.tensor([r['split']==s for r in rows]) for s in ['train','validation','test']}
    metrics, models = [], {}
    for li, layer in enumerate(LAYERS):
        for rank in [2, 4, 8, 16, 32, 64]:
            fit, curve = fit_model(x[masks['train'], :, li], rank)
            models[f'{layer}_{rank}'] = fit
            metric = {'layer':layer, 'rank_requested':rank, 'rank_actual':fit['basis'].shape[1],
                      'variance_fraction':fit['variance_fraction']}
            keep = torch.tensor([0,1,3,4])
            # Omitted level must also be absent from PCA fitting and centering.
            omitted_fit, _ = fit_model(x[masks['train'], :, li][:,keep],rank)
            small = Spline(keep[:,None].double(),omitted_fit['centroids'],bc='natural')
            pred_mid = small.evaluate(torch.tensor([[2.]],dtype=torch.float64)) @ omitted_fit['basis'].T
            lincoef = torch.linalg.lstsq(torch.stack([torch.ones(4),keep]).T.double(),omitted_fit['centroids']).solution
            linmid = torch.tensor([[1.,2.]],dtype=torch.float64) @ lincoef @ omitted_fit['basis'].T
            for split in ['validation','test']:
                y = x[masks[split], :, li].double()
                y = y-y.mean(1,keepdim=True)
                pred = curve.evaluate(torch.arange(5,dtype=torch.float64)[:,None]) @ fit['basis'].T
                line = torch.stack([torch.ones(5),torch.arange(5)]).T.double() @ fit['line'] @ fit['basis'].T
                # Full-dimensional held-out error; no cheap gains by dropping dimensions.
                metric[split+'_spline_mse'] = float((y-pred).square().mean())
                metric[split+'_line_mse'] = float((y-line).square().mean())
                # Omit middle knot to assess interpolation, not just centroid reconstruction.
                raw = x[masks[split], :, li].double()
                mid = raw[:,2]-raw[:,keep].mean(1)
                metric[split+'_omitted_mid_spline_mse'] = float((mid-pred_mid).square().mean())
                metric[split+'_omitted_mid_line_mse'] = float((mid-linmid).square().mean())
            metrics.append(metric)
    # Selection uses validation only. Test metrics are descriptive, not selection criteria.
    best = min(metrics,key=lambda m:m['validation_omitted_mid_spline_mse'])
    dump(OUT/'geometry_metrics.json',metrics)
    dump(OUT/'selected.json',best)
    torch.save(models,ROOT/'cache/fits.pt')
    key = f'{best["layer"]}_{best["rank_requested"]}'
    f = models[key]
    curve = Spline(torch.arange(5,dtype=torch.float64)[:,None],f['centroids'],bc='natural')
    u = torch.linspace(0,4,150,dtype=torch.float64)[:,None]
    p = curve.evaluate(u)
    plt.figure(figsize=(7,5))
    plt.plot(p[:,0],p[:,1],label='Natural spline')
    line = torch.cat([torch.ones_like(u),u],1) @ f['line']
    plt.plot(line[:,0],line[:,1],'--',label='Straight-line fit')
    for i,c in enumerate(f['centroids']):
        plt.scatter(c[0],c[1]); plt.annotate(str(i),(c[0],c[1]))
    plt.xlabel('PC1 (within-question changes)'); plt.ylabel('PC2')
    plt.title('Requested verbosity: 0 concise → 4 verbose\nCandidate geometry; not validated conciseness')
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/'geometry.png',dpi=160); plt.close()
    print(json.dumps(best,indent=2),flush=True)


def generations(stage, n_questions=2, max_tokens=128):
    rows=json.loads((ROOT/'data/pilot.json').read_text())
    split='validation' if stage=='gate' else 'test'
    rows=[r for r in rows if r['split']==split][:n_questions]
    runner=Runner()
    path=OUT/f'{stage}_generations.json'
    results=json.loads(path.read_text()) if path.exists() else []
    done={(r['question_id'],r['method']) for r in results}
    selected=json.loads((OUT/'selected.json').read_text())
    fits=torch.load(ROOT/'cache/fits.pt',weights_only=True)
    f=fits[f'{selected["layer"]}_{selected["rank_requested"]}']
    curve=Spline(torch.arange(5,dtype=torch.float64)[:,None],f['centroids'],bc='natural')
    for row in rows:
        template=1 if split=='validation' else 2
        methods=[('prompt_concise',0,None,None,None),('prompt_middle',2,None,None,None),('prompt_verbose',4,None,None,None)]
        cache=torch.load(ROOT/'cache'/(row['question_id']+'.pt'),weights_only=True)['activations']
        if stage=='gate':
            for li,l in enumerate(LAYERS):
                methods.append((f'donor_L{l}',2,l,cache[0,li],None))
        else:
            l=selected['layer']
            spline_delta=((curve.evaluate(torch.tensor([[0.5]],dtype=torch.float64))-curve.evaluate(torch.tensor([[2.]],dtype=torch.float64)))@f['basis'].T)[0]
            line_delta=(-1.5*f['line'][1])@f['basis'].T
            matched=line_delta*spline_delta.norm()/line_delta.norm().clamp_min(1e-12)
            methods += [('spline_half',2,l,None,spline_delta),('line_half',2,l,None,line_delta),('line_matched',2,l,None,matched)]
        for name,level,layer,donor,delta in methods:
            if (row['question_id'],name) in done: continue
            print('Generating',stage,row['question_id'],name,flush=True)
            result=runner.generate(runner.ids(row['prompt'],level,template),layer,donor,delta,max_tokens)
            results.append(dict(result,question_id=row['question_id'],question=row['prompt'],method=name,
                                source_concise=row['response_concise'],layer=layer,
                                intervention_norm=float(delta.norm()) if delta is not None else None))
            dump(path,results)
            print('Finished',result['tokens'],'tokens',round(result['seconds'],1),'sec',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','collect','analyze','gate','steer'])
    parser.add_argument('--questions',type=int,default=2,help='Generation questions per split (up to 6).')
    parser.add_argument('--max-tokens',type=int,default=128)
    args=parser.parse_args()
    if args.stage in ['gate','steer']: generations(args.stage,args.questions,args.max_tokens)
    else: globals()[args.stage]()
