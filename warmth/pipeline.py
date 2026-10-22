"""EPITOME ER pilot. Real labels; descriptive geometry followed by causal tests."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
UPSTREAM = REPO / 'vendor/causalab/causalab/methods/spline/cubic.py'
spec = importlib.util.spec_from_file_location('goodfire_cubic', UPSTREAM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Spline = mod.CubicSpline1D
FILES = ['emotional-reactions-reddit.csv', 'interpretations-reddit.csv', 'explorations-reddit.csv']
SYSTEM = 'Respond helpfully and accurately to the user. Do not invent facts or personal experiences.'


def digest(x):
    return hashlib.sha256(x).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def normalized(s):
    return ' '.join(s.lower().split())


def prepare(a):
    tables = []
    conflicts, duplicate_counts = set(), {}
    for name in FILES:
        with (a.data / name).open(newline='') as f:
            table = {}
            duplicate_counts[name] = 0
            for r in csv.DictReader(f):
                key = (r['sp_id'], r['rp_id'])
                if key in table:
                    duplicate_counts[name] += 1
                    if any(table[key][field] != r[field] for field in
                           ['level', 'seeker_post', 'response_post']):
                        conflicts.add(key)
                table[key] = r
            tables.append(table)
    candidates = []
    text_mismatches = 0
    seen = set()
    # Union seeker IDs and normalized text to prevent duplicate-prompt leakage.
    parent = {}
    def root(k):
        parent.setdefault(k, k)
        if parent[k] != k:
            parent[k] = root(parent[k])
        return parent[k]
    for r in tables[0].values():
        keys = ['id:' + r['sp_id'], 'text:' + digest(normalized(r['seeker_post']).encode())]
        parent[root(keys[1])] = root(keys[0])
    for key, r in tables[0].items():
        if key in conflicts or any(key not in t for t in tables):
            continue
        if any(normalized(t[key][field]) != normalized(r[field])
               for t in tables for field in ['response_post', 'seeker_post']):
            text_mismatches += 1
            continue
        label, ip, ex = [int(t[key]['level']) for t in tables]
        if any(v not in (0, 1, 2) for v in (label, ip, ex)):
            raise ValueError('Invalid label')
        words = len(r['response_post'].split())
        if not 4 <= words <= 256 or not r['seeker_post'].strip():
            continue
        # Remove repeated replies globally (before splitting).
        response_key = normalized(r['response_post'])
        if response_key in seen:
            continue
        seen.add(response_key)
        group = root('id:' + r['sp_id'])
        fraction = int(digest((str(a.seed) + group).encode())[:8], 16) / 2**32
        split = 'train' if fraction < .6 else 'validation' if fraction < .8 else 'test'
        candidates.append(dict(id=digest('|'.join(key).encode())[:16], group=group,
                               prompt=r['seeker_post'], response=r['response_post'],
                               er=label, ip=ip, ex=ex, words=words, split=split,
                               stratum=f'{ip}:{ex}:{int(math.log2(words))}'))
    rng = random.Random(a.seed)
    selected, report = [], {}
    for split, cap in [('train', a.train_per_level), ('validation', a.eval_per_level), ('test', a.eval_per_level)]:
        bins = {}
        for r in candidates:
            if r['split'] == split:
                bins.setdefault(r['stratum'], [[], [], []])[r['er']].append(r)
        triples = []
        for buckets in bins.values():
            for b in buckets:
                rng.shuffle(b)
            triples.extend(zip(*buckets))
        rng.shuffle(triples)
        triples = triples[:cap]
        if len(triples) < 3:
            raise ValueError(f'Only {len(triples)} matched triples for {split}; inspect data before proceeding')
        selected.extend(r for triple in triples for r in triple)
        report[split] = {'per_er_level': len(triples), 'requested_per_level': cap}
    manifest = {'seed': a.seed, 'counts': report,
                'duplicate_rows':duplicate_counts, 'excluded_conflicting_keys':len(conflicts),
                'excluded_cross_file_text_mismatches':text_mismatches,
                'source': 'https://github.com/behavioral-data/Empathy-Mental-Health',
                'source_hashes': {f: digest((a.data/f).read_bytes()) for f in FILES},
                'matching': 'Equal ER counts within IP, EX and log2 response-word-length bins, separately per split.',
                'rows': selected}
    save(a.run/'dataset.json', manifest)
    print(json.dumps(report, indent=2))


def config(a):
    return {'dataset_sha': digest((a.run/'dataset.json').read_bytes()),
            'pipeline_sha':digest(Path(__file__).read_bytes()),
            'model': str(a.model.resolve()), 'layers': a.layers, 'max_length': a.max_length,
            'system': SYSTEM, 'pooling': 'mean of all response content tokens, excluding EOT',
            'model_config_sha': digest((a.model/'config.json').read_bytes()),
            'tokenizer_sha': digest((a.model/'tokenizer_config.json').read_bytes())}


def signature(a):
    c = config(a)
    return digest(json.dumps(c, sort_keys=True).encode()), c


class Runner:
    def __init__(self, a):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.a = a
        self.tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            a.model, local_files_only=True, dtype=torch.bfloat16).to(a.device).eval()
        self.model.generation_config.pad_token_id = self.tok.eos_token_id
        if max(a.layers) > len(self.model.model.layers):
            raise ValueError('Layer outside model')

    def prompt(self, text):
        s = self.tok.apply_chat_template([{'role':'system','content':SYSTEM},
                    {'role':'user','content':text}], tokenize=False, add_generation_prompt=True)
        return self.tok.encode(s, add_special_tokens=False)

    def encode_row(self, row):
        prefix = self.prompt(row['prompt'])
        response = self.tok.encode(row['response'], add_special_tokens=False)
        return prefix, response

    def extract(self, row):
        return self.extract_batch([self.encode_row(row)])[0]

    @torch.inference_mode()
    def extract_batch(self, encoded):
        results = [None] * len(encoded)
        valid = []
        for i, (prefix, response) in enumerate(encoded):
            tokens = len(prefix) + len(response)
            if tokens > self.a.max_length or not response:
                results[i] = {'skipped': 'over_max_length' if tokens > self.a.max_length
                              else 'empty_response', 'tokens': tokens}
            else:
                valid.append((i, prefix, response))
        if not valid:
            return results
        width = max(len(p) + len(r) for _, p, r in valid)
        ids = torch.full((len(valid), width), self.tok.eos_token_id,
                         dtype=torch.long, device=self.a.device)
        mask = torch.zeros_like(ids)
        for j, (_, prefix, response) in enumerate(valid):
            n = len(prefix) + len(response)
            ids[j, :n] = torch.tensor(prefix + response, device=self.a.device)
            mask[j, :n] = 1
        acts, handles = {}, []
        for layer in self.a.layers:
            def hook(m, inp, out, layer=layer):
                h = out[0] if isinstance(out, tuple) else out
                # Slice before converting to float; exclude both prompt and right padding.
                acts[layer] = torch.stack([
                    h[j, len(prefix):len(prefix)+len(response)].float().mean(0)
                    for j, (_, prefix, response) in enumerate(valid)]).cpu()
            handles.append(self.model.model.layers[layer-1].register_forward_hook(hook))
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        for j, (i, prefix, response) in enumerate(valid):
            results[i] = {'activations': torch.stack([acts[l][j] for l in self.a.layers]),
                          'prompt_tokens': len(prefix), 'response_tokens': len(response)}
        return results

    @torch.inference_mode()
    def generate(self, prompt, layer, delta):
        prefix = self.prompt(prompt)
        if len(prefix) + self.a.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Generation would exceed the model context window')
        handle = None
        if delta is not None:
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Response-token intervention only; the first token is unsteered.
                if h.shape[1] != 1:
                    return out
                h = h.clone()
                h[:, -1] += delta.to(h)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            handle = self.model.model.layers[layer-1].register_forward_hook(hook)
        try:
            ids = torch.tensor([prefix], device=self.a.device)
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=self.a.max_tokens, do_sample=False, use_cache=True)
            new = output[0, len(prefix):]
            return {'text': self.tok.decode(new, skip_special_tokens=True),
                    'tokens': len(new), 'hit_cap': len(new) == self.a.max_tokens}
        finally:
            if handle is not None:
                handle.remove()


def collect(a):
    sig, cfg = signature(a)
    save(a.run/'collection_config.json', cfg)
    cache = a.run/'cache'/sig
    cache.mkdir(parents=True, exist_ok=True)
    rows = read(a.run/'dataset.json')['rows']
    pending = [r for r in rows if not (cache/(r['id']+'.pt')).exists()]
    if not pending:
        print(f'All {len(rows)} examples already cached.', flush=True)
        return
    runner = Runner(a)
    pending = [(row, runner.encode_row(row)) for row in pending]
    pending.sort(key=lambda item: sum(map(len, item[1])))
    done = len(rows) - len(pending)
    started = time.perf_counter()
    processed = 0
    for offset in range(0, len(pending), a.batch_size):
        batch = pending[offset:offset+a.batch_size]
        tick = time.perf_counter()
        results = runner.extract_batch([encoded for _, encoded in batch])
        for (row, _), result in zip(batch, results):
            p = cache/(row['id']+'.pt')
            torch.save(result, p.with_suffix('.tmp'))
            p.with_suffix('.tmp').replace(p)
        done += len(batch)
        processed += len(batch)
        print(f'{done}/{len(rows)} batch={len(batch)} '
              f'seconds={time.perf_counter()-tick:.1f} '
              f'examples/s={processed/(time.perf_counter()-started):.3f}', flush=True)


def fit_geometry(x, labels, rank):
    x = x.double()
    mean = x.mean(0)
    _, s, vh = torch.linalg.svd(x-mean, full_matrices=False)
    effective = int((s > s[0]*1e-8).sum()) if float(s[0]) > 0 else 0
    if effective == 0:
        raise ValueError('No activation variation')
    basis = vh[:min(rank, effective)].T
    z = (x-mean) @ basis
    centroids = torch.stack([z[labels == k].mean(0) for k in range(3)])
    curve = Spline(torch.arange(3, dtype=torch.float64)[:, None], centroids, bc='natural')
    return {'mean':mean, 'basis':basis, 'centroids':centroids,
            'variance_fraction':float(s[:basis.shape[1]].square().sum()/s.square().sum())}, curve


def displacement(f, target, method):
    c = f['centroids']
    curve = Spline(torch.arange(3, dtype=torch.float64)[:, None], c, bc='natural')
    spline = (curve.evaluate(torch.tensor([[target]], dtype=torch.float64))[0] - c[1]) @ f['basis'].T
    line = (target-1) * (c[2]-c[0])/2 @ f['basis'].T
    if method == 'spline':
        return spline
    if method == 'linear':
        return line
    return line * spline.norm() / line.norm().clamp_min(1e-12)


def fit(a):
    sig, cfg = signature(a)
    rows, acts, skipped = [], [], []
    for r in read(a.run/'dataset.json')['rows']:
        p = a.run/'cache'/sig/(r['id']+'.pt')
        if not p.exists():
            raise ValueError('Collection incomplete or configuration differs; run collect with the same arguments')
        d = torch.load(p, weights_only=True)
        if 'skipped' in d:
            skipped.append(r['id'])
        else:
            rows.append(r); acts.append(d['activations'])
    # Rebalance after token-length filtering, using the same exact strata.
    keep = []
    for split in ['train', 'validation', 'test']:
        for st in sorted({r['stratum'] for r in rows}):
            bins = [[i for i,r in enumerate(rows) if r['split']==split and r['stratum']==st and r['er']==k] for k in range(3)]
            n = min(map(len, bins))
            keep.extend(i for b in bins for i in b[:n])
    x = torch.stack(acts)[keep]
    rows = [rows[i] for i in keep]
    masks = {s: torch.tensor([r['split']==s for r in rows]) for s in ['train','validation','test']}
    labels = torch.tensor([r['er'] for r in rows])
    if any(int(m.sum()) < 9 for m in masks.values()):
        raise ValueError('Fewer than 3 examples per class after token filtering; increase data or max length')
    metrics, fits = [], {}
    for li, layer in enumerate(a.layers):
        for rank in a.ranks:
            f, curve = fit_geometry(x[masks['train'], li], labels[masks['train']], rank)
            key = f'{layer}_{rank}'
            fits[key] = f
            metric = {'key':key, 'layer':layer, 'rank':f['basis'].shape[1], 'variance_fraction':f['variance_fraction']}
            c = f['centroids']
            metric['midpoint_deviation'] = float((c[1]-(c[0]+c[2])/2).norm())
            for split in ['validation', 'test']:
                y = x[masks[split],li].double()
                lab = labels[masks[split]]
                pred = f['mean'] + c[lab] @ f['basis'].T
                straight = f['mean'] + (c[0] + lab[:,None]*(c[2]-c[0])/2) @ f['basis'].T
                # Normalize for layer scale; raw cross-layer MSE isn't comparable.
                denom = (y-f['mean']).square().mean().clamp_min(1e-12)
                metric[split+'_spline_nmse'] = float((y-pred).square().mean()/denom)
                metric[split+'_linear_nmse'] = float((y-straight).square().mean()/denom)
            metrics.append(metric)
    best = min(metrics, key=lambda m:m['validation_spline_nmse'])
    torch.save({'fits':fits, 'collection_signature':sig}, a.run/'fits.pt')
    save(a.run/'selected.json', best)
    save(a.run/'geometry_metrics.json', {'metrics':metrics, 'excluded_over_length':skipped,
         'retained_counts':{s: int(m.sum()) for s,m in masks.items()},
         'spline_sha':digest(UPSTREAM.read_bytes()), 'collection':cfg,
         'warning':'At integer knots spline predictions equal centroids; this is not evidence of nonlinear interpolation.'})
    save(a.run/'fitted_rows.json', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = fits[best['key']]
    c = f['centroids']
    curve = Spline(torch.arange(3,dtype=torch.float64)[:,None], c, bc='natural')
    p = curve.evaluate(torch.linspace(0,2,101,dtype=torch.float64)[:,None])
    def xy(v):
        return v[:,0], v[:,1] if v.shape[1]>1 else torch.zeros(len(v))
    plt.plot(*xy(p), label='Natural spline')
    plt.plot(*xy(c[[0,2]]), '--', label='Endpoint line')
    plt.scatter(*xy(c))
    for k,(xx,yy) in enumerate(zip(*xy(c))):
        plt.annotate(f'ER{k}', (xx,yy))
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    plt.title('EPITOME ER: candidate geometry, not validated control')
    plt.tight_layout(); plt.savefig(a.run/'geometry.png',dpi=160); plt.close()
    print(json.dumps(best,indent=2))


def steer(a):
    sig, _ = signature(a)
    bundle = torch.load(a.run/'fits.pt', weights_only=True)
    if bundle['collection_signature'] != sig:
        raise ValueError('Fitting/collection configuration mismatch')
    selected = read(a.run/'selected.json')
    f = bundle['fits'][selected['key']]
    rows = [r for r in read(a.run/'fitted_rows.json') if r['split']==a.split]
    unique = list({r['group']:r for r in rows}.values())[:a.questions]
    settings = {'fit_sha':digest((a.run/'fits.pt').read_bytes()), 'split':a.split,
                'targets':a.targets, 'strength':a.strength, 'max_tokens':a.max_tokens,
                'intervention':'fixed delta from reference ER1 at every response token; prefill untouched'}
    genid = digest(json.dumps(settings,sort_keys=True).encode())[:12]
    folder = a.run/('generations_'+genid)
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'settings.json',settings)
    results = read(folder/'outputs.json') if (folder/'outputs.json').exists() else []
    done = {r['id'] for r in results}
    runner = Runner(a)
    for row in unique:
        methods = [('baseline',None)] + [(m,t) for t in a.targets for m in ['spline','linear','linear_matched']]
        for method,target in methods:
            ident = digest(f'{row["id"]}:{method}:{target}:{genid}'.encode())[:16]
            if ident in done:
                continue
            print(f'Generate {row["id"]} {method} {target}',flush=True)
            delta = None if target is None else a.strength*displacement(f,target,method)
            start = time.time()
            result = runner.generate(row['prompt'], selected['layer'], delta)
            results.append(dict(result,id=ident,group=row['group'],prompt=row['prompt'],method=method,
                                target=target,seconds=time.time()-start,
                                delta_norm=0 if delta is None else float(delta.norm())))
            save(folder/'outputs.json',results)
    # Never overwrite human ratings when resuming or expanding generation.
    review = folder/'ratings.csv'
    existing = {}
    if review.exists():
        with review.open(newline='') as stream:
            existing = {r['id']:r for r in csv.DictReader(stream)}
    blinded = results.copy(); random.Random(a.seed).shuffle(blinded)
    fields = ['id','prompt','response','er_0_2','coherence_1_5','unsupported_claim_0_1','notes']
    with review.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for r in blinded:
            writer.writerow(existing.get(r['id'],dict(id=r['id'],prompt=r['prompt'],response=r['text'])))
    print('Review blinded outputs in',review,flush=True)


def score(a):
    if a.ratings is None:
        raise ValueError('Pass --ratings path/to/ratings.csv after filling all ratings')
    outputs = {r['id']:r for r in read(a.ratings.parent/'outputs.json')}
    groups = {}
    with a.ratings.open(newline='') as stream:
        ratings = list(csv.DictReader(stream))
    if len({r['id'] for r in ratings}) != len(ratings) or {r['id'] for r in ratings} != set(outputs):
        raise ValueError('Ratings must contain exactly one row per output')
    for r in ratings:
        vals = [int(r[k]) for k in ['er_0_2','coherence_1_5','unsupported_claim_0_1']]
        if vals[0] not in range(3) or vals[1] not in range(1,6) or vals[2] not in range(2):
            raise ValueError('Invalid rating')
        out = outputs[r['id']]
        if out.get('skipped'):
            raise ValueError('Skipped generation cannot be scored; use shorter prompts or larger max length')
        key = f'{out["method"]}:{out["target"]}'
        groups.setdefault(key,[]).append(vals+[out['tokens'],int(out['hit_cap'])])
    report = {k:dict(zip(['er','coherence','unsupported_claim_rate','tokens','cap_rate'],
                         torch.tensor(v,dtype=torch.float64).mean(0).tolist()),n=len(v)) for k,v in groups.items()}
    save(a.ratings.parent/'scores.json',report)
    print(json.dumps(report,indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','collect','fit','steer','score'])
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--run', type=Path, default=ROOT/'runs/pilot')
    p.add_argument('--model', type=Path, default=REPO/'conciseness/models/llama-3.1-8b-instruct')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2, help='Collection batch size; rows grouped by token length')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--layers',type=int,nargs='+',default=[12,16,20])
    p.add_argument('--ranks',type=int,nargs='+',default=[2,4,8,16])
    p.add_argument('--train-per-level',type=int,default=30)
    p.add_argument('--eval-per-level',type=int,default=10)
    p.add_argument('--max-length',type=int,default=768)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--questions',type=int,default=3)
    p.add_argument('--targets',type=float,nargs='+',default=[0,0.5,1.5,2])
    p.add_argument('--strength',type=float,default=1)
    p.add_argument('--ratings',type=Path)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if min(a.layers+a.ranks+[a.questions,a.max_length,a.max_tokens,a.threads,a.batch_size]) < 1:
        raise ValueError('Sizes and layers must be positive')
    if any(t < 0 or t > 2 for t in a.targets):
        raise ValueError('Targets must be in [0,2]')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    a.run.mkdir(parents=True,exist_ok=True)
    globals()[a.stage](a)
