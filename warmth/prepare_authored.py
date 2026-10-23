"""Import directly authored triplets for exploratory activation collection.

No model generation or judge approval. Intended levels are not observed ratings.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parent


def check(rows):
    if len(rows)!=30: raise ValueError('Expected 30 authored triplets')
    ids=set();groups=set();prompts=set();openings=set()
    for r in rows:
        for key in ['id','group','prompt','base_answer','invariants','source']:
            if not isinstance(r.get(key),str) or not r[key].strip(): raise ValueError(f'Missing {key}')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']): raise ValueError('Unsafe ID')
        prompt=' '.join(r['prompt'].lower().split())
        if r['id'] in ids or r['group'] in groups or prompt in prompts: raise ValueError('Duplicate scenario/group/prompt')
        ids.add(r['id']);groups.add(r['group']);prompts.add(prompt)
        if len(r['responses'])!=3 or len(r['openings'])!=3: raise ValueError('Three levels required')
        if r['intended_levels']!=['neutral','moderately_warm','strongly_warm']: raise ValueError('Unexpected level order')
        for opening,response in zip(r['openings'],r['responses']):
            if not isinstance(opening,str) or not opening.strip(): raise ValueError('Empty opening')
            if response!=opening+' '+r['base_answer']: raise ValueError(f'Substantive answer changed: {r["id"]}')
            normalized=' '.join(opening.lower().split())
            if normalized in openings: raise ValueError('Repeated opening')
            openings.add(normalized)
        lengths=[len(x.split()) for x in r['responses']]
        if max(lengths)/min(lengths)>1.1: raise ValueError(f'Within-triplet length difference exceeds 10%: {r["id"]}')
    categories=Counter(r['category'] for r in rows)
    if categories!=dict(emotional_support=10,advice=10,factual_disagreement=10): raise ValueError('Unexpected categories')
    for cat in categories:
        if Counter(r['split'] for r in rows if r['category']==cat)!=dict(train=6,validation=2,test=2):
            raise ValueError('Expected 6/2/2 split per category')
    return dict(triplets=30,responses=90,categories=dict(categories),
        splits=dict(Counter(r['split'] for r in rows)),unique_openings=len(openings),
        mean_words_by_level=[sum(len(r['responses'][k].split()) for r in rows)/30 for k in range(3)],
        max_within_triplet_length_ratio=max(max(len(s.split()) for s in r['responses'])/
                                          min(len(s.split()) for s in r['responses']) for r in rows))


def prepare(source,run):
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    report=check(rows)
    # This filename is the existing collector's input interface, not a claim of review.
    accepted=[dict(r,observed_warmth=[None]*3,human_reviewed=False,
                   admission_basis='direct_authoring_exploratory_not_independently_validated') for r in rows]
    payload=json.dumps(accepted,indent=2,ensure_ascii=False)+'\n'
    manifest=dict(protocol='direct_authoring_v1',source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                  accepted_sha=hashlib.sha256(payload.encode()).hexdigest(),
                  author='OpenAI assistant; model version not recorded',independently_validated=False)
    path=run/'authored_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest or not (run/'accepted.json').exists() or (run/'accepted.json').read_text()!=payload:
            raise ValueError('Authored dataset changed; use a fresh run')
        print('Authored dataset already prepared; existing outputs preserved.');return
    if run.exists() and any(run.iterdir()): raise ValueError('Use a fresh, empty run directory')
    run.mkdir(parents=True,exist_ok=True)
    (run/'accepted.json').write_text(payload)
    (run/'dataset_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    # Optional independent audit, never fabricated or required for exploratory collection.
    audit=[];key=[]
    for r in rows:
        for k,response in enumerate(r['responses']):
            ident=hashlib.sha256((r['id']+str(k)+response).encode()).hexdigest()[:20]
            audit.append(dict(id=ident,prompt=r['prompt'],response=response))
            key.append(dict(id=ident,scenario_id=r['id'],intended_level=k,split=r['split']))
    random.Random(42).shuffle(audit)
    with (run/'audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','prompt','response','warmth_0_4','naturalness_1_5','reviewer','notes'])
        writer.writeheader();writer.writerows(audit)
    (run/'audit_key.json').write_text(json.dumps(key,indent=2)+'\n')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('Ready for collect. Labels are authored intentions, not independently verified ratings.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/authored_warmth_v1.jsonl')
    p.add_argument('--run',type=Path,default=ROOT/'runs/authored_v1')
    a=p.parse_args();prepare(a.input,a.run)
